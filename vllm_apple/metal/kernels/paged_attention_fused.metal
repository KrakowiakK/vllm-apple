/**
 * PagedAttention Metal Compute Shader - FUSED KV Write + Attention
 *
 * ETAP 4: Fused KV-write z attention
 *
 * Architecture:
 * - Two-phase approach using same command buffer:
 *   1. kv_write_kernel: Fast kernel to write new K/V to cache
 *   2. paged_attention_fused_*: Attention kernel reads from fully populated cache
 * - Both kernels run in same GPU dispatch, no CPU sync between them
 * - Eliminates separate CPU-side KV update (was ~46% of total time)
 *
 * Memory layout:
 * - KV cache: [num_blocks, num_kv_heads, block_size, head_size]
 * - New K/V:  [num_seqs, num_kv_heads, head_size]
 * - Query:    [num_seqs, num_query_heads, head_size]
 * - Output:   [num_seqs, num_query_heads, head_size]
 */

#include <metal_stdlib>
using namespace metal;

// Params struct (extended for fused kernel)
struct FusedAttentionParams {
    float scale;
    uint num_seqs;
    uint max_seq_len;
    uint max_blocks_per_seq;
    uint head_size;
    uint block_size;
    uint num_kv_heads;
    uint num_query_heads;
    uint queries_per_kv;
    // KV cache strides
    uint k_stride_block;
    uint k_stride_head;
    uint k_stride_token;
    uint v_stride_block;
    uint v_stride_head;
    uint v_stride_token;
    // Query/Output strides
    uint q_stride_token;
    uint q_stride_head;
    uint o_stride_token;
    uint o_stride_head;
    // New K/V strides (for input tensors)
    uint new_kv_stride_token;
    uint new_kv_stride_head;
};


// ============================================================================
// KV WRITE KERNEL: Writes new K/V to cache (runs before attention)
// ============================================================================
kernel void kv_write_decode(
    device half* key_cache [[buffer(0)]],
    device half* value_cache [[buffer(1)]],
    device const int* block_table [[buffer(2)]],
    device const int* seq_lens [[buffer(3)]],
    constant FusedAttentionParams& params [[buffer(4)]],
    device const half* new_keys [[buffer(5)]],
    device const half* new_values [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Grid: (num_seqs, num_kv_heads)
    const uint seq_idx = gid.x;
    const uint kv_head_idx = gid.y;

    if (seq_idx >= params.num_seqs || kv_head_idx >= params.num_kv_heads) return;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    const uint bt_base = seq_idx * params.max_blocks_per_seq;

    // Compute position for the NEW token (seq_len already includes it)
    const uint new_token_pos = seq_len - 1;
    const uint new_block_idx = new_token_pos / params.block_size;
    const uint new_token_offset = new_token_pos % params.block_size;
    const int new_physical_block = block_table[bt_base + new_block_idx];

    // Compute write offset in KV cache
    const uint kv_write_base = new_physical_block * params.k_stride_block
                             + kv_head_idx * params.k_stride_head
                             + new_token_offset * params.k_stride_token;

    // Read new K/V from input tensors
    const uint new_kv_offset = seq_idx * params.new_kv_stride_token
                             + kv_head_idx * params.new_kv_stride_head;

    // Write head_size elements (32 lanes, 4 elements each for head_size=128)
    const uint dims_per_lane = (params.head_size + 31) / 32;
    for (uint i = 0; i < dims_per_lane && i < 4; i++) {
        uint dim_idx = simd_lane + i * 32;
        if (dim_idx < params.head_size) {
            key_cache[kv_write_base + dim_idx] = new_keys[new_kv_offset + dim_idx];
            value_cache[kv_write_base + dim_idx] = new_values[new_kv_offset + dim_idx];
        }
    }
}


// ============================================================================
// ATTENTION KERNEL for head_size=128 (reads from already-populated cache)
// ============================================================================
kernel void paged_attention_fused_h128(
    device const half* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],   // READ-ONLY (KV already written)
    device const half* value_cache [[buffer(2)]], // READ-ONLY
    device const int* block_table [[buffer(3)]],
    device const int* seq_lens [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant FusedAttentionParams& params [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint seq_idx = gid.x;
    const uint query_head_idx = gid.y;

    if (seq_idx >= params.num_seqs || query_head_idx >= params.num_query_heads) return;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    const uint kv_head_idx = query_head_idx / params.queries_per_kv;
    const uint bt_base = seq_idx * params.max_blocks_per_seq;

    // =========================================================================
    // COMPUTE ATTENTION (K/V already written by kv_write_decode kernel)
    // =========================================================================

    const uint q_offset = seq_idx * params.q_stride_token + query_head_idx * params.q_stride_head;

    // Load query (head_size=128, 4 values per lane)
    const float q0 = float(query[q_offset + simd_lane]);
    const float q1 = float(query[q_offset + 32 + simd_lane]);
    const float q2 = float(query[q_offset + 64 + simd_lane]);
    const float q3 = float(query[q_offset + 96 + simd_lane]);

    // Online softmax state
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const uint num_blocks = (seq_len + params.block_size - 1) / params.block_size;

    // Iterate over all blocks
    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_table[bt_base + block_idx];
        const uint tokens_in_block = min(params.block_size, uint(seq_len) - block_idx * params.block_size);
        const uint kv_block_base = physical_block * params.k_stride_block
                                 + kv_head_idx * params.k_stride_head;

        for (uint token_offset = 0; token_offset < tokens_in_block; token_offset++) {
            const uint kv_offset = kv_block_base + token_offset * params.k_stride_token;

            // Load K
            const float k0 = float(key_cache[kv_offset + simd_lane]);
            const float k1 = float(key_cache[kv_offset + 32 + simd_lane]);
            const float k2 = float(key_cache[kv_offset + 64 + simd_lane]);
            const float k3 = float(key_cache[kv_offset + 96 + simd_lane]);

            // Q * K^T
            float score = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
            score = simd_sum(score) * params.scale;

            // Online softmax update
            const float m_new = max(m_prev, score);
            const float exp_prev = exp(m_prev - m_new);
            const float exp_curr = exp(score - m_new);

            acc0 *= exp_prev;
            acc1 *= exp_prev;
            acc2 *= exp_prev;
            acc3 *= exp_prev;
            l_prev = l_prev * exp_prev + exp_curr;
            m_prev = m_new;

            // Load V and accumulate
            const float v0 = float(value_cache[kv_offset + simd_lane]);
            const float v1 = float(value_cache[kv_offset + 32 + simd_lane]);
            const float v2 = float(value_cache[kv_offset + 64 + simd_lane]);
            const float v3 = float(value_cache[kv_offset + 96 + simd_lane]);

            acc0 += exp_curr * v0;
            acc1 += exp_curr * v1;
            acc2 += exp_curr * v2;
            acc3 += exp_curr * v3;
        }
    }

    // Normalize and write output
    const uint o_offset = seq_idx * params.o_stride_token + query_head_idx * params.o_stride_head;
    const float inv_l = (l_prev > 0.0f) ? 1.0f / l_prev : 0.0f;

    output[o_offset + simd_lane] = half(acc0 * inv_l);
    output[o_offset + 32 + simd_lane] = half(acc1 * inv_l);
    output[o_offset + 64 + simd_lane] = half(acc2 * inv_l);
    output[o_offset + 96 + simd_lane] = half(acc3 * inv_l);
}


// ============================================================================
// FUSED KERNEL: Generic head_size (up to 128)
// Falls back to this for head_size != 128
// ============================================================================
kernel void paged_attention_fused_generic(
    device const half* query [[buffer(0)]],
    device half* key_cache [[buffer(1)]],
    device half* value_cache [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* seq_lens [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant FusedAttentionParams& params [[buffer(6)]],
    device const half* new_keys [[buffer(7)]],
    device const half* new_values [[buffer(8)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint seq_idx = gid.x;
    const uint query_head_idx = gid.y;

    if (seq_idx >= params.num_seqs || query_head_idx >= params.num_query_heads) return;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    const uint kv_head_idx = query_head_idx / params.queries_per_kv;
    const uint bt_base = seq_idx * params.max_blocks_per_seq;
    const uint head_size = params.head_size;
    const uint dims_per_lane = (head_size + 31) / 32;

    // PHASE 1: WRITE NEW K/V TO CACHE
    const uint new_token_pos = seq_len - 1;
    const uint new_block_idx = new_token_pos / params.block_size;
    const uint new_token_offset = new_token_pos % params.block_size;
    const int new_physical_block = block_table[bt_base + new_block_idx];

    const bool should_write_kv = (query_head_idx % params.queries_per_kv == 0);

    if (should_write_kv) {
        const uint kv_write_base = new_physical_block * params.k_stride_block
                                 + kv_head_idx * params.k_stride_head
                                 + new_token_offset * params.k_stride_token;

        const uint new_kv_offset = seq_idx * params.new_kv_stride_token
                                 + kv_head_idx * params.new_kv_stride_head;

        for (uint i = 0; i < dims_per_lane && i < 4; i++) {
            uint dim_idx = simd_lane + i * 32;
            if (dim_idx < head_size) {
                key_cache[kv_write_base + dim_idx] = new_keys[new_kv_offset + dim_idx];
                value_cache[kv_write_base + dim_idx] = new_values[new_kv_offset + dim_idx];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // PHASE 2: COMPUTE ATTENTION
    const uint q_offset = seq_idx * params.q_stride_token + query_head_idx * params.q_stride_head;

    float q_reg[4];
    for (uint i = 0; i < dims_per_lane && i < 4; i++) {
        uint dim_idx = i * 32 + simd_lane;
        q_reg[i] = (dim_idx < head_size) ? float(query[q_offset + dim_idx]) : 0.0f;
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint num_blocks = (seq_len + params.block_size - 1) / params.block_size;

    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_table[bt_base + block_idx];
        const uint tokens_in_block = min(params.block_size, uint(seq_len) - block_idx * params.block_size);
        const uint kv_block_base = physical_block * params.k_stride_block
                                 + kv_head_idx * params.k_stride_head;

        for (uint token_offset = 0; token_offset < tokens_in_block; token_offset++) {
            const uint kv_offset = kv_block_base + token_offset * params.k_stride_token;

            float score = 0.0f;
            for (uint i = 0; i < dims_per_lane && i < 4; i++) {
                uint dim_idx = i * 32 + simd_lane;
                if (dim_idx < head_size) {
                    float k_val = float(key_cache[kv_offset + dim_idx]);
                    score += q_reg[i] * k_val;
                }
            }
            score = simd_sum(score) * params.scale;

            const float m_new = max(m_prev, score);
            const float exp_prev = exp(m_prev - m_new);
            const float exp_curr = exp(score - m_new);

            for (uint i = 0; i < dims_per_lane && i < 4; i++) {
                acc[i] *= exp_prev;
            }
            l_prev = l_prev * exp_prev + exp_curr;
            m_prev = m_new;

            for (uint i = 0; i < dims_per_lane && i < 4; i++) {
                uint dim_idx = i * 32 + simd_lane;
                if (dim_idx < head_size) {
                    float v_val = float(value_cache[kv_offset + dim_idx]);
                    acc[i] += exp_curr * v_val;
                }
            }
        }
    }

    const uint o_offset = seq_idx * params.o_stride_token + query_head_idx * params.o_stride_head;
    const float inv_l = (l_prev > 0.0f) ? 1.0f / l_prev : 0.0f;

    for (uint i = 0; i < dims_per_lane && i < 4; i++) {
        uint dim_idx = i * 32 + simd_lane;
        if (dim_idx < head_size) {
            output[o_offset + dim_idx] = half(acc[i] * inv_l);
        }
    }
}

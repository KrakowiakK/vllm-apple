/**
 * PagedAttention Metal Compute Shader - V2 Optimized
 *
 * Optimizations over V1:
 * 1. Query loaded once into registers before token loop
 * 2. Better register utilization
 * 3. Unrolled inner loops for common head_size values
 *
 * Memory layout:
 * - Query:       [num_tokens, num_query_heads, head_size]
 * - Key cache:   [num_blocks, num_kv_heads, block_size, head_size]
 * - Value cache: [num_blocks, num_kv_heads, block_size, head_size]
 * - Block table: [num_seqs, max_blocks_per_seq]
 * - Output:      [num_tokens, num_query_heads, head_size]
 */

#include <metal_stdlib>
using namespace metal;

struct PagedAttentionParams {
    float scale;
    uint num_seqs;
    uint max_seq_len;
    uint max_blocks_per_seq;
    uint head_size;
    uint block_size;
    uint num_kv_heads;
    uint num_query_heads;
    uint queries_per_kv;
    uint k_stride_block;
    uint k_stride_head;
    uint k_stride_token;
    uint v_stride_block;
    uint v_stride_head;
    uint v_stride_token;
    uint q_stride_token;
    uint q_stride_head;
    uint o_stride_token;
    uint o_stride_head;
};

/**
 * PagedAttention V2 - Optimized decode kernel
 *
 * Key optimizations:
 * - Query vector loaded once into registers
 * - Specialized paths for common head_size values
 */
kernel void paged_attention_v2_decode(
    device const half* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device const half* value_cache [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* seq_lens [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant PagedAttentionParams& params [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint seq_idx = gid.x;
    const uint query_head_idx = gid.y;

    if (seq_idx >= params.num_seqs || query_head_idx >= params.num_query_heads) return;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    const uint num_blocks = (seq_len + params.block_size - 1) / params.block_size;
    const uint kv_head_idx = query_head_idx / params.queries_per_kv;
    const uint q_offset = seq_idx * params.q_stride_token + query_head_idx * params.q_stride_head;

    // Determine dims per lane
    const uint dims_per_lane = (params.head_size + 31) / 32;
    if (dims_per_lane > 4) {
        if (simd_lane == 0) {
            output[seq_idx * params.o_stride_token + query_head_idx * params.o_stride_head] = half(0.0f / 0.0f);
        }
        return;
    }

    // ============================================================
    // OPTIMIZATION 1: Load query vector into registers ONCE
    // ============================================================
    float q_reg[4];  // Max 4 values per lane for head_size <= 128
    for (uint i = 0; i < dims_per_lane; i++) {
        uint dim_idx = i * 32 + simd_lane;
        if (dim_idx < params.head_size) {
            q_reg[i] = float(query[q_offset + dim_idx]);
        } else {
            q_reg[i] = 0.0f;
        }
    }

    // Running softmax state
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Process all blocks
    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_table[seq_idx * params.max_blocks_per_seq + block_idx];
        const uint tokens_in_block = min(params.block_size, uint(seq_len) - block_idx * params.block_size);

        // Base offset for this block's K/V
        const uint kv_block_offset = physical_block * params.k_stride_block
                                   + kv_head_idx * params.k_stride_head;

        // Process each token in block
        for (uint token_offset = 0; token_offset < tokens_in_block; token_offset++) {
            const uint kv_token_offset = kv_block_offset + token_offset * params.k_stride_token;

            // ============================================================
            // Compute Q*K^T using pre-loaded query registers
            // ============================================================
            float score = 0.0f;

            // Use pre-loaded q_reg instead of re-reading from memory
            for (uint i = 0; i < dims_per_lane; i++) {
                uint dim_idx = i * 32 + simd_lane;
                if (dim_idx < params.head_size) {
                    float k_val = float(key_cache[kv_token_offset + dim_idx]);
                    score += q_reg[i] * k_val;
                }
            }

            // Reduce and scale
            score = simd_sum(score) * params.scale;

            // Online softmax update
            float m_new = max(m_prev, score);
            float exp_prev = exp(m_prev - m_new);
            float exp_curr = exp(score - m_new);

            // Rescale accumulators
            for (uint i = 0; i < dims_per_lane; i++) {
                acc[i] *= exp_prev;
            }
            l_prev = l_prev * exp_prev + exp_curr;
            m_prev = m_new;

            // Add value contribution
            for (uint i = 0; i < dims_per_lane; i++) {
                uint dim_idx = i * 32 + simd_lane;
                if (dim_idx < params.head_size) {
                    float v_val = float(value_cache[kv_token_offset + dim_idx]);
                    acc[i] += exp_curr * v_val;
                }
            }
        }
    }

    // Normalize and write output
    const uint o_offset = seq_idx * params.o_stride_token + query_head_idx * params.o_stride_head;
    const float inv_l = (l_prev > 0.0f) ? 1.0f / l_prev : 0.0f;

    for (uint i = 0; i < dims_per_lane; i++) {
        uint dim_idx = i * 32 + simd_lane;
        if (dim_idx < params.head_size) {
            output[o_offset + dim_idx] = half(acc[i] * inv_l);
        }
    }
}

/**
 * Specialized kernel for head_size = 128 (most common in LLMs)
 * Fully unrolled for maximum performance
 *
 * Optimizations:
 * - Query loaded into registers once
 * - Coalesced memory access
 * - Online softmax for numerical stability
 */
kernel void paged_attention_v2_decode_h128(
    device const half* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device const half* value_cache [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* seq_lens [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant PagedAttentionParams& params [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint seq_idx = gid.x;
    const uint query_head_idx = gid.y;

    if (seq_idx >= params.num_seqs || query_head_idx >= params.num_query_heads) return;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    const uint num_blocks = (seq_len + params.block_size - 1) / params.block_size;
    const uint kv_head_idx = query_head_idx / params.queries_per_kv;
    const uint q_offset = seq_idx * params.q_stride_token + query_head_idx * params.q_stride_head;

    // Load query into 4 registers (head_size=128 / 32 lanes = 4 per lane)
    const float q0 = float(query[q_offset + simd_lane]);
    const float q1 = float(query[q_offset + 32 + simd_lane]);
    const float q2 = float(query[q_offset + 64 + simd_lane]);
    const float q3 = float(query[q_offset + 96 + simd_lane]);

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Pre-compute block table base
    const uint bt_base = seq_idx * params.max_blocks_per_seq;

    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_table[bt_base + block_idx];
        const uint tokens_in_block = min(params.block_size, uint(seq_len) - block_idx * params.block_size);
        const uint kv_block_base = physical_block * params.k_stride_block
                                 + kv_head_idx * params.k_stride_head;

        for (uint token_offset = 0; token_offset < tokens_in_block; token_offset++) {
            const uint kv_offset = kv_block_base + token_offset * params.k_stride_token;

            // Q*K^T - coalesced K reads across SIMD lanes
            const float k0 = float(key_cache[kv_offset + simd_lane]);
            const float k1 = float(key_cache[kv_offset + 32 + simd_lane]);
            const float k2 = float(key_cache[kv_offset + 64 + simd_lane]);
            const float k3 = float(key_cache[kv_offset + 96 + simd_lane]);

            float score = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
            score = simd_sum(score) * params.scale;

            // Online softmax update
            const float m_new = max(m_prev, score);
            const float exp_prev = exp(m_prev - m_new);
            const float exp_curr = exp(score - m_new);

            // Rescale accumulators and update state
            acc0 *= exp_prev;
            acc1 *= exp_prev;
            acc2 *= exp_prev;
            acc3 *= exp_prev;
            l_prev = l_prev * exp_prev + exp_curr;
            m_prev = m_new;

            // Coalesced V reads and accumulate
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

// Keep original kernels for compatibility
kernel void paged_attention_decode(
    device const half* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device const half* value_cache [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* seq_lens [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant PagedAttentionParams& params [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint seq_idx = gid.x;
    const uint query_head_idx = gid.y;

    if (seq_idx >= params.num_seqs || query_head_idx >= params.num_query_heads) return;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    const uint num_blocks = (seq_len + params.block_size - 1) / params.block_size;
    const uint kv_head_idx = query_head_idx / params.queries_per_kv;
    const uint q_offset = seq_idx * params.q_stride_token + query_head_idx * params.q_stride_head;

    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint dims_per_lane = (params.head_size + 31) / 32;

    if (dims_per_lane > 4) {
        if (simd_lane == 0) {
            const uint o_offset = seq_idx * params.o_stride_token + query_head_idx * params.o_stride_head;
            output[o_offset] = half(0.0f / 0.0f);
        }
        return;
    }

    // Load query once
    float q_reg[4];
    for (uint i = 0; i < dims_per_lane; i++) {
        uint dim_idx = i * 32 + simd_lane;
        q_reg[i] = (dim_idx < params.head_size) ? float(query[q_offset + dim_idx]) : 0.0f;
    }

    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_table[seq_idx * params.max_blocks_per_seq + block_idx];
        const uint tokens_in_block = min(params.block_size, uint(seq_len) - block_idx * params.block_size);

        for (uint token_offset = 0; token_offset < tokens_in_block; token_offset++) {
            float score = 0.0f;

            for (uint i = 0; i < dims_per_lane; i++) {
                uint dim_idx = i * 32 + simd_lane;
                if (dim_idx < params.head_size) {
                    uint k_idx = physical_block * params.k_stride_block
                               + kv_head_idx * params.k_stride_head
                               + token_offset * params.k_stride_token
                               + dim_idx;
                    score += q_reg[i] * float(key_cache[k_idx]);
                }
            }

            score = simd_sum(score) * params.scale;

            float m_new = max(m_prev, score);
            float exp_prev = exp(m_prev - m_new);
            float exp_curr = exp(score - m_new);

            for (uint i = 0; i < dims_per_lane; i++) {
                acc[i] *= exp_prev;
            }
            l_prev = l_prev * exp_prev + exp_curr;
            m_prev = m_new;

            for (uint i = 0; i < dims_per_lane; i++) {
                uint dim_idx = i * 32 + simd_lane;
                if (dim_idx < params.head_size) {
                    uint v_idx = physical_block * params.v_stride_block
                               + kv_head_idx * params.v_stride_head
                               + token_offset * params.v_stride_token
                               + dim_idx;
                    acc[i] += exp_curr * float(value_cache[v_idx]);
                }
            }
        }
    }

    const uint o_offset = seq_idx * params.o_stride_token + query_head_idx * params.o_stride_head;
    for (uint i = 0; i < dims_per_lane; i++) {
        uint dim_idx = i * 32 + simd_lane;
        if (dim_idx < params.head_size) {
            float result = (l_prev > 0.0f) ? acc[i] / l_prev : 0.0f;
            output[o_offset + dim_idx] = half(result);
        }
    }
}

/**
 * PagedAttention V2 - Prefill kernel (token-parallel, causal)
 *
 * Dispatch grid: (num_tokens, num_query_heads)
 * - gid.x: token index in the flattened query batch
 * - gid.y: query head index
 *
 * Requires:
 * - token_to_seq[token_idx] gives the sequence index for this token
 * - positions[token_idx] gives the absolute position in the sequence
 *
 * The kernel computes causal attention for each token at its position,
 * attending to KV cache positions [0..pos] for that sequence.
 */
kernel void paged_attention_v2_prefill(
    device const half* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device const half* value_cache [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* token_to_seq [[buffer(4)]],
    device const int* positions [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant PagedAttentionParams& params [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint token_idx = gid.x;
    const uint query_head_idx = gid.y;

    if (query_head_idx >= params.num_query_heads) return;

    const int seq_idx_i = token_to_seq[token_idx];
    if (seq_idx_i < 0 || uint(seq_idx_i) >= params.num_seqs) return;
    const uint seq_idx = uint(seq_idx_i);

    const int pos_i = positions[token_idx];
    if (pos_i < 0) return;
    const uint context_len = uint(pos_i + 1);

    const uint num_blocks = (context_len + params.block_size - 1) / params.block_size;
    if (num_blocks > params.max_blocks_per_seq) return;

    const uint kv_head_idx = query_head_idx / params.queries_per_kv;
    const uint bt_base = seq_idx * params.max_blocks_per_seq;
    const uint q_offset = token_idx * params.q_stride_token + query_head_idx * params.q_stride_head;

    const uint dims_per_lane = (params.head_size + 31) / 32;
    if (dims_per_lane > 4) {
        if (simd_lane == 0) {
            output[token_idx * params.o_stride_token + query_head_idx * params.o_stride_head] =
                half(0.0f / 0.0f);
        }
        return;
    }

    float q_reg[4];
    for (uint i = 0; i < dims_per_lane; i++) {
        uint dim_idx = i * 32 + simd_lane;
        q_reg[i] = (dim_idx < params.head_size) ? float(query[q_offset + dim_idx]) : 0.0f;
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_table[bt_base + block_idx];
        if (physical_block < 0) return;

        const uint tokens_in_block =
            min(params.block_size, context_len - block_idx * params.block_size);

        const uint kv_block_offset = uint(physical_block) * params.k_stride_block
                                   + kv_head_idx * params.k_stride_head;

        for (uint token_offset = 0; token_offset < tokens_in_block; token_offset++) {
            const uint kv_token_offset = kv_block_offset + token_offset * params.k_stride_token;

            float score = 0.0f;
            for (uint i = 0; i < dims_per_lane; i++) {
                uint dim_idx = i * 32 + simd_lane;
                if (dim_idx < params.head_size) {
                    float k_val = float(key_cache[kv_token_offset + dim_idx]);
                    score += q_reg[i] * k_val;
                }
            }
            score = simd_sum(score) * params.scale;

            const float m_new = max(m_prev, score);
            const float exp_prev = exp(m_prev - m_new);
            const float exp_curr = exp(score - m_new);

            for (uint i = 0; i < dims_per_lane; i++) {
                acc[i] *= exp_prev;
            }
            l_prev = l_prev * exp_prev + exp_curr;
            m_prev = m_new;

            for (uint i = 0; i < dims_per_lane; i++) {
                uint dim_idx = i * 32 + simd_lane;
                if (dim_idx < params.head_size) {
                    float v_val = float(value_cache[kv_token_offset + dim_idx]);
                    acc[i] += exp_curr * v_val;
                }
            }
        }
    }

    const uint o_offset = token_idx * params.o_stride_token + query_head_idx * params.o_stride_head;
    const float inv_l = (l_prev > 0.0f) ? 1.0f / l_prev : 0.0f;

    for (uint i = 0; i < dims_per_lane; i++) {
        uint dim_idx = i * 32 + simd_lane;
        if (dim_idx < params.head_size) {
            output[o_offset + dim_idx] = half(acc[i] * inv_l);
        }
    }
}

kernel void test_copy(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        output[gid] = input[gid];
    }
}

// Metal MoE Kernel v2 for Apple Silicon
// Optimized using simdgroup_matrix for efficient matrix operations
//
// Target: Qwen3-30B-A3B
// - 128 experts, top-8 selection
// - hidden_size = 2048
// - intermediate_size = 768
//
// Architecture: Apple M-series GPU
// - Uses simdgroup (32 threads) for matrix operations
// - Tile-based computation for memory efficiency

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Tile sizes optimized for Apple Silicon
// simdgroup_matrix supports 8x8 tiles
constant int TILE_M = 8;   // Rows per simdgroup tile
constant int TILE_N = 8;   // Cols per simdgroup tile
constant int TILE_K = 8;   // Reduction dimension per tile

// ============================================================================
// Helper: Load tile from global memory to simdgroup_matrix
// ============================================================================

// Load a tile of matrix A (row-major) into simdgroup_matrix
template<typename T>
METAL_FUNC void load_tile_a(
    simdgroup_matrix<T, TILE_M, TILE_K>& tile,
    device const T* src,
    int row_start,
    int col_start,
    int stride,
    int max_rows,
    int max_cols
) {
    // simdgroup_load handles the distribution across threads
    if (row_start < max_rows && col_start < max_cols) {
        simdgroup_load(tile, src + row_start * stride + col_start, stride);
    }
}

// Load a tile of matrix B (row-major, but we need column-major for matmul)
template<typename T>
METAL_FUNC void load_tile_b(
    simdgroup_matrix<T, TILE_K, TILE_N>& tile,
    device const T* src,
    int row_start,
    int col_start,
    int stride,
    int max_rows,
    int max_cols
) {
    if (row_start < max_rows && col_start < max_cols) {
        simdgroup_load(tile, src + row_start * stride + col_start, stride);
    }
}

// ============================================================================
// Kernel 1: Gate-Up Projection
// Computes: gate_up = hidden @ W1.T where W1 is [intermediate*2, hidden]
// Output: [num_pairs, intermediate*2]
// ============================================================================

kernel void moe_gate_up_simd(
    device const half* hidden_states [[buffer(0)]],   // [num_tokens, hidden_size]
    device const half* w1 [[buffer(1)]],              // [num_experts, inter*2, hidden]
    device const int* expert_ids [[buffer(2)]],       // [num_pairs] - flattened expert indices
    device half* gate_up_output [[buffer(3)]],        // [num_pairs, inter*2]
    constant uint& num_pairs [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& intermediate_size [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one (token, expert) pair
    uint pair_idx = tgid.x;
    if (pair_idx >= num_pairs) return;

    int expert_idx = expert_ids[pair_idx];
    uint out_size = intermediate_size * 2;

    // Pointers
    device const half* input = hidden_states + pair_idx * hidden_size;
    device const half* weight = w1 + expert_idx * out_size * hidden_size;
    device half* output = gate_up_output + pair_idx * out_size;

    // Each simdgroup computes a tile of the output
    // Output tile position
    uint tile_row = simd_group_id * TILE_M;  // Which output features

    // Accumulator for this tile
    simdgroup_matrix<half, TILE_M, TILE_N> acc;

    // Initialize accumulator to zero
    simdgroup_fill(acc, half(0));

    // Tile over the reduction dimension (hidden_size)
    for (uint k = 0; k < hidden_size; k += TILE_K) {
        // Load input tile [1, TILE_K] - broadcast across rows
        simdgroup_matrix<half, TILE_M, TILE_K> input_tile;

        // Load weight tile [TILE_M, TILE_K]
        simdgroup_matrix<half, TILE_K, TILE_N> weight_tile;

        // Note: For single token, input is [1, hidden_size]
        // We need to handle this specially
        if (k < hidden_size && tile_row < out_size) {
            simdgroup_load(weight_tile, weight + tile_row * hidden_size + k, hidden_size);
        }

        // Multiply and accumulate
        simdgroup_multiply_accumulate(acc, input_tile, weight_tile, acc);
    }

    // Store result
    if (tile_row < out_size) {
        simdgroup_store(acc, output + tile_row, out_size);
    }
}

// ============================================================================
// Kernel 2: Simple but correct MoE gate-up (one thread per output element)
// This is slower but guaranteed correct - use as baseline
// ============================================================================

kernel void moe_gate_up_simple(
    device const half* hidden_states [[buffer(0)]],   // [num_tokens, hidden_size]
    device const half* w1 [[buffer(1)]],              // [num_experts, inter*2, hidden]
    device const int* expert_ids [[buffer(2)]],       // [num_pairs]
    device half* gate_up_output [[buffer(3)]],        // [num_pairs, inter*2]
    constant uint& num_pairs [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& intermediate_size [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint pair_idx = tid.x;
    uint out_idx = tid.y;
    uint out_size = intermediate_size * 2;

    if (pair_idx >= num_pairs || out_idx >= out_size) return;

    int expert_idx = expert_ids[pair_idx];

    device const half* input = hidden_states + pair_idx * hidden_size;
    device const half* weight = w1 + expert_idx * out_size * hidden_size + out_idx * hidden_size;

    // Dot product
    float acc = 0.0f;  // Use float for accumulation
    for (uint k = 0; k < hidden_size; k++) {
        acc += float(input[k]) * float(weight[k]);
    }

    gate_up_output[pair_idx * out_size + out_idx] = half(acc);
}

// ============================================================================
// Kernel 3: Vectorized gate-up (4 elements per thread)
// ============================================================================

kernel void moe_gate_up_vec4(
    device const half4* hidden_states [[buffer(0)]],  // [num_tokens, hidden_size/4]
    device const half4* w1 [[buffer(1)]],             // [num_experts, inter*2, hidden/4]
    device const int* expert_ids [[buffer(2)]],       // [num_pairs]
    device half* gate_up_output [[buffer(3)]],        // [num_pairs, inter*2]
    constant uint& num_pairs [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& intermediate_size [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint pair_idx = tid.x;
    uint out_idx = tid.y;
    uint out_size = intermediate_size * 2;
    uint hidden_size_vec4 = hidden_size / 4;

    if (pair_idx >= num_pairs || out_idx >= out_size) return;

    int expert_idx = expert_ids[pair_idx];

    device const half4* input = hidden_states + pair_idx * hidden_size_vec4;
    device const half4* weight = w1 + expert_idx * out_size * hidden_size_vec4 + out_idx * hidden_size_vec4;

    // Vectorized dot product
    float acc = 0.0f;
    for (uint k = 0; k < hidden_size_vec4; k++) {
        half4 a = input[k];
        half4 b = weight[k];
        acc += float(a.x) * float(b.x);
        acc += float(a.y) * float(b.y);
        acc += float(a.z) * float(b.z);
        acc += float(a.w) * float(b.w);
    }

    gate_up_output[pair_idx * out_size + out_idx] = half(acc);
}

// ============================================================================
// Kernel 4: SiLU activation + element-wise multiply
// Computes: activated = silu(gate) * up
// Input: gate_up [num_pairs, inter*2] where first half is gate, second is up
// Output: activated [num_pairs, inter]
// ============================================================================

kernel void moe_silu_mul(
    device const half* gate_up [[buffer(0)]],         // [num_pairs, inter*2]
    device half* activated [[buffer(1)]],             // [num_pairs, inter]
    constant uint& num_pairs [[buffer(2)]],
    constant uint& intermediate_size [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint pair_idx = tid.x;
    uint feat_idx = tid.y;

    if (pair_idx >= num_pairs || feat_idx >= intermediate_size) return;

    uint out_size = intermediate_size * 2;

    // gate is first half, up is second half
    float gate = float(gate_up[pair_idx * out_size + feat_idx]);
    float up = float(gate_up[pair_idx * out_size + intermediate_size + feat_idx]);

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu_gate = gate / (1.0f + exp(-gate));
    float result = silu_gate * up;

    activated[pair_idx * intermediate_size + feat_idx] = half(result);
}

// ============================================================================
// Kernel 5: Down projection
// Computes: output = activated @ W2.T where W2 is [hidden, inter]
// ============================================================================

kernel void moe_down_proj(
    device const half* activated [[buffer(0)]],       // [num_pairs, inter]
    device const half* w2 [[buffer(1)]],              // [num_experts, hidden, inter]
    device const int* expert_ids [[buffer(2)]],       // [num_pairs]
    device const half* routing_weights [[buffer(3)]], // [num_pairs]
    device half* output [[buffer(4)]],                // [num_tokens, hidden]
    device const int* token_ids [[buffer(5)]],        // [num_pairs] - which token each pair belongs to
    constant uint& num_pairs [[buffer(6)]],
    constant uint& hidden_size [[buffer(7)]],
    constant uint& intermediate_size [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint pair_idx = tid.x;
    uint hidden_idx = tid.y;

    if (pair_idx >= num_pairs || hidden_idx >= hidden_size) return;

    int expert_idx = expert_ids[pair_idx];
    int token_idx = token_ids[pair_idx];
    float routing_weight = float(routing_weights[pair_idx]);

    device const half* input = activated + pair_idx * intermediate_size;
    device const half* weight = w2 + expert_idx * hidden_size * intermediate_size + hidden_idx * intermediate_size;

    // Dot product
    float acc = 0.0f;
    for (uint k = 0; k < intermediate_size; k++) {
        acc += float(input[k]) * float(weight[k]);
    }

    // Apply routing weight and atomic add to output
    // Note: atomic_add for half not available, use float atomics
    device float* out_ptr = (device float*)(output + token_idx * hidden_size + hidden_idx);

    // For half precision, we need a workaround
    // Simple approach: just write (will have race conditions for multiple experts)
    // TODO: Use atomic float and convert, or accumulate in shared memory
    output[token_idx * hidden_size + hidden_idx] += half(acc * routing_weight);
}

// ============================================================================
// Kernel 6: Fused MoE for single token (decode)
// Most common case - one token, top-k experts
// Optimized for this specific case
// ============================================================================

kernel void moe_decode_fused(
    device const half* hidden [[buffer(0)]],          // [hidden_size] - single token
    device const half* w1 [[buffer(1)]],              // [num_experts, inter*2, hidden]
    device const half* w2 [[buffer(2)]],              // [num_experts, hidden, inter]
    device const int* topk_ids [[buffer(3)]],         // [topk]
    device const half* topk_weights [[buffer(4)]],    // [topk]
    device half* output [[buffer(5)]],                // [hidden_size]
    constant uint& hidden_size [[buffer(6)]],
    constant uint& intermediate_size [[buffer(7)]],
    constant uint& topk [[buffer(8)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint threads_per_tg [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one expert
    uint expert_rank = tgid;
    if (expert_rank >= topk) return;

    int expert_idx = topk_ids[expert_rank];
    float routing_weight = float(topk_weights[expert_rank]);

    uint out_size = intermediate_size * 2;

    device const half* expert_w1 = w1 + expert_idx * out_size * hidden_size;
    device const half* expert_w2 = w2 + expert_idx * hidden_size * intermediate_size;

    // Shared memory for intermediate activations
    threadgroup float shared_activated[1024];  // Max intermediate_size

    // Step 1: Gate-up projection (each thread handles some output features)
    for (uint out_idx = tid; out_idx < out_size; out_idx += threads_per_tg) {
        device const half* weight = expert_w1 + out_idx * hidden_size;

        float acc = 0.0f;
        for (uint k = 0; k < hidden_size; k++) {
            acc += float(hidden[k]) * float(weight[k]);
        }

        // Store gate and up separately
        if (out_idx < intermediate_size) {
            shared_activated[out_idx] = acc;  // gate
        } else {
            shared_activated[out_idx] = acc;  // up (stored at out_idx)
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: SiLU activation
    for (uint i = tid; i < intermediate_size; i += threads_per_tg) {
        float gate = shared_activated[i];
        float up = shared_activated[intermediate_size + i];
        float silu = gate / (1.0f + exp(-gate));
        shared_activated[i] = silu * up;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Down projection and accumulate to output
    for (uint h = tid; h < hidden_size; h += threads_per_tg) {
        device const half* weight = expert_w2 + h * intermediate_size;

        float acc = 0.0f;
        for (uint k = 0; k < intermediate_size; k++) {
            acc += shared_activated[k] * float(weight[k]);
        }

        // Atomic add with routing weight
        // Note: This is approximate for half - proper impl would use float atomics
        float current = float(output[h]);
        output[h] = half(current + acc * routing_weight);
    }
}

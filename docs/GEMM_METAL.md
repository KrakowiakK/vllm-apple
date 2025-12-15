# vLLM-Apple Metal Engine Performance Optimization Plan

**Document**: GEMM_METAL.md (Performance Unlock Plan)
**Date**: 2025-12-14
**Goal**: Replace MPS GEMM with custom Metal kernels + additional optimizations

---

## Executive Summary

The vLLM-Apple Metal Engine v2.0 works correctly but **GEMM operations (70-80% of compute) still use MPS**, not custom Metal kernels. This plan details the implementation of:

1. **Custom Metal GEMM Kernels** - Replace MPSMatrixMultiplication (PRIMARY)
2. **Top-K Logits Readback** - 60-90% readback reduction
3. **Selective LM Head** - 30-50% logits speedup in prefill
4. **Fused O-Projection + Residual** - 10-15% per layer
5. **Memory Barrier Reduction** - 5-10% overall

**Expected Total Impact**: 2-3x throughput improvement for decode, 1.5-2x for prefill

---

## Part 1: Custom Metal GEMM Kernels (PRIMARY FOCUS)

### 1.1 Current State Analysis

**Location**: `/vllm_apple/engine/ops/gemm.py`

Current implementation uses `MPSMatrixMultiplication.encodeToCommandBuffer()`:
- Encode-only API (correct, no waits)
- All calls use `alpha=1.0, beta=0.0, transpose_B=True`
- Weights stored as `[out_features, in_features]` (PyTorch convention)

**Typical GEMM Shapes** (Llama-like models):

| Operation | M (tokens) | K (in) | N (out) | Count/Layer |
|-----------|------------|--------|---------|-------------|
| QKV Projection | 1-2048 | 4096 | 12288 | 1 |
| O-Projection | 1-2048 | 4096 | 4096 | 1 |
| MLP Up/Gate | 1-2048 | 4096 | 11008 | 2 |
| MLP Down | 1-2048 | 11008 | 4096 | 1 |
| LM Head | 1-2048 | 4096 | 128256 | 1 (final) |

### 1.2 Implementation Phases

#### Phase 1: Naive Baseline Kernel (Correctness First)

**Objective**: Establish correct kernel infrastructure

**New File**: `/vllm_apple/engine/ops/gemm_metal.py`

```metal
// Naive GEMM: C[M,N] = A[M,K] @ B^T[N,K]
// Each thread computes one element of C
struct GEMMParams {
    uint M, N, K;
    uint lda, ldb, ldc;
};

kernel void gemm_naive_f16(
    device const half* A [[buffer(0)]],      // [M, K]
    device const half* B [[buffer(1)]],      // [N, K] (transposed)
    device half* C [[buffer(2)]],            // [M, N]
    constant GEMMParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // N dimension

    if (row >= params.M || col >= params.N) return;

    float acc = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        float a_val = float(A[row * params.K + k]);
        float b_val = float(B[col * params.K + k]);  // B^T[k,col] = B[col,k]
        acc += a_val * b_val;
    }

    C[row * params.N + col] = half(acc);
}
```

**Dispatch**:
```python
threads_per_threadgroup = MTLSize(16, 16, 1)  # 256 threads
threadgroups = MTLSize((N + 15) // 16, (M + 15) // 16, 1)
```

**Tests**:
- Correctness vs numpy reference (rtol=1e-2, atol=1e-3)
- Shapes: (1, 4096, 4096), (32, 4096, 4096), (128, 4096, 12288)

---

#### Phase 2: Tiled Kernel with Threadgroup Memory

**Objective**: 2-5x speedup via memory tiling

**Tile Configuration**:
- BM=64, BN=64, BK=16 (general case)
- BM=8, BN=64, BK=32 (small-M decode)
- Threadgroup memory: 4KB per tile (well within 32KB limit)

```metal
constant uint BM = 64;
constant uint BN = 64;
constant uint BK = 16;

kernel void gemm_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GEMMParams& params [[buffer(3)]],
    threadgroup half* As [[threadgroup(0)]],  // [BM, BK]
    threadgroup half* Bs [[threadgroup(1)]],  // [BN, BK]
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    // Thread block handles BM x BN tile of C
    uint c_row = tid.y * BM;
    uint c_col = tid.x * BN;

    // Per-thread: TM=4, TN=4 elements
    const uint TM = 4, TN = 4;
    float acc[TM][TN] = {{0.0f}};

    // Iterate K in tiles of BK
    for (uint kt = 0; kt < (params.K + BK - 1) / BK; kt++) {
        // Cooperative load A tile to shared
        // Cooperative load B tile to shared
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial results (outer product)
        for (uint k = 0; k < BK; k++) {
            // Load A[thread_row*TM : thread_row*TM+TM, k]
            // Load B[thread_col*TN : thread_col*TN+TN, k]
            // acc += outer_product(a, b)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results to C
}
```

**Dispatch**:
```python
threads_per_threadgroup = MTLSize(16, 16, 1)  # 16*4 = 64 rows, 16*4 = 64 cols
threadgroups = MTLSize((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
tgp_mem = BM * BK * 2 + BN * BK * 2  # ~4KB
```

**Target**: Within 50% of MPS performance

---

#### Phase 3: Optimized Kernel with simdgroup_matrix

**Objective**: Match or exceed MPS performance using Apple's hardware MMA

**Key Technique**: `simdgroup_matrix<half, 8, 8>` operations

```metal
#include <metal_simdgroup_matrix>

constant uint BM = 64, BN = 64, BK = 32;
constant uint WM = 2, WN = 2;  // Warp tiles

kernel void gemm_simdgroup_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GEMMParams& params [[buffer(3)]],
    threadgroup half* As [[threadgroup(0)]],
    threadgroup half* Bs [[threadgroup(1)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // 4 simdgroups per threadgroup
    // Each simdgroup computes (BM/WM) x (BN/WN) = 32x32 output elements

    // Fragment accumulators (TM x TN 8x8 matrices)
    const uint TM = 4, TN = 4;
    simdgroup_matrix<float, 8, 8> acc[TM][TN];

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        // Cooperative load with bank conflict avoidance (+8 padding)
        // ...
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kf = 0; kf < BK; kf += 8) {
            simdgroup_matrix<half, 8, 8> A_frag[TM], B_frag[TN];

            // Load fragments
            for (uint i = 0; i < TM; i++)
                simdgroup_load(A_frag[i], As + offset, stride);
            for (uint j = 0; j < TN; j++)
                simdgroup_load(B_frag[j], Bs + offset, stride);

            // MMA
            for (uint i = 0; i < TM; i++)
                for (uint j = 0; j < TN; j++)
                    simdgroup_multiply_accumulate(acc[i][j], A_frag[i], B_frag[j], acc[i][j]);
        }
    }

    // Store results
    for (uint i = 0; i < TM; i++)
        for (uint j = 0; j < TN; j++)
            simdgroup_store(acc[i][j], C + offset, params.N);
}
```

**Optimizations**:
- Bank conflict avoidance: +8 padding in threadgroup memory
- Tile swizzling for L2 cache reuse
- Vectorized loads (half4, half8)

**Target**: Within 5% of MPS performance (or faster)

---

#### Phase 4: Specialized Variants

**4.1 Small-M Kernel (Decode with M=1-8)**

```metal
// Each simdgroup handles one row, maximizes parallelism on N
kernel void gemm_small_m_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GEMMParams& params [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint row = tid.y;
    uint col_base = tid.x * 256 + simd_group_id * 64;

    if (row >= params.M) return;

    // Vectorized dot product with half4
    float2 acc = float2(0.0f);
    for (uint k = 0; k < params.K; k += 4) {
        half4 a_vec = *((device half4*)(A + row * params.K + k));
        // Process 2 columns per thread
        // ...
    }
}
```

**4.2 Large-N Kernel (LM Head with N=128K)**

```metal
// Cache A row in threadgroup memory, stream B columns
kernel void gemm_large_n_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GEMMParams& params [[buffer(3)]],
    threadgroup half* A_shared [[threadgroup(0)]],
    // ...
) {
    // Load entire A row to shared (K elements)
    // Each thread computes one output element
}
```

**4.3 Kernel Selection Logic**

```python
def _select_kernel(self, M: int, K: int, N: int):
    if M <= 8:
        return self._pipeline_small_m
    elif N > 65536:
        return self._pipeline_large_n
    elif K == 4096:  # Common hidden size
        return self._pipeline_k4096
    else:
        return self._pipeline_simdgroup
```

---

### 1.3 Integration Plan

**File Structure**:
```
vllm_apple/engine/ops/
    gemm.py              # Original MPS (fallback)
    gemm_metal.py        # NEW: Custom Metal kernels
    gemm_selector.py     # NEW: Unified interface
```

**Unified API**:
```python
class UnifiedGEMM:
    def __init__(self, context, backend="auto"):  # auto, mps, metal
        self._mps = EngineGEMM(context)
        self._metal = EngineGEMMMetal(context)

    def encode(self, step_ctx, A, B, C, **kwargs):
        if self._backend == "metal" or (self._backend == "auto" and self._prefer_metal(A, B)):
            return self._metal.encode(...)
        return self._mps.encode(...)
```

**Environment Variables**:
- `VLLM_GEMM_BACKEND=auto|mps|metal` (default: auto)
- `VLLM_METAL_GEMM_DEBUG=0|1` (debug logging)
- `VLLM_METAL_GEMM_VERIFY=0|1` (verify against MPS)

---

## Part 2: Top-K Logits Readback (60-90% Reduction)

### 2.1 Current State

- Config exists: `VLLM_METAL_TOPK_LOGITS` in config.py
- Placeholder: `encode_with_selection()` in lm_head.py
- Current readback: `num_tokens * vocab_size * 2` bytes (e.g., 4MB for 16 seqs @ 128K vocab)

### 2.2 Implementation

**New Metal Kernel**: `/vllm_apple/metal/kernels/topk_selection.metal`

```metal
struct TopKParams {
    uint num_tokens;
    uint vocab_size;
    uint k;
};

kernel void topk_selection(
    device const half* logits,      // [num_tokens, vocab_size]
    device int* topk_indices,        // [num_tokens, k]
    device half* topk_values,        // [num_tokens, k]
    constant TopKParams& params,
    uint token_idx [[threadgroup_position_in_grid.y]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Each threadgroup processes one token
    // Use bitonic sort within SIMD for thread-local top-k
    // Merge across threads using threadgroup memory
}
```

**New Op**: `/vllm_apple/engine/ops/topk.py`

```python
class EngineTopK:
    def encode(self, step_ctx, logits, indices_out, values_out, num_tokens, k):
        encoder = step_ctx.get_compute_encoder()
        # Set pipeline, buffers, dispatch
```

**LM Head Update**:
```python
def encode_with_topk(self, step_ctx, hidden_states, topk_indices, topk_values, num_tokens, k):
    # 1. Encode LM head GEMM
    self._gemm.encode(...)
    step_ctx.memory_barrier()

    # 2. Encode top-k selection
    self._topk.encode(step_ctx, self._logits_buffer, topk_indices, topk_values, num_tokens, k)
```

**Feature Flag**: `VLLM_METAL_TOPK_LOGITS=50` (or any k value)

---

## Part 3: Selective LM Head (30-50% Prefill Speedup)

### 3.1 Opportunity

In prefill, only the **last token per sequence** needs logits. Currently computes for all tokens.

### 3.2 Implementation

**Gather Kernel**:
```metal
kernel void gather_hidden_states(
    device const half* all_hidden,    // [total_tokens, hidden_size]
    device const int* indices,         // [num_selected]
    device half* selected_hidden,      // [num_selected, hidden_size]
    constant GatherParams& params,
    uint2 gid [[thread_position_in_grid]]
) {
    uint selected_idx = gid.y;
    uint hidden_idx = gid.x;

    if (selected_idx >= params.num_selected || hidden_idx >= params.hidden_size) return;

    int src_token = indices[selected_idx];
    selected_hidden[selected_idx * params.hidden_size + hidden_idx] =
        all_hidden[src_token * params.hidden_size + hidden_idx];
}
```

**LM Head Update**:
```python
def encode_selective(self, step_ctx, hidden_states, token_indices, output, num_selected):
    # 1. Gather selected tokens
    self._gather.encode(step_ctx, hidden_states, token_indices, selected_buffer, ...)
    step_ctx.memory_barrier()

    # 2. GEMM on smaller matrix (num_selected instead of num_tokens)
    self._gemm.encode(step_ctx, selected_buffer, self._weights, output, M=num_selected, ...)
```

**Runner Integration**:
```python
if step_desc.is_prefill:
    # Only last token per sequence needs logits
    last_token_indices = query_start_locs[1:] - 1
    lm_head.encode_selective(step_ctx, hidden, last_token_indices, logits, num_seqs)
```

**Feature Flag**: `VLLM_METAL_SELECTIVE_LM_HEAD=1`

---

## Part 4: Fused O-Projection + Residual (10-15% Per Layer)

### 4.1 Current Flow (runner.py:725-740)

```python
# Separate operations:
layer_ops.o_proj.encode(step_ctx, attn_output, hidden_states, num_tokens)
step_ctx.memory_barrier()
self._elementwise.encode_residual_add(step_ctx, hidden_states, residual, hidden_states, ...)
```

### 4.2 Fused Implementation

**Option A**: Use GEMM with beta=1.0
```python
def encode_with_residual(self, step_ctx, attn_output, residual, output, num_tokens):
    # Copy residual to output first
    self._elementwise.encode_copy(step_ctx, residual, output, ...)
    step_ctx.memory_barrier()

    # GEMM with beta=1.0: C = A @ B + C
    self._gemm.encode(step_ctx, attn_output, self._weights, C=output,
                      alpha=1.0, beta=1.0, ...)
```

**Option B**: Custom fused kernel
```metal
kernel void gemm_add_residual(...) {
    // Compute GEMM and add residual in single pass
}
```

**Feature Flag**: `VLLM_METAL_FUSED_O_PROJ_RESIDUAL=1`

---

## Part 5: Memory Barrier Reduction (5-10%)

### 5.1 Current State

11 memory barriers per layer in runner.py

### 5.2 Analysis

**Required barriers** (true dependencies):
- After MPS GEMM before compute kernel
- After KV write before attention
- After producer kernel before consumer

**Potentially unnecessary**:
- Between consecutive compute kernels on different buffers
- Between consecutive MPS operations

### 5.3 Implementation

```python
def memory_barrier_if_needed(self, step_ctx, prev_op, next_op, shared_buffers):
    """Conditional barrier based on actual dependencies."""
    needs_barrier = (
        prev_op.is_mps and not next_op.is_mps or  # MPS -> compute transition
        len(shared_buffers) > 0  # Producer-consumer relationship
    )
    if needs_barrier:
        step_ctx.memory_barrier()
```

**Feature Flag**: `VLLM_METAL_REDUCE_BARRIERS=1`

---

## Part 6: Testing Strategy

### 6.1 Unit Tests

```python
# tests/metal/test_gemm_metal.py
class TestGEMMMetalCorrectness:
    @pytest.mark.parametrize("M,K,N", [
        (1, 4096, 4096),      # Single token
        (128, 4096, 12288),   # QKV shape
        (256, 11008, 4096),   # MLP down
        (17, 4095, 4097),     # Non-aligned
    ])
    def test_vs_numpy_reference(self, M, K, N):
        A = np.random.randn(M, K).astype(np.float16)
        B = np.random.randn(N, K).astype(np.float16)
        expected = A @ B.T
        actual = run_metal_gemm(A, B, transpose_B=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)
```

### 6.2 Performance Tests

```python
class TestGEMMMetalPerformance:
    def test_vs_mps_baseline(self):
        shapes = [(128, 4096, 4096), (256, 4096, 12288), (1, 4096, 128256)]
        for M, K, N in shapes:
            mps_tflops = benchmark_mps(M, K, N)
            metal_tflops = benchmark_metal(M, K, N)
            # Phase 3 target: within 5% of MPS
            assert metal_tflops >= mps_tflops * 0.95
```

### 6.3 End-to-End Tests

```bash
# Full model correctness
VLLM_GEMM_BACKEND=metal VLLM_METAL_STRICT_NO_MPS=1 \
python -m pytest tests/e2e/test_model_output.py

# Benchmark
VLLM_GEMM_BACKEND=metal \
python benchmarks/test_devstral_24b.py --batch-sizes 1 4 8 16
```

---

## Part 7: Rollout Plan (PARALLEL APPROACH)

### Track A: Custom Metal GEMM (Primary)

**Week 1-2: Foundation**
- [ ] Implement naive GEMM kernel (Phase 1)
- [ ] Add unit tests for correctness
- [ ] Set up benchmark infrastructure

**Week 2-3: Tiled Kernel**
- [ ] Implement tiled GEMM (Phase 2)
- [ ] Tune tile sizes
- [ ] Benchmark vs MPS

**Week 3-4: simdgroup Optimization**
- [ ] Implement simdgroup kernel (Phase 3)
- [ ] Add bank conflict avoidance
- [ ] Achieve MPS parity

**Week 4-5: Specialized Variants**
- [ ] Small-M kernel for decode
- [ ] Large-N kernel for LM head
- [ ] Kernel selection logic

### Track B: Quick Wins (Parallel with Track A)

**Week 1-2: Top-K Logits (60-90% readback reduction)**
- [ ] Create topk_selection.metal kernel
- [ ] Create topk.py operation
- [ ] Integrate with lm_head.py
- [ ] Test readback size reduction

**Week 2-3: Selective LM Head (30-50% prefill speedup)**
- [ ] Create gather_hidden_states kernel
- [ ] Update lm_head.py encode_selective()
- [ ] Integrate with runner.py prefill path
- [ ] Benchmark prefill improvement

**Week 3-4: Fused Operations**
- [ ] Fused O-proj + residual
- [ ] Memory barrier reduction analysis

### Track C: Integration (After Track A + B)

**Week 5-6: Full Integration**
- [ ] Integrate all optimizations into runner.py
- [ ] End-to-end testing with all features
- [ ] Performance regression tests
- [ ] Documentation and GEMM_METAL.md finalization

---

## Part 8: Critical Files

| File | Purpose | Changes |
|------|---------|---------|
| `engine/ops/gemm.py` | Current MPS GEMM | Keep as fallback |
| `engine/ops/gemm_metal.py` | **NEW** Custom Metal GEMM | Create |
| `engine/ops/gemm_selector.py` | **NEW** Unified interface | Create |
| `engine/ops/topk.py` | **NEW** Top-K selection | Create |
| `engine/ops/lm_head.py` | LM head projection | Add selective/topk |
| `engine/ops/qkv.py` | O-projection | Add fused residual |
| `engine/runner.py` | Main execution | Integrate optimizations |
| `engine/config.py` | Configuration | Add new env vars |
| `metal/kernels/custom_gemm.metal` | **NEW** Metal kernels | Create |
| `metal/kernels/topk_selection.metal` | **NEW** Top-K kernel | Create |

---

## Part 9: Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_GEMM_BACKEND` | auto | GEMM backend: auto, mps, metal |
| `VLLM_METAL_GEMM_DEBUG` | 0 | Enable debug logging |
| `VLLM_METAL_GEMM_VERIFY` | 0 | Verify against MPS |
| `VLLM_METAL_TOPK_LOGITS` | None | Top-k for logits readback |
| `VLLM_METAL_SELECTIVE_LM_HEAD` | 1 | Enable selective LM head |
| `VLLM_METAL_FUSED_O_PROJ_RESIDUAL` | 1 | Enable fused O-proj+residual |
| `VLLM_METAL_REDUCE_BARRIERS` | 0 | Enable barrier reduction |

---

## Part 10: Success Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Decode tok/s (batch 1) | 12.7 | 25+ | Custom GEMM |
| Decode tok/s (batch 8) | 88.6 | 150+ | Custom GEMM + optimizations |
| Prefill tok/s (batch 8) | 498.5 | 800+ | Selective LM head |
| Logits readback time | 100% | 10% | Top-K |
| Per-layer time | 100% | 85% | Fused ops + barriers |

---

## Conclusion

This plan provides a comprehensive roadmap to unlock performance in the vLLM-Apple Metal Engine. The primary focus is replacing MPS GEMM with custom Metal kernels using simdgroup_matrix operations. Additional optimizations (top-k logits, selective LM head, fused operations) provide further gains.

All changes are behind feature flags for safe rollout with MPS fallback always available.

**Ready for implementation.**

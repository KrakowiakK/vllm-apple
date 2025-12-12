# Plan Optymalizacji vLLM-Apple dla 50+ TPS

## Current State (2024-12-11)
- **Qwen3-30B-A3B**: 4.2 TPS (chunked bmm MoE)
- **Theoretical limit** (PyTorch ops only): 6.5 TPS
- **Target**: 50+ TPS

## Performance Breakdown (per decode step)

| Component | Time/Layer | Total (48 layers) | % of Time |
|-----------|------------|-------------------|-----------|
| MoE       | 1.46ms     | 70ms              | 45%       |
| Attention | 1.16ms     | 56ms              | 36%       |
| RMSNorm   | 0.60ms     | 29ms              | 19%       |
| **Total** | 3.22ms     | 154ms             | 100%      |

Theoretical TPS: 1000/154 = 6.5 TPS
Actual vLLM TPS: 4.2 TPS (35% pipeline overhead)

## Optimization Strategy

### Phase 1: Reduce vLLM Pipeline Overhead (Target: 6-7 TPS)
**Goal**: Match theoretical limit by reducing overhead

1. [ ] Profile actual vLLM execution to find overhead sources
2. [ ] Optimize `_build_attn_metadata()` - currently creates many small tensors
3. [ ] Reduce Python overhead in model runner loop
4. [ ] Cache reusable tensors between steps

### Phase 2: Metal MoE Kernel (Target: 15-20 TPS)
**Goal**: Replace chunked bmm with fused Metal kernel

Current MoE bottlenecks:
- `w1[chunk_ids]` - index_select creates new tensor
- `torch.bmm` - not optimized for grouped GEMM
- Multiple kernel launches per chunk

Metal kernel approach:
1. [ ] Single kernel for all expert computations
2. [ ] Avoid memory allocation in hot path
3. [ ] Use Metal's simdgroup_matrix for matmul

Implementation steps:
1. [ ] Create `metal_moe.metal` shader file
2. [ ] Write Python bindings using PyObjC or custom extension
3. [ ] Benchmark vs chunked bmm
4. [ ] Integrate into vLLM worker

### Phase 3: Attention Optimization (Target: 25-30 TPS)
**Goal**: Reduce attention overhead

Current issues:
- SDPA not using Flash Attention on MPS
- KV cache gathering via Python indexing
- Per-sequence loop in decode

Optimizations:
1. [ ] Use contiguous KV cache layout
2. [ ] Batch multiple sequences in single SDPA call
3. [ ] Consider Metal attention kernel for decode

### Phase 4: MLX Integration (Target: 50+ TPS)
**Goal**: Leverage MLX's optimized Apple Silicon kernels

MLX advantages:
- Native Metal compute shaders
- Lazy evaluation reduces memory traffic
- Optimized for Apple Silicon architecture

Approach:
1. [ ] Create MLX model wrapper
2. [ ] Use MLX for compute, vLLM for serving
3. [ ] Hybrid: MLX for MoE, PyTorch for attention

## Implementation Order

### Immediate (today):
1. Profile vLLM to identify overhead sources
2. Start Metal MoE kernel prototype

### Short-term (this week):
3. Complete Metal MoE kernel
4. Optimize attention metadata building
5. Target: 15 TPS

### Medium-term:
6. MLX integration exploration
7. Full attention optimization
8. Target: 30+ TPS

## Test Protocol

For each optimization:
1. Isolated benchmark (no vLLM)
2. vLLM integration test
3. Correctness verification (compare outputs)
4. Memory usage check
5. Update KNOWLEDGE_BASE.md with findings

## Files to Modify/Create

- `vllm_apple/ops/metal_moe.metal` - Metal shader
- `vllm_apple/ops/metal_moe.py` - Python bindings
- `vllm_apple/v1/worker/apple_worker.py` - Integration
- `vllm_apple/v1/worker/apple_model_runner.py` - Attention optimizations
- `tests/benchmarks/benchmark_moe.py` - Benchmark suite

# MoE Readiness Checklist

**Status:** ALL SYSTEMS GO (v2.6-stable)
**Date:** 2025-12-16

## 1. Guarantees Satisfied
The following prerequisites for MoE routing are verified:

- [x] **Base Model Correctness**: Standard Dense Transformer execution is numerically equivalent to PyTorch.
- [x] **Memory Stability**: No leaks observed in long-running batch tests (Validation Suite).
- [x] **Latency Budget**: Decoding overhead is low enough (<5ms kernel overhead) to support router checks.
- [x] **Custom Ops**: Mechanism to dispatch custom Metal kernels (e.g., `execute_moe_layer`) is established via `step_context` and `ops/` structure.

## 2. MoE Integration Guidelines
When implementing MoE (e.g., Mixtral, Qwen-MoE):

1. **Routing**: Implement the router in `runner.py` (or `model_runner`) logic.
2. **Expert Dispatch**: Use `MPSMatrixMultiplication` for experts.
   - For small batch sizes, sequentially encoding experts is acceptable.
   - For large batches, explore batched GEMM or specialized sorting kernels later.
3. **Synchronization**: Maintain the "Encode-Only" invariant. Do NOT read back router decisions to CPU if possible. Use GPU-side predicates or indirect dispatch (advanced) if needed.
   - *Phase 1 Strategy*: Read router logits to CPU, dispatch expert commands from CPU (Hybrid). This is fine for initial validaton.

## 3. Invariants to Protect
**DO NOT BREAK**:
- `PagedAttention` correctness (Golden Record).
- `kv_write` layout (`[Blocks, Heads, Tokens, Dim]`).
- Pre-O-Proj capture hooks (for debugging experts).

## 4. Required Regression Tests
Before merging any MoE PR, you MUST run:
1. `tests/equivalence/test_layer_equivalence.py` (Ensures Dense path didn't break).
2. `tests/integration/test_chunking.py` (Ensures batching logic holds).

Signed,
*vLLM Apple Metal Maintenance Team*

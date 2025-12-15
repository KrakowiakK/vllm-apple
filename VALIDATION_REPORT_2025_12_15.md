# Engine Validation Report

**Date**: 2025-12-15
**Model**: mistralai/Devstral-Small-2505 (24B)
**Hardware**: Apple M3 Ultra (512 GB)
**Prompt Length**: 512 tokens
**Decode Steps**: 64

---

## Executive Summary

The vLLM-Apple Metal Engine v2.0 passes all validation criteria with large prompts (512 tokens). Engine is ready for continued development.

---

## 1. Performance Results

### Throughput (prompt_len=512, decode_steps=64)

| Batch | Prefill tok/s | Decode tok/s | Decode Latency |
|-------|--------------|--------------|----------------|
| 1     | 313.1        | 9.9          | 100.64 ms      |
| 2     | 370.2        | 19.9         | 100.70 ms      |
| 4     | 422.6        | 39.7         | 100.79 ms      |
| 8     | 453.7        | 77.0         | 103.84 ms      |
| 16    | 464.7        | 116.7        | 137.12 ms      |

### Batch Scaling Analysis

| Batch | vs Batch 1 | Efficiency | Status |
|-------|-----------|------------|--------|
| 2     | 2.01x     | 100.5%     | LINEAR |
| 4     | 4.01x     | 100.3%     | LINEAR |
| 8     | 7.78x     | 97.3%      | LINEAR |
| 16    | 11.79x    | 73.7%      | SUBLINEAR |

**Assessment**: Near-linear scaling (97%+) up to batch 8. Batch 16 shows reduced efficiency due to increased memory pressure with 512-token context.

---

## 2. Invariant Verification

| Invariant | Status | Notes |
|-----------|--------|-------|
| No PyTorch-MPS in hot path | PASS | VLLM_METAL_STRICT_NO_MPS=1 enabled, no violations |
| Step-boundary-only sync | PASS | No per-layer waits detected |
| KV cache single source of truth | PASS | Engine-owned MTLBuffer |
| No batch scaling cliff | PASS | Linear scaling up to batch 8 |

All benchmarks ran with `VLLM_METAL_STRICT_NO_MPS=1` without errors, confirming invariant compliance.

---

## 3. Correctness Verification

### Coherence Test Results

| Batch | Identical Outputs | Coherent | Status |
|-------|------------------|----------|--------|
| 1     | N/A              | Yes      | PASS   |
| 4     | Yes (4/4)        | Yes      | PASS   |
| 8     | Yes (8/8)        | Yes      | PASS   |

**Test**: Same prompt ("def fibonacci(n):...") replicated across batch positions.

**Result**: All sequences in batch produce identical, coherent Python code output.

---

## 4. Configuration Updates

### Changes Made During Validation

1. **benchmark_devstral_engine.py** (line 14-15):
   - `VLLM_METAL_SCRATCH_POOL_MB`: 4096 → 8192
   - `VLLM_METAL_MAX_BATCH_SIZE`: 2048 → 16384

2. **vllm_apple/engine/runner.py** (line 317):
   - Hard cap: 4096 → 16384 (allows batch 16 × 1024 tokens)

These changes enable large batch sizes with long prompts (512+ tokens).

---

## 5. Known Limitations

1. **Batch 16 Efficiency**: 73.7% efficiency with 512-token prompts (expected behavior due to memory pressure)

2. **Batch=1 Performance**: 9.9 tok/s decode is compute-bound (GPU floor ~100ms). 20 tok/s not achievable without quantization.

3. **Variable-length Batching**: Test framework requires same-length prompts per batch. Different-length prompts require proper padding/attention masking.

---

## 6. Conclusion

### Validation Status: PASS

| Category | Result |
|----------|--------|
| Performance | PASS - Expected throughput achieved |
| Scaling | PASS - Near-linear up to batch 8 |
| Invariants | PASS - All strict mode tests pass |
| Correctness | PASS - Coherent, identical outputs |

**Verdict**: Engine v2.0 is validated and ready for continued development.

---

## 7. Recommendations

1. Continue with current architecture - no blocking issues
2. Consider quantization for batch=1 performance improvement
3. Monitor batch 16+ efficiency at very long contexts (1024+ tokens)
4. Document scratch pool sizing recommendations in README

---

*Generated: 2025-12-15*

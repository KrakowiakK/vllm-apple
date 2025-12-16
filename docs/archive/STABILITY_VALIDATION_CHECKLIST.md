# vLLM-Apple Metal Engine v2.6 - Stability Validation Checklist

**Purpose:** Pre-MoE Stability Gate
**Date:** 2025-12-15
**Status:** BLOCKER IDENTIFIED - ENGINE PREFILL DIVERGENCE

---

## SECTION A: MUST-FIX ISSUES (BLOCKERS)

### A.1: Engine Prefill Numerical Divergence

**Status:** BLOCKER - ROOT CAUSE IDENTIFIED

**Previous Symptom:** "Mixed Prompt Length Bug" - certain prompt lengths producing early EOS.

**Definitive Root Cause:** Engine prefill produces different hidden states than PyTorch reference, causing wrong token generation.

**Evidence:**

| Test | PyTorch Path | Engine Path |
|------|--------------|-------------|
| 500-token "fox" prompt | 10 tokens (correct) | 9 tokens, ends with EOS |
| First token comparison | `10575` | `1531` (completely different) |
| Batch vs individual | Both work | Both fail (not batch-related) |

**Key Finding:** Bug is NOT batch-related. Individual prompts also fail when using engine prefill.

**Content Sensitivity Test (500 tokens):**
```
Template     Status
fox          FAIL (9 tokens, EOS)
code         OK (32 tokens)
story        OK (32 tokens)
instruction  OK (32 tokens)
```

**Likely Divergence Sources:**
1. **RoPE Application** (HIGH) - Metal kernel vs PyTorch position encoding
2. **Attention Computation** (HIGH) - SDPA vs Metal attention numerical differences
3. **GEMM Precision** (MEDIUM) - Metal kernels vs MPS matmul bfloat16 handling
4. **Chunked Prefill Boundary** (MEDIUM) - KV cache state between chunks

**Immediate Workaround:**
```bash
# Use PyTorch prefill + Engine decode
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=0
```

**Files to Investigate:**
| File | Area | Priority |
|------|------|----------|
| `vllm_apple/engine/ops/elementwise.py` | RoPE implementation | HIGH |
| `vllm_apple/v1/attention/backends/metal_attn.py` | Attention computation | HIGH |
| `vllm_apple/engine/ops/gemm_metal.py` | GEMM precision | MEDIUM |
| `vllm_apple/v1/worker/apple_model_runner.py` | Prefill orchestration | MEDIUM |

**Fix Criteria:**
- [x] Root cause identified with code evidence
- [ ] Divergence layer identified (RoPE, attention, or GEMM)
- [ ] Fix implemented
- [ ] Regression test added: `test_engine_pytorch_equivalence.py`
- [ ] Engine prefill matches PyTorch output for all test prompts

---

## SECTION B: REQUIRED ADDITIONAL TESTS

### B.1: Engine vs PyTorch Numerical Equivalence

| Test | Engine Prefill | PyTorch Prefill |
|------|----------------|-----------------|
| 100-token prompt | [ ] Match | [x] Reference |
| 500-token prompt | [ ] Match | [x] Reference |
| 1000-token prompt | [ ] Match | [x] Reference |
| Mixed content prompts | [ ] Match | [x] Reference |

### B.2: Prefill Profiling Matrix (Blocked)

| Prompt Length | Batch 1 | Batch 4 | Batch 8 |
|---------------|---------|---------|---------|
| 32            | [ ]     | [ ]     | [ ]     |
| 128           | [ ]     | [ ]     | [ ]     |
| 512           | [ ]     | [ ]     | [ ]     |
| 2048          | [ ]     | [ ]     | [ ]     |
| 8192          | [ ]     | [ ]     | [ ]     |
| 16384         | [ ]     | [ ]     | [ ]     |

**Status:** BLOCKED until engine prefill produces correct outputs

### B.3: Long-Running Stability (Decode-Only Mode)

With `VLLM_APPLE_ENGINE_PREFILL=0`:

| Configuration | Duration | Acceptance |
|---------------|----------|------------|
| Batch=1 | 30 min | Memory growth < 0.1 GB |
| Batch=8 | 30 min | Memory growth < 0.1 GB |
| Batch=1 | 60 min | Latency variance < 5% |

### B.4: Batch Edge Cases (Decode-Only Mode)

| Batch Size | Uniform Length | Mixed Length |
|------------|----------------|--------------|
| 3 | [ ] | [ ] |
| 5 | [ ] | [ ] |
| 7 | [ ] | [ ] |
| 9 | [ ] | [ ] |
| 12 | [ ] | [ ] |
| 15 | [ ] | [ ] |

---

## SECTION C: OPTIONAL VALIDATIONS (NON-BLOCKING)

- [ ] Layer-by-layer hidden state comparison (engine vs PyTorch)
- [ ] SDPA vs Metal kernel output comparison
- [ ] RoPE output comparison at various positions
- [ ] GEMM output comparison (Metal vs MPS)

---

## SECTION D: EXPLICITLY OUT OF SCOPE

| Item | Reason |
|------|--------|
| Batch=1 Performance Tuning | Compute-bound (~10-11 tok/s), no optimization path |
| GEMM / MPS Backend Changes | Stability freeze (but may be needed for fix) |
| Quantization | Feature work, blocked on stable baseline |
| Speculative Decoding | Feature work, blocked on stable baseline |
| MoE Routing | Blocked until Section A resolved |

**Statement:** These are intentionally deferred until after a stable baseline is frozen.

---

## SECTION E: DEFINITION OF DONE

### Tag: `v2.6-stable`

**Blockers Resolved:**
- [x] Engine prefill divergence root cause identified
- [ ] Divergence layer pinpointed (RoPE/attention/GEMM)
- [ ] Engine prefill produces identical outputs to PyTorch
- [ ] Regression test `test_engine_pytorch_equivalence.py` added and passing

**Required Tests Passed:**
- [ ] Engine vs PyTorch equivalence (B.1)
- [ ] Prefill matrix completed (B.2) - after fix
- [ ] Long-running stability 60 min, memory growth < 0.1 GB (B.3)
- [ ] Batch edge cases verified (B.4)

**Existing Tests Remain Green:**
- [x] Context knee point (512→8192) — no degradation
- [x] Memory stability — 0 growth
- [ ] Numerical correctness — BLOCKED by prefill divergence
- [x] Token cap boundary — all cases correct
- [x] Determinism — identical outputs (within same mode)

**Documentation:**
- [x] MIXED_LENGTH_BUG_DIAGNOSIS.md updated with definitive root cause
- [ ] VALIDATION_REPORT_v2.md updated
- [ ] Known issues documented
- [ ] Release notes drafted

---

## APPENDIX: Debug Commands

```bash
# Reproduce bug with engine prefill
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 \
VLLM_METAL_SCRATCH_POOL_MB=8192 VLLM_METAL_MAX_BATCH_SIZE=16384 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
python debug_mixed_length.py

# Test with PyTorch prefill (workaround)
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
python debug_mixed_length.py

# Full PyTorch reference (no engine)
VLLM_APPLE_USE_ENGINE=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
python debug_mixed_length.py

# Compare outputs at specific prompt length
python -c "
from vllm import LLM, SamplingParams
import os

# Test both paths
for mode in ['pytorch', 'engine']:
    if mode == 'pytorch':
        os.environ['VLLM_APPLE_USE_ENGINE'] = '0'
    else:
        os.environ['VLLM_APPLE_USE_ENGINE'] = '1'
        os.environ['VLLM_APPLE_ENGINE_PREFILL'] = '1'

    llm = LLM(model='mistralai/Devstral-Small-2505', trust_remote_code=True)
    prompt = 'The quick brown fox jumps over the lazy dog. ' * 50
    outputs = llm.generate([prompt], SamplingParams(max_tokens=10))
    print(f'{mode}: {list(outputs[0].outputs[0].token_ids)}')
"
```

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Engineer | | | |
| Reviewer | | | |
| Release Gate | | | |

---

*Document Version: 2.0*
*Updated: 2025-12-15*
*Root Cause Status: IDENTIFIED - Engine Prefill Numerical Divergence*

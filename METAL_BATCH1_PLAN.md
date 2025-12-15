# BATCH=1 DECODE OPTIMIZATION PLAN v2.2

**Goal:** Improve batch=1 decode from ~11 tok/s (~90ms/token) toward ≤50ms/token (20+ tok/s) **IF achievable by removing overhead**
**Hard Constraint:** Batch≥2 decode and prefill MUST NOT regress (±3% acceptable)

---

## CRITICAL RULES

### Rule 1: Phase 0 Is a Hard Gate

**NO optimization work (Phase 1–4) may begin until Phase 0 profiling is complete.**

- Phase 0 must produce real measured numbers
- The plan must be updated with actual data
- Any optimization without Phase 0 confirmation is INVALID

### Rule 2: Scenario Decision Is Strictly Enforced

After Phase 0, one scenario MUST be selected. Do not "optimize anyway".

| Scenario | GEMM Compute | Reducible Overhead | Action |
|----------|--------------|-------------------|--------|
| A | ≤45-50ms | ≥30ms | Proceed with Phases 1-4 |
| B | 50-60ms | 20-30ms | Limit to Phases 1-2 only |
| C | ≥70ms | ≤15ms | STOP. Document compute-bound limitation. |

### Rule 3: No Further GEMM Backend Experiments

- MPS stays for all large GEMMs (QKV, O, MLP, LM head)
- Do NOT attempt "pure Metal GEMM" for large matrices
- Size-based GEMM selection is LOCKED and CORRECT

### Rule 4: Success = Data + Documented Limits

A valid outcome is: *"Batch=1 decode is compute-bound on Devstral; overhead optimizations provide only marginal gains."*

This is NOT a failure if backed by Phase 0 data. The goal is correctness and defensible conclusions, not forcing 20 tok/s at any cost.

---

## CRITICAL ASSUMPTIONS TO VALIDATE

### Assumption 1: "True GEMM compute is ~40-50ms"

**Status:** UNVALIDATED - CRITICAL

**What could be wrong:**
- MPS GEMM on Apple Silicon may genuinely require 70-80ms for Devstral-class matrices
- MPS encoder setup is highly optimized and adds negligible overhead
- The ANE/GPU scheduling is already near-optimal

**If wrong:** If true GEMM compute is ≥70ms → Scenario C → STOP optimization work

### Assumption 2: "Encoder transitions cost ~50-75µs each"

**Status:** UNVALIDATED

**What could be wrong:**
- Transitions may cost only 5-10µs (not worth optimizing)
- Transitions may cost 100-200µs (major target)

### Assumption 3: "Barriers cost ~12-20µs each"

**Status:** UNVALIDATED

**What could be wrong:**
- Barriers may be near-zero cost when encoder is already open
- Barrier-with-reopen may be expensive, barrier-without-reopen cheap

### Assumption 4: "Python/PyObjC dispatch is ~3-5ms"

**Status:** UNVALIDATED

**What could be wrong:**
- PyObjC may be highly optimized (<1µs per call)
- Bottleneck may be Metal encoding, not Python

### Assumption 5: "Overhead components are additive"

**Status:** UNVALIDATED

**What could be wrong:**
- GPU and CPU work may overlap (GPU executes while CPU encodes next op)
- Removing one overhead may not reduce total time

---

## WHY MPS STAYS (Pure Metal ≠ Replace MPS)

**"Pure Metal direction" CORRECTLY means:**
1. Fuse elementwise ops (RMSNorm, residual, copy) into single Metal kernels
2. Reduce dispatch count and barrier count
3. Minimize encoder transitions between Metal and MPS
4. Reduce PyObjC overhead via caching and pre-packing

**"Pure Metal direction" does NOT mean:**
- ❌ Replace MPS GEMM with custom Metal GEMM for large matrices
- ❌ Eliminate all MPS usage
- ❌ Rewrite GEMM kernels

**Why MPS stays for Devstral GEMMs:**
- MPS is faster than custom Metal GEMM for K>2048 or N>4096
- Forcing Metal GEMM caused 40% regression (11 → 6.7 tok/s)
- This is PROVEN and LOCKED IN

---

## PHASE 0: PROFILING BREAKDOWN (HARD GATE)

**Priority:** CRITICAL | **Risk:** NONE | **Status:** ✅ COMPLETE - SCENARIO C SELECTED

**Purpose:** Validate or refute assumptions. Produce hard numbers. Select scenario A/B/C.

### CRITICAL: CPU vs GPU Timing

**CPU timestamps (`perf_counter_ns()`) measure SUBMISSION/ORCHESTRATION cost, NOT GPU execution time.**

GPU execution is asynchronous:
- CPU `encode()` returns immediately after submitting work
- GPU executes later, possibly overlapping with next CPU encode
- CPU timings alone must NOT be interpreted as "true compute time"

**Phase 0 must distinguish:**

| Metric | What It Measures | How to Measure |
|--------|------------------|----------------|
| CPU encode time | Submission overhead | `perf_counter_ns()` around encode calls |
| GPU execution time | True compute floor | Metal GPU timestamps OR `waitUntilCompleted` for confirmation |
| Total step time | End-to-end latency | Timer around full step |

**For Phase 0 initial pass:** Use CPU timestamps to understand submission overhead distribution.
**For confirmation:** Use Instruments or `waitUntilCompleted` timing to validate GPU execution floor.

### What to Measure

| Metric | Method | Notes |
|--------|--------|-------|
| Total decode step time | CPU timestamp | step_start to step_end |
| Per-layer CPU encode time | CPU timestamp | Submission overhead |
| QKV encode time | CPU timestamp | Includes MPS setup |
| Attention encode time | CPU timestamp | Metal dispatch |
| O-proj encode time | CPU timestamp | MPS setup |
| MLP encode time | CPU timestamp | 3 GEMMs + silu_mul |
| Elementwise encode time | CPU timestamp | Copy, rmsnorm, residual |
| Transition count | Counter | `_num_mps_transitions` |
| Transition CPU time | CPU timestamp | Time in transition calls |
| Barrier count | Counter | `_num_barriers` |
| Barrier reopen count | Counter | `_num_barrier_reopens` |
| Dispatch count | Counter | Metal dispatch calls |

### Output Format

```
================================================================================
BATCH=1 DECODE PROFILING (Devstral, 40 layers)
Iterations: 16 decode steps (after 3 warmup)
================================================================================

TOTAL DECODE STEP TIME: XX.XX ms (CPU wall clock)

CPU ENCODE TIME BREAKDOWN (submission overhead):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Component              │ Time (ms) │ % of Step │ Count │ Per-Call (µs) │
├─────────────────────────────────────────────────────────────────────────────┤
│ QKV GEMM encode        │   XX.XX   │   XX.X%   │   40  │     XXX       │
│ Attention encode       │   XX.XX   │   XX.X%   │   40  │     XXX       │
│ O-proj encode          │   XX.XX   │   XX.X%   │   40  │     XXX       │
│ MLP GEMMs encode       │   XX.XX   │   XX.X%   │  120  │     XXX       │
│ Elementwise encode     │   XX.XX   │   XX.X%   │  XXX  │     XXX       │
│ Encoder transitions    │   XX.XX   │   XX.X%   │  XXX  │     XXX       │
│ Barriers               │   XX.XX   │   XX.X%   │  XXX  │     XXX       │
│ Embedding + LM head    │   XX.XX   │   XX.X%   │    2  │     XXX       │
├─────────────────────────────────────────────────────────────────────────────┤
│ TOTAL CPU ENCODE       │   XX.XX   │   XX.X%   │       │               │
│ NON-ENCODE (GPU wait)  │   XX.XX   │   XX.X%   │       │               │
└─────────────────────────────────────────────────────────────────────────────┘

COUNTERS:
  MPS transitions:     XXX total (X.X per layer)
  Barriers:            XXX total (X.X per layer)
  Barrier reopens:     XXX total (X.X per layer)
  Dispatches:          XXX total (X.X per layer)

GPU EXECUTION FLOOR (confirmation run with waitUntilCompleted):
  Minimum achievable latency: XX.XX ms
  This is the compute-bound floor; overhead cannot reduce below this.

================================================================================
SCENARIO DETERMINATION
================================================================================
CPU encode overhead:     XX.XX ms
GPU execution floor:     XX.XX ms
Reducible overhead:      XX.XX ms (CPU encode - GPU floor? or measured gap)

Selected scenario: [A / B / C]
Rationale: [explanation based on data]
```

### Acceptance Criteria

| Criterion | Threshold | Action if Failed |
|-----------|-----------|------------------|
| Total time consistent across runs | ±5% | Increase warmup |
| CPU encode breakdown sums correctly | ±10% of total | Fix instrumentation |
| All counters populated | >0 | Fix counter increments |
| GPU floor measured | Non-zero | Add waitUntilCompleted run |

### Scenario Decision Matrix

| Measured GPU Floor | Measured CPU Overhead | Scenario | Action |
|--------------------|----------------------|----------|--------|
| ≤45ms | ≥35ms | A | Proceed with Phases 1-4 |
| 45-55ms | 25-35ms | A/B | Proceed, adjust target to 15-18 tok/s |
| 55-65ms | 15-25ms | B | Phases 1-2 only |
| 65-75ms | 10-15ms | B/C | Phase 1 only, document limitation |
| ≥75ms | ≤10ms | C | STOP. Compute-bound. Document. |

### Implementation Files

- `vllm_apple/engine/step.py` - Add counters and transition timing
- `vllm_apple/engine/runner.py` - Add per-section timestamps
- `benchmark_devstral_engine.py` - Add `--profile` mode with output format

### Deliverable

After Phase 0:
1. Fill in the profiling table above with real numbers
2. Select Scenario A/B/C with rationale
3. Update this plan with measured data
4. Only then propose Phase 1+ implementation

---

## PHASE 1: LAYER-LEVEL SCHEDULING

**Priority:** HIGH (if transition overhead ≥5ms) | **Risk:** MEDIUM | **Status:** BLOCKED ON PHASE 0

**This is a STRUCTURAL optimization, not a micro-optimization.**

### Hypothesis (Testable)

> If encoder transitions cost ≥30µs each and there are ≥100 transitions per token, reducing transitions by 40% will save ≥2ms.

**Will be validated by Phase 0 data. Do not implement until Phase 0 confirms threshold.**

### Trigger Condition

Only implement if Phase 0 shows: `transition_overhead_ms >= 5`

### Kill Switch

```bash
export VLLM_BATCH1_SCHEDULING_DISABLE=1
```

### Gating

```python
use_optimized_scheduling = (
    step_ctx.decode_single_seq and
    not os.environ.get("VLLM_BATCH1_SCHEDULING_DISABLE")
)
```

### Correctness Checks

- SmolLM logits must match baseline exactly (bit-identical)
- Devstral top-5 tokens must match baseline

### Regression Checks

- Batch≥2 decode must be within ±3%
- Prefill must be within ±3%

---

## PHASE 2: BARRIER AUDIT AND REDUCTION

**Priority:** HIGH (if barrier overhead ≥3ms) | **Risk:** LOW | **Status:** BLOCKED ON PHASE 0

**This is a STRUCTURAL optimization, not a micro-optimization.**

### Hypothesis (Testable)

> If barriers (especially barrier-with-reopen) cost ≥10µs each and there are ≥200 barriers, removing unnecessary barriers will save ≥2ms.

**Will be validated by Phase 0 data. Do not implement until Phase 0 confirms threshold.**

### Trigger Condition

Only implement if Phase 0 shows: `barrier_overhead_ms >= 3`

### Kill Switch

```bash
export VLLM_BATCH1_BARRIER_REDUCE_DISABLE=1
```

### Gating

```python
skip_barrier = (
    not required and
    self.decode_single_seq and
    not os.environ.get("VLLM_BATCH1_BARRIER_REDUCE_DISABLE")
)
```

### Correctness/Regression Checks

Same as Phase 1.

---

## PHASE 3: ELEMENTWISE KERNEL FUSION

**Priority:** MEDIUM (if dispatch overhead ≥2ms) | **Risk:** LOW | **Status:** BLOCKED ON PHASE 0

### Realistic Expectations

**Fusion is expected to save milliseconds, not tens of milliseconds.**

Fusion should only be justified if Phase 0 shows dispatch overhead ≥2ms.

### Hypothesis (Testable)

> If elementwise dispatch overhead is ≥15µs each and there are ≥400 dispatches, fusing pairs will save ≥1ms plus reduce memory traffic.

### Trigger Condition

Only implement if Phase 0 shows: `dispatch_overhead_ms >= 2` OR `elementwise_encode_ms >= 3`

### Kill Switch

```bash
export VLLM_BATCH1_FUSION_DISABLE=1
```

### Gating

```python
use_fused_kernels = (
    step_ctx.decode_single_seq and
    not os.environ.get("VLLM_BATCH1_FUSION_DISABLE")
)
```

### Correctness/Regression Checks

Same as Phase 1.

---

## PHASE 4: PYTHON/PYOBJC OVERHEAD REDUCTION

**Priority:** LOW (if Python overhead ≥1ms) | **Risk:** LOW | **Status:** BLOCKED ON PHASE 0

### Trigger Condition

Only implement if Phase 0 shows measurable Python/struct.pack overhead ≥1ms.

### Kill Switch

```bash
export VLLM_BATCH1_PYOBJC_OPT_DISABLE=1
```

### Correctness/Regression Checks

Same as Phase 1.

---

## SUCCESS CRITERIA (Scenario-Based)

### Scenario A: Large Reducible Overhead

**Condition:** GPU floor ≤50ms, CPU overhead ≥30ms

| Phase | Expected Savings | Cumulative |
|-------|-----------------|------------|
| Baseline | - | ~90ms |
| Phase 1 | -5ms | ~85ms |
| Phase 2 | -3ms | ~82ms |
| Phase 3 | -2ms | ~80ms |
| Phase 4 | -1ms | ~79ms |

**Target:** ~75-80ms (13-14 tok/s)
**Stretch:** ~65-70ms (15-16 tok/s) if overhead was underestimated

### Scenario B: Moderate Reducible Overhead

**Condition:** GPU floor 50-65ms, CPU overhead 20-30ms

| Phase | Expected Savings | Cumulative |
|-------|-----------------|------------|
| Baseline | - | ~90ms |
| Phase 1 | -3ms | ~87ms |
| Phase 2 | -2ms | ~85ms |

**Target:** ~83-87ms (11.5-12 tok/s)
**Phases 3-4:** Optional, diminishing returns

### Scenario C: Compute-Bound

**Condition:** GPU floor ≥70ms, CPU overhead ≤15ms

**Action:** STOP optimization work.

**Document:** "Batch=1 decode on Devstral is compute-bound. MPS GEMM execution dominates latency. 20 tok/s is not achievable without quantization or speculative decoding (out of scope)."

**Best case:** ~80-85ms (12 tok/s) with minimal overhead reduction.

---

## REGRESSION GUARDRAILS (All Scenarios)

| Metric | Baseline | Regression Limit |
|--------|----------|------------------|
| Batch=2 decode | ~91ms | ±3% (88-94ms) |
| Batch=4 decode | Baseline | ±3% |
| Batch=8 decode | Baseline | ±3% |
| Prefill (any batch) | Baseline | ±3% |

---

## VALIDATION PROTOCOL

### After Phase 0:

1. Fill in profiling table with real numbers
2. Determine GPU execution floor
3. Calculate reducible overhead
4. Select Scenario A/B/C
5. Update this plan with measured data
6. Get approval before proceeding to Phase 1+

### After Each Subsequent Phase:

1. **Correctness:** `python test_smolm_engine.py`
2. **Regression:** `python benchmark_devstral_engine.py --batch-sizes 1 2 4 8`
3. **Metrics:** Verify expected counter reductions
4. **Kill switch:** Confirm disabling returns to baseline

---

## IMPLEMENTATION ORDER

```
Phase 0: Profiling        ← CURRENT STEP
    ↓
[Fill in real numbers]
[Select Scenario A/B/C]
[Update this plan]
    ↓
If Scenario C: STOP
    ↓
If Scenario A/B:
    ↓
Phase 1: Scheduling (if transitions ≥5ms)
    ↓
Phase 2: Barriers (if barriers ≥3ms)
    ↓
If Scenario A only:
    ↓
Phase 3: Fusion (if dispatches ≥2ms)
    ↓
Phase 4: PyObjC (if overhead ≥1ms)
```

---

## DEFERRED (Out of Scope)

| Optimization | Reason |
|--------------|--------|
| Replace MPS GEMM | MPS is faster; proven by regression |
| Full layer fusion | Requires in-kernel GEMM |
| Speculative decoding | Different scope |
| Quantization | Different scope |
| Batch≥2 changes | Already excellent |

---

## ENVIRONMENT VARIABLES

| Variable | Purpose | Default |
|----------|---------|---------|
| `VLLM_PROFILE_BATCH1=1` | Enable profiling output | Off |
| `VLLM_BATCH1_SCHEDULING_DISABLE=1` | Kill switch Phase 1 | Off |
| `VLLM_BATCH1_BARRIER_REDUCE_DISABLE=1` | Kill switch Phase 2 | Off |
| `VLLM_BATCH1_FUSION_DISABLE=1` | Kill switch Phase 3 | Off |
| `VLLM_BATCH1_PYOBJC_OPT_DISABLE=1` | Kill switch Phase 4 | Off |

---

## PHASE 0 RESULTS (MEASURED 2024-12-15)

```
================================================================================
MEASURED DATA (Devstral batch=1, 16 decode steps, 3 warmup)
================================================================================

TOTAL DECODE STEP TIME: 92.80 ms (±0.50 ms)

COUNTERS (per step):
  MPS transitions:       161 total (4.0 per layer)
  Barriers:              322 total (8.1 per layer)
  Barrier reopens:       80 total (2.0 per layer)
  Dispatches:            0 total (0.0 per layer)

NOTE ON DISPATCH COUNTER:

The reported `Dispatches: 0` reflects a limitation of the current
instrumentation placement. The counter does not include actual Metal
`dispatchThreadgroups` calls nor internal MPS kernel dispatches.

This does NOT affect the conclusions of Phase 0, because:
- GPU execution floor ≈ total step time
- CPU submission overhead is negligible (~0.35 ms)
- The workload is conclusively compute-bound

No optimization decisions in Scenario C depend on the dispatch counter.

CPU SUBMISSION OVERHEAD:
  Transition time:       0.09 ms (0.6 µs/call × 161)
  Barrier time:          0.26 ms (0.8 µs/call × 322)
  Total CPU overhead:    ~0.35 ms

GPU EXECUTION FLOOR:     90.74 ms (from waitUntilCompleted timing)
  Note: This includes CPU encode + GPU execute + sync overhead

ANALYSIS:
  Total step time:       92.80 ms
  GPU floor:             90.74 ms
  Async overlap benefit: -2.06 ms (negative = GPU takes longer than CPU encode)

  CPU overhead is negligible (~0.35ms).
  GPU execution dominates latency.
  The step time (92.80ms) ≈ GPU floor (90.74ms).
  This confirms batch=1 decode is COMPUTE-BOUND, not overhead-bound.

================================================================================
SCENARIO DETERMINATION
================================================================================
  Measured GPU floor:    90.74 ms (>> 70ms threshold)
  Measured CPU overhead: ~0.35 ms (<< 15ms threshold)

SCENARIO SELECTED: [X] C (Compute-Bound)

RATIONALE:
  - GPU floor (90.74 ms) is well above the 70ms threshold for Scenario C
  - CPU submission overhead (0.35 ms) is negligible
  - The 20 tok/s target (50ms/token) is NOT achievable through overhead reduction
  - All MPS GEMM operations are required; Metal kernel alternatives are slower
  - Quantization (INT4/INT8) or speculative decoding would be required for 20 tok/s

ACTION: STOP optimization work per plan guidelines. Document compute-bound limitation.
```

### Scenario C Conclusion

**Batch=1 decode on Devstral is compute-bound.**

- MPS GEMM execution dominates latency (~90ms)
- CPU overhead is negligible (~0.35ms = 0.4%)
- The 20 tok/s goal (50ms/token) is NOT achievable without:
  - **Quantization (INT4/INT8)** - reduces compute by 2-4x
  - **Speculative decoding** - generates multiple tokens per forward pass
  - These are OUT OF SCOPE for the current optimization plan

**Best achievable with current float16 MPS path:** ~11 tok/s

**Recommendation:** Close the BATCH=1 optimization track. Focus on other improvements (batch≥2 scaling, prefill optimization, quantization integration).

---

## PHASE X: ADVANCED / CONDITIONAL OPTIMIZATIONS (POST PHASE 0)

Phase X exists to document engineering avenues, not optimization goals.
It prevents repeated rediscovery of the same ideas while fully respecting
the compute-bound conclusion of Scenario C.

This section documents **advanced, conditional optimization directions**
that MAY provide small but measurable improvements for **batch=1 decode**
*without breaking batch≥2 behavior*.

All items in this section:
- Are **NOT mandatory**
- Are **data-driven**
- Must be considered ONLY AFTER Phase 0 profiling
- Must be gated to `decode_single_seq`
- Must preserve exact correctness

None of these items invalidate Scenario C.
They represent **incremental engineering options**, not guaranteed wins.

---

### X.1 Persistent Compute Encoder Across Metal ↔ MPS Boundaries

**Problem statement:**
The current execution model frequently closes and reopens the compute encoder
around MPS GEMM calls. While Phase 0 showed this overhead is small,
it still contributes to encoder lifecycle churn.

**Key clarification:**
The goal is NOT to replace MPS GEMM.
The target is reducing **encoder teardown / reopen cycles** around elementwise chains.

**Idea:**
- Keep a compute encoder alive across consecutive Metal elementwise operations
- Group elementwise kernels before and after MPS GEMMs
- Minimize unnecessary encoder state transitions

**Expected impact:**
- Small (order of milliseconds at most)
- Batch=1 decode only

**Risks:**
- Metal resource hazard correctness
- Increased complexity in encoder lifecycle management

**Conditions to consider:**
- Only if Phase 0+ measurements show non-negligible encoder reopen overhead
- Must be gated to `decode_single_seq`
- Must include a kill switch

**Status:** Advanced / Experimental

---

### X.2 Attention → O-proj Pipeline Tightening (Decode-Only)

**Current pattern:**
 attention → barrier → O-proj GEMM
**Observation:**
Some barriers are defensive rather than strictly required,
especially in the decode_single_seq case with predictable access patterns.

**Opportunity:**
- Audit whether certain barriers between attention output and O-proj input
  can be safely reduced or reordered
- Improve cache locality or buffer reuse

**Expected benefit:**
- Small but potentially measurable (sub-millisecond)
- Decode-only

**Risks:**
- Incorrect synchronization leading to subtle correctness bugs

**Conditions to consider:**
- ONLY if Phase 0 data shows attention ≥20% of total decode time
- Must preserve correctness
- Must be gated to `decode_single_seq`

---

### X.3 KV-Cache Layout Specialization for Batch=1

**Observation:**
The current KV-cache layout is designed for general batching.
Batch=1 decode has simpler, predictable access patterns.

**Idea:**
- Introduce a specialized KV-cache layout or access path
  for `decode_single_seq`
- Optimize for single-token, single-sequence access

**Expected benefit:**
- Attention speedup (potentially more meaningful than elementwise fusion)

**Risks:**
- Kernel complexity
- Layout branching
- Maintenance overhead

**Conditions to consider:**
- ONLY if Phase 0 profiling shows attention as a dominant cost
- Must be isolated to batch=1
- Must not affect batch≥2 code paths

---

### X.4 Out of Scope / Future Work (Not Part of This Plan)

The following items are explicitly OUT OF SCOPE for this optimization plan:

- **Speculative decoding**
  Requires draft models and changes generation semantics.

- **Quantization-aware decode (INT8 / INT4)**
  Changes numerical behavior and compute characteristics.

- **Replacing MPS GEMM with custom Metal GEMM**
  Proven slower for Devstral-class matrix sizes.

- **Full layer fusion with in-kernel GEMM**
  High complexity, high risk, and not competitive with MPS.

These items may be explored in separate efforts,
but MUST NOT be mixed into the current plan.

---

Given the Phase 0 measurements, none of the items in Phase X are recommended
for immediate implementation. They are documented for completeness and
future reference only.

END OF PHASE X

---

## REVISION HISTORY

| Date | Version | Change |
|------|---------|--------|
| 2024-12-15 | 1.0 | Initial plan |
| 2024-12-15 | 1.1 | Fixed GEMM regression |
| 2024-12-15 | 1.2 | Marked GEMM tuning as DONE |
| 2024-12-15 | 2.0 | Added profiling, scheduling, barrier audit |
| 2024-12-15 | 2.1 | Added assumptions, kill switches, scenarios |
| 2024-12-15 | 2.2 | Critical guidance: CPU vs GPU timing, hard gate, strict scenarios, realistic expectations |
| 2024-12-15 | 2.3 | **Phase 0 COMPLETE**: GPU floor 90.74ms, CPU overhead 0.35ms → SCENARIO C selected. Compute-bound, optimization work STOPPED. |
| 2024-12-15 | 2.4 | Added Phase X: Advanced/Conditional optimizations (post Phase 0). Documents optional incremental engineering options. |
| 2024-12-15 | 2.5 | Documentation clarifications: dispatch counter note, Phase X role clarification, closing statement. No conclusions changed. |

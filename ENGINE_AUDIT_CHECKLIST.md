# vLLM-Apple Metal Engine — Implementation Checklist (Engine Audit)

Use this checklist to review engine code, PRs, or refactors against the
vLLM-Apple Metal v2.0 plan (MTLBuffer engine + vLLM orchestrator).

This is NOT a feature checklist — it is an invariants & architecture checklist.

See also: `METAL_PLAN.md`

Legend: `[x]` done, `[ ]` not yet, `[~]` partial/temporary.

Current status (engine mode):
- Decode path uses `EngineRunner` (MTLBuffer engine).
- Prefill routes through PyTorch and then syncs KV to engine by default (temporary; violates “single source of truth”).
- Prefill/mixed steps can run in-engine with `VLLM_APPLE_ENGINE_PREFILL=1` (avoids KV sync).
- Decode inputs are prepared on CPU in engine mode (no implicit MPS→CPU boundary conversion).
- KV sync calls `torch.mps.synchronize()` only when PyTorch prefill is used (not plan-compliant; engine prefill avoids it).

---

## 1. Engine / Orchestrator Boundary

- [~] Is vLLM used strictly as an orchestrator?
  - scheduler (prefill/decode/continuous batching)
  - paged attention metadata (`block_table`, `slot_mapping`, `seq_lens`)
  - sampling / logits processing / API

- [~] Is *all* data-plane execution delegated to the engine?
  - attention
  - KV cache writes/reads
  - (progressively) QKV / RMSNorm / MLP

- [x] Is the engine interface explicit?
  - step descriptor passed explicitly:
    - `step_kind: prefill | decode`
    - `num_scheduled_tokens`
    - `num_seqs_active`
  - NO inference from tensor shapes or heuristics

❌ Red flags:
- engine code reading scheduler internals
- engine inferring batch/concurrency from `num_decode_tokens`
- PyTorch tensors flowing deep into engine ops

---

## 2. PyTorch-MPS Invariants (Hard Stop)

- [ ] No `torch.device("mps")` tensors cross the vLLM ↔ engine boundary
- [ ] No `torch.mps.synchronize()` anywhere in engine execution paths
- [ ] No `.item()`, `.tolist()`, `.cpu()` on MPS tensors
- [ ] Engine mode passes with:

```bash
VLLM_APPLE_USE_ENGINE=1 VLLM_METAL_STRICT_NO_MPS=1 python -m pytest ...
```

❌ Red flags:
- "temporary" MPS usage for convenience
- sync hidden inside helpers or profiling code
- MPSGraph used via PyTorch tensors (instead of raw `MTLBuffer`)

---

## 3. Synchronization Discipline (Most Important)

### Step-boundary-only rule

- [~] One command buffer per scheduler step (prefill OR decode)
- [~] Exactly ONE wait point per step (after submit, before returning outputs)
- [x] NO waits inside:
  - per-layer code
  - per-op code
  - kernel bridges
  - MPSGraph / MPSMatrix helpers

### Forbidden in ENCODE / SUBMIT phases

- [x] `waitUntilCompleted`
- [x] `waitUntilScheduled`
- [x] polling `commandBuffer.status`
- [x] `synchronizeResource` / `synchronizeTexture`
- [x] reading `MTLBuffer.contents` or `getBytes`
- [x] `didModifyRange`
- [x] creating or committing a command buffer inside a helper
- [x] any API that implicitly runs / commits / waits

✅ Allowed:
- GPU-side barriers (`memoryBarrierWithScope:MTLBarrierScopeBuffers`)
- encode-only APIs

❌ Red flags:
- "just one wait here for correctness"
- per-op command buffers
- helpers that "execute" instead of "encode"

---

## 4. Kernel Library & Pipeline Management (Metal API Hygiene)

- [ ] Engine first loads precompiled `.metallib`
- [x] Falls back to runtime compilation from `.metal`
- [x] Resource path is overridable via env var (e.g., `VLLM_METAL_PATH_RESOURCES`)
- [ ] `MTLCompileOptions` uses preprocessor macros for feature gating
- [~] Kernel variants use:
  - function constants (`MTLFunctionConstantValues`)
  - NOT runtime branching
- [~] Pipeline states are cached by:
  - kernel name
  - constant values
  - device family

❌ Red flags:
- recompiling kernels per step
- branching inside kernels for head_size/mask/etc
- pipeline cache keyed only by name

---

## 5. Buffer & Memory Model

- [~] Engine owns all runtime state via `MTLBuffer`
- [ ] KV cache has a SINGLE source of truth (engine-owned)
- [ ] Any vLLM KV tensors are stubs only (no full duplication)

### Storage modes

- [ ] `MTLStorageModePrivate` for:
  - KV cache
  - activations
  - intermediates
- [ ] `MTLStorageModeShared` only for:
  - CPU inputs
  - CPU-visible outputs (logits)

### Transfers

- [ ] All shared ↔ private transfers are explicit
- [ ] Transfers encoded via blit command encoders
- [ ] No implicit sync-based transfers in hot path

❌ Red flags:
- duplicated KV buffers (torch + Metal)
- CPU touching private buffers
- hidden clears or implicit staging

---

## 6. Attention & KV Semantics

- [x] `block_table` values are treated as PHYSICAL block IDs
- [x] Engine range-checks block IDs
- [x] KV overwrite semantics are explicit and safe
- [ ] No Python per-token or per-sequence loops in hot paths
- [ ] Decode prefers fused KV-write + attention
- [x] Non-fused fallback obeys step-boundary-only sync

❌ Red flags:
- remapping block IDs inside engine
- per-sequence SDPA loops
- decode behavior dependent on runtime heuristics

---

## 7. Engine Compute Coverage (Progressive)

### Phase 1 (minimum acceptable)

- [~] Engine executes paged attention + KV-write
- [x] Returns logits at step boundary
- [x] LM head may be temporary on CPU

### Phase 2+

- [x] QKV projections in-engine (GEMM on `MTLBuffer`)
- [x] Output projection in-engine
- [x] No CPU staging between QKV → attention → O-proj

### Phase 3+

- [x] RMSNorm / LayerNorm in-engine
- [x] Elementwise ops in-engine (residuals, activation)
- [x] MLP / FFN in-engine (dense path first)

❌ Red flags:
- bouncing tensors CPU ↔ GPU between ops
- mixing engine tensors and torch tensors mid-layer

---

## 8. Logits & Readback Strategy

- [x] Stable vLLM-facing contract: engine returns logits
- [x] Logits readback happens ONLY at step boundary
- [x] Acknowledged risk for large vocab readback

Optional (opt-in only):
- [ ] top-k logits path
- [ ] engine-side sampling (restricted)
- [ ] logits compression

❌ Red flags:
- partial readbacks mid-step
- implicit readback inside attention or MLP

---

## 9. Fallback & Safety

- [x] Engine can DECLINE unsupported configs
- [~] Fallback is explicit and graceful
- [ ] Strict mode does NOT silently fallback to MPS
- [ ] Errors never crash the server

❌ Red flags:
- silent backend switching
- catching and ignoring Metal errors

---

## 10. Tests & Gates

- [x] Unit tests for engine primitives
- [x] Metal kernel tests pass
- [x] Strict mode tests exist and fail on violations
- [ ] Batch scaling tested: 1 / 4 / 8 / 16
- [ ] No batch-8 performance cliff

---

## Final Sanity Question

If we answer "YES" to all of the above:
→ This is a true MTLBuffer-based engine
→ vLLM is a pure orchestrator
→ Maximum Apple Silicon performance is achievable

If any "red flag" appears:
→ The design is leaking back toward a PyTorch-MPS hybrid
→ Performance cliffs and sync issues WILL return

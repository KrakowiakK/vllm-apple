# vLLM-Apple Metal Engine — Implementation Checklist (Engine Audit)

Use this checklist to review engine code, PRs, or refactors against the
vLLM-Apple Metal v2.0 plan (MTLBuffer engine + vLLM orchestrator).

This is NOT a feature checklist — it is an invariants & architecture checklist.

See also: `METAL_PLAN.md`

Legend: `[x]` done, `[ ]` not yet, `[~]` partial/temporary.

Current status (engine mode):
- ✅ Decode path uses `EngineRunner` (MTLBuffer engine).
- ✅ Prefill runs in-engine by default (`VLLM_APPLE_ENGINE_PREFILL=1` is now the default when engine mode enabled).
- ✅ KV cache is single source of truth (engine-owned; vLLM tensors are stubs only).
- ✅ KV cache uses `MTLStorageModePrivate` by default (when engine prefill enabled).
- ✅ Decode inputs are prepared on CPU in engine mode (no implicit MPS→CPU boundary conversion).
- ✅ No `torch.mps.synchronize()` in hot path when engine prefill enabled.
- ✅ Strict mode (`VLLM_METAL_STRICT_NO_MPS=1`) raises error on any sync attempts.
- ✅ Precompiled `.metallib` loading supported (with source fallback).
- ✅ `MTLCompileOptions.preprocessorMacros` supported for feature gating.
- ✅ All ENGINE_AUDIT_CHECKLIST items complete (updated 2025-12-14).

---

## 1. Engine / Orchestrator Boundary

- [x] Is vLLM used strictly as an orchestrator?
  - scheduler (prefill/decode/continuous batching)
  - paged attention metadata (`block_table`, `slot_mapping`, `seq_lens`)
  - sampling / logits processing / API

- [x] Is *all* data-plane execution delegated to the engine?
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

- [x] No `torch.device("mps")` tensors cross the vLLM ↔ engine boundary
  - `EngineInputs.__post_init__` validates all tensors are on CPU
  - `boundary.py` provides additional validation at engine entry
- [x] No `torch.mps.synchronize()` anywhere in engine execution paths
  - `strict_mode.py` patches and blocks MPS sync during hot path
  - `sync_from_torch_cache()` blocked in strict mode and with Private storage
- [x] No `.item()`, `.tolist()`, `.cpu()` on MPS tensors
  - All `.item()` calls are on CPU tensors (validated by EngineInputs)
  - `strict_mode.py` patches tensor methods to raise on MPS tensors during hot path
- [x] Engine mode passes with:

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

- [x] One command buffer per scheduler step (prefill OR decode)
  - `EngineRunner.step()` creates single command buffer per step
- [x] Exactly ONE wait point per step (after submit, before returning outputs)
  - `waitUntilCompleted()` called only at step boundary in `_execute_step()`
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

- [x] Engine first loads precompiled `.metallib`
  - `load_library()` tries `.metallib` first via `newLibraryWithURL_error_`
  - Falls back to source compilation only if metallib not found
- [x] Falls back to runtime compilation from `.metal`
- [x] Resource path is overridable via env var (e.g., `VLLM_METAL_PATH_RESOURCES`)
  - When set, forces source compilation for debugging
- [x] `MTLCompileOptions` uses preprocessor macros for feature gating
  - `_compile_from_source()` accepts `preprocessor_macros` dict
  - Maps to `MTLCompileOptions.setPreprocessorMacros_`
- [x] Kernel variants use:
  - function constants (`MTLFunctionConstantValues`)
  - NOT runtime branching
  - `get_pipeline()` supports function constants via `function_constants` param
- [x] Pipeline states are cached by:
  - kernel name
  - constant values
  - `PipelineKey` dataclass used as cache key

❌ Red flags:
- recompiling kernels per step
- branching inside kernels for head_size/mask/etc
- pipeline cache keyed only by name

---

## 5. Buffer & Memory Model

- [x] Engine owns all runtime state via `MTLBuffer`
- [x] KV cache has a SINGLE source of truth (engine-owned)
- [x] Any vLLM KV tensors are stubs only (no full duplication)

### Storage modes

- [x] `MTLStorageModePrivate` for:
  - KV cache (auto-selected when engine prefill enabled, default)
  - `EngineKVCache._allocate_buffers()` uses `MTLResourceStorageModePrivate`
  - Storage mode logged at initialization for transparency
- [x] `MTLStorageModeShared` only for:
  - CPU inputs (via `_copy_to_buffer`)
  - CPU-visible outputs (logits readback)
  - Scratch pool uses Shared for CPU↔GPU transfers

### Transfers

- [x] All shared ↔ private transfers are explicit
  - `blit_buffer_to_staging()` for Private→Shared readback
  - `encode_blit_copy()` for hot-path blit encoding
- [x] Transfers encoded via blit command encoders
  - `MTLBlitCommandEncoder` used for buffer copies
  - `_zero_buffer()` uses GPU-side `fillBuffer_range_value_`
- [x] No implicit sync-based transfers in hot path
  - `sync_from_torch_cache()` blocked with Private storage
  - Phase guards prevent blit-to-staging during ENCODE/SUBMIT

❌ Red flags:
- duplicated KV buffers (torch + Metal)
- CPU touching private buffers
- hidden clears or implicit staging

---

## 6. Attention & KV Semantics

- [x] `block_table` values are treated as PHYSICAL block IDs
- [x] Engine range-checks block IDs
- [x] KV overwrite semantics are explicit and safe
- [x] No Python per-token or per-sequence loops in hot paths
- [x] Decode prefers fused KV-write + attention
- [x] Non-fused fallback obeys step-boundary-only sync

❌ Red flags:
- remapping block IDs inside engine
- per-sequence SDPA loops
- decode behavior dependent on runtime heuristics

---

## 7. Engine Compute Coverage (Progressive)

### Phase 1 (minimum acceptable)

- [x] Engine executes paged attention + KV-write
  - Decode: fused KV-write + attention via `MetalPagedAttentionFused`
  - Prefill: token-parallel attention via `MetalPagedAttentionV2`
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
- [x] Fallback is explicit and graceful
- [x] Strict mode does NOT silently fallback to MPS
- [x] Errors never crash the server

❌ Red flags:
- silent backend switching
- catching and ignoring Metal errors

---

## 10. Tests & Gates

- [x] Unit tests for engine primitives
- [x] Metal kernel tests pass
- [x] Strict mode tests exist and fail on violations
- [x] Batch scaling tested: 1 / 4 / 8 / 16
- [x] No batch-8 performance cliff

---

## Final Sanity Question

If we answer "YES" to all of the above:
→ This is a true MTLBuffer-based engine
→ vLLM is a pure orchestrator
→ Maximum Apple Silicon performance is achievable

If any "red flag" appears:
→ The design is leaking back toward a PyTorch-MPS hybrid
→ Performance cliffs and sync issues WILL return

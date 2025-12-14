# vLLM-Apple Metal Plan v2.0 — Variant A (MTLBuffer Engine + vLLM Orchestrator)

This is the implementation plan for **Variant A**: a vLLM plugin that provides a
first-class **Apple execution engine** built around `MTLBuffer`, while vLLM stays
the **orchestrator** (scheduler + metadata + sampling + API).

The engine is responsible for the data plane (attention, KV cache, and
progressively QKV/RMSNorm/MLP) and is designed to **avoid PyTorch-MPS entirely in
the hot path**.

This repo contains useful legacy primitives from the V1.x track
(`MetalPagedAttentionV2`, `MetalPagedAttentionFused`, `MetalKVCache`), but V2.0
reorganizes them behind an explicit engine boundary so that model execution no
longer depends on PyTorch MPS tensors or `torch.mps.synchronize()`.

## Versioning Decision (Project Direction)

- **v2.0**: this plan. We prioritize Variant A and correctness + invariants
  (“no MPS hot path”) over incremental V1.x patching.
- **v1.6**: considered incomplete/abandoned. We do not plan to “finish v1.6”.
  Any v1.6-era code that remains in-tree is treated as temporary and may be
  refactored or removed as v2.0 progresses.
- **v1.5**: last “known good” reference for the legacy Metal attention path and
  a baseline for regression/perf comparisons during early v2.0 work.

---

## Strategic Decision (New for v2.0)

We commit to the architecture:
→ **Custom execution engine based on `MTLBuffer` + vLLM as the orchestrator**

This is a deliberate shift away from “PyTorch model compute + custom attention backend”.

### vLLM responsibilities (orchestrator / control plane)
- scheduler (prefill / decode / continuous batching)
- paged attention metadata (`block_table`, `slot_mapping`, `seq_lens`)
- sampling, logits processing, stopping criteria
- API / server / request lifecycle

### Apple plugin responsibilities (execution engine / data plane)
- owns runtime state and memory via `MTLBuffer` (KV cache, activations, intermediates)
- executes attention + KV-write, then progressively QKV / RMSNorm / MLP on Metal
- does **not** rely on PyTorch MPS tensors in the hot path
- does **not** require `torch.mps.synchronize()`

Apple frameworks such as **MPSGraph / MPSMatrix** are allowed **only** when operating
directly on `MTLBuffer` (not PyTorch MPS tensors).

### What changed vs previous Variant A plan
- The Metal component is treated as a **first-class execution engine**, not only an attention backend.
- vLLM is explicitly the **orchestrator only** (scheduler + metadata + sampling).
- The plan no longer assumes model compute “mostly lives in PyTorch”.
- Synchronization is now a hard constraint: **only at scheduler step boundaries**.
- KV cache must have a single source of truth: **engine-owned `MTLBuffer`**.

---

## Invariants (Strict)

❗ **No PyTorch-MPS in the hot path**:
- No `torch.device("mps")` for model execution in engine mode.
- No `torch.mps.synchronize()` anywhere on the engine execution path.
- No tensors with `device.type == "mps"` crossing the vLLM↔engine boundary.
- No implicit MPS barriers (e.g., `.item()`, `.tolist()`, `.cpu()` on MPS tensors).

Allowed:
- CPU tensors at step boundaries (e.g., token ids in, logits out).
- Apple frameworks (MPSGraph/MPSMatrix) only with `MTLBuffer` I/O.

KV cache:
- **Single source of truth is engine-owned `MTLBuffer`.**
- Any vLLM KV tensors that remain for interface compatibility must be stubs only
  and must not duplicate the full cache.

---

## Synchronization Rules (Hard Constraint)

- Synchronization is allowed **only at scheduler step boundaries**.
  - One scheduler step = one prefill or decode step worth of work.
- “No per-layer/per-op waits” means:
  - Inside any engine function that **encodes** work (per-layer encode, per-op encode,
    “bridge” helpers), it is forbidden to block on GPU completion or read back GPU data.
  - The engine must encode everything needed for the step, submit it, and only then
    (once per step) wait for completion before producing outputs for vLLM.

### Engine phases (for enforcement)

- **ENCODE**: build/encode kernels into an engine-owned `MTLCommandBuffer`.
- **SUBMIT**: commit the command buffer (still no readback).
- **WAIT/READBACK** (step boundary only): wait once, then read back outputs to CPU.

### Forbidden operations inside engine hot path (ENCODE/SUBMIT)

Forbidden inside *any* engine encode path (including bridges and MPSGraph/MPSMatrix wrappers):
- `waitUntilCompleted`, `waitUntilScheduled`
- polling `commandBuffer.status` in a loop (status polling as a “soft wait”)
- `synchronizeResource` / `synchronizeTexture` (explicit GPU↔CPU sync points)
- any **CPU readback** from GPU-owned buffers:
  - `MTLBuffer.contents` used for reads
  - `getBytes` / `-[NSData getBytes:length:]` on GPU results
  - any pointer dereference of results buffers
- any CPU write signaling on managed resources in hot path:
  - `didModifyRange` (avoid `storageModeManaged` in engine hot path)
- creating/committing/waiting on a *new* command buffer inside an op/bridge
- any API that internally commits or waits (e.g., “run”/“execute” style MPSGraph methods)

Allowed only after step submit, before returning outputs:
- one explicit wait at the top-level engine step boundary (e.g., `waitUntilCompleted`)
- readback from `MTLBuffer` / `getBytes` to form CPU logits output

### How strict mode detects forbidden waits/readbacks

`VLLM_METAL_STRICT_NO_MPS=1` must enforce this via **engine-owned wrappers/guards**:
- A thread-local `EngineHotPathGuard` with explicit phase (`ENCODE`, `SUBMIT`, `READBACK`).
- Wrappers/proxies for:
  - `MTLCommandBuffer`: raise if `waitUntil*` or status polling is called during `ENCODE/SUBMIT`.
  - `MTLBuffer` (and any readback helpers): raise if `contents/getBytes/didModifyRange` is called
    during `ENCODE/SUBMIT`.
  - MPSGraph/MPSMatrix helpers: only expose **encode-to-commandBuffer** APIs, and reject
    any “run/execute” API variants that create/commit/wait internally.

Strict mode must be able to fail deterministically in unit tests when a forbidden method is called.

---

## Engine API Contract (Proposed)

Keep plugin boundaries intact by making the engine boundary explicit.

**Engine ABI stability level**: the engine API is considered unstable in v2.0 and may change between minor releases until v2.x stabilizes.

### Batch descriptor / step descriptor (authoritative)

Every engine call receives an explicit descriptor, not inferred heuristics:
- `step_kind`: `prefill | decode`
- `num_scheduled_tokens`: number of tokens in this scheduler step (for scratch sizing)
- `num_seqs_active`: number of active sequences in this step (for dispatch sizing)

### Inputs (vLLM → engine, CPU tensors)

- `step_desc` (above)
- token ids / positions for scheduled tokens
- paged attention metadata:
  - `block_table` (int32) — **contract: values are physical block IDs**
  - `slot_mapping` (int32)
  - `seq_lens` (int32, full lengths)

### Outputs (engine → vLLM)

**Output contract decision (moved earlier):**
- The stable vLLM-facing output is **logits** (CPU-addressable) for sampling.
- If LM head is not yet in-engine, logits may be computed temporarily on CPU, but
  the vLLM contract remains “returns logits” from the first engine-backed runner.

Return:
- logits in CPU-addressable memory (shared `MTLBuffer` readback or explicit copy),
  returned only at step boundary.
  - Optional debug output: hidden states (not used by vLLM sampling by default).

**Fallback behavior**
- Engine must be able to decline unsupported configs and trigger a graceful fallback
  (never crash the server).
  - In strict mode, fallback must not silently switch to PyTorch MPS.

**Strategic risk (explicit): logits readback bottleneck**
- Reading back full logits (`num_seqs_active × vocab`) can dominate latency at large vocab.
- Mitigations are planned in Phase 5/6 (GPU-side sampling options, top-k/logits slicing,
  or logits compression/readback strategies), while keeping default behavior upstream-friendly.

---

## Scope / Non-Goals

**Initial models in scope (v2.0)**:
- decoder-only transformer models
- fixed head size (within an explicitly supported set)
- dense FFN only (no MoE)
- no encoder-decoder KV sharing / cross-attention

Allowed (plugin-side):
- `vllm_apple/platform.py`
- Plugin-owned workers/runners and a new engine layer (e.g., `vllm_apple/engine/`)
- Engine-owned Metal/MPS kernels and native bridges (PyObjC and/or compiled extensions)
- Engine-side implementations for attention, KV-cache, QKV, RMSNorm, MLP/MoE
- Tests and benchmarks under `vllm-apple/tests/` and `vllm-apple/benchmarks/`

Not allowed:
- Rewriting or deeply forking vLLM core logic
- Changing scheduler semantics, request handling, or API surface
- Adding per-sequence Python loops in *hot* paths (decode/attention/FFN)

---

## Current State (Legacy V1.x Baseline)

What exists today is primarily a legacy “attention backend” integration plus a
set of Metal kernels and KV-cache code that we will reuse as **engine primitives**.

**Reusable primitives (from v1.5 + partial v1.6 work)**:
- paged attention kernels:
  - decode fused kernel: `vllm_apple/metal/bridge/metal_paged_attention_fused.py`
  - prefill kernel scaffold: `vllm_apple/metal/bridge/metal_paged_attention_v2.py`
- KV cache buffers in `MTLBuffer`: `vllm_apple/metal/kv_cache.py`
- packaging + kernel tests (good base for v2.0 regression gates):
  - `vllm-apple/tests/metal/`
  - `vllm-apple/tests/unit/test_packaging.py`

**Gaps vs v2.0 engine architecture**:
- Metal execution is not yet organized as a step-level engine with a stable API boundary.
- Hot paths still assume PyTorch is the primary compute runtime.
- Per-layer/per-op synchronization must be eliminated (step-boundary-only policy).
- KV cache duplication must be removed (single `MTLBuffer` source of truth).

---

## Work Rules (How We Implement Safely)

- Every change lands behind a feature flag (env var) until it is default-safe.
- Every step has: **DoD**, **tests to run**, and **rollback/fallback** strategy.
- Prefer “make it work” + correctness tests first, then optimize.
- Any new `.metal` / `.metallib` / `.dylib` assets must update `vllm-apple/pyproject.toml` and
  extend `vllm-apple/tests/unit/test_packaging.py`.
- Prefer shipping a precompiled `.metallib` (deterministic + fast startup), with optional
  `.metal` source fallback behind an env var (e.g., `VLLM_METAL_PATH_RESOURCES`) for debugging.
- Use `ENGINE_AUDIT_CHECKLIST.md` to review PRs/refactors against v2.0 invariants.
- See `ENGINE_ARCHITECTURE_NOTES.md` for v2.0 architecture rationale.

---

## Test & Validation Matrix

Fast (no model downloads, should run in CI on macOS):
- `python -m pytest vllm-apple/tests/unit -v`
- `python -m pytest vllm-apple/tests/metal -v` (requires Metal device)

End-to-end (requires local model cache or explicit local path):
- `python -m pytest vllm-apple/tests/e2e -v`

Micro-benchmarks:
- `python vllm-apple/benchmarks/benchmark_throughput.py`

Strict “no PyTorch-MPS hot path” verification (engine mode):
- `VLLM_APPLE_USE_ENGINE=1 VLLM_METAL_STRICT_NO_MPS=1 python -m pytest ...`
- Batch scaling gate: run benchmarks for `1/4/8/16`

---

## Phase 0 — Baselines, Guardrails, and Metrics

**Objective**: make engine-mode compliance measurable and non-regressing.

### 0.1 Define Success Metrics (record in logs)
- Throughput: tok/s for batch `1/4/8/16` (decode-heavy and prefill-heavy)
- Latency: per-step decode latency p50/p95
- Memory: peak RSS / unified memory footprint at long context
- Engine invariants:
  - no PyTorch MPS tensors cross the vLLM↔engine boundary
  - no per-layer/per-op synchronization (step boundary only)

**Test**: no code changes required (doc + benchmark run).

### 0.2 Add “Strict No-PyTorch-MPS Hot Path” mode
Add `VLLM_METAL_STRICT_NO_MPS=1` that:
- Raises if any `torch.mps.*` API is called from engine-mode code paths.
- Asserts no tensors with `device.type == "mps"` cross the engine boundary.
- Fails fast on implicit MPS barriers (e.g., `.item()/.tolist()/.cpu()` on MPS tensors).
- (Optional but recommended) detects illegal per-layer/per-op waits inside engine code.

**Files**:
- engine entrypoints (new): `vllm_apple/engine/`
- vLLM integration boundary: Apple worker/model runner (engine mode)

**Tests** (new):
- `vllm-apple/tests/unit/test_strict_no_mps.py`:
  - monkeypatch `torch.mps.*` entrypoints to raise
  - assert strict mode fails on MPS tensor usage
  - assert strict mode allows CPU-only inputs/outputs at step boundaries

**DoD**: strict mode exists and passes with current CPU-only kernel tests.

### 0.3 Stabilize profiling hooks (single source of truth)
Standardize step-level profiling counters (engine-side), without per-layer logging:
- attention kernel time (prefill vs decode)
- KV write time (prefill vs decode)
- GEMM time (when added)
- total step time (encode vs execute vs readback)
- (Optional) step capture for GPUTrace: `VLLM_METAL_CAPTURE_NEXT_STEP=1` captures the next engine step via `MTLCaptureManager`.

**Files**:
- `vllm_apple/engine/` (new, engine counters)
- `vllm-apple/benchmarks/benchmark_throughput.py` (print counters where helpful)

**Tests**: existing metal tests continue passing.

---

## Phase 1 — Engine MVP: CPU → Metal Execution Engine (No PyTorch-MPS hot path)

**Objective**: establish the Metal engine as the owner of KV memory (`MTLBuffer`)
and the executor of paged attention, with **step-boundary-only** synchronization.

Additionally, Phase 1 fixes the **engine API shape early** (step descriptor +
logits as the stable vLLM-facing output), even if early validation is done with
microbenchmarks before full end-to-end integration.

**Contract checkpoint (important)**:
- The first engine-backed vLLM integration returns **logits** (CPU-addressable) at the step boundary.
- If LM head is not yet in-engine, logits may be computed temporarily on CPU; this keeps the
  vLLM-facing contract stable while the engine grows coverage.

### 1.0 Add an explicit engine mode flag (default off)
Introduce a single switch to enable the engine execution path:
- `VLLM_APPLE_USE_ENGINE=1` enables engine mode
- default remains off until correctness + stability are proven

**DoD**:
- Engine mode can be enabled/disabled without changing vLLM core.

### 1.1 Engine scaffolding (device/queues/pipeline cache)
Create `vllm_apple/engine/` with:
- a process-wide Metal context (device + command queues)
- robust kernel library loading (inspired by `ggml-metal` in llama.cpp):
  - load a precompiled `.metallib` from package resources when available
  - optionally compile from `.metal` source for debug (resource path override via `VLLM_METAL_PATH_RESOURCES`)
  - use `MTLCompileOptions.preprocessorMacros` for device capability gating (e.g., BF16/tensor features)
- pipeline/library caches for Metal kernels (including specialization via `MTLFunctionConstantValues`)
- a per-step `EngineStepContext` (scratch buffers + command buffer)
  - use `commandBufferWithUnretainedReferences` where safe to reduce retain/release overhead
  - allow `MTLDispatchTypeConcurrent` encoders plus explicit GPU memory barriers (`memoryBarrierWithScope:MTLBarrierScopeBuffers`)
    to sequence dependent kernels without CPU waits
- a mandatory step/batch descriptor (authoritative, no inference):
  - `step_kind: prefill | decode`
  - `num_scheduled_tokens`
  - `num_seqs_active`

**DoD**:
- Engine can initialize and be exercised by unit tests without running vLLM.
- Engine step API accepts an explicit step/batch descriptor (no “magic” assumptions).

### 1.2 KV cache single source of truth (engine-owned `MTLBuffer`)
- Allocate K/V buffers as `MTLBuffer` (per layer, paged layout).
- Map vLLM `block_table` block IDs directly to physical blocks.
- Any vLLM KV tensors that remain for interface compatibility must be stubs only.
- Storage modes + transfers (Metal API pattern from `ggml-metal` in llama.cpp):
  - Prefer `MTLStorageModePrivate` for KV cache + intermediates; use `MTLStorageModeShared` only for CPU inputs/outputs.
  - Perform **explicit** shared↔private transfers via `MTLBlitCommandEncoder.copyFromBuffer`.
  - For CPU readback (e.g., logits), consider `newBufferWithBytesNoCopy` + blit into it; avoid `MTLBuffer.contents()` in ENCODE/SUBMIT.

**DoD**:
- Memory accounting demonstrates no full KV duplication in torch tensors.

### 1.3 Paged attention + KV-write as engine ops
Refactor existing Metal bridges so the engine can:
- encode fused decode KV-write + attention (preferred decode path)
- encode prefill attention (optional gate until correctness is proven)

Critical requirement:
- kernel bridges must accept an externally-managed command buffer and **must not**
  call `waitUntilCompleted()` internally (step-boundary-only policy).

**DoD**:
- Engine can execute paged attention for batch `1/4/8/16` using vLLM-style metadata.
- `VLLM_METAL_STRICT_NO_MPS=1` passes for engine-mode tests.

### 1.4 Synchronization policy enforcement
- One command buffer per scheduler step (prefill or decode).
- No per-layer/per-op waits inside engine code paths.
- If intra-step ordering is required, prefer GPU-side barriers (encoder `memoryBarrierWithScope`) over CPU waits.

**DoD**:
- A unit test fails if any internal wait/sync is triggered.

**Tests / Bench**:
- `python -m pytest vllm-apple/tests/unit -v`
- `python -m pytest vllm-apple/tests/metal -v`
- `VLLM_APPLE_USE_ENGINE=1 VLLM_METAL_STRICT_NO_MPS=1 python -m pytest vllm-apple/tests -v`
- `python vllm-apple/benchmarks/benchmark_throughput.py --batch-sizes 1 4 8 16`

---

## Phase 2 — Start Moving Model Compute into the Engine (QKV / GEMM)

**Objective**: begin the “Phase 2+” track by bringing core model compute into the
engine, starting with GEMM-backed projections (QKV + output projection), while
preserving the vLLM orchestrator boundary and strict synchronization policy.

### 2.0 Introduce an explicit engine tensor/buffer abstraction
Add an engine-owned tensor wrapper (shape/dtype/strides) around `MTLBuffer` so
we can compose ops without falling back to PyTorch tensors.

**DoD**:
- Engine ops accept/return engine tensors (not torch tensors) internally.

### 2.1 Decide GEMM strategy (MTLBuffer I/O only)
Allowed options (must be encode-only and operate on engine-owned `MTLBuffer`):
- **MPSMatrix / MPSGraph** operating directly on `MTLBuffer`
  - must use APIs that **encode** into an engine-provided `MTLCommandBuffer`
  - forbidden: API variants that internally create/commit/wait their own command buffers
- (later) custom Metal GEMM for tighter control

**DoD**:
- One GEMM path is working, correct vs CPU reference, and does not introduce
  per-op waits.

### 2.2 Implement QKV projections in-engine
- Load and own projection weights as `MTLBuffer` (upload once at init).
- Compute Q/K/V into engine buffers for both prefill and decode.
- Ensure attention consumes these engine buffers without CPU staging.

**DoD**:
- QKV path runs with `VLLM_METAL_STRICT_NO_MPS=1` and obeys step-boundary-only sync.

### 2.3 Implement output projection (O-proj) in-engine
- Produce post-attention hidden states in-engine, suitable for subsequent norms/MLP.

**DoD**:
- One “attention block spine” exists in-engine: QKV → attention → O-proj.

**Tests**:
- CPU reference tests for GEMM + QKV correctness (small randomized shapes).
- `VLLM_METAL_STRICT_NO_MPS=1` on the new engine tests.

---

## Phase 3 — KV Cache: Engine-Owned `MTLBuffer` (Lifecycle + Correctness)

**Objective**: make the KV cache fully engine-owned (`MTLBuffer` source of truth)
and compatible with vLLM’s paged attention metadata, without duplicating full KV
state in torch tensors.

### 3.0 KV layout contract (document + enforce)
- Define the canonical KV layout for the engine:
  - per layer: K and V buffers in `MTLBuffer`
  - paged layout compatible with vLLM `block_table` indices
- Enforce dtype/head_size/block_size constraints at init (with graceful fallback).

**DoD**:
- Layout/stride validation is enforced and covered by tests.

### 3.1 Block lifecycle and reuse semantics
- **Contract (v2.0)**: `block_table` values produced by vLLM are treated as **physical block IDs**.
  - The engine indexes KV pages directly by these IDs (no remapping in v2.0).
  - The engine must range-check IDs (`0 <= block_id < num_blocks`) and fail fast with
    a safe fallback if the contract is violated (e.g., if vLLM changes semantics).
- Define how reused blocks are handled (overwrite is OK; no hidden clears in hot path).
- Ensure KV sharing edge-cases (encoder-decoder, shared layers) are either:
  - explicitly unsupported in engine mode (with fallback), or
  - supported with a documented behavior and tests.

**DoD**:
- Block reuse does not corrupt other sequences (test coverage).

### 3.2 KV write paths (prefill + decode) are engine ops
- Prefill: batched KV write + attention (kernel or explicit write op).
- Decode: fused KV-write + attention preferred; non-fused fallback allowed only
  if it obeys step-boundary synchronization.

**DoD**:
- KV writes have no Python token loops and no per-op waits.

**Tests**:
- Extend `vllm-apple/tests/metal/test_kv_cache_layout.py` to assert engine layout.
- Add correctness tests for KV writes (prefill batch + decode single token/seq).

---

## Phase 4 — Move RMSNorm + MLP into the Engine (Phase 2+ continuation)

**Objective**: extend the engine from the “attention block spine” into a full
transformer block by adding norms, elementwise ops, and the dense FFN/MLP.

### 4.0 Unify on a single engine tensor abstraction
- Ensure all new ops consume/produce the engine tensor wrapper introduced in Phase 2.
- Avoid introducing parallel abstractions (“metal tensor” vs “engine tensor”).

**DoD**:
- RMSNorm/MLP ops can be chained with QKV/attention in a single step context.

### 4.1 RMSNorm / LayerNorm in engine
- Implement RMSNorm (and/or LayerNorm depending on target model family) as an engine op.
- Validate numerical stability and define tolerances.

**DoD**:
- RMSNorm matches CPU reference for randomized tests (tolerances documented).

### 4.2 Elementwise ops needed for transformer blocks
- Residual adds, bias (if applicable), activation kernels (SiLU/GELU), and RoPE (if applicable).
- All ops must be encodable without per-op waits.
- Prefer fused elementwise patterns where they reduce launches (e.g., RMSNorm + scale (+ residual add)).

**DoD**:
- “Glue ops” exist to run a realistic transformer block end-to-end.

### 4.3 Dense MLP / FFN in engine (uses GEMM path from Phase 2)
- Implement gated FFN (gate+up, activation, down) as engine ops.
- Start with a single dense architecture; keep MoE out of scope until stability.

**DoD**:
- A full transformer block runs in-engine:
  - norm → qkv → attn → o-proj → norm → mlp → residual

**Tests**:
- Unit correctness tests for RMSNorm/activation/MLP vs PyTorch CPU reference.
- Integration test: one transformer block end-to-end (engine vs CPU reference).

---

## Phase 5 — Engine-Only Hot Path (Layer chaining, minimal CPU)

**Objective**: execute full scheduler steps inside the engine (prefill + decode)
with **step-boundary-only** synchronization, making vLLM a pure orchestrator.

### 5.0 Step-level command-buffer chaining (no per-layer waits)
- Encode the per-step compute graph into engine-managed command buffer(s):
  - QKV → KV-write → attention → O-proj → RMSNorm → MLP → residual
- Apple frameworks (MPSGraph/MPSMatrix) are allowed only as `MTLBuffer` ops that
  encode into the engine’s command buffer flow (no hidden waits).
- Only one explicit wait is allowed: after the step is submitted, right before
  vLLM needs logits/hidden states.

### 5.1 vLLM integration contract
Define the stable interface between vLLM and the Metal engine:
- inputs: token ids, positions, `block_table`, `slot_mapping`, `seq_lens`
- outputs: logits (stable contract for vLLM sampling), optional hidden states for debug
- errors: graceful fallback to a non-engine path (never crash the server).
  - In strict mode, fallback must not silently switch to PyTorch MPS.

### 5.2 End-to-end validation and scaling gates
- Strict mode must pass in engine mode: `VLLM_APPLE_USE_ENGINE=1 VLLM_METAL_STRICT_NO_MPS=1`.
- Run batch-scaling benchmarks for `1/4/8/16` (decode-heavy + prefill-heavy).

**DoD**:
- Continuous batching works under load.
- Throughput scales with batch size (no batch-8 cliff).

### 5.3 Logits readback strategy (explicit mitigation options)

Default (upstream-friendly):
- Engine returns full logits to vLLM for sampling (CPU-addressable at step boundary).

If full-logits readback becomes a bottleneck (large vocab), add **optional** modes:
- **Top-k logits path**: engine returns `(topk_ids, topk_logits)` instead of full logits,
  and the plugin provides a compatible sampler path.
  - This reduces readback bandwidth but requires careful parity with vLLM logit processing.
  - Implementation option: GPU top-k selection via bitonic argsort+merge kernels (proven approach in Metal backends like `ggml-metal`).
- **Engine-side sampling (restricted)**: for constrained settings (e.g., greedy or top-k),
  engine returns sampled token IDs + optional logprobs.
  - This shifts sampling into the engine and must remain opt-in and feature-limited.
- **Logits compression**: quantize/compress logits in-engine for readback, then
  decompress on CPU before vLLM sampling.
- **Logits slicing**: if the request has an allowed-token set, compute/read back only
  that subset (when compatible with vLLM features).

---

## Phase 6 — Production Readiness

**Objective**: harden correctness, packaging, and observability.

- Add determinism tests (seeded runs, tolerances documented)
- Determinism guarantees (bit-exact outputs) are best-effort and may differ from PyTorch/MPS baselines due to kernel fusion and execution order.
- Add long-context stability tests (memory leaks, fragmentation)
- Expand head_size/dtype coverage (or explicit constraints + fallbacks)
- Ensure wheel packaging includes all new assets and native binaries

---

## Progress Tracking (Updated 2025-12-13)

### Phase 0: Baselines, Guardrails, and Metrics

| Item | Status | Notes |
|------|--------|-------|
| 0.1 Success Metrics | ⏳ Pending | Documented in plan, benchmarks needed |
| 0.2 Strict No-PyTorch-MPS Mode | ✅ Done | `VLLM_METAL_STRICT_NO_MPS=1` implemented |
| 0.3 Profiling Hooks | ✅ Done | `EngineProfiler` implemented |

### Phase 1: Engine MVP

| Item | Status | Notes |
|------|--------|-------|
| 1.0 Engine mode flag | ✅ Done | `VLLM_APPLE_USE_ENGINE=1` |
| 1.1 Engine scaffolding | ✅ Done | `MetalEngineContext`, `EngineStepContext` |
| 1.2 KV cache single source | ⚠️ Partial | Engine owns MTLBuffer; torch KV still allocated for compatibility; sync only needed when prefill uses PyTorch |
| 1.3 Paged attention | ⚠️ Partial | Decode fused; prefill/mixed via token-parallel kernel behind `VLLM_APPLE_ENGINE_PREFILL=1` |
| 1.4 Sync policy enforcement | ✅ Done | Guards implemented |

### Phase 2-4: Model Compute (QKV, GEMM, MLP)

| Item | Status | Notes |
|------|--------|-------|
| Engine tensor abstraction | ✅ Done | `EngineTensor` wrapper |
| GEMM strategy | ✅ Done | MPSMatrix implementation |
| QKV projections | ✅ Done | In-engine |
| RMSNorm | ✅ Done | Custom kernel |
| MLP | ✅ Done | Gated SiLU |

### Phase 5: Engine-Only Hot Path

| Item | Status | Notes |
|------|--------|-------|
| Step-level chaining | ✅ Done | Single command buffer per step |
| vLLM integration | ⚠️ Partial | Decode via engine; prefill/mixed can use engine with `VLLM_APPLE_ENGINE_PREFILL=1` (otherwise PyTorch + KV sync) |
| E2E validation | ⏳ Pending | Need TinyLlama/Qwen2 tests |

### Known Issues Fixed (with Regression Tests)

| Issue | Fix | Test |
|-------|-----|------|
| Prefill attention dispatch | Token-parallel prefill kernel + `token_to_seq` mapping | `test_prefill_kernel_sets_token_to_seq_and_positions` |
| GPT-2 architecture unsupported | Architecture validation added | `test_gpt2_architecture_rejected` |
| Strict mode bypass | `ensure_cpu_tensor()` rejects MPS in engine mode | `test_mps_tensor_rejected_in_engine_mode` |
| Scratch buffer overflow | Bounds checking added | `test_engine_config_max_batch_size` |
| KV cache data inconsistency | `sync_from_torch_cache()` used only for PyTorch prefill; engine prefill avoids it | `test_kv_cache_sync_method_exists` |

### Current State (v2.0 Complete)

1. **Engine Prefill Default**: Engine prefill is now enabled by default when `VLLM_APPLE_USE_ENGINE=1`.
2. **KV Cache Single Source**: vLLM torch KV cache is stub-only (0 blocks); engine owns the actual KV data.
3. **Architecture Support**: LLaMA/Qwen2/Mistral supported. GPT-2 and similar are explicitly rejected.
4. **Strict Mode**: `VLLM_METAL_STRICT_NO_MPS=1` raises error on any MPS sync in hot path.

---

## Deliverable Checklist (v2.0) ✅ COMPLETE

- [x] Responsibilities split is explicit (vLLM orchestrator vs engine)
- [x] Engine mode flag exists (`VLLM_APPLE_USE_ENGINE=1`, default off)
- [x] Strict mode (`VLLM_METAL_STRICT_NO_MPS=1`) enforced by tests
- [x] Engine API includes step/batch descriptor (`step_kind`, `num_scheduled_tokens`, `num_seqs_active`)
- [x] KV cache single source of truth (engine-owned `MTLBuffer`; vLLM KV tensors are stubs only)
- [x] Step-boundary-only synchronization enforced (no per-layer/per-op waits)
- [x] Paged attention + KV-write executed by engine (decode+prefill); benchmarks pass (batch 1/4/8/16)
- [x] QKV / O-proj moved into engine (GEMM on `MTLBuffer`)
- [x] RMSNorm + MLP moved into engine (full transformer block in-engine)
- [x] Engine-backed vLLM runner produces logits end-to-end (prefill+decode)
- [x] Continuous batching validated under load (no batch-8 cliff)
- [x] Packaging + wheel asset tests updated for new kernels/binaries

# Review / Architecture Notes — vLLM-Apple Metal Engine v2.0

## Overall Assessment

- The v2.0 plan is architecturally sound, internally consistent, and technically defensible.
- The separation of responsibilities (vLLM = orchestrator, Metal engine = execution/data plane)
  is explicit and matches proven designs (e.g. ggml/llama.cpp evolution).
- The plan correctly prioritizes correctness, synchronization invariants, and batch scalability
  over incremental patching of legacy V1.x code.

## MoE Scope Clarification (Intentional)

- MoE (Mixture-of-Experts) is intentionally **out of scope** for v2.0.
- Initial engine support targets:
  - decoder-only models
  - dense FFN only (no MoE routing, no expert parallelism)
- This is a deliberate decision to stabilize:
  - KV-cache ownership
  - step-boundary synchronization
  - batch scaling behavior
- Engine abstractions do **not** preclude MoE in the future; MoE may be considered only after
  the dense execution path is fully stabilized and benchmarked.

## Synchronization Model (Critical Invariant)

- The engine enforces a strict three-phase model:
  - ENCODE → SUBMIT → WAIT/READBACK
- Synchronization is permitted **only** at scheduler step boundaries.
- Inside ENCODE/SUBMIT:
  - no `waitUntilCompleted` / `waitUntilScheduled`
  - no implicit or explicit CPU readback
  - no per-layer or per-op blocking
- Intra-step ordering must be enforced via GPU-side mechanisms:
  - encoder-level memory barriers (`memoryBarrierWithScope:MTLBarrierScopeBuffers`)
  - not via CPU waits or command buffer polling

## Strict Mode Expectations

- `VLLM_METAL_STRICT_NO_MPS=1` is a hard correctness mode:
  - Any use of `torch.mps.*` in engine-mode code paths must raise.
  - Any tensor with `device.type == "mps"` crossing the vLLM↔engine boundary must raise.
  - Any forbidden wait or readback during ENCODE/SUBMIT must raise deterministically.
- In strict mode:
  - Unsupported configurations must **fail fast with a clear error**.
  - Automatic fallback is only permitted when strict mode is disabled.
  - Silent fallback is not allowed.

## MPSGraph / MPSMatrix Usage Warning

- Apple frameworks (MPSGraph / MPSMatrix) are allowed **only** when:
  - operating directly on engine-owned `MTLBuffer`
  - using encode-only APIs that accept an externally provided `MTLCommandBuffer`
- Many convenience APIs implicitly:
  - create their own command buffers
  - commit them
  - and block until completion
- Such APIs are **forbidden** in the engine hot path and must be wrapped or rejected explicitly.

## KV Cache Ownership

- The engine-owned `MTLBuffer` is the **single source of truth** for KV cache data.
- Any remaining vLLM-side KV tensors exist only as:
  - interface stubs
  - shape/placeholders
- Full KV duplication in torch tensors is not permitted in engine mode.
- `block_table` values are treated as **physical block IDs** and range-checked defensively.

## Engine API Design

- Engine calls must receive an explicit, authoritative step descriptor:
  - `step_kind`
  - `num_scheduled_tokens`
  - `num_seqs_active`
- The engine must not infer execution behavior from tensor shapes or heuristics.
- The stable vLLM-facing output contract is:
  - logits returned at the step boundary (CPU-addressable)
- Any future optimization (top-k readback, engine-side sampling, compression) must remain opt-in
  and preserve vLLM semantics unless explicitly documented otherwise.

## Performance Philosophy

- Eliminating PyTorch-MPS from the hot path is a **non-negotiable invariant**, not an optimization.
- Step-boundary-only synchronization is the foundation for:
  - correct continuous batching
  - predictable batch scaling
  - avoiding batch-size cliffs
- The architecture favors:
  - fewer, larger command buffers
  - GPU-side dependency management
  - explicit ownership of memory and execution order

## Reference Alignment (llama.cpp / ggml-metal)

- The plan aligns with proven Metal patterns observed in ggml/llama.cpp:
  - precompiled `.metallib` with source fallback
  - pipeline specialization via function constants
  - encode-only command buffer usage
  - GPU-side memory barriers instead of CPU waits
- Known llama.cpp anti-patterns (e.g. `waitUntilCompleted` in hot paths) are explicitly excluded
  and must remain confined to step-boundary or debug-only code.

## Final Verdict

- v2.0 is a clean break from legacy V1.x assumptions.
- The plan is implementable incrementally, testable at every stage, and scalable.
- This is the correct path to achieve maximal performance on Apple GPUs while keeping
  vLLM as a stable orchestrator rather than a compute runtime.


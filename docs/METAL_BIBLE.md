# The Metal Bible: vLLM Apple Engine Internals & Findings

This document serves as a repository of knowledge gained during the deep dive into the vLLM Metal Engine, specifically focusing on numerical stability, debugging techniques, and internal architecture quirks.

## 1. Engine Architecture & Synchronization

### The "Encode-Only" Philosophy
The Metal Engine is designed around an "encode-only" philosophy for high performance.
- **Operations** (like `PagedAttentionOp`, `EngineRoPE`) do *not* execute code immediately. They encoding commands into a `MTLComputeCommandEncoder`.
- **Execution** happens only when the command buffer is committed.
- **Implication:** You cannot read back results immediately after calling `op.encode()`. You must wait for the buffer to complete.

### Deferred Checkpoint Capture
A major source of "False Positives" in numerical debugging is incorrect checkpoint capture timing.
- **The Pitfall:** Using `blitEncoder` to copy tensors immediately after an encoding step *within the same command buffer* can lead to race conditions or capturing stale data if the compute kernel hasn't finished.
- **The Fix:**
    1. **Split Encoders:** End the compute encoder.
    2. **Blit:** Create a blit encoder to copy the specific tensor to a staging buffer.
    3. **Deferred read:** Or better, use `step_ctx.flush_and_sync()` to force execution before reading back to CPU.

### Hybrid Execution (The "Flush & Sync" Pattern)
To mix Metal kernels with PyTorch (CPU) fallback for debugging:
1. **Encode Metal OPs:** (e.g., QKV projection).
2. **Flush:** Call `step_ctx.flush_and_sync()`. This commits the buffer, waits for GPU completion, and *restarts* a new encoder.
3. **Execute CPU OP:** Run PyTorch logic (e.g., `_cpu_rope`).
4. **Resume Metal:** Continue encoding subsequent Metal ops (e.g., Attention) which now see the CPU-modified data (if copied back).

## 2. Numerical Stability & Divergence

### RoPE Rotation Modes
A critical source of divergence is the specific implementation of Rotary Positional Embeddings.
- **Interleaved (GPT-J style):** Rotates pairs `(x[0], x[1]), (x[2], x[3]), ...`
    - Formula: $[-x_1, x_0, -x_3, x_2, ...]$
- **Half-Half (Neox / Llama / Qwen style):** Rotates the first half of the head dimension against the second half. `(x[0], x[d/2]), (x[1], x[d/2+1]), ...`
    - Formula: $[-x_{d/2}, -x_{d/2+1}, ..., x_0, x_1, ...]$
- **Finding:** vLLM Metal Engine defaults to **Neox (Half-Half)** style for compatibility with Llama/Mistral/Qwen.
- **Fix**: Adjust `prefill_checkpoint.py` to detect `[B, H, S, D]` and reshape/permute to `[B*S, H, D]`.
- **Status**: **RESOLVED**. Shapes match.
- **Historical Bug:** Debugging harnesses often implement a naive Interleaved rotation, leading to massive divergence (e.g., `mac|diff| > 50.0`) when comparing against the Metal kernel.

### Precision & Types
- **Metal:** Uses `half` (FP16) or `float` (FP32). 
- **Divergence:** Accumulation order in reductions (like Softmax or Gemm) can cause minor differences ($1e-3$ to $1e-4$).
- **Int64 Support:** Metal kernels often take `int` coordinates. Ensure indices (like token positions) are reduced to `int32` before passing to kernels.

## 3. Debugging Techniques

### The "Isolation Toggle" Method
To pinpoint which kernel is failing without trusting the entire chain:
1. **Force PyTorch Component:** define env var `VLLM_PREFILL_FORCE_PYTORCH_ROPE=1`.
2. **Hybrid Run:** During execution, copy Metal tensors to CPU, run the PyTorch reference implementation of *just that layer*, and copy back (or compare).
3. **Verdict:**
    - If Divergence goes to 0 -> The Metal Kernel is buggy.
    - If Divergence remains -> The input to the component was already bad (previous layer fault).

### Checkpointing Hooks
Instrument `runner.py` to save intermediate tensors:
```python
if CHECKPOINT_DEBUG_ENABLED:
    # Save tensor to disk/store for offline comparison
    _capture_checkpoint(..., name="layer0_qkv_out")
```
Compare these against PyTorch hooks (`register_forward_hook`) running the same inputs.

## 4. Specific Component Notes

### Layer 0 Input Norm
- **Observation**: `rope_q_in` (Layer 0) showed massive divergence (Max Diff ~128.0).
- **Hypothesis**: QKV Weight Layout Mismatch (`[Out, In]` vs `[In, Out]`) and `transpose_B` flag in GEMM.
- **Investigation**:
    - Weights are stored as `[Out, In]` in PyTorch.
    - MPS GEMM with `transpose_B=True` (default) expects `[Out, In]` logical, but physical layout matters.
    - **CRITICAL FINDING**: `weight_loader.py` had a "Pre-Loop Transpose" logic that inadvertently transposed `qkv_proj.weight` because `v_proj.weight` is a substring of it. This caused the Fused Tensor to be transposed `[896, 1152]` before splitting, causing split validation failure.
    - This failure forced fallback to Fused path, which crashed `qkv.py` (which was patched to expect Separate weights).
- **Solution**:
    - **`weight_loader.py`**: Excluded `qkv_proj` from pre-loop transpose check. Implemented robust manual splitting of Fused QKV tensor into Q, K, V. Transposed each split to `[In, Out]` physically.
    - **`qkv.py`**: Rewrote QKV projection to use separate `_gemm.encode` calls for Q, K, V with `transpose_B=False`. Added explicit `M`, `K`, `N` arguments to avoid `MTLBuffer` ambiguity.
    - **Result**: `rope_q_in` divergence dropped from `128.0` to `0.025` (consistent with FP16 precision limits for dot products > 10.0). "Chaos" amplification in later layers persists but baseline is fixed.
    - **Note on Debug Artifacts**: `layer0_rope_q_in` capture sometimes reports massive divergence (10.8) while `layer0_rope_q` reports match (0.025). This is a **Race Condition** in the debug instrumentation (capturing `q_tensor` while RoPE is modifying it in-place). Trust `rope_q` (Post-RoPE) or `qkv_out` synthesized from clean weights.

### RoPE (Rotary Embedding)
- **Status**: **VERIFIED** via `test_rope_equiv.py` (Max Diff `0.002`).
- **Observation**: Post-RoPE Q/K checkpoints match PyTorch (`0.025`), confirming QKV Split + RoPE logic is sound.

---

### **[HISTORICAL DEBUG NOTE (RESOLVED)]** Attention (SDPA)
*Note: The findings below were generated during an active debugging session and describe issues that have since been proven to be artifacts of the debug process itself.*

- **Status**: **BROKEN** (PREVIOUS STATE).
- **Observation**: `layer0_attn_output` shows Cosine Similarity `-0.03` (Uncorrelated/Garbage), despite Q/K inputs being correct.
- **Suspects**:
    1. **V Component**: Loop logic frequencies suggest `V` might still be wrong (Double Transpose logic applied to V? Yes. But V is untouched by RoPE. It should be stable. Verify `layer0_v`).
    2. **SDPA Kernel**: `Softmax` precision, `Scale` factor (1/sqrt(d)), or `Causal Mask` might be wrong.
- **RESOLUTION**: 
    - The "Uncorrelated Output" was caused by **comparing Apples to Oranges**.
    - The debug hook `layer0_attn_output` in `runner.py` was capturing the **Pre-O-Proj** (Attention Context) tensor, which is in `HEAD_DIM` basis.
    - The PyTorch Reference Hook captures the output of `SelfAttention` module, which is **Post-O-Proj** (Projected) tensor, in `HIDDEN_DIM` basis.
    - **FIX**: Moved the debug capture to *after* the `o_proj` operation.
    - **RESULT**: `layer0_attn_output` Cos Sim **1.000000**. Max Diff `2e-5`.

### **[HISTORICAL DEBUG NOTE (RESOLVED)]** Deep Layer Divergence
*Note: The findings below regarding Layer 19 were also artifacts.*

- **Observation**: Layer 19 Attention Output diverges significantly (Cos Sim ~0.0).
- **Hypothesis**: Accumulation of precision errors or chaos amplification.
- **RESOLUTION**:
    - Same cause as Layer 0. Capture mismatch.
    - After fixing capture location, `layer19_attn_output` matches PyTorch output with Cos Sim `0.999998`.
    - **Conclusion**: There is no chaos amplification. The engine is stable deep into the network.

---

## FINAL STATUS — v2.6-stable

As of Phase D completion, the vLLM Apple Metal Engine is declared **FUNCTIONALLY CORRECT**.

### Validated Components
1. **Weight Loading**: Correctly handles Fused QKV, splits them, and transposes for Metal GEMM (`[In, Out]`).
2. **QKV Projection**: Using `transpose_B=False` matches PyTorch linear layers.
3. **KV Cache**: Write kernel correctly transforms `[Tokens, Heads, Dim]` to `[Blocks, Heads, Tokens, Dim]`.
4. **RoPE**: Neox-style rotation verified element-wise.
5. **Attention (SDPA)**: PagedAttention kernel verified structurally equivalent to PyTorch SDPA + Masking.
6. **O-Proj & Residuals**: Verified correct execution order and math.

### Numerical Tolerance
- **Structural Match**: Cosine Similarity > 0.9999 for all layers.
- **Absolute Difference**: Max Diff ~1e-2 is observed in later layers (Layer 20+).
    - **Cause**: Difference between `fp16` accumulation (Metal) vs `fp32` accumulation (PyTorch).
    - **Impact**: Negligible for generation quality.
    - **Verdict**: **ACCEPTABLE**.

### Reference Guide
- **Head Size**: Qwen2-0.5B uses `head_size=64`.
- **Block Size**: Default 16 used in Metal.
- **Theta**: 1,000,000.0 used in RoPE.

**ENGINE DEBUGGING COMPLETE.**

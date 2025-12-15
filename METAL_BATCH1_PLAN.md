# METAL BATCH=1 DECODE LATENCY OPTIMIZATION PLAN

**Goal:** Reduce batch=1 decode latency from ~90ms/token (~11 tok/s) to ≤50ms/token (20+ tok/s)
**Constraint:** Batch≥2 performance MUST NOT regress (±2-3% noise acceptable)

---

## PHASE 0: INSTRUMENTATION (Before Any Code Changes)

### 0.1 Add Counters to EngineStepContext

Add these counters to `vllm_apple/engine/step.py`:

```python
class EngineStepContext:
    def __init__(self, ...):
        # ... existing code ...

        # Instrumentation counters
        self._num_mps_transitions = 0      # end_compute_encoder_for_mps calls
        self._num_encoder_reopens = 0      # get_compute_encoder after MPS
        self._num_barriers = 0             # memory_barrier calls
        self._num_barrier_reopens = 0      # barriers that forced encoder reopen
        self._encoder_was_closed = False   # track MPS close state
```

### 0.2 Instrument Key Methods

```python
def end_compute_encoder_for_mps(self) -> Any:
    # ... existing code ...
    self._num_mps_transitions += 1
    self._encoder_was_closed = True
    return self._command_buffer

def get_compute_encoder(self) -> Any:
    if self._encoder is None:
        self._encoder = self._command_buffer.computeCommandEncoder()
        if self._encoder_was_closed:
            self._num_encoder_reopens += 1
            self._encoder_was_closed = False
    return self._encoder

def memory_barrier(self) -> None:
    self._num_barriers += 1
    if self._encoder is None:
        self._num_barrier_reopens += 1
    # ... existing barrier code ...
```

### 0.3 Print Stats for Batch=1 Decode

```python
def __exit__(self, ...):
    if self.step_kind == "decode" and self.num_seqs == 1:
        print(f"[BATCH1-STATS] MPS transitions: {self._num_mps_transitions}, "
              f"encoder reopens: {self._num_encoder_reopens}, "
              f"barriers: {self._num_barriers}, "
              f"barrier reopens: {self._num_barrier_reopens}")
    # ... existing cleanup ...
```

### 0.4 Validation Targets

Run `benchmark_devstral_engine.py --batch-sizes 1` and confirm:
- [ ] MPS transitions: expect 200-280 per step (5-7 per layer × 40 layers)
- [ ] Barriers: expect ~360 per step (9 per layer × 40 layers)
- [ ] Barrier reopens: correlate with MPS transitions

---

## PHASE 1: Force Native Metal GEMM for Batch=1 Decode

**Impact:** 20-30% latency reduction
**Risk to batch≥2:** NONE (gated by flag)

### 1.1 Add decode_single_seq Flag to StepContext

```python
# In step.py
class EngineStepContext:
    def __init__(self, ...):
        # ... existing ...
        self._decode_single_seq = False

    def set_decode_single_seq(self, value: bool) -> None:
        self._decode_single_seq = value

    @property
    def decode_single_seq(self) -> bool:
        return self._decode_single_seq
```

### 1.2 Set Flag in Runner

```python
# In runner.py execute_step()
with EngineStepContext(...) as step_ctx:
    # Set flag for batch=1 decode optimization
    if step_desc.step_kind == "decode" and step_desc.num_seqs_active == 1:
        step_ctx.set_decode_single_seq(True)

    # ... rest of execute_step ...
```

### 1.3 Wire Through UnifiedGEMM / Selector

```python
# In gemm_selector.py or gemm.py
class UnifiedGEMM:
    def _should_use_metal(self, step_ctx, M: int, K: int, N: int) -> bool:
        # Batch=1 decode: always use native Metal (no MPS transitions)
        if hasattr(step_ctx, 'decode_single_seq') and step_ctx.decode_single_seq:
            return True

        # Existing heuristics for batch≥2
        return self._default_backend_selection(M, K, N)
```

### 1.4 Validation

After implementation:
- [ ] Re-run instrumentation: MPS transitions should drop to ~0 for batch=1
- [ ] Batch=1 decode: expect 20-30% speedup
- [ ] Batch≥2: must be unchanged (run full benchmark suite)

---

## PHASE 2: Coalesce Memory Barriers for Batch=1 Decode

**Impact:** 10-15% latency reduction
**Risk to batch≥2:** NONE (gated by flag, initially)

### 2.1 Current Barrier Locations (in _encode_transformer_layer)

```
1. After embedding lookup (once per step, not per layer)
2. After copy (residual save)           <- CAN REMOVE
3. After input_norm                     <- CAN COALESCE
4. After qkv_proj                       <- KEEP (QKV → RoPE)
5. After rope                           <- CAN COALESCE with attention barrier
6. After attention                      <- KEEP (attention → O-proj)
7. After o_proj + residual_add          <- KEEP (hidden update complete)
8. After copy (MLP residual save)       <- CAN REMOVE
9. After post_attn_norm                 <- CAN COALESCE
10. After mlp                           <- KEEP (MLP → residual add)
11. After final residual_add            <- KEEP (layer complete)
```

### 2.2 Minimal Required Barriers (4 per layer)

```python
# 1. After QKV GEMM + RoPE (before attention reads K/V cache)
step_ctx.memory_barrier()  # QKV + RoPE complete

# 2. After attention (before O-proj reads attention output)
step_ctx.memory_barrier()  # attention complete

# 3. After O-proj + residual (before post-attn norm reads hidden)
step_ctx.memory_barrier()  # attention block complete

# 4. After MLP + residual (layer complete, next layer can read)
step_ctx.memory_barrier()  # layer complete
```

### 2.3 Implementation (Gated)

```python
def _encode_transformer_layer(self, step_ctx, ...):
    # Use reduced barriers for batch=1 decode
    use_reduced_barriers = step_ctx.decode_single_seq

    # 1. Input LayerNorm + residual save
    self._elementwise.encode_copy(...)
    layer_ops.input_norm.encode(...)
    # NO barrier here if reduced mode

    # 2. QKV Projection
    layer_ops.qkv_proj.encode(...)
    # NO barrier here if reduced mode (coalesce with RoPE)

    # 3. RoPE
    layer_ops.rope.encode(...)
    step_ctx.memory_barrier()  # BARRIER 1: QKV+RoPE complete

    # 4. Attention
    layer_ops.attention.encode_decode_fused(...)
    step_ctx.memory_barrier()  # BARRIER 2: attention complete

    # 5. O Projection + residual
    layer_ops.o_proj.encode(...)
    self._elementwise.encode_residual_add(...)
    step_ctx.memory_barrier()  # BARRIER 3: attention block complete

    # 6. Post-attention norm + MLP residual save
    self._elementwise.encode_copy(...)
    layer_ops.post_attn_norm.encode(...)
    # NO barrier here if reduced mode

    # 7. MLP
    layer_ops.mlp.encode(...)
    # NO barrier here if reduced mode (inside MLP has its own)

    # 8. Final residual
    self._elementwise.encode_residual_add(...)
    step_ctx.memory_barrier()  # BARRIER 4: layer complete
```

### 2.4 Validation

- [ ] Barriers per step: 360 → ~160 (4 per layer × 40)
- [ ] No correctness issues (compare logits to baseline)
- [ ] Batch≥2: unchanged behavior (flag not set)

---

## PHASE 3: Precompute Constants

**Impact:** 5-10% latency reduction
**Risk to batch≥2:** NONE (global optimization)

### 3.1 Target Operations (Hot Path)

| Op | Constants to Prepack |
|----|---------------------|
| RMSNorm | hidden_size (uint), eps (float) |
| RoPE | head_size, num_heads, num_kv_heads, rotary_dim (all uint) |
| Elementwise | (none currently, but check) |

### 3.2 Implementation Example (RMSNorm)

```python
# In rmsnorm.py __init__:
import struct

class EngineRMSNorm:
    def __init__(self, ...):
        # ... existing ...

        # Precompute constant bytes
        self._hidden_size_bytes = struct.pack("I", hidden_size)
        self._eps_bytes = struct.pack("f", eps)

    def encode(self, ...):
        # ... existing buffer setup ...

        # Use precomputed bytes (no struct.pack in hot path)
        encoder.setBytes_length_atIndex_(self._hidden_size_bytes, 4, hidden_size_idx)
        encoder.setBytes_length_atIndex_(self._eps_bytes, 4, eps_idx)
```

### 3.3 Implementation Example (RoPE)

```python
# In elementwise.py EngineRoPE.__init__:
class EngineRoPE:
    def __init__(self, ...):
        # ... existing ...

        # Precompute constant bytes
        self._head_size_bytes = struct.pack("I", head_size)
        self._num_heads_bytes = struct.pack("I", num_heads)
        self._num_kv_heads_bytes = struct.pack("I", num_kv_heads)
        self._rotary_dim_bytes = struct.pack("I", self.rotary_dim)
```

---

## PHASE 4: Validation & Guardrails

### 4.1 Required Benchmarks

```bash
# Full benchmark suite
python benchmark_devstral_engine.py --batch-sizes 1 2 4 8 16

# Quick correctness check
python test_smolm_engine.py
```

### 4.2 Acceptance Criteria

| Metric | Baseline | Target | Regression Limit |
|--------|----------|--------|------------------|
| Batch=1 decode tok/s | ~11 | 20+ | N/A (improvement) |
| Batch=2 decode tok/s | ~22 | ~22 | ±3% |
| Batch=4 decode tok/s | ~44 | ~44 | ±3% |
| Batch=8 decode tok/s | ~85 | ~85 | ±3% |
| Batch=16 decode tok/s | ~127 | ~127 | ±3% |
| Prefill (all batches) | baseline | unchanged | ±5% |

### 4.3 Correctness Validation

- [ ] SmolLM test passes (exact logit match)
- [ ] Devstral top-5 predictions unchanged
- [ ] No NaN/Inf in outputs

---

## PHASE 5 (OPTIONAL): Fused Decode Kernels

Only if Phases 1-3 don't hit target. Lower priority fusions:

### 5.1 Easy Fusions (Low Risk)

- **RMSNorm + residual save**: Single kernel reads input, saves copy, outputs normalized
- **Residual add + RMSNorm**: Single kernel adds residual and normalizes
- **silu_mul already fused**: Check if gate+up GEMMs can share output buffer

### 5.2 Hard Fusions (High Risk, Defer)

- Full transformer layer fusion (requires in-kernel matmul)
- Cross-layer pipelining

---

## IMPLEMENTATION ORDER

```
Week 1:
├── Phase 0: Add instrumentation (1 day)
├── Validate diagnosis (1 day)
└── Phase 1: Native Metal GEMM for batch=1 (2-3 days)

Week 2:
├── Phase 2: Coalesce barriers (2 days)
├── Phase 3: Precompute constants (1 day)
└── Phase 4: Full validation (1-2 days)

If needed:
└── Phase 5: Selective fusions (1+ week)
```

---

## APPENDIX: Key Files to Modify

| File | Changes |
|------|---------|
| `vllm_apple/engine/step.py` | Add counters, decode_single_seq flag |
| `vllm_apple/engine/runner.py` | Set flag, reduce barriers |
| `vllm_apple/engine/ops/gemm.py` | Backend selection for batch=1 |
| `vllm_apple/engine/ops/gemm_selector.py` | Wire decode_single_seq check |
| `vllm_apple/engine/ops/rmsnorm.py` | Precompute constants |
| `vllm_apple/engine/ops/elementwise.py` | Precompute RoPE constants |

---

## APPENDIX: Current Decode Layer Sequence

From `runner.py:_encode_transformer_layer` (decode path):

```
1.  encode_copy (residual save)
2.  input_norm.encode
3.  memory_barrier
4.  qkv_proj.encode (GEMM - MPS transition if not native)
5.  memory_barrier
6.  rope.encode
7.  memory_barrier
8.  attention.encode_decode_fused
9.  memory_barrier
10. o_proj.encode (GEMM - MPS transition)
11. encode_residual_add
12. memory_barrier
13. encode_copy (MLP residual)
14. post_attn_norm.encode
15. memory_barrier
16. mlp.encode:
    - gate GEMM (MPS transition)
    - up GEMM (MPS transition)
    - memory_barrier
    - silu_mul
    - memory_barrier
    - down GEMM (MPS transition)
17. memory_barrier
18. encode_residual_add
19. memory_barrier
```

**Total per layer:** 9 barriers + 5-6 MPS transitions
**Total per step (40 layers):** ~360 barriers + ~200-240 MPS transitions

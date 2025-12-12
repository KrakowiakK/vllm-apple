# vLLM-Apple Knowledge Base

## Project Overview
Plugin enabling vLLM 0.12.0 to run on Apple Silicon (MPS backend).

## Architecture Map

```
vllm_apple/
├── platform.py                 # ApplePlatform - platform detection & MoE patches
├── v1/
│   ├── worker/
│   │   ├── apple_worker.py     # AppleWorker - subprocess worker, model loading
│   │   └── apple_model_runner.py # AppleModelRunner - input prep, forward, sampling
│   └── attention/
│       └── backends/
│           └── apple_attn.py   # AppleAttention - PyTorch SDPA attention backend
└── ops/
    └── apple_fused_moe.py      # MoE operations (not used - inline patches instead)
```

## Key Files and Responsibilities

### 1. `platform.py` (ApplePlatform)
- Platform detection for MPS
- Registers custom ops namespace
- **CRITICAL**: `_apply_deferred_moe_patches()` - patches fused_moe operations
- Patches applied: `topk_softmax_mps`, `fused_experts_mps`

### 2. `apple_worker.py` (AppleWorker)
- Runs in subprocess (important - patches must be applied here too!)
- `load_model()` - loads model, calls `_apply_moe_patches_after_load()`
- **CRITICAL**: `_apply_moe_patches_after_load()` - applies MoE patches after model loaded

### 3. `apple_model_runner.py` (AppleModelRunner)
- `_prepare_model_input()` - builds input_ids, positions from scheduler
- `_build_attn_metadata()` - **CRITICAL** - builds attention metadata
- `_execute_forward()` - runs model with set_forward_context
- `_sample_tokens()` - samples next tokens

### 4. `apple_attn.py` (AppleAttention)
- Uses PyTorch `scaled_dot_product_attention`
- `forward()` - handles prefill and decode
- `_compute_attention_no_cache()` - prefill (uses Q/K/V directly)
- `_compute_attention_with_cache()` - decode (reads K/V from cache)
- `_update_kv_cache()` - stores new K/V into cache

## Critical Discoveries

### 1. MoE Patching Must Happen in Worker Subprocess
**Problem**: Platform patches don't reach worker subprocess
**Solution**: Apply patches in `apple_worker.py._apply_moe_patches_after_load()` after model loading

### 2. PyTorch _OpNamespace Cannot Be Assigned
**Problem**: `torch.ops._moe_C.topk_softmax = ...` fails
**Solution**: Patch `sys.modules['vllm.model_executor.layers.fused_moe.fused_moe']` directly

### 3. Dtype Mismatch in fused_experts
**Problem**: `topk_weights` is float32, `hidden_states` is float16
**Solution**: Convert weights to hidden_states dtype before operations

### 4. seq_lens vs Full Sequence Length (FIXED 2024-12)
**Problem**: `seq_lens` in metadata was query length (1 for decode), not full sequence
**Solution**: Compute `full_seq_lens = num_computed_tokens + new_tokens`
- `seq_lens` tensor should be FULL sequence length
- `max_seq_len` should be max FULL sequence length
- `slot_mapping` must map to actual cache positions using block_ids

### 5. Qwen3 Uses QK-Norm
**Info**: Qwen3MoeAttention has `q_norm` and `k_norm` (RMSNorm on each head)
This is handled by the model itself, not attention backend.

### 6. KV Cache Scatter Optimization (FIXED 2024-12-11)
**Problem**: `scatter_` for KV cache updates is extremely slow on MPS (~1ms per op)
**Discovery**: 48 layers × 2 (K+V) × 1ms = 92ms overhead!
**Solution**: For single token (decode), use direct index assignment instead:
```python
# Old: 0.96ms per op
cache_flat.scatter_(0, indices, key)

# New: 0.006ms per op (160x faster!)
slot_idx = slot_mapping[0].item()
cache_flat[slot_idx] = key[0]
```
**Result**: Forward pass 235ms → 140ms, TPS 4.2 → 6.8 (+60%)

## Working Models
- GPT-2: 42 TPS ✓
- TinyLlama: 26 TPS ✓
- Qwen3-30B-A3B: **6.8 TPS** ✓ (after KV cache optimization)

## Performance Analysis (Qwen3-30B-A3B)

### Current Performance (After KV Cache Optimization)
- Forward pass: ~140ms
- Total step: ~145ms
- **TPS: 6.8-7.0**

### Breakdown (decode step)
| Component | Time | % of compute |
|-----------|------|--------------|
| MoE | 65ms | 46% |
| Attention (QKV+SDPA+O) | 52ms | 37% |
| RMSNorm | 24ms | 17% |
| **Pure compute** | 141ms | 100% |
| vLLM overhead | ~5ms | - |
| **Total** | ~146ms | - |

### MoE Benchmark Results
- **Izolowany MoE layer**: ~1.4ms per layer
- **48 MoE layers**: ~70ms total
- **Teoretyczny limit MoE-only**: ~14 TPS

### Chunk Size Benchmark
```
chunk_size=  8: 1.99ms per MoE layer
chunk_size= 16: 1.45ms
chunk_size= 32: 1.42ms
chunk_size= 64: 1.40ms (optimal)
chunk_size=128: 1.41ms
chunk_size=256: 1.41ms
```

## Key Finding: PyTorch MPS MoE Limit

**Discovery (2024-12-11)**: PyTorch bmm on MPS has a hard performance ceiling.

Benchmark results (single decode step):
- Chunked bmm (chunk=64): 1.45ms per MoE layer
- Single batch bmm: 1.41ms per MoE layer (only 3% faster!)
- **PyTorch MPS MoE ceiling: ~14-15 TPS** for Qwen3-30B-A3B

This means:
- Python loop overhead is negligible (<3%)
- The bottleneck is in PyTorch's MPS bmm implementation
- To exceed 14 TPS, we MUST use native Metal kernels

## Future Optimizations for 10+ TPS

Priority order:
1. **MoE optimization** (46% of compute) - Native Metal kernels or MLX
2. **Attention optimization** (37% of compute) - Better SDPA or custom kernels
3. **RMSNorm fusion** (17% of compute) - Fuse with other ops

Potential improvements:
| Optimization | Potential Gain | Effort |
|--------------|----------------|--------|
| MLX for MoE | 2-3x MoE speedup → ~9 TPS | Medium |
| Metal MoE kernels | 3-4x MoE speedup → ~10 TPS | High |
| Attention optimization | 1.5x attn speedup → ~8 TPS | Medium |

## Key Fix (2024-12-11): seq_lens in Attention Metadata
The critical bug was that `seq_lens` in AppleAttentionMetadata was set to the
query length (number of new tokens), not the FULL sequence length (prompt + generated).

**Before fix:**
- decode step: seq_lens=[1], max_seq_len=1
- Attention didn't know how many KV tokens to read from cache!

**After fix:**
- decode step: seq_lens=[prompt_len+generated], max_seq_len=full_length
- Attention correctly reads all past KV from cache

Changed in `apple_model_runner.py:_build_attn_metadata()`:
- Added `full_seq_lens = num_computed_tokens + new_tokens`
- Set `seq_lens_tensor = tensor(full_seq_lens)`
- Fixed `slot_mapping` to use actual block positions

## Common Issues and Solutions

### Circular Import with FusedMoEQuantConfig
Use `sys.modules` to access already-loaded modules instead of importing.

### vLLM sets fused_experts = None for non-CUDA
Apply patches after model loading when modules are already imported.

### MPS Scatter is Slow
Use direct index assignment for single-token updates (decode).

## Testing Commands

```bash
# Quick test
python test_vllm_apple.py

# Qwen3 test
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-30B-A3B', dtype='float16', max_model_len=256)
outputs = llm.generate(['Hello'], SamplingParams(max_tokens=20))
print(outputs[0].outputs[0].text)
"
```

## Attention Metadata Fields

- `num_actual_tokens`: Total tokens in batch
- `max_query_len`: Max NEW tokens per request
- `query_start_loc`: Cumsum of query lengths (for batching)
- `max_seq_len`: Max FULL sequence length (for KV cache)
- `seq_lens`: FULL sequence lengths tensor
- `block_table`: Block IDs for paged attention
- `slot_mapping`: Maps token positions to cache slots
- `num_decode_tokens`: Number of decode tokens (query_len=1)

## Performance History

| Date | Version | TPS | Notes |
|------|---------|-----|-------|
| 2024-12-10 | Initial | 0.3 | Python loop MoE |
| 2024-12-11 | Chunked bmm | 4.2 | Chunked batched MoE |
| 2024-12-11 | KV cache opt | 6.8 | Direct index assignment |
| 2024-12-11 | index_select | **10+** | 4.5x faster weight gather |
| 2024-12-11 | F.rms_norm | **11.5** | 5.8x faster RMSNorm |
| 2024-12-11 | Cached offsets | **11.6** | Attention index optimization |

## Key Optimizations Applied

### 1. KV Cache Write (scatter → direct assign)
- Problem: `scatter_` is slow on MPS (~1ms per op)
- Solution: Direct index assignment for single token
- Impact: 92ms → <1ms (96x faster)

### 2. Weight Gathering (indexing → index_select)
- Problem: `w1[ids]` is slow on MPS (88% of MoE time!)
- Solution: `torch.index_select(w1, 0, ids)`
- Impact: 0.586ms → 0.128ms (4.5x faster)
- Forward: 140ms → 84ms (~40% improvement)

### 3. RMSNorm (manual → F.rms_norm)
- Problem: Manual RMSNorm computation is slow on MPS
- Solution: Use `F.rms_norm()` which uses Metal acceleration
- Impact: 0.065ms → 0.011ms per call (5.8x faster)
- For 96 calls: ~6ms → ~1ms savings per forward
- Forward: 84ms → ~78ms

### 4. Attention Index Building (cached offsets)
- Problem: `torch.arange(block_size)` called on every attention layer
- Solution: Cache block offsets tensor and reuse
- Impact: ~1.4x speedup per layer, ~1.3ms total savings

### Current Breakdown (decode, 1 token)
| Component | Time |
|-----------|------|
| MoE (with index_select) | ~25ms |
| Attention | ~48ms |
| RMSNorm (with F.rms_norm) | ~4ms |
| Other (embedding, lm_head) | ~15ms |
| **Total forward** | ~75-85ms |
| vLLM overhead | ~5ms |
| **Total step** | ~80-90ms |
| **TPS** | **11.5** |

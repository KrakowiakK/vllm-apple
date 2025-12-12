# Metal Plugin V1.6 - Code Review Fixes Release

## Summary

V1.6 addresses all 5 critical issues identified in the code review, improving stability, reliability, and graceful degradation for the vLLM Apple plugin.

## Issues Fixed

### Issue 1: vLLM Fallback Disabled (HIGH)
**Problem**: `get_attn_backend_cls()` always returned `MetalAttentionBackend`, even when Metal was disabled or unsupported.

**Solution**:
- Created shared base classes (`base_attn.py`) for common SDPA logic
- Created new MPS-only backend (`mps_attn.py`) using PyTorch SDPA
- Fixed `platform.py` to return `MPSAttentionBackend` when:
  - `VLLM_METAL_ATTENTION=0`
  - Head size not in {32, 64, 96, 128}
  - Dtype is not float16

**Files Changed**:
- `vllm_apple/v1/attention/backends/base_attn.py` (NEW)
- `vllm_apple/v1/attention/backends/mps_attn.py` (NEW)
- `vllm_apple/platform.py`
- `tests/test_mps_fallback.py` (NEW)

### Issue 2: KV Sharing Can't Work (HIGH)
**Problem**: Each `MetalAttentionImpl` created its own `MetalKVCache`, breaking KV sharing for encoder-decoder models.

**Solution**:
- Added `_is_kv_shared` flag to track KV sharing status
- Skip `MetalKVCache` allocation when `kv_sharing_target_layer_name` is set
- Added safety assertions to prevent shared layers from using Metal zero-copy path
- Shared layers now use PyTorch KV cache tensor

**Files Changed**:
- `vllm_apple/v1/attention/backends/metal_attn.py`
- `tests/test_kv_sharing.py` (NEW)

### Issue 3: Metal Buffer Overflow (CRITICAL)
**Problem**: Hardcoded `max_batch_size=256` caused silent memory corruption for batches > 256.

**Solution**:
- Implemented `_ensure_batch_buffers()` for automatic buffer resizing with 1.5x growth
- All coupled buffers (query, output, seq_lens, block_table) resize together
- Changed default initial batch size from 256 to 64 (auto-resize handles larger batches)
- Removed hardcoded magic number 256

**Files Changed**:
- `vllm_apple/metal/bridge/metal_paged_attention_v2.py`
- `vllm_apple/v1/attention/backends/metal_attn.py`
- `tests/test_buffer_resize.py` (NEW)

### Issue 4: Assets Not Packaged (CRITICAL)
**Problem**: `pyproject.toml` specified non-existent paths, causing wheel to miss Metal shaders and dylib.

**Solution**:
- Updated `pyproject.toml` package-data to correct paths:
  ```toml
  vllm_apple = [
      "metal/kernels/*.metal",
      "metal/native/*.dylib",
      "ops/metal/*.metal"
  ]
  ```
- Added packaging test to verify assets in wheel

**Files Changed**:
- `pyproject.toml`
- `tests/test_packaging.py` (NEW)

### Issue 5: `.tolist()` Performance Footgun (MEDIUM)
**Problem**: `_compute_attention_no_cache()` called `.tolist()` every layer, forcing MPS device sync.

**Solution**:
- Added cached `query_start_loc_cpu` and `seq_lens_cpu` properties to `MetalAttentionMetadata`
- Properties cache the CPU list on first access, avoiding repeated device syncs
- Updated `_compute_attention_no_cache()` to use cached properties

**Files Changed**:
- `vllm_apple/v1/attention/backends/metal_attn.py`

## New Files Added

| File | Description |
|------|-------------|
| `vllm_apple/v1/attention/backends/base_attn.py` | Shared base classes for Apple attention backends |
| `vllm_apple/v1/attention/backends/mps_attn.py` | MPS-only backend using PyTorch SDPA |
| `tests/test_packaging.py` | Verify wheel contains all required assets |
| `tests/test_buffer_resize.py` | Test automatic buffer resizing |
| `tests/test_mps_fallback.py` | Test MPS fallback behavior |
| `tests/test_kv_sharing.py` | Test KV cache sharing |

## Directory Structure

```
vllm_apple/
├── __init__.py
├── platform.py                          # Updated: Fixed fallback logic
├── metal/
│   ├── bridge/
│   │   ├── metal_paged_attention_v2.py  # Updated: Auto-resize buffers
│   │   └── metal_paged_attention_fused.py
│   ├── kernels/
│   │   ├── paged_attention_v2.metal
│   │   └── paged_attention_fused.metal
│   ├── native/
│   │   └── libkv_write.dylib
│   └── kv_cache.py
├── ops/
│   └── metal/
│       └── moe_kernel_v2.metal
└── v1/
    └── attention/
        └── backends/
            ├── base_attn.py             # NEW: Shared base classes
            ├── mps_attn.py              # NEW: MPS fallback backend
            └── metal_attn.py            # Updated: KV sharing, caching
```

## Testing

Run the test suite:
```bash
cd /path/to/vllm-apple
pytest tests/ -v
```

### New Test Files
- `tests/test_packaging.py` - Verify wheel asset packaging
- `tests/test_buffer_resize.py` - Test buffer auto-resize
- `tests/test_mps_fallback.py` - Test MPS fallback selection
- `tests/test_kv_sharing.py` - Test KV cache sharing

## Validation Checklist

- [ ] Build wheel: `python -m build`
- [ ] Verify assets in wheel: `unzip -l dist/*.whl | grep -E "\.metal|\.dylib"`
- [ ] Test Metal backend: `VLLM_METAL_ATTENTION=1 python test_quick.py`
- [ ] Test MPS fallback: `VLLM_METAL_ATTENTION=0 python test_quick.py`
- [ ] Test large batch (>256): Verify auto-resize works
- [ ] Test encoder-decoder (if applicable): Verify KV sharing

## Breaking Changes

None. All changes are backward compatible.

## Performance Impact

- **Buffer resize**: Initial allocation reduced (64 vs 256), auto-resize adds minimal overhead
- **`.tolist()` caching**: Reduces device syncs per batch, improving prefill performance
- **MPS fallback**: Slower than Metal for decode, but provides graceful degradation

## Known Limitations

- MPS fallback is slower than Metal kernel for decode operations
- KV sharing tested with mocked encoder-decoder scenarios (no full model tests)
- Performance profiling for `.tolist()` caching improvement is manual

## Contributors

V1.6 Code Review Fixes implemented based on comprehensive code review findings.

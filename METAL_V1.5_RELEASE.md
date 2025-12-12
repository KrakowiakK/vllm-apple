# Metal Backend V1.5 FREEZE Report

**Date:** 2025-12-11
**Version:** V1.5 FREEZE (Final)

## Summary

Metal V1.5 is a stabilized release of the vLLM Apple Silicon plugin with:
- Clean, production-ready codebase
- Two-phase fused KV-write + attention decode kernel
- Full test coverage for Metal components
- Only h128 and generic kernels (h64 removed for simplicity)

## Performance Results

Tested on M3 Ultra with Qwen3-30B-A3B (MoE, 48 layers):

| Batch | tok/s | Scaling vs batch=1 |
|-------|-------|--------------------|
| 1     | 6.4   | 1.00x              |
| 2     | 11.2  | 1.73x              |
| 4     | 16.4  | 2.55x              |
| 8     | 22.5  | 3.50x              |
| 16    | 32.0  | 4.96x              |

### Attention Time Breakdown (batch=16)

```
metal_kernel: 87% of attention time
kv_update:    13% (prefill only, decode uses fused kernel)
sdpa:         <1% (prefill fallback)
```

## Architecture

### Two-Phase Fused Decode

The key optimization is the fused kernel that eliminates separate CPU KV update:

1. **Phase 1: `kv_write_decode`**
   - Writes new K/V for decode tokens directly to MTLBuffer
   - Grid: (num_seqs, num_kv_heads)
   - Each SIMD lane writes 4 dims for head_size=128

2. **Phase 2: `paged_attention_fused_h128`**
   - Reads from already-populated KV cache
   - Computes attention with online softmax
   - Grid: (num_seqs, num_query_heads)

Both kernels run in same command buffer - no CPU sync between them.

### Memory Layout

```
KV Cache: [num_blocks, num_kv_heads, block_size, head_size]
Query:    [num_seqs, num_query_heads, head_size]
Output:   [num_seqs, num_query_heads, head_size]
```

### Zero-Copy Architecture

- MetalKVCache owns physical KV cache in MTLBuffer (unified memory)
- vLLM's kv_cache tensor is ignored for decode - we use MetalKVCache
- Only query/output copied between MPS and Metal
- No 15GB KV cache copy per decode step

## Final File Structure

```
vllm_apple/
    __init__.py
    platform.py
    metal/
        __init__.py
        kv_cache.py
        block_allocator.py          # NEW - extracted from kv_cache
        bridge/
            __init__.py
            metal_runtime.py
            metal_paged_attention_v2.py
            metal_paged_attention_fused.py
        kernels/
            paged_attention_v2.metal
            paged_attention_fused.metal
        native/
            kv_write.c
            libkv_write.dylib
    ops/
        __init__.py
        apple_fused_moe.py
        mlx_moe.py
        metal/
            moe_kernel_v2.metal
            moe_metal.py
    v1/
        __init__.py
        attention/
            __init__.py
            backends/
                __init__.py
                metal_attn.py
        worker/
            __init__.py
            apple_input_batch.py
            apple_model_runner.py
            apple_worker.py
            gpu/
                attn_utils.py
tests/
    test_metal_attention.py
    test_basic_integration.py       # NEW
    test_fused_kernel.py            # NEW
    test_kv_cache_layout.py         # NEW
    benchmark_throughput.py         # NEW
```

## Metal Kernels (Final V1.5)

### paged_attention_fused.metal

3 kernels (h64 removed in V1.5):
1. `kv_write_decode` - Fast K/V write for decode tokens
2. `paged_attention_fused_h128` - Optimized attention for head_size=128
3. `paged_attention_fused_generic` - Fallback for other head sizes

### paged_attention_v2.metal

V2 kernel for prefill path (batched KV write + attention).

## API Surface

### Public Classes

```python
from vllm_apple.metal import (
    MetalKVCache,
    MetalPagedAttentionV2,
    MetalPagedAttentionFused,
    is_metal_available,
)

from vllm_apple.metal.block_allocator import (
    MetalBlockAllocator,
)

from vllm_apple.v1.attention import (
    MetalAttentionBackend,
    MetalAttentionImpl,
    MetalAttentionMetadata,
    MetalAttentionMetadataBuilder,
)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_METAL_ATTENTION` | `1` | Enable Metal attention |
| `VLLM_METAL_FUSED_KV` | `1` | Enable fused decode kernel |

### Removed Environment Variables (V1.5)

- `VLLM_METAL_TGMEM` - removed
- `VLLM_METAL_MULTISEQ` - removed
- `VLLM_METAL_PREFETCH` - removed
- `VLLM_METAL_V3` - removed
- `VLLM_METAL_UNROLL2` - removed

## Known Limitations

1. **MPS->CPU sync for prefill**: Prefill tokens still require MPS sync before CPU transfer
2. **FP16 only**: No FP8/INT8 quantized KV cache support
3. **No ALiBi**: Alibi slopes ignored (warning logged)
4. **No logits softcap**: Softcap ignored (warning logged)

## Changes in V1.5 Final

1. Extracted `MetalBlockAllocator` to separate file `block_allocator.py`
2. Removed `paged_attention_fused_h64` kernel (use generic instead)
3. Updated kernel selection: only h128 + generic
4. Added comprehensive test suite:
   - `test_basic_integration.py`
   - `test_fused_kernel.py`
   - `test_kv_cache_layout.py`
   - `benchmark_throughput.py`
5. Fixed imports from `apple_attn` to `metal_attn`
6. Restored `native/` directory with `libkv_write.dylib`

## Future Work (Post V1.5)

1. **Async prefill**: Hide MPS->CPU sync latency
2. **FP8 KV cache**: Memory savings for longer contexts
3. **Multi-query chunked prefill**: Better memory efficiency
4. **Speculative decoding**: Draft model acceleration

---

**Metal V1.5 FREEZE - Ready for production use**
**Project ready for Wariant A (full Metal backend)**

# Metal PagedAttention Backend for vLLM - MVP Summary

## Overview

This document summarizes the Metal PagedAttention backend implementation for vLLM on Apple Silicon. The backend provides a custom Metal compute kernel for decode attention, achieving significant throughput improvements at batch sizes >= 2.

## Performance Results (TinyLlama-1.1B, M-series Mac)

### Throughput Comparison

| Batch Size | Metal PagedAttention | MPS Baseline | Speedup |
|------------|---------------------|--------------|---------|
| 1          | 18.2 tok/s          | 37.1 tok/s   | 0.49x   |
| 2          | 33.9 tok/s          | 19.3 tok/s   | 1.76x   |
| 4          | 66.9 tok/s          | 28.4 tok/s   | 2.36x   |
| 8          | 129.7 tok/s         | 41.6 tok/s   | 3.12x   |

### Scaling Analysis

- **Metal PagedAttention**: Near-linear scaling with batch size
  - batch=1 -> batch=8: 7.1x improvement
- **MPS Baseline**: Sub-linear scaling
  - batch=1 -> batch=8: 1.1x improvement (essentially flat)

## What Works

### 1. Zero-Copy KV Cache (MetalKVCache)
- KV cache stored in MTLBuffer (unified memory)
- Direct writes to MTLBuffer without copying entire cache
- Memory layout: `[num_blocks, num_kv_heads, block_size, head_size]`

### 2. Continuous Batching
- Full support for vLLM's continuous batching scheduler
- Proper handling of decode/prefill split

### 3. Native KV Write
- C library (`libkv_write.dylib`) for fast strided memcpy
- Eliminates Python loop overhead (reduced from ~26ms to ~10ms per batch)
- Called via ctypes for minimal overhead

### 4. Metal Kernel Performance
- Custom Metal compute shader for PagedAttention
- Supports head_size: 32, 64, 96, 128
- Block sizes: 16 (configurable)
- GQA (Grouped Query Attention) support

## Known Limitations

### 1. Batch=1 Performance Overhead

For batch=1, MPS baseline is faster (37.1 vs 18.2 tok/s). This is due to **MPS -> CPU transfer overhead per layer**:

**Profiling breakdown (per 50 tokens, 22 layers, 1100 attention calls):**
```
kv_update_ms:    1281.6ms (47% of attention time)
  - kv_sync_ms:    569.9ms (MPS synchronize)
  - kv_to_cpu_ms:  693.3ms (tensor transfer to CPU)
  - kv_compute_ms:   3.5ms (block_id/offset calculation)
  - kv_native_ms:   10.0ms (native C memcpy)

metal_kernel_ms: 1433.3ms (53% of attention time)
```

**Root cause:** The model runs on MPS, but our attention backend and KV cache run in CPU/Metal unified memory. This requires `torch.mps.synchronize()` and `.cpu().numpy()` for every layer's key/value tensors.

### 2. Double KV Cache Allocation

vLLM allocates its own KV cache tensor, while MetalKVCache allocates MTLBuffer. This doubles memory usage for KV cache. The vLLM tensor is unused for decode (only for sizing).

### 3. Architecture Constraint

The bottleneck is architectural, not in the kernel or KV update code:
- Model (Q/K/V projections): MPS
- Attention backend (KV cache + kernel): CPU/Metal
- Every layer requires MPS -> CPU bridge

## File Structure

```
vllm_apple/
  v1/attention/backends/
    metal_attn.py          # MetalAttentionBackend, MetalAttentionImpl
  metal/
    bridge/
      metal_paged_attention_v2.py  # MetalPagedAttentionV2 kernel interface
    kernels/
      paged_attention_v2.metal     # Metal compute shader
    kv_cache.py            # MetalKVCache (MTLBuffer management)
    native/
      kv_write.c           # Native C library for fast KV writes
      libkv_write.dylib    # Compiled shared library
```

## Future Improvements (v2)

### Option A: Full Metal Path
- Move Q/K/V projections to Metal (bypass MPS for attention path)
- Eliminates MPS -> CPU transfer overhead
- Requires significant changes to model architecture

### Option B: CPU+Metal Model Runner
- Create separate model_runner that runs entirely on CPU+Metal
- No MPS involvement in attention path
- Uses Apple's Accelerate framework for linear operations

### Option C: Unified KV Cache
- Modify vLLM to accept external KV cache (MetalKVCache)
- Eliminates double memory allocation
- Requires changes to vLLM core

## Usage

Enable Metal PagedAttention:
```bash
export VLLM_METAL_ATTENTION=1
```

Disable (use MPS baseline):
```bash
export VLLM_METAL_ATTENTION=0
```

## Recommendation

- **batch >= 2**: Use Metal PagedAttention (2-3x faster)
- **batch = 1**: Consider MPS baseline for now (2x faster)

The current implementation is a functional MVP suitable for batched inference workloads. Single-request latency optimization requires the architectural changes outlined in "Future Improvements".

---

*Generated: 2025-12-11*
*vLLM version: 0.12.0*
*vllm-apple plugin: MVP Metal PagedAttention*

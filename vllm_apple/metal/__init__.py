"""Metal backend for vLLM on Apple Silicon.

This module provides Metal compute shader implementations for:
- PagedAttention V2 (prefill path with batched KV write)
- PagedAttention Fused (decode path with integrated KV write)
- MetalKVCache (unified GPU/CPU memory for KV cache)

Architecture:
- kv_cache.py: MetalKVCache with MTLBuffer for unified memory
- kernels/: Metal shader source files (.metal)
  - paged_attention_v2.metal: Batched prefill kernel
  - paged_attention_fused.metal: Fused decode kernel
- bridge/: Python <-> Metal interop layer
  - metal_paged_attention_v2.py: V2 prefill wrapper
  - metal_paged_attention_fused.py: Fused decode wrapper

Usage:
    from vllm_apple.metal import MetalKVCache, is_metal_available

    if is_metal_available():
        kv_cache = MetalKVCache(num_blocks, block_size, num_kv_heads, head_size)
"""

from .kv_cache import MetalKVCache

from .bridge import (
    MetalPagedAttentionV2,
    MetalPagedAttentionFused,
    is_metal_available,
)

__all__ = [
    # KV Cache
    "MetalKVCache",
    # Metal Runtime
    "MetalPagedAttentionV2",
    "MetalPagedAttentionFused",
    "is_metal_available",
]

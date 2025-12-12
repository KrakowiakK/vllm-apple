"""Metal bridge module for PyTorch <-> Metal interop.

Two primary implementations:
- MetalPagedAttentionV2: For prefill path (batched KV write + attention)
- MetalPagedAttentionFused: For decode path (fused KV write + attention)
"""

from .metal_runtime import is_metal_available

from .metal_paged_attention_v2 import (
    MetalPagedAttentionV2,
    metal_write_kv_batch,
)

from .metal_paged_attention_fused import (
    MetalPagedAttentionFused,
)

__all__ = [
    # V2 Prefill
    "MetalPagedAttentionV2",
    "metal_write_kv_batch",
    # Fused Decode
    "MetalPagedAttentionFused",
    # Utils
    "is_metal_available",
]

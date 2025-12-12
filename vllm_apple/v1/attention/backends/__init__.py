# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal attention backend implementation for vLLM v1.

Single unified backend using:
- MetalPagedAttentionV2 for prefill (batched KV write)
- MetalPagedAttentionFused for decode (fused KV write + attention)
"""

from vllm_apple.v1.attention.backends.metal_attn import (
    MetalAttentionBackend,
    MetalAttentionImpl,
    MetalAttentionMetadata,
    MetalAttentionMetadataBuilder,
)

__all__ = [
    "MetalAttentionBackend",
    "MetalAttentionImpl",
    "MetalAttentionMetadata",
    "MetalAttentionMetadataBuilder",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine Operations for vLLM-Apple Metal Engine v2.0.

This package provides encode-only operations for the Metal engine.
Each operation encodes commands to a command buffer without executing.
Execution happens at step boundary via single waitUntilCompleted().

Key principle: ALL ops use encode-only APIs. No internal waits.

Available Operations:
    attention.py    - Paged attention encoding
    kv_write.py     - KV cache write encoding
    gemm.py         - GEMM via MPSMatrixMultiplication
    qkv.py          - QKV projection
    rmsnorm.py      - RMSNorm kernel
    mlp.py          - MLP/FFN encoding
    elementwise.py  - Residual, activation, RoPE
    embedding.py    - Token embedding
    lm_head.py      - LM head projection

Usage:
    from vllm_apple.engine.ops import PagedAttentionOp, KVWriteOp

    # Create op
    attn_op = PagedAttentionOp(context, num_kv_heads=32, ...)

    # Encode to command buffer (no wait)
    attn_op.encode(step_ctx, query, kv_cache, block_table, output)
"""

# Lazy imports to avoid loading all ops when only some are needed
__all__ = [
    "PagedAttentionOp",
    "KVWriteOp",
    "EngineGEMM",
    "EngineQKVProjection",
    "EngineOProjection",
]


def __getattr__(name):
    """Lazy import of ops to reduce startup time."""
    if name == "PagedAttentionOp":
        from .attention import PagedAttentionOp
        return PagedAttentionOp
    elif name == "KVWriteOp":
        from .kv_write import KVWriteOp
        return KVWriteOp
    elif name == "EngineGEMM":
        from .gemm import EngineGEMM
        return EngineGEMM
    elif name == "EngineQKVProjection":
        from .qkv import EngineQKVProjection
        return EngineQKVProjection
    elif name == "EngineOProjection":
        from .qkv import EngineOProjection
        return EngineOProjection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

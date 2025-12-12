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
    embedding.py    - Token embedding lookup
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
    # Attention ops
    "PagedAttentionOp",
    "KVWriteOp",
    # GEMM and projections
    "EngineGEMM",
    "EngineQKVProjection",
    "EngineOProjection",
    # Normalization
    "EngineRMSNorm",
    # Elementwise ops
    "EngineElementwiseOps",
    "EngineRoPE",
    # MLP
    "EngineMLP",
    # Embedding and LM head
    "EngineEmbedding",
    "EngineLMHead",
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
    elif name == "EngineRMSNorm":
        from .rmsnorm import EngineRMSNorm
        return EngineRMSNorm
    elif name == "EngineElementwiseOps":
        from .elementwise import EngineElementwiseOps
        return EngineElementwiseOps
    elif name == "EngineRoPE":
        from .elementwise import EngineRoPE
        return EngineRoPE
    elif name == "EngineMLP":
        from .mlp import EngineMLP
        return EngineMLP
    elif name == "EngineEmbedding":
        from .embedding import EngineEmbedding
        return EngineEmbedding
    elif name == "EngineLMHead":
        from .lm_head import EngineLMHead
        return EngineLMHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

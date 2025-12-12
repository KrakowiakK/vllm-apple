# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QKV Projection Operation for vLLM-Apple Metal Engine v2.0.

This module provides encode-only QKV projection operations. The operation
encodes GEMM dispatches for computing Q, K, V from hidden states WITHOUT
executing. Execution happens at step boundary.

Key Principle: Encode-only API. No internal waits.

QKV projection: hidden_states @ W_qkv -> [Q, K, V]

The operation supports:
- Fused QKV projection (single GEMM)
- Separate Q, K, V projections (three GEMMs)
- With or without bias

Usage:
    from vllm_apple.engine.ops.qkv import EngineQKVProjection

    # Create op during initialization
    qkv_proj = EngineQKVProjection(
        context=engine_context,
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        weight_buffer=W_qkv_buffer,
    )

    # Encode QKV projection (no wait)
    with step_ctx:
        Q, K, V = qkv_proj.encode(
            step_ctx=step_ctx,
            hidden_states=hidden_buf,
            num_tokens=num_tokens,
        )
"""

from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from vllm.logger import init_logger
from ..tensor import EngineTensor, EngineDType
from .gemm import EngineGEMM

logger = init_logger(__name__)


@dataclass
class QKVProjectionConfig:
    """Configuration for QKV projection."""
    hidden_size: int
    num_heads: int  # Query heads
    num_kv_heads: int  # KV heads (for GQA)
    head_size: int
    fused: bool = True  # Whether weights are fused [hidden, (q + k + v)]
    has_bias: bool = False

    @property
    def q_size(self) -> int:
        """Total Q projection size."""
        return self.num_heads * self.head_size

    @property
    def k_size(self) -> int:
        """Total K projection size."""
        return self.num_kv_heads * self.head_size

    @property
    def v_size(self) -> int:
        """Total V projection size."""
        return self.num_kv_heads * self.head_size

    @property
    def qkv_size(self) -> int:
        """Total fused QKV size."""
        return self.q_size + self.k_size + self.v_size


class EngineQKVProjection:
    """Encode-only QKV projection operation.

    This op encodes GEMM for QKV projection: hidden @ W_qkv -> [Q, K, V]

    Supports both fused (single GEMM) and separate projections.

    After GEMM, returns views into the output buffer for Q, K, V:
    - Q: [num_tokens, num_heads, head_size]
    - K: [num_tokens, num_kv_heads, head_size]
    - V: [num_tokens, num_kv_heads, head_size]

    Attributes:
        context: MetalEngineContext
        config: QKVProjectionConfig
        gemm: EngineGEMM operation
    """

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        weight_buffer: Optional[Any] = None,  # MTLBuffer for W_qkv
        bias_buffer: Optional[Any] = None,  # MTLBuffer for bias
        fused: bool = True,
    ):
        """Initialize QKV projection op.

        Args:
            context: MetalEngineContext
            hidden_size: Input hidden dimension
            num_heads: Number of query heads
            num_kv_heads: Number of KV heads (for GQA)
            head_size: Dimension per head
            weight_buffer: Pre-loaded weight buffer [hidden_size, qkv_size]
            bias_buffer: Optional bias buffer [qkv_size]
            fused: Whether weights are fused
        """
        self._context = context
        self.config = QKVProjectionConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            fused=fused,
            has_bias=bias_buffer is not None,
        )

        # Store weights
        self._weight_buffer = weight_buffer
        self._bias_buffer = bias_buffer

        # Create GEMM op
        self._gemm = EngineGEMM(context)

        # Pre-compute offsets for Q, K, V splits
        self._q_offset = 0
        self._k_offset = self.config.q_size
        self._v_offset = self.config.q_size + self.config.k_size

        logger.info(
            f"EngineQKVProjection initialized: "
            f"hidden={hidden_size}, heads={num_heads}/{num_kv_heads}, "
            f"qkv_size={self.config.qkv_size}"
        )

    def set_weights(
        self,
        weight_buffer: Any,
        bias_buffer: Optional[Any] = None,
    ) -> None:
        """Set weight buffers.

        Args:
            weight_buffer: W_qkv [hidden_size, qkv_size]
            bias_buffer: Optional bias [qkv_size]
        """
        self._weight_buffer = weight_buffer
        self._bias_buffer = bias_buffer

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        hidden_states: Union[EngineTensor, Any],  # [num_tokens, hidden_size]
        output_buffer: Any,  # MTLBuffer for QKV output
        num_tokens: int,
    ) -> Tuple[Any, Any, Any]:
        """Encode QKV projection to command buffer.

        Computes: QKV = hidden_states @ W_qkv

        Returns views (byte offsets) into output_buffer for Q, K, V.

        Args:
            step_ctx: EngineStepContext with encoder
            hidden_states: Input [num_tokens, hidden_size]
            output_buffer: Output buffer for QKV [num_tokens, qkv_size]
            num_tokens: Number of tokens

        Returns:
            Tuple of (Q_info, K_info, V_info) with buffer offsets
        """
        if self._weight_buffer is None:
            raise RuntimeError("Weights not set - call set_weights() first")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # Encode GEMM: hidden @ W_qkv -> output
        # Shape: [num_tokens, hidden_size] @ [hidden_size, qkv_size] -> [num_tokens, qkv_size]
        self._gemm.encode(
            step_ctx=step_ctx,
            A=hidden_states,
            B=self._weight_buffer,
            C=output_buffer,
            M=num_tokens,
            K=self.config.hidden_size,
            N=self.config.qkv_size,
        )

        # If we have bias, need to add it
        if self._bias_buffer is not None:
            # TODO: Encode bias addition kernel
            logger.warning("QKV bias not yet implemented")

        # Memory barrier after GEMM
        step_ctx.memory_barrier()

        # Return info about Q, K, V locations in output
        # These are byte offsets into the output buffer
        element_size = 2  # float16
        q_info = {
            'buffer': output_buffer,
            'offset': 0,
            'shape': (num_tokens, self.config.num_heads, self.config.head_size),
            'size_bytes': num_tokens * self.config.q_size * element_size,
        }
        k_info = {
            'buffer': output_buffer,
            'offset': num_tokens * self.config.q_size * element_size,
            'shape': (num_tokens, self.config.num_kv_heads, self.config.head_size),
            'size_bytes': num_tokens * self.config.k_size * element_size,
        }
        v_info = {
            'buffer': output_buffer,
            'offset': num_tokens * (self.config.q_size + self.config.k_size) * element_size,
            'shape': (num_tokens, self.config.num_kv_heads, self.config.head_size),
            'size_bytes': num_tokens * self.config.v_size * element_size,
        }

        return q_info, k_info, v_info

    def get_output_size(self, num_tokens: int) -> int:
        """Get required output buffer size in bytes.

        Args:
            num_tokens: Number of tokens

        Returns:
            Size in bytes
        """
        element_size = 2  # float16
        return num_tokens * self.config.qkv_size * element_size

    def get_params_dict(self) -> Dict[str, Any]:
        """Get operation parameters as dict."""
        return {
            "hidden_size": self.config.hidden_size,
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "head_size": self.config.head_size,
            "q_size": self.config.q_size,
            "k_size": self.config.k_size,
            "v_size": self.config.v_size,
            "qkv_size": self.config.qkv_size,
            "fused": self.config.fused,
            "has_bias": self.config.has_bias,
        }


class EngineOProjection:
    """Encode-only output projection operation.

    This op encodes GEMM for output projection: attn_output @ W_o -> hidden_states

    Attributes:
        context: MetalEngineContext
        hidden_size: Output hidden dimension
        num_heads: Number of attention heads
        head_size: Dimension per head
    """

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        hidden_size: int,
        num_heads: int,
        head_size: int,
        weight_buffer: Optional[Any] = None,
        bias_buffer: Optional[Any] = None,
    ):
        """Initialize O projection op.

        Args:
            context: MetalEngineContext
            hidden_size: Output hidden dimension
            num_heads: Number of attention heads
            head_size: Dimension per head
            weight_buffer: W_o [num_heads * head_size, hidden_size]
            bias_buffer: Optional bias [hidden_size]
        """
        self._context = context
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.input_size = num_heads * head_size

        self._weight_buffer = weight_buffer
        self._bias_buffer = bias_buffer
        self._gemm = EngineGEMM(context)

        logger.info(
            f"EngineOProjection initialized: "
            f"input={self.input_size}, hidden={hidden_size}"
        )

    def set_weights(
        self,
        weight_buffer: Any,
        bias_buffer: Optional[Any] = None,
    ) -> None:
        """Set weight buffers."""
        self._weight_buffer = weight_buffer
        self._bias_buffer = bias_buffer

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        attn_output: Union[EngineTensor, Any],  # [num_tokens, num_heads * head_size]
        output_buffer: Any,  # MTLBuffer for output
        num_tokens: int,
    ) -> None:
        """Encode O projection to command buffer.

        Computes: output = attn_output @ W_o

        Args:
            step_ctx: EngineStepContext
            attn_output: Attention output [num_tokens, num_heads * head_size]
            output_buffer: Output buffer [num_tokens, hidden_size]
            num_tokens: Number of tokens
        """
        if self._weight_buffer is None:
            raise RuntimeError("Weights not set")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        self._gemm.encode(
            step_ctx=step_ctx,
            A=attn_output,
            B=self._weight_buffer,
            C=output_buffer,
            M=num_tokens,
            K=self.input_size,
            N=self.hidden_size,
        )

        if self._bias_buffer is not None:
            logger.warning("O projection bias not yet implemented")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LM Head Operation for vLLM-Apple Metal Engine v2.0.

This module provides encode-only language model head operations.
Projects hidden states to vocabulary logits.

Key Principle: Encode-only API. No internal waits.

Usage:
    from vllm_apple.engine.ops.lm_head import EngineLMHead

    # Create op during initialization
    lm_head = EngineLMHead(
        context=engine_context,
        hidden_size=4096,
        vocab_size=32000,
    )

    # Set weight buffer
    lm_head.set_weights(lm_head_weights)

    # Encode LM head projection (no wait)
    lm_head.encode(step_ctx, hidden_states, logits_buffer, num_tokens)
"""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

from vllm.logger import init_logger
from ..tensor import EngineTensor, EngineDType
from .gemm import EngineGEMM

logger = init_logger(__name__)


@dataclass
class LMHeadConfig:
    """Configuration for LM head operation."""
    hidden_size: int
    vocab_size: int
    dtype: EngineDType = EngineDType.FLOAT16
    tie_word_embeddings: bool = False


class EngineLMHead:
    """Encode-only language model head operation.

    This op projects hidden states to vocabulary logits:
        logits = hidden_states @ lm_head_weight.T

    Can optionally share weights with token embeddings (tie_word_embeddings).
    Uses GEMM for the projection.

    Attributes:
        context: MetalEngineContext
        config: LMHeadConfig
        weight_buffer: MTLBuffer containing weights [vocab_size, hidden_size]
    """

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        hidden_size: int,
        vocab_size: int,
        tie_word_embeddings: bool = False,
        dtype: EngineDType = EngineDType.FLOAT16,
    ):
        """Initialize LM head op.

        Args:
            context: MetalEngineContext
            hidden_size: Hidden dimension
            vocab_size: Vocabulary size
            tie_word_embeddings: Whether to share weights with embeddings
            dtype: Data type
        """
        self._context = context
        self.config = LMHeadConfig(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dtype=dtype,
            tie_word_embeddings=tie_word_embeddings,
        )
        self._weight_buffer = None

        # Use GEMM for projection
        self._gemm = EngineGEMM(context)

        logger.info(
            f"EngineLMHead initialized: hidden_size={hidden_size}, "
            f"vocab_size={vocab_size}, tie_embeddings={tie_word_embeddings}"
        )

    def set_weights(self, weight_buffer: Any) -> None:
        """Set LM head weight buffer.

        Args:
            weight_buffer: MTLBuffer containing [vocab_size, hidden_size] weights
        """
        self._weight_buffer = weight_buffer

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        hidden_states: Union[EngineTensor, Any],  # [num_tokens, hidden_size]
        output_buffer: Union[EngineTensor, Any],  # [num_tokens, vocab_size]
        num_tokens: int,
        selected_token_indices: Optional[Any] = None,  # For selecting specific tokens
    ) -> None:
        """Encode LM head projection to command buffer.

        Computes: logits = hidden_states @ weight.T
        Where weight is [vocab_size, hidden_size], so we compute:
            [num_tokens, hidden_size] @ [hidden_size, vocab_size] = [num_tokens, vocab_size]

        Args:
            step_ctx: EngineStepContext with encoder
            hidden_states: Input hidden states [num_tokens, hidden_size]
            output_buffer: Output logits [num_tokens, vocab_size]
            num_tokens: Number of tokens
            selected_token_indices: Optional indices for selecting which tokens
                                   to compute logits for (for efficiency in decode)
        """
        if self._weight_buffer is None:
            raise RuntimeError("Weights not set - call set_weights() first")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # LM head is: [num_tokens, hidden_size] @ [hidden_size, vocab_size] = [num_tokens, vocab_size]
        # Since weights are [vocab_size, hidden_size], we transpose B
        # This means: A @ B^T where A=[num_tokens, hidden_size], B=[vocab_size, hidden_size]
        #
        # With MPSMatrixMultiplication:
        # - A: [num_tokens, hidden_size] (M=num_tokens, K=hidden_size)
        # - B: [vocab_size, hidden_size] with transpose_B=True -> becomes [hidden_size, vocab_size]
        # - Result: [num_tokens, vocab_size]

        self._gemm.encode(
            step_ctx=step_ctx,
            A=hidden_states,
            B=self._weight_buffer,
            C=output_buffer,
            M=num_tokens,
            K=self.config.hidden_size,
            N=self.config.vocab_size,
            transpose_B=True,  # Weight is [vocab_size, hidden_size], need transposed
        )

    def encode_with_selection(
        self,
        step_ctx: Any,
        hidden_states: Union[EngineTensor, Any],  # [total_tokens, hidden_size]
        output_buffer: Union[EngineTensor, Any],  # [selected_tokens, vocab_size]
        token_indices: Any,  # [num_selected] indices to compute
        num_selected: int,
    ) -> None:
        """Encode LM head for selected tokens only.

        This is useful in decode mode where we only need logits for the last
        token of each sequence.

        Args:
            step_ctx: EngineStepContext
            hidden_states: All hidden states [total_tokens, hidden_size]
            output_buffer: Output buffer [num_selected, vocab_size]
            token_indices: Indices of tokens to compute [num_selected]
            num_selected: Number of selected tokens
        """
        # For now, just compute all logits
        # TODO: Implement optimized selection kernel
        self.encode(
            step_ctx=step_ctx,
            hidden_states=hidden_states,
            output_buffer=output_buffer,
            num_tokens=num_selected,
        )

    def get_output_size(self, num_tokens: int) -> int:
        """Get required output buffer size in bytes."""
        element_size = 2  # float16
        return num_tokens * self.config.vocab_size * element_size

    def get_weight_size(self) -> int:
        """Get required weight buffer size in bytes."""
        element_size = 2  # float16
        return self.config.vocab_size * self.config.hidden_size * element_size

    def get_params_dict(self) -> Dict[str, Any]:
        """Get operation parameters as dict."""
        return {
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "tie_word_embeddings": self.config.tie_word_embeddings,
        }

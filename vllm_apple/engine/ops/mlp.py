# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLP/FFN Operation for vLLM-Apple Metal Engine v2.0.

This module provides encode-only MLP (Feed-Forward Network) operations.
Supports both standard MLP and gated MLP (used in LLaMA, Qwen, etc.).

Key Principle: Encode-only API. No internal waits.

Standard MLP:
    output = down_proj(activation(up_proj(x)))

Gated MLP (LLaMA/Qwen style):
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

Usage:
    from vllm_apple.engine.ops.mlp import EngineMLP

    # Create op during initialization
    mlp = EngineMLP(
        context=engine_context,
        hidden_size=4096,
        intermediate_size=11008,
        gated=True,  # For LLaMA-style gated MLP
    )

    # Set weights
    mlp.set_weights(gate_proj, up_proj, down_proj)

    # Encode MLP (no wait)
    mlp.encode(step_ctx, hidden_states, output, num_tokens)
"""

from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from vllm.logger import init_logger
from ..tensor import EngineTensor, EngineDType
from .gemm import EngineGEMM
from .elementwise import EngineElementwiseOps

logger = init_logger(__name__)


@dataclass
class MLPConfig:
    """Configuration for MLP operation."""
    hidden_size: int
    intermediate_size: int
    gated: bool = True  # Gated MLP (LLaMA/Qwen) vs standard MLP
    activation: str = "silu"  # silu or gelu
    bias: bool = False


class EngineMLP:
    """Encode-only MLP/FFN operation.

    Supports two architectures:

    1. Gated MLP (gated=True, default for LLaMA/Qwen):
       intermediate = silu(gate_proj(x)) * up_proj(x)
       output = down_proj(intermediate)

    2. Standard MLP (gated=False):
       intermediate = activation(up_proj(x))
       output = down_proj(intermediate)

    Uses GEMM for projections and fused elementwise kernels for activation.

    Attributes:
        context: MetalEngineContext
        config: MLPConfig
    """

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        hidden_size: int,
        intermediate_size: int,
        gated: bool = True,
        activation: str = "silu",
        bias: bool = False,
    ):
        """Initialize MLP op.

        Args:
            context: MetalEngineContext
            hidden_size: Input/output hidden dimension
            intermediate_size: Intermediate (FFN) dimension
            gated: Whether to use gated MLP
            activation: Activation function ("silu" or "gelu")
            bias: Whether projections have bias
        """
        self._context = context
        self.config = MLPConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gated=gated,
            activation=activation,
            bias=bias,
        )

        # Weight buffers
        self._gate_proj = None  # [hidden_size, intermediate_size]
        self._up_proj = None    # [hidden_size, intermediate_size]
        self._down_proj = None  # [intermediate_size, hidden_size]

        # Bias buffers (optional)
        self._gate_bias = None
        self._up_bias = None
        self._down_bias = None

        # Sub-operations
        self._gemm = EngineGEMM(context)
        self._elementwise = EngineElementwiseOps(context)

        logger.info(
            f"EngineMLP initialized: hidden={hidden_size}, "
            f"intermediate={intermediate_size}, gated={gated}, "
            f"activation={activation}"
        )

    def set_weights(
        self,
        gate_proj: Optional[Any] = None,  # MTLBuffer [hidden, intermediate]
        up_proj: Optional[Any] = None,    # MTLBuffer [hidden, intermediate]
        down_proj: Optional[Any] = None,  # MTLBuffer [intermediate, hidden]
        gate_bias: Optional[Any] = None,
        up_bias: Optional[Any] = None,
        down_bias: Optional[Any] = None,
    ) -> None:
        """Set weight buffers.

        For gated MLP: provide gate_proj, up_proj, down_proj
        For standard MLP: provide up_proj, down_proj (gate_proj ignored)

        Args:
            gate_proj: Gate projection weights (gated MLP only)
            up_proj: Up projection weights
            down_proj: Down projection weights
            gate_bias: Gate projection bias
            up_bias: Up projection bias
            down_bias: Down projection bias
        """
        self._gate_proj = gate_proj
        self._up_proj = up_proj
        self._down_proj = down_proj
        self._gate_bias = gate_bias
        self._up_bias = up_bias
        self._down_bias = down_bias

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        hidden_states: Union[EngineTensor, Any],  # [num_tokens, hidden_size]
        output_buffer: Any,  # MTLBuffer for output [num_tokens, hidden_size]
        num_tokens: int,
        intermediate_buffer: Optional[Any] = None,  # Scratch for intermediate
    ) -> None:
        """Encode MLP to command buffer.

        Args:
            step_ctx: EngineStepContext with encoder
            hidden_states: Input [num_tokens, hidden_size]
            output_buffer: Output buffer [num_tokens, hidden_size]
            num_tokens: Number of tokens
            intermediate_buffer: Optional scratch buffer for intermediate activations.
                                If None, allocates from scratch pool.
        """
        if self._up_proj is None or self._down_proj is None:
            raise RuntimeError("Weights not set - call set_weights() first")

        if self.config.gated and self._gate_proj is None:
            raise RuntimeError("Gate projection weights not set for gated MLP")

        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # Allocate intermediate buffer if needed
        intermediate_size_bytes = (
            num_tokens * self.config.intermediate_size * 2  # float16
        )

        if self.config.gated:
            # Need two intermediate buffers: gate and up
            gate_intermediate = step_ctx.allocate_scratch(intermediate_size_bytes)
            up_intermediate = step_ctx.allocate_scratch(intermediate_size_bytes)
            fused_intermediate = step_ctx.allocate_scratch(intermediate_size_bytes)
        else:
            up_intermediate = step_ctx.allocate_scratch(intermediate_size_bytes)

        if self.config.gated:
            # Gated MLP: silu(gate_proj(x)) * up_proj(x) -> down_proj
            self._encode_gated_mlp(
                step_ctx=step_ctx,
                hidden_states=hidden_states,
                output_buffer=output_buffer,
                num_tokens=num_tokens,
                gate_intermediate=gate_intermediate,
                up_intermediate=up_intermediate,
                fused_intermediate=fused_intermediate,
            )
        else:
            # Standard MLP: activation(up_proj(x)) -> down_proj
            self._encode_standard_mlp(
                step_ctx=step_ctx,
                hidden_states=hidden_states,
                output_buffer=output_buffer,
                num_tokens=num_tokens,
                intermediate=up_intermediate,
            )

    def _encode_gated_mlp(
        self,
        step_ctx: Any,
        hidden_states: Union[EngineTensor, Any],
        output_buffer: Any,
        num_tokens: int,
        gate_intermediate: Any,
        up_intermediate: Any,
        fused_intermediate: Any,
    ) -> None:
        """Encode gated MLP.

        Computes: down_proj(silu(gate_proj(x)) * up_proj(x))
        """
        H = self.config.hidden_size
        I = self.config.intermediate_size

        # Step 1: gate_proj(x) -> gate_intermediate
        # [num_tokens, H] @ [H, I] -> [num_tokens, I]
        self._gemm.encode(
            step_ctx=step_ctx,
            A=hidden_states,
            B=self._gate_proj,
            C=gate_intermediate,
            M=num_tokens,
            K=H,
            N=I,
        )

        # Step 2: up_proj(x) -> up_intermediate
        # [num_tokens, H] @ [H, I] -> [num_tokens, I]
        self._gemm.encode(
            step_ctx=step_ctx,
            A=hidden_states,
            B=self._up_proj,
            C=up_intermediate,
            M=num_tokens,
            K=H,
            N=I,
        )

        # Memory barrier after GEMMs
        step_ctx.memory_barrier()

        # Step 3: silu(gate) * up -> fused_intermediate
        num_intermediate_elements = num_tokens * I
        self._elementwise.encode_silu_mul(
            step_ctx=step_ctx,
            gate=gate_intermediate,
            up=up_intermediate,
            output=fused_intermediate,
            num_elements=num_intermediate_elements,
        )

        # Memory barrier after elementwise
        step_ctx.memory_barrier()

        # Step 4: down_proj(fused_intermediate) -> output
        # [num_tokens, I] @ [I, H] -> [num_tokens, H]
        self._gemm.encode(
            step_ctx=step_ctx,
            A=fused_intermediate,
            B=self._down_proj,
            C=output_buffer,
            M=num_tokens,
            K=I,
            N=H,
        )

    def _encode_standard_mlp(
        self,
        step_ctx: Any,
        hidden_states: Union[EngineTensor, Any],
        output_buffer: Any,
        num_tokens: int,
        intermediate: Any,
    ) -> None:
        """Encode standard (non-gated) MLP.

        Computes: down_proj(activation(up_proj(x)))
        """
        H = self.config.hidden_size
        I = self.config.intermediate_size

        # Step 1: up_proj(x) -> intermediate
        self._gemm.encode(
            step_ctx=step_ctx,
            A=hidden_states,
            B=self._up_proj,
            C=intermediate,
            M=num_tokens,
            K=H,
            N=I,
        )

        # Memory barrier
        step_ctx.memory_barrier()

        # Step 2: activation(intermediate) in-place
        num_intermediate_elements = num_tokens * I
        if self.config.activation == "silu":
            self._elementwise.encode_silu(
                step_ctx=step_ctx,
                input_tensor=intermediate,
                output=intermediate,  # in-place
                num_elements=num_intermediate_elements,
                inplace=True,
            )
        else:  # gelu
            # GELU is not in-place, need another buffer
            activated = step_ctx.allocate_scratch(num_tokens * I * 2)
            self._elementwise.encode_gelu(
                step_ctx=step_ctx,
                input_tensor=intermediate,
                output=activated,
                num_elements=num_intermediate_elements,
            )
            intermediate = activated

        # Memory barrier
        step_ctx.memory_barrier()

        # Step 3: down_proj(intermediate) -> output
        self._gemm.encode(
            step_ctx=step_ctx,
            A=intermediate,
            B=self._down_proj,
            C=output_buffer,
            M=num_tokens,
            K=I,
            N=H,
        )

    def get_intermediate_size(self, num_tokens: int) -> int:
        """Get required intermediate buffer size in bytes.

        Args:
            num_tokens: Number of tokens

        Returns:
            Size in bytes for all intermediate buffers needed
        """
        element_size = 2  # float16
        per_intermediate = num_tokens * self.config.intermediate_size * element_size

        if self.config.gated:
            # Need 3 intermediate buffers: gate, up, fused
            return per_intermediate * 3
        else:
            # Need 1 intermediate buffer (possibly 2 for GELU)
            if self.config.activation == "gelu":
                return per_intermediate * 2
            return per_intermediate

    def get_output_size(self, num_tokens: int) -> int:
        """Get required output buffer size in bytes."""
        element_size = 2  # float16
        return num_tokens * self.config.hidden_size * element_size

    def get_params_dict(self) -> Dict[str, Any]:
        """Get operation parameters as dict."""
        return {
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "gated": self.config.gated,
            "activation": self.config.activation,
            "bias": self.config.bias,
        }

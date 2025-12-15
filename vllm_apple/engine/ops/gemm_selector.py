# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified GEMM Selector for vLLM-Apple Metal Engine v2.0.

This module provides a unified GEMM interface that selects between:
- MPSMatrixMultiplication (current default)
- Custom Metal kernels (performance optimized)

The backend selection is controlled by environment variables:
- VLLM_GEMM_BACKEND=auto  (default: uses heuristics to choose)
- VLLM_GEMM_BACKEND=mps   (force MPS backend)
- VLLM_GEMM_BACKEND=metal (force custom Metal backend)

Usage:
    from vllm_apple.engine.ops.gemm_selector import UnifiedGEMM

    # Create unified op
    gemm = UnifiedGEMM(context, backend="auto")

    # Encode GEMM (dispatches to appropriate backend)
    gemm.encode(step_ctx, A, B, C, M, K, N)
"""

import os
from typing import Any, Dict, Optional, Union

from vllm.logger import init_logger
from ..tensor import EngineTensor

logger = init_logger(__name__)

# Environment variable for backend selection
GEMM_BACKEND_ENV = "VLLM_GEMM_BACKEND"


def get_gemm_backend() -> str:
    """Get GEMM backend from environment.

    Returns:
        One of: "auto", "mps", "metal"
    """
    backend = os.environ.get(GEMM_BACKEND_ENV, "auto").lower()
    if backend not in ("auto", "mps", "metal"):
        logger.warning(f"Unknown GEMM backend '{backend}', using 'auto'")
        return "auto"
    return backend


class UnifiedGEMM:
    """Unified GEMM interface with backend selection.

    Provides a single API that can dispatch to either MPS or custom Metal
    GEMM implementations. The backend can be selected via environment
    variable or constructor parameter.

    Auto mode uses heuristics to choose the best backend:
    - Metal for small-M (decode) where custom kernel excels
    - MPS for large matrices where it's well-optimized
    - Metal when VLLM_GEMM_BACKEND=metal is set

    Attributes:
        context: MetalEngineContext
        backend: Selected backend ("auto", "mps", "metal")
    """

    def __init__(
        self,
        context: Any,  # MetalEngineContext
        backend: Optional[str] = None,
    ):
        """Initialize unified GEMM.

        Args:
            context: MetalEngineContext
            backend: Backend to use ("auto", "mps", "metal") or None for env default
        """
        self._context = context

        # Determine backend
        self._backend = backend or get_gemm_backend()

        # Lazy initialization of backend ops
        self._mps_gemm = None
        self._metal_gemm = None

        # Initialize based on backend
        if self._backend in ("auto", "mps"):
            self._init_mps()
        if self._backend in ("auto", "metal"):
            self._init_metal()

        logger.info(f"UnifiedGEMM initialized with backend: {self._backend}")

    def _init_mps(self) -> None:
        """Initialize MPS GEMM backend."""
        if self._mps_gemm is None:
            from .gemm import EngineGEMM
            self._mps_gemm = EngineGEMM(self._context)
            logger.debug("MPS GEMM backend initialized")

    def _init_metal(self) -> None:
        """Initialize custom Metal GEMM backend."""
        if self._metal_gemm is None:
            try:
                from .gemm_metal import EngineGEMMMetal
                self._metal_gemm = EngineGEMMMetal(self._context)
                logger.debug("Custom Metal GEMM backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Metal GEMM: {e}")
                self._metal_gemm = None

    def _should_use_metal(
        self,
        step_ctx: Any,
        M: int,
        K: int,
        N: int,
    ) -> bool:
        """Determine if Metal backend should be used.

        Heuristics for auto mode:
        - Use Metal for batch=1 decode (avoids MPS encoder transitions)
        - Use Metal for small M (decode phase)
        - Use Metal for very large N (LM head)
        - Use MPS as default (well-optimized by Apple)

        Args:
            step_ctx: EngineStepContext (may have decode_single_seq flag)
            M: Rows of output
            K: Inner dimension
            N: Columns of output

        Returns:
            True if Metal backend should be used
        """
        if self._backend == "metal":
            return True
        if self._backend == "mps":
            return False

        # === BATCH=1 DECODE OPTIMIZATION ===
        # When decode_single_seq is True, always use native Metal to avoid
        # MPS encoder transitions (end/reopen cycle). This is the primary
        # optimization for batch=1 decode latency.
        if hasattr(step_ctx, 'decode_single_seq') and step_ctx.decode_single_seq:
            return True

        # Auto mode heuristics
        # Small-M decode: Metal kernel is optimized for this
        if M <= 8:
            return True

        # Very large N (LM head with big vocab): Metal can stream better
        if N > 65536:
            return True

        # Default to MPS for general cases (well-optimized)
        return False

    def encode(
        self,
        step_ctx: Any,  # EngineStepContext
        A: Union[EngineTensor, Any],  # MTLBuffer
        B: Union[EngineTensor, Any],  # MTLBuffer
        C: Union[EngineTensor, Any],  # MTLBuffer
        M: Optional[int] = None,
        K: Optional[int] = None,
        N: Optional[int] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        transpose_A: bool = False,
        transpose_B: bool = True,  # Default matches weight storage
    ) -> None:
        """Encode GEMM to command buffer.

        Dispatches to appropriate backend based on configuration.

        Args:
            step_ctx: EngineStepContext with encoder
            A: Left matrix [M, K]
            B: Right matrix [N, K] if transpose_B else [K, N]
            C: Result matrix [M, N]
            M: Rows of A (inferred from tensor if not provided)
            K: Cols of A
            N: Cols of C
            alpha: Scaling for A @ B (only for MPS)
            beta: Scaling for C (only for MPS)
            transpose_A: Transpose A (only for MPS)
            transpose_B: Transpose B
        """
        # Infer dimensions
        if isinstance(A, EngineTensor):
            M = M or A.shape[0]
            K = K or A.shape[1]
        if isinstance(B, EngineTensor):
            if transpose_B:
                N = N or B.shape[0]
            else:
                N = N or B.shape[1]

        # Choose backend
        use_metal = (
            self._metal_gemm is not None and
            self._should_use_metal(step_ctx, M, K, N) and
            alpha == 1.0 and beta == 0.0 and not transpose_A
        )

        if use_metal:
            self._metal_gemm.encode(
                step_ctx=step_ctx,
                A=A,
                B=B,
                C=C,
                M=M,
                K=K,
                N=N,
                transpose_B=transpose_B,
            )
        else:
            # Fall back to MPS
            if self._mps_gemm is None:
                self._init_mps()
            self._mps_gemm.encode(
                step_ctx=step_ctx,
                A=A,
                B=B,
                C=C,
                M=M,
                K=K,
                N=N,
                alpha=alpha,
                beta=beta,
                transpose_A=transpose_A,
                transpose_B=transpose_B,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both backends."""
        stats = {
            "backend": self._backend,
        }
        if self._mps_gemm:
            stats["mps"] = self._mps_gemm.get_stats()
        if self._metal_gemm:
            stats["metal"] = self._metal_gemm.get_stats()
        return stats


def create_gemm(context: Any, backend: Optional[str] = None) -> UnifiedGEMM:
    """Factory function to create GEMM operation.

    Args:
        context: MetalEngineContext
        backend: Backend selection or None for auto

    Returns:
        UnifiedGEMM instance
    """
    return UnifiedGEMM(context, backend=backend)

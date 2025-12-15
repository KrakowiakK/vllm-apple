# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GEMM Operation for vLLM-Apple Metal Engine v2.0.

This module provides encode-only GEMM operations using MPSMatrixMultiplication.
The operation encodes matrix multiplications to a command buffer WITHOUT
executing. Execution happens at step boundary.

Key Principle: Encode-only API. No internal waits.

Uses Apple's MPSMatrixMultiplication which has an encode-only API:
    encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:

This allows GEMM to be encoded without synchronization, maintaining
the v2.0 "no waits in hot path" invariant.

IMPORTANT ENCODING CONTRACT:
    MPS operations (like MPSMatrixMultiplication) encode directly to a MTLCommandBuffer,
    NOT to a MTLComputeEncoder. This means:
    1. Any open compute encoder MUST be ended before MPS encoding
    2. After MPS encoding, a new compute encoder must be created for further compute ops
    3. The step_ctx.end_compute_encoder_for_mps() method handles this transition

    This is an encode-only exception allowed under v2.0 invariants because:
    - MPS encodeToCommandBuffer is a pure encode operation (no waits)
    - No hidden synchronization or GPU execution occurs

DTYPE CONTRACT:
    - MPS backend: Supports FLOAT16 and FLOAT32
    - Custom Metal backend: FLOAT16 only (hard fail on other dtypes)
    - EngineTensors validate dtype at creation time

Usage:
    from vllm_apple.engine.ops.gemm import EngineGEMM

    # Create op during initialization
    gemm = EngineGEMM(context)

    # Encode GEMM (no wait) - C = alpha * A @ B + beta * C
    gemm.encode(
        step_ctx=step_ctx,
        A=input_tensor,   # [M, K]
        B=weight_tensor,  # [K, N]
        C=output_tensor,  # [M, N]
        alpha=1.0,
        beta=0.0,
    )
"""

from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from vllm.logger import init_logger
from ..tensor import EngineTensor, EngineDType

logger = init_logger(__name__)

# Try to import Metal Performance Shaders
try:
    import Metal
    from Metal import (
        MTLResourceStorageModeShared,
        MTLOrigin,
    )
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    Metal = None

try:
    import MetalPerformanceShaders as MPS
    HAS_MPS = True
except ImportError:
    HAS_MPS = False
    MPS = None


@dataclass
class GEMMConfig:
    """Configuration for GEMM operation."""
    M: int  # Rows of A, rows of C
    K: int  # Cols of A, rows of B
    N: int  # Cols of B, cols of C
    alpha: float = 1.0
    beta: float = 0.0
    transpose_A: bool = False
    transpose_B: bool = False


class MPSMatrixWrapper:
    """Wrapper to create MPSMatrix from EngineTensor or MTLBuffer."""

    def __init__(self, context: Any):
        """Initialize wrapper.

        Args:
            context: MetalEngineContext
        """
        self._context = context
        self._device = context.device

    def create_matrix(
        self,
        buffer: Any,  # MTLBuffer
        rows: int,
        columns: int,
        row_bytes: int,
        dtype: EngineDType = EngineDType.FLOAT16,
        offset: int = 0,
    ) -> Any:
        """Create MPSMatrix from buffer.

        Args:
            buffer: MTLBuffer containing matrix data
            rows: Number of rows
            columns: Number of columns
            row_bytes: Bytes per row (must be multiple of 16 for Metal)
            dtype: Data type
            offset: Byte offset into buffer

        Returns:
            MPSMatrix
        """
        if not HAS_MPS:
            raise RuntimeError("MetalPerformanceShaders not available")

        # Map dtype to MPS type
        if dtype == EngineDType.FLOAT16:
            mps_dtype = MPS.MPSDataTypeFloat16
        elif dtype == EngineDType.FLOAT32:
            mps_dtype = MPS.MPSDataTypeFloat32
        else:
            raise ValueError(f"Unsupported dtype for GEMM: {dtype}")

        # Create matrix descriptor
        desc = MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            rows, columns, row_bytes, mps_dtype
        )

        # Create matrix
        matrix = MPS.MPSMatrix.alloc().initWithBuffer_offset_descriptor_(
            buffer, offset, desc
        )

        return matrix

    def create_from_tensor(self, tensor: EngineTensor) -> Any:
        """Create MPSMatrix from EngineTensor.

        Args:
            tensor: 2D EngineTensor

        Returns:
            MPSMatrix
        """
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {tensor.ndim}D")

        rows, cols = tensor.shape

        # Compute row bytes (ensure 16-byte alignment for Metal)
        row_bytes = cols * tensor.itemsize
        if row_bytes % 16 != 0:
            # Row alignment is required for correctness with MPS
            # This should only fail for malformed tensors or edge cases
            # Production weights should always be pre-aligned
            import os
            if os.environ.get("VLLM_METAL_STRICT_NO_MPS", "0") == "1":
                raise ValueError(
                    f"Row bytes {row_bytes} not 16-byte aligned. "
                    f"Metal requires 16-byte row alignment for correctness. "
                    f"Tensor shape: {tensor.shape}, itemsize: {tensor.itemsize}"
                )
            # Non-strict mode: allow with warning, MPS may handle padding internally
            logger.warning(
                f"Row bytes {row_bytes} not 16-byte aligned. "
                f"This may cause incorrect results. "
                f"Enable VLLM_METAL_STRICT_NO_MPS=1 to fail on alignment issues."
            )

        return self.create_matrix(
            buffer=tensor.buffer,
            rows=rows,
            columns=cols,
            row_bytes=row_bytes,
            dtype=tensor.dtype,
            offset=tensor.offset,
        )


class EngineGEMM:
    """Encode-only GEMM operation using MPSMatrixMultiplication or custom Metal.

    This op encodes GEMM operations to a command buffer WITHOUT executing.
    By default uses Apple's MPSMatrixMultiplication which provides an encode-only API.
    When VLLM_GEMM_BACKEND=metal, uses custom Metal kernels optimized for decode.

    Supports:
    - C = alpha * A @ B + beta * C
    - Transposed variants
    - Float16 and Float32

    Attributes:
        context: MetalEngineContext
    """

    def __init__(self, context: Any):  # MetalEngineContext
        """Initialize GEMM op.

        Args:
            context: MetalEngineContext
        """
        import os
        self._context = context
        self._device = context.device

        # Check GEMM backend setting
        backend = os.environ.get("VLLM_GEMM_BACKEND", "auto").lower()
        self._backend = backend  # Store for later use
        self._use_metal_backend = backend == "metal"
        self._metal_gemm = None

        # Initialize Metal backend if requested OR in auto mode (for decode_single_seq)
        if backend in ("metal", "auto"):
            try:
                from .gemm_metal import EngineGEMMMetal
                self._metal_gemm = EngineGEMMMetal(context)
                if backend == "metal":
                    logger.info("EngineGEMM initialized with custom Metal backend (forced)")
                else:
                    logger.debug("EngineGEMM: Metal backend available for auto selection")
            except Exception as e:
                logger.warning(f"Failed to initialize Metal GEMM, falling back to MPS: {e}")
                self._use_metal_backend = False
                self._metal_gemm = None

        # Always initialize MPS wrapper unless Metal is forced
        if backend != "metal":
            if not HAS_MPS:
                raise RuntimeError(
                    "MetalPerformanceShaders not available. "
                    "Install with: pip install pyobjc-framework-MetalPerformanceShaders"
                )
            self._matrix_wrapper = MPSMatrixWrapper(context)
            # Cache for MPSMatrixMultiplication objects
            # Key: (transpose_A, transpose_B, M, K, N)
            self._mm_cache: Dict[Tuple, Any] = {}
            logger.info("EngineGEMM initialized with MPS backend")

    def _validate_dtype(self, tensor: Union[EngineTensor, Any], name: str) -> None:
        """Validate tensor dtype.

        MPS backend supports FLOAT16 and FLOAT32.
        Metal backend (EngineGEMMMetal) validates FP16-only internally.

        Args:
            tensor: EngineTensor to validate
            name: Name for error message

        Raises:
            ValueError: If dtype is unsupported
        """
        if isinstance(tensor, EngineTensor):
            if tensor.dtype not in (EngineDType.FLOAT16, EngineDType.FLOAT32):
                raise ValueError(
                    f"EngineGEMM requires FLOAT16 or FLOAT32 tensors. "
                    f"Tensor '{name}' has dtype {tensor.dtype}."
                )
        # For raw MTLBuffers, caller must ensure correct dtype

    def _get_mm_kernel(
        self,
        transpose_A: bool,
        transpose_B: bool,
        M: int,
        K: int,
        N: int,
        alpha: float,
        beta: float,
    ) -> Any:
        """Get or create MPSMatrixMultiplication kernel.

        Args:
            transpose_A: Whether to transpose A
            transpose_B: Whether to transpose B
            M: Rows of result
            K: Inner dimension
            N: Columns of result
            alpha: Scaling for A @ B
            beta: Scaling for C

        Returns:
            MPSMatrixMultiplication
        """
        # IMPORTANT: alpha and beta are baked into MPSMatrixMultiplication at init time.
        # They MUST be part of the cache key to avoid wrong results.
        cache_key = (transpose_A, transpose_B, M, K, N, alpha, beta)

        if cache_key not in self._mm_cache:
            mm = MPS.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
                self._device,
                transpose_A,
                transpose_B,
                M,
                N,
                K,
                alpha,
                beta,
            )
            self._mm_cache[cache_key] = mm
            logger.debug(
                f"Created MPSMatrixMultiplication: M={M}, K={K}, N={N}, "
                f"alpha={alpha}, beta={beta}"
            )

        return self._mm_cache[cache_key]

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
        transpose_B: bool = False,
    ) -> None:
        """Encode GEMM to command buffer.

        Computes: C = alpha * A @ B + beta * C

        If A, B, C are EngineTensors, dimensions are inferred from shape.
        If they are raw MTLBuffers, M, K, N must be provided.

        Args:
            step_ctx: EngineStepContext with encoder
            A: Left matrix [M, K] (or [K, M] if transpose_A)
            B: Right matrix [K, N] (or [N, K] if transpose_B)
            C: Result matrix [M, N]
            M: Rows of A (and C)
            K: Cols of A (and rows of B)
            N: Cols of B (and C)
            alpha: Scaling factor for A @ B
            beta: Scaling factor for C
            transpose_A: Whether to transpose A
            transpose_B: Whether to transpose B
        """
        if not step_ctx.is_encoding:
            raise RuntimeError("encode() called outside ENCODE phase")

        # Validate dtype for EngineTensors
        self._validate_dtype(A, "A")
        self._validate_dtype(B, "B")
        self._validate_dtype(C, "C")

        # === BATCH=1 DECODE OPTIMIZATION ===
        # Check if we should use Metal backend for this encode:
        # 1. Metal backend is available
        # 2. Either forced (backend="metal") OR step_ctx.decode_single_seq is True
        # 3. Operation is supported by Metal (alpha=1, beta=0, no transpose_A)
        use_metal_for_this_call = (
            self._metal_gemm is not None and
            (self._use_metal_backend or
             (hasattr(step_ctx, 'decode_single_seq') and step_ctx.decode_single_seq)) and
            alpha == 1.0 and beta == 0.0 and not transpose_A
        )

        if use_metal_for_this_call:
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
            return

        # Fall through to MPS
        # Lazy init MPS wrapper if needed (when Metal was forced but op not supported)
        if not hasattr(self, '_matrix_wrapper'):
            self._matrix_wrapper = MPSMatrixWrapper(self._context)
            self._mm_cache = {}

        # Convert tensors to MPSMatrix
        if isinstance(A, EngineTensor):
            A_matrix = self._matrix_wrapper.create_from_tensor(A)
            A_shape = A.shape
        else:
            if M is None or K is None:
                raise ValueError("M, K must be provided for raw MTLBuffer A")
            A_shape = (M, K) if not transpose_A else (K, M)
            row_bytes = A_shape[1] * 2  # Assume float16
            A_matrix = self._matrix_wrapper.create_matrix(
                A, A_shape[0], A_shape[1], row_bytes
            )

        if isinstance(B, EngineTensor):
            B_matrix = self._matrix_wrapper.create_from_tensor(B)
            B_shape = B.shape
        else:
            if K is None or N is None:
                raise ValueError("K, N must be provided for raw MTLBuffer B")
            B_shape = (K, N) if not transpose_B else (N, K)
            row_bytes = B_shape[1] * 2
            B_matrix = self._matrix_wrapper.create_matrix(
                B, B_shape[0], B_shape[1], row_bytes
            )

        if isinstance(C, EngineTensor):
            C_matrix = self._matrix_wrapper.create_from_tensor(C)
            C_shape = C.shape
        else:
            if M is None or N is None:
                raise ValueError("M, N must be provided for raw MTLBuffer C")
            C_shape = (M, N)
            row_bytes = C_shape[1] * 2
            C_matrix = self._matrix_wrapper.create_matrix(
                C, C_shape[0], C_shape[1], row_bytes
            )

        # Infer dimensions
        if transpose_A:
            inferred_K, inferred_M = A_shape
        else:
            inferred_M, inferred_K = A_shape

        if transpose_B:
            inferred_N, inferred_K2 = B_shape
        else:
            inferred_K2, inferred_N = B_shape

        # Use provided or inferred dimensions
        M = M or inferred_M
        K = K or inferred_K
        N = N or inferred_N

        # Validate dimensions
        if inferred_K != inferred_K2:
            raise ValueError(f"Inner dimensions don't match: A has {inferred_K}, B has {inferred_K2}")

        # Get kernel
        mm = self._get_mm_kernel(transpose_A, transpose_B, M, K, N, alpha, beta)

        # MPS operations encode directly to command buffer, not to a compute encoder.
        # We must end any open compute encoder before MPS encoding.
        cmd_buffer = step_ctx.end_compute_encoder_for_mps()

        # Encode to command buffer
        # MPSMatrixMultiplication.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_
        mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
            cmd_buffer,
            A_matrix,
            B_matrix,
            C_matrix,
        )

    def encode_batched(
        self,
        step_ctx: Any,  # EngineStepContext
        A: EngineTensor,  # [batch, M, K]
        B: EngineTensor,  # [K, N] (broadcasted) or [batch, K, N]
        C: EngineTensor,  # [batch, M, N]
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        """Encode batched GEMM.

        For batched GEMM, we either:
        1. Use MPSMatrixMultiplication with batched matrices
        2. Encode multiple single-batch GEMMs

        Args:
            step_ctx: EngineStepContext
            A: Left matrices [batch, M, K]
            B: Right matrices [K, N] or [batch, K, N]
            C: Result matrices [batch, M, N]
            alpha: Scaling factor
            beta: Scaling factor
        """
        if A.ndim != 3 or C.ndim != 3:
            raise ValueError("Batched GEMM requires 3D A and C tensors")

        batch_size = A.shape[0]
        M = A.shape[1]
        K = A.shape[2]
        N = C.shape[2]

        # WARNING: Batched GEMM is implemented as a Python loop.
        # This is a temporary fallback and NOT suitable for hot paths.
        # TODO: Implement native batched GEMM using MPSMatrixMultiplication batched API
        if batch_size > 8:
            logger.warning(
                f"encode_batched called with batch_size={batch_size}. "
                f"This uses a Python loop and may be slow. "
                f"Consider restructuring to avoid batched GEMM in hot paths."
            )

        for i in range(batch_size):
            A_slice = A[i]
            C_slice = C[i]
            B_slice = B[i] if B.ndim == 3 else B

            self.encode(
                step_ctx=step_ctx,
                A=A_slice,
                B=B_slice,
                C=C_slice,
                alpha=alpha,
                beta=beta,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        return {
            "cached_kernels": len(self._mm_cache),
        }

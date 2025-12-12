# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine Tensor for vLLM-Apple Metal Engine v2.0.

This module provides EngineTensor, a lightweight wrapper around MTLBuffer
that tracks shape, dtype, and strides for use with Metal operations.

Key Features:
- Zero-copy view/slice operations
- Compatible with MPSMatrix for GEMM
- Conversion to/from numpy and torch.Tensor
- Efficient buffer reuse

Unlike PyTorch tensors, EngineTensor does NOT support automatic
differentiation or device management - it's a thin wrapper around
raw Metal buffers for engine operations.

Usage:
    from vllm_apple.engine.tensor import EngineTensor, EngineDType

    # Create from numpy
    tensor = EngineTensor.from_numpy(context, np_array)

    # Create view (zero-copy)
    view = tensor.view(new_shape)

    # Slice (zero-copy)
    slice_tensor = tensor[0:10]

    # Convert to numpy (copies data)
    result = tensor.to_numpy()

    # Get MTLBuffer for kernel
    buffer = tensor.buffer
"""

import numpy as np
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

# Try to import Metal
try:
    from Metal import MTLResourceStorageModeShared
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    MTLResourceStorageModeShared = None


class EngineDType(Enum):
    """Data types supported by EngineTensor."""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    INT32 = "int32"
    INT64 = "int64"

    @property
    def numpy_dtype(self):
        """Get numpy dtype."""
        return {
            EngineDType.FLOAT16: np.float16,
            EngineDType.FLOAT32: np.float32,
            EngineDType.INT32: np.int32,
            EngineDType.INT64: np.int64,
        }[self]

    @property
    def itemsize(self) -> int:
        """Get bytes per element."""
        return {
            EngineDType.FLOAT16: 2,
            EngineDType.FLOAT32: 4,
            EngineDType.INT32: 4,
            EngineDType.INT64: 8,
        }[self]

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> "EngineDType":
        """Convert numpy dtype to EngineDType."""
        dtype = np.dtype(dtype)
        mapping = {
            np.dtype(np.float16): cls.FLOAT16,
            np.dtype(np.float32): cls.FLOAT32,
            np.dtype(np.int32): cls.INT32,
            np.dtype(np.int64): cls.INT64,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]


@dataclass
class TensorMetadata:
    """Metadata for EngineTensor."""
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]  # In elements, not bytes
    dtype: EngineDType
    offset: int = 0  # Byte offset into buffer


class EngineTensor:
    """Lightweight tensor wrapper around MTLBuffer.

    This class provides shape/dtype/stride tracking for Metal buffers.
    It supports zero-copy views and slices, and conversion to/from numpy.

    Important: EngineTensor does NOT own the buffer - the caller must
    ensure the buffer outlives the tensor. For owned buffers, use
    EngineTensor.allocate() which stores a reference.

    Attributes:
        buffer: Underlying MTLBuffer
        shape: Tensor shape
        strides: Strides in elements
        dtype: Data type
        offset: Byte offset into buffer
    """

    def __init__(
        self,
        buffer: Any,  # MTLBuffer
        shape: Tuple[int, ...],
        dtype: EngineDType,
        strides: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
        _owns_buffer: bool = False,
    ):
        """Initialize EngineTensor.

        Args:
            buffer: MTLBuffer containing data
            shape: Tensor shape
            dtype: Data type
            strides: Strides in elements (default: contiguous)
            offset: Byte offset into buffer
            _owns_buffer: Whether this tensor owns the buffer (internal)
        """
        self._buffer = buffer
        self._shape = tuple(shape)
        self._dtype = dtype
        self._offset = offset
        self._owns_buffer = _owns_buffer

        # Compute strides if not provided (contiguous layout)
        if strides is None:
            strides = self._compute_contiguous_strides(shape)
        self._strides = tuple(strides)

        # Validate
        self._validate()

    def _compute_contiguous_strides(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute strides for contiguous layout."""
        if len(shape) == 0:
            return ()
        strides = [1]
        for dim in reversed(shape[1:]):
            strides.append(strides[-1] * dim)
        return tuple(reversed(strides))

    def _validate(self) -> None:
        """Validate tensor configuration."""
        if len(self._shape) != len(self._strides):
            raise ValueError(f"Shape {self._shape} and strides {self._strides} must have same length")

        # Check buffer size
        if self._buffer is not None:
            required_bytes = self._offset + self.nbytes
            buffer_size = self._buffer.length()
            if required_bytes > buffer_size:
                raise ValueError(
                    f"Buffer too small: need {required_bytes} bytes, have {buffer_size}"
                )

    @property
    def buffer(self) -> Any:
        """Get underlying MTLBuffer."""
        return self._buffer

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self._shape

    @property
    def strides(self) -> Tuple[int, ...]:
        """Get strides in elements."""
        return self._strides

    @property
    def dtype(self) -> EngineDType:
        """Get data type."""
        return self._dtype

    @property
    def offset(self) -> int:
        """Get byte offset into buffer."""
        return self._offset

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        return self._dtype.itemsize

    @property
    def nbytes(self) -> int:
        """Total bytes (for contiguous tensor)."""
        return self.numel * self.itemsize

    @property
    def is_contiguous(self) -> bool:
        """Check if tensor is contiguous in memory."""
        expected_strides = self._compute_contiguous_strides(self._shape)
        return self._strides == expected_strides

    def view(self, *shape: int) -> "EngineTensor":
        """Create a view with new shape (zero-copy).

        The tensor must be contiguous. Total elements must match.

        Args:
            *shape: New shape dimensions

        Returns:
            EngineTensor view with new shape
        """
        if not self.is_contiguous:
            raise ValueError("Cannot view non-contiguous tensor")

        new_numel = 1
        for dim in shape:
            new_numel *= dim

        if new_numel != self.numel:
            raise ValueError(f"Cannot view {self.numel} elements as shape {shape}")

        return EngineTensor(
            buffer=self._buffer,
            shape=shape,
            dtype=self._dtype,
            strides=None,  # Will compute contiguous strides
            offset=self._offset,
        )

    def reshape(self, *shape: int) -> "EngineTensor":
        """Reshape tensor (zero-copy if contiguous).

        Args:
            *shape: New shape dimensions

        Returns:
            EngineTensor with new shape
        """
        return self.view(*shape)

    def squeeze(self, dim: Optional[int] = None) -> "EngineTensor":
        """Remove dimensions of size 1.

        Args:
            dim: Specific dimension to squeeze, or None for all

        Returns:
            EngineTensor with squeezed shape
        """
        if dim is not None:
            if self._shape[dim] != 1:
                return self
            new_shape = list(self._shape)
            new_strides = list(self._strides)
            del new_shape[dim]
            del new_strides[dim]
        else:
            new_shape = []
            new_strides = []
            for s, st in zip(self._shape, self._strides):
                if s != 1:
                    new_shape.append(s)
                    new_strides.append(st)

        return EngineTensor(
            buffer=self._buffer,
            shape=tuple(new_shape) or (1,),
            dtype=self._dtype,
            strides=tuple(new_strides) or (1,),
            offset=self._offset,
        )

    def unsqueeze(self, dim: int) -> "EngineTensor":
        """Add dimension of size 1.

        Args:
            dim: Position for new dimension

        Returns:
            EngineTensor with added dimension
        """
        new_shape = list(self._shape)
        new_strides = list(self._strides)

        # Normalize negative dim
        if dim < 0:
            dim = len(new_shape) + dim + 1

        # Insert stride of 1 (doesn't affect memory layout)
        new_shape.insert(dim, 1)
        new_strides.insert(dim, self._strides[dim] if dim < len(self._strides) else 1)

        return EngineTensor(
            buffer=self._buffer,
            shape=tuple(new_shape),
            dtype=self._dtype,
            strides=tuple(new_strides),
            offset=self._offset,
        )

    def __getitem__(self, key) -> "EngineTensor":
        """Slice tensor (zero-copy when possible).

        Supports integer indexing and slices for first dimension.

        Args:
            key: Index or slice

        Returns:
            EngineTensor slice
        """
        if isinstance(key, int):
            # Single element - reduce dimension
            if self.ndim == 0:
                raise IndexError("Cannot index 0-dimensional tensor")
            if key < 0:
                key = self._shape[0] + key
            if key < 0 or key >= self._shape[0]:
                raise IndexError(f"Index {key} out of range [0, {self._shape[0]})")

            new_offset = self._offset + key * self._strides[0] * self.itemsize
            return EngineTensor(
                buffer=self._buffer,
                shape=self._shape[1:] or (1,),
                dtype=self._dtype,
                strides=self._strides[1:] or (1,),
                offset=new_offset,
            )

        elif isinstance(key, slice):
            # Slice first dimension
            start, stop, step = key.indices(self._shape[0])
            if step != 1:
                raise ValueError("Non-unit step slicing not supported")

            new_size = stop - start
            new_offset = self._offset + start * self._strides[0] * self.itemsize
            new_shape = (new_size,) + self._shape[1:]

            return EngineTensor(
                buffer=self._buffer,
                shape=new_shape,
                dtype=self._dtype,
                strides=self._strides,
                offset=new_offset,
            )

        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (copies data).

        This operation reads from GPU memory, so it's only allowed during
        READBACK phase or IDLE (not during ENCODE or SUBMIT).

        Returns:
            numpy array with tensor data

        Raises:
            RuntimeError: If called during ENCODE or SUBMIT phase
        """
        if self._buffer is None:
            raise ValueError("No buffer to convert")

        # Phase guard: buffer readback only allowed in READBACK/IDLE phase
        from .guards import guarded_buffer_contents

        # Read from buffer (with phase guard)
        contents = guarded_buffer_contents(self._buffer, f"EngineTensor.to_numpy()")
        total_bytes = self._buffer.length()
        mv = contents.as_buffer(total_bytes)

        # Create numpy array from buffer
        data = np.frombuffer(
            bytes(mv[self._offset:self._offset + self.nbytes]),
            dtype=self._dtype.numpy_dtype,
        )

        # Reshape (handles strides for contiguous case)
        if self.is_contiguous:
            return data.reshape(self._shape).copy()
        else:
            # For non-contiguous, we need strided copy
            # This is a simple implementation; could be optimized
            result = np.zeros(self._shape, dtype=self._dtype.numpy_dtype)
            # TODO: Implement strided copy
            raise NotImplementedError("Non-contiguous to_numpy not implemented")

    @classmethod
    def from_numpy(
        cls,
        context: Any,  # MetalEngineContext
        array: np.ndarray,
        copy: bool = True,
    ) -> "EngineTensor":
        """Create EngineTensor from numpy array.

        Args:
            context: MetalEngineContext for buffer allocation
            array: numpy array
            copy: Whether to copy data (always True currently)

        Returns:
            EngineTensor with data
        """
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        dtype = EngineDType.from_numpy(array.dtype)

        # Create buffer with data
        buffer = context.create_buffer_from_bytes(array.tobytes())

        return cls(
            buffer=buffer,
            shape=array.shape,
            dtype=dtype,
            _owns_buffer=True,
        )

    @classmethod
    def allocate(
        cls,
        context: Any,  # MetalEngineContext
        shape: Tuple[int, ...],
        dtype: EngineDType = EngineDType.FLOAT16,
        zero_init: bool = True,
    ) -> "EngineTensor":
        """Allocate new EngineTensor.

        Args:
            context: MetalEngineContext for buffer allocation
            shape: Tensor shape
            dtype: Data type
            zero_init: Whether to zero-initialize (uses GPU-side fillBuffer)

        Returns:
            EngineTensor with allocated buffer
        """
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * dtype.itemsize

        buffer = context.create_buffer(nbytes, storage_mode="shared")

        if zero_init:
            # Zero the buffer using GPU-side fillBuffer (avoids allocating CPU memory)
            cls._zero_buffer_gpu(context, buffer, nbytes)

        return cls(
            buffer=buffer,
            shape=shape,
            dtype=dtype,
            _owns_buffer=True,
        )

    @staticmethod
    def _zero_buffer_gpu(context: Any, buffer: Any, nbytes: int) -> None:
        """Zero-initialize buffer using GPU-side fillBuffer.

        This avoids allocating nbytes of CPU memory just for zeroing.
        Uses MTLBlitCommandEncoder.fillBuffer which is efficient GPU-side zeroing.
        """
        from Metal import NSMakeRange

        # Create a one-shot command buffer for initialization
        cmd_queue = context.command_queue
        cmd_buffer = cmd_queue.commandBuffer()
        if cmd_buffer is None:
            raise RuntimeError("Failed to create command buffer for buffer zeroing")

        # Create blit encoder
        blit_encoder = cmd_buffer.blitCommandEncoder()
        if blit_encoder is None:
            raise RuntimeError("Failed to create blit encoder for buffer zeroing")

        # Fill with zeros using GPU-side operation
        fill_range = NSMakeRange(0, nbytes)
        blit_encoder.fillBuffer_range_value_(buffer, fill_range, 0)

        # End encoding and commit
        blit_encoder.endEncoding()
        cmd_buffer.commit()

        # Wait for completion - OK here since this is allocation, not hot path
        cmd_buffer.waitUntilCompleted()

    def __repr__(self) -> str:
        return f"EngineTensor(shape={self._shape}, dtype={self._dtype.value}, offset={self._offset})"


def create_engine_tensor_from_torch(
    context: Any,
    tensor: "torch.Tensor",
) -> EngineTensor:
    """Create EngineTensor from PyTorch tensor.

    The tensor MUST be on CPU. This copies data.

    Args:
        context: MetalEngineContext
        tensor: PyTorch tensor on CPU

    Returns:
        EngineTensor with data
    """
    import torch
    if tensor.device.type != "cpu":
        raise ValueError(f"Tensor must be on CPU, got {tensor.device}")

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    return EngineTensor.from_numpy(context, tensor.numpy())


def engine_tensor_to_torch(
    tensor: EngineTensor,
    device: str = "cpu",
) -> "torch.Tensor":
    """Convert EngineTensor to PyTorch tensor.

    Args:
        tensor: EngineTensor
        device: Target device (only "cpu" supported)

    Returns:
        PyTorch tensor with data
    """
    import torch
    if device != "cpu":
        raise ValueError(f"Only CPU device supported, got {device}")

    np_array = tensor.to_numpy()

    # Map dtype
    dtype_map = {
        EngineDType.FLOAT16: torch.float16,
        EngineDType.FLOAT32: torch.float32,
        EngineDType.INT32: torch.int32,
        EngineDType.INT64: torch.int64,
    }

    return torch.from_numpy(np_array).to(dtype_map[tensor.dtype])

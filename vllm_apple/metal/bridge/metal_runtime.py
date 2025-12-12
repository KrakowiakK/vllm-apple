"""Metal Runtime Bridge for PyTorch.

This module provides a Python interface to compile and execute Metal compute shaders
from PyTorch tensors on Apple Silicon.

Key components:
1. MetalKernelCompiler - Compiles .metal shaders to MTLLibrary
2. MetalKernelRunner - Executes kernels with PyTorch tensor inputs
3. Buffer management for PyTorch <-> Metal interop

Usage:
    compiler = MetalKernelCompiler()
    library = compiler.compile_shader("path/to/shader.metal")

    runner = MetalKernelRunner(library)
    output = runner.run_kernel(
        "paged_attention_decode",
        inputs=[query, key_cache, value_cache, block_table, seq_lens],
        outputs=[output_tensor],
        params=params_buffer,
        grid_size=(num_seqs, num_query_heads, 1),
        threadgroup_size=(32, 1, 1)
    )
"""

import os
import ctypes
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import torch

# Try to import Metal Python bindings
try:
    import Metal
    import Foundation
    HAS_METAL_PYTHON = True
except ImportError:
    HAS_METAL_PYTHON = False

# Try to import PyObjC for Metal interop
try:
    from Foundation import NSData, NSError
    import Metal
    from Metal import (
        MTLCreateSystemDefaultDevice,
        MTLCompileOptions,
        MTLSize,
        MTLResourceStorageModeShared,
        MTLFunctionConstantValues,
    )
    import objc
    HAS_PYOBJC = True
except ImportError:
    HAS_PYOBJC = False
    Metal = None
    MTLFunctionConstantValues = None

from vllm.logger import init_logger

logger = init_logger(__name__)


class MetalDevice:
    """Wrapper for Metal device and command queue."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        if not HAS_PYOBJC:
            raise RuntimeError(
                "PyObjC is required for Metal bridge. "
                "Install with: pip install pyobjc-framework-Metal pyobjc-framework-Foundation"
            )

        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Failed to create Metal device. Is MPS available?")

        self.command_queue = self.device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create Metal command queue")

        self._libraries: Dict[str, Any] = {}
        self._pipelines: Dict[str, Any] = {}

        self._initialized = True
        logger.info(f"Metal device initialized: {self.device.name()}")

    def compile_shader(
        self,
        source_path: str,
        function_constants: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Compile Metal shader from source file.

        Args:
            source_path: Path to .metal shader file
            function_constants: Optional dict of function constants

        Returns:
            MTLLibrary object
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Metal shader not found: {source_path}")

        # Check cache
        cache_key = str(source_path)
        if cache_key in self._libraries:
            return self._libraries[cache_key]

        # Read shader source
        with open(source_path, "r") as f:
            source_code = f.read()

        # Compile options
        options = MTLCompileOptions.alloc().init()

        # Set function constants if provided
        if function_constants:
            constants = self.device.newMutableFunctionConstantsWithDictionary_(None)
            for name, value in function_constants.items():
                if isinstance(value, int):
                    constants.setConstantValue_type_atIndex_(
                        ctypes.c_uint(value),
                        0,  # MTLDataTypeUInt
                        int(name) if name.isdigit() else hash(name) % 100
                    )
            options.setConstantValues_(constants)

        # Compile
        library, error = self.device.newLibraryWithSource_options_error_(
            source_code, options, None
        )

        if library is None or error is not None:
            raise RuntimeError(f"Failed to compile Metal shader: {error}")

        self._libraries[cache_key] = library
        logger.info(f"Compiled Metal shader: {source_path}")

        return library

    def get_pipeline(
        self,
        library: Any,
        function_name: str,
        function_constants: Optional[Dict[int, int]] = None,
    ) -> Any:
        """Get or create compute pipeline for a function.

        Args:
            library: MTLLibrary containing the function
            function_name: Name of the kernel function
            function_constants: Optional dict mapping constant index to value

        Returns:
            MTLComputePipelineState
        """
        cache_key = f"{id(library)}_{function_name}_{function_constants}"
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        # Create function constants if needed
        if function_constants and MTLFunctionConstantValues is not None:
            constants = MTLFunctionConstantValues.alloc().init()
            for idx, value in function_constants.items():
                # Set uint constant at index
                import struct
                data = struct.pack("I", value)  # unsigned int
                constants.setConstantValue_type_atIndex_(
                    data,
                    Metal.MTLDataTypeUInt,  # MTLDataTypeUInt = 9
                    idx
                )
            error_ptr = objc.nil
            function = library.newFunctionWithName_constantValues_error_(
                function_name, constants, error_ptr
            )
            if function is None:
                # Fallback to non-specialized function
                logger.warning(f"Failed to create specialized function: {error_ptr}")
                function = library.newFunctionWithName_(function_name)
        else:
            function = library.newFunctionWithName_(function_name)

        if function is None:
            available = [str(f) for f in library.functionNames()]
            raise RuntimeError(
                f"Function '{function_name}' not found in library. "
                f"Available: {available}"
            )

        # Create pipeline
        error_ptr = objc.nil
        pipeline = self.device.newComputePipelineStateWithFunction_error_(
            function, error_ptr
        )

        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline: {error_ptr}")

        self._pipelines[cache_key] = pipeline
        return pipeline

    def create_buffer_from_tensor(self, tensor: torch.Tensor) -> Any:
        """Create MTLBuffer from PyTorch tensor.

        For now, we copy data to a Metal buffer. Zero-copy requires
        accessing PyTorch's internal MPS buffer, which is complex.

        Note: MPS tensors require synchronization before CPU transfer
        to avoid conflicts between PyTorch MPS and PyObjC Metal.
        """
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Synchronize MPS before CPU transfer to avoid conflicts
        if tensor.device.type == "mps":
            torch.mps.synchronize()

        # Move to CPU for buffer creation - use detach to avoid grad issues
        cpu_tensor = tensor.detach().cpu()

        # Get raw bytes from numpy
        numpy_data = cpu_tensor.numpy()
        data_bytes = numpy_data.tobytes()
        num_bytes = len(data_bytes)

        # Create Metal buffer from bytes
        buffer = self.device.newBufferWithBytes_length_options_(
            data_bytes,
            num_bytes,
            MTLResourceStorageModeShared
        )

        return buffer

    def run_kernel(
        self,
        pipeline: Any,
        buffers: List[Any],
        grid_size: Tuple[int, int, int],
        threadgroup_size: Tuple[int, int, int],
    ) -> None:
        """Execute a compute kernel.

        Args:
            pipeline: MTLComputePipelineState
            buffers: List of MTLBuffers for kernel arguments
            grid_size: (width, height, depth) of compute grid
            threadgroup_size: (width, height, depth) of threadgroups
        """
        # Synchronize MPS before running Metal kernel to avoid GPU conflicts
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        # Create command buffer
        command_buffer = self.command_queue.commandBuffer()

        # Create compute encoder
        encoder = command_buffer.computeCommandEncoder()

        # Set pipeline
        encoder.setComputePipelineState_(pipeline)

        # Set buffers
        for idx, buffer in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, idx)

        # Dispatch threads
        grid = MTLSize(grid_size[0], grid_size[1], grid_size[2])
        threadgroup = MTLSize(threadgroup_size[0], threadgroup_size[1], threadgroup_size[2])

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid, threadgroup)

        # End encoding
        encoder.endEncoding()

        # Commit and wait
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Check for errors
        error = command_buffer.error()
        if error is not None:
            raise RuntimeError(f"Metal kernel execution failed: {error}")


class PagedAttentionMetal:
    """Metal implementation of PagedAttention.

    This class provides a high-level interface for running PagedAttention
    on Metal, compatible with vLLM's attention backend interface.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_query_heads: int,
        head_size: int,
        block_size: int,
        scale: float,
    ):
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.head_size = head_size
        self.block_size = block_size
        self.scale = scale
        self.queries_per_kv = num_query_heads // num_kv_heads

        # Initialize Metal device
        self.device = MetalDevice()

        # Compile shader
        shader_path = Path(__file__).parent.parent / "kernels" / "paged_attention.metal"
        self.library = self.device.compile_shader(str(shader_path))

        # Create pipeline (no function constants - all params via buffer)
        self.pipeline = self.device.get_pipeline(
            self.library,
            "paged_attention_decode",
            function_constants=None
        )

        # Also get test kernels for verification
        self.test_copy_pipeline = self.device.get_pipeline(self.library, "test_copy")

        logger.info(
            f"PagedAttentionMetal initialized: "
            f"heads={num_query_heads}/{num_kv_heads}, "
            f"head_size={head_size}, block_size={block_size}"
        )

    def forward(
        self,
        query: torch.Tensor,          # [num_seqs, num_query_heads, head_size]
        key_cache: torch.Tensor,      # [num_blocks, num_kv_heads, block_size, head_size]
        value_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_size]
        block_table: torch.Tensor,    # [num_seqs, max_blocks_per_seq]
        seq_lens: torch.Tensor,       # [num_seqs]
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run PagedAttention on Metal.

        Args:
            query: Query tensor
            key_cache: Paged key cache
            value_cache: Paged value cache
            block_table: Block table mapping
            seq_lens: Sequence lengths
            output: Optional pre-allocated output tensor

        Returns:
            Attention output tensor
        """
        num_seqs = query.shape[0]

        # Allocate output if needed
        if output is None:
            output = torch.empty_like(query)

        # Create params buffer matching PagedAttentionParams struct
        import struct
        params_data = struct.pack(
            "fIIIIIIIIIIIIIIIIII",  # float + 18 uints (19 values total)
            self.scale,                                          # scale
            num_seqs,                                            # num_seqs
            int(seq_lens.max().item()),                          # max_seq_len
            block_table.shape[1],                                # max_blocks_per_seq
            # Model config
            self.head_size,                                      # head_size
            self.block_size,                                     # block_size
            self.num_kv_heads,                                   # num_kv_heads
            self.num_query_heads,                                # num_query_heads
            self.queries_per_kv,                                 # queries_per_kv
            # K strides
            self.num_kv_heads * self.block_size * self.head_size,  # k_stride_block
            self.block_size * self.head_size,                      # k_stride_head
            self.head_size,                                        # k_stride_token
            # V strides (same)
            self.num_kv_heads * self.block_size * self.head_size,  # v_stride_block
            self.block_size * self.head_size,                      # v_stride_head
            self.head_size,                                        # v_stride_token
            # Q strides
            self.num_query_heads * self.head_size,                 # q_stride_token
            self.head_size,                                        # q_stride_head
            # O strides (same as Q)
            self.num_query_heads * self.head_size,                 # o_stride_token
            self.head_size,                                        # o_stride_head
        )

        params_buffer = self.device.device.newBufferWithBytes_length_options_(
            params_data, len(params_data), MTLResourceStorageModeShared
        )

        # Create input buffers
        buffers = [
            self.device.create_buffer_from_tensor(query),
            self.device.create_buffer_from_tensor(key_cache),
            self.device.create_buffer_from_tensor(value_cache),
            self.device.create_buffer_from_tensor(block_table.int()),
            self.device.create_buffer_from_tensor(seq_lens.int()),
        ]

        # Create output buffer
        output_bytes = output.numel() * output.element_size()
        output_buffer = self.device.device.newBufferWithLength_options_(
            output_bytes, MTLResourceStorageModeShared
        )
        buffers.append(output_buffer)
        buffers.append(params_buffer)

        # Compute grid size
        grid_size = (num_seqs, self.num_query_heads, 1)
        threadgroup_size = (32, 1, 1)  # One SIMD group per (seq, head)

        # Run kernel
        self.device.run_kernel(self.pipeline, buffers, grid_size, threadgroup_size)

        # Copy result back to output tensor
        import numpy as np
        output_ptr = output_buffer.contents()
        # Create numpy array from buffer contents - use bytes() for safety
        output_np = np.frombuffer(
            bytes(output_ptr.as_buffer(output_bytes)),
            dtype=np.float16
        ).reshape(output.shape)
        output.copy_(torch.from_numpy(output_np.copy()).to(output.device))

        return output


def is_metal_available() -> bool:
    """Check if Metal runtime is available."""
    return HAS_PYOBJC and torch.backends.mps.is_available()


# Convenience function
def get_metal_device() -> MetalDevice:
    """Get singleton Metal device."""
    return MetalDevice()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal PagedAttention Fused Kernel Bridge - KV Write + Attention in one pass.

ETAP 4: Fuzja KV-write z attention

This module provides Python bindings for the fused Metal compute kernel that:
1. Writes new K/V for the current decode token directly to KV-cache
2. Computes attention over all cached tokens in the same kernel dispatch
3. Eliminates separate CPU-side KV update (was ~46% of total time)

Key benefits:
- No separate _update_kv_cache_metal() call for decode path
- Single dispatch instead of CPU sync + write + dispatch
- K/V data stays in GPU registers/TGMEM after write for immediate use

Environment variables:
- VLLM_METAL_FUSED_KV=1: Enable fused KV-write + attention (default: 1)
- VLLM_METAL_FUSED_KV=0: Disable fused path (use separate KV update)
"""

import ctypes
import os
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import torch

from Metal import (
    MTLCreateSystemDefaultDevice,
    MTLResourceStorageModeShared,
    MTLSize,
)

logger = logging.getLogger(__name__)

# Environment variable control
ENABLE_FUSED_KV = os.environ.get("VLLM_METAL_FUSED_KV", "1") == "1"


def is_metal_available() -> bool:
    """Check if Metal is available."""
    try:
        device = MTLCreateSystemDefaultDevice()
        return device is not None
    except Exception:
        return False


class FusedAttentionParams(ctypes.Structure):
    """Parameters struct matching Metal shader FusedAttentionParams."""
    _fields_ = [
        ("scale", ctypes.c_float),
        ("num_seqs", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("max_blocks_per_seq", ctypes.c_uint32),
        ("head_size", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("num_kv_heads", ctypes.c_uint32),
        ("num_query_heads", ctypes.c_uint32),
        ("queries_per_kv", ctypes.c_uint32),
        # KV cache strides
        ("k_stride_block", ctypes.c_uint32),
        ("k_stride_head", ctypes.c_uint32),
        ("k_stride_token", ctypes.c_uint32),
        ("v_stride_block", ctypes.c_uint32),
        ("v_stride_head", ctypes.c_uint32),
        ("v_stride_token", ctypes.c_uint32),
        # Query/Output strides
        ("q_stride_token", ctypes.c_uint32),
        ("q_stride_head", ctypes.c_uint32),
        ("o_stride_token", ctypes.c_uint32),
        ("o_stride_head", ctypes.c_uint32),
        # New K/V strides
        ("new_kv_stride_token", ctypes.c_uint32),
        ("new_kv_stride_head", ctypes.c_uint32),
    ]


class MetalPagedAttentionFused:
    """Metal PagedAttention with fused KV-write + attention.

    This class manages the fused kernel that writes new K/V to cache
    and computes attention in a single dispatch.

    Attributes:
        num_kv_heads: Number of KV attention heads
        num_query_heads: Number of query heads
        head_size: Size of each attention head
        block_size: Number of tokens per KV cache block
        scale: Attention scaling factor (1/sqrt(head_size))
        kernel_name: Name of the selected Metal kernel
        using_fused: Whether fused kernel is being used
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_query_heads: int,
        head_size: int,
        block_size: int,
        scale: Optional[float] = None,
        max_num_blocks: int = 4096,
        max_batch_size: int = 64,
    ):
        """Initialize fused Metal PagedAttention.

        Args:
            num_kv_heads: Number of KV heads
            num_query_heads: Number of query heads
            head_size: Head dimension size
            block_size: Tokens per block
            scale: Attention scale factor (default: 1/sqrt(head_size))
            max_num_blocks: Maximum number of KV cache blocks
            max_batch_size: Maximum batch size
        """
        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.head_size = head_size
        self.block_size = block_size
        self.scale = scale if scale is not None else 1.0 / (head_size ** 0.5)
        self.max_num_blocks = max_num_blocks
        self.max_batch_size = max_batch_size
        self.queries_per_kv = num_query_heads // num_kv_heads

        # Create Metal device
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Failed to create Metal device")

        # Compute strides for KV cache: [num_blocks, num_kv_heads, block_size, head_size]
        self.kv_stride_block = num_kv_heads * block_size * head_size
        self.kv_stride_head = block_size * head_size
        self.kv_stride_token = head_size

        # Compute strides for Q/O: [num_seqs, num_query_heads, head_size]
        self.q_stride_token = num_query_heads * head_size
        self.q_stride_head = head_size
        self.o_stride_token = num_query_heads * head_size
        self.o_stride_head = head_size

        # Compute strides for new K/V: [num_seqs, num_kv_heads, head_size]
        self.new_kv_stride_token = num_kv_heads * head_size
        self.new_kv_stride_head = head_size

        # Load and compile shader
        self._load_shader()
        self._select_kernel()

        logger.info(
            f"MetalPagedAttentionFused initialized: "
            f"kernel={self.kernel_name}, fused={self.using_fused}, "
            f"num_kv_heads={num_kv_heads}, num_query_heads={num_query_heads}, "
            f"head_size={head_size}, block_size={block_size}"
        )

    def _load_shader(self) -> None:
        """Load and compile Metal shader."""
        # Find shader file
        shader_path = Path(__file__).parent.parent / "kernels" / "paged_attention_fused.metal"
        if not shader_path.exists():
            raise FileNotFoundError(f"Shader not found: {shader_path}")

        with open(shader_path, "r") as f:
            source = f.read()

        # Compile shader
        options = None
        library, error = self.device.newLibraryWithSource_options_error_(
            source, options, None
        )
        if library is None:
            raise RuntimeError(f"Failed to compile shader: {error}")

        self.library = library

        # Get available function names
        self.available_functions = list(library.functionNames())

    def _select_kernel(self) -> None:
        """Select appropriate kernels for two-phase fused execution."""
        self.using_fused = ENABLE_FUSED_KV

        # Phase 1: KV write kernel
        self.kv_write_kernel_name = "kv_write_decode"
        if self.kv_write_kernel_name in self.available_functions:
            kv_write_function = self.library.newFunctionWithName_(self.kv_write_kernel_name)
            if kv_write_function is None:
                raise RuntimeError(f"Failed to get KV write function")
            self.kv_write_pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
                kv_write_function, None
            )
            if self.kv_write_pipeline is None:
                raise RuntimeError(f"Failed to create KV write pipeline: {error}")
        else:
            self.using_fused = False
            self.kv_write_pipeline = None

        # Phase 2: Attention kernel selection based on head_size
        # V1.5: Only h128 (optimized) and generic (fallback) kernels
        if self.using_fused:
            if self.head_size == 128 and "paged_attention_fused_h128" in self.available_functions:
                self.kernel_name = "paged_attention_fused_h128"
            elif "paged_attention_fused_generic" in self.available_functions:
                self.kernel_name = "paged_attention_fused_generic"
            else:
                self.using_fused = False
                raise RuntimeError(f"No suitable fused kernel for head_size={self.head_size}")
        else:
            raise RuntimeError("Fused KV disabled - use MetalPagedAttentionV2 for non-fused path")

        # Create attention pipeline state
        function = self.library.newFunctionWithName_(self.kernel_name)
        if function is None:
            raise RuntimeError(f"Failed to get function: {self.kernel_name}")

        pipeline_state, error = self.device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if pipeline_state is None:
            raise RuntimeError(f"Failed to create pipeline state: {error}")

        self.pipeline_state = pipeline_state

        # Create command queue
        self.command_queue = self.device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create command queue")

    def forward_fused(
        self,
        query: torch.Tensor,           # [num_seqs, num_query_heads, head_size]
        new_keys: torch.Tensor,        # [num_seqs, num_kv_heads, head_size]
        new_values: torch.Tensor,      # [num_seqs, num_kv_heads, head_size]
        key_buffer,                    # MTLBuffer for key cache
        value_buffer,                  # MTLBuffer for value cache
        block_table: torch.Tensor,     # [num_seqs, max_blocks_per_seq]
        seq_lens: torch.Tensor,        # [num_seqs]
        output: torch.Tensor,          # [num_seqs, num_query_heads, head_size]
    ) -> torch.Tensor:
        """Execute fused KV-write + attention kernel.

        This writes new_keys/new_values to KV cache and computes attention
        in a single kernel dispatch.

        Args:
            query: Query tensor [num_seqs, num_query_heads, head_size]
            new_keys: New key tensor for current decode token [num_seqs, num_kv_heads, head_size]
            new_values: New value tensor for current decode token [num_seqs, num_kv_heads, head_size]
            key_buffer: MTLBuffer containing key cache
            value_buffer: MTLBuffer containing value cache
            block_table: Block table mapping logical to physical blocks
            seq_lens: Sequence lengths (including new token)
            output: Output tensor [num_seqs, num_query_heads, head_size]

        Returns:
            Output tensor with attention results
        """
        assert self.using_fused, "Fused kernel not available"

        num_seqs = query.shape[0]
        max_blocks_per_seq = block_table.shape[1] if block_table.dim() > 1 else 1

        # Prepare parameters
        params = FusedAttentionParams(
            scale=self.scale,
            num_seqs=num_seqs,
            max_seq_len=int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0,
            max_blocks_per_seq=max_blocks_per_seq,
            head_size=self.head_size,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            num_query_heads=self.num_query_heads,
            queries_per_kv=self.queries_per_kv,
            # KV cache strides
            k_stride_block=self.kv_stride_block,
            k_stride_head=self.kv_stride_head,
            k_stride_token=self.kv_stride_token,
            v_stride_block=self.kv_stride_block,
            v_stride_head=self.kv_stride_head,
            v_stride_token=self.kv_stride_token,
            # Query/Output strides
            q_stride_token=self.q_stride_token,
            q_stride_head=self.q_stride_head,
            o_stride_token=self.o_stride_token,
            o_stride_head=self.o_stride_head,
            # New K/V strides
            new_kv_stride_token=self.new_kv_stride_token,
            new_kv_stride_head=self.new_kv_stride_head,
        )

        # Ensure tensors are contiguous and on CPU
        query_np = query.contiguous().numpy() if query.device.type == "cpu" else query.cpu().numpy()
        new_keys_np = new_keys.contiguous().numpy() if new_keys.device.type == "cpu" else new_keys.cpu().numpy()
        new_values_np = new_values.contiguous().numpy() if new_values.device.type == "cpu" else new_values.cpu().numpy()
        block_table_np = block_table.to(torch.int32).contiguous().numpy() if block_table.device.type == "cpu" else block_table.to(torch.int32).cpu().numpy()
        seq_lens_np = seq_lens.to(torch.int32).contiguous().numpy() if seq_lens.device.type == "cpu" else seq_lens.to(torch.int32).cpu().numpy()
        output_np = np.zeros((num_seqs, self.num_query_heads, self.head_size), dtype=np.float16)

        # Create Metal buffers for input tensors
        query_buffer_local = self._create_buffer(query_np)
        new_keys_buffer = self._create_buffer(new_keys_np)
        new_values_buffer = self._create_buffer(new_values_np)
        block_table_buffer = self._create_buffer(block_table_np)
        seq_lens_buffer = self._create_buffer(seq_lens_np)
        output_buffer = self._create_buffer(output_np)
        params_buffer = self._create_buffer(bytes(params))

        # Create command buffer
        command_buffer = self.command_queue.commandBuffer()

        if self.kernel_name == "paged_attention_fused_h128":
            # =================================================================
            # Two-phase path (preferred for head_size=128):
            #   1) KV write kernel
            #   2) Attention kernel that reads already-written KV
            # =================================================================
            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(self.kv_write_pipeline)

            # KV write kernel buffer layout:
            # buffer(0): key_cache (writable)
            # buffer(1): value_cache (writable)
            # buffer(2): block_table
            # buffer(3): seq_lens
            # buffer(4): params
            # buffer(5): new_keys
            # buffer(6): new_values
            encoder.setBuffer_offset_atIndex_(key_buffer, 0, 0)
            encoder.setBuffer_offset_atIndex_(value_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(seq_lens_buffer, 0, 3)
            encoder.setBuffer_offset_atIndex_(params_buffer, 0, 4)
            encoder.setBuffer_offset_atIndex_(new_keys_buffer, 0, 5)
            encoder.setBuffer_offset_atIndex_(new_values_buffer, 0, 6)

            # KV write dispatch: (num_seqs, num_kv_heads) threadgroups
            threads_per_threadgroup = MTLSize(32, 1, 1)
            kv_write_threadgroups = MTLSize(num_seqs, self.num_kv_heads, 1)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                kv_write_threadgroups, threads_per_threadgroup
            )
            encoder.endEncoding()

            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(self.pipeline_state)

            # Attention kernel buffer layout:
            # buffer(0): query
            # buffer(1): key_cache (read-only)
            # buffer(2): value_cache (read-only)
            # buffer(3): block_table
            # buffer(4): seq_lens
            # buffer(5): output
            # buffer(6): params
            encoder.setBuffer_offset_atIndex_(query_buffer_local, 0, 0)
            encoder.setBuffer_offset_atIndex_(key_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(value_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 3)
            encoder.setBuffer_offset_atIndex_(seq_lens_buffer, 0, 4)
            encoder.setBuffer_offset_atIndex_(output_buffer, 0, 5)
            encoder.setBuffer_offset_atIndex_(params_buffer, 0, 6)

            threads_per_threadgroup = MTLSize(32, 1, 1)
            threadgroups = MTLSize(num_seqs, self.num_query_heads, 1)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                threadgroups, threads_per_threadgroup
            )
            encoder.endEncoding()
        elif self.kernel_name == "paged_attention_fused_generic":
            # =================================================================
            # Single-dispatch fused path:
            #   - Kernel writes new K/V and computes attention.
            # =================================================================
            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(self.pipeline_state)

            # Generic fused kernel buffer layout:
            # buffer(0): query
            # buffer(1): key_cache (writable)
            # buffer(2): value_cache (writable)
            # buffer(3): block_table
            # buffer(4): seq_lens
            # buffer(5): output
            # buffer(6): params
            # buffer(7): new_keys
            # buffer(8): new_values
            encoder.setBuffer_offset_atIndex_(query_buffer_local, 0, 0)
            encoder.setBuffer_offset_atIndex_(key_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(value_buffer, 0, 2)
            encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 3)
            encoder.setBuffer_offset_atIndex_(seq_lens_buffer, 0, 4)
            encoder.setBuffer_offset_atIndex_(output_buffer, 0, 5)
            encoder.setBuffer_offset_atIndex_(params_buffer, 0, 6)
            encoder.setBuffer_offset_atIndex_(new_keys_buffer, 0, 7)
            encoder.setBuffer_offset_atIndex_(new_values_buffer, 0, 8)

            threads_per_threadgroup = MTLSize(32, 1, 1)
            threadgroups = MTLSize(num_seqs, self.num_query_heads, 1)
            encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                threadgroups, threads_per_threadgroup
            )
            encoder.endEncoding()
        else:
            raise RuntimeError(f"Unsupported fused kernel: {self.kernel_name}")

        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Copy output back
        output_mv = output_buffer.contents().as_buffer(output_buffer.length())
        output_np = np.frombuffer(output_mv, dtype=np.float16).reshape(
            num_seqs, self.num_query_heads, self.head_size
        ).copy()

        output.copy_(torch.from_numpy(output_np))
        return output

    def forward_with_metal_buffers(
        self,
        query: torch.Tensor,
        key_buffer,
        value_buffer,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Execute non-fused attention kernel (for fallback/comparison).

        This is the standard attention kernel that reads from already-populated
        KV cache. Used when fused mode is disabled.

        Args:
            query: Query tensor [num_seqs, num_query_heads, head_size]
            key_buffer: MTLBuffer containing key cache
            value_buffer: MTLBuffer containing value cache
            block_table: Block table
            seq_lens: Sequence lengths
            output: Output tensor

        Returns:
            Output tensor with attention results
        """
        num_seqs = query.shape[0]
        max_blocks_per_seq = block_table.shape[1] if block_table.dim() > 1 else 1

        # Prepare parameters
        params = FusedAttentionParams(
            scale=self.scale,
            num_seqs=num_seqs,
            max_seq_len=int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0,
            max_blocks_per_seq=max_blocks_per_seq,
            head_size=self.head_size,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            num_query_heads=self.num_query_heads,
            queries_per_kv=self.queries_per_kv,
            k_stride_block=self.kv_stride_block,
            k_stride_head=self.kv_stride_head,
            k_stride_token=self.kv_stride_token,
            v_stride_block=self.kv_stride_block,
            v_stride_head=self.kv_stride_head,
            v_stride_token=self.kv_stride_token,
            q_stride_token=self.q_stride_token,
            q_stride_head=self.q_stride_head,
            o_stride_token=self.o_stride_token,
            o_stride_head=self.o_stride_head,
            new_kv_stride_token=0,
            new_kv_stride_head=0,
        )

        # Prepare numpy arrays
        query_np = query.contiguous().numpy() if query.device.type == "cpu" else query.cpu().numpy()
        block_table_np = block_table.to(torch.int32).contiguous().numpy() if block_table.device.type == "cpu" else block_table.to(torch.int32).cpu().numpy()
        seq_lens_np = seq_lens.to(torch.int32).contiguous().numpy() if seq_lens.device.type == "cpu" else seq_lens.to(torch.int32).cpu().numpy()
        output_np = np.zeros((num_seqs, self.num_query_heads, self.head_size), dtype=np.float16)

        # Create buffers
        query_buffer = self._create_buffer(query_np)
        block_table_buffer = self._create_buffer(block_table_np)
        seq_lens_buffer = self._create_buffer(seq_lens_np)
        output_buffer = self._create_buffer(output_np)
        params_buffer = self._create_buffer(bytes(params))

        # Execute kernel (non-fused version)
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline_state)

        encoder.setBuffer_offset_atIndex_(query_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(key_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(value_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(block_table_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(seq_lens_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, 5)
        encoder.setBuffer_offset_atIndex_(params_buffer, 0, 6)

        threads_per_threadgroup = MTLSize(32, 1, 1)
        threadgroups = MTLSize(num_seqs, self.num_query_heads, 1)

        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups, threads_per_threadgroup
        )

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Copy output
        output_mv = output_buffer.contents().as_buffer(output_buffer.length())
        output_np = np.frombuffer(output_mv, dtype=np.float16).reshape(
            num_seqs, self.num_query_heads, self.head_size
        ).copy()

        output.copy_(torch.from_numpy(output_np))
        return output

    def _create_buffer(self, data) -> 'MTLBuffer':
        """Create MTLBuffer from numpy array or bytes."""
        if isinstance(data, bytes):
            buffer = self.device.newBufferWithBytes_length_options_(
                data, len(data), MTLResourceStorageModeShared
            )
        else:
            buffer = self.device.newBufferWithBytes_length_options_(
                data.tobytes(), data.nbytes, MTLResourceStorageModeShared
            )
        if buffer is None:
            raise RuntimeError("Failed to create Metal buffer")
        return buffer

    @property
    def is_fused_available(self) -> bool:
        """Check if fused kernel is available and enabled."""
        return self.using_fused and ENABLE_FUSED_KV

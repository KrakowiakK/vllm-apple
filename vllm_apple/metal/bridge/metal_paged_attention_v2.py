"""Metal PagedAttention V2 - Optimized implementation.

Key optimizations over V1:
1. Query vector loaded into registers once (kernel optimization)
2. Specialized kernel for head_size=128
3. KV cache buffers are persistent - no re-copy on each forward
4. Validation of head_size in Python
5. Pre-allocated output buffer
"""

import struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch

from Metal import (
    MTLCreateSystemDefaultDevice,
    MTLCompileOptions,
    MTLSize,
    MTLResourceStorageModeShared,
)


class MetalPagedAttentionV2:
    """Optimized Metal PagedAttention with persistent KV cache buffers.

    This version:
    - Does NOT copy KV cache every forward (huge savings!)
    - Requires explicit set_kv_cache() call when cache changes
    - Uses specialized kernel for head_size=128
    """

    # Supported head sizes
    SUPPORTED_HEAD_SIZES = {32, 64, 96, 128}

    def __init__(
        self,
        num_kv_heads: int,
        num_query_heads: int,
        head_size: int,
        block_size: int,
        scale: float,
        max_num_blocks: int = 256,
        max_batch_size: int = 64,
    ):
        # Validate head_size
        if head_size not in self.SUPPORTED_HEAD_SIZES:
            raise ValueError(
                f"head_size={head_size} not supported. "
                f"Supported: {sorted(self.SUPPORTED_HEAD_SIZES)}"
            )

        self.num_kv_heads = num_kv_heads
        self.num_query_heads = num_query_heads
        self.head_size = head_size
        self.block_size = block_size
        self.scale = scale
        self.queries_per_kv = num_query_heads // num_kv_heads
        self.max_num_blocks = max_num_blocks
        self.max_batch_size = max_batch_size

        # Pre-compute strides (never change)
        self._k_stride_block = num_kv_heads * block_size * head_size
        self._k_stride_head = block_size * head_size
        self._k_stride_token = head_size
        self._v_stride_block = num_kv_heads * block_size * head_size
        self._v_stride_head = block_size * head_size
        self._v_stride_token = head_size
        self._q_stride_token = num_query_heads * head_size
        self._q_stride_head = head_size
        self._o_stride_token = num_query_heads * head_size
        self._o_stride_head = head_size

        # Create Metal device and queue
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Failed to create Metal device")

        self.command_queue = self.device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create command queue")

        # Compile shader (V2)
        shader_path = Path(__file__).parent.parent / "kernels" / "paged_attention_v2.metal"
        if not shader_path.exists():
            # Fallback to V1
            shader_path = Path(__file__).parent.parent / "kernels" / "paged_attention.metal"

        with open(shader_path) as f:
            shader_source = f.read()

        options = MTLCompileOptions.alloc().init()
        library, error = self.device.newLibraryWithSource_options_error_(
            shader_source, options, None
        )
        if library is None:
            raise RuntimeError(f"Failed to compile shader: {error}")

        # Select kernel based on head_size
        if head_size == 128 and "paged_attention_v2_decode_h128" in list(library.functionNames()):
            kernel_name = "paged_attention_v2_decode_h128"
        elif "paged_attention_v2_decode" in list(library.functionNames()):
            kernel_name = "paged_attention_v2_decode"
        else:
            kernel_name = "paged_attention_decode"

        function = library.newFunctionWithName_(kernel_name)
        if function is None:
            raise RuntimeError(f"Function '{kernel_name}' not found")

        self.pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if self.pipeline is None:
            raise RuntimeError(f"Failed to create pipeline: {error}")

        # Pre-allocate buffers
        self._init_buffers()

        # Track if KV cache is set
        self._kv_cache_set = False
        self._kv_cache_version = 0

    def _init_buffers(self):
        """Pre-allocate Metal buffers."""
        # Query: [max_batch, num_query_heads, head_size]
        query_size = self.max_batch_size * self.num_query_heads * self.head_size * 2
        self.query_buf = self.device.newBufferWithLength_options_(
            query_size, MTLResourceStorageModeShared
        )

        # KV cache: [max_blocks, num_kv_heads, block_size, head_size]
        # Will be resized dynamically if needed
        kv_size = self.max_num_blocks * self.num_kv_heads * self.block_size * self.head_size * 2
        self.key_buf = self.device.newBufferWithLength_options_(
            kv_size, MTLResourceStorageModeShared
        )
        self.value_buf = self.device.newBufferWithLength_options_(
            kv_size, MTLResourceStorageModeShared
        )
        self._current_kv_buf_size = kv_size

        # Block table: estimate max_blocks_per_seq (will be resized if needed)
        max_seq_len = self.max_num_blocks * self.block_size
        max_blocks_per_seq = (max_seq_len + self.block_size - 1) // self.block_size
        self.max_blocks_per_seq = max_blocks_per_seq
        block_table_size = self.max_batch_size * max_blocks_per_seq * 4
        self.block_table_buf = self.device.newBufferWithLength_options_(
            block_table_size, MTLResourceStorageModeShared
        )
        self._current_block_table_buf_size = block_table_size

        # Seq lens
        self.seq_lens_buf = self.device.newBufferWithLength_options_(
            self.max_batch_size * 4, MTLResourceStorageModeShared
        )

        # Output
        output_size = self.max_batch_size * self.num_query_heads * self.head_size * 2
        self.output_buf = self.device.newBufferWithLength_options_(
            output_size, MTLResourceStorageModeShared
        )

        # Params (fixed size struct)
        self.params_buf = self.device.newBufferWithLength_options_(
            76, MTLResourceStorageModeShared
        )

    def _ensure_kv_buffer_size(self, required_size: int):
        """Ensure KV buffers are large enough, reallocating if needed."""
        if required_size > self._current_kv_buf_size:
            # Grow by 50% more than needed to reduce future reallocations
            new_size = int(required_size * 1.5)
            old_size = self._current_kv_buf_size
            self.key_buf = self.device.newBufferWithLength_options_(
                new_size, MTLResourceStorageModeShared
            )
            self.value_buf = self.device.newBufferWithLength_options_(
                new_size, MTLResourceStorageModeShared
            )
            self._current_kv_buf_size = new_size
            self._kv_cache_set = False  # Need to re-copy cache

            # Log reallocation for debugging
            import os
            if os.environ.get("METAL_PROFILE_DETAIL"):
                print(f"[METAL REALLOC] KV buffers: {old_size} -> {new_size} bytes (required: {required_size})")

    def _ensure_block_table_buffer_size(self, required_size: int):
        """Ensure block table buffer is large enough."""
        if required_size > self._current_block_table_buf_size:
            old_size = self._current_block_table_buf_size
            new_size = int(required_size * 1.5)
            self.block_table_buf = self.device.newBufferWithLength_options_(
                new_size, MTLResourceStorageModeShared
            )
            self._current_block_table_buf_size = new_size

            # Log reallocation for debugging
            import os
            if os.environ.get("METAL_PROFILE_DETAIL"):
                print(f"[METAL REALLOC] Block table buffer: {old_size} -> {new_size} bytes (required: {required_size})")

    def _ensure_batch_buffers(self, required_batch_size: int):
        """Grow all batch-related buffers together if needed.

        This method ensures query_buf, output_buf, seq_lens_buf, and block_table_buf
        are all large enough for the required batch size. All buffers are resized
        together to maintain consistency.

        Args:
            required_batch_size: Number of sequences in the batch
        """
        if required_batch_size <= self.max_batch_size:
            return

        # Grow by 50% more than needed to reduce future reallocations
        new_size = int(required_batch_size * 1.5)
        old_size = self.max_batch_size
        self.max_batch_size = new_size

        # Log reallocation for debugging
        import os
        if os.environ.get("METAL_PROFILE_DETAIL"):
            print(f"[METAL REALLOC] Batch buffers: {old_size} -> {new_size} (required: {required_batch_size})")

        # Resize query buffer: [max_batch, num_query_heads, head_size]
        query_size = new_size * self.num_query_heads * self.head_size * 2  # float16
        self.query_buf = self.device.newBufferWithLength_options_(
            query_size, MTLResourceStorageModeShared
        )

        # Resize output buffer: [max_batch, num_query_heads, head_size]
        output_size = new_size * self.num_query_heads * self.head_size * 2  # float16
        self.output_buf = self.device.newBufferWithLength_options_(
            output_size, MTLResourceStorageModeShared
        )

        # Resize seq_lens buffer: [max_batch]
        self.seq_lens_buf = self.device.newBufferWithLength_options_(
            new_size * 4, MTLResourceStorageModeShared  # int32
        )

        # Resize block_table buffer: [max_batch, max_blocks_per_seq]
        block_table_size = new_size * self.max_blocks_per_seq * 4  # int32
        self.block_table_buf = self.device.newBufferWithLength_options_(
            block_table_size, MTLResourceStorageModeShared
        )
        self._current_block_table_buf_size = block_table_size

        # Params buffer is fixed size (76 bytes), no need to resize

    def _update_buffer(self, buffer, data: bytes, offset: int = 0):
        """Update buffer contents efficiently."""
        contents = buffer.contents()
        mv = contents.as_buffer(buffer.length())
        mv[offset:offset + len(data)] = data

    def set_kv_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ):
        """Set KV cache buffers. Call this when cache is updated.

        For decode, this should be called once at start, then only when
        new tokens are appended (incremental update possible).

        Args:
            key_cache: [num_blocks, num_kv_heads, block_size, head_size]
            value_cache: [num_blocks, num_kv_heads, block_size, head_size]
        """
        if key_cache.device.type == "mps":
            torch.mps.synchronize()
            key_cache = key_cache.cpu()
            value_cache = value_cache.cpu()

        key_bytes = key_cache.detach().numpy().tobytes()
        value_bytes = value_cache.detach().numpy().tobytes()

        self._update_buffer(self.key_buf, key_bytes)
        self._update_buffer(self.value_buf, value_bytes)
        self._kv_cache_set = True
        self._kv_cache_version += 1

    def update_kv_cache_block(
        self,
        block_idx: int,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
    ):
        """Update a single block in KV cache (for incremental decode).

        Args:
            block_idx: Physical block index to update
            key_block: [num_kv_heads, block_size, head_size]
            value_block: [num_kv_heads, block_size, head_size]
        """
        if key_block.device.type == "mps":
            torch.mps.synchronize()
            key_block = key_block.cpu()
            value_block = value_block.cpu()

        block_size_bytes = self.num_kv_heads * self.block_size * self.head_size * 2
        offset = block_idx * block_size_bytes

        self._update_buffer(self.key_buf, key_block.detach().numpy().tobytes(), offset)
        self._update_buffer(self.value_buf, value_block.detach().numpy().tobytes(), offset)

    def forward(
        self,
        query: torch.Tensor,          # [num_seqs, num_query_heads, head_size]
        key_cache: torch.Tensor,      # [num_blocks, num_kv_heads, block_size, head_size]
        value_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_size]
        block_table: torch.Tensor,    # [num_seqs, max_blocks_per_seq]
        seq_lens: torch.Tensor,       # [num_seqs]
        output: Optional[torch.Tensor] = None,
        skip_kv_copy: bool = False,   # Set True if KV cache already set via set_kv_cache()
    ) -> torch.Tensor:
        """Run PagedAttention.

        Args:
            skip_kv_copy: If True, assumes KV cache was set via set_kv_cache().
                          This provides significant speedup for decode.
        """
        num_seqs = query.shape[0]
        original_device = query.device
        is_mps = query.device.type == "mps"

        # Ensure batch buffers are large enough (auto-resize if needed)
        self._ensure_batch_buffers(num_seqs)

        if output is None:
            output = torch.zeros_like(query)

        # Sync MPS once at the start
        if is_mps:
            torch.mps.synchronize()

        # Fast path for CPU tensors (avoid repeated device checks)
        if is_mps:
            query_cpu = query.cpu()
            block_table_cpu = block_table.int().cpu()
            seq_lens_cpu = seq_lens.int().cpu()
        else:
            query_cpu = query.detach()
            block_table_cpu = block_table.int().detach()
            seq_lens_cpu = seq_lens.int().detach()

        # Update query (always changes) - direct numpy access
        self._update_buffer(self.query_buf, query_cpu.numpy().tobytes())

        # Update KV cache only if not skipping
        if not skip_kv_copy:
            if is_mps:
                key_cpu = key_cache.cpu()
                value_cpu = value_cache.cpu()
            else:
                key_cpu = key_cache.detach()
                value_cpu = value_cache.detach()

            # Ensure KV buffers are large enough
            kv_bytes = key_cpu.numel() * 2  # float16 = 2 bytes
            self._ensure_kv_buffer_size(kv_bytes)

            self._update_buffer(self.key_buf, key_cpu.numpy().tobytes())
            self._update_buffer(self.value_buf, value_cpu.numpy().tobytes())

        # Ensure block table buffer is large enough
        block_table_bytes = block_table_cpu.numel() * 4  # int32 = 4 bytes
        self._ensure_block_table_buffer_size(block_table_bytes)

        # Update block table and seq lens (small, always update)
        self._update_buffer(self.block_table_buf, block_table_cpu.numpy().tobytes())
        self._update_buffer(self.seq_lens_buf, seq_lens_cpu.numpy().tobytes())

        # max_seq_len already on CPU
        max_seq_len = int(seq_lens_cpu.max().item())

        # Update params - use pre-computed strides
        params_data = struct.pack(
            "fIIIIIIIIIIIIIIIIII",
            self.scale,
            num_seqs,
            max_seq_len,
            block_table.shape[1],
            self.head_size,
            self.block_size,
            self.num_kv_heads,
            self.num_query_heads,
            self.queries_per_kv,
            self._k_stride_block,
            self._k_stride_head,
            self._k_stride_token,
            self._v_stride_block,
            self._v_stride_head,
            self._v_stride_token,
            self._q_stride_token,
            self._q_stride_head,
            self._o_stride_token,
            self._o_stride_head,
        )
        self._update_buffer(self.params_buf, params_data)

        # Run kernel
        cmd = self.command_queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipeline)

        enc.setBuffer_offset_atIndex_(self.query_buf, 0, 0)
        enc.setBuffer_offset_atIndex_(self.key_buf, 0, 1)
        enc.setBuffer_offset_atIndex_(self.value_buf, 0, 2)
        enc.setBuffer_offset_atIndex_(self.block_table_buf, 0, 3)
        enc.setBuffer_offset_atIndex_(self.seq_lens_buf, 0, 4)
        enc.setBuffer_offset_atIndex_(self.output_buf, 0, 5)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 6)

        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            MTLSize(num_seqs, self.num_query_heads, 1),
            MTLSize(32, 1, 1)
        )

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        error = cmd.error()
        if error:
            raise RuntimeError(f"Metal kernel failed: {error}")

        # Copy result back - avoid extra bytes() copy
        output_size = output.numel() * output.element_size()
        output_ptr = self.output_buf.contents()
        # Direct memoryview access without bytes() intermediate
        output_mv = output_ptr.as_buffer(output_size)
        output_np = np.frombuffer(output_mv, dtype=np.float16).reshape(output.shape).copy()

        if original_device.type == "cpu":
            output.copy_(torch.from_numpy(output_np))
        else:
            output.copy_(torch.from_numpy(output_np).to(original_device))

        return output

    def forward_with_metal_buffers(
        self,
        query: torch.Tensor,          # [num_seqs, num_query_heads, head_size] CPU
        key_buffer,                   # MTLBuffer for key cache
        value_buffer,                 # MTLBuffer for value cache
        block_table: torch.Tensor,    # [num_seqs, max_blocks_per_seq] CPU int32
        seq_lens: torch.Tensor,       # [num_seqs] CPU int32
        output: torch.Tensor,         # [num_seqs, num_query_heads, head_size] CPU
    ) -> torch.Tensor:
        """Run PagedAttention with external MTLBuffers for KV cache.

        This method avoids copying KV cache - it uses the provided MTLBuffers
        directly. The KV cache must already be in the correct format:
        [num_blocks, num_kv_heads, block_size, head_size] as contiguous float16.

        Args:
            query: Query tensor on CPU [num_seqs, num_query_heads, head_size]
            key_buffer: MTLBuffer containing key cache
            value_buffer: MTLBuffer containing value cache
            block_table: Block table on CPU [num_seqs, max_blocks_per_seq]
            seq_lens: Sequence lengths on CPU [num_seqs]
            output: Output tensor on CPU [num_seqs, num_query_heads, head_size]

        Returns:
            Output tensor with attention results
        """
        num_seqs = query.shape[0]

        # Ensure batch buffers are large enough (auto-resize if needed)
        self._ensure_batch_buffers(num_seqs)

        # Update query buffer
        self._update_buffer(self.query_buf, query.numpy().tobytes())

        # Update block table and seq lens
        block_table_int = block_table.int()
        seq_lens_int = seq_lens.int()

        # Ensure block table buffer is large enough
        block_table_bytes = block_table_int.numel() * 4
        self._ensure_block_table_buffer_size(block_table_bytes)

        self._update_buffer(self.block_table_buf, block_table_int.numpy().tobytes())
        self._update_buffer(self.seq_lens_buf, seq_lens_int.numpy().tobytes())

        max_seq_len = int(seq_lens_int.max().item())

        # Update params
        params_data = struct.pack(
            "fIIIIIIIIIIIIIIIIII",
            self.scale,
            num_seqs,
            max_seq_len,
            block_table.shape[1],
            self.head_size,
            self.block_size,
            self.num_kv_heads,
            self.num_query_heads,
            self.queries_per_kv,
            self._k_stride_block,
            self._k_stride_head,
            self._k_stride_token,
            self._v_stride_block,
            self._v_stride_head,
            self._v_stride_token,
            self._q_stride_token,
            self._q_stride_head,
            self._o_stride_token,
            self._o_stride_head,
        )
        self._update_buffer(self.params_buf, params_data)

        # Run kernel with external KV buffers
        cmd = self.command_queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipeline)

        enc.setBuffer_offset_atIndex_(self.query_buf, 0, 0)
        enc.setBuffer_offset_atIndex_(key_buffer, 0, 1)      # External MTLBuffer
        enc.setBuffer_offset_atIndex_(value_buffer, 0, 2)    # External MTLBuffer
        enc.setBuffer_offset_atIndex_(self.block_table_buf, 0, 3)
        enc.setBuffer_offset_atIndex_(self.seq_lens_buf, 0, 4)
        enc.setBuffer_offset_atIndex_(self.output_buf, 0, 5)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 6)

        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            MTLSize(num_seqs, self.num_query_heads, 1),
            MTLSize(32, 1, 1)
        )

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        error = cmd.error()
        if error:
            raise RuntimeError(f"Metal kernel failed: {error}")

        # Copy result back
        output_size = output.numel() * output.element_size()
        output_ptr = self.output_buf.contents()
        output_mv = output_ptr.as_buffer(output_size)
        output_np = np.frombuffer(output_mv, dtype=np.float16).reshape(output.shape).copy()
        output.copy_(torch.from_numpy(output_np))

        return output


import ctypes
from pathlib import Path

# Load native KV write library
_native_lib_path = Path(__file__).parent.parent / "native" / "libkv_write.dylib"
_kv_lib = None
_native_write_kv_batch = None
_native_write_kv_single = None

def _load_native_lib():
    """Lazy load native library."""
    global _kv_lib, _native_write_kv_batch, _native_write_kv_single
    if _kv_lib is not None:
        return

    if not _native_lib_path.exists():
        raise RuntimeError(f"Native KV library not found: {_native_lib_path}")

    _kv_lib = ctypes.CDLL(str(_native_lib_path))

    # metal_write_kv_batch signature
    _native_write_kv_batch = _kv_lib.metal_write_kv_batch
    _native_write_kv_batch.argtypes = [
        ctypes.c_void_p,  # key_dst
        ctypes.c_void_p,  # value_dst
        ctypes.c_void_p,  # key_src
        ctypes.c_void_p,  # value_src
        ctypes.c_void_p,  # block_ids
        ctypes.c_void_p,  # token_offsets
        ctypes.c_int,     # num_tokens
        ctypes.c_int,     # num_kv_heads
        ctypes.c_int,     # head_size
        ctypes.c_int,     # stride_block
        ctypes.c_int,     # stride_head
        ctypes.c_int,     # stride_token
    ]
    _native_write_kv_batch.restype = None

    # metal_write_kv_single signature (optimized for decode)
    _native_write_kv_single = _kv_lib.metal_write_kv_single
    _native_write_kv_single.argtypes = [
        ctypes.c_void_p,  # key_dst
        ctypes.c_void_p,  # value_dst
        ctypes.c_void_p,  # key_src
        ctypes.c_void_p,  # value_src
        ctypes.c_int64,   # block_id
        ctypes.c_int64,   # token_offset
        ctypes.c_int,     # num_kv_heads
        ctypes.c_int,     # head_size
        ctypes.c_int,     # stride_block
        ctypes.c_int,     # stride_head
        ctypes.c_int,     # stride_token
    ]
    _native_write_kv_single.restype = None


def metal_write_kv_batch(
    key_mv: memoryview,
    value_mv: memoryview,
    key_np: np.ndarray,
    value_np: np.ndarray,
    block_ids: np.ndarray,
    token_offsets: np.ndarray,
    num_tokens: int,
    num_kv_heads: int,
    head_size: int,
    stride_block: int,
    stride_head: int,
    stride_token: int,
) -> None:
    """Native KV write - ALL loops in C, zero Python iteration.

    This function calls native C code via ctypes for maximum performance.
    The C function handles all token/head iteration with direct memcpy.

    Layout: [num_blocks, num_kv_heads, block_size, head_size]
    """
    # Ensure native library is loaded
    _load_native_lib()

    # Ensure data is contiguous float16
    if not key_np.flags['C_CONTIGUOUS'] or key_np.dtype != np.float16:
        key_np = np.ascontiguousarray(key_np, dtype=np.float16)
    if not value_np.flags['C_CONTIGUOUS'] or value_np.dtype != np.float16:
        value_np = np.ascontiguousarray(value_np, dtype=np.float16)

    # Ensure block_ids and token_offsets are contiguous int64
    if not block_ids.flags['C_CONTIGUOUS'] or block_ids.dtype != np.int64:
        block_ids = np.ascontiguousarray(block_ids, dtype=np.int64)
    if not token_offsets.flags['C_CONTIGUOUS'] or token_offsets.dtype != np.int64:
        token_offsets = np.ascontiguousarray(token_offsets, dtype=np.int64)

    # Get raw pointers
    key_dst_ptr = ctypes.addressof(ctypes.c_char.from_buffer(key_mv))
    value_dst_ptr = ctypes.addressof(ctypes.c_char.from_buffer(value_mv))

    # For single token (decode), use optimized single-token function
    if num_tokens == 1:
        _native_write_kv_single(
            key_dst_ptr,
            value_dst_ptr,
            key_np.ctypes.data,
            value_np.ctypes.data,
            int(block_ids[0]),
            int(token_offsets[0]),
            num_kv_heads,
            head_size,
            stride_block,
            stride_head,
            stride_token,
        )
    else:
        # Multi-token path (prefill)
        _native_write_kv_batch(
            key_dst_ptr,
            value_dst_ptr,
            key_np.ctypes.data,
            value_np.ctypes.data,
            block_ids.ctypes.data,
            token_offsets.ctypes.data,
            num_tokens,
            num_kv_heads,
            head_size,
            stride_block,
            stride_head,
            stride_token,
        )


def is_metal_available() -> bool:
    """Check if Metal is available."""
    try:
        device = MTLCreateSystemDefaultDevice()
        return device is not None and torch.backends.mps.is_available()
    except Exception:
        return False

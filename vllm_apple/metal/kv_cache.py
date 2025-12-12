# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal KV Cache Management.

This module provides Metal-native KV cache management for vLLM on Apple Silicon.
The KV cache is stored in unified memory as MTLBuffer, accessible directly
by the Metal compute kernel without data copies.

Key components:
- MetalKVCache: Manages MTLBuffer allocation and layout for KV cache
- MetalBlockAllocator: Maps logical block IDs to physical offsets

Memory layout:
- KV cache is stored as two MTLBuffers (key_buffer, value_buffer)
- Each buffer: [num_blocks * num_kv_heads * block_size * head_size] flat float16
- Metal kernel uses strides to navigate the cache
"""

from typing import Optional
import numpy as np

from Metal import (
    MTLCreateSystemDefaultDevice,
    MTLResourceStorageModeShared,
)


class MetalBlockAllocator:
    """Block allocator mapping logical block IDs to physical offsets.

    This allocator tracks which blocks are allocated and maps block_id
    to the physical offset in the MTLBuffer.

    Attributes:
        num_blocks: Total number of blocks in the cache
        block_size_bytes: Size of each block in bytes
        free_blocks: Set of free block IDs
        allocated_blocks: Set of allocated block IDs
    """

    def __init__(
        self,
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_size: int,
        dtype_size: int = 2,  # float16 = 2 bytes
    ):
        """Initialize block allocator.

        Args:
            num_blocks: Total number of blocks
            num_kv_heads: Number of KV heads
            block_size: Number of tokens per block
            head_size: Size of each head
            dtype_size: Size of dtype in bytes (2 for float16)
        """
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_size = head_size
        self.dtype_size = dtype_size

        # Block size in elements and bytes
        self.block_elements = num_kv_heads * block_size * head_size
        self.block_size_bytes = self.block_elements * dtype_size

        # Track free and allocated blocks
        self.free_blocks = set(range(num_blocks))
        self.allocated_blocks: set[int] = set()

    def allocate_block(self) -> Optional[int]:
        """Allocate a single block.

        Returns:
            Block ID if successful, None if no free blocks
        """
        if not self.free_blocks:
            return None

        block_id = self.free_blocks.pop()
        self.allocated_blocks.add(block_id)
        return block_id

    def allocate_blocks(self, num_blocks: int) -> list[int]:
        """Allocate multiple blocks.

        Args:
            num_blocks: Number of blocks to allocate

        Returns:
            List of block IDs, may be shorter than requested if not enough free blocks
        """
        blocks = []
        for _ in range(num_blocks):
            block_id = self.allocate_block()
            if block_id is None:
                break
            blocks.append(block_id)
        return blocks

    def free_block(self, block_id: int) -> None:
        """Free a single block.

        Args:
            block_id: Block ID to free
        """
        if block_id in self.allocated_blocks:
            self.allocated_blocks.remove(block_id)
            self.free_blocks.add(block_id)

    def free_blocks(self, block_ids: list[int]) -> None:
        """Free multiple blocks.

        Args:
            block_ids: List of block IDs to free
        """
        for block_id in block_ids:
            self.free_block(block_id)

    def get_block_offset(self, block_id: int) -> int:
        """Get byte offset for a block in the MTLBuffer.

        Args:
            block_id: Logical block ID

        Returns:
            Byte offset in the buffer
        """
        return block_id * self.block_size_bytes

    def get_num_free_blocks(self) -> int:
        """Get number of free blocks."""
        return len(self.free_blocks)

    def get_num_allocated_blocks(self) -> int:
        """Get number of allocated blocks."""
        return len(self.allocated_blocks)

    def reset(self) -> None:
        """Reset allocator, freeing all blocks."""
        self.free_blocks = set(range(self.num_blocks))
        self.allocated_blocks.clear()


class MetalKVCache:
    """Metal KV cache with MTLBuffer storage.

    This class manages KV cache in Metal unified memory. The cache is stored
    as MTLBuffer which can be directly accessed by Metal compute kernels
    without copying data.

    Memory layout (per buffer):
    - Shape: [num_blocks, num_kv_heads, block_size, head_size]
    - Stored as contiguous float16 in unified memory

    Attributes:
        device: Metal device
        key_buffer: MTLBuffer for key cache
        value_buffer: MTLBuffer for value cache
        allocator: Block allocator
    """

    def __init__(
        self,
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_size: int,
        num_layers: int = 1,
    ):
        """Initialize Metal KV cache.

        Args:
            num_blocks: Number of blocks per layer
            num_kv_heads: Number of KV heads
            block_size: Tokens per block
            head_size: Size of each head
            num_layers: Number of attention layers
        """
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_size = head_size
        self.num_layers = num_layers
        self.dtype_size = 2  # float16

        # Create Metal device
        self.metal_device = MTLCreateSystemDefaultDevice()
        if self.metal_device is None:
            raise RuntimeError("Failed to create Metal device")

        # Calculate buffer size
        self.elements_per_layer = num_blocks * num_kv_heads * block_size * head_size
        self.bytes_per_layer = self.elements_per_layer * self.dtype_size

        # Allocate MTLBuffers for each layer
        self.key_buffers: list = []
        self.value_buffers: list = []
        self._allocate_buffers()

        # Create allocators for each layer
        self.allocators = [
            MetalBlockAllocator(
                num_blocks=num_blocks,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                head_size=head_size,
            )
            for _ in range(num_layers)
        ]

        # Pre-compute strides for Metal kernel
        self.strides = {
            'block': num_kv_heads * block_size * head_size,
            'head': block_size * head_size,
            'token': head_size,
            'element': 1,
        }

    def _allocate_buffers(self) -> None:
        """Allocate MTLBuffers for KV cache."""
        for layer_idx in range(self.num_layers):
            key_buf = self.metal_device.newBufferWithLength_options_(
                self.bytes_per_layer, MTLResourceStorageModeShared
            )
            value_buf = self.metal_device.newBufferWithLength_options_(
                self.bytes_per_layer, MTLResourceStorageModeShared
            )

            if key_buf is None or value_buf is None:
                raise RuntimeError(f"Failed to allocate MTLBuffer for layer {layer_idx}")

            # Zero-initialize buffers
            self._zero_buffer(key_buf)
            self._zero_buffer(value_buf)

            self.key_buffers.append(key_buf)
            self.value_buffers.append(value_buf)

    def _zero_buffer(self, buffer) -> None:
        """Zero-initialize an MTLBuffer."""
        contents = buffer.contents()
        mv = contents.as_buffer(buffer.length())
        mv[:] = bytes(buffer.length())

    def get_buffers(self, layer_idx: int) -> tuple:
        """Get MTLBuffers for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (key_buffer, value_buffer)
        """
        return self.key_buffers[layer_idx], self.value_buffers[layer_idx]

    def get_allocator(self, layer_idx: int) -> MetalBlockAllocator:
        """Get allocator for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Block allocator for the layer
        """
        return self.allocators[layer_idx]

    def allocate_blocks_for_sequence(self, num_tokens: int, layer_idx: int = 0) -> list[int]:
        """Allocate blocks for a sequence.

        Args:
            num_tokens: Number of tokens in the sequence
            layer_idx: Layer index (default 0, same allocation for all layers)

        Returns:
            List of allocated block IDs
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return self.allocators[layer_idx].allocate_blocks(num_blocks_needed)

    def write_kv(
        self,
        layer_idx: int,
        block_id: int,
        token_offset: int,
        key: np.ndarray,  # [num_kv_heads, head_size]
        value: np.ndarray,  # [num_kv_heads, head_size]
    ) -> None:
        """Write key-value pair to cache with correct strided layout.

        Memory layout: [num_blocks, num_kv_heads, block_size, head_size]

        For a given (block_id, token_offset), we write each head separately
        because stride_head = block_size * head_size (non-contiguous for heads).

        Args:
            layer_idx: Layer index
            block_id: Block ID
            token_offset: Token offset within block (0 to block_size-1)
            key: Key tensor [num_kv_heads, head_size]
            value: Value tensor [num_kv_heads, head_size]
        """
        if token_offset >= self.block_size:
            raise ValueError(f"token_offset {token_offset} >= block_size {self.block_size}")

        key_f16 = key.astype(np.float16)
        value_f16 = value.astype(np.float16)

        # Get buffer memoryviews
        key_buf = self.key_buffers[layer_idx]
        value_buf = self.value_buffers[layer_idx]
        key_mv = key_buf.contents().as_buffer(key_buf.length())
        value_mv = value_buf.contents().as_buffer(value_buf.length())

        # Write each head separately (non-contiguous layout)
        # Layout: [block, head, token, dim]
        # offset = block_id * stride_block + head_idx * stride_head + token_offset * stride_token
        stride_block = self.strides['block']
        stride_head = self.strides['head']
        stride_token = self.strides['token']
        head_bytes = self.head_size * self.dtype_size

        for head_idx in range(self.num_kv_heads):
            element_offset = (
                block_id * stride_block +
                head_idx * stride_head +
                token_offset * stride_token
            )
            byte_offset = element_offset * self.dtype_size

            # Write key and value for this head
            key_mv[byte_offset:byte_offset + head_bytes] = key_f16[head_idx, :].tobytes()
            value_mv[byte_offset:byte_offset + head_bytes] = value_f16[head_idx, :].tobytes()

    def _write_to_buffer(self, buffer, offset: int, data: bytes) -> None:
        """Write data to MTLBuffer at offset."""
        contents = buffer.contents()
        mv = contents.as_buffer(buffer.length())
        mv[offset:offset + len(data)] = data

    def get_buffer_pointer(self, layer_idx: int, is_key: bool) -> int:
        """Get raw pointer to buffer for Metal kernel.

        Args:
            layer_idx: Layer index
            is_key: True for key buffer, False for value buffer

        Returns:
            Buffer contents pointer
        """
        buffer = self.key_buffers[layer_idx] if is_key else self.value_buffers[layer_idx]
        return buffer.contents()

    def get_cache_info(self) -> dict:
        """Get cache statistics."""
        total_allocated = sum(a.get_num_allocated_blocks() for a in self.allocators)
        total_free = sum(a.get_num_free_blocks() for a in self.allocators)

        return {
            'num_layers': self.num_layers,
            'num_blocks_per_layer': self.num_blocks,
            'total_blocks': self.num_blocks * self.num_layers,
            'allocated_blocks': total_allocated,
            'free_blocks': total_free,
            'bytes_per_layer': self.bytes_per_layer,
            'total_bytes': self.bytes_per_layer * self.num_layers * 2,  # K and V
            'total_gb': (self.bytes_per_layer * self.num_layers * 2) / (1024**3),
        }

    def reset(self) -> None:
        """Reset cache, freeing all blocks and zeroing buffers."""
        for layer_idx in range(self.num_layers):
            self.allocators[layer_idx].reset()
            self._zero_buffer(self.key_buffers[layer_idx])
            self._zero_buffer(self.value_buffers[layer_idx])


class MetalKVCacheManager:
    """Global manager for MetalKVCache instances across layers.

    This singleton manages KV cache for all attention layers, providing
    a unified interface for the Metal attention backend.
    """

    _instance: Optional['MetalKVCacheManager'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if MetalKVCacheManager._initialized:
            return
        MetalKVCacheManager._initialized = True

        self.metal_device = MTLCreateSystemDefaultDevice()
        if self.metal_device is None:
            raise RuntimeError("Failed to create Metal device")

        self.caches: dict[str, MetalKVCache] = {}  # layer_name -> cache
        self.layer_to_cache: dict[str, MetalKVCache] = {}

        # Global config (set during initialization)
        self.num_blocks: int = 0
        self.block_size: int = 16
        self.dtype_size: int = 2  # float16

    def initialize(
        self,
        num_blocks: int,
        block_size: int,
        layer_specs: dict[str, tuple[int, int]],  # layer_name -> (num_kv_heads, head_size)
    ) -> None:
        """Initialize KV caches for all layers.

        Args:
            num_blocks: Number of blocks per layer
            block_size: Tokens per block
            layer_specs: Dict mapping layer_name to (num_kv_heads, head_size)
        """
        self.num_blocks = num_blocks
        self.block_size = block_size

        for layer_name, (num_kv_heads, head_size) in layer_specs.items():
            cache = MetalKVCache(
                num_blocks=num_blocks,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                head_size=head_size,
                num_layers=1,
            )
            self.caches[layer_name] = cache
            self.layer_to_cache[layer_name] = cache

    def get_cache(self, layer_name: str) -> Optional[MetalKVCache]:
        """Get KV cache for a layer."""
        return self.caches.get(layer_name)

    def get_kv_buffers(self, layer_name: str) -> tuple:
        """Get key and value MTLBuffers for a layer.

        Returns:
            (key_buffer, value_buffer) MTLBuffer tuple
        """
        cache = self.caches.get(layer_name)
        if cache is None:
            raise ValueError(f"No cache for layer {layer_name}")
        return cache.get_buffers(0)

    def write_kv_token(
        self,
        layer_name: str,
        slot_idx: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Write key-value for a single token to cache.

        Args:
            layer_name: Layer name
            slot_idx: Absolute slot index (block_id * block_size + token_offset)
            key: Key tensor [num_kv_heads, head_size]
            value: Value tensor [num_kv_heads, head_size]
        """
        cache = self.caches.get(layer_name)
        if cache is None:
            raise ValueError(f"No cache for layer {layer_name}")

        block_id = slot_idx // self.block_size
        token_offset = slot_idx % self.block_size

        cache.write_kv(
            layer_idx=0,
            block_id=block_id,
            token_offset=token_offset,
            key=key,
            value=value,
        )

    def reset(self) -> None:
        """Reset all caches."""
        for cache in self.caches.values():
            cache.reset()

    def is_initialized(self) -> bool:
        """Check if manager is initialized with caches."""
        return len(self.caches) > 0


def get_kv_cache_manager() -> MetalKVCacheManager:
    """Get the global KV cache manager instance."""
    return MetalKVCacheManager()


def is_metal_available() -> bool:
    """Check if Metal is available."""
    try:
        device = MTLCreateSystemDefaultDevice()
        return device is not None
    except Exception:
        return False

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal Block Allocator for KV Cache.

This module provides block allocation for Metal KV cache management.
The allocator maps logical block IDs to physical offsets in MTLBuffer.

Memory layout:
- KV cache layout: [num_blocks, num_kv_heads, block_size, head_size]
- stride_block = num_kv_heads * block_size * head_size
- stride_head = block_size * head_size
- stride_token = head_size
"""

from typing import Optional


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
        self._free_blocks = set(range(num_blocks))
        self._allocated_blocks: set[int] = set()

        # Strides for Metal kernel (in elements, not bytes)
        self.strides = {
            'block': num_kv_heads * block_size * head_size,
            'head': block_size * head_size,
            'token': head_size,
            'element': 1,
        }

    def allocate_block(self) -> Optional[int]:
        """Allocate a single block.

        Returns:
            Block ID if successful, None if no free blocks
        """
        if not self._free_blocks:
            return None

        block_id = self._free_blocks.pop()
        self._allocated_blocks.add(block_id)
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
        if block_id in self._allocated_blocks:
            self._allocated_blocks.remove(block_id)
            self._free_blocks.add(block_id)

    def free_blocks_list(self, block_ids: list[int]) -> None:
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

    def get_block_element_offset(self, block_id: int) -> int:
        """Get element offset for a block (for Metal kernel).

        Args:
            block_id: Logical block ID

        Returns:
            Element offset in the buffer
        """
        return block_id * self.block_elements

    def get_num_free_blocks(self) -> int:
        """Get number of free blocks."""
        return len(self._free_blocks)

    def get_num_allocated_blocks(self) -> int:
        """Get number of allocated blocks."""
        return len(self._allocated_blocks)

    @property
    def free_blocks(self) -> set[int]:
        """Get set of free block IDs (read-only copy)."""
        return self._free_blocks.copy()

    @property
    def allocated_blocks(self) -> set[int]:
        """Get set of allocated block IDs (read-only copy)."""
        return self._allocated_blocks.copy()

    def reset(self) -> None:
        """Reset allocator, freeing all blocks."""
        self._free_blocks = set(range(self.num_blocks))
        self._allocated_blocks.clear()

    def __repr__(self) -> str:
        return (
            f"MetalBlockAllocator(num_blocks={self.num_blocks}, "
            f"allocated={len(self._allocated_blocks)}, "
            f"free={len(self._free_blocks)})"
        )

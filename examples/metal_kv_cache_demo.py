#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demo of Metal KV cache with zero-copy architecture.

This example demonstrates the Metal KV cache which uses unified
memory (MTLBuffer) for zero-copy GPU/CPU access.

Usage:
    python examples/metal_kv_cache_demo.py

Requirements:
    - Apple Silicon Mac (M1/M2/M3/M4)
    - vllm-apple plugin installed
"""

import numpy as np

# Check Metal availability
try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    print("ERROR: Metal not available. Install pyobjc-framework-Metal.")
    exit(1)

from vllm_apple.metal.kv_cache import MetalKVCache
from vllm_apple.metal.block_allocator import MetalBlockAllocator


def demo_kv_cache():
    """Demonstrate Metal KV cache functionality."""
    print("=" * 60)
    print("Metal KV Cache Demo")
    print("=" * 60)

    # Configuration
    num_blocks = 32
    num_kv_heads = 4
    block_size = 16
    head_size = 128
    num_layers = 2

    # Create KV cache
    print(f"\nCreating MetalKVCache:")
    print(f"  num_blocks:   {num_blocks}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  block_size:   {block_size}")
    print(f"  head_size:    {head_size}")
    print(f"  num_layers:   {num_layers}")

    cache = MetalKVCache(
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_size=head_size,
        num_layers=num_layers,
    )

    # Show memory layout
    print(f"\nMemory Layout (strides in elements):")
    print(f"  block:   {cache.strides['block']:,}")
    print(f"  head:    {cache.strides['head']:,}")
    print(f"  token:   {cache.strides['token']:,}")
    print(f"  element: {cache.strides['element']}")

    # Calculate memory usage
    bytes_per_buffer = num_blocks * num_kv_heads * block_size * head_size * 2  # float16
    total_bytes = bytes_per_buffer * 2 * num_layers  # K + V buffers
    print(f"\nMemory Usage:")
    print(f"  Per buffer:  {bytes_per_buffer / 1024 / 1024:.2f} MB")
    print(f"  Total:       {total_bytes / 1024 / 1024:.2f} MB")

    # Write some test data
    print(f"\nWriting test data to block 0, token 0...")
    key = np.ones((num_kv_heads, head_size), dtype=np.float16) * 1.5
    value = np.ones((num_kv_heads, head_size), dtype=np.float16) * 2.5

    cache.write_kv(
        layer_idx=0,
        block_id=0,
        token_offset=0,
        key=key,
        value=value,
    )

    # Verify data was written
    key_buf, val_buf = cache.get_buffers(0)
    print(f"  Key buffer length:   {key_buf.length():,} bytes")
    print(f"  Value buffer length: {val_buf.length():,} bytes")

    # Read back first element to verify
    key_mv = key_buf.contents().as_buffer(key_buf.length())
    first_value = np.frombuffer(key_mv[:2], dtype=np.float16)[0]
    print(f"  First key value:     {first_value} (expected: 1.5)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def demo_block_allocator():
    """Demonstrate block allocator functionality."""
    print("\n" + "=" * 60)
    print("Block Allocator Demo")
    print("=" * 60)

    allocator = MetalBlockAllocator(
        num_blocks=64,
        num_kv_heads=4,
        block_size=16,
        head_size=128,
    )

    print(f"\nInitial state:")
    print(f"  Free blocks:      {allocator.get_num_free_blocks()}")
    print(f"  Allocated blocks: {allocator.get_num_allocated_blocks()}")

    # Allocate some blocks
    print(f"\nAllocating 10 blocks...")
    block_ids = allocator.allocate_blocks(10)
    print(f"  Allocated block IDs: {block_ids}")
    print(f"  Free blocks:      {allocator.get_num_free_blocks()}")
    print(f"  Allocated blocks: {allocator.get_num_allocated_blocks()}")

    # Free some blocks
    print(f"\nFreeing first 5 blocks...")
    allocator.free_blocks_list(block_ids[:5])
    print(f"  Free blocks:      {allocator.get_num_free_blocks()}")
    print(f"  Allocated blocks: {allocator.get_num_allocated_blocks()}")

    # Show block offsets
    print(f"\nBlock offsets (in bytes):")
    for i in range(3):
        offset = allocator.get_block_offset(i)
        print(f"  Block {i}: {offset:,} bytes")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_kv_cache()
    demo_block_allocator()

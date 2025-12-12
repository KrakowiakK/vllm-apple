# SPDX-License-Identifier: Apache-2.0
"""Test memory calculation without full vLLM stack."""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"

import torch
import psutil

GiB_bytes = 1024 * 1024 * 1024


def test_memory_calculation():
    """Test memory calculation logic."""
    print("\n" + "="*60)
    print("  Memory Calculation Test")
    print("="*60)

    # Simulate what apple_worker does
    total_memory = psutil.virtual_memory().total
    gpu_memory_utilization = 0.5

    # Model memory (simulated - GPT-2 is ~500MB)
    model_memory_usage = int(0.5 * GiB_bytes)  # 500 MB

    # Activation memory (simulated)
    activation_memory_bytes = int(0.03 * GiB_bytes)  # ~30 MB

    # Default max KV cache
    DEFAULT_MAX_KV_CACHE_GIB = 16
    max_kv_cache_bytes = DEFAULT_MAX_KV_CACHE_GIB * GiB_bytes

    # Calculate requested memory from utilization
    requested_memory_from_utilization = int(
        total_memory * gpu_memory_utilization
    )

    # Check environment variable
    from vllm import envs
    print(f"\n  VLLM_CPU_KVCACHE_SPACE env: {envs.VLLM_CPU_KVCACHE_SPACE}")

    if envs.VLLM_CPU_KVCACHE_SPACE is not None:
        requested_memory = int(envs.VLLM_CPU_KVCACHE_SPACE * GiB_bytes)
        print(f"  Using explicit VLLM_CPU_KVCACHE_SPACE: {envs.VLLM_CPU_KVCACHE_SPACE} GiB")
    else:
        requested_memory = min(
            requested_memory_from_utilization,
            max_kv_cache_bytes + model_memory_usage,
        )
        print(f"  Using calculated limit")

    # Buffer
    buffer_bytes = 500 * (1 << 20)  # 500 MiB

    non_kv_cache_memory = (
        model_memory_usage + activation_memory_bytes + buffer_bytes
    )

    available_kv_cache_memory_bytes = max(
        requested_memory - non_kv_cache_memory,
        0,
    )

    print(f"\n  Memory calculation:")
    print(f"    Total memory: {total_memory / GiB_bytes:.2f} GiB")
    print(f"    GPU utilization: {gpu_memory_utilization}")
    print(f"    Requested from utilization: {requested_memory_from_utilization / GiB_bytes:.2f} GiB")
    print(f"    Max KV cache: {max_kv_cache_bytes / GiB_bytes:.2f} GiB")
    print(f"    Final requested memory: {requested_memory / GiB_bytes:.2f} GiB")
    print(f"    Model memory: {model_memory_usage / GiB_bytes:.2f} GiB")
    print(f"    Activation memory: {activation_memory_bytes / GiB_bytes:.3f} GiB")
    print(f"    Buffer: {buffer_bytes / GiB_bytes:.2f} GiB")
    print(f"    Non-KV cache memory: {non_kv_cache_memory / GiB_bytes:.2f} GiB")
    print(f"    Available for KV cache: {available_kv_cache_memory_bytes / GiB_bytes:.2f} GiB")

    # Calculate how many blocks this gives
    # GPT-2 has 12 layers, 12 heads, head_dim=64
    # Block size = 16 tokens
    # KV cache per block per layer = 2 * head_dim * num_kv_heads * dtype_size * block_size
    # = 2 * 64 * 12 * 2 (fp16) * 16 = 49152 bytes per layer per block
    num_layers = 12
    bytes_per_block_per_layer = 2 * 64 * 12 * 2 * 16  # 49152
    bytes_per_block = bytes_per_block_per_layer * num_layers

    num_blocks = available_kv_cache_memory_bytes // bytes_per_block
    total_tokens = num_blocks * 16

    print(f"\n  KV cache block calculation:")
    print(f"    Bytes per block (all layers): {bytes_per_block} ({bytes_per_block / 1024:.1f} KB)")
    print(f"    Number of blocks: {num_blocks}")
    print(f"    Total tokens: {total_tokens}")

    if available_kv_cache_memory_bytes > GiB_bytes:
        print(f"\n  [PASS] KV cache memory looks reasonable")
    else:
        print(f"\n  [WARN] KV cache memory might be too small")


if __name__ == "__main__":
    test_memory_calculation()

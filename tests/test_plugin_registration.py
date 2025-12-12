# SPDX-License-Identifier: Apache-2.0
"""Test vllm-apple plugin registration."""

import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_plugin_register():
    """Test that the plugin registers correctly."""
    from vllm_apple import register

    result = register()
    print(f"register() returned: {result}")

    assert result is not None, "Plugin should register on MPS-capable system"
    assert result == "vllm_apple.platform.ApplePlatform", f"Expected ApplePlatform path, got {result}"
    print("✓ Plugin registration OK")


def test_platform_import():
    """Test that ApplePlatform can be imported."""
    from vllm_apple.platform import ApplePlatform

    print(f"ApplePlatform class: {ApplePlatform}")
    assert ApplePlatform.device_name == "mps"
    assert ApplePlatform.device_type == "mps"
    print("✓ Platform import OK")


def test_worker_import():
    """Test that AppleWorker can be imported."""
    from vllm_apple.v1.worker.apple_worker import AppleWorker

    print(f"AppleWorker class: {AppleWorker}")
    print("✓ Worker import OK")


def test_model_runner_import():
    """Test that AppleModelRunner can be imported."""
    from vllm_apple.v1.worker.apple_model_runner import AppleModelRunner

    print(f"AppleModelRunner class: {AppleModelRunner}")
    print("✓ ModelRunner import OK")


def test_input_batch_import():
    """Test that AppleInputBatch can be imported."""
    from vllm_apple.v1.worker.apple_input_batch import AppleInputBatch

    print(f"AppleInputBatch class: {AppleInputBatch}")
    print("✓ InputBatch import OK")


def test_attention_backend_import():
    """Test that AppleAttentionBackend can be imported."""
    from vllm_apple.v1.attention.backends.apple_attn import AppleAttentionBackend

    print(f"AppleAttentionBackend class: {AppleAttentionBackend}")
    print("✓ AttentionBackend import OK")


def test_ops_import():
    """Test that Apple ops can be imported."""
    try:
        from vllm_apple.ops import apple_fused_moe
        print(f"apple_fused_moe module: {apple_fused_moe}")
        print("✓ Ops import OK")
    except ImportError as e:
        print(f"⚠ Ops import failed (may be expected): {e}")


def test_mps_available():
    """Test MPS availability."""
    import torch

    is_available = torch.backends.mps.is_available()
    is_built = torch.backends.mps.is_built()

    print(f"MPS available: {is_available}")
    print(f"MPS built: {is_built}")

    assert is_available, "MPS should be available on Apple Silicon"
    print("✓ MPS availability OK")


if __name__ == "__main__":
    print("=" * 60)
    print("vllm-apple Plugin Registration Tests")
    print("=" * 60)
    print()

    tests = [
        test_mps_available,
        test_plugin_register,
        test_platform_import,
        test_worker_import,
        test_model_runner_import,
        test_input_batch_import,
        test_attention_backend_import,
        test_ops_import,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

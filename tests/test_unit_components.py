# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm-apple plugin components.

Run with: python tests/test_unit_components.py
"""

import os
import sys

# Apply MPS patches before importing anything else
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_1_mps_available():
    """Test 1: Check if MPS is available."""
    print_header("Test 1: MPS Availability")

    has_mps = torch.backends.mps.is_available()
    is_built = torch.backends.mps.is_built()

    print(f"  MPS available: {has_mps}")
    print(f"  MPS built: {is_built}")

    if has_mps:
        # Test creating tensor on MPS
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device="mps")
            print(f"  Created tensor on MPS: {x.device}")
            print(f"  [PASS] MPS is working")
            return True
        except Exception as e:
            print(f"  [FAIL] Cannot create tensor on MPS: {e}")
            return False
    else:
        print(f"  [FAIL] MPS not available")
        return False


def test_2_plugin_registration():
    """Test 2: Check if plugin is properly registered."""
    print_header("Test 2: Plugin Registration")

    # Check entry points
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group='vllm.platform_plugins')
        plugin_names = [ep.name for ep in eps]
        print(f"  Registered platform plugins: {plugin_names}")

        if 'apple' in plugin_names:
            print(f"  [PASS] Apple plugin is registered")
            return True
        else:
            print(f"  [FAIL] Apple plugin not found in entry_points")
            return False
    except Exception as e:
        print(f"  [FAIL] Error checking entry_points: {e}")
        return False


def test_3_platform_detection():
    """Test 3: Check if ApplePlatform is detected."""
    print_header("Test 3: Platform Detection")

    try:
        from vllm.platforms import current_platform

        platform_name = type(current_platform).__name__
        platform_enum = current_platform._enum
        device_name = current_platform.device_name

        print(f"  Current platform: {platform_name}")
        print(f"  Platform enum: {platform_enum}")
        print(f"  Device name: {device_name}")

        if "Apple" in platform_name:
            print(f"  [PASS] ApplePlatform detected")
            return True
        else:
            print(f"  [WARN] Expected ApplePlatform, got {platform_name}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error detecting platform: {e}")
        return False


def test_4_attention_backend_import():
    """Test 4: Check if AppleAttentionBackend can be imported."""
    print_header("Test 4: Attention Backend Import")

    try:
        from vllm_apple.v1.attention.backends.apple_attn import AppleAttentionBackend

        print(f"  AppleAttentionBackend class: {AppleAttentionBackend}")
        print(f"  Backend name: {AppleAttentionBackend.get_name()}")
        print(f"  [PASS] AppleAttentionBackend imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_5_worker_import():
    """Test 5: Check if AppleWorker can be imported."""
    print_header("Test 5: Worker Import")

    try:
        from vllm_apple.v1.worker.apple_worker import AppleWorker

        print(f"  AppleWorker class: {AppleWorker}")
        print(f"  [PASS] AppleWorker imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_6_model_runner_import():
    """Test 6: Check if AppleModelRunner can be imported."""
    print_header("Test 6: Model Runner Import")

    try:
        from vllm_apple.v1.worker.apple_model_runner import AppleModelRunner

        print(f"  AppleModelRunner class: {AppleModelRunner}")
        print(f"  [PASS] AppleModelRunner imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_7_basic_mps_operations():
    """Test 7: Test basic MPS tensor operations."""
    print_header("Test 7: Basic MPS Operations")

    try:
        device = torch.device("mps")

        # Test basic tensor creation
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)

        # Test matmul
        c = torch.matmul(a, b)
        print(f"  Matmul result device: {c.device}")

        # Test softmax
        d = torch.softmax(a, dim=-1)
        print(f"  Softmax result device: {d.device}")

        # Test attention-like operation
        q = torch.randn(2, 8, 32, 64, device=device)  # [batch, heads, seq, head_dim]
        k = torch.randn(2, 8, 32, 64, device=device)
        v = torch.randn(2, 8, 32, 64, device=device)

        # Scaled dot-product attention
        scale = 1.0 / (64 ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        print(f"  Attention output device: {out.device}")
        print(f"  Attention output shape: {out.shape}")

        # Synchronize
        torch.mps.synchronize()

        print(f"  [PASS] Basic MPS operations work")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_8_platform_get_attn_backend():
    """Test 8: Test get_attn_backend_cls returns valid path."""
    print_header("Test 8: Platform get_attn_backend_cls")

    try:
        from vllm.platforms import current_platform

        # Call get_attn_backend_cls with typical parameters
        backend_path = current_platform.get_attn_backend_cls(
            selected_backend=None,
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype=None,
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )

        print(f"  Backend path: {backend_path}")

        # Try to import the backend
        from vllm.utils.import_utils import resolve_obj_by_qualname
        backend_cls = resolve_obj_by_qualname(backend_path)

        print(f"  Resolved backend class: {backend_cls}")
        print(f"  [PASS] get_attn_backend_cls works")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_9_memory_info():
    """Test 9: Test memory info functions."""
    print_header("Test 9: Memory Info")

    try:
        from vllm.platforms import current_platform
        import psutil

        # Get device total memory
        total_mem = current_platform.get_device_total_memory()
        free_mem, total_mem2 = current_platform.mem_get_info()

        print(f"  Device total memory: {total_mem / (1024**3):.2f} GiB")
        print(f"  mem_get_info: free={free_mem / (1024**3):.2f} GiB, total={total_mem2 / (1024**3):.2f} GiB")
        print(f"  System memory: {psutil.virtual_memory().total / (1024**3):.2f} GiB")

        print(f"  [PASS] Memory info works")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_10_torch_sdpa():
    """Test 10: Test PyTorch SDPA (Scaled Dot-Product Attention)."""
    print_header("Test 10: PyTorch SDPA on MPS")

    try:
        device = torch.device("mps")

        # Create Q, K, V tensors
        batch_size = 2
        num_heads = 8
        seq_len = 64
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

        # Test SDPA
        with torch.no_grad():
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )

        print(f"  SDPA output device: {out.device}")
        print(f"  SDPA output shape: {out.shape}")
        print(f"  SDPA output dtype: {out.dtype}")

        torch.mps.synchronize()

        print(f"  [PASS] SDPA works on MPS")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("  vllm-apple Plugin Unit Tests")
    print("="*60)

    tests = [
        ("MPS Availability", test_1_mps_available),
        ("Plugin Registration", test_2_plugin_registration),
        ("Platform Detection", test_3_platform_detection),
        ("Attention Backend Import", test_4_attention_backend_import),
        ("Worker Import", test_5_worker_import),
        ("Model Runner Import", test_6_model_runner_import),
        ("Basic MPS Operations", test_7_basic_mps_operations),
        ("Platform get_attn_backend_cls", test_8_platform_get_attn_backend),
        ("Memory Info", test_9_memory_info),
        ("PyTorch SDPA", test_10_torch_sdpa),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  [ERROR] Test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    failed = len(results) - passed

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed}/{len(results)} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

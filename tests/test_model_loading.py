# SPDX-License-Identifier: Apache-2.0
"""Test model loading to diagnose memory issues.

This test loads a very small model to check if weights go to MPS or CPU.
"""

import os
import sys
import gc

# Apply MPS patches before importing anything else
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"  # 4GB for testing

import torch
import psutil


def print_memory():
    """Print current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()

    print(f"  Process RSS: {mem_info.rss / (1024**3):.2f} GiB")
    print(f"  System used: {vm.used / (1024**3):.2f} GiB / {vm.total / (1024**3):.2f} GiB")
    print(f"  System available: {vm.available / (1024**3):.2f} GiB")


def check_tensor_devices(model):
    """Check which devices model parameters are on."""
    device_counts = {}
    total_params = 0
    total_bytes = 0

    for name, param in model.named_parameters():
        device = str(param.device)
        if device not in device_counts:
            device_counts[device] = {"count": 0, "bytes": 0}
        device_counts[device]["count"] += 1
        device_counts[device]["bytes"] += param.numel() * param.element_size()
        total_params += 1
        total_bytes += param.numel() * param.element_size()

    print(f"\n  Model parameter distribution:")
    for device, info in device_counts.items():
        pct = 100 * info["bytes"] / total_bytes if total_bytes > 0 else 0
        print(f"    {device}: {info['count']} params, {info['bytes'] / (1024**3):.3f} GiB ({pct:.1f}%)")

    print(f"  Total: {total_params} params, {total_bytes / (1024**3):.3f} GiB")
    return device_counts


def test_tiny_model_loading():
    """Test loading a tiny model to check memory behavior."""
    print("\n" + "="*60)
    print("  Test: Tiny Model Loading (GPT-2)")
    print("="*60)

    print("\n1. Initial memory state:")
    print_memory()

    print("\n2. Loading GPT-2 (small) model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"  # ~500MB model

    # Load on CPU first
    print("   Loading to CPU...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\n3. After loading to CPU:")
    print_memory()
    check_tensor_devices(model)

    # Move to MPS
    print("\n4. Moving model to MPS...")
    try:
        model = model.to("mps")
        torch.mps.synchronize()
        print("   Model moved to MPS")
    except Exception as e:
        print(f"   ERROR moving to MPS: {e}")
        return False

    print("\n5. After moving to MPS:")
    print_memory()
    device_info = check_tensor_devices(model)

    # Check if all params are on MPS
    mps_bytes = device_info.get("mps:0", {}).get("bytes", 0)
    cpu_bytes = device_info.get("cpu", {}).get("bytes", 0)

    if mps_bytes > 0 and cpu_bytes == 0:
        print("\n   [PASS] All model parameters are on MPS")
    elif mps_bytes > 0 and cpu_bytes > 0:
        print("\n   [WARN] Model has parameters on both MPS and CPU")
    else:
        print("\n   [FAIL] Model is not on MPS")

    # Test inference
    print("\n6. Testing inference...")
    inputs = tokenizer("Hello, I am", return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    result = tokenizer.decode(outputs[0])
    print(f"   Generated: {result[:80]}...")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()

    print("\n7. After cleanup:")
    print_memory()

    print("\n   [PASS] Tiny model test completed")
    return True


def test_model_runner_direct():
    """Test AppleModelRunner directly (without full vLLM stack)."""
    print("\n" + "="*60)
    print("  Test: AppleModelRunner Direct Instantiation")
    print("="*60)

    print("\n1. Initial memory state:")
    print_memory()

    print("\n2. Importing components...")
    try:
        from vllm.config import ModelConfig, CacheConfig, SchedulerConfig, ParallelConfig
        from vllm_apple.v1.worker.apple_model_runner import AppleModelRunner

        print("   Imports successful")
    except Exception as e:
        print(f"   [FAIL] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n3. After imports:")
    print_memory()

    # We can't easily instantiate ModelRunner without full config
    # but we can test the class exists and basic attributes
    print("\n4. Checking AppleModelRunner class:")
    print(f"   Class: {AppleModelRunner}")
    print(f"   Module: {AppleModelRunner.__module__}")

    print("\n   [PASS] AppleModelRunner check completed")
    return True


def main():
    """Run all model loading tests."""
    print("\n" + "="*60)
    print("  Model Loading Diagnostics")
    print("="*60)

    results = []

    # Test 1: Tiny model loading
    try:
        result = test_tiny_model_loading()
        results.append(("Tiny Model Loading", result))
    except Exception as e:
        print(f"\n  [ERROR] Test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Tiny Model Loading", False))

    # Test 2: ModelRunner direct
    try:
        result = test_model_runner_direct()
        results.append(("ModelRunner Direct", result))
    except Exception as e:
        print(f"\n  [ERROR] Test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ModelRunner Direct", False))

    # Summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    return all(r for _, r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

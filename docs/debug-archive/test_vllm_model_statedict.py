#!/usr/bin/env python3
"""Direct test to print vLLM model state dict keys."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '0'  # Disable engine


def main():
    print("=== Checking vLLM Model Structure ===")

    # Look at vLLM model class structure directly
    print("Looking at vLLM model class structure...")

    from vllm.model_executor.models.mistral import MistralForCausalLM, MistralModel
    print(f"MistralForCausalLM class: {MistralForCausalLM}")
    print(f"MistralModel class: {MistralModel}")

    # The key is to check if vLLM's MistralModel has packed_modules or similar
    print("\nChecking for packed modules...")
    if hasattr(MistralForCausalLM, 'packed_modules_mapping'):
        print(f"packed_modules_mapping: {MistralForCausalLM.packed_modules_mapping}")
    if hasattr(MistralForCausalLM, 'supported_lora_modules'):
        print(f"supported_lora_modules: {MistralForCausalLM.supported_lora_modules}")

    # Check the base model class
    for base in MistralForCausalLM.__mro__:
        if hasattr(base, 'packed_modules_mapping'):
            print(f"{base.__name__}.packed_modules_mapping: {base.packed_modules_mapping}")
            break

    # Look at MistralAttention
    from vllm.model_executor.models.mistral import MistralAttention
    print(f"\nMistralAttention: {MistralAttention}")

    # Check what linear layers it has
    import inspect
    source = inspect.getsource(MistralAttention.__init__)
    print("\nMistralAttention.__init__ source (key lines):")
    for line in source.split('\n'):
        if 'proj' in line.lower() or 'linear' in line.lower():
            print(f"  {line.strip()}")


if __name__ == "__main__":
    main()

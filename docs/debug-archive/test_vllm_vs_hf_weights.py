#!/usr/bin/env python3
"""Compare vLLM model state dict vs HuggingFace model state dict."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '0'  # Disable engine for testing

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = 'mistralai/Devstral-Small-2505'

    print("=== Loading HuggingFace Model ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    hf_state = hf_model.state_dict()

    print(f"\nHuggingFace state dict keys (layer 0):")
    hf_layer0_keys = sorted([k for k in hf_state.keys() if 'layers.0.' in k])
    for k in hf_layer0_keys:
        print(f"  {k}: {hf_state[k].shape}")

    # Check for fused vs separate
    print(f"\nHF patterns:")
    print(f"  Has qkv_proj: {any('qkv_proj' in k for k in hf_state.keys())}")
    print(f"  Has q_proj: {any('q_proj' in k and 'qkv_proj' not in k for k in hf_state.keys())}")
    print(f"  Has gate_up_proj: {any('gate_up_proj' in k for k in hf_state.keys())}")
    print(f"  Has gate_proj: {any('gate_proj' in k and 'gate_up_proj' not in k for k in hf_state.keys())}")

    print("\n=== Loading vLLM Model ===")
    from vllm import LLM

    llm = LLM(
        model=model_name,
        max_model_len=128,
        dtype='float16',
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Get the model from vLLM
    # In V1 engine, the model is in a separate worker process
    # But we can check the model's state dict through a different approach

    # Actually, let's check what the AppleModelRunner sees
    print("\nvLLM loaded. Checking model type...")

    # The model in vLLM goes through a transformation that fuses weights
    # Let's check by looking at what model architecture vLLM uses
    print(f"Model config type: {type(llm.llm_engine.model_config)}")
    print(f"HF config model_type: {llm.llm_engine.model_config.hf_config.model_type}")

    # Try to access the underlying model state
    # This might not work with V1 engine due to multiprocessing
    try:
        # V0 engine style
        if hasattr(llm.llm_engine, 'model_executor'):
            executor = llm.llm_engine.model_executor
            if hasattr(executor, 'driver_worker'):
                model = executor.driver_worker.model_runner.model
                vllm_state = model.state_dict()

                print(f"\nvLLM state dict keys (layer 0):")
                vllm_layer0_keys = sorted([k for k in vllm_state.keys() if 'layers.0.' in k])
                for k in vllm_layer0_keys:
                    print(f"  {k}: {vllm_state[k].shape}")

                print(f"\nvLLM patterns:")
                print(f"  Has qkv_proj: {any('qkv_proj' in k for k in vllm_state.keys())}")
                print(f"  Has q_proj: {any('q_proj' in k and 'qkv_proj' not in k for k in vllm_state.keys())}")
                print(f"  Has gate_up_proj: {any('gate_up_proj' in k for k in vllm_state.keys())}")
                print(f"  Has gate_proj: {any('gate_proj' in k and 'gate_up_proj' not in k for k in vllm_state.keys())}")
    except Exception as e:
        print(f"\nCouldn't access vLLM model state directly: {e}")

    print("\n=== Key Insight ===")
    print("vLLM transforms HuggingFace models during loading.")
    print("It fuses q_proj+k_proj+v_proj -> qkv_proj")
    print("It fuses gate_proj+up_proj -> gate_up_proj")
    print("")
    print("If the engine's weight_loader expects the HF format but receives")
    print("vLLM's fused format (or vice versa), the weights will be misaligned!")


if __name__ == "__main__":
    main()

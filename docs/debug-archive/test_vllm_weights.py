#!/usr/bin/env python3
"""Check what state dict vLLM produces (fused vs separate QKV)."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '0'  # Don't use engine

def main():
    from vllm import LLM

    llm = LLM(
        model='mistralai/Devstral-Small-2505',
        max_model_len=128,
        dtype='float16',
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Get the model from the engine core
    # This is the PyTorch model that vLLM uses
    model = None
    try:
        # Try to access model through various paths
        if hasattr(llm, 'model'):
            model = llm.model
        elif hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
            model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except Exception as e:
        print(f"Could not access model: {e}")
        return

    if model is None:
        print("Could not access model")
        return

    state_dict = model.state_dict()

    print("=== vLLM Model State Dict Keys (Layer 0) ===")
    layer0_keys = [k for k in sorted(state_dict.keys()) if 'layers.0.' in k]
    for key in layer0_keys:
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")

    # Check for fused vs separate
    has_fused_qkv = any('qkv_proj' in k for k in state_dict.keys())
    has_separate_qkv = any('q_proj' in k and 'qkv_proj' not in k for k in state_dict.keys())
    has_fused_gate_up = any('gate_up_proj' in k for k in state_dict.keys())
    has_separate_gate_up = any('gate_proj' in k and 'gate_up_proj' not in k for k in state_dict.keys())

    print(f"\n=== Weight Pattern ===")
    print(f"  Fused qkv_proj: {has_fused_qkv}")
    print(f"  Separate q/k/v_proj: {has_separate_qkv}")
    print(f"  Fused gate_up_proj: {has_fused_gate_up}")
    print(f"  Separate gate/up_proj: {has_separate_gate_up}")

if __name__ == "__main__":
    main()

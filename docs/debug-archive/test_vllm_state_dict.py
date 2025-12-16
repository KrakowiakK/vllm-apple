#!/usr/bin/env python3
"""Check vLLM's internal state dict structure."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '0'

def main():
    from vllm import LLM

    print("Loading vLLM model to check state dict...")

    llm = LLM(
        model='mistralai/Devstral-Small-2505',
        max_model_len=128,
        dtype='float16',
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Access the internal model
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        state_dict = model.state_dict()

        print(f"\n=== vLLM State Dict (Layer 0) ===")
        layer0_keys = sorted([k for k in state_dict.keys() if 'layers.0.' in k])
        for key in layer0_keys:
            shape = state_dict[key].shape
            print(f"  {key}: {shape}")

        # Check for fused patterns
        has_qkv_proj = any('qkv_proj' in k for k in state_dict.keys())
        has_separate_qkv = any('q_proj' in k and 'qkv_proj' not in k for k in state_dict.keys())
        has_gate_up_proj = any('gate_up_proj' in k for k in state_dict.keys())
        has_separate_gate_up = any('gate_proj' in k and 'gate_up_proj' not in k for k in state_dict.keys())

        print(f"\n=== Weight Patterns ===")
        print(f"  Has fused qkv_proj: {has_qkv_proj}")
        print(f"  Has separate q/k/v_proj: {has_separate_qkv}")
        print(f"  Has fused gate_up_proj: {has_gate_up_proj}")
        print(f"  Has separate gate/up_proj: {has_separate_gate_up}")

        # Check QKV dimensions if fused
        if has_qkv_proj:
            qkv_key = [k for k in state_dict.keys() if 'layers.0' in k and 'qkv_proj' in k][0]
            qkv_shape = state_dict[qkv_key].shape
            print(f"\n=== Fused QKV Analysis ===")
            print(f"  QKV shape: {qkv_shape}")
            # qkv_size = (num_heads + 2*num_kv_heads) * head_size
            # For Devstral: (32 + 2*8) * 128 = 48 * 128 = 6144
            qkv_size = qkv_shape[0]
            hidden_size = qkv_shape[1]
            print(f"  QKV output size: {qkv_size}")
            print(f"  Hidden size: {hidden_size}")

            # Try to decode
            for head_size in [128, 64, 96, 32]:
                if qkv_size % head_size == 0:
                    total_heads = qkv_size // head_size
                    print(f"  If head_size={head_size}: total_heads={total_heads}")
                    # For GQA: num_heads + 2*num_kv_heads = total_heads
                    # Try different splits
                    for num_kv in range(1, total_heads // 2):
                        num_q = total_heads - 2 * num_kv
                        if num_q > 0 and num_q % num_kv == 0:
                            ratio = num_q // num_kv
                            print(f"    Possible: num_q_heads={num_q}, num_kv_heads={num_kv}, ratio={ratio}")

        if has_gate_up_proj:
            gate_up_key = [k for k in state_dict.keys() if 'layers.0' in k and 'gate_up_proj' in k][0]
            gate_up_shape = state_dict[gate_up_key].shape
            print(f"\n=== Fused Gate-Up Analysis ===")
            print(f"  Gate-up shape: {gate_up_shape}")
            print(f"  Intermediate size: {gate_up_shape[0] // 2}")

    except Exception as e:
        print(f"Error accessing model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

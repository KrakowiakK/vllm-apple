#!/usr/bin/env python3
"""Debug weight loading for Devstral model."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '0'  # Use non-engine path to load model

import torch

def main():
    # Load model directly using transformers
    from transformers import AutoModelForCausalLM, AutoConfig

    print("Loading model config...")
    config = AutoConfig.from_pretrained('mistralai/Devstral-Small-2505', trust_remote_code=True)

    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Devstral-Small-2505',
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("\n=== Model State Dict Keys ===")
    state_dict = model.state_dict()

    # Group by prefix
    from collections import defaultdict
    groups = defaultdict(list)
    for key in sorted(state_dict.keys()):
        parts = key.split('.')
        if len(parts) > 2:
            prefix = '.'.join(parts[:3])
        else:
            prefix = parts[0]
        groups[prefix].append(key)

    # Print grouped
    for prefix, keys in sorted(groups.items()):
        if len(keys) <= 5:
            for key in keys:
                shape = state_dict[key].shape
                print(f"  {key}: {shape}")
        else:
            print(f"  {prefix}.* ({len(keys)} keys)")
            # Show first and last
            for key in keys[:2]:
                shape = state_dict[key].shape
                print(f"    {key}: {shape}")
            print(f"    ...")
            for key in keys[-2:]:
                shape = state_dict[key].shape
                print(f"    {key}: {shape}")

    # Check specific attention weights for layer 0
    print("\n=== Layer 0 Attention Weights ===")
    layer0_keys = [k for k in state_dict.keys() if 'layers.0.' in k and ('proj' in k or 'attn' in k)]
    for key in sorted(layer0_keys):
        shape = state_dict[key].shape
        dtype = state_dict[key].dtype
        print(f"  {key}: shape={shape}, dtype={dtype}")

    # Check if qkv is fused or separate
    has_fused_qkv = any('qkv_proj' in k for k in state_dict.keys())
    has_separate_qkv = any('q_proj' in k and 'qkv_proj' not in k for k in state_dict.keys())
    print(f"\n=== QKV Weight Pattern ===")
    print(f"  Fused qkv_proj: {has_fused_qkv}")
    print(f"  Separate q/k/v_proj: {has_separate_qkv}")

    # Check MLP weights
    has_fused_gate_up = any('gate_up_proj' in k for k in state_dict.keys())
    has_separate_gate_up = any('gate_proj' in k and 'gate_up_proj' not in k for k in state_dict.keys())
    print(f"\n=== MLP Weight Pattern ===")
    print(f"  Fused gate_up_proj: {has_fused_gate_up}")
    print(f"  Separate gate/up_proj: {has_separate_gate_up}")

    # Print expected sizes
    print("\n=== Expected Dimensions ===")
    config = model.config
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  vocab_size: {config.vocab_size}")

    # Calculate expected shapes
    hidden = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv_size = q_size + 2 * kv_size

    print(f"\n=== Expected Weight Shapes ===")
    print(f"  q_proj: [{q_size}, {hidden}]")
    print(f"  k_proj: [{kv_size}, {hidden}]")
    print(f"  v_proj: [{kv_size}, {hidden}]")
    print(f"  qkv_proj (if fused): [{qkv_size}, {hidden}]")
    print(f"  o_proj: [{hidden}, {q_size}]")

if __name__ == "__main__":
    main()

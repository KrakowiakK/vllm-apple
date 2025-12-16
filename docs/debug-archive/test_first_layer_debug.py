#!/usr/bin/env python3
"""Debug the first transformer layer output to find where divergence starts."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '0'  # Load without engine first

import numpy as np
import torch

def main():
    print("Loading model without engine (for weight reference)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = 'mistralai/Devstral-Small-2505'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Print model config
    config = model.config
    print(f"\n=== Model Config ===")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {config.num_key_value_heads}")
    print(f"head_dim: {config.head_dim}")
    print(f"intermediate_size: {config.intermediate_size}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")

    # Get state dict to examine weights
    state_dict = model.state_dict()

    # Print layer 0 weight shapes
    print(f"\n=== Layer 0 Weights ===")
    layer0_keys = sorted([k for k in state_dict.keys() if 'layers.0.' in k])
    for key in layer0_keys:
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")

    # Check QKV structure
    print(f"\n=== QKV Analysis ===")
    q_key = 'model.layers.0.self_attn.q_proj.weight'
    k_key = 'model.layers.0.self_attn.k_proj.weight'
    v_key = 'model.layers.0.self_attn.v_proj.weight'

    if q_key in state_dict:
        q_shape = state_dict[q_key].shape
        k_shape = state_dict[k_key].shape
        v_shape = state_dict[v_key].shape
        print(f"Separate Q/K/V:")
        print(f"  Q: {q_shape}")
        print(f"  K: {k_shape}")
        print(f"  V: {v_shape}")
        qkv_size = q_shape[0] + k_shape[0] + v_shape[0]
        print(f"  Total QKV size: {qkv_size}")

    # Check if vLLM would fuse them
    print(f"\n=== Expected QKV after vLLM fusion ===")
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    hidden = config.hidden_size

    q_size = num_heads * head_dim
    k_size = num_kv_heads * head_dim
    v_size = num_kv_heads * head_dim
    fused_qkv_size = q_size + k_size + v_size

    print(f"  Q size: {q_size} ({num_heads} heads * {head_dim} dim)")
    print(f"  K size: {k_size} ({num_kv_heads} kv_heads * {head_dim} dim)")
    print(f"  V size: {v_size} ({num_kv_heads} kv_heads * {head_dim} dim)")
    print(f"  Fused QKV total: {fused_qkv_size}")
    print(f"  Expected fused shape: [{fused_qkv_size}, {hidden}]")

    # Quick sanity check: run a forward pass
    print(f"\n=== Running Reference Forward Pass ===")
    prompt = "def add(a, b):"
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"Input token IDs: {inputs.input_ids[0].tolist()}")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Check first layer output
    first_layer_hidden = outputs.hidden_states[1]  # After layer 0
    last_layer_hidden = outputs.hidden_states[-1]  # After all layers

    print(f"\nAfter embedding (hidden_states[0]):")
    print(f"  Shape: {outputs.hidden_states[0].shape}")
    print(f"  Sample: {outputs.hidden_states[0][0, 0, :5]}")

    print(f"\nAfter layer 0 (hidden_states[1]):")
    print(f"  Shape: {first_layer_hidden.shape}")
    print(f"  Sample: {first_layer_hidden[0, 0, :5]}")

    print(f"\nFinal hidden (hidden_states[-1]):")
    print(f"  Shape: {last_layer_hidden.shape}")
    print(f"  Sample: {last_layer_hidden[0, -1, :5]}")

    # Get logits
    logits = outputs.logits
    print(f"\nLogits shape: {logits.shape}")
    print(f"Last token logits range: [{logits[0, -1].min():.2f}, {logits[0, -1].max():.2f}]")

    # Get top-5 predictions for last position
    probs = torch.softmax(logits[0, -1], dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"\nTop-5 predictions (last position):")
    for i in range(5):
        token_id = top5.indices[i].item()
        prob = top5.values[i].item()
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id}: {token!r} (prob={prob:.4f})")

    # Also show top-5 by raw logits
    top5_logits = torch.topk(logits[0, -1], 5)
    print(f"\nTop-5 by logits (last position):")
    for i in range(5):
        token_id = top5_logits.indices[i].item()
        logit = top5_logits.values[i].item()
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. Token {token_id}: {token!r} (logit={logit:.4f})")


if __name__ == "__main__":
    main()

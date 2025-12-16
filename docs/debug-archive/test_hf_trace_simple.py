#!/usr/bin/env python3
"""Simple trace of HuggingFace model values."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = 'mistralai/Devstral-Small-2505'

    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    config = model.config
    print(f"\n=== Model Config ===")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {config.num_key_value_heads}")
    print(f"head_dim: {config.head_dim}")
    print(f"rope_theta: {config.rope_theta}")

    # Dictionary to store captured values
    captured = {}

    # Simple hooks that just capture output
    def make_output_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[name] = output[0].detach().clone()
            else:
                captured[name] = output.detach().clone()
        return hook

    # Register hooks on layer 0
    layer0 = model.model.layers[0]
    layer0.input_layernorm.register_forward_hook(make_output_hook('input_ln'))
    layer0.self_attn.q_proj.register_forward_hook(make_output_hook('q_proj'))
    layer0.self_attn.k_proj.register_forward_hook(make_output_hook('k_proj'))
    layer0.self_attn.v_proj.register_forward_hook(make_output_hook('v_proj'))
    layer0.self_attn.o_proj.register_forward_hook(make_output_hook('o_proj'))
    layer0.post_attention_layernorm.register_forward_hook(make_output_hook('post_attn_ln'))

    # Run forward pass
    prompt = "def add(a, b):"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)

    print(f"\n=== Input ===")
    print(f"Prompt: {prompt!r}")
    print(f"Token IDs: {token_ids.tolist()}")
    print(f"Num tokens: {num_tokens}")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Print captured values
    print(f"\n=== HuggingFace Internal Values ===")

    # Embedding
    embedding = outputs.hidden_states[0][0]
    print(f"\n1. Embedding:")
    print(f"   Shape: {embedding.shape}")
    print(f"   [0, :5]: {embedding[0, :5]}")

    # Input LayerNorm
    print(f"\n2. After Input LayerNorm:")
    print(f"   Shape: {captured['input_ln'].shape}")
    print(f"   [0, 0, :5]: {captured['input_ln'][0, 0, :5]}")

    # Q projection (this is BEFORE RoPE)
    print(f"\n3. Q Projection (before RoPE):")
    print(f"   Shape: {captured['q_proj'].shape}")
    print(f"   [0, 0, :5]: {captured['q_proj'][0, 0, :5]}")

    # K projection (before RoPE)
    print(f"\n4. K Projection (before RoPE):")
    print(f"   Shape: {captured['k_proj'].shape}")
    print(f"   [0, 0, :5]: {captured['k_proj'][0, 0, :5]}")

    # V projection
    print(f"\n5. V Projection:")
    print(f"   Shape: {captured['v_proj'].shape}")
    print(f"   [0, 0, :5]: {captured['v_proj'][0, 0, :5]}")

    # O projection output
    print(f"\n6. O Projection Output:")
    print(f"   Shape: {captured['o_proj'].shape}")
    print(f"   [0, 0, :5]: {captured['o_proj'][0, 0, :5]}")

    # After layer 0
    after_layer0 = outputs.hidden_states[1][0]
    print(f"\n7. After Full Layer 0:")
    print(f"   Shape: {after_layer0.shape}")
    print(f"   [0, :5]: {after_layer0[0, :5]}")

    # Now verify our manual computation
    print(f"\n\n=== Manual Verification ===")

    state_dict = model.state_dict()
    q_weight = state_dict['model.layers.0.self_attn.q_proj.weight']
    k_weight = state_dict['model.layers.0.self_attn.k_proj.weight']
    v_weight = state_dict['model.layers.0.self_attn.v_proj.weight']
    input_ln_weight = state_dict['model.layers.0.input_layernorm.weight']
    embed_weight = state_dict['model.embed_tokens.weight']

    # Get first token
    first_embed = embed_weight[token_ids[0:1]]  # [1, hidden]

    print(f"\nFirst token embedding:")
    print(f"  Shape: {first_embed.shape}")
    print(f"  [:5]: {first_embed[0, :5]}")
    print(f"  HF[:5]: {embedding[0, :5]}")
    print(f"  Match: {torch.allclose(first_embed[0], embedding[0], atol=1e-3)}")

    # RMSNorm
    def rmsnorm(x, w, eps=1e-5):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x.float() * torch.rsqrt(variance + eps)
        return (x_norm * w.float()).to(x.dtype)

    manual_normed = rmsnorm(first_embed, input_ln_weight)
    print(f"\nManual RMSNorm:")
    print(f"  Shape: {manual_normed.shape}")
    print(f"  [:5]: {manual_normed[0, :5]}")
    print(f"  HF[:5]: {captured['input_ln'][0, 0, :5]}")
    print(f"  Diff: {torch.abs(manual_normed[0, :5] - captured['input_ln'][0, 0, :5])}")

    # QKV projection (note: HuggingFace hooks capture after reshape)
    manual_q = manual_normed @ q_weight.T  # [1, q_size]
    manual_k = manual_normed @ k_weight.T  # [1, k_size]
    manual_v = manual_normed @ v_weight.T  # [1, v_size]

    print(f"\nManual Q projection:")
    print(f"  Shape: {manual_q.shape}")
    print(f"  [:5]: {manual_q[0, :5]}")

    # HF q_proj output is reshaped: [batch, seq, num_heads, head_dim]
    print(f"  HF shape: {captured['q_proj'].shape}")
    # Reshape HF to compare: [batch, seq, num_heads * head_dim]
    hf_q_flat = captured['q_proj'].view(1, num_tokens, -1)
    print(f"  HF reshaped: {hf_q_flat.shape}")
    print(f"  HF[:5]: {hf_q_flat[0, 0, :5]}")
    print(f"  Diff: {torch.abs(manual_q[0, :5] - hf_q_flat[0, 0, :5])}")

    # Check shapes match our expectation
    print(f"\n=== Shape Analysis ===")
    print(f"Q proj weight: {q_weight.shape} (should be [q_size, hidden])")
    print(f"K proj weight: {k_weight.shape} (should be [k_size, hidden])")
    print(f"V proj weight: {v_weight.shape} (should be [v_size, hidden])")
    print(f"Expected Q size: {config.num_attention_heads * config.head_dim}")
    print(f"Expected K size: {config.num_key_value_heads * config.head_dim}")
    print(f"Expected V size: {config.num_key_value_heads * config.head_dim}")


if __name__ == "__main__":
    main()

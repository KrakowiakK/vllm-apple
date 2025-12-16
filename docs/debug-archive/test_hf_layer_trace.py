#!/usr/bin/env python3
"""Trace HuggingFace model internal values with hooks."""
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

    # Hook for input layernorm
    def capture_input_ln_output(module, input, output):
        captured['input_ln_output'] = output.detach().clone()

    # Hook for attention (self_attn module)
    def capture_attn_input(module, input, output):
        # input[0] is hidden states after input_layernorm
        captured['attn_input'] = input[0].detach().clone()
        # output[0] is attention output
        captured['attn_output'] = output[0].detach().clone()

    # Hook for q_proj
    def capture_q_proj(module, input, output):
        captured['q_proj_input'] = input[0].detach().clone()
        captured['q_proj_output'] = output.detach().clone()

    # Hook for k_proj
    def capture_k_proj(module, input, output):
        captured['k_proj_output'] = output.detach().clone()

    # Hook for v_proj
    def capture_v_proj(module, input, output):
        captured['v_proj_output'] = output.detach().clone()

    # Hook for o_proj
    def capture_o_proj(module, input, output):
        captured['o_proj_input'] = input[0].detach().clone()
        captured['o_proj_output'] = output.detach().clone()

    # Register hooks on layer 0
    layer0 = model.model.layers[0]
    layer0.input_layernorm.register_forward_hook(capture_input_ln_output)
    layer0.self_attn.register_forward_hook(capture_attn_input)
    layer0.self_attn.q_proj.register_forward_hook(capture_q_proj)
    layer0.self_attn.k_proj.register_forward_hook(capture_k_proj)
    layer0.self_attn.v_proj.register_forward_hook(capture_v_proj)
    layer0.self_attn.o_proj.register_forward_hook(capture_o_proj)

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
    print(f"\n=== Captured Values ===")

    # Embedding (from hidden_states[0])
    embedding = outputs.hidden_states[0][0]
    print(f"\n1. Embedding:")
    print(f"   Shape: {embedding.shape}")
    print(f"   Sample [0, :5]: {embedding[0, :5]}")

    # After input layernorm
    print(f"\n2. After Input LayerNorm:")
    print(f"   Shape: {captured['input_ln_output'].shape}")
    print(f"   Sample [0, 0, :5]: {captured['input_ln_output'][0, 0, :5]}")

    # Q projection
    print(f"\n3. Q Projection Output:")
    print(f"   Shape: {captured['q_proj_output'].shape}")
    print(f"   Sample [0, 0, :5]: {captured['q_proj_output'][0, 0, :5]}")

    # K projection
    print(f"\n4. K Projection Output:")
    print(f"   Shape: {captured['k_proj_output'].shape}")
    print(f"   Sample [0, 0, :5]: {captured['k_proj_output'][0, 0, :5]}")

    # V projection
    print(f"\n5. V Projection Output:")
    print(f"   Shape: {captured['v_proj_output'].shape}")
    print(f"   Sample [0, 0, :5]: {captured['v_proj_output'][0, 0, :5]}")

    # O projection input (attention output before O-proj)
    print(f"\n6. O Projection Input (attention output):")
    print(f"   Shape: {captured['o_proj_input'].shape}")
    print(f"   Sample [0, 0, :5]: {captured['o_proj_input'][0, 0, :5]}")

    # O projection output
    print(f"\n7. O Projection Output:")
    print(f"   Shape: {captured['o_proj_output'].shape}")
    print(f"   Sample [0, 0, :5]: {captured['o_proj_output'][0, 0, :5]}")

    # After layer 0 (from hidden_states[1])
    after_layer0 = outputs.hidden_states[1][0]
    print(f"\n8. After Full Layer 0 (including residual):")
    print(f"   Shape: {after_layer0.shape}")
    print(f"   Sample [0, :5]: {after_layer0[0, :5]}")

    # Check attention module structure
    print(f"\n=== Attention Module Structure ===")
    attn = layer0.self_attn
    print(f"Module type: {type(attn)}")
    print(f"Q proj: {attn.q_proj}")
    print(f"K proj: {attn.k_proj}")
    print(f"V proj: {attn.v_proj}")
    print(f"O proj: {attn.o_proj}")

    # Check if there's a rotary embedding
    if hasattr(attn, 'rotary_emb'):
        print(f"Rotary Emb: {attn.rotary_emb}")
        if hasattr(attn.rotary_emb, 'cos_cached'):
            print(f"  cos_cached shape: {attn.rotary_emb.cos_cached.shape if attn.rotary_emb.cos_cached is not None else None}")
    else:
        print("No rotary_emb attribute")

    # Manual Q check
    print(f"\n=== Manual Q Check ===")
    state_dict = model.state_dict()
    q_weight = state_dict['model.layers.0.self_attn.q_proj.weight']
    input_ln_weight = state_dict['model.layers.0.input_layernorm.weight']

    # Get embedding for first token
    embed_weight = state_dict['model.embed_tokens.weight']
    first_token_embed = embed_weight[token_ids[0]]
    print(f"First token embedding sample: {first_token_embed[:5]}")

    # Manual RMSNorm
    def rmsnorm(x, w, eps=1e-5):
        variance = x.float().pow(2).mean().unsqueeze(0)
        x_norm = x.float() * torch.rsqrt(variance + eps)
        return (x_norm * w.float()).to(x.dtype)

    manual_normed = rmsnorm(first_token_embed, input_ln_weight)
    print(f"Manual normed sample: {manual_normed[:5]}")
    print(f"HF normed sample: {captured['input_ln_output'][0, 0, :5]}")

    # Manual Q projection
    manual_q = manual_normed @ q_weight.T
    print(f"\nManual Q shape: {manual_q.shape}")
    print(f"Manual Q sample: {manual_q[:5]}")
    print(f"HF Q sample: {captured['q_proj_output'][0, 0, :5]}")

    diff = torch.abs(manual_q[:5] - captured['q_proj_output'][0, 0, :5])
    print(f"Q diff: {diff}")


if __name__ == "__main__":
    main()

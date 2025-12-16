#!/usr/bin/env python3
"""Trace a single transformer layer through engine vs HuggingFace.

This test loads the same weights and compares outputs at each step:
1. Input hidden states
2. After RMSNorm
3. After QKV projection
4. After RoPE
5. After attention
6. After O-proj + residual
"""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def reference_rmsnorm(x, weight, eps=1e-5):
    """Reference RMSNorm."""
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_f32 * torch.rsqrt(variance + eps)
    return (x_norm * weight.float()).to(x.dtype)


def reference_rope(q, k, positions, head_size, rope_theta=10000.0):
    """Reference RoPE implementation."""
    seq_len = q.shape[0]
    rotary_dim = head_size

    # Compute frequency table
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))

    # Compute position angles
    freqs = torch.outer(positions.float(), inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    def apply_rope(x):
        """Apply RoPE to tensor [seq_len, num_heads, head_size]."""
        x_f32 = x.float()
        out = torch.zeros_like(x_f32)

        for i in range(0, rotary_dim, 2):
            cos_val = cos[:, i // 2].unsqueeze(1)  # [seq, 1]
            sin_val = sin[:, i // 2].unsqueeze(1)  # [seq, 1]

            x0 = x_f32[:, :, i]
            x1 = x_f32[:, :, i + 1]

            out[:, :, i] = x0 * cos_val - x1 * sin_val
            out[:, :, i + 1] = x0 * sin_val + x1 * cos_val

        return out.to(x.dtype)

    return apply_rope(q), apply_rope(k)


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
    print(f"intermediate_size: {config.intermediate_size}")
    print(f"rope_theta: {config.rope_theta}")

    # Get layer 0 weights
    state_dict = model.state_dict()
    q_weight = state_dict['model.layers.0.self_attn.q_proj.weight']  # [q_size, hidden]
    k_weight = state_dict['model.layers.0.self_attn.k_proj.weight']  # [k_size, hidden]
    v_weight = state_dict['model.layers.0.self_attn.v_proj.weight']  # [v_size, hidden]
    o_weight = state_dict['model.layers.0.self_attn.o_proj.weight']  # [hidden, o_in]
    input_ln_weight = state_dict['model.layers.0.input_layernorm.weight']  # [hidden]

    print(f"\n=== Layer 0 Weight Shapes ===")
    print(f"q_proj: {q_weight.shape}")
    print(f"k_proj: {k_weight.shape}")
    print(f"v_proj: {v_weight.shape}")
    print(f"o_proj: {o_weight.shape}")
    print(f"input_layernorm: {input_ln_weight.shape}")

    # Tokenize input
    prompt = "def add(a, b):"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)
    positions = torch.arange(num_tokens, dtype=torch.long)

    print(f"\n=== Input ===")
    print(f"Prompt: {prompt!r}")
    print(f"Token IDs: {token_ids.tolist()}")
    print(f"Num tokens: {num_tokens}")

    # Get embedding
    embed_weight = state_dict['model.embed_tokens.weight']  # [vocab, hidden]
    hidden_states = embed_weight[token_ids]  # [num_tokens, hidden]

    print(f"\n=== Step 1: Embedding ===")
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Hidden states sample [0, :5]: {hidden_states[0, :5]}")

    # Apply input layernorm
    normed = reference_rmsnorm(hidden_states, input_ln_weight)

    print(f"\n=== Step 2: After RMSNorm ===")
    print(f"Normed shape: {normed.shape}")
    print(f"Normed sample [0, :5]: {normed[0, :5]}")

    # QKV projection
    Q = normed @ q_weight.T  # [num_tokens, q_size]
    K = normed @ k_weight.T  # [num_tokens, k_size]
    V = normed @ v_weight.T  # [num_tokens, v_size]

    print(f"\n=== Step 3: After QKV Projection ===")
    print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"Q sample [0, :5]: {Q[0, :5]}")
    print(f"K sample [0, :5]: {K[0, :5]}")
    print(f"V sample [0, :5]: {V[0, :5]}")

    # Reshape for attention
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    Q_reshaped = Q.view(num_tokens, num_heads, head_dim)
    K_reshaped = K.view(num_tokens, num_kv_heads, head_dim)
    V_reshaped = V.view(num_tokens, num_kv_heads, head_dim)

    print(f"\n=== Step 4: Reshape for Attention ===")
    print(f"Q_reshaped: {Q_reshaped.shape}")
    print(f"K_reshaped: {K_reshaped.shape}")
    print(f"V_reshaped: {V_reshaped.shape}")

    # Apply RoPE
    Q_rope, K_rope = reference_rope(Q_reshaped, K_reshaped, positions, head_dim, config.rope_theta)

    print(f"\n=== Step 5: After RoPE ===")
    print(f"Q_rope sample [0, 0, :5]: {Q_rope[0, 0, :5]}")
    print(f"K_rope sample [0, 0, :5]: {K_rope[0, 0, :5]}")

    # GQA: expand KV heads
    # K_expanded: [num_tokens, num_heads, head_dim]
    kv_groups = num_heads // num_kv_heads
    K_expanded = K_rope.repeat_interleave(kv_groups, dim=1)
    V_expanded = V_reshaped.repeat_interleave(kv_groups, dim=1)

    print(f"\n=== Step 6: KV Expansion (GQA) ===")
    print(f"K_expanded: {K_expanded.shape}")
    print(f"V_expanded: {V_expanded.shape}")

    # Attention: Q @ K^T -> scores -> softmax -> @ V
    # Q: [num_tokens, num_heads, head_dim]
    # K: [num_tokens, num_heads, head_dim]
    scale = 1.0 / (head_dim ** 0.5)

    # Compute attention scores: [num_tokens, num_heads, num_tokens]
    scores = torch.einsum('thd,shd->ths', Q_rope.float(), K_expanded.float()) * scale

    # Causal mask
    causal_mask = torch.triu(torch.ones(num_tokens, num_tokens, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(1), float('-inf'))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    print(f"\n=== Step 7: Attention Scores ===")
    print(f"Scores shape: {scores.shape}")
    print(f"Attn weights shape: {attn_weights.shape}")
    print(f"Attn weights [0, 0, :]: {attn_weights[0, 0, :]}")  # First token, first head

    # Apply attention: [num_tokens, num_heads, num_tokens] @ [num_tokens, num_heads, head_dim]
    attn_output = torch.einsum('ths,shd->thd', attn_weights, V_expanded.float())
    attn_output = attn_output.to(hidden_states.dtype)

    print(f"\n=== Step 8: Attention Output ===")
    print(f"Attn output shape: {attn_output.shape}")
    print(f"Attn output sample [0, 0, :5]: {attn_output[0, 0, :5]}")

    # Flatten attention output for O-proj
    attn_flat = attn_output.reshape(num_tokens, num_heads * head_dim)

    print(f"\n=== Step 9: Flattened for O-proj ===")
    print(f"Attn flat shape: {attn_flat.shape}")
    print(f"Attn flat sample [0, :5]: {attn_flat[0, :5]}")

    # O-projection + residual
    o_out = attn_flat @ o_weight.T
    output = hidden_states + o_out

    print(f"\n=== Step 10: After O-proj + Residual ===")
    print(f"O-proj output shape: {o_out.shape}")
    print(f"O-proj sample [0, :5]: {o_out[0, :5]}")
    print(f"Layer output shape: {output.shape}")
    print(f"Layer output sample [0, :5]: {output[0, :5]}")

    # Compare with HuggingFace layer 0 output
    print(f"\n=== Comparison with HuggingFace Full Forward ===")
    with torch.no_grad():
        hf_outputs = model(inputs.input_ids, output_hidden_states=True)

    hf_after_layer0 = hf_outputs.hidden_states[1][0]  # [num_tokens, hidden]

    print(f"HF after layer 0 shape: {hf_after_layer0.shape}")
    print(f"HF sample [0, :5]: {hf_after_layer0[0, :5]}")
    print(f"Our sample [0, :5]: {output[0, :5]}")

    diff = torch.abs(output - hf_after_layer0)
    print(f"\nMax diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")

    if diff.max() < 0.01:
        print("\n=== Reference Layer 0 PASSES ===")
    else:
        print("\n=== Reference Layer 0 DIFFERS ===")
        # Find where the largest diff is
        max_idx = torch.argmax(diff)
        token_idx = max_idx // diff.shape[1]
        hidden_idx = max_idx % diff.shape[1]
        print(f"Largest diff at token {token_idx}, hidden {hidden_idx}")
        print(f"Our value: {output[token_idx, hidden_idx]}")
        print(f"HF value: {hf_after_layer0[token_idx, hidden_idx]}")


if __name__ == "__main__":
    main()

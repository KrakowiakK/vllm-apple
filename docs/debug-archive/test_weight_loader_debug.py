#!/usr/bin/env python3
"""Debug what weight_loader produces from vLLM model."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '0'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = 'mistralai/Devstral-Small-2505'

    print("=== Loading HuggingFace model directly ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Import weight loader
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.weight_loader import EngineWeightLoader

    print("\n=== Running EngineWeightLoader on HF model ===")
    ctx = MetalEngineContext()
    loader = EngineWeightLoader(ctx, model_config=None)

    # Load weights from HF model
    weights = loader.load_from_hf_model(hf_model, arch='llama')

    # Check what we got
    print(f"\n=== Loaded Weight Structure ===")
    print(f"num_layers: {weights.num_layers}")
    print(f"hidden_size: {weights.hidden_size}")
    print(f"vocab_size: {weights.vocab_size}")
    print(f"intermediate_size: {weights.intermediate_size}")
    print(f"num_attention_heads: {weights.num_attention_heads}")
    print(f"num_kv_heads: {weights.num_kv_heads}")
    print(f"head_size: {weights.head_size}")

    print(f"\n=== Layer 0 Weights ===")
    lw = weights.layers[0]
    print(f"  qkv_proj (fused): {lw.qkv_proj is not None}")
    print(f"  q_proj (separate): {lw.q_proj is not None}")
    print(f"  k_proj (separate): {lw.k_proj is not None}")
    print(f"  v_proj (separate): {lw.v_proj is not None}")
    print(f"  o_proj: {lw.o_proj is not None}")
    print(f"  input_layernorm: {lw.input_layernorm is not None}")
    print(f"  post_attention_layernorm: {lw.post_attention_layernorm is not None}")
    print(f"  gate_up_proj (fused): {lw.gate_up_proj is not None}")
    print(f"  gate_proj (separate): {lw.gate_proj is not None}")
    print(f"  up_proj (separate): {lw.up_proj is not None}")
    print(f"  down_proj: {lw.down_proj is not None}")

    # Print buffer sizes
    def buf_size(buf):
        return buf.length() if buf is not None else 0

    print(f"\n=== Layer 0 Buffer Sizes ===")
    if lw.qkv_proj:
        print(f"  qkv_proj: {buf_size(lw.qkv_proj)} bytes")
    if lw.q_proj:
        print(f"  q_proj: {buf_size(lw.q_proj)} bytes")
    if lw.k_proj:
        print(f"  k_proj: {buf_size(lw.k_proj)} bytes")
    if lw.v_proj:
        print(f"  v_proj: {buf_size(lw.v_proj)} bytes")
    print(f"  o_proj: {buf_size(lw.o_proj)} bytes")
    print(f"  input_layernorm: {buf_size(lw.input_layernorm)} bytes")

    if lw.gate_up_proj:
        print(f"  gate_up_proj: {buf_size(lw.gate_up_proj)} bytes")
    if lw.gate_proj:
        print(f"  gate_proj: {buf_size(lw.gate_proj)} bytes")
    if lw.up_proj:
        print(f"  up_proj: {buf_size(lw.up_proj)} bytes")
    print(f"  down_proj: {buf_size(lw.down_proj)} bytes")

    # Expected sizes for Devstral
    print(f"\n=== Expected Sizes (Devstral) ===")
    hidden = 5120
    heads = 32
    kv_heads = 8
    head_dim = 128
    intermediate = 32768
    q_size = heads * head_dim  # 4096
    k_size = kv_heads * head_dim  # 1024
    v_size = kv_heads * head_dim  # 1024
    print(f"  q_proj: {q_size * hidden * 2} bytes = [4096, 5120] * 2 (fp16)")
    print(f"  k_proj: {k_size * hidden * 2} bytes = [1024, 5120] * 2")
    print(f"  v_proj: {v_size * hidden * 2} bytes = [1024, 5120] * 2")
    print(f"  o_proj: {hidden * q_size * 2} bytes = [5120, 4096] * 2")
    print(f"  gate_proj: {intermediate * hidden * 2} bytes = [32768, 5120] * 2")
    print(f"  up_proj: {intermediate * hidden * 2} bytes = [32768, 5120] * 2")
    print(f"  down_proj: {hidden * intermediate * 2} bytes = [5120, 32768] * 2")


if __name__ == "__main__":
    main()

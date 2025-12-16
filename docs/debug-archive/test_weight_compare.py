#!/usr/bin/env python3
"""Compare weights between manual loading and EngineWeightLoader."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from Metal import MTLResourceStorageModeShared

    model_name = 'mistralai/Devstral-Small-2505'

    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    config = model.config
    state_dict = model.state_dict()

    ctx = MetalEngineContext()

    # Manual weight loading (like test_full_model_logits.py does)
    def manual_to_buffer(tensor):
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        tensor = tensor.cpu().contiguous()
        np_arr = tensor.numpy()
        np_arr = np.ascontiguousarray(np_arr)
        return ctx.device.newBufferWithBytes_length_options_(
            np_arr.tobytes(), np_arr.nbytes, MTLResourceStorageModeShared
        )

    # Load via EngineWeightLoader
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')

    print("\n=== Comparing Layer 0 Weights ===")

    # Q projection
    print("\n--- Q Projection ---")
    q_weight_hf = state_dict['model.layers.0.self_attn.q_proj.weight']
    q_manual_buf = manual_to_buffer(q_weight_hf)
    q_loader_buf = weights.layers[0].q_proj

    print(f"HF q_proj shape: {q_weight_hf.shape}")
    print(f"Manual buffer size: {q_manual_buf.length()}")
    print(f"Loader buffer size: {q_loader_buf.length()}")

    # Read back both buffers
    q_manual_view = q_manual_buf.contents().as_buffer(q_manual_buf.length())
    q_manual_np = np.frombuffer(q_manual_view, dtype=np.float16).copy()

    q_loader_view = q_loader_buf.contents().as_buffer(q_loader_buf.length())
    q_loader_np = np.frombuffer(q_loader_view, dtype=np.float16).copy()

    print(f"Manual first 5: {q_manual_np[:5]}")
    print(f"Loader first 5: {q_loader_np[:5]}")
    print(f"Max diff: {np.abs(q_manual_np - q_loader_np).max()}")
    print(f"Q proj MATCH: {np.allclose(q_manual_np, q_loader_np)}")

    # K projection
    print("\n--- K Projection ---")
    k_weight_hf = state_dict['model.layers.0.self_attn.k_proj.weight']
    k_manual_buf = manual_to_buffer(k_weight_hf)
    k_loader_buf = weights.layers[0].k_proj

    print(f"HF k_proj shape: {k_weight_hf.shape}")
    print(f"Manual buffer size: {k_manual_buf.length()}")
    print(f"Loader buffer size: {k_loader_buf.length()}")

    k_manual_view = k_manual_buf.contents().as_buffer(k_manual_buf.length())
    k_manual_np = np.frombuffer(k_manual_view, dtype=np.float16).copy()

    k_loader_view = k_loader_buf.contents().as_buffer(k_loader_buf.length())
    k_loader_np = np.frombuffer(k_loader_view, dtype=np.float16).copy()

    print(f"Manual first 5: {k_manual_np[:5]}")
    print(f"Loader first 5: {k_loader_np[:5]}")
    print(f"Max diff: {np.abs(k_manual_np - k_loader_np).max()}")
    print(f"K proj MATCH: {np.allclose(k_manual_np, k_loader_np)}")

    # V projection
    print("\n--- V Projection ---")
    v_weight_hf = state_dict['model.layers.0.self_attn.v_proj.weight']
    v_manual_buf = manual_to_buffer(v_weight_hf)
    v_loader_buf = weights.layers[0].v_proj

    print(f"HF v_proj shape: {v_weight_hf.shape}")
    print(f"Manual buffer size: {v_manual_buf.length()}")
    print(f"Loader buffer size: {v_loader_buf.length()}")

    v_manual_view = v_manual_buf.contents().as_buffer(v_manual_buf.length())
    v_manual_np = np.frombuffer(v_manual_view, dtype=np.float16).copy()

    v_loader_view = v_loader_buf.contents().as_buffer(v_loader_buf.length())
    v_loader_np = np.frombuffer(v_loader_view, dtype=np.float16).copy()

    print(f"Manual first 5: {v_manual_np[:5]}")
    print(f"Loader first 5: {v_loader_np[:5]}")
    print(f"Max diff: {np.abs(v_manual_np - v_loader_np).max()}")
    print(f"V proj MATCH: {np.allclose(v_manual_np, v_loader_np)}")

    # O projection
    print("\n--- O Projection ---")
    o_weight_hf = state_dict['model.layers.0.self_attn.o_proj.weight']
    o_manual_buf = manual_to_buffer(o_weight_hf)
    o_loader_buf = weights.layers[0].o_proj

    print(f"HF o_proj shape: {o_weight_hf.shape}")
    print(f"Manual buffer size: {o_manual_buf.length()}")
    print(f"Loader buffer size: {o_loader_buf.length()}")

    o_manual_view = o_manual_buf.contents().as_buffer(o_manual_buf.length())
    o_manual_np = np.frombuffer(o_manual_view, dtype=np.float16).copy()

    o_loader_view = o_loader_buf.contents().as_buffer(o_loader_buf.length())
    o_loader_np = np.frombuffer(o_loader_view, dtype=np.float16).copy()

    print(f"Manual first 5: {o_manual_np[:5]}")
    print(f"Loader first 5: {o_loader_np[:5]}")
    print(f"Max diff: {np.abs(o_manual_np - o_loader_np).max()}")
    print(f"O proj MATCH: {np.allclose(o_manual_np, o_loader_np)}")

    # Input LayerNorm
    print("\n--- Input LayerNorm ---")
    ln_weight_hf = state_dict['model.layers.0.input_layernorm.weight']
    ln_manual_buf = manual_to_buffer(ln_weight_hf)
    ln_loader_buf = weights.layers[0].input_layernorm

    print(f"HF input_ln shape: {ln_weight_hf.shape}")
    print(f"Manual buffer size: {ln_manual_buf.length()}")
    print(f"Loader buffer size: {ln_loader_buf.length()}")

    ln_manual_view = ln_manual_buf.contents().as_buffer(ln_manual_buf.length())
    ln_manual_np = np.frombuffer(ln_manual_view, dtype=np.float16).copy()

    ln_loader_view = ln_loader_buf.contents().as_buffer(ln_loader_buf.length())
    ln_loader_np = np.frombuffer(ln_loader_view, dtype=np.float16).copy()

    print(f"Manual first 5: {ln_manual_np[:5]}")
    print(f"Loader first 5: {ln_loader_np[:5]}")
    print(f"Max diff: {np.abs(ln_manual_np - ln_loader_np).max()}")
    print(f"Input LN MATCH: {np.allclose(ln_manual_np, ln_loader_np)}")

    # MLP gate
    print("\n--- MLP Gate Projection ---")
    gate_weight_hf = state_dict['model.layers.0.mlp.gate_proj.weight']
    gate_manual_buf = manual_to_buffer(gate_weight_hf)
    gate_loader_buf = weights.layers[0].gate_proj

    print(f"HF gate_proj shape: {gate_weight_hf.shape}")
    print(f"Manual buffer size: {gate_manual_buf.length()}")
    print(f"Loader buffer size: {gate_loader_buf.length()}")

    gate_manual_view = gate_manual_buf.contents().as_buffer(gate_manual_buf.length())
    gate_manual_np = np.frombuffer(gate_manual_view, dtype=np.float16).copy()

    gate_loader_view = gate_loader_buf.contents().as_buffer(gate_loader_buf.length())
    gate_loader_np = np.frombuffer(gate_loader_view, dtype=np.float16).copy()

    print(f"Manual first 5: {gate_manual_np[:5]}")
    print(f"Loader first 5: {gate_loader_np[:5]}")
    print(f"Max diff: {np.abs(gate_manual_np - gate_loader_np).max()}")
    print(f"Gate proj MATCH: {np.allclose(gate_manual_np, gate_loader_np)}")

    # Embedding
    print("\n--- Embedding ---")
    embed_weight_hf = state_dict['model.embed_tokens.weight']
    embed_manual_buf = manual_to_buffer(embed_weight_hf)
    embed_loader_buf = weights.embedding

    print(f"HF embedding shape: {embed_weight_hf.shape}")
    print(f"Manual buffer size: {embed_manual_buf.length()}")
    print(f"Loader buffer size: {embed_loader_buf.length()}")

    embed_manual_view = embed_manual_buf.contents().as_buffer(min(1000, embed_manual_buf.length()))
    embed_manual_np = np.frombuffer(embed_manual_view, dtype=np.float16).copy()

    embed_loader_view = embed_loader_buf.contents().as_buffer(min(1000, embed_loader_buf.length()))
    embed_loader_np = np.frombuffer(embed_loader_view, dtype=np.float16).copy()

    print(f"Manual first 10: {embed_manual_np[:10]}")
    print(f"Loader first 10: {embed_loader_np[:10]}")
    print(f"Max diff (first 500 elements): {np.abs(embed_manual_np - embed_loader_np).max()}")
    print(f"Embedding MATCH: {np.allclose(embed_manual_np, embed_loader_np)}")

    # LM Head
    print("\n--- LM Head ---")
    lm_weight_hf = state_dict['lm_head.weight']
    lm_manual_buf = manual_to_buffer(lm_weight_hf)
    lm_loader_buf = weights.lm_head

    print(f"HF lm_head shape: {lm_weight_hf.shape}")
    print(f"Manual buffer size: {lm_manual_buf.length()}")
    print(f"Loader buffer size: {lm_loader_buf.length()}")

    lm_manual_view = lm_manual_buf.contents().as_buffer(min(1000, lm_manual_buf.length()))
    lm_manual_np = np.frombuffer(lm_manual_view, dtype=np.float16).copy()

    lm_loader_view = lm_loader_buf.contents().as_buffer(min(1000, lm_loader_buf.length()))
    lm_loader_np = np.frombuffer(lm_loader_view, dtype=np.float16).copy()

    print(f"Manual first 10: {lm_manual_np[:10]}")
    print(f"Loader first 10: {lm_loader_np[:10]}")
    print(f"Max diff (first 500 elements): {np.abs(lm_manual_np - lm_loader_np).max()}")
    print(f"LM Head MATCH: {np.allclose(lm_manual_np, lm_loader_np)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test Metal Engine with SmolLM-135M (small model for fast testing)."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.runner import EngineRunner
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import (
        ModelDescriptor,
        StepDescriptor,
        EngineInputs,
    )

    model_name = 'HuggingFaceTB/SmolLM-135M'

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    config = model.config
    print(f"Model: {config.hidden_size} hidden, {config.num_attention_heads} heads, {config.num_hidden_layers} layers")
    print(f"KV heads: {config.num_key_value_heads}, head_dim: {config.head_dim}")
    print(f"rms_norm_eps: {config.rms_norm_eps}")

    # HF prediction
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)

    with torch.no_grad():
        hf_outputs = model(**inputs)
        hf_logits = hf_outputs.logits[0, -1]
        hf_probs = torch.softmax(hf_logits, dim=-1)
        hf_top5 = torch.topk(hf_probs, 5)
        hf_pred = hf_top5.indices[0].item()

    print(f"\n=== HuggingFace Prediction ===")
    print(f"Input: '{prompt}' ({num_tokens} tokens)")
    print(f"Top 5: {[tokenizer.decode([t]) for t in hf_top5.indices.tolist()]}")
    print(f"Predicted: '{tokenizer.decode([hf_pred])}'")

    # Engine setup
    ctx = MetalEngineContext()

    model_desc = ModelDescriptor(
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_size=config.head_dim,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        rope_theta=getattr(config, 'rope_theta', 10000.0),
        max_position_embeddings=getattr(config, 'max_position_embeddings', 2048),
        architecture="llama",
        rms_norm_eps=config.rms_norm_eps,
    )

    kv_desc = model_desc.get_kv_cache_descriptor(num_blocks=32, block_size=16)
    kv_cache = EngineKVCache(ctx, kv_desc)

    print("\nLoading weights...")
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='llama')

    print("Creating EngineRunner...")
    runner = EngineRunner(
        context=ctx,
        model_desc=model_desc,
        weights=weights,
        kv_cache=kv_cache,
    )

    # Create inputs
    positions = torch.arange(num_tokens, dtype=torch.int64)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    block_table = torch.zeros((1, 4), dtype=torch.int64)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int64)
    query_start_locs = torch.tensor([0, num_tokens], dtype=torch.int64)

    engine_inputs = EngineInputs(
        token_ids=token_ids.cpu(),
        positions=positions.cpu(),
        block_table=block_table.cpu(),
        slot_mapping=slot_mapping.cpu(),
        seq_lens=seq_lens.cpu(),
        query_start_locs=query_start_locs.cpu(),
        max_decode_seq_len=0,
    )

    step_desc = StepDescriptor(
        step_id=1,
        step_kind="prefill",
        num_scheduled_tokens=num_tokens,
        num_seqs_active=1,
        max_num_blocks_per_seq=4,
        is_first_step=True,
        cache_enabled=True,
    )

    # Execute
    print("\n=== Executing EngineRunner ===")
    outputs = runner.execute_step(step_desc, engine_inputs)

    engine_logits = outputs.logits[-1].numpy()
    engine_probs = np.exp(engine_logits - engine_logits.max())
    engine_probs = engine_probs / engine_probs.sum()
    top5_idx = np.argsort(engine_probs)[-5:][::-1]
    engine_pred = top5_idx[0]

    print(f"\n=== Engine Prediction ===")
    print(f"Top 5: {[tokenizer.decode([t]) for t in top5_idx.tolist()]}")
    print(f"Predicted: '{tokenizer.decode([engine_pred])}'")

    # Compare
    print(f"\n=== Comparison ===")
    print(f"HF predicted: '{tokenizer.decode([hf_pred])}' (id={hf_pred})")
    print(f"Engine predicted: '{tokenizer.decode([engine_pred])}' (id={engine_pred})")

    logits_diff = np.abs(engine_logits - hf_logits.numpy())
    print(f"Logits max diff: {logits_diff.max():.4f}")

    if engine_pred == hf_pred:
        print("\n✓ MATCH!")
    else:
        # Check if HF pred is in engine top 5
        if hf_pred in top5_idx:
            rank = np.where(top5_idx == hf_pred)[0][0] + 1
            print(f"\n⚠ Different top-1, but HF pred is rank {rank} in engine")
        else:
            print("\n✗ MISMATCH")


if __name__ == "__main__":
    main()

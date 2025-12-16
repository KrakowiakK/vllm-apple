#!/usr/bin/env python3
"""Test EngineRunner directly to verify vLLM integration."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Test EngineRunner directly."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.runner import EngineRunner
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import (
        ModelDescriptor,
        KVCacheDescriptor,
        StepDescriptor,
        EngineInputs,
    )

    model_name = 'mistralai/Devstral-Small-2505'

    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    config = model.config
    print(f"Model: {config.hidden_size} hidden, {config.num_attention_heads} heads")

    # Get HF prediction
    prompt = "def add(a, b):"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)

    with torch.no_grad():
        hf_outputs = model(**inputs)
        hf_logits = hf_outputs.logits[0]
        hf_last_logits = hf_logits[-1]
        hf_probs = torch.softmax(hf_last_logits, dim=-1)
        hf_top5 = torch.topk(hf_probs, 5)
        hf_pred_token = hf_top5.indices[0].item()

    print(f"\n=== HuggingFace Prediction ===")
    print(f"Top 5 tokens: {[tokenizer.decode([t]) for t in hf_top5.indices.tolist()]}")
    print(f"Predicted: '{tokenizer.decode([hf_pred_token])}'")

    # Setup engine
    ctx = MetalEngineContext()

    # Create ModelDescriptor
    model_desc = ModelDescriptor(
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_size=config.head_dim,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        rope_theta=config.rope_theta,
        max_position_embeddings=8192,
        architecture="mistral",
        rms_norm_eps=config.rms_norm_eps,
    )

    # Create KV cache
    kv_desc = model_desc.get_kv_cache_descriptor(
        num_blocks=64,
        block_size=16,
    )
    kv_cache = EngineKVCache(ctx, kv_desc)

    # Load weights using EngineWeightLoader
    print("\n=== Loading weights via EngineWeightLoader ===")
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')

    print(f"Weights loaded:")
    print(f"  num_layers: {weights.num_layers}")
    print(f"  hidden_size: {weights.hidden_size}")
    print(f"  vocab_size: {weights.vocab_size}")

    # Create EngineRunner
    print("\n=== Creating EngineRunner ===")
    runner = EngineRunner(
        context=ctx,
        model_desc=model_desc,
        weights=weights,
        kv_cache=kv_cache,
    )

    # Create inputs
    print("\n=== Creating EngineInputs ===")
    positions = torch.arange(num_tokens, dtype=torch.int64)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    block_table = torch.zeros((1, 4), dtype=torch.int64)  # 1 seq, max 4 blocks
    block_table[0, 0] = 0  # First block
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

    print(f"Input tokens: {token_ids.tolist()}")
    print(f"Positions: {positions.tolist()}")

    # Execute step
    print("\n=== Executing EngineRunner.execute_step() ===")
    outputs = runner.execute_step(step_desc, engine_inputs)

    print(f"Output logits shape: {outputs.logits.shape}")

    # Get last token logits
    engine_last_logits = outputs.logits[-1].numpy()

    # Check for NaN/Inf
    if np.any(np.isnan(engine_last_logits)):
        print("ERROR: Engine logits contain NaN!")
        return
    if np.any(np.isinf(engine_last_logits)):
        print("ERROR: Engine logits contain Inf!")
        return

    # Softmax
    logits_max = np.max(engine_last_logits)
    exp_logits = np.exp(engine_last_logits - logits_max)
    engine_probs = exp_logits / np.sum(exp_logits)

    # Top 5
    top5_indices = np.argsort(engine_probs)[-5:][::-1]
    top5_probs = engine_probs[top5_indices]
    engine_pred_token = top5_indices[0]

    print(f"\n=== EngineRunner Prediction ===")
    print(f"Top 5 tokens: {[tokenizer.decode([t]) for t in top5_indices.tolist()]}")
    print(f"Top 5 probs: {top5_probs.tolist()}")
    print(f"Predicted: '{tokenizer.decode([engine_pred_token])}'")

    # Compare
    print(f"\n=== Comparison ===")
    print(f"HF predicted: '{tokenizer.decode([hf_pred_token])}' (id={hf_pred_token})")
    print(f"Engine predicted: '{tokenizer.decode([engine_pred_token])}' (id={engine_pred_token})")

    if engine_pred_token == hf_pred_token:
        print("\n✓ MATCH! EngineRunner predicts correctly!")
    else:
        print("\n✗ MISMATCH!")
        if hf_pred_token in top5_indices:
            rank = np.where(top5_indices == hf_pred_token)[0][0] + 1
            print(f"  HF's prediction is rank {rank} in engine's predictions")


if __name__ == "__main__":
    main()

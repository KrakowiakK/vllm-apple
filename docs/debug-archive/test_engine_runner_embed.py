#!/usr/bin/env python3
"""Test EngineRunner with direct HF embeddings to isolate embedding issue."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Test EngineRunner with direct HF embeddings."""
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

    # Test 1: Verify embedding lookup
    print("\n=== Test 1: Verify Embedding Lookup ===")
    from vllm_apple.engine.ops.embedding import EngineEmbedding
    from vllm_apple.engine.step import EngineStepContext

    state_dict = model.state_dict()
    embed_weight = state_dict['model.embed_tokens.weight']

    # HF embedding lookup (reference)
    hf_embed = embed_weight[token_ids].cpu().numpy().astype(np.float16)
    print(f"HF embedding [0, :5]: {hf_embed[0, :5]}")

    # Engine embedding lookup
    embed_weight_buf = ctx.device.newBufferWithBytes_length_options_(
        embed_weight.cpu().numpy().astype(np.float16).tobytes(),
        embed_weight.numel() * 2,
        MTLResourceStorageModeShared
    )

    engine_embed = EngineEmbedding(
        context=ctx,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
    )
    engine_embed.set_weights(embed_weight_buf)

    # Prepare token IDs buffer
    token_ids_int32 = token_ids.to(torch.int32).numpy()
    token_ids_buf = ctx.device.newBufferWithBytes_length_options_(
        token_ids_int32.tobytes(),
        token_ids_int32.nbytes,
        MTLResourceStorageModeShared
    )

    # Output buffer
    embed_out_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * config.hidden_size * 2,
        MTLResourceStorageModeShared
    )

    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        engine_embed.encode(
            step_ctx=step_ctx,
            token_ids=token_ids_buf,
            output_buffer=embed_out_buf,
            num_tokens=num_tokens,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read back
    embed_view = embed_out_buf.contents().as_buffer(num_tokens * config.hidden_size * 2)
    engine_embed_out = np.frombuffer(embed_view, dtype=np.float16).reshape(num_tokens, config.hidden_size).copy()

    print(f"Engine embedding [0, :5]: {engine_embed_out[0, :5]}")
    embed_diff = np.abs(engine_embed_out - hf_embed)
    print(f"Embedding max diff: {embed_diff.max():.6f}")
    print(f"Embedding PASS: {embed_diff.max() < 0.001}")

    # Test 2: Run EngineRunner with correct config
    print("\n=== Test 2: EngineRunner Full Test ===")

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
    )

    # Create KV cache
    kv_desc = model_desc.get_kv_cache_descriptor(
        num_blocks=64,
        block_size=16,
    )
    kv_cache = EngineKVCache(ctx, kv_desc)

    # Load weights using EngineWeightLoader
    print("Loading weights via EngineWeightLoader...")
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')

    print(f"Weights loaded:")
    print(f"  num_layers: {weights.num_layers}")
    print(f"  hidden_size: {weights.hidden_size}")
    print(f"  vocab_size: {weights.vocab_size}")
    print(f"  head_size (inferred): {weights.head_size}")
    print(f"  num_attention_heads (inferred): {weights.num_attention_heads}")
    print(f"  num_kv_heads (inferred): {weights.num_kv_heads}")

    # Verify weight loading for layer 0
    print("\n=== Verify Layer 0 Weights ===")
    layer0_w = weights.layers[0]

    # Check q_proj weight dimensions
    q_proj_hf = state_dict['model.layers.0.self_attn.q_proj.weight']
    print(f"HF q_proj shape: {q_proj_hf.shape}")

    if layer0_w.q_proj is not None:
        print(f"Engine q_proj buffer size: {layer0_w.q_proj.length()} bytes")
        expected_q_size = q_proj_hf.numel() * 2  # float16
        print(f"Expected q_proj size: {expected_q_size} bytes")
        print(f"Q_proj size match: {layer0_w.q_proj.length() == expected_q_size}")

    if layer0_w.qkv_proj is not None:
        print(f"Engine qkv_proj buffer size: {layer0_w.qkv_proj.length()} bytes")

    # Create EngineRunner
    print("\n=== Creating EngineRunner ===")
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
    block_table[0, 0] = 0
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

        # Debug: Compare logits
        hf_logits_np = hf_last_logits.numpy()
        logits_diff = np.abs(engine_last_logits - hf_logits_np)
        print(f"\n  Logits max diff: {logits_diff.max():.4f}")
        print(f"  Logits mean diff: {logits_diff.mean():.4f}")

        # Check if logits look like initialization
        print(f"\n  Engine logits stats: min={engine_last_logits.min():.4f}, max={engine_last_logits.max():.4f}, mean={engine_last_logits.mean():.4f}")
        print(f"  HF logits stats: min={hf_logits_np.min():.4f}, max={hf_logits_np.max():.4f}, mean={hf_logits_np.mean():.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Debug embedding weight loading path."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from vllm_apple.engine.ops.embedding import EngineEmbedding
    from vllm_apple.engine.step import EngineStepContext
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

    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)

    print(f"Token IDs: {token_ids.tolist()}")

    # Reference embedding
    embed_weight = state_dict['model.embed_tokens.weight']
    hf_embed = embed_weight[token_ids].cpu().numpy().astype(np.float16)
    print(f"\nHF embedding [0, :5]: {hf_embed[0, :5]}")
    print(f"HF embed_weight shape: {embed_weight.shape}")

    ctx = MetalEngineContext()

    # Method 1: Manual weight loading (WORKING in test_engine_runner_embed.py)
    print("\n=== Method 1: Manual Weight Loading ===")
    embed_weight_np = embed_weight.cpu().numpy().astype(np.float16)
    embed_weight_buf_manual = ctx.device.newBufferWithBytes_length_options_(
        embed_weight_np.tobytes(),
        embed_weight_np.nbytes,
        MTLResourceStorageModeShared
    )
    print(f"Manual buffer size: {embed_weight_buf_manual.length()} bytes")
    print(f"Expected size: {embed_weight_np.nbytes} bytes")

    engine_embed_manual = EngineEmbedding(
        context=ctx,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
    )
    engine_embed_manual.set_weights(embed_weight_buf_manual)

    token_ids_int32 = token_ids.to(torch.int32).numpy()
    token_ids_buf = ctx.device.newBufferWithBytes_length_options_(
        token_ids_int32.tobytes(),
        token_ids_int32.nbytes,
        MTLResourceStorageModeShared
    )
    embed_out_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * config.hidden_size * 2,
        MTLResourceStorageModeShared
    )

    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        engine_embed_manual.encode(
            step_ctx=step_ctx,
            token_ids=token_ids_buf,
            output_buffer=embed_out_buf,
            num_tokens=num_tokens,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    embed_view = embed_out_buf.contents().as_buffer(num_tokens * config.hidden_size * 2)
    engine_embed_manual_out = np.frombuffer(embed_view, dtype=np.float16).reshape(num_tokens, config.hidden_size).copy()
    print(f"Manual embedding [0, :5]: {engine_embed_manual_out[0, :5]}")
    print(f"Manual max diff: {np.abs(engine_embed_manual_out - hf_embed).max():.6f}")

    # Method 2: Via EngineWeightLoader
    print("\n=== Method 2: Via EngineWeightLoader ===")
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')

    print(f"Loader embedding buffer: {weights.embedding}")
    print(f"Loader embedding buffer size: {weights.embedding.length()} bytes")
    print(f"Expected size: {config.vocab_size * config.hidden_size * 2} bytes")

    # Read back first few elements from the weights.embedding buffer
    embed_loaded_view = weights.embedding.contents().as_buffer(min(100, weights.embedding.length()))
    embed_loaded_sample = np.frombuffer(embed_loaded_view, dtype=np.float16)[:10]
    print(f"Loaded embed buffer [0:10]: {embed_loaded_sample}")

    # Compare with expected values
    expected_sample = embed_weight_np.flatten()[:10]
    print(f"Expected embed buffer [0:10]: {expected_sample}")

    # Now use this buffer with EngineEmbedding
    engine_embed_loader = EngineEmbedding(
        context=ctx,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
    )
    engine_embed_loader.set_weights(weights.embedding)

    embed_out_buf2 = ctx.device.newBufferWithLength_options_(
        num_tokens * config.hidden_size * 2,
        MTLResourceStorageModeShared
    )

    with EngineStepContext(ctx, step_id=2, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        engine_embed_loader.encode(
            step_ctx=step_ctx,
            token_ids=token_ids_buf,
            output_buffer=embed_out_buf2,
            num_tokens=num_tokens,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    embed_view2 = embed_out_buf2.contents().as_buffer(num_tokens * config.hidden_size * 2)
    engine_embed_loader_out = np.frombuffer(embed_view2, dtype=np.float16).reshape(num_tokens, config.hidden_size).copy()
    print(f"Loader embedding [0, :5]: {engine_embed_loader_out[0, :5]}")
    print(f"Loader max diff: {np.abs(engine_embed_loader_out - hf_embed).max():.6f}")

    # Method 3: Via EngineRunner
    print("\n=== Method 3: Via EngineRunner ===")
    from vllm_apple.engine.runner import EngineRunner
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import ModelDescriptor

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
    kv_desc = model_desc.get_kv_cache_descriptor(num_blocks=64, block_size=16)
    kv_cache = EngineKVCache(ctx, kv_desc)

    runner = EngineRunner(
        context=ctx,
        model_desc=model_desc,
        weights=weights,
        kv_cache=kv_cache,
    )

    print(f"Runner embedding weight buffer: {runner._embedding._weight_buffer}")
    print(f"Is same buffer as weights.embedding? {runner._embedding._weight_buffer is weights.embedding}")

    embed_out_buf3 = ctx.device.newBufferWithLength_options_(
        num_tokens * config.hidden_size * 2,
        MTLResourceStorageModeShared
    )

    with EngineStepContext(ctx, step_id=3, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        runner._embedding.encode(
            step_ctx=step_ctx,
            token_ids=token_ids_buf,
            output_buffer=embed_out_buf3,
            num_tokens=num_tokens,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    embed_view3 = embed_out_buf3.contents().as_buffer(num_tokens * config.hidden_size * 2)
    engine_embed_runner_out = np.frombuffer(embed_view3, dtype=np.float16).reshape(num_tokens, config.hidden_size).copy()
    print(f"Runner embedding [0, :5]: {engine_embed_runner_out[0, :5]}")
    print(f"Runner max diff: {np.abs(engine_embed_runner_out - hf_embed).max():.6f}")

    # Summary
    print("\n=== Summary ===")
    print(f"Method 1 (Manual): PASS={np.abs(engine_embed_manual_out - hf_embed).max() < 0.001}")
    print(f"Method 2 (Loader): PASS={np.abs(engine_embed_loader_out - hf_embed).max() < 0.001}")
    print(f"Method 3 (Runner): PASS={np.abs(engine_embed_runner_out - hf_embed).max() < 0.001}")


if __name__ == "__main__":
    main()

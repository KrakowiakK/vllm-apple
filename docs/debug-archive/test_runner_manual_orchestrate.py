#!/usr/bin/env python3
"""Use EngineRunner's ops but manually orchestrate like test_full_model_logits.py."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.runner import EngineRunner
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import ModelDescriptor
    from vllm_apple.engine.tensor import EngineTensor, EngineDType
    from Metal import MTLResourceStorageModeShared

    model_name = 'mistralai/Devstral-Small-2505'
    num_layers = 40

    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    config = model.config
    state_dict = model.state_dict()

    prompt = "def add(a, b):"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)

    # Get HF prediction
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

    ctx = MetalEngineContext()

    model_desc = ModelDescriptor(
        num_layers=num_layers,
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
    kv_desc = model_desc.get_kv_cache_descriptor(num_blocks=64, block_size=16)
    kv_cache = EngineKVCache(ctx, kv_desc)

    # Load weights via EngineWeightLoader
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')

    # Create EngineRunner to get the ops
    runner = EngineRunner(
        context=ctx,
        model_desc=model_desc,
        weights=weights,
        kv_cache=kv_cache,
    )

    # Extract config values
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = config.head_dim
    block_size = 16

    q_size = num_heads * head_size
    k_size = num_kv_heads * head_size
    qkv_size = q_size + 2 * k_size

    # Get HF embeddings (like test_full_model_logits.py does)
    embed_weight = state_dict['model.embed_tokens.weight']
    embed_np = embed_weight[token_ids].cpu().numpy().astype(np.float16)
    embed_np = np.ascontiguousarray(embed_np)

    # Create buffers (like test_full_model_logits.py)
    hidden_a = ctx.device.newBufferWithBytes_length_options_(embed_np.tobytes(), embed_np.nbytes, MTLResourceStorageModeShared)
    hidden_b = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    residual_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    normed_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    qkv_buf = ctx.device.newBufferWithLength_options_(num_tokens * qkv_size * 2, MTLResourceStorageModeShared)
    attn_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * q_size * 2, MTLResourceStorageModeShared)
    o_proj_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    mlp_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    final_normed_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    logits_buf = ctx.device.newBufferWithLength_options_(num_tokens * config.vocab_size * 2, MTLResourceStorageModeShared)

    positions = np.arange(num_tokens, dtype=np.int32)
    positions_buf = ctx.device.newBufferWithBytes_length_options_(positions.tobytes(), positions.nbytes, MTLResourceStorageModeShared)
    slot_mapping = np.arange(num_tokens, dtype=np.int32)
    slot_buf = ctx.device.newBufferWithBytes_length_options_(slot_mapping.tobytes(), slot_mapping.nbytes, MTLResourceStorageModeShared)
    block_table = np.array([[0]], dtype=np.int32)
    block_table_buf = ctx.device.newBufferWithBytes_length_options_(block_table.tobytes(), block_table.nbytes, MTLResourceStorageModeShared)
    token_to_seq = np.zeros(num_tokens, dtype=np.int32)
    token_to_seq_buf = ctx.device.newBufferWithBytes_length_options_(token_to_seq.tobytes(), token_to_seq.nbytes, MTLResourceStorageModeShared)

    print("\n=== Running Full Model with EngineRunner Ops ===")

    current_hidden = hidden_a
    other_hidden = hidden_b

    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        for layer_idx in range(num_layers):
            layer_ops = runner._layers[layer_idx]
            k_cache, v_cache = kv_cache.get_buffers(layer_idx)

            # Copy hidden to residual
            runner._elementwise.encode_copy(step_ctx=step_ctx, input_buffer=current_hidden, output=residual_buf, num_elements=num_tokens * hidden_size)

            # Input LayerNorm (to normed_buf, NOT in-place)
            layer_ops.input_norm.encode(step_ctx=step_ctx, input_buffer=current_hidden, output_buffer=normed_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            # QKV Projection
            layer_ops.qkv_proj.encode(step_ctx=step_ctx, hidden_states=normed_buf, qkv_output=qkv_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            # RoPE
            k_tensor = EngineTensor(buffer=qkv_buf, shape=(num_tokens, num_kv_heads, head_size), dtype=EngineDType.FLOAT16, offset=num_tokens * q_size * 2)
            layer_ops.rope.encode(step_ctx=step_ctx, query=qkv_buf, key=k_tensor, positions=positions_buf, num_tokens=num_tokens, max_position_in_batch=num_tokens - 1)
            step_ctx.memory_barrier()

            # KV Write
            layer_ops.kv_write.encode_prefill(
                step_ctx=step_ctx,
                new_keys_buffer=qkv_buf, new_values_buffer=qkv_buf,
                key_buffer=k_cache, value_buffer=v_cache,
                slot_mapping_buffer=slot_buf, num_tokens=num_tokens,
                new_keys_offset=num_tokens * q_size * 2,
                new_values_offset=num_tokens * (q_size + k_size) * 2,
            )
            step_ctx.memory_barrier()

            # Attention
            layer_ops.attention.encode_prefill(
                step_ctx=step_ctx,
                query_buffer=qkv_buf, key_buffer=k_cache, value_buffer=v_cache,
                block_table_buffer=block_table_buf, token_to_seq_buffer=token_to_seq_buf,
                positions_buffer=positions_buf, output_buffer=attn_out_buf,
                num_tokens=num_tokens, num_seqs=1, max_seq_len=num_tokens, max_blocks_per_seq=1,
            )
            step_ctx.memory_barrier()

            # O Projection
            layer_ops.o_proj.encode(step_ctx=step_ctx, attn_output=attn_out_buf, output_buffer=o_proj_out_buf, num_tokens=num_tokens)

            # Residual add
            runner._elementwise.encode_residual_add(step_ctx=step_ctx, x=o_proj_out_buf, residual=residual_buf, output=current_hidden, num_elements=num_tokens * hidden_size)
            step_ctx.memory_barrier()

            # Copy for MLP residual
            runner._elementwise.encode_copy(step_ctx=step_ctx, input_buffer=current_hidden, output=residual_buf, num_elements=num_tokens * hidden_size)

            # Post LayerNorm (to normed_buf)
            layer_ops.post_attn_norm.encode(step_ctx=step_ctx, input_buffer=current_hidden, output_buffer=normed_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            # MLP
            layer_ops.mlp.encode(step_ctx=step_ctx, hidden_states=normed_buf, output_buffer=mlp_out_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            # Final residual
            runner._elementwise.encode_residual_add(step_ctx=step_ctx, x=mlp_out_buf, residual=residual_buf, output=other_hidden, num_elements=num_tokens * hidden_size)
            step_ctx.memory_barrier()

            current_hidden, other_hidden = other_hidden, current_hidden

        # Final norm
        runner._final_norm.encode(step_ctx=step_ctx, input_buffer=current_hidden, output_buffer=final_normed_buf, num_tokens=num_tokens)
        step_ctx.memory_barrier()

        # LM head
        runner._lm_head.encode(step_ctx=step_ctx, hidden_states=final_normed_buf, output_buffer=logits_buf, num_tokens=num_tokens)

        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read logits
    logits_view = logits_buf.contents().as_buffer(num_tokens * config.vocab_size * 2)
    engine_logits = np.frombuffer(logits_view, dtype=np.float16).reshape(num_tokens, config.vocab_size).copy()

    engine_last_logits = engine_logits[-1]

    if np.any(np.isnan(engine_last_logits)):
        print("ERROR: Engine logits contain NaN!")
        return
    if np.any(np.isinf(engine_last_logits)):
        print("ERROR: Engine logits contain Inf!")
        return

    logits_max = np.max(engine_last_logits)
    exp_logits = np.exp(engine_last_logits - logits_max)
    engine_probs = exp_logits / np.sum(exp_logits)

    top5_indices = np.argsort(engine_probs)[-5:][::-1]
    top5_probs = engine_probs[top5_indices]
    engine_pred_token = top5_indices[0]

    print(f"\n=== Engine Prediction (Runner ops, manual orchestration) ===")
    print(f"Top 5 tokens: {[tokenizer.decode([t]) for t in top5_indices.tolist()]}")
    print(f"Predicted: '{tokenizer.decode([engine_pred_token])}'")

    print(f"\n=== Comparison ===")
    print(f"HF predicted: '{tokenizer.decode([hf_pred_token])}' (id={hf_pred_token})")
    print(f"Engine predicted: '{tokenizer.decode([engine_pred_token])}' (id={engine_pred_token})")

    if engine_pred_token == hf_pred_token:
        print("\n✓ MATCH!")
    else:
        print("\n✗ MISMATCH!")
        logits_diff = np.abs(engine_last_logits - hf_last_logits.numpy())
        print(f"Logits max diff: {logits_diff.max():.4f}")


if __name__ == "__main__":
    main()

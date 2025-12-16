#!/usr/bin/env python3
"""Test full model including LM head to verify logits are reasonable."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Test full model logits."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.qkv import EngineQKVProjection, EngineOProjection
    from vllm_apple.engine.ops.rmsnorm import EngineRMSNorm
    from vllm_apple.engine.ops.elementwise import EngineElementwiseOps, EngineRoPE
    from vllm_apple.engine.ops.mlp import EngineMLP
    from vllm_apple.engine.ops.attention import PagedAttentionOp
    from vllm_apple.engine.ops.kv_write import KVWriteOp
    from vllm_apple.engine.ops.lm_head import EngineLMHead
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import KVCacheDescriptor
    from vllm_apple.engine.tensor import EngineTensor, EngineDType
    from Metal import MTLResourceStorageModeShared

    model_name = 'mistralai/Devstral-Small-2505'
    num_layers = 40

    print(f"Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    config = model.config
    print(f"Model: {config.hidden_size} hidden, {config.num_attention_heads} heads, vocab={config.vocab_size}")

    prompt = "def add(a, b):"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)
    print(f"Input: {num_tokens} tokens: {token_ids.tolist()}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")

    # Get HF prediction
    with torch.no_grad():
        hf_outputs = model(**inputs)
        hf_logits = hf_outputs.logits[0]  # [seq_len, vocab]
        hf_last_logits = hf_logits[-1]  # Last token logits
        hf_probs = torch.softmax(hf_last_logits, dim=-1)
        hf_top5 = torch.topk(hf_probs, 5)
        hf_pred_token = hf_top5.indices[0].item()

    print(f"\n=== HuggingFace Prediction ===")
    print(f"Top 5 tokens: {[tokenizer.decode([t]) for t in hf_top5.indices.tolist()]}")
    print(f"Top 5 probs: {hf_top5.values.tolist()}")
    print(f"Predicted next token: '{tokenizer.decode([hf_pred_token])}'")

    # Setup engine
    ctx = MetalEngineContext()

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = config.head_dim
    intermediate_size = config.intermediate_size
    vocab_size = config.vocab_size
    rope_theta = config.rope_theta
    block_size = 16

    q_size = num_heads * head_size
    k_size = num_kv_heads * head_size
    qkv_size = q_size + 2 * k_size

    # Create KV cache
    kv_desc = KVCacheDescriptor(
        num_blocks=16,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_layers=num_layers,
    )
    kv_cache = EngineKVCache(ctx, kv_desc)

    state_dict = model.state_dict()

    def to_buffer(tensor, name):
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        tensor = tensor.cpu().contiguous()
        np_arr = tensor.numpy()
        np_arr = np.ascontiguousarray(np_arr)
        return ctx.device.newBufferWithBytes_length_options_(
            np_arr.tobytes(), np_arr.nbytes, MTLResourceStorageModeShared
        )

    # Create layers
    print("\nLoading layer weights...")
    layers = []
    for i in range(num_layers):
        layer = {
            'input_norm': EngineRMSNorm(ctx, hidden_size=hidden_size, eps=1e-5),
            'qkv_proj': EngineQKVProjection(
                ctx, hidden_size=hidden_size, num_heads=num_heads,
                num_kv_heads=num_kv_heads, head_size=head_size, fused=False,
            ),
            'rope': EngineRoPE(
                ctx, head_size=head_size, num_heads=num_heads,
                num_kv_heads=num_kv_heads, max_position=8192, base=rope_theta,
            ),
            'kv_write': KVWriteOp(ctx, num_kv_heads=num_kv_heads, head_size=head_size, block_size=block_size),
            'attn_op': PagedAttentionOp(
                ctx, num_kv_heads=num_kv_heads, num_query_heads=num_heads,
                head_size=head_size, block_size=block_size,
            ),
            'o_proj': EngineOProjection(ctx, hidden_size=hidden_size, num_heads=num_heads, head_size=head_size),
            'post_norm': EngineRMSNorm(ctx, hidden_size=hidden_size, eps=1e-5),
            'mlp': EngineMLP(ctx, hidden_size=hidden_size, intermediate_size=intermediate_size, gated=True, activation="silu"),
        }

        layer['input_norm'].set_weights(to_buffer(state_dict[f'model.layers.{i}.input_layernorm.weight'], f"l{i}_in"))
        layer['qkv_proj'].set_weights(
            q_weight=to_buffer(state_dict[f'model.layers.{i}.self_attn.q_proj.weight'], f"l{i}_q"),
            k_weight=to_buffer(state_dict[f'model.layers.{i}.self_attn.k_proj.weight'], f"l{i}_k"),
            v_weight=to_buffer(state_dict[f'model.layers.{i}.self_attn.v_proj.weight'], f"l{i}_v"),
        )
        layer['o_proj'].set_weights(to_buffer(state_dict[f'model.layers.{i}.self_attn.o_proj.weight'], f"l{i}_o"))
        layer['post_norm'].set_weights(to_buffer(state_dict[f'model.layers.{i}.post_attention_layernorm.weight'], f"l{i}_post"))
        layer['mlp'].set_weights(
            gate_proj=to_buffer(state_dict[f'model.layers.{i}.mlp.gate_proj.weight'], f"l{i}_gate"),
            up_proj=to_buffer(state_dict[f'model.layers.{i}.mlp.up_proj.weight'], f"l{i}_up"),
            down_proj=to_buffer(state_dict[f'model.layers.{i}.mlp.down_proj.weight'], f"l{i}_down"),
        )

        layers.append(layer)

    # Final norm and LM head
    final_norm = EngineRMSNorm(ctx, hidden_size=hidden_size, eps=1e-5)
    final_norm.set_weights(to_buffer(state_dict['model.norm.weight'], "final_norm"))

    lm_head = EngineLMHead(ctx, hidden_size=hidden_size, vocab_size=vocab_size)
    lm_head.set_weights(to_buffer(state_dict['lm_head.weight'], "lm_head"))

    elementwise = EngineElementwiseOps(ctx)

    # Create buffers
    embed_weight = state_dict['model.embed_tokens.weight']
    embed_np = embed_weight[token_ids].cpu().numpy().astype(np.float16)
    embed_np = np.ascontiguousarray(embed_np)

    hidden_a = ctx.device.newBufferWithBytes_length_options_(embed_np.tobytes(), embed_np.nbytes, MTLResourceStorageModeShared)
    hidden_b = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    residual_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    normed_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    qkv_buf = ctx.device.newBufferWithLength_options_(num_tokens * qkv_size * 2, MTLResourceStorageModeShared)
    attn_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * q_size * 2, MTLResourceStorageModeShared)
    o_proj_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    mlp_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    final_normed_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    logits_buf = ctx.device.newBufferWithLength_options_(num_tokens * vocab_size * 2, MTLResourceStorageModeShared)

    positions = np.arange(num_tokens, dtype=np.int32)
    positions_buf = ctx.device.newBufferWithBytes_length_options_(positions.tobytes(), positions.nbytes, MTLResourceStorageModeShared)
    slot_mapping = np.arange(num_tokens, dtype=np.int32)
    slot_buf = ctx.device.newBufferWithBytes_length_options_(slot_mapping.tobytes(), slot_mapping.nbytes, MTLResourceStorageModeShared)
    block_table = np.array([[0]], dtype=np.int32)
    block_table_buf = ctx.device.newBufferWithBytes_length_options_(block_table.tobytes(), block_table.nbytes, MTLResourceStorageModeShared)
    token_to_seq = np.zeros(num_tokens, dtype=np.int32)
    token_to_seq_buf = ctx.device.newBufferWithBytes_length_options_(token_to_seq.tobytes(), token_to_seq.nbytes, MTLResourceStorageModeShared)

    print("\n=== Running Full Model ===")

    current_hidden = hidden_a
    other_hidden = hidden_b

    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        for layer_idx, layer in enumerate(layers):
            k_cache, v_cache = kv_cache.get_buffers(layer_idx)

            elementwise.encode_copy(step_ctx=step_ctx, input_buffer=current_hidden, output=residual_buf, num_elements=num_tokens * hidden_size)
            layer['input_norm'].encode(step_ctx=step_ctx, input_buffer=current_hidden, output_buffer=normed_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            layer['qkv_proj'].encode(step_ctx=step_ctx, hidden_states=normed_buf, qkv_output=qkv_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            k_tensor = EngineTensor(buffer=qkv_buf, shape=(num_tokens, num_kv_heads, head_size), dtype=EngineDType.FLOAT16, offset=num_tokens * q_size * 2)
            layer['rope'].encode(step_ctx=step_ctx, query=qkv_buf, key=k_tensor, positions=positions_buf, num_tokens=num_tokens, max_position_in_batch=num_tokens - 1)
            step_ctx.memory_barrier()

            layer['kv_write'].encode_prefill(
                step_ctx=step_ctx,
                new_keys_buffer=qkv_buf, new_values_buffer=qkv_buf,
                key_buffer=k_cache, value_buffer=v_cache,
                slot_mapping_buffer=slot_buf, num_tokens=num_tokens,
                new_keys_offset=num_tokens * q_size * 2,
                new_values_offset=num_tokens * (q_size + k_size) * 2,
            )
            step_ctx.memory_barrier()

            layer['attn_op'].encode_prefill(
                step_ctx=step_ctx,
                query_buffer=qkv_buf, key_buffer=k_cache, value_buffer=v_cache,
                block_table_buffer=block_table_buf, token_to_seq_buffer=token_to_seq_buf,
                positions_buffer=positions_buf, output_buffer=attn_out_buf,
                num_tokens=num_tokens, num_seqs=1, max_seq_len=num_tokens, max_blocks_per_seq=1,
            )
            step_ctx.memory_barrier()

            layer['o_proj'].encode(step_ctx=step_ctx, attn_output=attn_out_buf, output_buffer=o_proj_out_buf, num_tokens=num_tokens)
            elementwise.encode_residual_add(step_ctx=step_ctx, x=o_proj_out_buf, residual=residual_buf, output=current_hidden, num_elements=num_tokens * hidden_size)
            step_ctx.memory_barrier()

            elementwise.encode_copy(step_ctx=step_ctx, input_buffer=current_hidden, output=residual_buf, num_elements=num_tokens * hidden_size)
            layer['post_norm'].encode(step_ctx=step_ctx, input_buffer=current_hidden, output_buffer=normed_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            layer['mlp'].encode(step_ctx=step_ctx, hidden_states=normed_buf, output_buffer=mlp_out_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            elementwise.encode_residual_add(step_ctx=step_ctx, x=mlp_out_buf, residual=residual_buf, output=other_hidden, num_elements=num_tokens * hidden_size)
            step_ctx.memory_barrier()

            current_hidden, other_hidden = other_hidden, current_hidden

        # Final norm
        final_norm.encode(step_ctx=step_ctx, input_buffer=current_hidden, output_buffer=final_normed_buf, num_tokens=num_tokens)
        step_ctx.memory_barrier()

        # LM head
        lm_head.encode(step_ctx=step_ctx, hidden_states=final_normed_buf, output_buffer=logits_buf, num_tokens=num_tokens)

        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read logits
    logits_view = logits_buf.contents().as_buffer(num_tokens * vocab_size * 2)
    engine_logits = np.frombuffer(logits_view, dtype=np.float16).reshape(num_tokens, vocab_size).copy()

    # Get last token logits
    engine_last_logits = engine_logits[-1]

    print(f"\n=== Engine Prediction ===")

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

    print(f"Top 5 tokens: {[tokenizer.decode([t]) for t in top5_indices.tolist()]}")
    print(f"Top 5 probs: {top5_probs.tolist()}")
    print(f"Predicted next token: '{tokenizer.decode([engine_pred_token])}'")

    # Compare
    print(f"\n=== Comparison ===")
    print(f"HF predicted: '{tokenizer.decode([hf_pred_token])}' (id={hf_pred_token})")
    print(f"Engine predicted: '{tokenizer.decode([engine_pred_token])}' (id={engine_pred_token})")

    logits_diff = np.abs(engine_last_logits - hf_last_logits.numpy())
    print(f"\nLogits max diff: {logits_diff.max():.4f}")
    print(f"Logits mean diff: {logits_diff.mean():.4f}")

    if engine_pred_token == hf_pred_token:
        print("\n✓ MATCH! Engine and HF predict the same token!")
    else:
        print("\n✗ MISMATCH! Engine and HF predict different tokens.")
        # Check if HF's prediction is in engine's top 5
        if hf_pred_token in top5_indices:
            rank = np.where(top5_indices == hf_pred_token)[0][0] + 1
            print(f"  HF's prediction is rank {rank} in engine's predictions")


if __name__ == "__main__":
    main()

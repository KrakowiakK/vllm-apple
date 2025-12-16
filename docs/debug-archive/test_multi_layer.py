#!/usr/bin/env python3
"""Test multiple transformer layers to check error accumulation."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Test multiple layers."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.qkv import EngineQKVProjection, EngineOProjection
    from vllm_apple.engine.ops.rmsnorm import EngineRMSNorm
    from vllm_apple.engine.ops.elementwise import EngineElementwiseOps, EngineRoPE
    from vllm_apple.engine.ops.mlp import EngineMLP
    from vllm_apple.engine.ops.attention import PagedAttentionOp
    from vllm_apple.engine.ops.kv_write import KVWriteOp
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import KVCacheDescriptor
    from vllm_apple.engine.tensor import EngineTensor, EngineDType
    from Metal import MTLResourceStorageModeShared

    model_name = 'mistralai/Devstral-Small-2505'
    num_test_layers = 40  # Test all layers

    print(f"Loading HuggingFace model (testing first {num_test_layers} layers)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    config = model.config
    print(f"Model: {config.hidden_size} hidden, {config.num_attention_heads} heads, {config.num_key_value_heads} kv_heads")

    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)
    print(f"Input: {num_tokens} tokens: {token_ids.tolist()}")

    # Capture HF outputs after each layer
    hf_outputs = {}

    def make_hook(name):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                hf_outputs[name] = out[0].detach().clone()
            else:
                hf_outputs[name] = out.detach().clone()
        return hook

    for i in range(num_test_layers):
        model.model.layers[i].register_forward_hook(make_hook(f'layer{i}'))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Setup engine
    ctx = MetalEngineContext()

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = config.head_dim
    intermediate_size = config.intermediate_size
    rope_theta = config.rope_theta
    block_size = 16

    q_size = num_heads * head_size
    k_size = num_kv_heads * head_size
    v_size = num_kv_heads * head_size
    qkv_size = q_size + k_size + v_size

    # Create KV cache
    kv_desc = KVCacheDescriptor(
        num_blocks=8,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_layers=num_test_layers,
    )
    kv_cache = EngineKVCache(ctx, kv_desc)

    # Create ops for each layer
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

    layers = []
    for i in range(num_test_layers):
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

        # Load weights
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

    elementwise = EngineElementwiseOps(ctx)

    # Create buffers
    hf_embed = outputs.hidden_states[0][0]
    embed_np = hf_embed.cpu().numpy().astype(np.float16)
    embed_np = np.ascontiguousarray(embed_np)

    hidden_a = ctx.device.newBufferWithBytes_length_options_(
        embed_np.tobytes(), embed_np.nbytes, MTLResourceStorageModeShared
    )
    hidden_b = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    residual_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    normed_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    qkv_buf = ctx.device.newBufferWithLength_options_(num_tokens * qkv_size * 2, MTLResourceStorageModeShared)
    attn_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * q_size * 2, MTLResourceStorageModeShared)
    o_proj_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    mlp_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)

    positions = np.arange(num_tokens, dtype=np.int32)
    positions_buf = ctx.device.newBufferWithBytes_length_options_(positions.tobytes(), positions.nbytes, MTLResourceStorageModeShared)
    slot_mapping = np.arange(num_tokens, dtype=np.int32)
    slot_buf = ctx.device.newBufferWithBytes_length_options_(slot_mapping.tobytes(), slot_mapping.nbytes, MTLResourceStorageModeShared)
    block_table = np.array([[0]], dtype=np.int32)
    block_table_buf = ctx.device.newBufferWithBytes_length_options_(block_table.tobytes(), block_table.nbytes, MTLResourceStorageModeShared)
    token_to_seq = np.zeros(num_tokens, dtype=np.int32)
    token_to_seq_buf = ctx.device.newBufferWithBytes_length_options_(token_to_seq.tobytes(), token_to_seq.nbytes, MTLResourceStorageModeShared)

    print(f"\n=== Running {num_test_layers} Layers ===")

    current_hidden = hidden_a
    other_hidden = hidden_b

    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        for layer_idx, layer in enumerate(layers):
            k_cache, v_cache = kv_cache.get_buffers(layer_idx)

            # 1. Copy hidden to residual
            elementwise.encode_copy(step_ctx=step_ctx, input_buffer=current_hidden, output=residual_buf, num_elements=num_tokens * hidden_size)

            # 2. Input LayerNorm
            layer['input_norm'].encode(step_ctx=step_ctx, input_buffer=current_hidden, output_buffer=normed_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            # 3. QKV Projection
            layer['qkv_proj'].encode(step_ctx=step_ctx, hidden_states=normed_buf, qkv_output=qkv_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            # 4. RoPE
            k_tensor = EngineTensor(buffer=qkv_buf, shape=(num_tokens, num_kv_heads, head_size), dtype=EngineDType.FLOAT16, offset=num_tokens * q_size * 2)
            layer['rope'].encode(step_ctx=step_ctx, query=qkv_buf, key=k_tensor, positions=positions_buf, num_tokens=num_tokens, max_position_in_batch=num_tokens - 1)
            step_ctx.memory_barrier()

            # 5. KV Write
            layer['kv_write'].encode_prefill(
                step_ctx=step_ctx,
                new_keys_buffer=qkv_buf, new_values_buffer=qkv_buf,
                key_buffer=k_cache, value_buffer=v_cache,
                slot_mapping_buffer=slot_buf, num_tokens=num_tokens,
                new_keys_offset=num_tokens * q_size * 2,
                new_values_offset=num_tokens * (q_size + k_size) * 2,
            )
            step_ctx.memory_barrier()

            # 6. Attention
            layer['attn_op'].encode_prefill(
                step_ctx=step_ctx,
                query_buffer=qkv_buf, key_buffer=k_cache, value_buffer=v_cache,
                block_table_buffer=block_table_buf, token_to_seq_buffer=token_to_seq_buf,
                positions_buffer=positions_buf, output_buffer=attn_out_buf,
                num_tokens=num_tokens, num_seqs=1, max_seq_len=num_tokens, max_blocks_per_seq=1,
            )
            step_ctx.memory_barrier()

            # 7. O Projection
            layer['o_proj'].encode(step_ctx=step_ctx, attn_output=attn_out_buf, output_buffer=o_proj_out_buf, num_tokens=num_tokens)

            # 8. Residual add
            elementwise.encode_residual_add(step_ctx=step_ctx, x=o_proj_out_buf, residual=residual_buf, output=current_hidden, num_elements=num_tokens * hidden_size)
            step_ctx.memory_barrier()

            # 9. Copy for MLP residual
            elementwise.encode_copy(step_ctx=step_ctx, input_buffer=current_hidden, output=residual_buf, num_elements=num_tokens * hidden_size)

            # 10. Post LayerNorm
            layer['post_norm'].encode(step_ctx=step_ctx, input_buffer=current_hidden, output_buffer=normed_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            # 11. MLP
            layer['mlp'].encode(step_ctx=step_ctx, hidden_states=normed_buf, output_buffer=mlp_out_buf, num_tokens=num_tokens)
            step_ctx.memory_barrier()

            # 12. Final residual
            elementwise.encode_residual_add(step_ctx=step_ctx, x=mlp_out_buf, residual=residual_buf, output=other_hidden, num_elements=num_tokens * hidden_size)
            step_ctx.memory_barrier()

            # Swap buffers
            current_hidden, other_hidden = other_hidden, current_hidden

        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read final output
    output_view = current_hidden.contents().as_buffer(num_tokens * hidden_size * 2)
    engine_output = np.frombuffer(output_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    # Compare with HF output after last tested layer
    hf_last = hf_outputs[f'layer{num_test_layers - 1}'][0].numpy()

    print(f"\n=== Results After {num_test_layers} Layers ===")
    print(f"Engine output [0, :5]: {engine_output[0, :5]}")
    print(f"HF output [0, :5]: {hf_last[0, :5]}")

    max_diff = np.abs(engine_output - hf_last).max()
    mean_diff = np.abs(engine_output - hf_last).mean()

    print(f"\nMax diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")

    if np.any(np.isnan(engine_output)):
        print("ERROR: Engine output contains NaN!")
    if np.any(np.isinf(engine_output)):
        print("ERROR: Engine output contains Inf!")

    print(f"\nResult: {'PASS' if max_diff < 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()

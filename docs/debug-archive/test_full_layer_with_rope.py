#!/usr/bin/env python3
"""Test full transformer layer with RoPE against HuggingFace."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Test full layer including RoPE and attention."""
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

    # Simple prompt
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)

    print(f"\n=== Input ===")
    print(f"Token IDs: {token_ids.tolist()}")
    print(f"Num tokens: {num_tokens}")

    # Capture HF outputs
    captured = {}

    def make_hook(name):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                captured[name] = out[0].detach().clone()
            else:
                captured[name] = out.detach().clone()
        return hook

    layer0 = model.model.layers[0]
    model.model.embed_tokens.register_forward_hook(make_hook('embedding'))
    layer0.register_forward_hook(make_hook('layer0_output'))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hf_embed = outputs.hidden_states[0][0]  # [num_tokens, hidden]
    hf_after_layer0 = outputs.hidden_states[1][0]  # [num_tokens, hidden]

    print(f"\nHF embedding shape: {hf_embed.shape}")
    print(f"HF after layer0 shape: {hf_after_layer0.shape}")
    print(f"HF after layer0 [0, :5]: {hf_after_layer0[0, :5]}")

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
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_layers=1,
    )
    kv_cache = EngineKVCache(ctx, kv_desc)

    # Create ops
    input_norm = EngineRMSNorm(ctx, hidden_size=hidden_size, eps=1e-5)
    qkv_proj = EngineQKVProjection(
        ctx,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        fused=False,
    )
    rope = EngineRoPE(
        ctx,
        head_size=head_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_position=8192,
        base=rope_theta,
    )
    kv_write = KVWriteOp(
        ctx,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
    )
    attn_op = PagedAttentionOp(
        ctx,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
    )
    o_proj = EngineOProjection(
        ctx,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_size=head_size,
    )
    post_norm = EngineRMSNorm(ctx, hidden_size=hidden_size, eps=1e-5)
    mlp = EngineMLP(
        ctx,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gated=True,
        activation="silu",
    )
    elementwise = EngineElementwiseOps(ctx)

    # Load weights
    state_dict = model.state_dict()

    def to_buffer(tensor, name):
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        tensor = tensor.cpu().contiguous()
        np_arr = tensor.numpy()
        np_arr = np.ascontiguousarray(np_arr)
        buf = ctx.device.newBufferWithBytes_length_options_(
            np_arr.tobytes(), np_arr.nbytes, MTLResourceStorageModeShared
        )
        return buf

    print("\n=== Loading weights ===")
    input_norm.set_weights(to_buffer(state_dict['model.layers.0.input_layernorm.weight'], "input_ln"))
    qkv_proj.set_weights(
        q_weight=to_buffer(state_dict['model.layers.0.self_attn.q_proj.weight'], "q"),
        k_weight=to_buffer(state_dict['model.layers.0.self_attn.k_proj.weight'], "k"),
        v_weight=to_buffer(state_dict['model.layers.0.self_attn.v_proj.weight'], "v"),
    )
    o_proj.set_weights(to_buffer(state_dict['model.layers.0.self_attn.o_proj.weight'], "o"))
    post_norm.set_weights(to_buffer(state_dict['model.layers.0.post_attention_layernorm.weight'], "post_ln"))
    mlp.set_weights(
        gate_proj=to_buffer(state_dict['model.layers.0.mlp.gate_proj.weight'], "gate"),
        up_proj=to_buffer(state_dict['model.layers.0.mlp.up_proj.weight'], "up"),
        down_proj=to_buffer(state_dict['model.layers.0.mlp.down_proj.weight'], "down"),
    )

    # Create buffers
    embed_np = hf_embed.cpu().numpy().astype(np.float16)
    embed_np = np.ascontiguousarray(embed_np)
    hidden_buf = ctx.device.newBufferWithBytes_length_options_(
        embed_np.tobytes(), embed_np.nbytes, MTLResourceStorageModeShared
    )
    residual_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * hidden_size * 2, MTLResourceStorageModeShared
    )
    normed_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * hidden_size * 2, MTLResourceStorageModeShared
    )
    qkv_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * qkv_size * 2, MTLResourceStorageModeShared
    )
    attn_out_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * q_size * 2, MTLResourceStorageModeShared
    )
    o_proj_out_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * hidden_size * 2, MTLResourceStorageModeShared
    )
    mlp_out_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * hidden_size * 2, MTLResourceStorageModeShared
    )
    final_out_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * hidden_size * 2, MTLResourceStorageModeShared
    )

    # Positions, slot mapping, block table
    positions = np.arange(num_tokens, dtype=np.int32)
    positions_buf = ctx.device.newBufferWithBytes_length_options_(
        positions.tobytes(), positions.nbytes, MTLResourceStorageModeShared
    )

    slot_mapping = np.arange(num_tokens, dtype=np.int32)  # Simple: token i -> slot i
    slot_buf = ctx.device.newBufferWithBytes_length_options_(
        slot_mapping.tobytes(), slot_mapping.nbytes, MTLResourceStorageModeShared
    )

    block_table = np.array([[0]], dtype=np.int32)  # Single sequence, single block
    block_table_buf = ctx.device.newBufferWithBytes_length_options_(
        block_table.tobytes(), block_table.nbytes, MTLResourceStorageModeShared
    )

    token_to_seq = np.zeros(num_tokens, dtype=np.int32)  # All tokens belong to seq 0
    token_to_seq_buf = ctx.device.newBufferWithBytes_length_options_(
        token_to_seq.tobytes(), token_to_seq.nbytes, MTLResourceStorageModeShared
    )

    seq_lens = np.array([num_tokens], dtype=np.int32)
    seq_lens_buf = ctx.device.newBufferWithBytes_length_options_(
        seq_lens.tobytes(), seq_lens.nbytes, MTLResourceStorageModeShared
    )

    k_cache, v_cache = kv_cache.get_buffers(0)

    k_offset = q_size * 2
    v_offset = k_offset + k_size * 2 * num_tokens

    print("\n=== Running Full Layer ===")

    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        # 1. Copy hidden to residual
        elementwise.encode_copy(
            step_ctx=step_ctx,
            input_buffer=hidden_buf,
            output=residual_buf,
            num_elements=num_tokens * hidden_size,
        )

        # 2. Input LayerNorm
        input_norm.encode(
            step_ctx=step_ctx,
            input_buffer=hidden_buf,
            output_buffer=normed_buf,
            num_tokens=num_tokens,
        )

        step_ctx.memory_barrier()

        # 3. QKV Projection
        qkv_proj.encode(
            step_ctx=step_ctx,
            hidden_states=normed_buf,
            qkv_output=qkv_buf,
            num_tokens=num_tokens,
        )

        step_ctx.memory_barrier()

        # 4. RoPE on Q and K
        k_tensor = EngineTensor(
            buffer=qkv_buf,
            shape=(num_tokens, num_kv_heads, head_size),
            dtype=EngineDType.FLOAT16,
            offset=num_tokens * q_size * 2,
        )

        rope.encode(
            step_ctx=step_ctx,
            query=qkv_buf,
            key=k_tensor,
            positions=positions_buf,
            num_tokens=num_tokens,
            max_position_in_batch=num_tokens - 1,
        )

        step_ctx.memory_barrier()

        # 5. KV Write
        kv_write.encode_prefill(
            step_ctx=step_ctx,
            new_keys_buffer=qkv_buf,
            new_values_buffer=qkv_buf,
            key_buffer=k_cache,
            value_buffer=v_cache,
            slot_mapping_buffer=slot_buf,
            num_tokens=num_tokens,
            new_keys_offset=num_tokens * q_size * 2,
            new_values_offset=num_tokens * (q_size + k_size) * 2,
        )

        step_ctx.memory_barrier()

        # 6. Attention
        attn_op.encode_prefill(
            step_ctx=step_ctx,
            query_buffer=qkv_buf,
            key_buffer=k_cache,
            value_buffer=v_cache,
            block_table_buffer=block_table_buf,
            token_to_seq_buffer=token_to_seq_buf,
            positions_buffer=positions_buf,
            output_buffer=attn_out_buf,
            num_tokens=num_tokens,
            num_seqs=1,
            max_seq_len=num_tokens,
            max_blocks_per_seq=1,
        )

        step_ctx.memory_barrier()

        # 7. O Projection
        o_proj.encode(
            step_ctx=step_ctx,
            attn_output=attn_out_buf,
            output_buffer=o_proj_out_buf,
            num_tokens=num_tokens,
        )

        # 8. Residual add (attn_out + residual -> hidden_buf)
        elementwise.encode_residual_add(
            step_ctx=step_ctx,
            x=o_proj_out_buf,
            residual=residual_buf,
            output=hidden_buf,
            num_elements=num_tokens * hidden_size,
        )

        step_ctx.memory_barrier()

        # 9. Copy hidden to residual for MLP
        elementwise.encode_copy(
            step_ctx=step_ctx,
            input_buffer=hidden_buf,
            output=residual_buf,
            num_elements=num_tokens * hidden_size,
        )

        # 10. Post-attention LayerNorm
        post_norm.encode(
            step_ctx=step_ctx,
            input_buffer=hidden_buf,
            output_buffer=normed_buf,
            num_tokens=num_tokens,
        )

        step_ctx.memory_barrier()

        # 11. MLP
        mlp.encode(
            step_ctx=step_ctx,
            hidden_states=normed_buf,
            output_buffer=mlp_out_buf,
            num_tokens=num_tokens,
        )

        step_ctx.memory_barrier()

        # 12. Final residual add
        elementwise.encode_residual_add(
            step_ctx=step_ctx,
            x=mlp_out_buf,
            residual=residual_buf,
            output=final_out_buf,
            num_elements=num_tokens * hidden_size,
        )

        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read output
    output_view = final_out_buf.contents().as_buffer(num_tokens * hidden_size * 2)
    engine_output = np.frombuffer(output_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    print(f"\n=== Comparison ===")
    print(f"Engine output shape: {engine_output.shape}")
    print(f"Engine output [0, :5]: {engine_output[0, :5]}")
    print(f"HF output [0, :5]: {hf_after_layer0[0, :5].numpy()}")

    max_diff = np.abs(engine_output - hf_after_layer0.numpy()).max()
    mean_diff = np.abs(engine_output - hf_after_layer0.numpy()).mean()

    print(f"\nMax diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")

    # Check for NaN/Inf
    if np.any(np.isnan(engine_output)):
        print("ERROR: Engine output contains NaN!")
    if np.any(np.isinf(engine_output)):
        print("ERROR: Engine output contains Inf!")

    print(f"\nResult: {'PASS' if max_diff < 1.0 else 'FAIL'}")

    # If fail, analyze difference distribution
    if max_diff >= 1.0:
        diff = np.abs(engine_output - hf_after_layer0.numpy())
        print(f"\nDifference analysis:")
        print(f"  95th percentile: {np.percentile(diff, 95):.6f}")
        print(f"  99th percentile: {np.percentile(diff, 99):.6f}")
        print(f"  Token with max diff: {np.unravel_index(diff.argmax(), diff.shape)}")


if __name__ == "__main__":
    main()

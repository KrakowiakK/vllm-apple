#!/usr/bin/env python3
"""Test EngineRunner layer by layer to find divergence point."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Test EngineRunner layer by layer."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.qkv import EngineQKVProjection, EngineOProjection
    from vllm_apple.engine.ops.rmsnorm import EngineRMSNorm
    from vllm_apple.engine.ops.elementwise import EngineElementwiseOps, EngineRoPE
    from vllm_apple.engine.ops.mlp import EngineMLP
    from vllm_apple.engine.ops.attention import PagedAttentionOp
    from vllm_apple.engine.ops.kv_write import KVWriteOp
    from vllm_apple.engine.ops.embedding import EngineEmbedding
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import KVCacheDescriptor, ModelDescriptor
    from vllm_apple.engine.tensor import EngineTensor, EngineDType
    from vllm_apple.engine.runner import EngineRunner
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
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = config.head_dim
    intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers

    print(f"Config: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, head_size={head_size}")

    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)

    print(f"Input: {num_tokens} tokens")

    # Capture HF layer outputs
    hf_layer_outputs = {}

    def make_hook(name):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                hf_layer_outputs[name] = out[0].detach().clone()
            else:
                hf_layer_outputs[name] = out.detach().clone()
        return hook

    model.model.embed_tokens.register_forward_hook(make_hook('embed'))
    for i in range(num_layers):
        model.model.layers[i].register_forward_hook(make_hook(f'layer{i}'))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    state_dict = model.state_dict()

    # Setup engine
    ctx = MetalEngineContext()

    model_desc = ModelDescriptor(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        intermediate_size=intermediate_size,
        vocab_size=config.vocab_size,
        rope_theta=config.rope_theta,
        max_position_embeddings=8192,
        architecture="mistral",
    )

    kv_desc = model_desc.get_kv_cache_descriptor(num_blocks=64, block_size=16)
    kv_cache = EngineKVCache(ctx, kv_desc)

    # Load weights
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')

    # Create EngineRunner
    runner = EngineRunner(
        context=ctx,
        model_desc=model_desc,
        weights=weights,
        kv_cache=kv_cache,
    )

    # ===== TEST 1: Compare embeddings =====
    print("\n=== Test 1: Embeddings ===")

    def to_buffer(tensor, name):
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        tensor = tensor.cpu().contiguous()
        np_arr = tensor.numpy()
        np_arr = np.ascontiguousarray(np_arr)
        return ctx.device.newBufferWithBytes_length_options_(
            np_arr.tobytes(), np_arr.nbytes, MTLResourceStorageModeShared
        )

    # HF embedding
    hf_embed = hf_layer_outputs['embed'][0].cpu().numpy()
    print(f"HF embed [0, :5]: {hf_embed[0, :5]}")

    # Engine embedding using runner's embedding op
    token_ids_int32 = token_ids.to(torch.int32).numpy()
    token_ids_buf = to_buffer(torch.from_numpy(token_ids_int32), "token_ids")
    embed_out_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * hidden_size * 2, MTLResourceStorageModeShared
    )

    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        runner._embedding.encode(
            step_ctx=step_ctx,
            token_ids=token_ids_buf,
            output_buffer=embed_out_buf,
            num_tokens=num_tokens,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    embed_view = embed_out_buf.contents().as_buffer(num_tokens * hidden_size * 2)
    engine_embed = np.frombuffer(embed_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    print(f"Engine embed [0, :5]: {engine_embed[0, :5]}")
    embed_diff = np.abs(engine_embed - hf_embed).max()
    print(f"Embed max diff: {embed_diff:.6f}")
    print(f"Embed PASS: {embed_diff < 0.001}")

    # ===== TEST 2: Run first layer only =====
    print("\n=== Test 2: Single Layer Comparison ===")

    # Create fresh KV cache for this test
    kv_cache2 = EngineKVCache(ctx, kv_desc)
    runner2 = EngineRunner(
        context=ctx,
        model_desc=model_desc,
        weights=weights,
        kv_cache=kv_cache2,
    )

    # Create a modified runner that only runs 1 layer
    # We'll manually run just the first layer

    q_size = num_heads * head_size
    k_size = num_kv_heads * head_size
    v_size = num_kv_heads * head_size
    qkv_size = q_size + k_size + v_size

    # Allocate buffers
    hidden_a = to_buffer(torch.from_numpy(hf_embed.astype(np.float16)), "hidden_a")
    hidden_b = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    residual_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    qkv_buf = ctx.device.newBufferWithLength_options_(num_tokens * qkv_size * 2, MTLResourceStorageModeShared)
    attn_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * q_size * 2, MTLResourceStorageModeShared)
    mlp_out_buf = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)

    # Positions, slots, etc.
    positions = np.arange(num_tokens, dtype=np.int32)
    positions_buf = to_buffer(torch.from_numpy(positions), "positions")
    slot_mapping = np.arange(num_tokens, dtype=np.int32)
    slot_buf = to_buffer(torch.from_numpy(slot_mapping), "slots")
    block_table = np.array([[0]], dtype=np.int32)
    block_table_buf = to_buffer(torch.from_numpy(block_table), "block_table")
    token_to_seq = np.zeros(num_tokens, dtype=np.int32)
    token_to_seq_buf = to_buffer(torch.from_numpy(token_to_seq), "token_to_seq")

    layer_ops = runner2._layers[0]
    k_cache, v_cache = kv_cache2.get_buffers(0)
    elementwise = runner2._elementwise

    with EngineStepContext(ctx, step_id=2, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        # 1. Copy hidden to residual
        elementwise.encode_copy(step_ctx=step_ctx, input_buffer=hidden_a, output=residual_buf, num_elements=num_tokens * hidden_size)

        # 2. Input LayerNorm (in-place)
        layer_ops.input_norm.encode(step_ctx=step_ctx, input_buffer=hidden_a, output_buffer=hidden_a, num_tokens=num_tokens)
        step_ctx.memory_barrier()

        # 3. QKV projection
        layer_ops.qkv_proj.encode(step_ctx=step_ctx, hidden_states=hidden_a, qkv_output=qkv_buf, num_tokens=num_tokens)
        step_ctx.memory_barrier()

        # 4. RoPE
        k_tensor = EngineTensor(buffer=qkv_buf, shape=(num_tokens, num_kv_heads, head_size), dtype=EngineDType.FLOAT16, offset=num_tokens * q_size * 2)
        layer_ops.rope.encode(step_ctx=step_ctx, query=qkv_buf, key=k_tensor, positions=positions_buf, num_tokens=num_tokens, max_position_in_batch=num_tokens - 1)
        step_ctx.memory_barrier()

        # 5. KV Write
        layer_ops.kv_write.encode_prefill(
            step_ctx=step_ctx,
            new_keys_buffer=qkv_buf, new_values_buffer=qkv_buf,
            key_buffer=k_cache, value_buffer=v_cache,
            slot_mapping_buffer=slot_buf, num_tokens=num_tokens,
            new_keys_offset=num_tokens * q_size * 2,
            new_values_offset=num_tokens * (q_size + k_size) * 2,
        )
        step_ctx.memory_barrier()

        # 6. Attention
        layer_ops.attention.encode_prefill(
            step_ctx=step_ctx,
            query_buffer=qkv_buf, key_buffer=k_cache, value_buffer=v_cache,
            block_table_buffer=block_table_buf, token_to_seq_buffer=token_to_seq_buf,
            positions_buffer=positions_buf, output_buffer=attn_out_buf,
            num_tokens=num_tokens, num_seqs=1, max_seq_len=num_tokens, max_blocks_per_seq=1,
        )
        step_ctx.memory_barrier()

        # 7. O projection
        layer_ops.o_proj.encode(step_ctx=step_ctx, attn_output=attn_out_buf, output_buffer=hidden_a, num_tokens=num_tokens)

        # 8. Residual add
        elementwise.encode_residual_add(step_ctx=step_ctx, x=hidden_a, residual=residual_buf, output=hidden_a, num_elements=num_tokens * hidden_size)
        step_ctx.memory_barrier()

        # 9. Copy for MLP residual
        elementwise.encode_copy(step_ctx=step_ctx, input_buffer=hidden_a, output=residual_buf, num_elements=num_tokens * hidden_size)

        # 10. Post norm
        layer_ops.post_attn_norm.encode(step_ctx=step_ctx, input_buffer=hidden_a, output_buffer=hidden_a, num_tokens=num_tokens)
        step_ctx.memory_barrier()

        # 11. MLP
        layer_ops.mlp.encode(step_ctx=step_ctx, hidden_states=hidden_a, output_buffer=mlp_out_buf, num_tokens=num_tokens)
        step_ctx.memory_barrier()

        # 12. Final residual
        elementwise.encode_residual_add(step_ctx=step_ctx, x=mlp_out_buf, residual=residual_buf, output=hidden_b, num_elements=num_tokens * hidden_size)

        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read layer 0 output
    output_view = hidden_b.contents().as_buffer(num_tokens * hidden_size * 2)
    engine_layer0 = np.frombuffer(output_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    hf_layer0 = hf_layer_outputs['layer0'][0].cpu().numpy()

    print(f"Engine layer0 [0, :5]: {engine_layer0[0, :5]}")
    print(f"HF layer0 [0, :5]: {hf_layer0[0, :5]}")

    layer0_diff = np.abs(engine_layer0 - hf_layer0)
    print(f"Layer 0 max diff: {layer0_diff.max():.6f}")
    print(f"Layer 0 mean diff: {layer0_diff.mean():.6f}")
    print(f"Layer 0 PASS: {layer0_diff.max() < 1.0}")

    # ===== TEST 3: Compare what runner2._layers[0].qkv_proj.config says =====
    print("\n=== Test 3: QKV Config Verification ===")
    qkv_config = runner2._layers[0].qkv_proj.config
    print(f"QKV config:")
    print(f"  hidden_size: {qkv_config.hidden_size}")
    print(f"  num_heads: {qkv_config.num_heads}")
    print(f"  num_kv_heads: {qkv_config.num_kv_heads}")
    print(f"  head_size: {qkv_config.head_size}")
    print(f"  fused: {qkv_config.fused}")
    print(f"  q_size: {qkv_config.q_size}")
    print(f"  k_size: {qkv_config.k_size}")
    print(f"  v_size: {qkv_config.v_size}")

    print("\nExpected:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_size: {head_size}")
    print(f"  q_size: {num_heads * head_size}")
    print(f"  k_size: {num_kv_heads * head_size}")
    print(f"  v_size: {num_kv_heads * head_size}")

    config_ok = (
        qkv_config.hidden_size == hidden_size and
        qkv_config.num_heads == num_heads and
        qkv_config.num_kv_heads == num_kv_heads and
        qkv_config.head_size == head_size
    )
    print(f"\nConfig MATCH: {config_ok}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare layer 0 intermediate outputs between manual ops and EngineRunner ops."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Compare layer 0 outputs."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.qkv import EngineQKVProjection
    from vllm_apple.engine.ops.rmsnorm import EngineRMSNorm
    from vllm_apple.engine.runner import EngineRunner
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import ModelDescriptor, KVCacheDescriptor
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

    prompt = "def add(a, b):"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)

    # Extract config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = config.head_dim

    q_size = num_heads * head_size
    k_size = num_kv_heads * head_size
    qkv_size = q_size + 2 * k_size

    print(f"Config: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, head_size={head_size}")
    print(f"q_size={q_size}, k_size={k_size}, qkv_size={qkv_size}")
    print(f"Tokens: {num_tokens}")

    ctx = MetalEngineContext()

    # Get HF embedding as input
    embed_weight = state_dict['model.embed_tokens.weight']
    embed_np = embed_weight[token_ids].cpu().numpy().astype(np.float16)
    embed_np = np.ascontiguousarray(embed_np)
    print(f"\nInput embedding [0, :5]: {embed_np[0, :5]}")

    # === METHOD 1: Manual ops (like test_full_model_logits.py) ===
    print("\n=== Method 1: Manual Ops ===")

    def to_buffer(tensor):
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        tensor = tensor.cpu().contiguous()
        np_arr = tensor.numpy()
        np_arr = np.ascontiguousarray(np_arr)
        return ctx.device.newBufferWithBytes_length_options_(
            np_arr.tobytes(), np_arr.nbytes, MTLResourceStorageModeShared
        )

    # Create manual ops
    manual_norm = EngineRMSNorm(ctx, hidden_size=hidden_size, eps=1e-5)
    manual_norm.set_weights(to_buffer(state_dict['model.layers.0.input_layernorm.weight']))

    manual_qkv = EngineQKVProjection(
        ctx, hidden_size=hidden_size, num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_size=head_size, fused=False,
    )
    manual_qkv.set_weights(
        q_weight=to_buffer(state_dict['model.layers.0.self_attn.q_proj.weight']),
        k_weight=to_buffer(state_dict['model.layers.0.self_attn.k_proj.weight']),
        v_weight=to_buffer(state_dict['model.layers.0.self_attn.v_proj.weight']),
    )

    # Buffers for manual ops
    hidden_manual = ctx.device.newBufferWithBytes_length_options_(embed_np.tobytes(), embed_np.nbytes, MTLResourceStorageModeShared)
    normed_manual = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    qkv_manual = ctx.device.newBufferWithLength_options_(num_tokens * qkv_size * 2, MTLResourceStorageModeShared)

    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        # Input norm
        manual_norm.encode(step_ctx=step_ctx, input_buffer=hidden_manual, output_buffer=normed_manual, num_tokens=num_tokens)
        step_ctx.memory_barrier()

        # QKV projection
        manual_qkv.encode(step_ctx=step_ctx, hidden_states=normed_manual, qkv_output=qkv_manual, num_tokens=num_tokens)

        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read results
    normed_view = normed_manual.contents().as_buffer(num_tokens * hidden_size * 2)
    manual_normed = np.frombuffer(normed_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    qkv_view = qkv_manual.contents().as_buffer(num_tokens * qkv_size * 2)
    manual_qkv_out = np.frombuffer(qkv_view, dtype=np.float16).copy()

    print(f"Manual normed [0, :5]: {manual_normed[0, :5]}")
    print(f"Manual Q [0, :5]: {manual_qkv_out[:5]}")  # Q starts at offset 0
    print(f"Manual K [0, :5]: {manual_qkv_out[num_tokens * q_size : num_tokens * q_size + 5]}")  # K starts at q_size * num_tokens

    # === METHOD 2: EngineRunner ops ===
    print("\n=== Method 2: EngineRunner Ops ===")

    model_desc = ModelDescriptor(
        num_layers=40,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        rope_theta=config.rope_theta,
        max_position_embeddings=8192,
        architecture="mistral",
    )
    kv_desc = model_desc.get_kv_cache_descriptor(num_blocks=64, block_size=16)
    kv_cache = EngineKVCache(ctx, kv_desc)

    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')

    runner = EngineRunner(context=ctx, model_desc=model_desc, weights=weights, kv_cache=kv_cache)

    # Get layer 0 ops
    layer_ops = runner._layers[0]

    # Buffers for runner ops
    hidden_runner = ctx.device.newBufferWithBytes_length_options_(embed_np.tobytes(), embed_np.nbytes, MTLResourceStorageModeShared)
    normed_runner = ctx.device.newBufferWithLength_options_(num_tokens * hidden_size * 2, MTLResourceStorageModeShared)
    qkv_runner = ctx.device.newBufferWithLength_options_(num_tokens * qkv_size * 2, MTLResourceStorageModeShared)

    with EngineStepContext(ctx, step_id=2, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        # Input norm
        layer_ops.input_norm.encode(step_ctx=step_ctx, input_buffer=hidden_runner, output_buffer=normed_runner, num_tokens=num_tokens)
        step_ctx.memory_barrier()

        # QKV projection
        layer_ops.qkv_proj.encode(step_ctx=step_ctx, hidden_states=normed_runner, qkv_output=qkv_runner, num_tokens=num_tokens)

        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read results
    normed_view = normed_runner.contents().as_buffer(num_tokens * hidden_size * 2)
    runner_normed = np.frombuffer(normed_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    qkv_view = qkv_runner.contents().as_buffer(num_tokens * qkv_size * 2)
    runner_qkv_out = np.frombuffer(qkv_view, dtype=np.float16).copy()

    print(f"Runner normed [0, :5]: {runner_normed[0, :5]}")
    print(f"Runner Q [0, :5]: {runner_qkv_out[:5]}")
    print(f"Runner K [0, :5]: {runner_qkv_out[num_tokens * q_size : num_tokens * q_size + 5]}")

    # === Compare ===
    print("\n=== Comparison ===")
    norm_diff = np.abs(manual_normed - runner_normed).max()
    print(f"Normed max diff: {norm_diff:.6f}")
    print(f"Normed MATCH: {norm_diff < 0.001}")

    qkv_diff = np.abs(manual_qkv_out - runner_qkv_out).max()
    print(f"QKV max diff: {qkv_diff:.6f}")
    print(f"QKV MATCH: {qkv_diff < 0.001}")

    if qkv_diff > 0.001:
        # Find where differences occur
        print("\nAnalyzing QKV differences...")
        q_manual = manual_qkv_out[:num_tokens * q_size]
        q_runner = runner_qkv_out[:num_tokens * q_size]
        k_manual = manual_qkv_out[num_tokens * q_size:num_tokens * (q_size + k_size)]
        k_runner = runner_qkv_out[num_tokens * q_size:num_tokens * (q_size + k_size)]
        v_manual = manual_qkv_out[num_tokens * (q_size + k_size):]
        v_runner = runner_qkv_out[num_tokens * (q_size + k_size):]

        print(f"Q diff: max={np.abs(q_manual - q_runner).max():.6f}, mean={np.abs(q_manual - q_runner).mean():.6f}")
        print(f"K diff: max={np.abs(k_manual - k_runner).max():.6f}, mean={np.abs(k_manual - k_runner).mean():.6f}")
        print(f"V diff: max={np.abs(v_manual - v_runner).max():.6f}, mean={np.abs(v_manual - v_runner).mean():.6f}")


if __name__ == "__main__":
    main()

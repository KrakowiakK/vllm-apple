#!/usr/bin/env python3
"""Compare engine QKV projection output against HuggingFace."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.qkv import EngineQKVProjection
    from vllm_apple.engine.ops.rmsnorm import EngineRMSNorm
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

    # Get weights
    state_dict = model.state_dict()
    embed_weight = state_dict['model.embed_tokens.weight']
    input_ln_weight = state_dict['model.layers.0.input_layernorm.weight']
    q_weight = state_dict['model.layers.0.self_attn.q_proj.weight']
    k_weight = state_dict['model.layers.0.self_attn.k_proj.weight']
    v_weight = state_dict['model.layers.0.self_attn.v_proj.weight']

    # Tokenize
    prompt = "def add(a, b):"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0]
    num_tokens = len(token_ids)

    print(f"\n=== Input ===")
    print(f"Token IDs: {token_ids.tolist()}")
    print(f"Num tokens: {num_tokens}")

    # Get HuggingFace reference values
    print(f"\n=== HuggingFace Reference ===")
    captured = {}

    def make_hook(name):
        def hook(mod, inp, out):
            captured[name] = out.detach().clone() if not isinstance(out, tuple) else out[0].detach().clone()
        return hook

    layer0 = model.model.layers[0]
    layer0.input_layernorm.register_forward_hook(make_hook('input_ln'))
    layer0.self_attn.q_proj.register_forward_hook(make_hook('q_proj'))
    layer0.self_attn.k_proj.register_forward_hook(make_hook('k_proj'))
    layer0.self_attn.v_proj.register_forward_hook(make_hook('v_proj'))

    with torch.no_grad():
        model(**inputs)

    hf_normed = captured['input_ln'][0]  # [num_tokens, hidden]
    hf_q = captured['q_proj'][0]  # [num_tokens, q_size]
    hf_k = captured['k_proj'][0]  # [num_tokens, k_size]
    hf_v = captured['v_proj'][0]  # [num_tokens, v_size]

    print(f"HF normed shape: {hf_normed.shape}")
    print(f"HF Q shape: {hf_q.shape}")
    print(f"HF K shape: {hf_k.shape}")
    print(f"HF V shape: {hf_v.shape}")
    print(f"HF normed [0, :5]: {hf_normed[0, :5]}")
    print(f"HF Q [0, :5]: {hf_q[0, :5]}")

    # Create engine context
    print(f"\n=== Engine Computation ===")
    ctx = MetalEngineContext()

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = config.head_dim
    q_size = num_heads * head_size
    k_size = num_kv_heads * head_size
    v_size = num_kv_heads * head_size
    qkv_size = q_size + k_size + v_size

    # Create ops
    rmsnorm = EngineRMSNorm(ctx, hidden_size=hidden_size, eps=1e-5)
    qkv_proj = EngineQKVProjection(
        ctx,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        fused=False,  # Using separate weights
    )

    # Load weights to buffers
    def to_buffer(tensor, name):
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        tensor = tensor.cpu().contiguous()
        np_arr = tensor.numpy()
        np_arr = np.ascontiguousarray(np_arr)
        buf = ctx.device.newBufferWithBytes_length_options_(
            np_arr.tobytes(), np_arr.nbytes, MTLResourceStorageModeShared
        )
        print(f"  {name}: {tensor.shape} -> {np_arr.nbytes} bytes")
        return buf

    print("\nLoading weights to MTLBuffers:")
    ln_weight_buf = to_buffer(input_ln_weight, "input_ln")
    q_weight_buf = to_buffer(q_weight, "q_weight")
    k_weight_buf = to_buffer(k_weight, "k_weight")
    v_weight_buf = to_buffer(v_weight, "v_weight")

    rmsnorm.set_weights(ln_weight_buf)
    qkv_proj.set_weights(q_weight=q_weight_buf, k_weight=k_weight_buf, v_weight=v_weight_buf)

    # Create input buffer (embeddings)
    embed_np = embed_weight[token_ids].cpu().numpy().astype(np.float16)
    embed_np = np.ascontiguousarray(embed_np)
    input_buf = ctx.device.newBufferWithBytes_length_options_(
        embed_np.tobytes(), embed_np.nbytes, MTLResourceStorageModeShared
    )

    # Create output buffers
    normed_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * hidden_size * 2, MTLResourceStorageModeShared
    )
    qkv_buf = ctx.device.newBufferWithLength_options_(
        num_tokens * qkv_size * 2, MTLResourceStorageModeShared
    )

    print(f"\n=== Running Engine RMSNorm ===")
    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        rmsnorm.encode(
            step_ctx=step_ctx,
            input_buffer=input_buf,
            output_buffer=normed_buf,
            num_tokens=num_tokens,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read normed output
    normed_view = normed_buf.contents().as_buffer(num_tokens * hidden_size * 2)
    engine_normed = np.frombuffer(normed_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    print(f"Engine normed shape: {engine_normed.shape}")
    print(f"Engine normed [0, :5]: {engine_normed[0, :5]}")
    print(f"HF normed [0, :5]: {hf_normed[0, :5].numpy()}")
    normed_diff = np.abs(engine_normed[0, :5] - hf_normed[0, :5].numpy())
    print(f"Diff: {normed_diff}")
    print(f"Max diff: {np.abs(engine_normed - hf_normed.numpy()).max()}")

    print(f"\n=== Running Engine QKV Projection ===")
    with EngineStepContext(ctx, step_id=2, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        qkv_proj.encode(
            step_ctx=step_ctx,
            hidden_states=normed_buf,
            qkv_output=qkv_buf,
            num_tokens=num_tokens,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read QKV output
    qkv_view = qkv_buf.contents().as_buffer(num_tokens * qkv_size * 2)
    engine_qkv = np.frombuffer(qkv_view, dtype=np.float16).copy()

    # QKV layout is stacked: [Q_all_tokens][K_all_tokens][V_all_tokens]
    engine_q = engine_qkv[:num_tokens * q_size].reshape(num_tokens, q_size)
    engine_k = engine_qkv[num_tokens * q_size:num_tokens * (q_size + k_size)].reshape(num_tokens, k_size)
    engine_v = engine_qkv[num_tokens * (q_size + k_size):].reshape(num_tokens, v_size)

    print(f"Engine Q shape: {engine_q.shape}")
    print(f"Engine Q [0, :5]: {engine_q[0, :5]}")
    print(f"HF Q [0, :5]: {hf_q[0, :5].numpy()}")
    q_diff = np.abs(engine_q[0, :5] - hf_q[0, :5].numpy())
    print(f"Q Diff: {q_diff}")
    print(f"Q Max diff: {np.abs(engine_q - hf_q.numpy()).max()}")

    print(f"\nEngine K [0, :5]: {engine_k[0, :5]}")
    print(f"HF K [0, :5]: {hf_k[0, :5].numpy()}")
    k_diff = np.abs(engine_k[0, :5] - hf_k[0, :5].numpy())
    print(f"K Diff: {k_diff}")
    print(f"K Max diff: {np.abs(engine_k - hf_k.numpy()).max()}")

    print(f"\nEngine V [0, :5]: {engine_v[0, :5]}")
    print(f"HF V [0, :5]: {hf_v[0, :5].numpy()}")
    v_diff = np.abs(engine_v[0, :5] - hf_v[0, :5].numpy())
    print(f"V Diff: {v_diff}")
    print(f"V Max diff: {np.abs(engine_v - hf_v.numpy()).max()}")

    # Summary
    print(f"\n=== Summary ===")
    normed_max_diff = np.abs(engine_normed - hf_normed.numpy()).max()
    q_max_diff = np.abs(engine_q - hf_q.numpy()).max()
    k_max_diff = np.abs(engine_k - hf_k.numpy()).max()
    v_max_diff = np.abs(engine_v - hf_v.numpy()).max()

    print(f"RMSNorm max diff: {normed_max_diff:.6f} {'PASS' if normed_max_diff < 0.01 else 'FAIL'}")
    print(f"Q proj max diff: {q_max_diff:.6f} {'PASS' if q_max_diff < 0.01 else 'FAIL'}")
    print(f"K proj max diff: {k_max_diff:.6f} {'PASS' if k_max_diff < 0.01 else 'FAIL'}")
    print(f"V proj max diff: {v_max_diff:.6f} {'PASS' if v_max_diff < 0.01 else 'FAIL'}")


if __name__ == "__main__":
    main()

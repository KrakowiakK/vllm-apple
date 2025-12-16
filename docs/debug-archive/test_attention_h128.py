#!/usr/bin/env python3
"""Test paged attention with head_size=128 (Devstral config)."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch


def test_h128_prefill():
    """Test prefill attention with head_size=128."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.attention import PagedAttentionOp
    from vllm_apple.engine.ops.kv_write import KVWriteOp
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import KVCacheDescriptor
    from Metal import MTLResourceStorageModeShared

    print("=== Testing Attention with head_size=128 (Devstral config) ===")

    # Devstral config
    num_kv_heads = 8
    num_query_heads = 32  # 4 queries per kv head
    head_size = 128
    block_size = 16
    num_blocks = 4
    num_layers = 1

    ctx = MetalEngineContext()

    # Create KV cache
    kv_desc = KVCacheDescriptor(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_layers=num_layers,
    )
    kv_cache = EngineKVCache(ctx, kv_desc)

    # Create attention op
    attn_op = PagedAttentionOp(
        context=ctx,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
        head_size=head_size,
        block_size=block_size,
    )

    # Create KV write op
    kv_write_op = KVWriteOp(
        context=ctx,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
    )

    # Test parameters
    num_tokens = 3
    num_seqs = 1

    # QKV sizes
    q_size = num_tokens * num_query_heads * head_size
    k_size = num_tokens * num_kv_heads * head_size
    v_size = num_tokens * num_kv_heads * head_size
    qkv_size = q_size + k_size + v_size

    print(f"\nConfig:")
    print(f"  num_query_heads: {num_query_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_size: {head_size}")
    print(f"  num_tokens: {num_tokens}")
    print(f"  q_size: {q_size} elements")
    print(f"  k_size: {k_size} elements")
    print(f"  v_size: {v_size} elements")

    # Create input data
    np.random.seed(42)
    qkv_data = np.random.randn(qkv_size).astype(np.float16) * 0.1

    # Create buffers
    qkv_buf = ctx.device.newBufferWithBytes_length_options_(
        qkv_data.tobytes(), qkv_data.nbytes, MTLResourceStorageModeShared
    )

    output_size = num_tokens * num_query_heads * head_size
    output_buf = ctx.device.newBufferWithLength_options_(
        output_size * 2, MTLResourceStorageModeShared
    )

    # Single sequence with 3 tokens
    slot_mapping = np.array([0, 1, 2], dtype=np.int32)
    slot_buf = ctx.device.newBufferWithBytes_length_options_(
        slot_mapping.tobytes(), slot_mapping.nbytes, MTLResourceStorageModeShared
    )

    block_table = np.array([[0]], dtype=np.int32)
    block_table_buf = ctx.device.newBufferWithBytes_length_options_(
        block_table.tobytes(), block_table.nbytes, MTLResourceStorageModeShared
    )

    seq_lens = np.array([3], dtype=np.int32)
    seq_lens_buf = ctx.device.newBufferWithBytes_length_options_(
        seq_lens.tobytes(), seq_lens.nbytes, MTLResourceStorageModeShared
    )

    token_to_seq = np.array([0, 0, 0], dtype=np.int32)
    token_to_seq_buf = ctx.device.newBufferWithBytes_length_options_(
        token_to_seq.tobytes(), token_to_seq.nbytes, MTLResourceStorageModeShared
    )

    positions = np.array([0, 1, 2], dtype=np.int32)
    positions_buf = ctx.device.newBufferWithBytes_length_options_(
        positions.tobytes(), positions.nbytes, MTLResourceStorageModeShared
    )

    k_cache, v_cache = kv_cache.get_buffers(0)
    k_offset = q_size * 2
    v_offset = k_offset + k_size * 2

    # Step 1: Write KV
    print("\n=== Step 1: Write KV to cache ===")
    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=num_seqs) as step_ctx:
        kv_write_op.encode_prefill(
            step_ctx=step_ctx,
            new_keys_buffer=qkv_buf,
            new_values_buffer=qkv_buf,
            key_buffer=k_cache,
            value_buffer=v_cache,
            slot_mapping_buffer=slot_buf,
            num_tokens=num_tokens,
            new_keys_offset=k_offset,
            new_values_offset=v_offset,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    print("KV write completed")

    # Step 2: Run attention
    print("\n=== Step 2: Run Prefill Attention ===")
    with EngineStepContext(ctx, step_id=2, step_kind="prefill", num_tokens=num_tokens, num_seqs=num_seqs) as step_ctx:
        attn_op.encode_prefill(
            step_ctx=step_ctx,
            query_buffer=qkv_buf,
            key_buffer=k_cache,
            value_buffer=v_cache,
            block_table_buffer=block_table_buf,
            token_to_seq_buffer=token_to_seq_buf,
            positions_buffer=positions_buf,
            output_buffer=output_buf,
            num_tokens=num_tokens,
            num_seqs=num_seqs,
            max_seq_len=3,
            max_blocks_per_seq=1,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read output
    output_view = output_buf.contents().as_buffer(output_size * 2)
    engine_output = np.frombuffer(output_view, dtype=np.float16).reshape(
        num_tokens, num_query_heads, head_size
    ).copy()

    print(f"\nAttention output shape: {engine_output.shape}")
    print(f"Output [0, 0, :5]: {engine_output[0, 0, :5]}")
    print(f"Output [1, 0, :5]: {engine_output[1, 0, :5]}")
    print(f"Output [2, 0, :5]: {engine_output[2, 0, :5]}")

    # Check for NaN/Inf
    if np.any(np.isnan(engine_output)):
        print("ERROR: Engine output contains NaN!")
        return False
    if np.any(np.isinf(engine_output)):
        print("ERROR: Engine output contains Inf!")
        return False

    # Compute reference
    print("\n=== Computing Reference ===")
    q_data = qkv_data[:q_size].reshape(num_tokens, num_query_heads, head_size)
    k_data = qkv_data[q_size:q_size + k_size].reshape(num_tokens, num_kv_heads, head_size)
    v_data = qkv_data[q_size + k_size:].reshape(num_tokens, num_kv_heads, head_size)

    scale = 1.0 / np.sqrt(head_size)
    ref_output = np.zeros((num_tokens, num_query_heads, head_size), dtype=np.float16)
    queries_per_kv = num_query_heads // num_kv_heads

    for token_idx in range(num_tokens):
        position = positions[token_idx]
        seq_tokens = list(range(position + 1))  # Causal: can only attend up to current pos

        for qh in range(num_query_heads):
            kv_h = qh // queries_per_kv

            q_vec = q_data[token_idx, qh, :].astype(np.float32)
            k_vecs = np.array([k_data[t, kv_h, :] for t in seq_tokens]).astype(np.float32)
            v_vecs = np.array([v_data[t, kv_h, :] for t in seq_tokens]).astype(np.float32)

            scores = (q_vec @ k_vecs.T) * scale
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            weights = exp_scores / np.sum(exp_scores)
            out = weights @ v_vecs
            ref_output[token_idx, qh, :] = out.astype(np.float16)

    print(f"Reference output [0, 0, :5]: {ref_output[0, 0, :5]}")
    print(f"Reference output [1, 0, :5]: {ref_output[1, 0, :5]}")
    print(f"Reference output [2, 0, :5]: {ref_output[2, 0, :5]}")

    max_diff = np.abs(engine_output - ref_output).max()
    print(f"\nMax diff: {max_diff:.6f}")
    print(f"PREFILL Result: {'PASS' if max_diff < 0.1 else 'FAIL'}")

    return max_diff < 0.1


def test_h128_decode():
    """Test decode attention with head_size=128."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.attention import PagedAttentionOp
    from vllm_apple.engine.ops.kv_write import KVWriteOp
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import KVCacheDescriptor
    from Metal import MTLResourceStorageModeShared

    print("\n\n=== Testing DECODE Attention with head_size=128 ===")

    # Devstral config
    num_kv_heads = 8
    num_query_heads = 32
    head_size = 128
    block_size = 16
    num_blocks = 4
    num_layers = 1

    ctx = MetalEngineContext()

    kv_desc = KVCacheDescriptor(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_layers=num_layers,
    )
    kv_cache = EngineKVCache(ctx, kv_desc)

    attn_op = PagedAttentionOp(
        context=ctx,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
        head_size=head_size,
        block_size=block_size,
    )

    kv_write_op = KVWriteOp(
        context=ctx,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
    )

    # First, populate KV cache with 3 prefill tokens
    prefill_tokens = 3
    num_seqs = 1

    q_size_prefill = prefill_tokens * num_query_heads * head_size
    k_size_prefill = prefill_tokens * num_kv_heads * head_size
    v_size_prefill = prefill_tokens * num_kv_heads * head_size

    np.random.seed(42)
    qkv_prefill = np.random.randn(q_size_prefill + k_size_prefill + v_size_prefill).astype(np.float16) * 0.1
    qkv_prefill_buf = ctx.device.newBufferWithBytes_length_options_(
        qkv_prefill.tobytes(), qkv_prefill.nbytes, MTLResourceStorageModeShared
    )

    slot_mapping_prefill = np.array([0, 1, 2], dtype=np.int32)
    slot_prefill_buf = ctx.device.newBufferWithBytes_length_options_(
        slot_mapping_prefill.tobytes(), slot_mapping_prefill.nbytes, MTLResourceStorageModeShared
    )

    k_cache, v_cache = kv_cache.get_buffers(0)
    k_offset_prefill = q_size_prefill * 2
    v_offset_prefill = k_offset_prefill + k_size_prefill * 2

    # Prefill KV write
    print("Prefill: Writing initial KV cache...")
    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=prefill_tokens, num_seqs=num_seqs) as step_ctx:
        kv_write_op.encode_prefill(
            step_ctx=step_ctx,
            new_keys_buffer=qkv_prefill_buf,
            new_values_buffer=qkv_prefill_buf,
            key_buffer=k_cache,
            value_buffer=v_cache,
            slot_mapping_buffer=slot_prefill_buf,
            num_tokens=prefill_tokens,
            new_keys_offset=k_offset_prefill,
            new_values_offset=v_offset_prefill,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Now test decode step (1 new token)
    print("\n=== Decode Step ===")
    decode_tokens = 1
    q_size_decode = decode_tokens * num_query_heads * head_size
    k_size_decode = decode_tokens * num_kv_heads * head_size
    v_size_decode = decode_tokens * num_kv_heads * head_size

    np.random.seed(123)
    qkv_decode = np.random.randn(q_size_decode + k_size_decode + v_size_decode).astype(np.float16) * 0.1
    qkv_decode_buf = ctx.device.newBufferWithBytes_length_options_(
        qkv_decode.tobytes(), qkv_decode.nbytes, MTLResourceStorageModeShared
    )

    # Output buffer for decode
    output_size = decode_tokens * num_query_heads * head_size
    output_buf = ctx.device.newBufferWithLength_options_(
        output_size * 2, MTLResourceStorageModeShared
    )

    # Block table, seq_lens for decode
    block_table = np.array([[0]], dtype=np.int32)
    block_table_buf = ctx.device.newBufferWithBytes_length_options_(
        block_table.tobytes(), block_table.nbytes, MTLResourceStorageModeShared
    )

    seq_lens = np.array([4], dtype=np.int32)  # 3 prefill + 1 decode = 4 total
    seq_lens_buf = ctx.device.newBufferWithBytes_length_options_(
        seq_lens.tobytes(), seq_lens.nbytes, MTLResourceStorageModeShared
    )

    k_offset_decode = q_size_decode * 2
    v_offset_decode = k_offset_decode + k_size_decode * 2

    print(f"Decode config:")
    print(f"  seq_lens: {seq_lens}")
    print(f"  q_size: {q_size_decode}")
    print(f"  k_offset: {k_offset_decode}")
    print(f"  v_offset: {v_offset_decode}")

    # Run decode attention (fused)
    with EngineStepContext(ctx, step_id=2, step_kind="decode", num_tokens=decode_tokens, num_seqs=num_seqs) as step_ctx:
        attn_op.encode_decode_fused(
            step_ctx=step_ctx,
            query_buffer=qkv_decode_buf,
            new_keys_buffer=qkv_decode_buf,
            new_values_buffer=qkv_decode_buf,
            key_buffer=k_cache,
            value_buffer=v_cache,
            block_table_buffer=block_table_buf,
            seq_lens_buffer=seq_lens_buf,
            output_buffer=output_buf,
            num_seqs=num_seqs,
            max_seq_len=4,
            max_blocks_per_seq=1,
            query_offset=0,
            new_keys_offset=k_offset_decode,
            new_values_offset=v_offset_decode,
            output_offset=0,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read output
    output_view = output_buf.contents().as_buffer(output_size * 2)
    engine_output = np.frombuffer(output_view, dtype=np.float16).reshape(
        decode_tokens, num_query_heads, head_size
    ).copy()

    print(f"\nDecode output shape: {engine_output.shape}")
    print(f"Output [0, 0, :5]: {engine_output[0, 0, :5]}")

    # Check for NaN/Inf
    if np.any(np.isnan(engine_output)):
        print("ERROR: Engine output contains NaN!")
        return False
    if np.any(np.isinf(engine_output)):
        print("ERROR: Engine output contains Inf!")
        return False

    # Compute reference
    # The decode token (position 3) can attend to positions 0, 1, 2, 3
    # Positions 0-2 have K/V from prefill, position 3 has new K/V from decode
    k_prefill = qkv_prefill[q_size_prefill:q_size_prefill + k_size_prefill].reshape(prefill_tokens, num_kv_heads, head_size)
    v_prefill = qkv_prefill[q_size_prefill + k_size_prefill:].reshape(prefill_tokens, num_kv_heads, head_size)
    k_decode = qkv_decode[q_size_decode:q_size_decode + k_size_decode].reshape(decode_tokens, num_kv_heads, head_size)
    v_decode = qkv_decode[q_size_decode + k_size_decode:].reshape(decode_tokens, num_kv_heads, head_size)
    q_decode = qkv_decode[:q_size_decode].reshape(decode_tokens, num_query_heads, head_size)

    # Full K/V history: [prefill K/V, decode K/V]
    k_all = np.concatenate([k_prefill, k_decode], axis=0)  # [4, num_kv_heads, head_size]
    v_all = np.concatenate([v_prefill, v_decode], axis=0)

    scale = 1.0 / np.sqrt(head_size)
    ref_output = np.zeros((decode_tokens, num_query_heads, head_size), dtype=np.float16)
    queries_per_kv = num_query_heads // num_kv_heads

    for qh in range(num_query_heads):
        kv_h = qh // queries_per_kv
        q_vec = q_decode[0, qh, :].astype(np.float32)
        k_vecs = k_all[:, kv_h, :].astype(np.float32)  # All 4 positions
        v_vecs = v_all[:, kv_h, :].astype(np.float32)

        scores = (q_vec @ k_vecs.T) * scale
        scores_max = np.max(scores)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores)
        out = weights @ v_vecs
        ref_output[0, qh, :] = out.astype(np.float16)

    print(f"Reference output [0, 0, :5]: {ref_output[0, 0, :5]}")

    max_diff = np.abs(engine_output - ref_output).max()
    print(f"\nMax diff: {max_diff:.6f}")
    print(f"DECODE Result: {'PASS' if max_diff < 0.1 else 'FAIL'}")

    return max_diff < 0.1


if __name__ == "__main__":
    prefill_ok = test_h128_prefill()
    decode_ok = test_h128_decode()

    print("\n" + "=" * 50)
    print(f"PREFILL h128: {'PASS' if prefill_ok else 'FAIL'}")
    print(f"DECODE h128: {'PASS' if decode_ok else 'FAIL'}")

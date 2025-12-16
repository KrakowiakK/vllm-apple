#!/usr/bin/env python3
"""Test basic paged attention functionality."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np
import torch


def test_simple_attention():
    """Test attention with a simple known input."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.attention import PagedAttentionOp
    from vllm_apple.engine.ops.kv_write import KVWriteOp
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import KVCacheDescriptor
    from Metal import MTLResourceStorageModeShared

    print("=== Testing Basic Attention ===")

    # Small config for testing
    num_kv_heads = 2
    num_query_heads = 4  # 2 queries per kv head
    head_size = 64
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
    num_tokens = 4
    num_seqs = 2

    # Create input data
    # Q shape: [num_tokens, num_query_heads, head_size]
    q_size = num_tokens * num_query_heads * head_size
    k_size = num_tokens * num_kv_heads * head_size
    v_size = num_tokens * num_kv_heads * head_size
    qkv_size = q_size + k_size + v_size

    # Use known values for easier debugging
    np.random.seed(42)
    qkv_data = np.random.randn(qkv_size).astype(np.float16)

    # Create buffers
    qkv_buf = ctx.device.newBufferWithBytes_length_options_(
        qkv_data.tobytes(), qkv_data.nbytes, MTLResourceStorageModeShared
    )

    output_size = num_tokens * num_query_heads * head_size
    output_buf = ctx.device.newBufferWithLength_options_(
        output_size * 2, MTLResourceStorageModeShared
    )

    # Create slot mapping (tokens 0,1 -> seq 0, tokens 2,3 -> seq 1)
    # Slot = seq_idx * block_size + position_in_seq
    slot_mapping = np.array([0, 1, 16, 17], dtype=np.int32)
    slot_buf = ctx.device.newBufferWithBytes_length_options_(
        slot_mapping.tobytes(), slot_mapping.nbytes, MTLResourceStorageModeShared
    )

    # Create block table (2 seqs, 1 block each)
    # seq 0 -> block 0, seq 1 -> block 1
    block_table = np.array([[0], [1]], dtype=np.int32)
    block_table_buf = ctx.device.newBufferWithBytes_length_options_(
        block_table.tobytes(), block_table.nbytes, MTLResourceStorageModeShared
    )

    # Sequence lengths (seq 0 has 2 tokens, seq 1 has 2 tokens)
    seq_lens = np.array([2, 2], dtype=np.int32)
    seq_lens_buf = ctx.device.newBufferWithBytes_length_options_(
        seq_lens.tobytes(), seq_lens.nbytes, MTLResourceStorageModeShared
    )

    # Token to sequence mapping (for prefill)
    token_to_seq = np.array([0, 0, 1, 1], dtype=np.int32)
    token_to_seq_buf = ctx.device.newBufferWithBytes_length_options_(
        token_to_seq.tobytes(), token_to_seq.nbytes, MTLResourceStorageModeShared
    )

    # Positions
    positions = np.array([0, 1, 0, 1], dtype=np.int32)
    positions_buf = ctx.device.newBufferWithBytes_length_options_(
        positions.tobytes(), positions.nbytes, MTLResourceStorageModeShared
    )

    # Get KV cache buffers
    k_cache, v_cache = kv_cache.get_buffers(0)

    k_offset = q_size * 2  # bytes
    v_offset = k_offset + k_size * 2

    print(f"\nTest config:")
    print(f"  num_tokens: {num_tokens}")
    print(f"  num_seqs: {num_seqs}")
    print(f"  num_query_heads: {num_query_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_size: {head_size}")
    print(f"  block_size: {block_size}")
    print(f"  q_size: {q_size}")
    print(f"  k_size: {k_size}")
    print(f"  v_size: {v_size}")
    print(f"  k_offset: {k_offset}")
    print(f"  v_offset: {v_offset}")

    # Step 1: Write KV cache
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

    # Read back KV cache to verify
    # KV cache layout: [num_blocks, num_kv_heads, block_size, head_size]
    one_kv_size = num_kv_heads * block_size * head_size  # elements per block
    cache_total_bytes = one_kv_size * num_blocks * 2  # bytes (float16)
    k_cache_view = k_cache.contents().as_buffer(cache_total_bytes)
    k_cache_data = np.frombuffer(k_cache_view, dtype=np.float16).copy()
    print(f"K cache total elements: {k_cache_data.shape}")
    print(f"K cache first 10: {k_cache_data[:10]}")

    # Step 2: Run attention
    print("\n=== Step 2: Run Prefill Attention ===")
    with EngineStepContext(ctx, step_id=2, step_kind="prefill", num_tokens=num_tokens, num_seqs=num_seqs) as step_ctx:
        attn_op.encode_prefill(
            step_ctx=step_ctx,
            query_buffer=qkv_buf,  # Q is at offset 0
            key_buffer=k_cache,
            value_buffer=v_cache,
            block_table_buffer=block_table_buf,
            token_to_seq_buffer=token_to_seq_buf,
            positions_buffer=positions_buf,
            output_buffer=output_buf,
            num_tokens=num_tokens,
            num_seqs=num_seqs,
            max_seq_len=2,
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
    print(f"Output [0, 1, :5]: {engine_output[0, 1, :5]}")

    # Compute reference attention manually
    print("\n=== Computing Reference Attention ===")

    # Extract Q, K, V from qkv_data
    q_data = qkv_data[:q_size].reshape(num_tokens, num_query_heads, head_size)
    k_data = qkv_data[q_size:q_size + k_size].reshape(num_tokens, num_kv_heads, head_size)
    v_data = qkv_data[q_size + k_size:].reshape(num_tokens, num_kv_heads, head_size)

    print(f"Q shape: {q_data.shape}")
    print(f"K shape: {k_data.shape}")
    print(f"V shape: {v_data.shape}")

    # Compute attention for token 0 (position 0, seq 0)
    # Token 0 can only attend to itself (causal)
    # For GQA, query heads 0,1 use kv head 0; query heads 2,3 use kv head 1

    scale = 1.0 / np.sqrt(head_size)
    ref_output = np.zeros_like(q_data)

    for token_idx in range(num_tokens):
        seq_idx = token_to_seq[token_idx]
        position = positions[token_idx]

        # Get tokens in this sequence up to current position (causal)
        seq_start = 0 if seq_idx == 0 else 2  # seq 0 has tokens 0,1; seq 1 has tokens 2,3
        seq_tokens = list(range(seq_start, seq_start + position + 1))

        print(f"\nToken {token_idx} (seq={seq_idx}, pos={position}):")
        print(f"  Can attend to tokens: {seq_tokens}")

        for qh in range(num_query_heads):
            kv_h = qh // (num_query_heads // num_kv_heads)  # Which KV head to use

            q_vec = q_data[token_idx, qh, :].astype(np.float32)

            # Gather K and V from attended tokens
            k_vecs = np.array([k_data[t, kv_h, :] for t in seq_tokens]).astype(np.float32)
            v_vecs = np.array([v_data[t, kv_h, :] for t in seq_tokens]).astype(np.float32)

            # Compute attention scores
            scores = (q_vec @ k_vecs.T) * scale

            # Softmax
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            weights = exp_scores / np.sum(exp_scores)

            # Weighted sum of values
            out = weights @ v_vecs
            ref_output[token_idx, qh, :] = out.astype(np.float16)

            if token_idx == 0 and qh == 0:
                print(f"  Query head 0, KV head {kv_h}:")
                print(f"    Q: {q_vec[:5]}")
                print(f"    K: {k_vecs[:, :5]}")
                print(f"    scores: {scores}")
                print(f"    weights: {weights}")
                print(f"    output: {out[:5]}")

    print(f"\n=== Comparison ===")
    print(f"Reference output [0, 0, :5]: {ref_output[0, 0, :5]}")
    print(f"Engine output [0, 0, :5]: {engine_output[0, 0, :5]}")

    max_diff = np.abs(engine_output - ref_output).max()
    print(f"\nMax diff: {max_diff:.6f}")
    print(f"Result: {'PASS' if max_diff < 0.1 else 'FAIL'}")

    # Check for NaN/Inf
    if np.any(np.isnan(engine_output)):
        print("WARNING: Engine output contains NaN!")
    if np.any(np.isinf(engine_output)):
        print("WARNING: Engine output contains Inf!")

    return max_diff < 0.1


if __name__ == "__main__":
    test_simple_attention()

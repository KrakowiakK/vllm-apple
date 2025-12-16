#!/usr/bin/env python3
"""Debug each layer component to find where garbage output originates."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '1'

import numpy as np
import torch

def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.gemm import EngineGEMM
    from vllm_apple.engine.ops.rmsnorm import EngineRMSNorm
    from vllm_apple.engine.ops.embedding import EngineEmbedding
    from vllm_apple.engine.ops.qkv import EngineQKVProjection
    from vllm_apple.engine.ops.elementwise import EngineRoPE
    from vllm_apple.engine.ops.kv_write import KVWriteOp
    from vllm_apple.engine.ops.attention import PagedAttentionOp
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import KVCacheDescriptor
    from Metal import MTLResourceStorageModeShared

    ctx = MetalEngineContext()

    # Small test configuration (mimicking Devstral but smaller)
    hidden_size = 256  # Small for debugging
    num_heads = 4
    num_kv_heads = 2
    head_size = 64
    num_tokens = 4
    num_seqs = 1
    block_size = 16
    vocab_size = 1000
    max_position_embeddings = 512
    rope_theta = 10000.0

    print(f"Test config: hidden_size={hidden_size}, heads={num_heads}/{num_kv_heads}, head_size={head_size}")
    print(f"tokens={num_tokens}, seqs={num_seqs}, block_size={block_size}")

    # Create KV cache (force shared storage for debugging)
    kv_desc = KVCacheDescriptor(
        num_blocks=64,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_layers=1,
    )
    kv_cache = EngineKVCache(ctx, kv_desc, use_private_storage=False)
    print(f"KV cache using private storage: {kv_cache.uses_private_storage}")

    # Create ops
    embedding = EngineEmbedding(ctx, vocab_size=vocab_size, hidden_size=hidden_size)
    input_norm = EngineRMSNorm(ctx, hidden_size=hidden_size, eps=1e-5)
    qkv_proj = EngineQKVProjection(
        ctx, hidden_size=hidden_size, num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_size=head_size
    )
    rope = EngineRoPE(
        ctx, head_size=head_size, num_heads=num_heads, num_kv_heads=num_kv_heads,
        max_position=max_position_embeddings, base=rope_theta
    )
    kv_write = KVWriteOp(ctx, num_kv_heads=num_kv_heads, head_size=head_size, block_size=block_size)
    attention = PagedAttentionOp(
        ctx, num_kv_heads=num_kv_heads, num_query_heads=num_heads,
        head_size=head_size, block_size=block_size
    )

    # Initialize weights with known values
    np.random.seed(42)

    # Embedding weights
    embed_np = np.random.randn(vocab_size, hidden_size).astype(np.float16)
    embed_bytes = embed_np.tobytes()
    embed_buffer = ctx.device.newBufferWithBytes_length_options_(
        embed_bytes, len(embed_bytes), MTLResourceStorageModeShared
    )
    embedding.set_weights(embed_buffer)

    # RMSNorm weights (all ones for simplicity)
    norm_np = np.ones(hidden_size, dtype=np.float16)
    norm_bytes = norm_np.tobytes()
    norm_buffer = ctx.device.newBufferWithBytes_length_options_(
        norm_bytes, len(norm_bytes), MTLResourceStorageModeShared
    )
    input_norm.set_weights(norm_buffer)

    # QKV weights: [qkv_size, hidden_size]
    qkv_size = (num_heads + 2 * num_kv_heads) * head_size  # Q + K + V
    qkv_weight_np = np.random.randn(qkv_size, hidden_size).astype(np.float16) * 0.02
    qkv_weight_bytes = qkv_weight_np.tobytes()
    qkv_weight_buffer = ctx.device.newBufferWithBytes_length_options_(
        qkv_weight_bytes, len(qkv_weight_bytes), MTLResourceStorageModeShared
    )
    qkv_proj.set_weights(qkv_weight=qkv_weight_buffer)

    # Create input data
    token_ids = np.array([100, 200, 300, 400], dtype=np.int32)  # 4 tokens
    positions = np.array([0, 1, 2, 3], dtype=np.int32)  # Positions 0-3
    seq_lens = np.array([4], dtype=np.int32)  # One sequence of length 4

    # slot_mapping: maps each token to its slot in KV cache
    # For prefill of seq 0 with positions 0-3, slots should be 0-3
    slot_mapping = np.array([0, 1, 2, 3], dtype=np.int32)

    # block_table: maps sequence -> physical blocks
    # Seq 0 needs 1 block (4 tokens < block_size=16)
    # Physical block 0 is used
    block_table = np.array([[0, -1, -1, -1]], dtype=np.int32)  # [num_seqs, max_blocks]

    # token_to_seq: which sequence each token belongs to
    token_to_seq = np.array([0, 0, 0, 0], dtype=np.int32)

    # Create MTL buffers
    token_ids_buffer = ctx.device.newBufferWithBytes_length_options_(
        token_ids.tobytes(), token_ids.nbytes, MTLResourceStorageModeShared
    )
    positions_buffer = ctx.device.newBufferWithBytes_length_options_(
        positions.tobytes(), positions.nbytes, MTLResourceStorageModeShared
    )
    seq_lens_buffer = ctx.device.newBufferWithBytes_length_options_(
        seq_lens.tobytes(), seq_lens.nbytes, MTLResourceStorageModeShared
    )
    slot_mapping_buffer = ctx.device.newBufferWithBytes_length_options_(
        slot_mapping.tobytes(), slot_mapping.nbytes, MTLResourceStorageModeShared
    )
    block_table_buffer = ctx.device.newBufferWithBytes_length_options_(
        block_table.tobytes(), block_table.nbytes, MTLResourceStorageModeShared
    )
    token_to_seq_buffer = ctx.device.newBufferWithBytes_length_options_(
        token_to_seq.tobytes(), token_to_seq.nbytes, MTLResourceStorageModeShared
    )

    # Allocate output buffers
    hidden_size_bytes = num_tokens * hidden_size * 2
    hidden_buffer = ctx.device.newBufferWithLength_options_(
        hidden_size_bytes, MTLResourceStorageModeShared
    )
    normed_buffer = ctx.device.newBufferWithLength_options_(
        hidden_size_bytes, MTLResourceStorageModeShared
    )
    qkv_buffer_size = num_tokens * qkv_size * 2
    qkv_buffer = ctx.device.newBufferWithLength_options_(
        qkv_buffer_size, MTLResourceStorageModeShared
    )
    attn_out_size = num_tokens * num_heads * head_size * 2
    attn_out_buffer = ctx.device.newBufferWithLength_options_(
        attn_out_size, MTLResourceStorageModeShared
    )

    # ===== Step 1: Embedding =====
    print("\n=== Step 1: Embedding ===")
    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=num_seqs) as step_ctx:
        embedding.encode(step_ctx, token_ids_buffer, hidden_buffer, num_tokens)
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read back embedding output
    hidden_view = hidden_buffer.contents().as_buffer(hidden_size_bytes)
    hidden_np = np.frombuffer(hidden_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    # Verify embedding
    expected_embed = embed_np[token_ids]
    embed_diff = np.abs(hidden_np.astype(np.float32) - expected_embed.astype(np.float32))
    print(f"Embedding max diff: {embed_diff.max():.6f}")
    if embed_diff.max() > 0.01:
        print("  ✗ Embedding FAILED")
        return False
    print("  ✓ Embedding OK")

    # ===== Step 2: RMSNorm =====
    print("\n=== Step 2: RMSNorm ===")
    with EngineStepContext(ctx, step_id=2, step_kind="prefill", num_tokens=num_tokens, num_seqs=num_seqs) as step_ctx:
        input_norm.encode(step_ctx, hidden_buffer, normed_buffer, num_tokens)
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    normed_view = normed_buffer.contents().as_buffer(hidden_size_bytes)
    normed_np = np.frombuffer(normed_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    # Verify RMSNorm
    def ref_rmsnorm(x, w, eps=1e-5):
        x_f32 = x.astype(np.float32)
        w_f32 = w.astype(np.float32)
        mean_sq = np.mean(x_f32 ** 2, axis=-1, keepdims=True)
        rsqrt = 1.0 / np.sqrt(mean_sq + eps)
        return (x_f32 * rsqrt * w_f32).astype(np.float16)

    expected_normed = ref_rmsnorm(hidden_np, norm_np)
    norm_diff = np.abs(normed_np.astype(np.float32) - expected_normed.astype(np.float32))
    print(f"RMSNorm max diff: {norm_diff.max():.6f}")
    if norm_diff.max() > 0.01:
        print("  ✗ RMSNorm FAILED")
        return False
    print("  ✓ RMSNorm OK")

    # ===== Step 3: QKV Projection =====
    print("\n=== Step 3: QKV Projection ===")
    # Need to copy normed to new buffer for input
    normed_input_buffer = ctx.device.newBufferWithBytes_length_options_(
        normed_np.tobytes(), normed_np.nbytes, MTLResourceStorageModeShared
    )

    with EngineStepContext(ctx, step_id=3, step_kind="prefill", num_tokens=num_tokens, num_seqs=num_seqs) as step_ctx:
        qkv_proj.encode(step_ctx, normed_input_buffer, qkv_buffer, num_tokens)
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    qkv_view = qkv_buffer.contents().as_buffer(qkv_buffer_size)
    qkv_np = np.frombuffer(qkv_view, dtype=np.float16).copy()

    # QKV output layout for multi-token: [Q_all][K_all][V_all]
    q_size = num_tokens * num_heads * head_size
    k_size = num_tokens * num_kv_heads * head_size
    v_size = num_tokens * num_kv_heads * head_size

    q_np = qkv_np[:q_size].reshape(num_tokens, num_heads * head_size)
    k_np = qkv_np[q_size:q_size+k_size].reshape(num_tokens, num_kv_heads * head_size)
    v_np = qkv_np[q_size+k_size:].reshape(num_tokens, num_kv_heads * head_size)

    # Verify QKV projection against numpy reference
    # Split weights
    q_weight = qkv_weight_np[:num_heads * head_size, :]
    k_weight = qkv_weight_np[num_heads * head_size:num_heads * head_size + num_kv_heads * head_size, :]
    v_weight = qkv_weight_np[num_heads * head_size + num_kv_heads * head_size:, :]

    expected_q = normed_np.astype(np.float32) @ q_weight.T.astype(np.float32)
    expected_k = normed_np.astype(np.float32) @ k_weight.T.astype(np.float32)
    expected_v = normed_np.astype(np.float32) @ v_weight.T.astype(np.float32)

    q_diff = np.abs(q_np.astype(np.float32) - expected_q)
    k_diff = np.abs(k_np.astype(np.float32) - expected_k)
    v_diff = np.abs(v_np.astype(np.float32) - expected_v)

    print(f"Q max diff: {q_diff.max():.6f}, mean: {q_diff.mean():.6f}")
    print(f"K max diff: {k_diff.max():.6f}, mean: {k_diff.mean():.6f}")
    print(f"V max diff: {v_diff.max():.6f}, mean: {v_diff.mean():.6f}")

    if q_diff.max() > 0.1 or k_diff.max() > 0.1 or v_diff.max() > 0.1:
        print("  ✗ QKV Projection FAILED")
        # Debug: print some values
        print(f"  Q sample (actual): {q_np[0, :5]}")
        print(f"  Q sample (expected): {expected_q[0, :5].astype(np.float16)}")
        return False
    print("  ✓ QKV Projection OK")

    # ===== Step 4: RoPE =====
    print("\n=== Step 4: RoPE ===")
    # Need fresh QKV buffer with expected values for RoPE test
    # RoPE modifies Q and K in-place

    # Actually, let's skip RoPE verification for now since it's complex
    # and focus on attention

    # ===== Step 5: KV Write (Prefill) =====
    print("\n=== Step 5: KV Write (Prefill) ===")

    # Get KV cache buffers
    k_cache, v_cache = kv_cache.get_buffers(0)

    # Prepare K and V from QKV buffer (after the Q section)
    k_offset_bytes = q_size * 2
    v_offset_bytes = (q_size + k_size) * 2

    with EngineStepContext(ctx, step_id=5, step_kind="prefill", num_tokens=num_tokens, num_seqs=num_seqs) as step_ctx:
        kv_write.encode_prefill(
            step_ctx=step_ctx,
            new_keys_buffer=qkv_buffer,
            new_values_buffer=qkv_buffer,
            key_buffer=k_cache,
            value_buffer=v_cache,
            slot_mapping_buffer=slot_mapping_buffer,
            num_tokens=num_tokens,
            new_keys_offset=k_offset_bytes,
            new_values_offset=v_offset_bytes,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Verify KV cache was written correctly
    # Read back KV cache
    if k_cache is None:
        print("  ✗ k_cache is None!")
        return False

    k_cache_size = 64 * num_kv_heads * block_size * head_size * 2  # 64 blocks
    print(f"  k_cache buffer length: {k_cache.length()}, expected: {k_cache_size}")
    k_cache_view = k_cache.contents().as_buffer(k_cache_size)
    k_cache_np = np.frombuffer(k_cache_view, dtype=np.float16).copy()

    # Check the first block (where our 4 tokens should be)
    # KV cache layout: [num_blocks, num_kv_heads, block_size, head_size]
    # So block 0, head 0, token 0 starts at offset 0
    # Block 0, head 0, token 1 starts at offset head_size

    print(f"KV cache K sample (block 0, head 0, tokens 0-3):")
    for t in range(4):
        offset = 0 * num_kv_heads * block_size * head_size + 0 * block_size * head_size + t * head_size
        cached_k = k_cache_np[offset:offset+5]  # First 5 elements
        expected_k_t = k_np[t, :5]  # Token t, head 0, first 5 elements
        print(f"  Token {t}: cached={cached_k}, expected K (pre-RoPE)={expected_k_t}")

    print("  ✓ KV Write OK (values written, correctness depends on RoPE)")

    # ===== Step 6: Prefill Attention =====
    print("\n=== Step 6: Prefill Attention ===")

    with EngineStepContext(ctx, step_id=6, step_kind="prefill", num_tokens=num_tokens, num_seqs=num_seqs) as step_ctx:
        attention.encode_prefill(
            step_ctx=step_ctx,
            query_buffer=qkv_buffer,  # Q is at start
            key_buffer=k_cache,
            value_buffer=v_cache,
            block_table_buffer=block_table_buffer,
            token_to_seq_buffer=token_to_seq_buffer,
            positions_buffer=positions_buffer,
            output_buffer=attn_out_buffer,
            num_tokens=num_tokens,
            num_seqs=num_seqs,
            max_seq_len=4,
            max_blocks_per_seq=4,
            query_offset=0,
            output_offset=0,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read back attention output
    attn_view = attn_out_buffer.contents().as_buffer(attn_out_size)
    attn_np = np.frombuffer(attn_view, dtype=np.float16).reshape(num_tokens, num_heads * head_size).copy()

    print(f"Attention output shape: {attn_np.shape}")
    print(f"Attention output sample (token 0): {attn_np[0, :10]}")
    print(f"Attention output sample (token 3): {attn_np[3, :10]}")

    # Check if attention output is all zeros or NaN
    if np.isnan(attn_np).any():
        print("  ✗ Attention output contains NaN!")
        return False
    if np.allclose(attn_np, 0):
        print("  ✗ Attention output is all zeros!")
        return False

    print("  ✓ Attention produces non-trivial output")

    # ===== Summary =====
    print("\n=== SUMMARY ===")
    print("All components produce reasonable output.")
    print("If full model still fails, issue may be in:")
    print("  1. Weight loading (wrong weights for actual model)")
    print("  2. Layer stacking (wrong buffer usage between layers)")
    print("  3. RoPE (complex math that wasn't validated)")
    print("  4. Scale factors or other hyperparameters")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

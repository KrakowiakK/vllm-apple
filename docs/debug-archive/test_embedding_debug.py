#!/usr/bin/env python3
"""Debug embedding lookup in engine."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
os.environ['VLLM_APPLE_USE_ENGINE'] = '1'
os.environ['VLLM_GEMM_BACKEND'] = 'mps'

import torch
import numpy as np

def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.ops.embedding import EngineEmbedding
    from Metal import MTLResourceStorageModeShared

    # Create context
    ctx = MetalEngineContext()

    # Create test embedding weights - small vocab for testing
    vocab_size = 100
    hidden_size = 16

    # Create simple embedding: each row is [token_id, token_id, token_id, ...]
    # So token_id=5 gives [5, 5, 5, ...]
    embedding_weights = np.zeros((vocab_size, hidden_size), dtype=np.float16)
    for i in range(vocab_size):
        embedding_weights[i, :] = float(i)

    # Convert to MTLBuffer
    weights_bytes = embedding_weights.tobytes()
    weight_buffer = ctx.device.newBufferWithBytes_length_options_(
        weights_bytes, len(weights_bytes), MTLResourceStorageModeShared
    )

    # Create embedding op
    embedding = EngineEmbedding(ctx, vocab_size=vocab_size, hidden_size=hidden_size)
    embedding.set_weights(weight_buffer)

    # Test token IDs
    token_ids = np.array([0, 5, 10, 50], dtype=np.int32)
    num_tokens = len(token_ids)

    # Create input buffer
    token_ids_bytes = token_ids.tobytes()
    token_ids_buffer = ctx.device.newBufferWithBytes_length_options_(
        token_ids_bytes, len(token_ids_bytes), MTLResourceStorageModeShared
    )

    # Create output buffer
    output_size = num_tokens * hidden_size * 2  # float16 = 2 bytes
    output_buffer = ctx.device.newBufferWithLength_options_(
        output_size, MTLResourceStorageModeShared
    )

    # Create step context and run embedding
    from vllm_apple.engine.step import EngineStepContext

    with EngineStepContext(ctx, step_id=1, step_kind="decode", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        embedding.encode(step_ctx, token_ids_buffer, output_buffer, num_tokens)
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read back output using as_buffer method
    buffer_view = output_buffer.contents().as_buffer(output_size)
    output = np.frombuffer(buffer_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    print("Token IDs:", token_ids)
    print("Expected embedding values (first element of each row):")
    for i, tid in enumerate(token_ids):
        print(f"  Token {tid}: expected={tid}, got={output[i, 0]}, row={output[i, :4]}")

    # Check correctness
    correct = True
    for i, tid in enumerate(token_ids):
        expected = float(tid)
        if abs(output[i, 0] - expected) > 0.1:
            print(f"  MISMATCH: token {tid} expected {expected}, got {output[i, 0]}")
            correct = False

    if correct:
        print("\n✓ Embedding lookup works correctly!")
    else:
        print("\n✗ Embedding lookup has errors!")

    return correct

if __name__ == "__main__":
    main()

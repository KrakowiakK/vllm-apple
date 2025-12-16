#!/usr/bin/env python3
"""Verify RMSNorm correctness in isolation."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np

def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.ops.rmsnorm import EngineRMSNorm
    from vllm_apple.engine.step import EngineStepContext
    from Metal import MTLResourceStorageModeShared

    # Create context
    ctx = MetalEngineContext()

    # Test parameters
    num_tokens = 4
    hidden_size = 128
    eps = 1e-5

    # Create test input
    np.random.seed(42)
    input_np = np.random.randn(num_tokens, hidden_size).astype(np.float16)
    weight_np = np.random.randn(hidden_size).astype(np.float16)

    # Reference RMSNorm implementation
    def reference_rmsnorm(x, w, eps):
        x_f32 = x.astype(np.float32)
        w_f32 = w.astype(np.float32)
        mean_sq = np.mean(x_f32 ** 2, axis=-1, keepdims=True)
        rsqrt = 1.0 / np.sqrt(mean_sq + eps)
        return (x_f32 * rsqrt * w_f32).astype(np.float16)

    expected = reference_rmsnorm(input_np, weight_np, eps)

    # Create MTLBuffers
    input_bytes = input_np.tobytes()
    input_buffer = ctx.device.newBufferWithBytes_length_options_(
        input_bytes, len(input_bytes), MTLResourceStorageModeShared
    )

    weight_bytes = weight_np.tobytes()
    weight_buffer = ctx.device.newBufferWithBytes_length_options_(
        weight_bytes, len(weight_bytes), MTLResourceStorageModeShared
    )

    output_size = num_tokens * hidden_size * 2
    output_buffer = ctx.device.newBufferWithLength_options_(
        output_size, MTLResourceStorageModeShared
    )

    # Create op and run
    rmsnorm = EngineRMSNorm(ctx, hidden_size=hidden_size, eps=eps)
    rmsnorm.set_weights(weight_buffer)

    with EngineStepContext(ctx, step_id=1, step_kind="decode", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        rmsnorm.encode(
            step_ctx=step_ctx,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            num_tokens=num_tokens,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read back result
    output_view = output_buffer.contents().as_buffer(output_size)
    output_np = np.frombuffer(output_view, dtype=np.float16).reshape(num_tokens, hidden_size).copy()

    # Compare
    diff = np.abs(expected.astype(np.float32) - output_np.astype(np.float32))
    print("Input shape:", input_np.shape)
    print("Input sample:", input_np[0, :5])
    print("\nWeight sample:", weight_np[:5])
    print("\nExpected output sample:", expected[0, :5])
    print("Actual output sample:", output_np[0, :5])
    print(f"\nMax diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")

    # Check if close enough (allow for FP16 precision)
    if diff.max() < 0.01:
        print("\n✓ RMSNorm is correct!")
        return True
    else:
        print("\n✗ RMSNorm has errors!")
        return False

if __name__ == "__main__":
    main()

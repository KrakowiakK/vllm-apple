#!/usr/bin/env python3
"""Verify RoPE correctness against reference implementation."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np


def reference_rope(q, k, positions, head_size, rope_theta=10000.0):
    """Reference RoPE implementation.

    Args:
        q: Query tensor [num_tokens, num_heads, head_size]
        k: Key tensor [num_tokens, num_kv_heads, head_size]
        positions: Position indices [num_tokens]
        head_size: Dimension per head
        rope_theta: Base for rotation frequencies

    Returns:
        (q_rot, k_rot): Rotated tensors
    """
    num_tokens = q.shape[0]
    rotary_dim = head_size  # Apply to all dimensions

    # Compute frequency table: inv_freq = 1 / (theta ^ (2i / d))
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))

    # Compute position angles: [num_tokens, rotary_dim/2]
    # freqs[i, j] = positions[i] * inv_freq[j]
    freqs = np.outer(positions.astype(np.float32), inv_freq)

    # Get cos/sin: [num_tokens, rotary_dim/2]
    cos = np.cos(freqs)
    sin = np.sin(freqs)

    def apply_rope(x):
        """Apply RoPE to tensor [num_tokens, num_heads, head_size]."""
        x_f32 = x.astype(np.float32)
        out = np.zeros_like(x_f32)

        for t in range(num_tokens):
            for h in range(x.shape[1]):
                for i in range(0, rotary_dim, 2):
                    cos_val = cos[t, i // 2]
                    sin_val = sin[t, i // 2]

                    x0 = x_f32[t, h, i]
                    x1 = x_f32[t, h, i + 1]

                    # Rotation: [x0, x1] @ [[cos, -sin], [sin, cos]]
                    out[t, h, i] = x0 * cos_val - x1 * sin_val
                    out[t, h, i + 1] = x0 * sin_val + x1 * cos_val

        return out.astype(x.dtype)

    q_rot = apply_rope(q)
    k_rot = apply_rope(k)

    return q_rot, k_rot


def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.step import EngineStepContext
    from vllm_apple.engine.ops.elementwise import EngineRoPE
    from vllm_apple.engine.tensor import EngineTensor, EngineDType
    from Metal import MTLResourceStorageModeShared

    ctx = MetalEngineContext()

    # Test parameters
    num_tokens = 4
    num_heads = 4
    num_kv_heads = 2
    head_size = 64
    rope_theta = 10000.0

    print(f"Test config: tokens={num_tokens}, heads={num_heads}/{num_kv_heads}, head_size={head_size}")

    # Create RoPE op
    rope = EngineRoPE(
        ctx, head_size=head_size, num_heads=num_heads, num_kv_heads=num_kv_heads,
        max_position=512, base=rope_theta
    )

    # Create test input
    np.random.seed(42)

    # Q: [num_tokens, num_heads * head_size]
    q_np = np.random.randn(num_tokens, num_heads * head_size).astype(np.float16)
    # K: [num_tokens, num_kv_heads * head_size]
    k_np = np.random.randn(num_tokens, num_kv_heads * head_size).astype(np.float16)
    # Positions: [num_tokens]
    positions = np.array([0, 1, 2, 3], dtype=np.int32)

    # Reshape for reference computation
    q_shaped = q_np.reshape(num_tokens, num_heads, head_size)
    k_shaped = k_np.reshape(num_tokens, num_kv_heads, head_size)

    # Compute reference
    q_ref, k_ref = reference_rope(q_shaped, k_shaped, positions, head_size, rope_theta)

    # Create buffers for Metal
    # QKV layout: [Q][K][V] stacked
    q_size = num_tokens * num_heads * head_size
    k_size = num_tokens * num_kv_heads * head_size

    # Create Q buffer
    q_bytes = q_np.tobytes()
    q_buffer = ctx.device.newBufferWithBytes_length_options_(
        q_bytes, len(q_bytes), MTLResourceStorageModeShared
    )

    # Create K buffer
    k_bytes = k_np.tobytes()
    k_buffer = ctx.device.newBufferWithBytes_length_options_(
        k_bytes, len(k_bytes), MTLResourceStorageModeShared
    )

    # Create positions buffer
    positions_buffer = ctx.device.newBufferWithBytes_length_options_(
        positions.tobytes(), positions.nbytes, MTLResourceStorageModeShared
    )

    # Create K tensor (for the API)
    k_tensor = EngineTensor(
        buffer=k_buffer,
        shape=(num_tokens, num_kv_heads, head_size),
        dtype=EngineDType.FLOAT16,
        offset=0,
    )

    # Run RoPE
    with EngineStepContext(ctx, step_id=1, step_kind="prefill", num_tokens=num_tokens, num_seqs=1) as step_ctx:
        rope.encode(
            step_ctx=step_ctx,
            query=q_buffer,
            key=k_tensor,
            positions=positions_buffer,
            num_tokens=num_tokens,
            max_position_in_batch=3,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read back results
    q_result_view = q_buffer.contents().as_buffer(len(q_bytes))
    q_result = np.frombuffer(q_result_view, dtype=np.float16).reshape(num_tokens, num_heads, head_size).copy()

    k_result_view = k_buffer.contents().as_buffer(len(k_bytes))
    k_result = np.frombuffer(k_result_view, dtype=np.float16).reshape(num_tokens, num_kv_heads, head_size).copy()

    # Compare
    q_diff = np.abs(q_result.astype(np.float32) - q_ref.astype(np.float32))
    k_diff = np.abs(k_result.astype(np.float32) - k_ref.astype(np.float32))

    print(f"\nQ RoPE max diff: {q_diff.max():.6f}, mean: {q_diff.mean():.6f}")
    print(f"K RoPE max diff: {k_diff.max():.6f}, mean: {k_diff.mean():.6f}")

    # Show samples
    print(f"\nQ sample (token 0, head 0, first 8):")
    print(f"  Input:    {q_np[0, :8]}")
    print(f"  Expected: {q_ref[0, 0, :8].astype(np.float16)}")
    print(f"  Actual:   {q_result[0, 0, :8]}")

    print(f"\nQ sample (token 3, head 0, first 8):")
    print(f"  Input:    {q_np[3, :8]}")
    print(f"  Expected: {q_ref[3, 0, :8].astype(np.float16)}")
    print(f"  Actual:   {q_result[3, 0, :8]}")

    # Position 0 should have no rotation (cos=1, sin=0)
    print(f"\nPosition 0 rotation check (should be same as input for first few dims):")
    q_pos0_diff = np.abs(q_np[0].astype(np.float32) - q_result[0].flatten().astype(np.float32))
    print(f"  Q pos=0 diff: max={q_pos0_diff.max():.6f}")

    # Check if position matters
    print(f"\nPosition 3 vs position 0 Q[0,:]:")
    print(f"  Position 0: {q_result[0, 0, :5]}")
    print(f"  Position 3: {q_result[3, 0, :5]}")

    # Verification
    if q_diff.max() > 0.1 or k_diff.max() > 0.1:
        print("\n✗ RoPE has significant errors!")
        return False
    else:
        print("\n✓ RoPE is correct!")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

#!/usr/bin/env python3
"""Verify GEMM correctness in isolation."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'

import numpy as np

def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.ops.gemm import EngineGEMM
    from vllm_apple.engine.step import EngineStepContext
    from Metal import MTLResourceStorageModeShared

    # Create context
    ctx = MetalEngineContext()
    gemm = EngineGEMM(ctx)

    # Test GEMM with known values
    # A: [M, K] = [4, 8]
    # B: [K, N] = [8, 16]  (or [N, K] = [16, 8] if transpose_B)
    # C: [M, N] = [4, 16]
    M, K, N = 4, 8, 16

    # Create test inputs - simple pattern for debugging
    A = np.arange(M * K, dtype=np.float16).reshape(M, K)
    # Weight stored as [N, K] for transpose_B=True (PyTorch convention)
    B = np.arange(N * K, dtype=np.float16).reshape(N, K)

    # Expected output: A @ B.T
    expected = A.astype(np.float32) @ B.T.astype(np.float32)
    expected = expected.astype(np.float16)

    # Create MTLBuffers
    A_bytes = A.tobytes()
    A_buffer = ctx.device.newBufferWithBytes_length_options_(
        A_bytes, len(A_bytes), MTLResourceStorageModeShared
    )

    B_bytes = B.tobytes()
    B_buffer = ctx.device.newBufferWithBytes_length_options_(
        B_bytes, len(B_bytes), MTLResourceStorageModeShared
    )

    C_size = M * N * 2
    C_buffer = ctx.device.newBufferWithLength_options_(
        C_size, MTLResourceStorageModeShared
    )

    # Run GEMM
    with EngineStepContext(ctx, step_id=1, step_kind="decode", num_tokens=M, num_seqs=1) as step_ctx:
        gemm.encode(
            step_ctx=step_ctx,
            A=A_buffer,
            B=B_buffer,
            C=C_buffer,
            M=M,
            K=K,
            N=N,
            transpose_B=True,
        )
        step_ctx.end_encoding()
        step_ctx.submit()
        step_ctx.wait_until_completed()

    # Read back result
    C_view = C_buffer.contents().as_buffer(C_size)
    C_result = np.frombuffer(C_view, dtype=np.float16).reshape(M, N).copy()

    # Compare
    print("Input A:")
    print(A)
    print("\nWeight B (will be transposed):")
    print(B)
    print("\nExpected (A @ B.T):")
    print(expected)
    print("\nActual:")
    print(C_result)
    print("\nDifference:")
    diff = np.abs(expected.astype(np.float32) - C_result.astype(np.float32))
    print(diff)
    print(f"\nMax diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")

    # Check if close enough (allow for FP16 precision)
    if diff.max() < 0.1:
        print("\n✓ GEMM is correct!")
        return True
    else:
        print("\n✗ GEMM has errors!")
        return False

if __name__ == "__main__":
    main()

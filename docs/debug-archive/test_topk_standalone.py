#!/usr/bin/env python3
"""Standalone test for custom Metal Top-K selection kernel.

Tests correctness against numpy reference and benchmarks performance.
"""

import numpy as np
import time
import sys
import traceback


def test_metal_available():
    """Check if Metal is available."""
    try:
        import Metal
        from Metal import MTLCreateSystemDefaultDevice
        device = MTLCreateSystemDefaultDevice()
        if device is None:
            print("ERROR: Metal device not available")
            return False
        print(f"Metal device: {device.name()}")
        return True
    except ImportError as e:
        print(f"ERROR: Metal not available: {e}")
        return False


def create_metal_buffer(device, data, dtype=np.float16):
    """Create MTLBuffer from numpy array."""
    from Metal import MTLResourceStorageModeShared
    data_bytes = data.astype(dtype).tobytes()
    buffer = device.newBufferWithBytes_length_options_(
        data_bytes, len(data_bytes), MTLResourceStorageModeShared
    )
    return buffer


def read_metal_buffer(buffer, shape, dtype=np.float16):
    """Read numpy array from MTLBuffer."""
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buffer_view = buffer.contents().as_buffer(size)
    return np.frombuffer(buffer_view, dtype=dtype).reshape(shape).copy()


def read_metal_buffer_int(buffer, shape):
    """Read int32 array from MTLBuffer."""
    size = int(np.prod(shape)) * 4  # int32
    buffer_view = buffer.contents().as_buffer(size)
    return np.frombuffer(buffer_view, dtype=np.int32).reshape(shape).copy()


def topk_reference(logits, k):
    """Numpy reference for top-k selection."""
    num_tokens, vocab_size = logits.shape
    indices = np.zeros((num_tokens, k), dtype=np.int32)
    values = np.zeros((num_tokens, k), dtype=np.float16)

    for i in range(num_tokens):
        row = logits[i].astype(np.float32)
        top_indices = np.argsort(row)[-k:][::-1]
        indices[i] = top_indices
        values[i] = logits[i][top_indices]

    return indices, values


class MockStepContext:
    """Mock step context for testing kernel encoding."""

    def __init__(self, device, command_queue):
        self._device = device
        self._command_queue = command_queue
        self._cmd_buffer = None
        self._encoder = None
        self._is_encoding = True

    @property
    def is_encoding(self):
        return self._is_encoding

    def _ensure_cmd_buffer(self):
        if self._cmd_buffer is None:
            self._cmd_buffer = self._command_queue.commandBuffer()

    def get_compute_encoder(self):
        self._ensure_cmd_buffer()
        if self._encoder is None:
            self._encoder = self._cmd_buffer.computeCommandEncoder()
        return self._encoder

    def commit_and_wait(self):
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None
        if self._cmd_buffer is not None:
            self._cmd_buffer.commit()
            self._cmd_buffer.waitUntilCompleted()
            self._cmd_buffer = None


def test_topk_correctness(num_tokens, vocab_size, k, topk_op, engine_context):
    """Test Top-K correctness for given parameters."""
    np.random.seed(42)
    logits = np.random.randn(num_tokens, vocab_size).astype(np.float16)

    expected_indices, expected_values = topk_reference(logits, k)

    # Create buffers
    logits_buf = create_metal_buffer(engine_context.device, logits)
    indices_buf = engine_context.device.newBufferWithLength_options_(
        num_tokens * k * 4, 0  # int32
    )
    values_buf = engine_context.device.newBufferWithLength_options_(
        num_tokens * k * 2, 0  # float16
    )

    # Create step context
    step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)

    # Encode Top-K
    topk_op.encode(step_ctx, logits_buf, indices_buf, values_buf, num_tokens, vocab_size, k=k)

    # Execute
    step_ctx.commit_and_wait()

    # Read results
    actual_indices = read_metal_buffer_int(indices_buf, (num_tokens, k))
    actual_values = read_metal_buffer(values_buf, (num_tokens, k))

    # Compare - indices might differ for equal values, so check values
    # Sort both by index to compare
    all_correct = True
    for i in range(num_tokens):
        expected_set = set(expected_indices[i])
        actual_set = set(actual_indices[i])

        # Check if top-k indices produce same values
        expected_vals_sorted = sorted(expected_values[i], reverse=True)
        actual_vals_sorted = sorted(actual_values[i], reverse=True)

        # Check values match (allowing for ties)
        val_diff = np.max(np.abs(np.array(expected_vals_sorted, dtype=np.float32) -
                                  np.array(actual_vals_sorted, dtype=np.float32)))
        if val_diff > 0.001:
            all_correct = False

    return all_correct, actual_indices, actual_values, expected_indices, expected_values


def test_topk_performance(num_tokens, vocab_size, k, topk_op, engine_context, num_iters=50):
    """Benchmark Top-K performance."""
    np.random.seed(42)
    logits = np.random.randn(num_tokens, vocab_size).astype(np.float16)

    logits_buf = create_metal_buffer(engine_context.device, logits)
    indices_buf = engine_context.device.newBufferWithLength_options_(
        num_tokens * k * 4, 0
    )
    values_buf = engine_context.device.newBufferWithLength_options_(
        num_tokens * k * 2, 0
    )

    # Warmup
    for _ in range(5):
        step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)
        topk_op.encode(step_ctx, logits_buf, indices_buf, values_buf, num_tokens, vocab_size, k=k)
        step_ctx.commit_and_wait()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        step_ctx = MockStepContext(engine_context.device, engine_context.command_queue)
        topk_op.encode(step_ctx, logits_buf, indices_buf, values_buf, num_tokens, vocab_size, k=k)
        step_ctx.commit_and_wait()
    elapsed = time.perf_counter() - start

    avg_time_us = (elapsed / num_iters) * 1e6

    # Calculate data sizes
    full_readback_bytes = num_tokens * vocab_size * 2
    topk_readback_bytes = num_tokens * k * 6  # int32 + float16
    reduction_factor = full_readback_bytes / topk_readback_bytes

    return avg_time_us, full_readback_bytes, topk_readback_bytes, reduction_factor


def main():
    print("=" * 60)
    print("Custom Metal Top-K Selection Kernel Tests")
    print("=" * 60)

    # Check Metal availability
    if not test_metal_available():
        return 1

    # Import engine components
    try:
        from vllm_apple.engine.context import MetalEngineContext
        from vllm_apple.engine.ops.topk import EngineTopK
        print("Engine modules imported successfully")
    except ImportError as e:
        print(f"ERROR: Failed to import engine modules: {e}")
        traceback.print_exc()
        return 1

    # Create context and Top-K op
    try:
        engine_context = MetalEngineContext.get_instance()
        topk_op = EngineTopK(engine_context, k=50)
        print("Top-K operation created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create Top-K op: {e}")
        traceback.print_exc()
        return 1

    # Test configurations
    test_configs = [
        (1, 32000, 50, "Single token, small vocab"),
        (1, 128000, 50, "Single token, large vocab"),
        (8, 32000, 50, "Batch 8, small vocab"),
        (8, 128000, 50, "Batch 8, large vocab"),
        (16, 128000, 50, "Batch 16, large vocab"),
        (1, 128000, 8, "Single token, small k"),
        (8, 128000, 8, "Batch 8, small k"),
    ]

    print("\n" + "=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    all_passed = True
    for num_tokens, vocab_size, k, desc in test_configs:
        try:
            passed, actual_idx, actual_val, exp_idx, exp_val = test_topk_correctness(
                num_tokens, vocab_size, k, topk_op, engine_context
            )
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            print(f"[{status}] {desc} ({num_tokens}x{vocab_size}, k={k})")
        except Exception as e:
            all_passed = False
            print(f"[ERROR] {desc}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)
    print(f"\n{'Config':<40} {'Time (us)':<12} {'Full (KB)':<12} {'TopK (B)':<12} {'Reduction':<12}")
    print("-" * 90)

    perf_configs = [
        (1, 32000, 50, "Decode, 32K vocab"),
        (1, 128000, 50, "Decode, 128K vocab"),
        (8, 128000, 50, "Batch 8, 128K vocab"),
        (16, 128000, 50, "Batch 16, 128K vocab"),
        (32, 128000, 50, "Batch 32, 128K vocab"),
    ]

    for num_tokens, vocab_size, k, desc in perf_configs:
        try:
            time_us, full_bytes, topk_bytes, reduction = test_topk_performance(
                num_tokens, vocab_size, k, topk_op, engine_context
            )
            print(f"{desc:<40} {time_us:<12.1f} {full_bytes/1024:<12.1f} {topk_bytes:<12} {reduction:<12.0f}x")
        except Exception as e:
            print(f"{desc:<40} ERROR: {e}")

    print("\n" + "=" * 60)
    print("DATA TRANSFER SAVINGS")
    print("=" * 60)
    print("\nFor a model with 128K vocab and k=50:")
    full_size = 128000 * 2  # 256KB per token
    topk_size = 50 * 6  # 300 bytes per token
    print(f"  Full logits readback: {full_size / 1024:.1f} KB per token")
    print(f"  Top-K readback: {topk_size} bytes per token")
    print(f"  Reduction: {full_size / topk_size:.0f}x")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if all_passed:
        print("All correctness tests PASSED")
    else:
        print("Some correctness tests FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

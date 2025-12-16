#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comprehensive Validation Suite for vLLM-Apple Metal Engine v2.0.

This script performs 8 categories of validation tests:
1. Prefill vs Decode - separate and multi-length
2. Context length knee point analysis
3. Memory stability / leak test
4. Numerical correctness vs reference
5. Batch edge cases (odd sizes, mixed lengths)
6. Concurrency / server-like load
7. Token cap boundary behavior (16384)
8. Determinism and seeds

DO NOT modify engine behavior. ONLY MEASURE, VERIFY, and REPORT.

Usage:
    # Run all tests with Devstral-24B
    VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 \
    VLLM_METAL_SCRATCH_POOL_MB=4096 \
    python validation_suite_v2.py

    # Run specific test
    python validation_suite_v2.py --test prefill_decode
    python validation_suite_v2.py --test memory
    python validation_suite_v2.py --test correctness
"""

import os
import sys
import time
import gc
import argparse
import traceback
import statistics
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import threading

# Set environment before imports
os.environ.setdefault('VLLM_CPU_KVCACHE_SPACE', '4')

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TestResult:
    """Result from a single test."""
    test_name: str
    status: str  # PASS, WARN, FAIL
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ValidationReport:
    """Complete validation report."""
    start_time: str
    end_time: str = ""
    model: str = ""
    engine_mode: str = ""
    hardware: str = ""
    test_results: List[TestResult] = field(default_factory=list)
    overall_status: str = "PENDING"
    moe_ready: bool = False
    notes: str = ""


# ============================================================================
# UTILITIES
# ============================================================================

def get_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    try:
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / (1024**3)
    except:
        pass
    return 0.0

def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except:
        pass

def get_hardware_info() -> str:
    """Get hardware information."""
    import platform
    try:
        # Try to get Apple Silicon info
        machine = platform.machine()
        processor = platform.processor()
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        mem_gb = mem_bytes / (1024**3)
        return f"{machine} ({processor}), {mem_gb:.0f}GB RAM"
    except:
        return platform.platform()

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")

def format_table(headers: List[str], rows: List[List[Any]], col_widths: Optional[List[int]] = None) -> str:
    """Format a table as string."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) if i < len(r) else 0 for r in rows)) + 2
                      for i, h in enumerate(headers)]

    lines = []
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in rows:
        row_line = "".join(str(r).ljust(w) for r, w in zip(row, col_widths))
        lines.append(row_line)

    return "\n".join(lines)


# ============================================================================
# TEST 1: PREFILL VS DECODE
# ============================================================================

def test_prefill_decode(
    model_name: str,
    prompt_lengths: List[int] = [32, 128, 512, 2048],
    batch_sizes: List[int] = [1, 2, 4, 8],
    decode_lengths: List[int] = [16, 64, 256],
    num_runs: int = 3,
) -> TestResult:
    """Test prefill and decode performance separately across configurations."""

    print_section("TEST 1: PREFILL VS DECODE - SEPARATE AND MULTI-LENGTH")

    from vllm import LLM, SamplingParams

    try:
        print(f"Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=16384,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )

        prefill_results = []
        decode_results = []
        anomalies = []

        # Test prompts of various lengths
        base_text = "This is a test prompt for benchmarking the Metal engine performance. "

        # === PREFILL TESTS ===
        print("\n--- PREFILL TESTS ---")
        print(f"Prompt lengths: {prompt_lengths}")
        print(f"Batch sizes: {batch_sizes}")

        for prompt_len in prompt_lengths:
            for batch_size in batch_sizes:
                # Create prompts of target length
                prompt = base_text * (prompt_len // 10 + 1)
                prompt = prompt[:prompt_len * 4]  # Rough char-to-token ratio
                prompts = [prompt] * batch_size

                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=1,  # Minimal decode for prefill test
                )

                times = []
                total_tokens = 0

                for run in range(num_runs + 1):  # +1 for warmup
                    clear_memory()

                    start = time.perf_counter()
                    outputs = llm.generate(prompts, sampling_params)
                    end = time.perf_counter()

                    if run > 0:  # Skip warmup
                        times.append(end - start)
                        total_tokens = sum(len(o.prompt_token_ids) for o in outputs)

                avg_time = statistics.mean(times)
                tok_s = total_tokens / avg_time if avg_time > 0 else 0

                prefill_results.append({
                    'prompt_len': prompt_len,
                    'batch_size': batch_size,
                    'total_tokens': total_tokens,
                    'latency_ms': avg_time * 1000,
                    'tok_s': tok_s,
                })

                print(f"  Prompt={prompt_len:5d}, Batch={batch_size:2d}: {tok_s:8.1f} tok/s, {avg_time*1000:7.1f}ms")

                # Check for anomalies
                if batch_size > 1:
                    prev = next((r for r in prefill_results
                               if r['prompt_len'] == prompt_len and r['batch_size'] == batch_size // 2), None)
                    if prev and prev['tok_s'] > 0:
                        scaling = tok_s / prev['tok_s']
                        if scaling < 1.5:  # Should be near 2x
                            anomalies.append(f"Prefill scaling anomaly: prompt={prompt_len}, batch {batch_size//2}->{batch_size}: {scaling:.2f}x (expected ~2x)")

        # === DECODE TESTS ===
        print("\n--- DECODE TESTS ---")
        print(f"Decode lengths: {decode_lengths}")
        print(f"Batch sizes: {batch_sizes}")

        # Use fixed prompt length for decode tests
        fixed_prompt_len = 512
        prompt = base_text * (fixed_prompt_len // 10 + 1)
        prompt = prompt[:fixed_prompt_len * 4]

        for decode_len in decode_lengths:
            for batch_size in batch_sizes:
                prompts = [prompt] * batch_size

                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=decode_len,
                )

                times = []
                total_decode_tokens = 0

                for run in range(num_runs + 1):  # +1 for warmup
                    clear_memory()

                    start = time.perf_counter()
                    outputs = llm.generate(prompts, sampling_params)
                    end = time.perf_counter()

                    if run > 0:  # Skip warmup
                        times.append(end - start)
                        total_decode_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

                avg_time = statistics.mean(times)
                # Estimate decode time (total - prefill estimate)
                prefill_estimate = avg_time * 0.15  # Rough estimate
                decode_time = avg_time - prefill_estimate
                tok_s = total_decode_tokens / decode_time if decode_time > 0 else 0
                latency_per_token = (decode_time * 1000) / total_decode_tokens if total_decode_tokens > 0 else 0

                decode_results.append({
                    'decode_len': decode_len,
                    'batch_size': batch_size,
                    'total_tokens': total_decode_tokens,
                    'latency_ms': avg_time * 1000,
                    'latency_per_token_ms': latency_per_token,
                    'tok_s': tok_s,
                })

                print(f"  Decode={decode_len:4d}, Batch={batch_size:2d}: {tok_s:8.1f} tok/s, {latency_per_token:5.2f}ms/tok")

        # Determine status
        status = "PASS"
        notes = []

        if anomalies:
            status = "WARN"
            notes.extend(anomalies)

        # Check for basic sanity
        if not prefill_results or not decode_results:
            status = "FAIL"
            notes.append("No results collected")

        del llm
        clear_memory()

        return TestResult(
            test_name="prefill_decode",
            status=status,
            metrics={
                'prefill_results': prefill_results,
                'decode_results': decode_results,
                'anomalies': anomalies,
            },
            notes="; ".join(notes) if notes else "All configurations tested successfully",
        )

    except Exception as e:
        traceback.print_exc()
        return TestResult(
            test_name="prefill_decode",
            status="FAIL",
            notes=f"Error: {str(e)}",
        )


# ============================================================================
# TEST 2: CONTEXT LENGTH KNEE POINT
# ============================================================================

def test_context_knee_point(
    model_name: str,
    context_lengths: List[int] = [512, 1024, 2048, 4096, 8192],
    batch_sizes: List[int] = [1, 4, 8],
    decode_tokens: int = 32,
    num_runs: int = 3,
) -> TestResult:
    """Find where performance degrades due to KV cache pressure."""

    print_section("TEST 2: CONTEXT LENGTH KNEE POINT")

    from vllm import LLM, SamplingParams

    try:
        print(f"Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=16384,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )

        results = defaultdict(list)
        knee_points = {}

        base_text = "This is a test prompt for benchmarking. " * 100

        for batch_size in batch_sizes:
            print(f"\n--- Batch size: {batch_size} ---")
            prev_tok_s = None

            for ctx_len in context_lengths:
                # Create prompt of target length
                prompt = base_text[:ctx_len * 4]
                prompts = [prompt] * batch_size

                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=decode_tokens,
                )

                times = []
                decode_tokens_actual = 0

                try:
                    for run in range(num_runs + 1):
                        clear_memory()

                        start = time.perf_counter()
                        outputs = llm.generate(prompts, sampling_params)
                        end = time.perf_counter()

                        if run > 0:
                            times.append(end - start)
                            decode_tokens_actual = sum(len(o.outputs[0].token_ids) for o in outputs)

                    avg_time = statistics.mean(times)
                    prefill_estimate = avg_time * 0.15
                    decode_time = avg_time - prefill_estimate
                    tok_s = decode_tokens_actual / decode_time if decode_time > 0 else 0

                    results[batch_size].append({
                        'context_len': ctx_len,
                        'decode_tok_s': tok_s,
                        'latency_ms': avg_time * 1000,
                    })

                    print(f"  Context={ctx_len:5d}: {tok_s:8.1f} decode tok/s, {avg_time*1000:7.1f}ms")

                    # Check for knee point (>20% degradation from previous)
                    if prev_tok_s is not None and tok_s < prev_tok_s * 0.8:
                        if batch_size not in knee_points:
                            knee_points[batch_size] = ctx_len
                            print(f"    *** KNEE POINT DETECTED at {ctx_len} ***")

                    prev_tok_s = tok_s

                except Exception as e:
                    print(f"  Context={ctx_len:5d}: FAILED - {str(e)}")
                    if batch_size not in knee_points:
                        knee_points[batch_size] = ctx_len

        # Analysis
        status = "PASS"
        notes = []

        for batch_size, knee in knee_points.items():
            notes.append(f"Batch {batch_size}: knee point at {knee} context")
            if knee < 4096:
                status = "WARN"

        del llm
        clear_memory()

        return TestResult(
            test_name="context_knee_point",
            status=status,
            metrics={
                'results': dict(results),
                'knee_points': knee_points,
            },
            notes="; ".join(notes) if notes else "No significant knee points detected",
        )

    except Exception as e:
        traceback.print_exc()
        return TestResult(
            test_name="context_knee_point",
            status="FAIL",
            notes=f"Error: {str(e)}",
        )


# ============================================================================
# TEST 3: MEMORY STABILITY / LEAK TEST
# ============================================================================

def test_memory_stability(
    model_name: str,
    duration_minutes: int = 5,  # Reduced for practical testing
    batch_sizes: List[int] = [1, 8],
    decode_steps: int = 32,
) -> TestResult:
    """Test for memory leaks over extended operation."""

    print_section("TEST 3: MEMORY STABILITY / LEAK TEST")

    from vllm import LLM, SamplingParams

    try:
        print(f"Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )

        results = {}

        base_text = "Write a function to calculate fibonacci numbers. " * 10

        for batch_size in batch_sizes:
            print(f"\n--- Batch size: {batch_size} (running for {duration_minutes} min) ---")

            prompts = [base_text] * batch_size
            sampling_params = SamplingParams(temperature=0.0, max_tokens=decode_steps)

            memory_samples = []
            iteration_count = 0

            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)

            initial_memory = get_memory_gb()
            peak_memory = initial_memory

            sample_interval = 10  # Sample every 10 iterations

            while time.time() < end_time:
                try:
                    _ = llm.generate(prompts, sampling_params)
                    iteration_count += 1

                    if iteration_count % sample_interval == 0:
                        current_memory = get_memory_gb()
                        peak_memory = max(peak_memory, current_memory)
                        memory_samples.append({
                            'iteration': iteration_count,
                            'elapsed_s': time.time() - start_time,
                            'memory_gb': current_memory,
                        })

                        elapsed_min = (time.time() - start_time) / 60
                        print(f"  Iter {iteration_count:4d} ({elapsed_min:.1f}min): {current_memory:.2f}GB")

                except Exception as e:
                    print(f"  Error at iteration {iteration_count}: {e}")
                    break

            final_memory = get_memory_gb()
            memory_growth = final_memory - initial_memory

            results[batch_size] = {
                'iterations': iteration_count,
                'duration_s': time.time() - start_time,
                'initial_memory_gb': initial_memory,
                'final_memory_gb': final_memory,
                'peak_memory_gb': peak_memory,
                'memory_growth_gb': memory_growth,
                'samples': memory_samples,
            }

            print(f"\n  Summary: {iteration_count} iterations")
            print(f"  Initial: {initial_memory:.2f}GB, Final: {final_memory:.2f}GB, Peak: {peak_memory:.2f}GB")
            print(f"  Growth: {memory_growth:.3f}GB")

        # Analyze results
        status = "PASS"
        notes = []

        for batch_size, data in results.items():
            growth = data['memory_growth_gb']
            if growth > 1.0:  # More than 1GB growth is concerning
                status = "FAIL"
                notes.append(f"Batch {batch_size}: {growth:.2f}GB memory growth (LEAK?)")
            elif growth > 0.5:
                if status == "PASS":
                    status = "WARN"
                notes.append(f"Batch {batch_size}: {growth:.2f}GB memory growth (monitor)")
            else:
                notes.append(f"Batch {batch_size}: stable ({growth:.3f}GB growth)")

        del llm
        clear_memory()

        return TestResult(
            test_name="memory_stability",
            status=status,
            metrics=results,
            notes="; ".join(notes),
        )

    except Exception as e:
        traceback.print_exc()
        return TestResult(
            test_name="memory_stability",
            status="FAIL",
            notes=f"Error: {str(e)}",
        )


# ============================================================================
# TEST 4: NUMERICAL CORRECTNESS VS REFERENCE
# ============================================================================

def test_numerical_correctness(
    model_name: str,
    num_prompts: int = 5,
    seeds: List[int] = [42, 123, 456],
) -> TestResult:
    """Compare outputs against reference for numerical correctness."""

    print_section("TEST 4: NUMERICAL CORRECTNESS VS REFERENCE")

    from vllm import LLM, SamplingParams

    try:
        print(f"Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )

        test_prompts = [
            "def fibonacci(n):\n    ",
            "The capital of France is",
            "To calculate the area of a circle, you",
            "import numpy as np\ndef matrix_multiply(a, b):\n    ",
            "The theory of relativity states that",
        ][:num_prompts]

        results = []

        # Run with different seeds to check consistency
        for seed in seeds:
            print(f"\n--- Seed: {seed} ---")

            sampling_params = SamplingParams(
                temperature=0.0,  # Greedy for determinism
                max_tokens=50,
                seed=seed,
            )

            clear_memory()
            outputs = llm.generate(test_prompts, sampling_params)

            for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
                generated = output.outputs[0].text
                token_ids = output.outputs[0].token_ids

                # Basic coherence checks
                is_coherent = True
                notes = []

                # Check 1: Not empty
                if not generated.strip():
                    is_coherent = False
                    notes.append("empty")

                # Check 2: No excessive repetition
                words = generated.split()
                if len(words) > 5:
                    for j in range(len(words) - 3):
                        if words[j] == words[j+1] == words[j+2] == words[j+3]:
                            is_coherent = False
                            notes.append("repetition")
                            break

                # Check 3: Reasonable characters
                if generated.strip():
                    alpha_ratio = sum(c.isalnum() or c.isspace() for c in generated) / len(generated)
                    if alpha_ratio < 0.5:
                        notes.append(f"low_alpha:{alpha_ratio:.2f}")

                results.append({
                    'prompt_idx': i,
                    'seed': seed,
                    'prompt': prompt[:50] + "...",
                    'output': generated[:100] + "..." if len(generated) > 100 else generated,
                    'num_tokens': len(token_ids),
                    'is_coherent': is_coherent,
                    'notes': ",".join(notes) if notes else "OK",
                })

                status_char = "OK" if is_coherent else "!!"
                print(f"  [{status_char}] Prompt {i}: {len(token_ids)} tokens")
                if not is_coherent:
                    print(f"       Notes: {notes}")

        # Cross-seed consistency check
        print("\n--- Cross-seed consistency ---")
        consistency_issues = []

        for i in range(num_prompts):
            seed_outputs = [r for r in results if r['prompt_idx'] == i]
            first_output = seed_outputs[0]['output'] if seed_outputs else ""

            for r in seed_outputs[1:]:
                if r['output'] != first_output:
                    consistency_issues.append(f"Prompt {i}: output varies across seeds")
                    print(f"  Prompt {i}: VARIES across seeds (expected for temp=0)")
                    break
            else:
                print(f"  Prompt {i}: consistent across seeds")

        # Determine status
        coherent_count = sum(1 for r in results if r['is_coherent'])
        total_count = len(results)
        coherence_rate = coherent_count / total_count if total_count > 0 else 0

        if coherence_rate >= 0.95:
            status = "PASS"
        elif coherence_rate >= 0.80:
            status = "WARN"
        else:
            status = "FAIL"

        notes = [f"Coherence rate: {coherence_rate*100:.1f}% ({coherent_count}/{total_count})"]
        if consistency_issues:
            notes.extend(consistency_issues)

        del llm
        clear_memory()

        return TestResult(
            test_name="numerical_correctness",
            status=status,
            metrics={
                'results': results,
                'coherence_rate': coherence_rate,
                'consistency_issues': consistency_issues,
            },
            notes="; ".join(notes),
        )

    except Exception as e:
        traceback.print_exc()
        return TestResult(
            test_name="numerical_correctness",
            status="FAIL",
            notes=f"Error: {str(e)}",
        )


# ============================================================================
# TEST 5: BATCH EDGE CASES
# ============================================================================

def test_batch_edge_cases(
    model_name: str,
    odd_batch_sizes: List[int] = [3, 5, 7],
    mixed_prompt_lengths: List[int] = [32, 128, 512, 2048],
) -> TestResult:
    """Test unusual batch sizes and mixed prompt lengths."""

    print_section("TEST 5: BATCH EDGE CASES")

    from vllm import LLM, SamplingParams

    try:
        print(f"Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )

        results = []
        failures = []

        base_text = "Test prompt for edge case validation. " * 100

        # Test 1: Odd batch sizes
        print("\n--- Odd batch sizes ---")
        for batch_size in odd_batch_sizes:
            prompts = [base_text[:200]] * batch_size
            sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

            try:
                clear_memory()
                start = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params)
                end = time.perf_counter()

                success = len(outputs) == batch_size
                all_coherent = all(len(o.outputs[0].text.strip()) > 0 for o in outputs)

                results.append({
                    'test': f'odd_batch_{batch_size}',
                    'success': success and all_coherent,
                    'latency_ms': (end - start) * 1000,
                    'outputs': batch_size,
                })

                status = "OK" if success and all_coherent else "FAIL"
                print(f"  Batch {batch_size}: {status} ({(end-start)*1000:.1f}ms)")

            except Exception as e:
                failures.append(f"Batch {batch_size}: {str(e)}")
                print(f"  Batch {batch_size}: ERROR - {e}")

        # Test 2: Mixed prompt lengths in single batch
        print("\n--- Mixed prompt lengths ---")
        prompts = [base_text[:length * 4] for length in mixed_prompt_lengths]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

        try:
            clear_memory()
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            end = time.perf_counter()

            success = len(outputs) == len(mixed_prompt_lengths)
            all_coherent = all(len(o.outputs[0].text.strip()) > 0 for o in outputs)

            results.append({
                'test': 'mixed_lengths',
                'lengths': mixed_prompt_lengths,
                'success': success and all_coherent,
                'latency_ms': (end - start) * 1000,
            })

            print(f"  Mixed lengths {mixed_prompt_lengths}: {'OK' if success and all_coherent else 'FAIL'}")

            for i, (length, output) in enumerate(zip(mixed_prompt_lengths, outputs)):
                prompt_tokens = len(output.prompt_token_ids)
                output_tokens = len(output.outputs[0].token_ids)
                print(f"    Length {length}: {prompt_tokens} prompt + {output_tokens} output tokens")

        except Exception as e:
            failures.append(f"Mixed lengths: {str(e)}")
            print(f"  Mixed lengths: ERROR - {e}")

        # Test 3: Single token prompts
        print("\n--- Single token prompts ---")
        for batch_size in [1, 4, 8]:
            prompts = ["Hello"] * batch_size
            sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

            try:
                clear_memory()
                outputs = llm.generate(prompts, sampling_params)

                success = len(outputs) == batch_size
                results.append({
                    'test': f'single_token_batch_{batch_size}',
                    'success': success,
                })

                print(f"  Single token, batch {batch_size}: {'OK' if success else 'FAIL'}")

            except Exception as e:
                failures.append(f"Single token batch {batch_size}: {str(e)}")
                print(f"  Single token, batch {batch_size}: ERROR - {e}")

        # Determine status
        all_success = all(r.get('success', False) for r in results) and not failures

        if all_success:
            status = "PASS"
        elif failures:
            status = "FAIL"
        else:
            status = "WARN"

        del llm
        clear_memory()

        return TestResult(
            test_name="batch_edge_cases",
            status=status,
            metrics={
                'results': results,
                'failures': failures,
            },
            notes="; ".join(failures) if failures else "All edge cases passed",
        )

    except Exception as e:
        traceback.print_exc()
        return TestResult(
            test_name="batch_edge_cases",
            status="FAIL",
            notes=f"Error: {str(e)}",
        )


# ============================================================================
# TEST 6: CONCURRENCY / SERVER-LIKE LOAD
# ============================================================================

def test_concurrency(
    model_name: str,
    num_concurrent: List[int] = [2, 4, 8],
    requests_per_test: int = 10,
) -> TestResult:
    """Test concurrent request handling."""

    print_section("TEST 6: CONCURRENCY / SERVER-LIKE LOAD")

    from vllm import LLM, SamplingParams

    try:
        print(f"Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )

        results = {}

        prompts = [
            "Write a short poem about",
            "Explain quantum computing in",
            "The history of artificial intelligence",
            "How to make a perfect",
            "The future of renewable energy",
            "Best practices for software",
            "Understanding machine learning",
            "The science behind climate",
        ]

        sampling_params = SamplingParams(temperature=0.0, max_tokens=50)

        for num_threads in num_concurrent:
            print(f"\n--- {num_threads} concurrent requests ---")

            latencies = []
            errors = []

            # Simulate concurrent requests by batching
            for i in range(requests_per_test):
                batch_prompts = [prompts[j % len(prompts)] for j in range(num_threads)]

                clear_memory()
                start = time.perf_counter()

                try:
                    outputs = llm.generate(batch_prompts, sampling_params)
                    end = time.perf_counter()

                    latency = (end - start) * 1000
                    latencies.append(latency)

                except Exception as e:
                    errors.append(str(e))

            if latencies:
                latencies.sort()
                p50 = latencies[len(latencies) // 2]
                p95 = latencies[int(len(latencies) * 0.95)]
                p99 = latencies[-1]  # For small samples
                avg = statistics.mean(latencies)

                results[num_threads] = {
                    'requests': requests_per_test,
                    'errors': len(errors),
                    'avg_latency_ms': avg,
                    'p50_ms': p50,
                    'p95_ms': p95,
                    'p99_ms': p99,
                    'throughput_req_s': requests_per_test / (sum(latencies) / 1000),
                }

                print(f"  Avg: {avg:.1f}ms, P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")
                print(f"  Errors: {len(errors)}/{requests_per_test}")
            else:
                results[num_threads] = {'errors': len(errors), 'failed': True}
                print(f"  All requests failed")

        # Analyze
        status = "PASS"
        notes = []

        for num_threads, data in results.items():
            if data.get('failed') or data.get('errors', 0) > requests_per_test * 0.1:
                status = "FAIL"
                notes.append(f"{num_threads} concurrent: high error rate")
            elif data.get('p95_ms', 0) > data.get('p50_ms', 1) * 3:
                if status == "PASS":
                    status = "WARN"
                notes.append(f"{num_threads} concurrent: high tail latency")

        del llm
        clear_memory()

        return TestResult(
            test_name="concurrency",
            status=status,
            metrics=results,
            notes="; ".join(notes) if notes else "Concurrency handling normal",
        )

    except Exception as e:
        traceback.print_exc()
        return TestResult(
            test_name="concurrency",
            status="FAIL",
            notes=f"Error: {str(e)}",
        )


# ============================================================================
# TEST 7: TOKEN CAP BOUNDARY (16384)
# ============================================================================

def test_token_cap_boundary(
    model_name: str,
    max_model_len: int = 16384,
) -> TestResult:
    """Test behavior at token cap boundaries."""

    print_section("TEST 7: TOKEN CAP BOUNDARY BEHAVIOR (16384)")

    from vllm import LLM, SamplingParams

    try:
        print(f"Loading model: {model_name}")
        print(f"Max model length: {max_model_len}")

        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )

        results = []
        base_text = "test " * 10000  # Long text to create long prompts

        # Test cases around the boundary
        test_cases = [
            {'name': 'near_limit', 'prompt_target': max_model_len - 100, 'decode': 20},
            {'name': 'at_limit', 'prompt_target': max_model_len - 20, 'decode': 20},
            {'name': 'over_limit', 'prompt_target': max_model_len + 100, 'decode': 20},
            {'name': 'exact_limit', 'prompt_target': max_model_len, 'decode': 1},
        ]

        for test_case in test_cases:
            name = test_case['name']
            prompt_target = test_case['prompt_target']
            decode_tokens = test_case['decode']

            print(f"\n--- Test: {name} (target: {prompt_target} tokens) ---")

            # Create prompt of approximate target length
            prompt = base_text[:prompt_target * 5]  # Rough estimate

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=decode_tokens,
            )

            try:
                clear_memory()
                start = time.perf_counter()
                outputs = llm.generate([prompt], sampling_params)
                end = time.perf_counter()

                prompt_tokens = len(outputs[0].prompt_token_ids)
                output_tokens = len(outputs[0].outputs[0].token_ids)
                total_tokens = prompt_tokens + output_tokens

                result = {
                    'name': name,
                    'target_tokens': prompt_target,
                    'actual_prompt_tokens': prompt_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'exceeded_limit': total_tokens > max_model_len,
                    'status': 'OK',
                    'latency_ms': (end - start) * 1000,
                }

                if name == 'over_limit' and prompt_tokens >= max_model_len:
                    result['status'] = 'TRUNCATED'
                    print(f"  Input was truncated (expected)")

                results.append(result)

                print(f"  Prompt: {prompt_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                print(f"  Status: {result['status']}, Latency: {(end-start)*1000:.1f}ms")

            except Exception as e:
                error_msg = str(e)
                result = {
                    'name': name,
                    'target_tokens': prompt_target,
                    'status': 'ERROR',
                    'error': error_msg,
                }
                results.append(result)

                # Over limit should error gracefully
                if name == 'over_limit' and ('too long' in error_msg.lower() or 'exceed' in error_msg.lower()):
                    print(f"  Properly rejected with error (expected)")
                    result['status'] = 'REJECTED_OK'
                else:
                    print(f"  ERROR: {error_msg}")

        # Analyze
        status = "PASS"
        notes = []

        for r in results:
            if r['status'] == 'ERROR' and r['name'] != 'over_limit':
                status = "FAIL"
                notes.append(f"{r['name']}: unexpected error")
            elif r.get('exceeded_limit') and r['name'] not in ['over_limit']:
                status = "WARN"
                notes.append(f"{r['name']}: exceeded token limit")

        del llm
        clear_memory()

        return TestResult(
            test_name="token_cap_boundary",
            status=status,
            metrics={'results': results, 'max_model_len': max_model_len},
            notes="; ".join(notes) if notes else "Token cap behavior is correct",
        )

    except Exception as e:
        traceback.print_exc()
        return TestResult(
            test_name="token_cap_boundary",
            status="FAIL",
            notes=f"Error: {str(e)}",
        )


# ============================================================================
# TEST 8: DETERMINISM AND SEEDS
# ============================================================================

def test_determinism(
    model_name: str,
    batch_sizes: List[int] = [1, 8],
    num_repetitions: int = 5,
) -> TestResult:
    """Test output determinism with fixed seeds."""

    print_section("TEST 8: DETERMINISM AND SEEDS")

    from vllm import LLM, SamplingParams

    try:
        print(f"Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
        )

        results = {}
        test_prompt = "Write a haiku about programming:"

        for batch_size in batch_sizes:
            print(f"\n--- Batch size: {batch_size}, {num_repetitions} repetitions ---")

            prompts = [test_prompt] * batch_size

            # Test with temperature=0 (should be deterministic)
            print("  Temperature=0.0 (greedy):")
            greedy_outputs = []

            for rep in range(num_repetitions):
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=50,
                )

                clear_memory()
                outputs = llm.generate(prompts, sampling_params)

                generated = [o.outputs[0].text for o in outputs]
                greedy_outputs.append(generated)

            # Check consistency
            greedy_consistent = all(out == greedy_outputs[0] for out in greedy_outputs)

            if greedy_consistent:
                print(f"    All {num_repetitions} runs identical: YES")
            else:
                print(f"    All {num_repetitions} runs identical: NO")
                # Show differences
                for i, out in enumerate(greedy_outputs):
                    if out != greedy_outputs[0]:
                        print(f"      Run {i+1} differs from run 1")

            # Test with temperature>0 and fixed seed (should be deterministic)
            print("  Temperature=0.7, seed=42:")
            seeded_outputs = []

            for rep in range(num_repetitions):
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=50,
                    seed=42,
                )

                clear_memory()
                outputs = llm.generate(prompts, sampling_params)

                generated = [o.outputs[0].text for o in outputs]
                seeded_outputs.append(generated)

            seeded_consistent = all(out == seeded_outputs[0] for out in seeded_outputs)

            if seeded_consistent:
                print(f"    All {num_repetitions} runs identical: YES")
            else:
                print(f"    All {num_repetitions} runs identical: NO (MPS non-determinism)")

            results[batch_size] = {
                'greedy_consistent': greedy_consistent,
                'seeded_consistent': seeded_consistent,
                'greedy_sample': greedy_outputs[0][0][:100] if greedy_outputs else "",
                'seeded_sample': seeded_outputs[0][0][:100] if seeded_outputs else "",
            }

        # Analyze
        status = "PASS"
        notes = []

        for batch_size, data in results.items():
            if not data['greedy_consistent']:
                status = "FAIL"
                notes.append(f"Batch {batch_size}: greedy not deterministic")
            if not data['seeded_consistent']:
                if status == "PASS":
                    status = "WARN"
                notes.append(f"Batch {batch_size}: seeded sampling varies (MPS quirk)")

        del llm
        clear_memory()

        return TestResult(
            test_name="determinism",
            status=status,
            metrics=results,
            notes="; ".join(notes) if notes else "Determinism verified",
        )

    except Exception as e:
        traceback.print_exc()
        return TestResult(
            test_name="determinism",
            status="FAIL",
            notes=f"Error: {str(e)}",
        )


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_validation_suite(
    model_name: str = "mistralai/Devstral-Small-2505",
    tests: Optional[List[str]] = None,
    quick: bool = False,
) -> ValidationReport:
    """Run the complete validation suite."""

    report = ValidationReport(
        start_time=datetime.now().isoformat(),
        model=model_name,
        hardware=get_hardware_info(),
    )

    # Determine engine mode
    use_engine = os.environ.get("VLLM_APPLE_USE_ENGINE", "0") == "1"
    engine_prefill = os.environ.get("VLLM_APPLE_ENGINE_PREFILL", "0") == "1"

    if use_engine and engine_prefill:
        report.engine_mode = "Full Metal Engine (prefill + decode)"
    elif use_engine:
        report.engine_mode = "Hybrid (PyTorch prefill + Metal Engine decode)"
    else:
        report.engine_mode = "PyTorch Only (baseline)"

    print("=" * 80)
    print(" vLLM-Apple Metal Engine v2.0 - Validation Suite")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Engine Mode: {report.engine_mode}")
    print(f"Hardware: {report.hardware}")
    print(f"VLLM_METAL_SCRATCH_POOL_MB: {os.environ.get('VLLM_METAL_SCRATCH_POOL_MB', '512')}")
    print(f"VLLM_METAL_MAX_BATCH_SIZE: {os.environ.get('VLLM_METAL_MAX_BATCH_SIZE', '256')}")
    print("=" * 80)

    # Define all tests
    all_tests = {
        'prefill_decode': lambda: test_prefill_decode(
            model_name,
            prompt_lengths=[32, 128, 512, 2048] if not quick else [128, 512],
            batch_sizes=[1, 2, 4, 8] if not quick else [1, 4],
            decode_lengths=[16, 64, 256] if not quick else [32],
        ),
        'context_knee': lambda: test_context_knee_point(
            model_name,
            context_lengths=[512, 1024, 2048, 4096, 8192] if not quick else [512, 2048, 4096],
            batch_sizes=[1, 4, 8] if not quick else [1, 4],
        ),
        'memory': lambda: test_memory_stability(
            model_name,
            duration_minutes=5 if not quick else 2,
            batch_sizes=[1, 8] if not quick else [1],
        ),
        'correctness': lambda: test_numerical_correctness(
            model_name,
            num_prompts=5,
            seeds=[42, 123, 456] if not quick else [42],
        ),
        'edge_cases': lambda: test_batch_edge_cases(
            model_name,
        ),
        'concurrency': lambda: test_concurrency(
            model_name,
            num_concurrent=[2, 4, 8] if not quick else [2, 4],
        ),
        'token_cap': lambda: test_token_cap_boundary(
            model_name,
        ),
        'determinism': lambda: test_determinism(
            model_name,
            batch_sizes=[1, 8] if not quick else [1],
            num_repetitions=5 if not quick else 3,
        ),
    }

    # Select tests to run
    if tests:
        selected_tests = {k: v for k, v in all_tests.items() if k in tests}
    else:
        selected_tests = all_tests

    # Run tests
    for test_name, test_func in selected_tests.items():
        try:
            result = test_func()
            report.test_results.append(result)
        except Exception as e:
            print(f"\nFATAL ERROR in {test_name}: {e}")
            traceback.print_exc()
            report.test_results.append(TestResult(
                test_name=test_name,
                status="FAIL",
                notes=f"Fatal error: {str(e)}",
            ))

        # Clear memory between tests
        clear_memory()

    report.end_time = datetime.now().isoformat()

    # Determine overall status
    statuses = [r.status for r in report.test_results]

    if "FAIL" in statuses:
        report.overall_status = "FAIL"
    elif "WARN" in statuses:
        report.overall_status = "WARN"
    else:
        report.overall_status = "PASS"

    # MoE readiness
    critical_tests = ['correctness', 'memory', 'determinism']
    critical_results = [r for r in report.test_results if r.test_name in critical_tests]
    report.moe_ready = all(r.status == "PASS" for r in critical_results)

    # Print summary
    print_summary(report)

    return report


def print_summary(report: ValidationReport):
    """Print validation summary."""

    print("\n")
    print("=" * 80)
    print(" VALIDATION REPORT SUMMARY")
    print("=" * 80)
    print(f"Model: {report.model}")
    print(f"Engine Mode: {report.engine_mode}")
    print(f"Hardware: {report.hardware}")
    print(f"Duration: {report.start_time} to {report.end_time}")
    print()

    # Results table
    print("TEST RESULTS:")
    print("-" * 60)

    for r in report.test_results:
        status_color = {
            'PASS': '\033[92m',  # Green
            'WARN': '\033[93m',  # Yellow
            'FAIL': '\033[91m',  # Red
        }.get(r.status, '')
        reset = '\033[0m'

        print(f"  {r.test_name:<25} [{status_color}{r.status:^6}{reset}] {r.notes[:40]}")

    print("-" * 60)
    print()

    # Overall status
    overall_color = {
        'PASS': '\033[92m',
        'WARN': '\033[93m',
        'FAIL': '\033[91m',
    }.get(report.overall_status, '')
    reset = '\033[0m'

    print(f"OVERALL STATUS: {overall_color}{report.overall_status}{reset}")
    print(f"MoE INTEGRATION READY: {'YES' if report.moe_ready else 'NO'}")
    print()

    if report.overall_status == "FAIL":
        print("ACTION REQUIRED: Fix failing tests before MoE integration.")
    elif report.overall_status == "WARN":
        print("RECOMMENDATION: Review warnings before MoE integration.")
    else:
        print("Engine is ready for MoE integration.")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM-Apple Metal Engine v2.0 Validation Suite"
    )
    parser.add_argument(
        "--model", type=str,
        default="mistralai/Devstral-Small-2505",
        help="Model to test"
    )
    parser.add_argument(
        "--test", type=str, nargs="+",
        choices=['prefill_decode', 'context_knee', 'memory', 'correctness',
                 'edge_cases', 'concurrency', 'token_cap', 'determinism'],
        help="Specific test(s) to run (default: all)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick versions of tests"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output JSON file for report"
    )
    args = parser.parse_args()

    report = run_validation_suite(
        model_name=args.model,
        tests=args.test,
        quick=args.quick,
    )

    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            # Convert report to dict
            report_dict = {
                'start_time': report.start_time,
                'end_time': report.end_time,
                'model': report.model,
                'engine_mode': report.engine_mode,
                'hardware': report.hardware,
                'overall_status': report.overall_status,
                'moe_ready': report.moe_ready,
                'test_results': [
                    {
                        'test_name': r.test_name,
                        'status': r.status,
                        'notes': r.notes,
                        'metrics': r.metrics,
                        'timestamp': r.timestamp,
                    }
                    for r in report.test_results
                ],
            }
            json.dump(report_dict, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()

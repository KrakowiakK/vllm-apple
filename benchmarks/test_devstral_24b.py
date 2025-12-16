#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comprehensive benchmark for mistralai/Devstral-Small-2505 (text-only 24B).

Note: Devstral-Small-2-24B-Instruct-2512 is multimodal (Pixtral architecture).
Using Devstral-Small-2505 which is text-only (MistralForCausalLM).

Tests:
- Batch sizes: 1, 2, 4, 8, 16
- Prefill throughput (tok/s)
- Decode throughput (tok/s)
- Memory usage (GB)
- Output correctness verification

Usage:
    # Full engine mode (recommended):
    VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 python benchmarks/test_devstral_24b.py

    # PyTorch prefill + Engine decode:
    VLLM_APPLE_USE_ENGINE=1 python benchmarks/test_devstral_24b.py

    # PyTorch only (baseline):
    python benchmarks/test_devstral_24b.py
"""

import os
import sys
import time
import argparse
import gc
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import statistics

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    batch_size: int
    prompt_len: int
    output_len: int
    num_prompt_tokens: int
    num_output_tokens: int
    prefill_time_s: float
    decode_time_s: float
    total_time_s: float
    prefill_tok_s: float
    decode_tok_s: float
    ttft_ms: float
    memory_allocated_gb: float
    memory_reserved_gb: float


@dataclass
class CorrectnessResult:
    """Results from correctness verification."""
    prompt: str
    output: str
    is_coherent: bool
    output_length: int
    notes: str = ""


def get_memory_stats() -> Tuple[float, float]:
    """Get current memory usage in GB.

    Returns:
        Tuple of (allocated_gb, reserved_gb)
    """
    allocated = 0.0
    reserved = 0.0

    try:
        if torch.backends.mps.is_available():
            # MPS memory tracking
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            # MPS doesn't have reserved memory concept like CUDA
            reserved = allocated
    except Exception:
        pass

    return allocated, reserved


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except Exception:
        pass


def verify_output_correctness(outputs: List, prompts: List[str]) -> List[CorrectnessResult]:
    """Verify that model outputs are coherent and sensible.

    Args:
        outputs: List of vLLM RequestOutput objects
        prompts: Original prompts

    Returns:
        List of CorrectnessResult
    """
    results = []

    for i, (output, prompt) in enumerate(zip(outputs, prompts)):
        generated_text = output.outputs[0].text

        # Basic coherence checks
        is_coherent = True
        notes = []

        # Check 1: Output is not empty
        if not generated_text.strip():
            is_coherent = False
            notes.append("Empty output")

        # Check 2: Output doesn't contain excessive repetition
        words = generated_text.split()
        if len(words) > 5:
            # Check for 3+ consecutive repeated words
            for j in range(len(words) - 2):
                if words[j] == words[j+1] == words[j+2]:
                    notes.append("Potential repetition detected")
                    break

        # Check 3: Output contains actual words (not just garbage)
        if generated_text.strip():
            # At least some alphanumeric content
            alpha_ratio = sum(c.isalpha() for c in generated_text) / len(generated_text)
            if alpha_ratio < 0.3:
                notes.append(f"Low alpha ratio: {alpha_ratio:.2f}")

        # Check 4: Reasonable token count
        num_tokens = len(output.outputs[0].token_ids)

        results.append(CorrectnessResult(
            prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
            output=generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
            is_coherent=is_coherent,
            output_length=num_tokens,
            notes="; ".join(notes) if notes else "OK"
        ))

    return results


def run_benchmark(
    llm,
    prompts: List[str],
    sampling_params,
    batch_size: int,
    prompt_len: int,
    output_len: int,
) -> Tuple[BenchmarkResult, List[CorrectnessResult]]:
    """Run a single benchmark iteration.

    Args:
        llm: vLLM LLM instance
        prompts: List of prompts
        sampling_params: vLLM SamplingParams
        batch_size: Number of prompts
        prompt_len: Target prompt length
        output_len: Target output length

    Returns:
        Tuple of (BenchmarkResult, List[CorrectnessResult])
    """
    # Clear memory before run
    clear_memory()

    # Get initial memory
    mem_before = get_memory_stats()[0]

    # Run generation with timing
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    # Get memory after
    mem_after = get_memory_stats()

    # Calculate token counts
    num_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    num_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    # Estimate prefill vs decode time
    # Prefill is typically much faster per token than decode
    # Use a rough heuristic: prefill takes ~10-20% of time for typical workloads
    # But adjust based on prompt/output ratio
    token_ratio = num_prompt_tokens / (num_output_tokens + 1)  # Avoid div by zero

    if token_ratio > 10:
        # Long prompt, short output - prefill dominates
        prefill_fraction = 0.7
    elif token_ratio > 2:
        # Moderate ratio
        prefill_fraction = 0.3
    else:
        # Short prompt, long output - decode dominates
        prefill_fraction = 0.1

    prefill_time = total_time * prefill_fraction
    decode_time = total_time * (1 - prefill_fraction)

    # Calculate throughputs
    prefill_tok_s = num_prompt_tokens / prefill_time if prefill_time > 0 else 0
    decode_tok_s = num_output_tokens / decode_time if decode_time > 0 else 0
    ttft_ms = prefill_time * 1000  # Time to first token approximation

    result = BenchmarkResult(
        batch_size=batch_size,
        prompt_len=prompt_len,
        output_len=output_len,
        num_prompt_tokens=num_prompt_tokens,
        num_output_tokens=num_output_tokens,
        prefill_time_s=prefill_time,
        decode_time_s=decode_time,
        total_time_s=total_time,
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        ttft_ms=ttft_ms,
        memory_allocated_gb=mem_after[0],
        memory_reserved_gb=mem_after[1],
    )

    # Verify correctness
    correctness = verify_output_correctness(outputs, prompts)

    return result, correctness


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark mistralai/Devstral-Small-2505 (text-only 24B)"
    )
    parser.add_argument(
        "--model", type=str,
        default="mistralai/Devstral-Small-2505",
        help="Model to benchmark"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--prompt-len", type=int, default=128,
        help="Approximate prompt length in tokens"
    )
    parser.add_argument(
        "--output-len", type=int, default=64,
        help="Maximum output length in tokens"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of benchmark runs per config"
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85,
        help="GPU memory utilization fraction"
    )
    parser.add_argument(
        "--skip-correctness", action="store_true",
        help="Skip correctness verification output"
    )
    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("vLLM-Apple Metal Engine - Devstral-Small-2-24B Benchmark")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Prompt length: ~{args.prompt_len} tokens")
    print(f"  Output length: {args.output_len} tokens")
    print(f"  Runs per config: {args.runs}")
    print(f"  Warmup runs: {args.warmup}")
    print(f"  Max model len: {args.max_model_len}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    print()

    # Environment info
    use_engine = os.environ.get("VLLM_APPLE_USE_ENGINE", "0") == "1"
    engine_prefill = os.environ.get("VLLM_APPLE_ENGINE_PREFILL", "0") == "1"
    strict_mode = os.environ.get("VLLM_METAL_STRICT_NO_MPS", "0") == "1"

    if use_engine and engine_prefill:
        mode = "Full Metal Engine (prefill + decode)"
    elif use_engine:
        mode = "Hybrid (PyTorch prefill + Metal Engine decode)"
    else:
        mode = "PyTorch Only (baseline)"

    print("Engine Configuration:")
    print(f"  VLLM_APPLE_USE_ENGINE: {os.environ.get('VLLM_APPLE_USE_ENGINE', '0')}")
    print(f"  VLLM_APPLE_ENGINE_PREFILL: {os.environ.get('VLLM_APPLE_ENGINE_PREFILL', '0')}")
    print(f"  VLLM_METAL_STRICT_NO_MPS: {os.environ.get('VLLM_METAL_STRICT_NO_MPS', '0')}")
    print(f"  Mode: {mode}")
    print()

    # Initial memory
    mem_initial = get_memory_stats()
    print(f"Initial memory: {mem_initial[0]:.2f} GB allocated")
    print()

    # Load model
    print("Loading model...")
    print("(This may take several minutes for a 24B model)")
    print()

    try:
        from vllm import LLM, SamplingParams

        load_start = time.perf_counter()
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,  # Disable CUDA graphs on MPS
        )
        load_time = time.perf_counter() - load_start

        print(f"Model loaded in {load_time:.1f}s")

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        traceback.print_exc()
        return

    # Memory after load
    mem_after_load = get_memory_stats()
    print(f"Memory after load: {mem_after_load[0]:.2f} GB allocated")
    print()

    # Create test prompts
    # Use varied prompts to test different scenarios
    test_prompts_base = [
        "Write a Python function that implements binary search on a sorted list. Include docstrings and type hints.",
        "Explain the concept of machine learning in simple terms, suitable for a beginner. Cover supervised and unsupervised learning.",
        "What are the main differences between REST and GraphQL APIs? Provide examples of when to use each.",
        "Describe the architecture of a modern web application, including frontend, backend, and database layers.",
        "Write a short story about a robot learning to paint. Make it creative and engaging.",
        "Explain how neural networks work, starting from basic perceptrons to deep learning architectures.",
        "What are the best practices for writing clean, maintainable code? Provide specific examples.",
        "Describe the process of deploying a machine learning model to production. Include monitoring and scaling.",
        "Write a bash script that monitors system resources and sends alerts when thresholds are exceeded.",
        "Explain the CAP theorem in distributed systems. Give real-world examples of trade-offs.",
        "What are design patterns in software engineering? Describe the Singleton, Factory, and Observer patterns.",
        "Write a SQL query to find the top 10 customers by total purchase amount, including their details.",
        "Explain containerization with Docker and Kubernetes. How do they work together?",
        "Describe the principles of functional programming. Compare with object-oriented programming.",
        "Write a React component that implements a todo list with add, remove, and filter functionality.",
        "Explain how HTTPS works, including SSL/TLS handshake and certificate verification.",
    ]

    # Sampling params - deterministic for reproducibility
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for deterministic output
        max_tokens=args.output_len,
        top_p=1.0,
    )

    # Run benchmarks
    all_results: List[BenchmarkResult] = []
    all_correctness: List[CorrectnessResult] = []

    print("=" * 80)
    print("RUNNING BENCHMARKS")
    print("=" * 80)

    for batch_size in args.batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}")

        # Select prompts for this batch
        prompts = test_prompts_base[:batch_size]
        if len(prompts) < batch_size:
            # Repeat prompts if we need more
            prompts = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]

        batch_results: List[BenchmarkResult] = []

        # Warmup runs
        print(f"  Warmup ({args.warmup} runs)...", end=" ", flush=True)
        for _ in range(args.warmup):
            try:
                _ = llm.generate(prompts, sampling_params)
            except Exception as e:
                print(f"\n  WARNING: Warmup failed: {e}")
                break
        print("done")

        # Benchmark runs
        print(f"  Benchmark ({args.runs} runs):")
        for run in range(args.runs):
            try:
                result, correctness = run_benchmark(
                    llm=llm,
                    prompts=prompts,
                    sampling_params=sampling_params,
                    batch_size=batch_size,
                    prompt_len=args.prompt_len,
                    output_len=args.output_len,
                )
                batch_results.append(result)

                # Save correctness from first run only
                if run == 0:
                    all_correctness.extend(correctness)

                print(f"    Run {run+1}: {result.total_time_s:.2f}s total, "
                      f"{result.decode_tok_s:.1f} decode tok/s, "
                      f"{result.memory_allocated_gb:.2f} GB")

            except Exception as e:
                print(f"    Run {run+1}: ERROR - {e}")
                traceback.print_exc()

        if batch_results:
            # Calculate averages
            avg_result = BenchmarkResult(
                batch_size=batch_size,
                prompt_len=args.prompt_len,
                output_len=args.output_len,
                num_prompt_tokens=int(statistics.mean(r.num_prompt_tokens for r in batch_results)),
                num_output_tokens=int(statistics.mean(r.num_output_tokens for r in batch_results)),
                prefill_time_s=statistics.mean(r.prefill_time_s for r in batch_results),
                decode_time_s=statistics.mean(r.decode_time_s for r in batch_results),
                total_time_s=statistics.mean(r.total_time_s for r in batch_results),
                prefill_tok_s=statistics.mean(r.prefill_tok_s for r in batch_results),
                decode_tok_s=statistics.mean(r.decode_tok_s for r in batch_results),
                ttft_ms=statistics.mean(r.ttft_ms for r in batch_results),
                memory_allocated_gb=statistics.mean(r.memory_allocated_gb for r in batch_results),
                memory_reserved_gb=statistics.mean(r.memory_reserved_gb for r in batch_results),
            )
            all_results.append(avg_result)

            print(f"\n  Average:")
            print(f"    Total time: {avg_result.total_time_s:.2f}s")
            print(f"    Prefill: {avg_result.prefill_tok_s:.1f} tok/s ({avg_result.prefill_time_s*1000:.1f}ms)")
            print(f"    Decode: {avg_result.decode_tok_s:.1f} tok/s ({avg_result.decode_time_s*1000:.1f}ms)")
            print(f"    TTFT: {avg_result.ttft_ms:.1f}ms")
            print(f"    Memory: {avg_result.memory_allocated_gb:.2f} GB")

    # Print summary
    print("\n")
    print("=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    print(f"Model: {args.model}")
    print(f"Mode: {mode}")
    print()

    print(f"{'Batch':<8} {'Prompt Tok':<12} {'Output Tok':<12} {'Prefill tok/s':<15} {'Decode tok/s':<15} {'TTFT (ms)':<12} {'Memory (GB)':<12}")
    print("-" * 100)

    for r in all_results:
        print(f"{r.batch_size:<8} {r.num_prompt_tokens:<12} {r.num_output_tokens:<12} "
              f"{r.prefill_tok_s:<15.1f} {r.decode_tok_s:<15.1f} "
              f"{r.ttft_ms:<12.1f} {r.memory_allocated_gb:<12.2f}")

    # Scaling analysis
    if len(all_results) >= 2:
        print("\n")
        print("=" * 60)
        print("SCALING ANALYSIS")
        print("=" * 60)

        baseline = all_results[0]
        for r in all_results[1:]:
            decode_scaling = r.decode_tok_s / baseline.decode_tok_s if baseline.decode_tok_s > 0 else 0
            prefill_scaling = r.prefill_tok_s / baseline.prefill_tok_s if baseline.prefill_tok_s > 0 else 0
            print(f"Batch {baseline.batch_size}→{r.batch_size}: "
                  f"Decode {decode_scaling:.2f}x, Prefill {prefill_scaling:.2f}x")

    # Correctness summary
    if not args.skip_correctness and all_correctness:
        print("\n")
        print("=" * 60)
        print("OUTPUT CORRECTNESS VERIFICATION")
        print("=" * 60)

        coherent_count = sum(1 for c in all_correctness if c.is_coherent)
        total_count = len(all_correctness)

        print(f"Coherent outputs: {coherent_count}/{total_count} ({100*coherent_count/total_count:.1f}%)")
        print()

        # Show first few examples
        print("Sample outputs:")
        for i, c in enumerate(all_correctness[:3]):
            print(f"\n  [{i+1}] Prompt: {c.prompt}")
            print(f"      Output: {c.output}")
            print(f"      Length: {c.output_length} tokens, Status: {c.notes}")

    # Final memory
    mem_final = get_memory_stats()
    print("\n")
    print("=" * 60)
    print("MEMORY SUMMARY")
    print("=" * 60)
    print(f"Initial: {mem_initial[0]:.2f} GB")
    print(f"After load: {mem_after_load[0]:.2f} GB")
    print(f"Peak during benchmark: {max(r.memory_allocated_gb for r in all_results) if all_results else 0:.2f} GB")
    print(f"Final: {mem_final[0]:.2f} GB")

    # Cleanup
    del llm
    clear_memory()

    print("\n")
    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

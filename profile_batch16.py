#!/usr/bin/env python3
"""
METAL V1.5 PROFILING SCRIPT - Batch 16 Regression Analysis
============================================================

This script performs deep profiling at three levels:
1. Python-level: metal_attn sync/copy operations
2. Metal kernel: fused kernel execution time
3. Scheduler: decode vs prefill tokens

Run with:
    python profile_batch16.py

Output: Full profiling report with regression analysis.
"""

import os
os.environ["VLLM_METAL_ATTENTION"] = "1"
os.environ["VLLM_METAL_FUSED_KV"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "32"
# Enable detailed profiling
os.environ["METAL_PROFILE_DETAIL"] = "1"

import time
import torch
from collections import defaultdict
from vllm import LLM, SamplingParams


# =============================================================================
# PROFILING COUNTERS
# =============================================================================
PROFILE_DATA = {
    'mps_sync_count': 0,
    'mps_sync_ms': 0.0,
    'cpu_copy_count': 0,
    'cpu_copy_ms': 0.0,
    'fused_kernel_calls': 0,
    'fused_kernel_ms': 0.0,
    'forward_pass_count': 0,
    'forward_pass_ms': 0.0,
    'per_batch_timing': defaultdict(list),
    'decode_vs_prefill': [],
    'kernel_batch_timing': defaultdict(list),
}


def patch_profiling():
    """Patch metal_attn.py to add detailed profiling."""
    import vllm_apple.v1.attention.backends.metal_attn as metal_attn

    # Store original methods
    original_compute_attention_fused = None
    if hasattr(metal_attn.MetalAttentionImpl, '_compute_attention_fused'):
        original_compute_attention_fused = metal_attn.MetalAttentionImpl._compute_attention_fused

    original_forward = metal_attn.MetalAttentionImpl.forward

    # Patched forward to track decode/prefill
    def patched_forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                       output_scale=None, output_block_scale=None):
        num_actual_tokens = query.shape[0]

        # Track decode vs prefill
        if attn_metadata is not None:
            num_decode = attn_metadata.num_decode_tokens if hasattr(attn_metadata, 'num_decode_tokens') else 0
            num_prefill = num_actual_tokens - num_decode
            PROFILE_DATA['decode_vs_prefill'].append((num_decode, num_prefill))

            if len(PROFILE_DATA['decode_vs_prefill']) <= 10 or len(PROFILE_DATA['decode_vs_prefill']) % 48 == 0:
                print(f"[PROFILE] forward: tokens={num_actual_tokens}, decode={num_decode}, prefill={num_prefill}")

        t_start = time.perf_counter()
        result = original_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                                 output, output_scale, output_block_scale)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        PROFILE_DATA['forward_pass_count'] += 1
        PROFILE_DATA['forward_pass_ms'] += elapsed_ms

        return result

    metal_attn.MetalAttentionImpl.forward = patched_forward

    # Patch torch.mps.synchronize
    original_mps_sync = torch.mps.synchronize

    def patched_mps_sync():
        t_start = time.perf_counter()
        original_mps_sync()
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        PROFILE_DATA['mps_sync_count'] += 1
        PROFILE_DATA['mps_sync_ms'] += elapsed_ms

    torch.mps.synchronize = patched_mps_sync

    print("[PROFILE] Profiling patches applied")


def run_profiled_inference(llm, prompts, sampling_params, batch_label):
    """Run inference with profiling."""
    print(f"\n{'='*60}")
    print(f"PROFILING: {batch_label}")
    print(f"{'='*60}")

    # Reset counters
    PROFILE_DATA['mps_sync_count'] = 0
    PROFILE_DATA['mps_sync_ms'] = 0.0
    PROFILE_DATA['cpu_copy_count'] = 0
    PROFILE_DATA['cpu_copy_ms'] = 0.0
    PROFILE_DATA['fused_kernel_calls'] = 0
    PROFILE_DATA['fused_kernel_ms'] = 0.0
    PROFILE_DATA['forward_pass_count'] = 0
    PROFILE_DATA['forward_pass_ms'] = 0.0
    PROFILE_DATA['decode_vs_prefill'] = []

    # Get Metal profile
    try:
        from vllm_apple.v1.attention.backends.metal_attn import reset_metal_profile, get_metal_profile
        reset_metal_profile()
    except ImportError:
        pass

    # Run inference
    torch.mps.synchronize() if torch.backends.mps.is_available() else None
    t_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    torch.mps.synchronize() if torch.backends.mps.is_available() else None
    elapsed = time.time() - t_start

    # Calculate metrics
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec = total_output_tokens / elapsed

    # Get Metal profile
    metal_profile = {}
    try:
        metal_profile = get_metal_profile()
    except:
        pass

    # Print results
    print(f"\n[RESULTS for {batch_label}]")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Output tokens: {total_output_tokens}")
    print(f"  Throughput: {tok_per_sec:.1f} tok/s")
    print(f"\n[SYNC/COPY STATS]")
    print(f"  torch.mps.synchronize() calls: {PROFILE_DATA['mps_sync_count']}")
    print(f"  torch.mps.synchronize() total: {PROFILE_DATA['mps_sync_ms']:.1f}ms")
    print(f"  Forward pass calls: {PROFILE_DATA['forward_pass_count']}")
    print(f"  Forward pass total: {PROFILE_DATA['forward_pass_ms']:.1f}ms")

    if PROFILE_DATA['forward_pass_count'] > 0:
        avg_forward = PROFILE_DATA['forward_pass_ms'] / PROFILE_DATA['forward_pass_count']
        print(f"  Avg forward per call: {avg_forward:.2f}ms")

    print(f"\n[DECODE vs PREFILL]")
    decode_counts = [x[0] for x in PROFILE_DATA['decode_vs_prefill']]
    prefill_counts = [x[1] for x in PROFILE_DATA['decode_vs_prefill']]
    if decode_counts:
        print(f"  Decode token range: {min(decode_counts)} - {max(decode_counts)}")
        print(f"  Prefill token range: {min(prefill_counts)} - {max(prefill_counts)}")
        print(f"  Total calls with prefill > 0: {sum(1 for x in prefill_counts if x > 0)}")
        print(f"  Total calls with decode > 0: {sum(1 for x in decode_counts if x > 0)}")

    if metal_profile:
        print(f"\n[METAL PROFILE]")
        print(f"  calls: {metal_profile.get('call_count', 0)}")
        print(f"  kv_update_ms: {metal_profile.get('kv_update_ms', 0):.1f}")
        print(f"  kv_sync_ms: {metal_profile.get('kv_sync_ms', 0):.1f}")
        print(f"  kv_to_cpu_ms: {metal_profile.get('kv_to_cpu_ms', 0):.1f}")
        print(f"  metal_compute_ms: {metal_profile.get('metal_compute_ms', 0):.1f}")
        print(f"  sdpa_compute_ms: {metal_profile.get('sdpa_compute_ms', 0):.1f}")
        print(f"  total_forward_ms: {metal_profile.get('total_forward_ms', 0):.1f}")

    # Store for report
    PROFILE_DATA['per_batch_timing'][batch_label].append({
        'elapsed': elapsed,
        'tok_per_sec': tok_per_sec,
        'output_tokens': total_output_tokens,
        'mps_sync_count': PROFILE_DATA['mps_sync_count'],
        'mps_sync_ms': PROFILE_DATA['mps_sync_ms'],
        'forward_pass_count': PROFILE_DATA['forward_pass_count'],
        'forward_pass_ms': PROFILE_DATA['forward_pass_ms'],
        'metal_profile': metal_profile.copy(),
        'decode_prefill': list(PROFILE_DATA['decode_vs_prefill']),
    })

    return outputs


def main():
    print("="*70)
    print("METAL V1.5 PROFILING - Batch 16 Regression Analysis")
    print("="*70)

    # Apply profiling patches
    patch_profiling()

    # Load model
    print("\nLoading Qwen3-30B-A3B...")
    start = time.time()

    llm = LLM(
        model="Qwen/Qwen3-30B-A3B",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=512,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    print(f"Model loaded in {time.time()-start:.1f}s")

    # Test prompts
    prompts = [
        "What is 2+2?",
        "Name the capital of France.",
        "Write a haiku about coding.",
        "Explain AI in one sentence.",
        "What color is the sky?",
        "Count from 1 to 5.",
        "Say hello in Spanish.",
        "What is Python?",
        "Name a prime number.",
        "What is the sun?",
        "Define gravity briefly.",
        "What is water made of?",
        "Name a continent.",
        "What is 10 times 10?",
        "Say goodbye.",
        "What day comes after Monday?",
    ]

    params = SamplingParams(temperature=0.0, max_tokens=32)

    # Warmup
    print("\nWarmup...")
    _ = llm.generate(prompts[:1], params)

    # Profile each batch size
    batch_sizes = [1, 2, 4, 8, 16]

    for batch_size in batch_sizes:
        batch_prompts = prompts[:batch_size]
        run_profiled_inference(llm, batch_prompts, params, f"batch={batch_size}")

    # Generate report
    print("\n" + "="*70)
    print("PROFILING REPORT")
    print("="*70)

    print("\n1. THROUGHPUT COMPARISON:")
    print("-" * 50)
    print(f"{'Batch':<10} {'tok/s':<10} {'time(s)':<10} {'forward_ms':<12} {'sync_count':<12}")
    print("-" * 50)

    baseline_tps = None
    for batch_size in batch_sizes:
        key = f"batch={batch_size}"
        data = PROFILE_DATA['per_batch_timing'][key][0]
        tps = data['tok_per_sec']
        if baseline_tps is None:
            baseline_tps = tps
        scaling = tps / baseline_tps

        avg_forward = data['forward_pass_ms'] / max(data['forward_pass_count'], 1)

        print(f"{batch_size:<10} {tps:<10.1f} {data['elapsed']:<10.2f} {avg_forward:<12.1f} {data['mps_sync_count']:<12} ({scaling:.2f}x)")

    print("\n2. DECODE vs PREFILL ANALYSIS:")
    print("-" * 50)
    for batch_size in batch_sizes:
        key = f"batch={batch_size}"
        data = PROFILE_DATA['per_batch_timing'][key][0]
        decode_prefill = data['decode_prefill']

        if decode_prefill:
            prefill_calls = sum(1 for d, p in decode_prefill if p > 0)
            decode_calls = sum(1 for d, p in decode_prefill if d > 0)
            max_prefill = max(p for d, p in decode_prefill)
            max_decode = max(d for d, p in decode_prefill)

            print(f"Batch {batch_size}:")
            print(f"  Total layer calls: {len(decode_prefill)}")
            print(f"  Calls with prefill: {prefill_calls}")
            print(f"  Calls with decode: {decode_calls}")
            print(f"  Max prefill tokens: {max_prefill}")
            print(f"  Max decode tokens: {max_decode}")

    print("\n3. METAL PROFILE COMPARISON:")
    print("-" * 50)
    for batch_size in batch_sizes:
        key = f"batch={batch_size}"
        data = PROFILE_DATA['per_batch_timing'][key][0]
        mp = data['metal_profile']

        if mp.get('call_count', 0) > 0:
            print(f"Batch {batch_size}:")
            print(f"  kv_update: {mp.get('kv_update_ms', 0):.1f}ms ({100*mp.get('kv_update_ms', 0)/max(mp.get('total_forward_ms', 1), 1):.0f}%)")
            print(f"  kv_sync: {mp.get('kv_sync_ms', 0):.1f}ms")
            print(f"  kv_to_cpu: {mp.get('kv_to_cpu_ms', 0):.1f}ms")
            print(f"  metal_kernel: {mp.get('metal_compute_ms', 0):.1f}ms ({100*mp.get('metal_compute_ms', 0)/max(mp.get('total_forward_ms', 1), 1):.0f}%)")
            print(f"  total: {mp.get('total_forward_ms', 0):.1f}ms")

    print("\n4. REGRESSION ANALYSIS:")
    print("-" * 50)

    # Calculate expected vs actual for batch=16
    batch8_data = PROFILE_DATA['per_batch_timing']['batch=8'][0]
    batch16_data = PROFILE_DATA['per_batch_timing']['batch=16'][0]

    batch8_forward = batch8_data['forward_pass_ms'] / max(batch8_data['forward_pass_count'], 1)
    batch16_forward = batch16_data['forward_pass_ms'] / max(batch16_data['forward_pass_count'], 1)

    expected_batch16_forward = batch8_forward * 2  # Linear scaling expectation
    actual_batch16_forward = batch16_forward
    regression_factor = actual_batch16_forward / expected_batch16_forward

    print(f"Batch 8 avg forward: {batch8_forward:.1f}ms")
    print(f"Batch 16 avg forward: {batch16_forward:.1f}ms")
    print(f"Expected (2x linear): {expected_batch16_forward:.1f}ms")
    print(f"Regression factor: {regression_factor:.2f}x slower than expected")

    # Identify bottleneck
    batch16_mp = batch16_data['metal_profile']
    batch8_mp = batch8_data['metal_profile']

    print("\n5. BOTTLENECK IDENTIFICATION:")
    print("-" * 50)

    if batch16_mp.get('total_forward_ms', 0) > 0 and batch8_mp.get('total_forward_ms', 0) > 0:
        kv_update_ratio = batch16_mp.get('kv_update_ms', 0) / max(batch8_mp.get('kv_update_ms', 1), 1)
        kv_sync_ratio = batch16_mp.get('kv_sync_ms', 0) / max(batch8_mp.get('kv_sync_ms', 1), 1)
        metal_ratio = batch16_mp.get('metal_compute_ms', 0) / max(batch8_mp.get('metal_compute_ms', 1), 1)

        print(f"KV Update slowdown: {kv_update_ratio:.2f}x")
        print(f"KV Sync slowdown: {kv_sync_ratio:.2f}x")
        print(f"Metal Kernel slowdown: {metal_ratio:.2f}x")

        # Identify biggest contributor
        contributors = [
            ('KV Update', kv_update_ratio, batch16_mp.get('kv_update_ms', 0) - batch8_mp.get('kv_update_ms', 0)),
            ('KV Sync', kv_sync_ratio, batch16_mp.get('kv_sync_ms', 0) - batch8_mp.get('kv_sync_ms', 0)),
            ('Metal Kernel', metal_ratio, batch16_mp.get('metal_compute_ms', 0) - batch8_mp.get('metal_compute_ms', 0)),
        ]

        biggest = max(contributors, key=lambda x: x[2])
        print(f"\nBiggest contributor to slowdown: {biggest[0]}")
        print(f"  Ratio: {biggest[1]:.2f}x")
        print(f"  Added time: {biggest[2]:.1f}ms")

    print("\n" + "="*70)
    print("END OF PROFILING REPORT")
    print("="*70)


if __name__ == "__main__":
    main()

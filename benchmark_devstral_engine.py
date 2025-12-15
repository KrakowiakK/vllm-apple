#!/usr/bin/env python3
"""Benchmark Devstral-24B with Metal Engine.

Measures prefill and decode throughput for batch sizes 1, 2, 4, 8, 16.

Phase 0 Profiling Mode:
    VLLM_PROFILE_BATCH1=1 python benchmark_devstral_engine.py --batch-sizes 1 --profile

This produces a detailed breakdown of CPU encode time vs GPU execution time
to validate assumptions in the BATCH=1 optimization plan.
"""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '4'
os.environ['VLLM_METAL_SCRATCH_POOL_MB'] = '4096'  # 4GB scratch pool
os.environ['VLLM_METAL_MAX_BATCH_SIZE'] = '2048'  # Allow larger batches

import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_engine(
    model_name: str = 'mistralai/Devstral-Small-2505',
    batch_sizes: list = [1, 2, 4, 8, 16],
    prompt_len: int = 128,
    decode_steps: int = 32,
    warmup_steps: int = 5,
):
    """Benchmark EngineRunner with Devstral model."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.runner import EngineRunner
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import (
        ModelDescriptor,
        StepDescriptor,
        EngineInputs,
    )

    print("=" * 80)
    print(f"Devstral Metal Engine Benchmark")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Prompt length: {prompt_len}")
    print(f"Decode steps: {decode_steps}")
    print()

    # Load model
    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    config = model.config
    print(f"Model config: {config.hidden_size} hidden, {config.num_attention_heads} heads, {config.num_hidden_layers} layers")

    # Setup engine
    ctx = MetalEngineContext()

    model_desc = ModelDescriptor(
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_size=config.head_dim,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        rope_theta=config.rope_theta,
        max_position_embeddings=config.max_position_embeddings,
        architecture="mistral",
        rms_norm_eps=config.rms_norm_eps,
    )

    # Calculate KV cache blocks needed
    block_size = 16
    max_batch = max(batch_sizes)
    max_seq_len = prompt_len + decode_steps + 64
    blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_batch * blocks_per_seq + 32

    kv_desc = model_desc.get_kv_cache_descriptor(
        num_blocks=num_blocks,
        block_size=block_size,
    )
    kv_cache = EngineKVCache(ctx, kv_desc)

    # Load weights
    print("Loading weights via EngineWeightLoader...")
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')

    # Create EngineRunner
    print("Creating EngineRunner...")
    runner = EngineRunner(
        context=ctx,
        model_desc=model_desc,
        weights=weights,
        kv_cache=kv_cache,
    )

    # Create dummy prompt tokens
    prompt_tokens = torch.randint(1, config.vocab_size, (prompt_len,), dtype=torch.int64)

    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}")

        # Create fresh context for each batch size to avoid scratch pool exhaustion
        ctx = MetalEngineContext()

        # Reload weights (they reference the context)
        loader = EngineWeightLoader(ctx, model_config=None)
        weights = loader.load_from_hf_model(model, arch='mistral')

        # Create fresh KV cache
        kv_cache = EngineKVCache(ctx, kv_desc)
        runner = EngineRunner(
            context=ctx,
            model_desc=model_desc,
            weights=weights,
            kv_cache=kv_cache,
        )

        # ===== PREFILL =====
        total_prefill_tokens = batch_size * prompt_len

        # Create batched prefill inputs
        token_ids = prompt_tokens.repeat(batch_size)
        positions = torch.arange(prompt_len, dtype=torch.int64).repeat(batch_size)
        for seq_idx in range(batch_size):
            positions[seq_idx * prompt_len:(seq_idx + 1) * prompt_len] = torch.arange(prompt_len)

        # Slot mapping: each sequence gets its own blocks
        slot_mapping = torch.zeros(total_prefill_tokens, dtype=torch.int64)
        for seq_idx in range(batch_size):
            seq_offset = seq_idx * blocks_per_seq * block_size
            for tok_idx in range(prompt_len):
                slot_mapping[seq_idx * prompt_len + tok_idx] = seq_offset + tok_idx

        # Block table
        block_table = torch.zeros((batch_size, blocks_per_seq + 4), dtype=torch.int64)
        for seq_idx in range(batch_size):
            for blk_idx in range(blocks_per_seq):
                block_table[seq_idx, blk_idx] = seq_idx * blocks_per_seq + blk_idx

        seq_lens = torch.full((batch_size,), prompt_len, dtype=torch.int64)
        query_start_locs = torch.zeros(batch_size + 1, dtype=torch.int64)
        for i in range(batch_size + 1):
            query_start_locs[i] = i * prompt_len

        prefill_inputs = EngineInputs(
            token_ids=token_ids.cpu(),
            positions=positions.cpu(),
            block_table=block_table.cpu(),
            slot_mapping=slot_mapping.cpu(),
            seq_lens=seq_lens.cpu(),
            query_start_locs=query_start_locs.cpu(),
            max_decode_seq_len=0,
        )

        prefill_step = StepDescriptor(
            step_id=0,
            step_kind="prefill",
            num_scheduled_tokens=total_prefill_tokens,
            num_seqs_active=batch_size,
            max_num_blocks_per_seq=blocks_per_seq + 4,
            is_first_step=True,
            cache_enabled=True,
        )

        # No warmup - fresh context already

        # Benchmark prefill
        prefill_start = time.perf_counter()
        outputs = runner.execute_step(prefill_step, prefill_inputs)
        prefill_end = time.perf_counter()

        prefill_time_ms = (prefill_end - prefill_start) * 1000
        prefill_tok_s = total_prefill_tokens / (prefill_end - prefill_start)

        print(f"PREFILL: {total_prefill_tokens} tokens in {prefill_time_ms:.1f} ms = {prefill_tok_s:.1f} tok/s")

        # ===== DECODE =====
        # Create fresh context for decode (avoids scratch pool exhaustion)
        ctx = MetalEngineContext()
        loader = EngineWeightLoader(ctx, model_config=None)
        weights = loader.load_from_hf_model(model, arch='mistral')
        kv_cache = EngineKVCache(ctx, kv_desc)
        runner = EngineRunner(
            context=ctx,
            model_desc=model_desc,
            weights=weights,
            kv_cache=kv_cache,
        )

        # Run prefill first to populate cache
        _ = runner.execute_step(prefill_step, prefill_inputs)

        # Prepare decode inputs
        decode_times = []

        for step in range(warmup_steps + decode_steps):
            cur_pos = prompt_len + (step if step >= warmup_steps else 0)

            # One new token per sequence
            decode_token_ids = torch.randint(1, config.vocab_size, (batch_size,), dtype=torch.int64)
            decode_positions = torch.full((batch_size,), cur_pos, dtype=torch.int64)

            # Slot mapping for new tokens
            decode_slot_mapping = torch.zeros(batch_size, dtype=torch.int64)
            for seq_idx in range(batch_size):
                seq_offset = seq_idx * blocks_per_seq * block_size
                decode_slot_mapping[seq_idx] = seq_offset + cur_pos

            decode_seq_lens = torch.full((batch_size,), cur_pos + 1, dtype=torch.int64)
            decode_query_start_locs = torch.arange(batch_size + 1, dtype=torch.int64)

            decode_inputs = EngineInputs(
                token_ids=decode_token_ids.cpu(),
                positions=decode_positions.cpu(),
                block_table=block_table.cpu(),
                slot_mapping=decode_slot_mapping.cpu(),
                seq_lens=decode_seq_lens.cpu(),
                query_start_locs=decode_query_start_locs.cpu(),
                max_decode_seq_len=cur_pos + 1,
            )

            decode_step = StepDescriptor(
                step_id=step + 1,
                step_kind="decode",
                num_scheduled_tokens=batch_size,
                num_seqs_active=batch_size,
                max_num_blocks_per_seq=blocks_per_seq + 4,
                is_first_step=False,
                cache_enabled=True,
            )

            step_start = time.perf_counter()
            _ = runner.execute_step(decode_step, decode_inputs)
            step_end = time.perf_counter()

            if step >= warmup_steps:
                decode_times.append(step_end - step_start)

        avg_decode_time_ms = np.mean(decode_times) * 1000
        decode_tok_s = batch_size / np.mean(decode_times)

        print(f"DECODE:  {batch_size} tokens/step, avg {avg_decode_time_ms:.2f} ms = {decode_tok_s:.1f} tok/s")

        results.append({
            'batch_size': batch_size,
            'prefill_tok_s': prefill_tok_s,
            'prefill_ms': prefill_time_ms,
            'decode_tok_s': decode_tok_s,
            'decode_ms': avg_decode_time_ms,
        })

    # Summary
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Batch':<8} {'Prefill tok/s':<15} {'Prefill ms':<12} {'Decode tok/s':<15} {'Decode ms':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r['batch_size']:<8} {r['prefill_tok_s']:<15.1f} {r['prefill_ms']:<12.1f} {r['decode_tok_s']:<15.1f} {r['decode_ms']:<12.2f}")

    # Scaling analysis
    if len(results) >= 2:
        print("\n")
        print("SCALING ANALYSIS:")
        b1 = results[0]
        for r in results[1:]:
            prefill_scale = r['prefill_tok_s'] / b1['prefill_tok_s']
            decode_scale = r['decode_tok_s'] / b1['decode_tok_s']
            print(f"  Batch {r['batch_size']:2d} vs 1: Prefill {prefill_scale:.2f}x, Decode {decode_scale:.2f}x")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)

    return results


def profile_batch1_decode(
    model_name: str = 'mistralai/Devstral-Small-2505',
    prompt_len: int = 64,
    decode_steps: int = 16,
    warmup_steps: int = 3,
):
    """Phase 0 Profiling: Detailed breakdown for batch=1 decode.

    Produces output in the format required by METAL_BATCH1_PLAN.md Phase 0.

    IMPORTANT: CPU timestamps measure SUBMISSION overhead, NOT GPU execution time.
    GPU execution is asynchronous - use the GPU floor measurement for true compute time.
    """
    # Enable profiling
    os.environ["VLLM_PROFILE_BATCH1"] = "1"

    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.runner import EngineRunner
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import (
        ModelDescriptor,
        StepDescriptor,
        EngineInputs,
    )

    print("=" * 80)
    print("BATCH=1 DECODE PROFILING (Phase 0)")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Prompt length: {prompt_len}")
    print(f"Decode steps: {decode_steps} (after {warmup_steps} warmup)")
    print()

    # Load model
    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    config = model.config
    num_layers = config.num_hidden_layers
    print(f"Model config: {config.hidden_size} hidden, {config.num_attention_heads} heads, {num_layers} layers")

    # Setup engine
    ctx = MetalEngineContext()

    model_desc = ModelDescriptor(
        num_layers=num_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_size=config.head_dim,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        rope_theta=config.rope_theta,
        max_position_embeddings=config.max_position_embeddings,
        architecture="mistral",
        rms_norm_eps=config.rms_norm_eps,
    )

    # Calculate KV cache blocks
    block_size = 16
    max_seq_len = prompt_len + decode_steps + 64
    blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = blocks_per_seq + 32

    kv_desc = model_desc.get_kv_cache_descriptor(
        num_blocks=num_blocks,
        block_size=block_size,
    )
    kv_cache = EngineKVCache(ctx, kv_desc)

    # Load weights
    print("Loading weights...")
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')

    # Create runner
    runner = EngineRunner(
        context=ctx,
        model_desc=model_desc,
        weights=weights,
        kv_cache=kv_cache,
    )

    # Create dummy prompt
    prompt_tokens = torch.randint(1, config.vocab_size, (prompt_len,), dtype=torch.int64)

    # Prefill
    slot_mapping = torch.arange(prompt_len, dtype=torch.int64)
    block_table = torch.arange(blocks_per_seq, dtype=torch.int64).unsqueeze(0)
    seq_lens = torch.tensor([prompt_len], dtype=torch.int64)
    query_start_locs = torch.tensor([0, prompt_len], dtype=torch.int64)

    prefill_inputs = EngineInputs(
        token_ids=prompt_tokens.cpu(),
        positions=torch.arange(prompt_len, dtype=torch.int64).cpu(),
        block_table=block_table.cpu(),
        slot_mapping=slot_mapping.cpu(),
        seq_lens=seq_lens.cpu(),
        query_start_locs=query_start_locs.cpu(),
        max_decode_seq_len=0,
    )

    prefill_step = StepDescriptor(
        step_id=0,
        step_kind="prefill",
        num_scheduled_tokens=prompt_len,
        num_seqs_active=1,
        max_num_blocks_per_seq=blocks_per_seq,
        is_first_step=True,
        cache_enabled=True,
    )

    print("Running prefill...")
    _ = runner.execute_step(prefill_step, prefill_inputs)

    # Decode profiling
    print(f"Running {warmup_steps} warmup + {decode_steps} profiled decode steps...")

    step_times = []
    profiling_stats = []

    for step in range(warmup_steps + decode_steps):
        cur_pos = prompt_len + step

        decode_token_ids = torch.randint(1, config.vocab_size, (1,), dtype=torch.int64)
        decode_positions = torch.tensor([cur_pos], dtype=torch.int64)
        decode_slot_mapping = torch.tensor([cur_pos], dtype=torch.int64)
        decode_seq_lens = torch.tensor([cur_pos + 1], dtype=torch.int64)
        decode_query_start_locs = torch.tensor([0, 1], dtype=torch.int64)

        decode_inputs = EngineInputs(
            token_ids=decode_token_ids.cpu(),
            positions=decode_positions.cpu(),
            block_table=block_table.cpu(),
            slot_mapping=decode_slot_mapping.cpu(),
            seq_lens=decode_seq_lens.cpu(),
            query_start_locs=decode_query_start_locs.cpu(),
            max_decode_seq_len=cur_pos + 1,
        )

        decode_step = StepDescriptor(
            step_id=step + 1,
            step_kind="decode",
            num_scheduled_tokens=1,
            num_seqs_active=1,
            max_num_blocks_per_seq=blocks_per_seq,
            is_first_step=False,
            cache_enabled=True,
        )

        step_start = time.perf_counter()
        outputs, stats = runner.execute_step(decode_step, decode_inputs, return_step_ctx=True)
        step_end = time.perf_counter()

        if step >= warmup_steps:
            step_times.append(step_end - step_start)
            if stats is not None:
                profiling_stats.append(stats)

    # Calculate averages
    avg_step_time_ms = np.mean(step_times) * 1000
    avg_step_time_std = np.std(step_times) * 1000

    # Aggregate profiling stats
    if profiling_stats:
        avg_stats = {
            "mps_transitions": np.mean([s["mps_transitions"] for s in profiling_stats]),
            "encoder_reopens": np.mean([s["encoder_reopens"] for s in profiling_stats]),
            "barriers": np.mean([s["barriers"] for s in profiling_stats]),
            "barrier_reopens": np.mean([s["barrier_reopens"] for s in profiling_stats]),
            "dispatches": np.mean([s["dispatches"] for s in profiling_stats]),
            "transition_time_ms": np.mean([s["transition_time_ms"] for s in profiling_stats]),
            "barrier_time_ms": np.mean([s["barrier_time_ms"] for s in profiling_stats]),
        }
    else:
        avg_stats = None

    # GPU floor measurement: run with explicit sync per step
    print("\nMeasuring GPU execution floor (with waitUntilCompleted sync)...")
    gpu_floor_times = []

    # Fresh context for GPU floor measurement
    ctx = MetalEngineContext()
    loader = EngineWeightLoader(ctx, model_config=None)
    weights = loader.load_from_hf_model(model, arch='mistral')
    kv_cache = EngineKVCache(ctx, kv_desc)
    runner = EngineRunner(
        context=ctx,
        model_desc=model_desc,
        weights=weights,
        kv_cache=kv_cache,
    )

    # Run prefill again
    _ = runner.execute_step(prefill_step, prefill_inputs)

    # Measure GPU floor (fewer iterations as this is slower)
    for step in range(warmup_steps + 8):
        cur_pos = prompt_len + step

        decode_token_ids = torch.randint(1, config.vocab_size, (1,), dtype=torch.int64)
        decode_positions = torch.tensor([cur_pos], dtype=torch.int64)
        decode_slot_mapping = torch.tensor([cur_pos], dtype=torch.int64)
        decode_seq_lens = torch.tensor([cur_pos + 1], dtype=torch.int64)
        decode_query_start_locs = torch.tensor([0, 1], dtype=torch.int64)

        decode_inputs = EngineInputs(
            token_ids=decode_token_ids.cpu(),
            positions=decode_positions.cpu(),
            block_table=block_table.cpu(),
            slot_mapping=decode_slot_mapping.cpu(),
            seq_lens=decode_seq_lens.cpu(),
            query_start_locs=decode_query_start_locs.cpu(),
            max_decode_seq_len=cur_pos + 1,
        )

        decode_step = StepDescriptor(
            step_id=step + 1,
            step_kind="decode",
            num_scheduled_tokens=1,
            num_seqs_active=1,
            max_num_blocks_per_seq=blocks_per_seq,
            is_first_step=False,
            cache_enabled=True,
        )

        # GPU floor: include full sync
        step_start = time.perf_counter()
        _ = runner.execute_step(decode_step, decode_inputs)
        step_end = time.perf_counter()

        if step >= warmup_steps:
            gpu_floor_times.append(step_end - step_start)

    gpu_floor_ms = np.mean(gpu_floor_times) * 1000

    # Print results
    print("\n")
    print("=" * 80)
    print(f"BATCH=1 DECODE PROFILING (Devstral, {num_layers} layers)")
    print(f"Iterations: {decode_steps} decode steps (after {warmup_steps} warmup)")
    print("=" * 80)
    print()
    print(f"TOTAL DECODE STEP TIME: {avg_step_time_ms:.2f} ms (±{avg_step_time_std:.2f} ms)")
    print()

    if avg_stats:
        print("COUNTERS (per step):")
        print(f"  MPS transitions:     {avg_stats['mps_transitions']:.0f} total ({avg_stats['mps_transitions']/num_layers:.1f} per layer)")
        print(f"  Barriers:            {avg_stats['barriers']:.0f} total ({avg_stats['barriers']/num_layers:.1f} per layer)")
        print(f"  Barrier reopens:     {avg_stats['barrier_reopens']:.0f} total ({avg_stats['barrier_reopens']/num_layers:.1f} per layer)")
        print(f"  Dispatches:          {avg_stats['dispatches']:.0f} total ({avg_stats['dispatches']/num_layers:.1f} per layer)")
        print()
        print("CPU SUBMISSION OVERHEAD (measured with perf_counter):")
        print(f"  Transition time:     {avg_stats['transition_time_ms']:.2f} ms")
        print(f"  Barrier time:        {avg_stats['barrier_time_ms']:.2f} ms")
        print()

    print(f"GPU EXECUTION FLOOR:     {gpu_floor_ms:.2f} ms (from waitUntilCompleted timing)")
    print("  Note: This includes CPU encode + GPU execute + sync overhead")
    print()

    # Calculate reducible overhead estimate
    # This is approximate: total_time - gpu_floor gives us an idea of async overlap benefit
    overhead_estimate = avg_step_time_ms - gpu_floor_ms
    print("=" * 80)
    print("SCENARIO DETERMINATION")
    print("=" * 80)
    print(f"Total step time:         {avg_step_time_ms:.2f} ms")
    print(f"GPU floor (with sync):   {gpu_floor_ms:.2f} ms")
    print(f"Async overlap benefit:   {-overhead_estimate:.2f} ms (negative = GPU takes longer than CPU encode)")
    print()

    # Determine scenario
    if gpu_floor_ms <= 55:
        if avg_step_time_ms - 50 >= 30:
            scenario = "A"
            rationale = "GPU floor ≤55ms, large reducible overhead"
        else:
            scenario = "A/B"
            rationale = "GPU floor ≤55ms, moderate overhead"
    elif gpu_floor_ms <= 70:
        scenario = "B"
        rationale = "GPU floor 55-70ms, limited optimization potential"
    else:
        scenario = "C"
        rationale = "GPU floor ≥70ms, compute-bound, 20 tok/s NOT achievable"

    print(f"Selected scenario: {scenario}")
    print(f"Rationale: {rationale}")
    print()

    if scenario == "C":
        print("RECOMMENDATION: STOP optimization work.")
        print("Batch=1 decode is compute-bound. 20 tok/s requires quantization or")
        print("speculative decoding (out of scope for current plan).")
    elif scenario == "B":
        print("RECOMMENDATION: Proceed with Phase 1 (Scheduling) and Phase 2 (Barriers) only.")
        print("Phases 3-4 have diminishing returns.")
    else:
        print("RECOMMENDATION: Proceed with Phases 1-4 as planned.")

    print()
    print("=" * 80)
    print("Copy the above output to METAL_BATCH1_PLAN.md Phase 0 Results section")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--profile", action="store_true", help="Run Phase 0 profiling for batch=1")
    args = parser.parse_args()

    if args.profile:
        profile_batch1_decode(
            prompt_len=args.prompt_len,
            decode_steps=args.decode_steps,
        )
    else:
        benchmark_engine(
            batch_sizes=args.batch_sizes,
            prompt_len=args.prompt_len,
            decode_steps=args.decode_steps,
        )

#!/usr/bin/env python3
"""Benchmark Devstral-24B with Metal Engine.

Measures prefill and decode throughput for batch sizes 1, 2, 4, 8, 16.
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    args = parser.parse_args()

    benchmark_engine(
        batch_sizes=args.batch_sizes,
        prompt_len=args.prompt_len,
        decode_steps=args.decode_steps,
    )

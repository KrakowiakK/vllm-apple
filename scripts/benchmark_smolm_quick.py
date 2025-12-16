#!/usr/bin/env python3
"""Quick benchmark with SmolLM-135M for testing batch scaling."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '1'
os.environ['VLLM_METAL_SCRATCH_POOL_MB'] = '1024'
os.environ['VLLM_METAL_MAX_BATCH_SIZE'] = '1024'  # Allow larger batches

import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.runner import EngineRunner
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import (
        ModelDescriptor,
        StepDescriptor,
        EngineInputs,
    )

    model_name = 'HuggingFaceTB/SmolLM-135M'
    batch_sizes = [1, 2, 4, 8, 16]
    prompt_len = 64
    decode_steps = 16

    print("=" * 70)
    print("SmolLM-135M Quick Benchmark")
    print("=" * 70)

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    config = model.config

    model_desc = ModelDescriptor(
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_size=config.head_dim,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        rope_theta=getattr(config, 'rope_theta', 10000.0),
        max_position_embeddings=getattr(config, 'max_position_embeddings', 2048),
        architecture="llama",
        rms_norm_eps=config.rms_norm_eps,
    )

    block_size = 16
    max_batch = max(batch_sizes)
    max_seq_len = prompt_len + decode_steps + 32
    blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_batch * blocks_per_seq + 16

    kv_desc = model_desc.get_kv_cache_descriptor(num_blocks=num_blocks, block_size=block_size)

    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*50}")

        # Fresh context for each batch
        ctx = MetalEngineContext()
        loader = EngineWeightLoader(ctx, model_config=None)
        weights = loader.load_from_hf_model(model, arch='llama')
        kv_cache = EngineKVCache(ctx, kv_desc)
        runner = EngineRunner(context=ctx, model_desc=model_desc, weights=weights, kv_cache=kv_cache)

        total_tokens = batch_size * prompt_len
        prompt_tokens = torch.randint(1, config.vocab_size, (prompt_len,), dtype=torch.int64)

        # Prefill inputs
        token_ids = prompt_tokens.repeat(batch_size)
        positions = torch.zeros(total_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(total_tokens, dtype=torch.int64)
        for seq_idx in range(batch_size):
            for tok_idx in range(prompt_len):
                global_idx = seq_idx * prompt_len + tok_idx
                positions[global_idx] = tok_idx
                slot_mapping[global_idx] = seq_idx * blocks_per_seq * block_size + tok_idx

        block_table = torch.zeros((batch_size, blocks_per_seq + 2), dtype=torch.int64)
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
            num_scheduled_tokens=total_tokens,
            num_seqs_active=batch_size,
            max_num_blocks_per_seq=blocks_per_seq + 2,
            is_first_step=True,
            cache_enabled=True,
        )

        # Warmup
        _ = runner.execute_step(prefill_step, prefill_inputs)

        # Benchmark prefill
        ctx2 = MetalEngineContext()
        loader2 = EngineWeightLoader(ctx2, model_config=None)
        weights2 = loader2.load_from_hf_model(model, arch='llama')
        kv_cache2 = EngineKVCache(ctx2, kv_desc)
        runner2 = EngineRunner(context=ctx2, model_desc=model_desc, weights=weights2, kv_cache=kv_cache2)

        prefill_start = time.perf_counter()
        _ = runner2.execute_step(prefill_step, prefill_inputs)
        prefill_end = time.perf_counter()

        prefill_ms = (prefill_end - prefill_start) * 1000
        prefill_tok_s = total_tokens / (prefill_end - prefill_start)

        print(f"PREFILL: {total_tokens} tok in {prefill_ms:.1f} ms = {prefill_tok_s:.0f} tok/s")

        # Decode benchmark
        ctx3 = MetalEngineContext()
        loader3 = EngineWeightLoader(ctx3, model_config=None)
        weights3 = loader3.load_from_hf_model(model, arch='llama')
        kv_cache3 = EngineKVCache(ctx3, kv_desc)
        runner3 = EngineRunner(context=ctx3, model_desc=model_desc, weights=weights3, kv_cache=kv_cache3)

        # Prefill first
        _ = runner3.execute_step(prefill_step, prefill_inputs)

        # Decode steps
        decode_times = []
        for step in range(decode_steps):
            cur_pos = prompt_len + step
            decode_token_ids = torch.randint(1, config.vocab_size, (batch_size,), dtype=torch.int64)
            decode_positions = torch.full((batch_size,), cur_pos, dtype=torch.int64)
            decode_slot_mapping = torch.zeros(batch_size, dtype=torch.int64)
            for seq_idx in range(batch_size):
                decode_slot_mapping[seq_idx] = seq_idx * blocks_per_seq * block_size + cur_pos

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
                max_num_blocks_per_seq=blocks_per_seq + 2,
                is_first_step=False,
                cache_enabled=True,
            )

            step_start = time.perf_counter()
            _ = runner3.execute_step(decode_step, decode_inputs)
            step_end = time.perf_counter()
            decode_times.append(step_end - step_start)

        avg_decode_ms = np.mean(decode_times) * 1000
        decode_tok_s = batch_size / np.mean(decode_times)

        print(f"DECODE:  {batch_size} tok/step, avg {avg_decode_ms:.1f} ms = {decode_tok_s:.0f} tok/s")

        results.append({
            'batch': batch_size,
            'prefill_tok_s': prefill_tok_s,
            'prefill_ms': prefill_ms,
            'decode_tok_s': decode_tok_s,
            'decode_ms': avg_decode_ms,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Batch':<8} {'Prefill tok/s':<15} {'Prefill ms':<12} {'Decode tok/s':<15} {'Decode ms':<12}")
    print("-" * 62)
    for r in results:
        print(f"{r['batch']:<8} {r['prefill_tok_s']:<15.0f} {r['prefill_ms']:<12.1f} {r['decode_tok_s']:<15.0f} {r['decode_ms']:<12.1f}")

    # Scaling
    if len(results) >= 2:
        print("\nSCALING (vs batch 1):")
        b1 = results[0]
        for r in results[1:]:
            print(f"  Batch {r['batch']:2d}: Prefill {r['prefill_tok_s']/b1['prefill_tok_s']:.2f}x, Decode {r['decode_tok_s']/b1['decode_tok_s']:.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Simple coherence test - same prompt replicated across batch."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '4'
os.environ['VLLM_METAL_SCRATCH_POOL_MB'] = '8192'
os.environ['VLLM_METAL_MAX_BATCH_SIZE'] = '16384'
os.environ['VLLM_APPLE_USE_ENGINE'] = '1'
os.environ['VLLM_APPLE_ENGINE_PREFILL'] = '1'
os.environ['VLLM_METAL_STRICT_NO_MPS'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_coherence_simple(batch_sizes=[1, 4, 8, 16], max_tokens=40):
    """Test coherence with same prompt replicated across batch."""
    from vllm_apple.engine.context import MetalEngineContext
    from vllm_apple.engine.runner import EngineRunner
    from vllm_apple.engine.weight_loader import EngineWeightLoader
    from vllm_apple.engine.kv_cache import EngineKVCache
    from vllm_apple.engine.descriptors import (
        ModelDescriptor,
        StepDescriptor,
        EngineInputs,
    )

    model_name = 'mistralai/Devstral-Small-2505'

    print("=" * 80)
    print("SIMPLE COHERENCE TEST")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Same prompt replicated across batch")
    print(f"Batch sizes: {batch_sizes}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    config = model.config

    # Test prompts
    test_prompts = [
        ("fibonacci", "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n"),
        ("quicksort", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"),
        ("binary_search", "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n"),
        ("palindrome", "def is_palindrome(s):\n    \"\"\"Check if string is palindrome.\"\"\"\n"),
    ]

    results = []

    for prompt_name, prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Testing prompt: {prompt_name}")
        print(f"{'='*60}")

        prompt_results = []

        for batch_size in batch_sizes:
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

            block_size = 16
            max_seq_len = 512 + max_tokens
            blocks_per_seq = (max_seq_len + block_size - 1) // block_size
            num_blocks = batch_size * blocks_per_seq + 32

            kv_desc = model_desc.get_kv_cache_descriptor(num_blocks=num_blocks, block_size=block_size)
            kv_cache = EngineKVCache(ctx, kv_desc)
            loader = EngineWeightLoader(ctx, model_config=None)
            weights = loader.load_from_hf_model(model, arch='mistral')
            runner = EngineRunner(context=ctx, model_desc=model_desc, weights=weights, kv_cache=kv_cache)

            # Tokenize - same prompt for all
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(tokens)
            total_prefill_tokens = batch_size * prompt_len

            # Replicate for batch
            token_ids = torch.tensor(tokens * batch_size, dtype=torch.int64)
            positions = torch.arange(prompt_len, dtype=torch.int64).repeat(batch_size)
            for seq_idx in range(batch_size):
                positions[seq_idx * prompt_len:(seq_idx + 1) * prompt_len] = torch.arange(prompt_len)

            slot_mapping = torch.zeros(total_prefill_tokens, dtype=torch.int64)
            for seq_idx in range(batch_size):
                seq_offset = seq_idx * blocks_per_seq * block_size
                for tok_idx in range(prompt_len):
                    slot_mapping[seq_idx * prompt_len + tok_idx] = seq_offset + tok_idx

            block_table = torch.zeros((batch_size, blocks_per_seq + 4), dtype=torch.int64)
            for seq_idx in range(batch_size):
                for blk_idx in range(blocks_per_seq):
                    block_table[seq_idx, blk_idx] = seq_idx * blocks_per_seq + blk_idx

            seq_lens = torch.full((batch_size,), prompt_len, dtype=torch.int64)
            query_start_locs = torch.zeros(batch_size + 1, dtype=torch.int64)
            for i in range(batch_size + 1):
                query_start_locs[i] = i * prompt_len

            prefill_inputs = EngineInputs(
                token_ids=token_ids, positions=positions, block_table=block_table,
                slot_mapping=slot_mapping, seq_lens=seq_lens,
                query_start_locs=query_start_locs, max_decode_seq_len=0)

            prefill_step = StepDescriptor(
                step_id=0, step_kind="prefill", num_scheduled_tokens=total_prefill_tokens,
                num_seqs_active=batch_size, max_num_blocks_per_seq=blocks_per_seq + 4,
                is_first_step=True, cache_enabled=True)

            outputs = runner.execute_step(prefill_step, prefill_inputs)

            # Get last token logits for each sequence
            generated = []
            for seq_idx in range(batch_size):
                seq_end = (seq_idx + 1) * prompt_len - 1
                token = torch.argmax(outputs.logits[seq_end]).item()
                generated.append([token])

            # Decode
            for step in range(max_tokens - 1):
                cur_pos = prompt_len + step

                decode_token_ids = torch.tensor([g[-1] for g in generated], dtype=torch.int64)
                decode_positions = torch.full((batch_size,), cur_pos, dtype=torch.int64)

                decode_slot_mapping = torch.zeros(batch_size, dtype=torch.int64)
                for seq_idx in range(batch_size):
                    seq_offset = seq_idx * blocks_per_seq * block_size
                    decode_slot_mapping[seq_idx] = seq_offset + cur_pos

                decode_seq_lens = torch.full((batch_size,), cur_pos + 1, dtype=torch.int64)
                decode_query_start_locs = torch.arange(batch_size + 1, dtype=torch.int64)

                decode_inputs = EngineInputs(
                    token_ids=decode_token_ids, positions=decode_positions, block_table=block_table,
                    slot_mapping=decode_slot_mapping, seq_lens=decode_seq_lens,
                    query_start_locs=decode_query_start_locs, max_decode_seq_len=cur_pos + 1)

                decode_step = StepDescriptor(
                    step_id=step+1, step_kind="decode", num_scheduled_tokens=batch_size,
                    num_seqs_active=batch_size, max_num_blocks_per_seq=blocks_per_seq + 4,
                    is_first_step=False, cache_enabled=True)

                outputs = runner.execute_step(decode_step, decode_inputs)

                for seq_idx in range(batch_size):
                    token = torch.argmax(outputs.logits[seq_idx]).item()
                    generated[seq_idx].append(token)

            # Check coherence
            coherent_count = 0
            for seq_idx in range(batch_size):
                output = tokenizer.decode(generated[seq_idx], skip_special_tokens=True)

                # Check for meaningful Python content
                meaningful = any(k in output for k in ['if', 'return', 'def', 'else', 'for', 'while', ':', '='])
                # Check for garbage (excessive repetition)
                garbage = any(c * 15 in output for c in 'abcdefghijklmnopqrstuvwxyz0123456789')

                if meaningful and not garbage:
                    coherent_count += 1

            rate = coherent_count / batch_size * 100
            prompt_results.append({'batch': batch_size, 'coherent': coherent_count, 'total': batch_size, 'rate': rate})

            # Print sample output
            sample = tokenizer.decode(generated[0], skip_special_tokens=True)
            status = "OK" if rate >= 80 else "WARN" if rate >= 50 else "FAIL"
            print(f"  Batch {batch_size:2d}: {coherent_count}/{batch_size} coherent ({rate:.0f}%) [{status}]")
            print(f"    Sample: {sample[:50]}...")

        results.extend(prompt_results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_coherent = sum(r['coherent'] for r in results)
    total_tested = sum(r['total'] for r in results)
    overall_rate = total_coherent / total_tested * 100 if total_tested > 0 else 0

    print(f"Total: {total_coherent}/{total_tested} ({overall_rate:.0f}%)")

    if overall_rate >= 90:
        print("\nVERDICT: PASS - Engine produces coherent outputs")
        return True
    elif overall_rate >= 70:
        print("\nVERDICT: ACCEPTABLE - Minor coherence issues")
        return True
    else:
        print("\nVERDICT: FAIL - Significant coherence issues")
        return False


if __name__ == "__main__":
    success = test_coherence_simple(batch_sizes=[1, 4, 8, 16])
    exit(0 if success else 1)

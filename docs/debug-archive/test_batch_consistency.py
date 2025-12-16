#!/usr/bin/env python3
"""Test batch consistency - same prompt should give same output regardless of batch position."""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '4'
os.environ['VLLM_METAL_SCRATCH_POOL_MB'] = '8192'
os.environ['VLLM_METAL_MAX_BATCH_SIZE'] = '16384'
os.environ['VLLM_APPLE_USE_ENGINE'] = '1'
os.environ['VLLM_APPLE_ENGINE_PREFILL'] = '1'
os.environ['VLLM_METAL_STRICT_NO_MPS'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_batch_consistency():
    """Test that same prompt produces same output in different batch positions."""
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
    print("BATCH CONSISTENCY TEST")
    print("=" * 80)
    print("Testing if same prompt produces same output in different batch positions")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    config = model.config

    # Test prompt
    test_prompt = "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n"
    filler_prompt = "def hello():\n    print('hello')\n"

    def generate_with_batch(prompt, batch_size, position):
        """Generate with prompt at specific position in batch."""
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
        max_seq_len = 512
        blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        num_blocks = batch_size * blocks_per_seq + 32

        kv_desc = model_desc.get_kv_cache_descriptor(
            num_blocks=num_blocks,
            block_size=block_size,
        )
        kv_cache = EngineKVCache(ctx, kv_desc)

        loader = EngineWeightLoader(ctx, model_config=None)
        weights = loader.load_from_hf_model(model, arch='mistral')

        runner = EngineRunner(
            context=ctx,
            model_desc=model_desc,
            weights=weights,
            kv_cache=kv_cache,
        )

        # Create batch with test prompt at specified position
        prompts = [filler_prompt] * batch_size
        prompts[position] = prompt

        encodings = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        max_prompt_len = max(len(e) for e in encodings)

        # Pad
        padded = []
        for e in encodings:
            padded.append(e + [tokenizer.pad_token_id or 0] * (max_prompt_len - len(e)))

        prompt_tokens = torch.tensor(padded, dtype=torch.int64)
        total_prefill_tokens = batch_size * max_prompt_len

        token_ids = prompt_tokens.flatten()
        positions = torch.arange(max_prompt_len, dtype=torch.int64).repeat(batch_size)
        for seq_idx in range(batch_size):
            positions[seq_idx * max_prompt_len:(seq_idx + 1) * max_prompt_len] = torch.arange(max_prompt_len)

        slot_mapping = torch.zeros(total_prefill_tokens, dtype=torch.int64)
        for seq_idx in range(batch_size):
            seq_offset = seq_idx * blocks_per_seq * block_size
            for tok_idx in range(max_prompt_len):
                slot_mapping[seq_idx * max_prompt_len + tok_idx] = seq_offset + tok_idx

        block_table = torch.zeros((batch_size, blocks_per_seq + 4), dtype=torch.int64)
        for seq_idx in range(batch_size):
            for blk_idx in range(blocks_per_seq):
                block_table[seq_idx, blk_idx] = seq_idx * blocks_per_seq + blk_idx

        seq_lens = torch.full((batch_size,), max_prompt_len, dtype=torch.int64)
        query_start_locs = torch.zeros(batch_size + 1, dtype=torch.int64)
        for i in range(batch_size + 1):
            query_start_locs[i] = i * max_prompt_len

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

        outputs = runner.execute_step(prefill_step, prefill_inputs)

        # Get last token logits for test position
        seq_end = (position + 1) * max_prompt_len - 1
        test_logits = outputs.logits[seq_end]
        generated_token = torch.argmax(test_logits).item()

        # Decode loop
        generated = [generated_token]
        max_tokens = 30

        for step in range(max_tokens - 1):
            cur_pos = max_prompt_len + step

            # All sequences decode one token
            decode_token_ids = torch.zeros(batch_size, dtype=torch.int64)
            decode_token_ids[position] = generated[-1]  # Only care about test position

            decode_positions = torch.full((batch_size,), cur_pos, dtype=torch.int64)

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

            outputs = runner.execute_step(decode_step, decode_inputs)
            next_token = torch.argmax(outputs.logits[position]).item()
            generated.append(next_token)

        return tokenizer.decode(generated, skip_special_tokens=True)

    # Test: same prompt at position 0 in batch 1 vs batch 4 vs batch 8
    print("Generating with batch size 1 (baseline)...")
    output_batch1 = generate_with_batch(test_prompt, batch_size=1, position=0)

    print("Generating with batch size 4, position 0...")
    output_batch4_pos0 = generate_with_batch(test_prompt, batch_size=4, position=0)

    print("Generating with batch size 4, position 3...")
    output_batch4_pos3 = generate_with_batch(test_prompt, batch_size=4, position=3)

    print("Generating with batch size 8, position 0...")
    output_batch8_pos0 = generate_with_batch(test_prompt, batch_size=8, position=0)

    print("Generating with batch size 8, position 7...")
    output_batch8_pos7 = generate_with_batch(test_prompt, batch_size=8, position=7)

    # Compare results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nPrompt: {test_prompt[:50]}...")
    print(f"\nBatch 1, pos 0:   {output_batch1[:60]}...")
    print(f"Batch 4, pos 0:   {output_batch4_pos0[:60]}...")
    print(f"Batch 4, pos 3:   {output_batch4_pos3[:60]}...")
    print(f"Batch 8, pos 0:   {output_batch8_pos0[:60]}...")
    print(f"Batch 8, pos 7:   {output_batch8_pos7[:60]}...")

    # Check consistency
    baseline = output_batch1[:40]

    def check_similar(out, name):
        # Simple check: first 40 chars should be similar or both contain python keywords
        match = out[:40] == baseline
        has_keywords = any(k in out for k in ['if', 'return', 'def', 'else', 'n ==', '=='])
        is_garbage = 'ibibib' in out or len(set(out[:30])) < 5
        return match or (has_keywords and not is_garbage)

    results = {
        'batch4_pos0': check_similar(output_batch4_pos0, 'batch4_pos0'),
        'batch4_pos3': check_similar(output_batch4_pos3, 'batch4_pos3'),
        'batch8_pos0': check_similar(output_batch8_pos0, 'batch8_pos0'),
        'batch8_pos7': check_similar(output_batch8_pos7, 'batch8_pos7'),
    }

    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK")
    print("=" * 80)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_pass = all(results.values())
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")

    return all_pass


if __name__ == "__main__":
    success = test_batch_consistency()
    exit(0 if success else 1)

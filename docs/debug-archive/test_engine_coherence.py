#!/usr/bin/env python3
"""Test engine output coherence with real prompts.

This tests that the Metal Engine produces coherent, meaningful outputs
across different batch sizes and prompt lengths.
"""
import os
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '4'
os.environ['VLLM_METAL_SCRATCH_POOL_MB'] = '8192'
os.environ['VLLM_METAL_MAX_BATCH_SIZE'] = '16384'
os.environ['VLLM_APPLE_USE_ENGINE'] = '1'
os.environ['VLLM_APPLE_ENGINE_PREFILL'] = '1'
os.environ['VLLM_METAL_STRICT_NO_MPS'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_coherence(batch_sizes=[1, 4, 8], max_tokens=50):
    """Generate text and check for coherence."""
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
    print("ENGINE COHERENCE TEST")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Batch sizes to test: {batch_sizes}")
    print(f"Max tokens: {max_tokens}")
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

    # Test prompts of varying complexity
    prompts = [
        "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n",
        "# Python function to sort a list\ndef bubble_sort(arr):\n",
        "Write a function to check if a string is a palindrome:\n\ndef is_palindrome(s):\n",
        "# Binary search implementation\ndef binary_search(arr, target):\n",
        "class LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, data):\n",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
        "# Calculate factorial\ndef factorial(n):\n",
        "def merge_sort(arr):\n    \"\"\"Merge sort implementation.\"\"\"\n",
        "# HTTP request handler\nasync def handle_request(request):\n",
        "def validate_email(email):\n    \"\"\"Check if email is valid.\"\"\"\n",
        "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n",
        "def bfs(graph, start):\n    \"\"\"Breadth-first search.\"\"\"\n",
        "# Matrix multiplication\ndef matmul(A, B):\n",
        "def dijkstra(graph, start):\n    \"\"\"Dijkstra's shortest path.\"\"\"\n",
        "# Hash table implementation\nclass HashTable:\n    def __init__(self, size=100):\n",
        "def depth_first_search(graph, node, visited=None):\n",
    ]

    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*60}")

        # Create fresh context
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

        # KV cache
        block_size = 16
        max_seq_len = 1024 + max_tokens
        blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        num_blocks = batch_size * blocks_per_seq + 32

        kv_desc = model_desc.get_kv_cache_descriptor(
            num_blocks=num_blocks,
            block_size=block_size,
        )
        kv_cache = EngineKVCache(ctx, kv_desc)

        # Load weights
        loader = EngineWeightLoader(ctx, model_config=None)
        weights = loader.load_from_hf_model(model, arch='mistral')

        # Create runner
        runner = EngineRunner(
            context=ctx,
            model_desc=model_desc,
            weights=weights,
            kv_cache=kv_cache,
        )

        # Select prompts for this batch
        batch_prompts = prompts[:batch_size]

        # Tokenize
        encodings = [tokenizer.encode(p, add_special_tokens=False) for p in batch_prompts]
        max_prompt_len = max(len(e) for e in encodings)

        # Pad to same length
        padded = []
        for e in encodings:
            padded.append(e + [tokenizer.pad_token_id or 0] * (max_prompt_len - len(e)))

        prompt_tokens = torch.tensor(padded, dtype=torch.int64)
        total_prefill_tokens = batch_size * max_prompt_len

        # Create prefill inputs
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

        # Run prefill
        outputs = runner.execute_step(prefill_step, prefill_inputs)

        # Get initial tokens (greedy sampling)
        logits = outputs.logits
        # Get last token logits for each sequence AT ACTUAL CONTENT END (not padded end)
        actual_lens = [len(e) for e in encodings]
        last_logits = []
        for seq_idx in range(batch_size):
            # Position of actual last token for this sequence
            actual_end = seq_idx * max_prompt_len + actual_lens[seq_idx] - 1
            last_logits.append(logits[actual_end])
        last_logits = torch.stack(last_logits)
        generated_tokens = [torch.argmax(last_logits[i]).item() for i in range(batch_size)]

        # Track generated sequences
        generated = [[t] for t in generated_tokens]

        # Decode loop
        for step in range(max_tokens - 1):
            cur_pos = max_prompt_len + step

            decode_token_ids = torch.tensor(generated_tokens, dtype=torch.int64)
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

            # Greedy sampling
            generated_tokens = [torch.argmax(outputs.logits[i]).item() for i in range(batch_size)]
            for i, t in enumerate(generated_tokens):
                generated[i].append(t)

        # Decode and print results
        batch_coherent = 0
        for seq_idx in range(batch_size):
            prompt = batch_prompts[seq_idx]
            output_text = tokenizer.decode(generated[seq_idx], skip_special_tokens=True)
            full_text = prompt + output_text

            # Basic coherence check: output should contain Python keywords
            # and not be garbage/repetitive
            coherent = True

            # Check for meaningful content
            meaningful_tokens = ['def', 'return', 'if', 'else', 'for', 'while', 'self',
                               '(', ')', ':', '=', 'in', 'not', 'and', 'or']
            has_meaningful = any(t in output_text for t in meaningful_tokens)

            # Check for excessive repetition (same char repeated > 20 times)
            max_repeat = 0
            for i in range(len(output_text) - 1):
                count = 1
                while i + count < len(output_text) and output_text[i] == output_text[i + count]:
                    count += 1
                max_repeat = max(max_repeat, count)

            excessive_repeat = max_repeat > 20

            if not has_meaningful or excessive_repeat:
                coherent = False

            if coherent:
                batch_coherent += 1

            print(f"\n[Seq {seq_idx}] {'COHERENT' if coherent else 'INCOHERENT'}")
            print(f"Prompt: {prompt[:50]}...")
            print(f"Output: {output_text[:100]}...")

        coherence_rate = batch_coherent / batch_size * 100
        print(f"\nBatch {batch_size}: {batch_coherent}/{batch_size} coherent ({coherence_rate:.0f}%)")

        results.append({
            'batch_size': batch_size,
            'coherent': batch_coherent,
            'total': batch_size,
            'rate': coherence_rate,
        })

    # Summary
    print("\n")
    print("=" * 80)
    print("COHERENCE TEST SUMMARY")
    print("=" * 80)

    total_coherent = sum(r['coherent'] for r in results)
    total_tested = sum(r['total'] for r in results)
    overall_rate = total_coherent / total_tested * 100 if total_tested > 0 else 0

    for r in results:
        print(f"Batch {r['batch_size']:2d}: {r['coherent']}/{r['total']} coherent ({r['rate']:.0f}%)")

    print(f"\nOverall: {total_coherent}/{total_tested} ({overall_rate:.0f}%)")

    if overall_rate >= 80:
        print("\nVERDICT: PASS - Engine produces coherent outputs")
    else:
        print("\nVERDICT: FAIL - Engine outputs are not coherent")

    return overall_rate >= 80


if __name__ == "__main__":
    success = test_coherence(batch_sizes=[1, 4, 8])
    exit(0 if success else 1)

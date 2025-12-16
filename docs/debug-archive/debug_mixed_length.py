#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Debug script for mixed prompt length bug.

This script reproduces the mixed prompt length issue and captures
detailed diagnostic information to identify the root cause.

Usage:
    VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 \
    VLLM_METAL_SCRATCH_POOL_MB=4096 \
    python debug_mixed_length.py
"""

import os
import sys
import logging

# Set environment variables before importing vllm
os.environ.setdefault("VLLM_APPLE_USE_ENGINE", "1")
os.environ.setdefault("VLLM_APPLE_ENGINE_PREFILL", "1")
os.environ.setdefault("VLLM_METAL_SCRATCH_POOL_MB", "4096")

# Enable verbose logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from vllm import LLM, SamplingParams


def get_eos_token_id(tokenizer) -> int:
    """Get EOS token ID from tokenizer."""
    if hasattr(tokenizer, 'eos_token_id'):
        return tokenizer.eos_token_id
    if hasattr(tokenizer, 'eos_id'):
        return tokenizer.eos_id
    return 2  # Common default


def create_prompts_with_lengths(tokenizer, target_lengths: list[int]) -> list[str]:
    """Create prompts with specific token lengths."""
    base_text = "The quick brown fox jumps over the lazy dog. " * 50
    prompts = []

    for target_len in target_lengths:
        # Binary search for character count that gives target token count
        low, high = target_len, target_len * 10
        best_prompt = ""
        best_diff = float('inf')

        while low <= high:
            mid = (low + high) // 2
            candidate = base_text[:mid]
            tokens = tokenizer.encode(candidate)
            diff = abs(len(tokens) - target_len)

            if diff < best_diff:
                best_diff = diff
                best_prompt = candidate

            if len(tokens) < target_len:
                low = mid + 1
            elif len(tokens) > target_len:
                high = mid - 1
            else:
                break

        prompts.append(best_prompt)
        actual_len = len(tokenizer.encode(best_prompt))
        logger.info(f"Target {target_len} tokens -> actual {actual_len} tokens")

    return prompts


def analyze_output(output, target_len: int, eos_token_id: int) -> dict:
    """Analyze a single output for diagnostic information."""
    prompt_tokens = len(output.prompt_token_ids)
    output_tokens = output.outputs[0].token_ids
    output_text = output.outputs[0].text
    finish_reason = output.outputs[0].finish_reason

    analysis = {
        'target_prompt_len': target_len,
        'actual_prompt_len': prompt_tokens,
        'num_output_tokens': len(output_tokens),
        'finish_reason': str(finish_reason),
        'first_output_token': output_tokens[0] if output_tokens else None,
        'last_output_token': output_tokens[-1] if output_tokens else None,
        'is_first_token_eos': output_tokens[0] == eos_token_id if output_tokens else False,
        'is_last_token_eos': output_tokens[-1] == eos_token_id if output_tokens else False,
        'output_preview': output_text[:100] if output_text else '',
    }
    return analysis


def run_diagnostic():
    """Run the diagnostic test."""
    model_name = os.environ.get("MODEL_NAME", "mistralai/Devstral-Small-2505")

    print("=" * 80)
    print("MIXED PROMPT LENGTH BUG DIAGNOSTIC")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"VLLM_APPLE_USE_ENGINE: {os.environ.get('VLLM_APPLE_USE_ENGINE')}")
    print(f"VLLM_APPLE_ENGINE_PREFILL: {os.environ.get('VLLM_APPLE_ENGINE_PREFILL')}")
    print("=" * 80)

    # Load model
    logger.info(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        enforce_eager=True,  # Disable CUDA graphs
    )

    tokenizer = llm.get_tokenizer()
    eos_token_id = get_eos_token_id(tokenizer)
    logger.info(f"EOS token ID: {eos_token_id}")

    # Target prompt lengths that reproduce the bug
    # 379 tokens is the problematic length from the validation report
    target_lengths = [32, 128, 379, 702]

    print("\n--- Creating prompts ---")
    prompts = create_prompts_with_lengths(tokenizer, target_lengths)

    # Verify actual token counts
    actual_lengths = [len(tokenizer.encode(p)) for p in prompts]
    print(f"Actual token counts: {actual_lengths}")

    # Test parameters
    max_tokens = 32
    params = SamplingParams(
        temperature=0.0,  # Greedy for reproducibility
        max_tokens=max_tokens,
    )

    print(f"\n--- Running generation (max_tokens={max_tokens}) ---")
    outputs = llm.generate(prompts, params)

    print("\n--- Results ---")
    print(f"{'Target':<8} {'Prompt':<8} {'Output':<8} {'Status':<8} {'Finish Reason':<20} {'First Token':<12}")
    print("-" * 80)

    all_pass = True
    for i, (target, output) in enumerate(zip(target_lengths, outputs)):
        analysis = analyze_output(output, target, eos_token_id)
        status = "OK" if analysis['num_output_tokens'] == max_tokens else "FAIL"
        if status == "FAIL":
            all_pass = False

        first_tok = analysis['first_output_token']
        first_tok_str = f"{first_tok}" + (" (EOS)" if analysis['is_first_token_eos'] else "")

        print(f"{target:<8} {analysis['actual_prompt_len']:<8} {analysis['num_output_tokens']:<8} "
              f"{status:<8} {analysis['finish_reason']:<20} {first_tok_str:<12}")

        if status == "FAIL":
            print(f"  >>> FAILURE DETAILS:")
            print(f"      Output tokens: {analysis['num_output_tokens']}")
            print(f"      Finish reason: {analysis['finish_reason']}")
            print(f"      First token: {analysis['first_output_token']}")
            print(f"      Is EOS: {analysis['is_first_token_eos']}")
            print(f"      Output preview: {analysis['output_preview'][:50]}...")

    print("\n" + "=" * 80)
    if all_pass:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED - BUG REPRODUCED")
    print("=" * 80)

    # Additional diagnostics for failures
    if not all_pass:
        print("\n--- Additional Diagnostics ---")

        # Test each prompt individually to rule out batch interaction
        print("\nRunning prompts individually to check if batch-related:")
        individual_results = []
        for i, (prompt, target) in enumerate(zip(prompts, target_lengths)):
            single_output = llm.generate([prompt], params)[0]
            analysis = analyze_output(single_output, target, eos_token_id)
            status = "OK" if analysis['num_output_tokens'] == max_tokens else "FAIL"
            individual_results.append(status)
            print(f"  Prompt {i} (target {target}): {analysis['num_output_tokens']} tokens [{status}]")

        if all(r == "OK" for r in individual_results):
            print("\n>>> Individual runs all pass - BUG IS BATCH-RELATED")
        else:
            print("\n>>> Individual runs also fail - BUG IS NOT BATCH-RELATED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_diagnostic())

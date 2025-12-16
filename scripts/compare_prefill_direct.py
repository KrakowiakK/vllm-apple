#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Direct comparison of Engine vs PyTorch prefill in a single run.

This script performs a single comparison without subprocess isolation.
Run it twice with different environment variables to compare:

# PyTorch-MPS (reference):
VLLM_APPLE_USE_ENGINE=0 python scripts/compare_prefill_direct.py --output pytorch_result.json

# Engine with prefill:
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 python scripts/compare_prefill_direct.py --output engine_result.json

# Then compare:
python scripts/compare_results.py pytorch_result.json engine_result.json
"""

import argparse
import json
import os
import sys

# Configuration
MODEL_NAME = "mistralai/Devstral-Small-2505"
DTYPE = "bfloat16"

def main():
    parser = argparse.ArgumentParser(description="Run prefill and capture output")
    parser.add_argument("--tokens", type=int, default=500, help="Target prompt length")
    parser.add_argument("--template", type=str, default="fox",
                        choices=["fox", "code", "story", "instruction"])
    parser.add_argument("--max-output", type=int, default=32, help="Max output tokens")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--prompt", type=str, help="Custom prompt text")
    args = parser.parse_args()

    # Import after argument parsing
    from vllm import LLM, SamplingParams

    # Report environment
    use_engine = os.environ.get("VLLM_APPLE_USE_ENGINE", "0")
    engine_prefill = os.environ.get("VLLM_APPLE_ENGINE_PREFILL", "0")
    mode = f"engine(prefill={engine_prefill})" if use_engine == "1" else "pytorch"

    print(f"Mode: {mode}")
    print(f"VLLM_APPLE_USE_ENGINE: {use_engine}")
    print(f"VLLM_APPLE_ENGINE_PREFILL: {engine_prefill}")
    print(f"Model: {MODEL_NAME}")

    # Load model
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        enforce_eager=True,
        dtype=DTYPE,
        max_model_len=4096,  # Reduced for testing to conserve memory
    )
    tokenizer = llm.get_tokenizer()

    # Generate prompt
    if args.prompt:
        prompt = args.prompt
    else:
        templates = {
            "fox": "The quick brown fox jumps over the lazy dog. ",
            "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n# ",
            "story": "Once upon a time, in a land far away, there lived a brave knight. ",
            "instruction": "Please explain the following concept in detail: ",
        }
        base = templates.get(args.template, templates["fox"])
        estimated_chars = args.tokens * 4
        repetitions = (estimated_chars // len(base)) + 1
        prompt = (base * repetitions)[:estimated_chars]

    # Tokenize
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    token_count = len(input_ids)
    print(f"Input tokens: {token_count}")

    # Run inference
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_output,
        seed=42,
        logprobs=10,  # Request top-10 logprobs
    )

    outputs = llm.generate([prompt], params, use_tqdm=False)
    output = outputs[0]

    # Extract results
    output_tokens = list(output.outputs[0].token_ids)
    output_text = output.outputs[0].text
    finish_reason = str(output.outputs[0].finish_reason)

    # Get logprobs
    logprobs_info = []
    if output.outputs[0].logprobs:
        for pos_logprobs in output.outputs[0].logprobs:
            sorted_lp = sorted(pos_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)[:10]
            logprobs_info.append([(tok_id, float(lp.logprob)) for tok_id, lp in sorted_lp])

    result = {
        "mode": mode,
        "use_engine": use_engine,
        "engine_prefill": engine_prefill,
        "model": MODEL_NAME,
        "dtype": DTYPE,
        "input_token_count": token_count,
        "output_token_count": len(output_tokens),
        "output_tokens": output_tokens,
        "output_text": output_text[:500],
        "finish_reason": finish_reason,
        "first_token": output_tokens[0] if output_tokens else None,
        "last_token": output_tokens[-1] if output_tokens else None,
        "eos_token_id": tokenizer.eos_token_id,
        "ends_with_eos": output_tokens[-1] == tokenizer.eos_token_id if output_tokens else False,
        "input_ids_last10": input_ids[-10:],
        "logprobs_first_position": logprobs_info[0] if logprobs_info else [],
        "logprobs_all": logprobs_info,
    }

    # Print summary
    print(f"\n--- Results ({mode}) ---")
    print(f"Output tokens: {len(output_tokens)}")
    print(f"First token: {output_tokens[0] if output_tokens else 'N/A'}")
    print(f"Output: {output_tokens[:20]}...")
    print(f"Finish reason: {finish_reason}")
    print(f"Ends with EOS: {result['ends_with_eos']}")

    if logprobs_info:
        print(f"\nFirst position top-5:")
        for i, (tok_id, logprob) in enumerate(logprobs_info[0][:5]):
            token_str = tokenizer.decode([tok_id])
            print(f"  [{i}] {tok_id} ({repr(token_str)}): {logprob:.4f}")

    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print(f"\nJSON: {json.dumps(result)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

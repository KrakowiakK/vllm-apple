#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI tool to reproduce and diagnose engine prefill divergence.

This script provides a simple interface to compare Engine vs PyTorch prefill
outputs and identify divergence.

Usage:
    # Quick test with default prompt
    python scripts/repro_prefill_divergence.py

    # Test with custom prompt
    python scripts/repro_prefill_divergence.py --prompt "Your custom prompt here"

    # Test with specific token length
    python scripts/repro_prefill_divergence.py --tokens 500

    # Test known-failing fox prompt
    python scripts/repro_prefill_divergence.py --fox

    # Verbose output with logits
    python scripts/repro_prefill_divergence.py --verbose

    # Export results to JSON
    python scripts/repro_prefill_divergence.py --output results.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, Any, Optional

# Configuration
MODEL_NAME = "mistralai/Devstral-Small-2505"
# CRITICAL: Use float16 for equivalence testing.
# bfloat16 is model's native dtype but MPS has limited bfloat16 support,
# and engine uses float16 throughout. Standardize to float16 for fair comparison.
DTYPE = "float16"


def create_prompt_with_tokens(target_length: int, template: str = "fox") -> str:
    """Create a prompt with approximately target_length tokens."""
    templates = {
        "fox": "The quick brown fox jumps over the lazy dog. ",
        "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n# ",
        "story": "Once upon a time, in a land far away, there lived a brave knight. ",
        "instruction": "Please explain the following concept in detail: ",
    }

    base = templates.get(template, templates["fox"])
    # Estimate ~4 chars per token on average
    estimated_chars = target_length * 4
    repetitions = (estimated_chars // len(base)) + 1
    return (base * repetitions)[:estimated_chars]


def run_inference(prompt: str, use_engine: bool, engine_prefill: bool,
                  max_tokens: int = 10, verbose: bool = False) -> Dict[str, Any]:
    """Run inference in a subprocess and return results."""

    script = f'''
import os
import json

# Set environment BEFORE imports
os.environ["VLLM_APPLE_USE_ENGINE"] = "{'1' if use_engine else '0'}"
os.environ["VLLM_APPLE_ENGINE_PREFILL"] = "{'1' if engine_prefill else '0'}"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_METAL_SCRATCH_POOL_MB"] = "8192"
os.environ["VLLM_METAL_MAX_BATCH_SIZE"] = "16384"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # Required for subprocess execution

from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="{MODEL_NAME}",
    trust_remote_code=True,
    enforce_eager=True,
    dtype="{DTYPE}",
    max_model_len=8192,  # Reduce from default 131072 to fit in memory
)
tokenizer = llm.get_tokenizer()

# Encode prompt - use raw string to avoid escaping issues
prompt = {repr(prompt)}
input_ids = tokenizer.encode(prompt, add_special_tokens=False)
token_count = len(input_ids)

# Run generation with logprobs for comparison
params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,  # Disable top_k
    max_tokens={max_tokens},
    seed=42,
    logprobs=20,  # Request top-20 logprobs for comparison
)

outputs = llm.generate([prompt], params, use_tqdm=False)
output = outputs[0]

# Get output tokens
output_tokens = list(output.outputs[0].token_ids)
output_text = output.outputs[0].text
finish_reason = str(output.outputs[0].finish_reason)

# Get logprobs if available (for top tokens)
logprobs_info = []
if output.outputs[0].logprobs:
    for pos_logprobs in output.outputs[0].logprobs[:5]:  # First 5 positions
        sorted_lp = sorted(pos_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)[:10]
        logprobs_info.append([(tok_id, lp.logprob) for tok_id, lp in sorted_lp])

result = {{
    "mode": "engine" if {use_engine} else "pytorch",
    "engine_prefill": {engine_prefill},
    "input_token_count": token_count,
    "output_token_count": len(output_tokens),
    "output_tokens": output_tokens,
    "output_text": output_text[:200],
    "finish_reason": finish_reason,
    "first_token": output_tokens[0] if output_tokens else None,
    "last_token": output_tokens[-1] if output_tokens else None,
    "eos_token_id": tokenizer.eos_token_id,
    "ends_with_eos": output_tokens[-1] == tokenizer.eos_token_id if output_tokens else False,
    "logprobs_top10": logprobs_info,
}}

print("RESULT:" + json.dumps(result))
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        env = os.environ.copy()
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        if verbose:
            print(f"  Running {'Engine' if use_engine else 'PyTorch'} path...")

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        # Parse result
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT:"):
                return json.loads(line[7:])

        # Error case
        print("ERROR: No result found in subprocess output")
        if verbose:
            print(f"STDOUT (last 3000 chars):\n{result.stdout[-3000:]}")
            print(f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}")
        return {"error": "No result found", "stderr": result.stderr[-500:]}

    except subprocess.TimeoutExpired:
        return {"error": "Subprocess timeout"}
    finally:
        os.unlink(script_path)


def compare_results(pytorch: Dict, engine: Dict, verbose: bool = False) -> Dict[str, Any]:
    """Compare PyTorch and Engine results."""

    comparison = {
        "pytorch_tokens": pytorch.get("output_tokens", []),
        "engine_tokens": engine.get("output_tokens", []),
        "pytorch_first": pytorch.get("first_token"),
        "engine_first": engine.get("first_token"),
        "first_token_match": pytorch.get("first_token") == engine.get("first_token"),
        "pytorch_count": pytorch.get("output_token_count", 0),
        "engine_count": engine.get("output_token_count", 0),
        "count_match": pytorch.get("output_token_count") == engine.get("output_token_count"),
        "pytorch_eos": pytorch.get("ends_with_eos", False),
        "engine_eos": engine.get("ends_with_eos", False),
    }

    # Check token sequence match
    pt_tokens = pytorch.get("output_tokens", [])
    en_tokens = engine.get("output_tokens", [])
    min_len = min(len(pt_tokens), len(en_tokens))

    matching_prefix = 0
    for i in range(min_len):
        if pt_tokens[i] == en_tokens[i]:
            matching_prefix += 1
        else:
            break

    comparison["matching_prefix_length"] = matching_prefix
    comparison["full_match"] = pt_tokens == en_tokens

    # Verdict
    if comparison["first_token_match"] and comparison["count_match"] and comparison["full_match"]:
        comparison["verdict"] = "PASS"
    else:
        comparison["verdict"] = "FAIL"

    return comparison


def print_results(prompt: str, pytorch: Dict, engine: Dict, comparison: Dict, verbose: bool = False):
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("ENGINE vs PYTORCH PREFILL COMPARISON")
    print("=" * 70)

    print(f"\nPrompt: {prompt[:100]}..." if len(prompt) > 100 else f"\nPrompt: {prompt}")
    print(f"Input tokens: {pytorch.get('input_token_count', 'N/A')}")

    print("\n--- PyTorch Path ---")
    print(f"Output tokens: {pytorch.get('output_token_count', 'N/A')}")
    print(f"First token: {pytorch.get('first_token', 'N/A')}")
    print(f"Tokens: {pytorch.get('output_tokens', [])[:20]}{'...' if len(pytorch.get('output_tokens', [])) > 20 else ''}")
    print(f"Finish: {pytorch.get('finish_reason', 'N/A')}")
    print(f"Ends with EOS: {pytorch.get('ends_with_eos', 'N/A')}")

    print("\n--- Engine Path ---")
    print(f"Output tokens: {engine.get('output_token_count', 'N/A')}")
    print(f"First token: {engine.get('first_token', 'N/A')}")
    print(f"Tokens: {engine.get('output_tokens', [])[:20]}{'...' if len(engine.get('output_tokens', [])) > 20 else ''}")
    print(f"Finish: {engine.get('finish_reason', 'N/A')}")
    print(f"Ends with EOS: {engine.get('ends_with_eos', 'N/A')}")

    print("\n--- Comparison ---")
    print(f"First token match: {comparison['first_token_match']}")
    print(f"Token count match: {comparison['count_match']}")
    print(f"Full sequence match: {comparison['full_match']}")
    print(f"Matching prefix: {comparison['matching_prefix_length']} tokens")

    if verbose:
        print(f"\nDebug: PyTorch has {len(pytorch.get('logprobs_top10', []))} logprob positions")
        print(f"Debug: Engine has {len(engine.get('logprobs_top10', []))} logprob positions")

    if verbose and pytorch.get("logprobs_top10") and engine.get("logprobs_top10"):
        # Show logprobs for first 3 positions
        for pos in range(min(3, len(pytorch.get("logprobs_top10", [])), len(engine.get("logprobs_top10", [])))):
            print(f"\n--- Position {pos} Top-10 Logprobs ---")
            pt_lp = pytorch["logprobs_top10"][pos] if len(pytorch["logprobs_top10"]) > pos else []
            en_lp = engine["logprobs_top10"][pos] if len(engine["logprobs_top10"]) > pos else []

            print("PyTorch:")
            for i, (tok_id, logprob) in enumerate(pt_lp[:10]):
                print(f"  [{i}] {tok_id}: {logprob:.4f}")

            print("Engine:")
            for i, (tok_id, logprob) in enumerate(en_lp[:10]):
                print(f"  [{i}] {tok_id}: {logprob:.4f}")

    print("\n" + "=" * 70)
    verdict_str = comparison['verdict']
    if verdict_str == "PASS":
        print(f"VERDICT: {verdict_str}")
    else:
        print(f"VERDICT: {verdict_str} - DIVERGENCE DETECTED")
        print("\nDivergence details:")
        print(f"  PyTorch first token: {comparison['pytorch_first']}")
        print(f"  Engine first token:  {comparison['engine_first']}")
        if not comparison['first_token_match']:
            print("  >>> FIRST TOKEN MISMATCH - Prefill hidden states diverged")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce and diagnose engine prefill divergence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt text")
    parser.add_argument("--tokens", type=int, default=500, help="Target prompt length in tokens")
    parser.add_argument("--template", type=str, default="fox",
                        choices=["fox", "code", "story", "instruction"],
                        help="Template for generating prompt")
    parser.add_argument("--fox", action="store_true", help="Use known-failing 500-token fox prompt")
    parser.add_argument("--max-output", type=int, default=32, help="Max output tokens")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output with logprobs")
    parser.add_argument("--output", "-o", type=str, help="Export results to JSON file")
    parser.add_argument("--pytorch-only", action="store_true", help="Run only PyTorch path")
    parser.add_argument("--engine-only", action="store_true", help="Run only Engine path")

    args = parser.parse_args()

    # Determine prompt
    if args.prompt:
        prompt = args.prompt
    elif args.fox:
        prompt = create_prompt_with_tokens(500, "fox")
    else:
        prompt = create_prompt_with_tokens(args.tokens, args.template)

    print(f"Model: {MODEL_NAME}")
    print(f"Dtype: {DTYPE}")
    print(f"Prompt length: ~{args.tokens if not args.prompt else len(prompt)//4} tokens")

    results = {"prompt_preview": prompt[:200], "pytorch": None, "engine": None, "comparison": None}

    # Run PyTorch
    if not args.engine_only:
        print("\nRunning PyTorch-MPS reference...")
        pytorch_result = run_inference(prompt, use_engine=False, engine_prefill=False,
                                       max_tokens=args.max_output, verbose=args.verbose)
        results["pytorch"] = pytorch_result

    # Run Engine
    if not args.pytorch_only:
        print("Running Engine with prefill...")
        engine_result = run_inference(prompt, use_engine=True, engine_prefill=True,
                                      max_tokens=args.max_output, verbose=args.verbose)
        results["engine"] = engine_result

    # Compare
    if results["pytorch"] and results["engine"]:
        comparison = compare_results(results["pytorch"], results["engine"], args.verbose)
        results["comparison"] = comparison
        print_results(prompt, results["pytorch"], results["engine"], comparison, args.verbose)

        # Export if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults exported to: {args.output}")

        return 0 if comparison["verdict"] == "PASS" else 1

    elif results["pytorch"]:
        print("\n--- PyTorch Results ---")
        print(json.dumps(results["pytorch"], indent=2))
        return 0

    elif results["engine"]:
        print("\n--- Engine Results ---")
        print(json.dumps(results["engine"], indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

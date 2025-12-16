#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bisection test to identify first divergent checkpoint in prefill.

This test runs both PyTorch-MPS and Engine paths with checkpoint capture enabled,
then compares checkpoints to identify the first point of divergence.

Usage:
    VLLM_PREFILL_EQ_DEBUG=1 pytest tests/test_prefill_bisection.py -v -s

Environment:
    VLLM_PREFILL_EQ_DEBUG=1  Required - enables checkpoint capture
    VLLM_ENGINE_DEBUG=1      Optional - verbose engine logging
"""

import os
import sys
import json
import pytest
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np

# Constants
MODEL_NAME = "mistralai/Devstral-Small-2505"
DTYPE = "float16"  # Standardized dtype


class TestPrefillBisection:
    """Bisection test to find first divergent checkpoint."""

    @pytest.fixture
    def short_prompt(self):
        """Short prompt for quick bisection."""
        return "The quick brown fox jumps over the lazy dog."

    @pytest.fixture
    def medium_prompt(self):
        """Medium prompt (approx 128 tokens)."""
        base = "The quick brown fox jumps over the lazy dog. "
        return base * 15  # ~120 tokens

    def _run_inference_with_checkpoints(
        self,
        prompt: str,
        use_engine: bool,
        engine_prefill: bool,
    ) -> Dict[str, Any]:
        """Run inference in subprocess with checkpoint capture.

        Returns dict with:
            - top1_id: Top-1 token ID
            - top1_logit: Top-1 logit value
            - top5_ids: Top-5 token IDs
            - checkpoints: Dict of checkpoint data (if VLLM_PREFILL_EQ_DEBUG=1)
        """
        escaped_prompt = prompt.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

        script = f'''
import os
import sys
import json
import torch
import numpy as np

# Set environment BEFORE imports
os.environ["VLLM_APPLE_USE_ENGINE"] = "{'1' if use_engine else '0'}"
os.environ["VLLM_APPLE_ENGINE_PREFILL"] = "{'1' if engine_prefill else '0'}"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_METAL_SCRATCH_POOL_MB"] = "8192"
os.environ["VLLM_METAL_MAX_BATCH_SIZE"] = "16384"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # Required for subprocess execution
# Keep VLLM_PREFILL_EQ_DEBUG from parent environment
os.environ["VLLM_PREFILL_EQ_DEBUG"] = "{os.environ.get('VLLM_PREFILL_EQ_DEBUG', '0')}"

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

# Encode prompt
prompt = """{escaped_prompt}"""
input_ids = tokenizer.encode(prompt, add_special_tokens=False)
token_count = len(input_ids)

# Run generation
params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=1,
    seed=42,
    logprobs=20,
)

outputs = llm.generate([prompt], params, use_tqdm=False)
output = outputs[0]

# Extract logprobs
logprobs = output.outputs[0].logprobs
if logprobs and len(logprobs) > 0:
    first_logprobs = logprobs[0]
    sorted_lp = sorted(first_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
    top20 = [(tok_id, lp.logprob) for tok_id, lp in sorted_lp[:20]]
else:
    sampled = output.outputs[0].token_ids[0] if output.outputs[0].token_ids else 0
    top20 = [(sampled, 0.0)]

# Get checkpoints if debug enabled
checkpoints = {{}}
if os.environ.get("VLLM_PREFILL_EQ_DEBUG") == "1":
    try:
        from vllm_apple.debug import get_checkpoint_store
        source = "engine" if {'true' if use_engine and engine_prefill else 'false'} else "pytorch"
        store = get_checkpoint_store(source)
        for name in store.list_checkpoints():
            ckpt = store.get(name)
            if ckpt:
                checkpoints[name] = {{
                    "shape": list(ckpt.shape),
                    "min": float(ckpt.min_val),
                    "max": float(ckpt.max_val),
                    "mean": float(ckpt.mean_val),
                    "has_nan": ckpt.has_nan,
                    "has_inf": ckpt.has_inf,
                }}
    except Exception as e:
        print(f"[WARN] Failed to get checkpoints: {{e}}", file=sys.stderr)

result = {{
    "token_count": token_count,
    "top1_id": top20[0][0],
    "top1_logit": float(top20[0][1]),
    "top5_ids": [t[0] for t in top20[:5]],
    "top20": top20,
    "use_engine": {'true' if use_engine else 'false'},
    "engine_prefill": {'true' if engine_prefill else 'false'},
    "checkpoints": checkpoints,
}}

print("RESULT_JSON:" + json.dumps(result))
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            env = os.environ.copy()
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            # Parse result
            for line in result.stdout.split('\n'):
                if line.startswith("RESULT_JSON:"):
                    return json.loads(line[12:])

            print(f"STDERR (last 2000 chars):\n{result.stderr[-2000:]}")
            raise RuntimeError("No result found in subprocess output")

        finally:
            os.unlink(script_path)

    def _compare_results(
        self,
        pytorch_result: Dict,
        engine_result: Dict,
    ) -> Dict[str, Any]:
        """Compare PyTorch and Engine results."""
        comparison = {
            "top1_match": pytorch_result["top1_id"] == engine_result["top1_id"],
            "pytorch_top1": pytorch_result["top1_id"],
            "engine_top1": engine_result["top1_id"],
            "pytorch_top1_logit": pytorch_result["top1_logit"],
            "engine_top1_logit": engine_result["top1_logit"],
            "logit_diff": abs(pytorch_result["top1_logit"] - engine_result["top1_logit"]),
            "top5_overlap": len(set(pytorch_result["top5_ids"]) & set(engine_result["top5_ids"])),
        }

        # Compute logprob differences for common tokens
        pytorch_dict = {t[0]: t[1] for t in pytorch_result["top20"]}
        engine_dict = {t[0]: t[1] for t in engine_result["top20"]}
        common = set(pytorch_dict.keys()) & set(engine_dict.keys())

        if common:
            diffs = [abs(pytorch_dict[t] - engine_dict[t]) for t in common]
            comparison["max_logprob_diff"] = max(diffs)
            comparison["mean_logprob_diff"] = sum(diffs) / len(diffs)
        else:
            comparison["max_logprob_diff"] = float('inf')
            comparison["mean_logprob_diff"] = float('inf')

        # Verdict
        if comparison["top1_match"] and comparison["top5_overlap"] >= 5:
            comparison["verdict"] = "PASS"
        else:
            comparison["verdict"] = "FAIL"

        return comparison

    def test_bisection_short_prompt(self, short_prompt):
        """Run bisection on short prompt."""
        print(f"\n{'='*70}")
        print("PREFILL BISECTION TEST - SHORT PROMPT")
        print(f"Prompt: {short_prompt[:50]}...")
        print(f"VLLM_PREFILL_EQ_DEBUG: {os.environ.get('VLLM_PREFILL_EQ_DEBUG', '0')}")
        print(f"{'='*70}\n")

        # Run PyTorch path
        print("Running PyTorch-MPS reference...")
        pytorch_result = self._run_inference_with_checkpoints(
            short_prompt, use_engine=False, engine_prefill=False
        )
        print(f"PyTorch top-1: {pytorch_result['top1_id']} (logit={pytorch_result['top1_logit']:.4f})")

        # Run Engine path
        print("\nRunning Engine path...")
        engine_result = self._run_inference_with_checkpoints(
            short_prompt, use_engine=True, engine_prefill=True
        )
        print(f"Engine top-1: {engine_result['top1_id']} (logit={engine_result['top1_logit']:.4f})")

        # Compare
        comparison = self._compare_results(pytorch_result, engine_result)

        print(f"\n--- Comparison ---")
        print(f"Top-1 match: {comparison['top1_match']}")
        print(f"Top-5 overlap: {comparison['top5_overlap']}/5")
        print(f"Logit diff: {comparison['logit_diff']:.6f}")
        print(f"Max logprob diff: {comparison['max_logprob_diff']:.6f}")
        print(f"Mean logprob diff: {comparison['mean_logprob_diff']:.6f}")

        # Print checkpoint info if available
        if pytorch_result.get("checkpoints"):
            print(f"\nPyTorch checkpoints: {list(pytorch_result['checkpoints'].keys())}")
        if engine_result.get("checkpoints"):
            print(f"Engine checkpoints: {list(engine_result['checkpoints'].keys())}")

        print(f"\nVERDICT: {comparison['verdict']}")

        assert comparison["verdict"] == "PASS", f"Divergence detected: {comparison}"

    def test_bisection_medium_prompt(self, medium_prompt):
        """Run bisection on medium prompt (~128 tokens)."""
        print(f"\n{'='*70}")
        print("PREFILL BISECTION TEST - MEDIUM PROMPT (~128 tokens)")
        print(f"VLLM_PREFILL_EQ_DEBUG: {os.environ.get('VLLM_PREFILL_EQ_DEBUG', '0')}")
        print(f"{'='*70}\n")

        # Run PyTorch path
        print("Running PyTorch-MPS reference...")
        pytorch_result = self._run_inference_with_checkpoints(
            medium_prompt, use_engine=False, engine_prefill=False
        )
        print(f"PyTorch top-1: {pytorch_result['top1_id']} (logit={pytorch_result['top1_logit']:.4f})")
        print(f"Token count: {pytorch_result['token_count']}")

        # Run Engine path
        print("\nRunning Engine path...")
        engine_result = self._run_inference_with_checkpoints(
            medium_prompt, use_engine=True, engine_prefill=True
        )
        print(f"Engine top-1: {engine_result['top1_id']} (logit={engine_result['top1_logit']:.4f})")

        # Compare
        comparison = self._compare_results(pytorch_result, engine_result)

        print(f"\n--- Comparison ---")
        print(f"Top-1 match: {comparison['top1_match']}")
        print(f"Top-5 overlap: {comparison['top5_overlap']}/5")
        print(f"Logit diff: {comparison['logit_diff']:.6f}")
        print(f"Max logprob diff: {comparison['max_logprob_diff']:.6f}")

        print(f"\nVERDICT: {comparison['verdict']}")

        assert comparison["verdict"] == "PASS", f"Divergence detected: {comparison}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

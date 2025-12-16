#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test suite for Engine vs PyTorch prefill equivalence.

This test verifies that the Metal engine prefill path produces identical
logits to the PyTorch-MPS reference path.

Usage:
    pytest tests/test_engine_pytorch_equivalence.py -v -s

Environment:
    VLLM_ALLOW_INSECURE_SERIALIZATION=1
    VLLM_METAL_SCRATCH_POOL_MB=8192
    VLLM_METAL_MAX_BATCH_SIZE=16384
"""

import os
import sys
import subprocess
import json
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pytest

# Constants
MODEL_NAME = "mistralai/Devstral-Small-2505"
# CRITICAL: Use float16 for equivalence testing.
# bfloat16 is model's native dtype but MPS has limited bfloat16 support,
# and engine uses float16 throughout. Standardize to float16 for fair comparison.
DTYPE = "float16"
ROPE_THETA = 1_000_000_000.0  # 1e9 for Devstral

# Acceptance thresholds (initial, will tighten after measurement)
MAX_DIFF_THRESHOLD = 1e-2  # Start lenient, tighten after finding fix
MEAN_DIFF_THRESHOLD = 1e-3
TOP1_MATCH_REQUIRED = True
TOP5_MATCH_REQUIRED = True


@dataclass
class LogitsComparison:
    """Result of comparing logits between two paths."""
    prompt_name: str
    token_count: int
    last_pos: int
    dtype: str

    # PyTorch results
    pytorch_top1_id: int
    pytorch_top1_logit: float
    pytorch_top5_ids: List[int]
    pytorch_top10: List[Tuple[int, float]]

    # Engine results
    engine_top1_id: int
    engine_top1_logit: float
    engine_top5_ids: List[int]
    engine_top10: List[Tuple[int, float]]

    # Comparison metrics
    top1_match: bool
    top5_overlap: int
    max_diff: float
    mean_diff: float

    # Input details for debugging
    input_ids_last5: List[int]
    position_ids_last5: List[int]
    chunking_enabled: bool

    def passed(self) -> bool:
        """Check if comparison passed acceptance criteria."""
        return (
            self.top1_match and
            self.top5_overlap >= 5 and
            self.max_diff < MAX_DIFF_THRESHOLD and
            self.mean_diff < MEAN_DIFF_THRESHOLD
        )

    def report(self) -> str:
        """Generate detailed report string."""
        lines = [
            f"\n{'='*70}",
            f"PROMPT: {self.prompt_name} ({self.token_count} tokens)",
            f"{'='*70}",
            f"Last position: {self.last_pos}",
            f"Dtype: {self.dtype}",
            f"Chunking enabled: {self.chunking_enabled}",
            f"Input IDs (last 5): {self.input_ids_last5}",
            f"Position IDs (last 5): {self.position_ids_last5}",
            "",
            f"PyTorch top-1: {self.pytorch_top1_id} (logit={self.pytorch_top1_logit:.4f})",
            f"Engine  top-1: {self.engine_top1_id} (logit={self.engine_top1_logit:.4f})",
            f"TOP-1 MATCH: {'PASS' if self.top1_match else 'FAIL'}",
            "",
            f"PyTorch top-5: {self.pytorch_top5_ids}",
            f"Engine  top-5: {self.engine_top5_ids}",
            f"TOP-5 OVERLAP: {self.top5_overlap}/5 {'PASS' if self.top5_overlap >= 5 else 'FAIL'}",
            "",
            f"max|diff|: {self.max_diff:.6f} {'PASS' if self.max_diff < MAX_DIFF_THRESHOLD else 'FAIL'}",
            f"mean|diff|: {self.mean_diff:.6f} {'PASS' if self.mean_diff < MEAN_DIFF_THRESHOLD else 'FAIL'}",
            "",
            "PyTorch top-10:",
        ]
        for i, (tok_id, logit) in enumerate(self.pytorch_top10):
            lines.append(f"  [{i}] {tok_id}: {logit:.4f}")
        lines.append("")
        lines.append("Engine top-10:")
        for i, (tok_id, logit) in enumerate(self.engine_top10):
            lines.append(f"  [{i}] {tok_id}: {logit:.4f}")
        lines.append("")
        lines.append(f"VERDICT: {'PASS' if self.passed() else 'FAIL'}")
        lines.append("="*70)
        return "\n".join(lines)


def create_prompt_with_exact_tokens(tokenizer, target_length: int, base_text: str) -> str:
    """Create a prompt with exactly target_length tokens.

    Uses binary search to find the right character count.
    """
    # Extend base text if needed
    extended = base_text * ((target_length * 10 // len(base_text)) + 1)

    # Binary search for exact token count
    low, high = target_length, len(extended)
    best_prompt = ""
    best_diff = float('inf')

    while low <= high:
        mid = (low + high) // 2
        candidate = extended[:mid]
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        diff = abs(len(tokens) - target_length)

        if diff < best_diff:
            best_diff = diff
            best_prompt = candidate

        if len(tokens) < target_length:
            low = mid + 1
        elif len(tokens) > target_length:
            high = mid - 1
        else:
            # Exact match
            return candidate

    # Trim or pad to get exact length
    tokens = tokenizer.encode(best_prompt, add_special_tokens=False)
    if len(tokens) > target_length:
        # Decode subset
        best_prompt = tokenizer.decode(tokens[:target_length])

    return best_prompt


def get_test_prompts(tokenizer) -> Dict[str, str]:
    """Generate test prompts with exact token lengths."""
    base_texts = {
        "short_32": "The quick brown fox jumps over the lazy dog. ",
        "medium_128": "The quick brown fox jumps over the lazy dog. ",
        "long_512": "The quick brown fox jumps over the lazy dog. ",
        "fox_500": "The quick brown fox jumps over the lazy dog. ",  # Known failing
        "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n# ",
        "multilingual": "Hello world. Bonjour le monde. Hola mundo. Guten Tag. ",
    }

    target_lengths = {
        "short_32": 32,
        "medium_128": 128,
        "long_512": 512,
        "fox_500": 500,
        "code": 300,
        "multilingual": 200,
    }

    prompts = {}
    for name, base in base_texts.items():
        target = target_lengths[name]
        prompts[name] = create_prompt_with_exact_tokens(tokenizer, target, base)
        actual = len(tokenizer.encode(prompts[name], add_special_tokens=False))
        print(f"Prompt '{name}': target={target}, actual={actual} tokens")

    return prompts


def run_prefill_subprocess(prompt: str, use_engine: bool, engine_prefill: bool) -> Dict:
    """Run prefill in subprocess and return logits info.

    We use subprocess to ensure clean environment state between runs.
    """
    # Escape prompt carefully for embedding in Python string
    escaped_prompt = prompt.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

    script = f'''
import os
import sys
import json
import torch

# Set environment BEFORE imports
os.environ["VLLM_APPLE_USE_ENGINE"] = "{'1' if use_engine else '0'}"
os.environ["VLLM_APPLE_ENGINE_PREFILL"] = "{'1' if engine_prefill else '0'}"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_METAL_SCRATCH_POOL_MB"] = "8192"
os.environ["VLLM_METAL_MAX_BATCH_SIZE"] = "16384"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # Required for subprocess execution
# Disable chunked prefill for consistency (if it exists)
os.environ.setdefault("VLLM_DISABLE_CHUNKED_PREFILL", "0")

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
last_pos = token_count - 1

# Log diagnostics to stderr (not stdout where JSON goes)
print(f"[DIAG] dtype={'{DTYPE}'}, use_engine={'1' if use_engine else '0'}, engine_prefill={'1' if engine_prefill else '0'}", file=sys.stderr)
print(f"[DIAG] token_count={{token_count}}, last_pos={{last_pos}}", file=sys.stderr)
print(f"[DIAG] input_ids[-5:]={{input_ids[-5:]}}", file=sys.stderr)
print(f"[DIAG] position_ids[-5:]={{list(range(max(0, token_count-5), token_count))}}", file=sys.stderr)

# Run generation to get logits (single token)
params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,  # Disable top_k
    max_tokens=1,
    seed=42,
    logprobs=20,  # Request top-20 logprobs for comparison
)

outputs = llm.generate([prompt], params, use_tqdm=False)
output = outputs[0]

# Get the sampled token and its logit
sampled_token = output.outputs[0].token_ids[0] if output.outputs[0].token_ids else -1

# Get logprobs if available
logprobs = output.outputs[0].logprobs
if logprobs and len(logprobs) > 0:
    first_logprobs = logprobs[0]
    # Sort by logprob (descending)
    sorted_logprobs = sorted(first_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
    top20 = [(tok_id, lp.logprob) for tok_id, lp in sorted_logprobs[:20]]
    top10 = top20[:10]
    top5_ids = [tok_id for tok_id, _ in top10[:5]]
    top1_id = top10[0][0]
    top1_logit = top10[0][1]
    print(f"[DIAG] top1_id={{top1_id}}, top1_logit={{top1_logit:.4f}}", file=sys.stderr)
    print(f"[DIAG] top5_ids={{top5_ids}}", file=sys.stderr)
else:
    # Fallback: use sampled token
    print("[DIAG] WARNING: No logprobs returned, using sampled token as fallback", file=sys.stderr)
    top10 = [(sampled_token, 0.0)]
    top5_ids = [sampled_token]
    top1_id = sampled_token
    top1_logit = 0.0

# Check chunking status (from env var at runtime)
chunking_enabled = os.environ.get("VLLM_APPLE_ENGINE_PREFILL") == "1"

result = {{
    "token_count": token_count,
    "last_pos": last_pos,
    "dtype": "{DTYPE}",
    "top1_id": top1_id,
    "top1_logit": float(top1_logit),
    "top5_ids": top5_ids,
    "top10": top10,
    "input_ids_last5": input_ids[-5:],
    "position_ids_last5": list(range(max(0, token_count-5), token_count)),
    "chunking_enabled": chunking_enabled,
    "use_engine": {'true' if use_engine else 'false'},
    "engine_prefill": {'true' if engine_prefill else 'false'},
}}

print("RESULT_JSON:" + json.dumps(result))
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        env = os.environ.copy()
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        # Parse output
        for line in result.stdout.split('\n'):
            if line.startswith("RESULT_JSON:"):
                json_str = line[len("RESULT_JSON:"):]
                return json.loads(json_str)

        # If no result found, print error info
        print(f"STDOUT: {result.stdout[-2000:]}")
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"No result found in subprocess output")

    finally:
        os.unlink(script_path)


def compare_prefill(prompt_name: str, prompt: str) -> LogitsComparison:
    """Compare prefill outputs between PyTorch and Engine paths."""

    print(f"\n--- Testing prompt: {prompt_name} ---")

    # Run PyTorch reference (no engine)
    print("Running PyTorch-MPS reference...")
    pytorch_result = run_prefill_subprocess(prompt, use_engine=False, engine_prefill=False)

    # Run Engine prefill
    print("Running Engine prefill...")
    engine_result = run_prefill_subprocess(prompt, use_engine=True, engine_prefill=True)

    # Compare
    top1_match = pytorch_result["top1_id"] == engine_result["top1_id"]

    pytorch_top5 = set(pytorch_result["top5_ids"])
    engine_top5 = set(engine_result["top5_ids"])
    top5_overlap = len(pytorch_top5 & engine_top5)

    # Compute logit differences (using available top-10)
    pytorch_logits = {tok_id: logit for tok_id, logit in pytorch_result["top10"]}
    engine_logits = {tok_id: logit for tok_id, logit in engine_result["top10"]}

    # Find common tokens to compare
    common_tokens = set(pytorch_logits.keys()) & set(engine_logits.keys())
    if common_tokens:
        diffs = [abs(pytorch_logits[t] - engine_logits[t]) for t in common_tokens]
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
    else:
        # No common tokens - big divergence
        max_diff = float('inf')
        mean_diff = float('inf')

    return LogitsComparison(
        prompt_name=prompt_name,
        token_count=pytorch_result["token_count"],
        last_pos=pytorch_result["last_pos"],
        dtype=pytorch_result["dtype"],
        pytorch_top1_id=pytorch_result["top1_id"],
        pytorch_top1_logit=pytorch_result["top1_logit"],
        pytorch_top5_ids=pytorch_result["top5_ids"],
        pytorch_top10=pytorch_result["top10"],
        engine_top1_id=engine_result["top1_id"],
        engine_top1_logit=engine_result["top1_logit"],
        engine_top5_ids=engine_result["top5_ids"],
        engine_top10=engine_result["top10"],
        top1_match=top1_match,
        top5_overlap=top5_overlap,
        max_diff=max_diff,
        mean_diff=mean_diff,
        input_ids_last5=pytorch_result["input_ids_last5"],
        position_ids_last5=pytorch_result["position_ids_last5"],
        chunking_enabled=engine_result["chunking_enabled"],
    )


class TestEnginePyTorchEquivalence:
    """Test class for engine vs PyTorch equivalence."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load tokenizer once for all tests."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    @pytest.fixture(scope="class")
    def test_prompts(self, tokenizer):
        """Generate test prompts."""
        return get_test_prompts(tokenizer)

    def test_short_32(self, test_prompts):
        """Test short 32-token prompt."""
        result = compare_prefill("short_32", test_prompts["short_32"])
        print(result.report())
        assert result.passed(), f"short_32 failed: {result.report()}"

    def test_medium_128(self, test_prompts):
        """Test medium 128-token prompt."""
        result = compare_prefill("medium_128", test_prompts["medium_128"])
        print(result.report())
        assert result.passed(), f"medium_128 failed: {result.report()}"

    def test_long_512(self, test_prompts):
        """Test long 512-token prompt."""
        result = compare_prefill("long_512", test_prompts["long_512"])
        print(result.report())
        assert result.passed(), f"long_512 failed: {result.report()}"

    def test_fox_500(self, test_prompts):
        """Test known-failing fox 500-token prompt."""
        result = compare_prefill("fox_500", test_prompts["fox_500"])
        print(result.report())
        assert result.passed(), f"fox_500 failed: {result.report()}"

    def test_code(self, test_prompts):
        """Test code prompt."""
        result = compare_prefill("code", test_prompts["code"])
        print(result.report())
        assert result.passed(), f"code failed: {result.report()}"

    def test_multilingual(self, test_prompts):
        """Test multilingual prompt."""
        result = compare_prefill("multilingual", test_prompts["multilingual"])
        print(result.report())
        assert result.passed(), f"multilingual failed: {result.report()}"


def run_all_comparisons():
    """Run all comparisons and print summary."""
    from transformers import AutoTokenizer

    print("="*70)
    print("ENGINE vs PYTORCH PREFILL EQUIVALENCE TEST")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Dtype: {DTYPE}")
    print(f"RoPE theta: {ROPE_THETA}")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompts = get_test_prompts(tokenizer)

    results = []
    for name, prompt in prompts.items():
        result = compare_prefill(name, prompt)
        results.append(result)
        print(result.report())

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r.passed())
    total = len(results)

    print(f"Passed: {passed}/{total}")
    print()

    for r in results:
        status = "PASS" if r.passed() else "FAIL"
        print(f"  {r.prompt_name}: {status} (top1={r.top1_match}, top5={r.top5_overlap}/5, max_diff={r.max_diff:.6f})")

    print()
    if passed == total:
        print("OVERALL: PASS")
        return 0
    else:
        print("OVERALL: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_comparisons())

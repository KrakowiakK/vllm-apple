#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare decode logits between Engine and PyTorch paths.

This script captures and compares actual logit values for the first decode step
to identify numerical divergence between Engine and PyTorch.
"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np

MODEL_NAME = "mistralai/Devstral-Small-2505"
DTYPE = "float16"


def run_decode_logits(prompt: str, use_engine: bool) -> dict:
    """Run inference and capture logits for first decode step."""

    script = f'''
import os
import sys
import json
import torch

# Set environment BEFORE imports
os.environ["VLLM_APPLE_USE_ENGINE"] = "{'1' if use_engine else '0'}"
os.environ["VLLM_APPLE_ENGINE_PREFILL"] = "{'1' if use_engine else '0'}"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_METAL_SCRATCH_POOL_MB"] = "8192"
os.environ["VLLM_METAL_MAX_BATCH_SIZE"] = "16384"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="{MODEL_NAME}",
    trust_remote_code=True,
    enforce_eager=True,
    dtype="{DTYPE}",
    max_model_len=8192,
)
tokenizer = llm.get_tokenizer()

# Encode prompt
prompt = {repr(prompt)}
input_ids = tokenizer.encode(prompt, add_special_tokens=False)
token_count = len(input_ids)

# Hook to capture logits
captured_logits = {{}}

# Patch the model runner to capture logits
original_sample = llm.llm_engine.model_executor.driver_worker.model_runner._sample_tokens

def patched_sample(hidden_states, model_input, scheduler_output, is_logits=False):
    step_num = len(captured_logits) + 1

    if is_logits:
        logits = hidden_states
    else:
        # Extract logits manually
        logits_hidden = hidden_states[-model_input.num_reqs:] if hidden_states.dim() == 2 else hidden_states
        if hasattr(llm.llm_engine.model_executor.driver_worker.model_runner.model, 'compute_logits'):
            logits = llm.llm_engine.model_executor.driver_worker.model_runner.model.compute_logits(logits_hidden)
        else:
            logits = logits_hidden

    # Capture logits for this step (detach and move to CPU)
    captured_logits[step_num] = {{
        "shape": list(logits.shape),
        "logits_slice": logits[0, :100].detach().cpu().numpy().tolist() if logits.numel() > 0 else [],
        "top_5_indices": logits[0].topk(5).indices.cpu().numpy().tolist() if logits.numel() > 0 else [],
        "top_5_values": logits[0].topk(5).values.cpu().numpy().tolist() if logits.numel() > 0 else [],
    }}

    return original_sample(hidden_states, model_input, scheduler_output, is_logits)

llm.llm_engine.model_executor.driver_worker.model_runner._sample_tokens = patched_sample

# Run generation
params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=3,
    seed=42,
)

outputs = llm.generate([prompt], params, use_tqdm=False)
output = outputs[0]

output_tokens = list(output.outputs[0].token_ids)

# Get top-5 for specific tokens 1115 and 2136
all_logits = {{}}
for step_num, logits_info in captured_logits.items():
    if logits_info.get("logits_slice"):
        top5 = list(zip(logits_info["top_5_indices"], logits_info["top_5_values"]))
        all_logits[step_num] = {{
            "top_5": top5,
        }}

result = {{
    "mode": "engine" if {use_engine} else "pytorch",
    "input_token_count": token_count,
    "output_tokens": output_tokens,
    "captured_logits": all_logits,
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

        for line in result.stdout.split('\n'):
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[12:])

        print(f"STDERR:\n{result.stderr[-2000:]}")
        return {"error": "No result", "stderr": result.stderr[-500:]}
    finally:
        os.unlink(script_path)


def main():
    # Use the same prompt as the divergence test
    base = "The quick brown fox jumps over the lazy dog. "
    estimated_chars = 500 * 4  # Target ~500 tokens
    repetitions = (estimated_chars // len(base)) + 1
    prompt = (base * repetitions)[:estimated_chars]

    print("=" * 70)
    print("DECODE LOGITS COMPARISON")
    print("=" * 70)

    print("\n[1] Running PyTorch path...")
    pytorch_result = run_decode_logits(prompt, use_engine=False)

    if "error" in pytorch_result:
        print(f"ERROR: {pytorch_result}")
        return 1

    print(f"Output tokens: {pytorch_result['output_tokens']}")
    print(f"Captured {len(pytorch_result['captured_logits'])} steps")

    print("\n[2] Running Engine path...")
    engine_result = run_decode_logits(prompt, use_engine=True)

    if "error" in engine_result:
        print(f"ERROR: {engine_result}")
        return 1

    print(f"Output tokens: {engine_result['output_tokens']}")
    print(f"Captured {len(engine_result['captured_logits'])} steps")

    # Compare logits
    print("\n" + "=" * 70)
    print("LOGITS COMPARISON")
    print("=" * 70)

    for step in sorted(set(pytorch_result.get('captured_logits', {}).keys()) &
                       set(engine_result.get('captured_logits', {}).keys())):
        pt_logits = pytorch_result['captured_logits'].get(step, {})
        en_logits = engine_result['captured_logits'].get(step, {})

        print(f"\nStep {step}:")
        if pt_logits.get('top_5'):
            print(f"  PyTorch top-5: {pt_logits['top_5']}")
        if en_logits.get('top_5'):
            print(f"  Engine top-5:  {en_logits['top_5']}")

        # Check if top-1 differs
        if pt_logits.get('top_5') and en_logits.get('top_5'):
            pt_top1 = pt_logits['top_5'][0][0]
            en_top1 = en_logits['top_5'][0][0]
            if pt_top1 != en_top1:
                print(f"  >>> TOP-1 MISMATCH: PyTorch={pt_top1}, Engine={en_top1}")
                # Show logit difference for the two candidates
                pt_dict = {t[0]: t[1] for t in pt_logits['top_5']}
                en_dict = {t[0]: t[1] for t in en_logits['top_5']}
                if pt_top1 in en_dict:
                    print(f"      Token {pt_top1}: PyTorch={pt_dict.get(pt_top1, 'N/A'):.4f}, Engine={en_dict.get(pt_top1, 'N/A'):.4f}")
                if en_top1 in pt_dict:
                    print(f"      Token {en_top1}: PyTorch={pt_dict.get(en_top1, 'N/A'):.4f}, Engine={en_dict.get(en_top1, 'N/A'):.4f}")
            else:
                print(f"  ✓ Top-1 match: {pt_top1}")

    # Final verdict
    print("\n" + "=" * 70)
    if pytorch_result['output_tokens'] == engine_result['output_tokens']:
        print("VERDICT: PASS - Outputs match")
        return 0
    else:
        print("VERDICT: FAIL - Outputs diverge")
        print(f"  PyTorch: {pytorch_result['output_tokens']}")
        print(f"  Engine:  {engine_result['output_tokens']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

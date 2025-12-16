#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Debug script to trace decode divergence by capturing step-by-step details.

This script captures and compares the exact values being passed during:
1. Prefill step
2. First decode step

Usage:
    python scripts/debug_decode_divergence.py
"""

import os
import sys
import json
import subprocess
import tempfile

MODEL_NAME = "mistralai/Devstral-Small-2505"
DTYPE = "float16"


def run_with_debug(prompt: str, max_tokens: int, use_engine: bool) -> dict:
    """Run inference with detailed debug logging."""

    script = f'''
import os
import sys
import json

# Set environment BEFORE imports
os.environ["VLLM_APPLE_USE_ENGINE"] = "{'1' if use_engine else '0'}"
os.environ["VLLM_APPLE_ENGINE_PREFILL"] = "{'1' if use_engine else '0'}"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_METAL_SCRATCH_POOL_MB"] = "8192"
os.environ["VLLM_METAL_MAX_BATCH_SIZE"] = "16384"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Patch the model runner to capture step details
_step_details = []

from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="{MODEL_NAME}",
    trust_remote_code=True,
    enforce_eager=True,
    dtype="{DTYPE}",
    max_model_len=8192,
)

# Patch to capture attention metadata
original_execute_model = llm.llm_engine.model_executor.driver_worker.model_runner.execute_model

def patched_execute_model(scheduler_output, *args, **kwargs):
    """Capture step details before executing."""
    runner = llm.llm_engine.model_executor.driver_worker.model_runner

    # Build model input to get attn_metadata
    model_input = runner._prepare_model_input(scheduler_output)

    step_info = {{
        "step_num": len(_step_details) + 1,
        "num_scheduled_tokens": model_input.num_scheduled_tokens,
        "num_reqs": model_input.num_reqs,
        "req_ids": model_input.req_ids,
        "seq_lens_query": model_input.seq_lens,  # Query lengths (new tokens per request)
        "input_ids": model_input.input_ids.tolist()[:10],  # First 10 for brevity
        "positions": model_input.positions.tolist()[-5:] if len(model_input.positions) > 0 else [],  # Last 5
    }}

    attn = model_input.attn_metadata
    step_info["attn"] = {{
        "num_actual_tokens": attn.num_actual_tokens,
        "num_decode_tokens": attn.num_decode_tokens,
        "max_query_len": attn.max_query_len,
        "max_seq_len": attn.max_seq_len,
        "seq_lens_full": attn.seq_lens.tolist() if attn.seq_lens is not None else [],  # Full seq lens
    }}

    # Check request states
    for req_id in model_input.req_ids[:1]:  # First request only
        if req_id in runner.requests:
            req_state = runner.requests[req_id]
            step_info["req_state"] = {{
                "num_computed_tokens": req_state.num_computed_tokens,
                "output_token_count": len(req_state.output_token_ids),
            }}

    _step_details.append(step_info)

    return original_execute_model(scheduler_output, *args, **kwargs)

llm.llm_engine.model_executor.driver_worker.model_runner.execute_model = patched_execute_model

# Run generation
tokenizer = llm.get_tokenizer()
prompt = {repr(prompt)}
input_ids = tokenizer.encode(prompt, add_special_tokens=False)

params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens={max_tokens},
    seed=42,
    logprobs=5,
)

outputs = llm.generate([prompt], params, use_tqdm=False)
output = outputs[0]

# Collect results
output_tokens = list(output.outputs[0].token_ids)
result = {{
    "mode": "engine" if {use_engine} else "pytorch",
    "input_token_count": len(input_ids),
    "output_tokens": output_tokens[:10],  # First 10
    "output_token_count": len(output_tokens),
    "step_details": _step_details,
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
    # Create a prompt that will cause divergence (around 445 tokens)
    # Use the EXACT same prompt generation as repro_prefill_divergence.py
    base = "The quick brown fox jumps over the lazy dog. "
    estimated_chars = 500 * 4  # Target ~500 tokens
    repetitions = (estimated_chars // len(base)) + 1
    prompt = (base * repetitions)[:estimated_chars]

    print("=" * 70)
    print("DEBUG DECODE DIVERGENCE")
    print("=" * 70)

    # Run PyTorch path
    print("\n[1] Running PyTorch path with 3 output tokens...")
    pytorch_result = run_with_debug(prompt, max_tokens=3, use_engine=False)

    if "error" in pytorch_result:
        print(f"ERROR: {pytorch_result}")
        return 1

    print(f"Input tokens: {pytorch_result['input_token_count']}")
    print(f"Output tokens: {pytorch_result['output_tokens']}")
    print(f"\nStep details:")
    for step in pytorch_result['step_details'][:3]:  # First 3 steps
        print(f"  Step {step['step_num']}:")
        print(f"    num_scheduled_tokens: {step['num_scheduled_tokens']}")
        print(f"    seq_lens_query: {step['seq_lens_query']}")
        print(f"    positions (last 5): {step['positions']}")
        print(f"    attn.seq_lens_full: {step['attn']['seq_lens_full']}")
        print(f"    attn.num_decode_tokens: {step['attn']['num_decode_tokens']}")
        if 'req_state' in step:
            print(f"    req_state.num_computed_tokens: {step['req_state']['num_computed_tokens']}")

    # Run Engine path
    print("\n[2] Running Engine path with 3 output tokens...")
    engine_result = run_with_debug(prompt, max_tokens=3, use_engine=True)

    if "error" in engine_result:
        print(f"ERROR: {engine_result}")
        return 1

    print(f"Input tokens: {engine_result['input_token_count']}")
    print(f"Output tokens: {engine_result['output_tokens']}")
    print(f"\nStep details:")
    for step in engine_result['step_details'][:3]:
        print(f"  Step {step['step_num']}:")
        print(f"    num_scheduled_tokens: {step['num_scheduled_tokens']}")
        print(f"    seq_lens_query: {step['seq_lens_query']}")
        print(f"    positions (last 5): {step['positions']}")
        print(f"    attn.seq_lens_full: {step['attn']['seq_lens_full']}")
        print(f"    attn.num_decode_tokens: {step['attn']['num_decode_tokens']}")
        if 'req_state' in step:
            print(f"    req_state.num_computed_tokens: {step['req_state']['num_computed_tokens']}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"PyTorch output: {pytorch_result['output_tokens']}")
    print(f"Engine output:  {engine_result['output_tokens']}")

    if pytorch_result['output_tokens'] == engine_result['output_tokens']:
        print("\n✓ PASS - Outputs match")
        return 0
    else:
        print("\n✗ FAIL - Outputs diverge")

        # Show step-by-step comparison
        pt_steps = pytorch_result['step_details']
        en_steps = engine_result['step_details']

        for i in range(min(3, len(pt_steps), len(en_steps))):
            print(f"\nStep {i+1} comparison:")
            pt = pt_steps[i]
            en = en_steps[i]

            pt_seqlen = pt['attn']['seq_lens_full']
            en_seqlen = en['attn']['seq_lens_full']

            if pt_seqlen != en_seqlen:
                print(f"  ✗ seq_lens_full DIFFER: PyTorch={pt_seqlen}, Engine={en_seqlen}")
            else:
                print(f"  ✓ seq_lens_full match: {pt_seqlen}")

            pt_pos = pt['positions']
            en_pos = en['positions']
            if pt_pos != en_pos:
                print(f"  ✗ positions DIFFER: PyTorch={pt_pos}, Engine={en_pos}")
            else:
                print(f"  ✓ positions match: {pt_pos}")

        return 1


if __name__ == "__main__":
    sys.exit(main())

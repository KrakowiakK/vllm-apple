#!/bin/bash
export MODEL_NAME=Qwen/Qwen2-0.5B
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
PYTHON=./.venv/bin/python

# Verification Run: 551 tokens (previously crashed)
echo "Running Engine Prefill with 551 tokens..."
VLLM_APPLE_USE_ENGINE=1 VLLM_APPLE_ENGINE_PREFILL=1 TARGET_LEN=551 $PYTHON diagnose_worker.py > res_verify_fix.json
cat res_verify_fix.json

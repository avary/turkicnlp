#!/usr/bin/env bash
# Start a local OpenAI-compatible LLM server with vLLM for the TurkicNLP LLM backend.
#
#   bash scripts/start_vllm_server.sh                       # Qwen/Qwen2.5-1.5B-Instruct on port 8000
#   MODEL=Qwen/Qwen3-8B PORT=8001 bash scripts/start_vllm_server.sh
#   GPUS=0,1 TP=2 MODEL=Qwen/Qwen3-32B bash scripts/start_vllm_server.sh
#
# Requirements: Linux with an NVIDIA GPU (a 1.5B model needs ~6 GB of GPU memory) and
#   pip install vllm          # use its own environment; vLLM pins its own torch version
# On macOS or without a GPU use Ollama instead (see the README, "LLM Backend").
#
# Then, in Python:
#   nlp = turkicnlp.Pipeline("kaz", processors=["tokenize", "ner"], ner_backend="llm",
#                            llm_model="Qwen/Qwen2.5-1.5B-Instruct",
#                            llm_base_url="http://localhost:8000/v1", llm_api_key="turkicnlp-local")
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"   # any chat model on the Hugging Face Hub
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-turkicnlp-local}"         # clients must send this as llm_api_key
MAX_LEN="${MAX_LEN:-8192}"                     # prompts of the toolkit are short; 8k is plenty
GPU_UTIL="${GPU_UTIL:-0.85}"
TP="${TP:-1}"                                  # tensor parallel size (number of GPUs)
[ -n "${GPUS:-}" ] && export CUDA_VISIBLE_DEVICES="$GPUS"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed. Install it in its own environment:  pip install vllm" >&2
  exit 1
fi

echo "Serving $MODEL on http://$HOST:$PORT/v1 (API key: $API_KEY)"
exec vllm serve "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --api-key "$API_KEY" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size "$TP"

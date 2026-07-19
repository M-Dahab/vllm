#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_ID="${MODEL_ID:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen36-vllm}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-10}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-modelopt}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-throughput}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

ARGS=(
  "$MODEL_ID"
  --served-model-name "$MODEL_ID"
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --dtype "$DTYPE"
  --quantization "$QUANTIZATION"
  --tensor-parallel-size 1
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --block-size "$BLOCK_SIZE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --attention-backend "$ATTENTION_BACKEND"
  --performance-mode "$PERFORMANCE_MODE"
  --enforce-eager
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
  --reasoning-parser qwen3
)
if [[ "$LOAD_FORMAT" != "auto" ]]; then
  ARGS+=(--load-format "$LOAD_FORMAT")
fi

echo "Stopping old container if present: $CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting $IMAGE on port $PORT with model $MODEL_ID"
echo "vLLM args: ${ARGS[*]}"

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --restart "$RESTART_POLICY" \
  --network host \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface \
  -e CUDA_MODULE_LOADING=LAZY \
  -v /home/mohammad/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-cache-qwen36:/root/.cache/vllm \
  "$IMAGE" \
  "${ARGS[@]}"

echo "Container started. Follow logs with: docker logs -f $CONTAINER_NAME"
echo "Health: curl -s http://localhost:$PORT/v1/models | jq"
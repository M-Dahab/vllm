#!/usr/bin/env bash
set -Eeuo pipefail

# Optimized vLLM runner for google/gemma-4-12B-it on DGX Spark / GB10.
#
# Usage:
#   ./run-gemma4-12b-vllm.sh                                # max-context default
#   MODE=throughput ./run-gemma4-12b-vllm.sh                # 32k, higher parallel throughput
#   MAX_MODEL_LEN=131072 MAX_NUM_SEQS=4 ./run-gemma4-12b-vllm.sh
#   GPU_MEMORY_UTILIZATION=0.92 ./run-gemma4-12b-vllm.sh   # tighter VRAM headroom for 262k

MODEL_ID="${MODEL_ID:-google/gemma-4-12B-it}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-/home/mohammad/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205}"
CONTAINER_NAME="${CONTAINER_NAME:-gemma4-12b-vllm}"
IMAGE="${IMAGE:-vllm/vllm-openai:gemma4-unified}"
PORT="${PORT:-8000}"
MODE="${MODE:-max-context}"

# Per-mode presets. Everything is still env-overridable on the command line.
BLOCK_SIZE="${BLOCK_SIZE:-128}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
DTYPE="${DTYPE:-auto}"
# Gemma4 unified has partial multimodal-token full attention; flashinfer rejects that config.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-throughput}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"

# Safer 262K defaults — the previous 0.94 OOMed on this GB10.
# 0.92 leaves more headroom for KV cache at 262k and avoids a CUDA OOM loop.
case "$MODE" in
  max-context|opencode|long-context)
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
    # Keep the advertised context at 262K, but avoid profiling/processing
    # a giant full-context prefill batch at once. 65K improves stability and
    # scheduler flexibility for mixed opencode/coding-agent requests.
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
    ;;
  latency)
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
    PERFORMANCE_MODE="${PERFORMANCE_MODE:-latency}"
    ;;
  throughput)
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
    ;;
  max-throughput)
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
    ;;
  *)
    echo "ERROR: Unknown MODE=$MODE. Use max-context, opencode, long-context, latency, throughput, or max-throughput." >&2
    exit 1
    ;;
esac

if [[ ! -f "$HOST_MODEL_PATH/config.json" ]]; then
  echo "ERROR: HOST_MODEL_PATH does not contain config.json: $HOST_MODEL_PATH" >&2
  echo "Set HOST_MODEL_PATH=/path/to/local/snapshot" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found" >&2
  exit 1
fi

echo "Stopping old container if present: $CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

ARGS=(
  "$MODEL_PATH"
  --served-model-name "$MODEL_ID"
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --dtype "$DTYPE"
  --tensor-parallel-size 1
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --block-size "$BLOCK_SIZE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --attention-backend "$ATTENTION_BACKEND"
  --performance-mode "$PERFORMANCE_MODE"
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --reasoning-parser gemma4
)

if [[ "$LOAD_FORMAT" != "auto" ]]; then
  ARGS+=(--load-format "$LOAD_FORMAT")
fi
if [[ -n "$SPECULATIVE_CONFIG" ]]; then
  ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
fi

echo "Starting $IMAGE on port $PORT"
echo "Model snapshot: $HOST_MODEL_PATH"
echo "vLLM args: ${ARGS[*]}"

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --restart unless-stopped \
  --network host \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_ATTENTION_BACKEND="$ATTENTION_BACKEND" \
  -e CUDA_MODULE_LOADING=LAZY \
  -v /home/mohammad/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \
  "$IMAGE" \
  vllm serve \
  "${ARGS[@]}"

echo "Container started. Follow logs with: docker logs -f $CONTAINER_NAME"
echo "Health: curl -s http://localhost:$PORT/v1/models | jq"

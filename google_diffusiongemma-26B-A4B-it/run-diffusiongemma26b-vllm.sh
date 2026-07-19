#!/usr/bin/env bash
set -Eeuo pipefail

# Optimized vLLM runner for google/diffusiongemma-26B-A4B-it on DGX Spark / GB10.
#
# Architecture: DiffusionGemma — encoder-decoder, discrete diffusion text generation.
#   - 25.2B total params, 3.8B active (8/128 MoE + 1 shared)
#   - Canvas length: 256 tokens per diffusion block
#   - Multimodal: text + image (variable aspect ratio/resolution)
#   - Context: up to 256K tokens
#   - Weights: ~53GB safetensors (11 shards, FP16)
#
# Diffusion-based text generation (NOT autoregressive):
#   Encoder caches prompt context in KV cache.
#   Decoder iteratively denoises a full 256-token canvas via bidirectional attention,
#   accessing cached context through cross-attention.
#   Adaptive early-stop: simpler prompts need fewer denoising steps.
#   Each canvas is fully denoised → encoder processes it → appended to KV cache → next canvas.
#
# Memory profile (GB10, 122GB, gpu-util=0.7 → 85GB usable):
#   Weights: ~50GB (25.2B × 2 bytes)
#   Encoder KV cache: ~10-15GB at 262K context (encoder-decoder, MoE sparse)
#   Diffusion state (canvas + sampler buffers): ~3-5GB per sequence
#   Single-sequence peak at 262K context: ~70-75GB — fits but leaves little headroom.
#   Two concurrent sequences would OOM — design is for low-batch, single-user.
#
# Usage:
#   ./run-diffusiongemma26b-vllm.sh                                # max-context default
#   MAX_MODEL_LEN=32768 ./run-diffusiongemma26b-vllm.sh            # shorter context, less KV cache
#   MAX_NUM_SEQS=4 ./run-diffusiongemma26b-vllm.sh                 # more concurrent sessions
#   GPU_MEMORY_UTILIZATION=0.85 ./run-diffusiongemma26b-vllm.sh    # tighter VRAM headroom

MODEL_ID="${MODEL_ID:-google/diffusiongemma-26B-A4B-it}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/0f28bc42f588fbd8f71e08102b1c3960298a1358}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-/home/mohammad/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/0f28bc42f588fbd8f71e08102b1c3960298a1358}"
CONTAINER_NAME="${CONTAINER_NAME:-diffusiongemma26b-vllm}"
IMAGE="${IMAGE:-vllm/vllm-openai:gemma4-unified}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
DTYPE="${DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-throughput}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

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

echo ""
echo "Starting $IMAGE on port $PORT"
echo "  Model:       $MODEL_ID"
echo "  Snapshot:    $HOST_MODEL_PATH"
echo "  GPU mem:     $GPU_MEMORY_UTILIZATION"
echo "  Context:     $MAX_MODEL_LEN"
echo "  Canvas:      256 tokens (diffusion)"
echo "  Concurrent:  $MAX_NUM_SEQS"
echo "  Args:        ${ARGS:-none}"
echo ""
echo "Note: installing latest transformers inside container before vLLM starts..."
echo ""
echo "Applying vLLM PR #45643 patch for DiffusionGemma transformers fallback..."
echo ""

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
  --enforce-eager
)

if [[ -n "$QUANTIZATION" ]]; then
  ARGS+=(--quantization "$QUANTIZATION")
fi

if [[ "$LOAD_FORMAT" != "auto" ]]; then
  ARGS+=(--load-format "$LOAD_FORMAT")
fi

echo "vLLM args: ${ARGS[*]}"
echo ""

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
  -v vllm-cache-diffusiongemma26b:/root/.cache/vllm \
  "$IMAGE" \
  bash -c 'TARGET=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/transformers/base.py && python3 -c "
import sys
target = \"$TARGET\"
old = \"\"\"        self._decorate_for_torch_compile(**from_config_kwargs)\"\"\"
new = \"\"\"        should_decorate_for_compile = not getattr(
            self.model_config.hf_config,
            \"model_type\",
            None,
        ) == \"diffusion_gemma\"
        if should_decorate_for_compile:
            self._decorate_for_torch_compile(**from_config_kwargs)\"\"\"
src = open(target).read()
if \"should_decorate_for_compile\" in src:
    print(\"[skip] base.py patch already applied\")
elif old not in src:
    print(\"[error] base.py: original code not found\")
    sys.exit(1)
else:
    open(target, \"w\").write(src.replace(old, new))
    print(\"[ok] base.py: patch applied\")
" && python3 -c "
import sys
target = \"/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/transformers/moe.py\"
old = \"\"\"        top_k = getattr_iter(text_config, [\"num_experts_per_tok\", \"top_k\"], None)\"\"\"
new = \"\"\"        top_k = getattr_iter(text_config, [\"num_experts_per_tok\", \"top_k\", \"top_k_experts\"], None)\"\"\"
src = open(target).read()
if \"top_k_experts\" in src:
    print(\"[skip] moe.py patch already applied\")
elif old not in src:
    print(\"[error] moe.py: original code not found\")
    sys.exit(1)
else:
    open(target, \"w\").write(src.replace(old, new))
    print(\"[ok] moe.py: patch applied\")
" && pip install --no-cache-dir -U "transformers>=5.0.0" && exec vllm serve '"${ARGS[*]}"

echo "Container started. Follow logs:   docker logs -f $CONTAINER_NAME"
echo "Health:                            curl -s http://localhost:$PORT/v1/models | jq"
echo ""
echo "Chat (text only):"
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"model\": \"$MODEL_ID\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 100}'"
echo ""
echo "Chat (multimodal — text + image):"
echo "  curl http://localhost:$PORT/v1/chat/completions \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"model\": \"$MODEL_ID\", \"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": {\"url\": \"https://example.com/chart.png\"}}, {\"type\": \"text\", \"text\": \"What does this show?\"}]}], \"max_tokens\": 100}'"

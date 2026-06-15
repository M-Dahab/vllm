# vLLM Serving Configs & Benchmarks for DGX Spark (GB10)

Optimized Docker-based vLLM profiles for serving large language models on NVIDIA **DGX Spark** / **GB10** unified memory.

## Profiles

### [google/gemma-4-12B-it](./google_gemma-4-12B-it)

Multimodal, dispatch MoE. Uses `vllm/vllm-openai:gemma4-unified` image.

| Profile | Context | Batched Tokens | Max Seqs | TPS (c4) |
|---------|---------|----------------|----------|----------|
| max-context (default) | 262,144 | 65,536 | 4 | ~39 tok/s |
| throughput | 32,768 | 65,536 | 32 | ~261 tok/s (c32) |

```bash
cd google_gemma-4-12B-it
./run-gemma4-12b-vllm.sh                    # max-context
MODE=throughput ./run-gemma4-12b-vllm.sh    # high parallel throughput
```

### [Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF](./qwopus3.6-27B-Coder-MTP-GGUF)

Coder-finetuned 27B dense model with Multi-Token Prediction (MTP) head. GGUF Q4_K_M quant (~15.6 GB). Loaded via vLLM's GGUF loader on `vllm/vllm-openai:latest`. Strong coding/agentic reasoning — 67.0% SWE-bench Verified.

```bash
cd qwopus3.6-27B-Coder-MTP-GGUF
./run-qwopus36-vllm.sh
ENABLE_MTP=true ./run-qwopus36-vllm.sh    # enable MTP speculation (experimental)
```

### [RedHatAI/Qwen3.6-35B-A3B-NVFP4](./qwen3.6-35B-A3B-NVFP4)

Multimodal MoE (35B total, ~3B active). NVFP4 quantized via compressed-tensors. Uses `vllm/vllm-openai:latest` image. **Highest aggregate throughput tested** on this hardware: **127 tok/s at concurrency 4** with full 262K context.

```bash
cd qwen3.6-35B-A3B-NVFP4
./run-qwen36-vllm.sh
# Benchmarked: 40 tok/s (c1), 80 tok/s (c2), 127 tok/s (c4) — PP=512, TG=128
```

### [cyburn/Qwopus3.6-35B-A3B-NVFP4 (MoE + MTP)](./qwopus3.6-35B-A3B)

Multimodal MoE (35B total, 3.5B active per token), 256 experts × 8 active, MTP 1 head, 40 layers with hybrid Mamba+Attention. NVFP4+BF16 mixed quant optimized for Blackwell (4.75 bits avg, ~23.5 GB). Native vLLM support via compressed-tensors — no patches needed. Same config as the original Jackrong model.

```bash
cd qwopus3.6-35B-A3B
./run-qwopus36-35b-a3b.sh
```

## Hardware Context

- **GPU**: NVIDIA GB10 (Blackwell CC 12.1), 119.7 GiB unified memory
- **Arch**: aarch64 (ARM Grace)
- **GPU memory utilization**: all profiles now default to **0.70** for reliability across models; override via `GPU_MEMORY_UTILIZATION=0.85 ./run-*.sh`
- **Restart safety**: default `--restart unless-stopped` for production; pass `RESTART_POLICY=no` for tuning

## Benchmarking

Each project folder includes a `bench-qwen36.sh` / `bench-gemma4-12b.sh` script powered by `llama-benchy`:

```bash
CONCURRENCY='1 2 4' PP=512 TG=128 RUNS=1 ./bench-qwen36.sh
```

Results saved to `bench-results/` as JSON + summary text.

## Local Setup

Models are expected in HuggingFace cache at `~/.cache/huggingface/hub/`. The scripts mount this path into the Docker container at `/root/.cache/huggingface`.
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

### [RedHatAI/Qwen3.6-35B-A3B-NVFP4](./qwen3.6-35B-A3B-NVFP4)

Multimodal MoE (35B total, ~3B active). NVFP4 quantized via compressed-tensors. Uses `vllm/vllm-openai:latest` image. **Highest aggregate throughput tested** on this hardware: **127 tok/s at concurrency 4** with full 262K context.

```bash
cd qwen3.6-35B-A3B-NVFP4
./run-qwen36-vllm.sh
# Benchmarked: 40 tok/s (c1), 80 tok/s (c2), 127 tok/s (c4) — PP=512, TG=128
```

## Hardware Context

- **GPU**: NVIDIA GB10 (Blackwell CC 12.1), 119.7 GiB unified memory
- **Arch**: aarch64 (ARM Grace)
- **Best practice**: 0.92–0.94 `--gpu-memory-utilization` for max-context profiles (0.95+ fails the startup free-memory check)
- **Restart safety**: default `--restart unless-stopped` for production; pass `RESTART_POLICY=no` for tuning

## Benchmarking

Each project folder includes a `bench-qwen36.sh` / `bench-gemma4-12b.sh` script powered by `llama-benchy`:

```bash
CONCURRENCY='1 2 4' PP=512 TG=128 RUNS=1 ./bench-qwen36.sh
```

Results saved to `bench-results/` as JSON + summary text.

## Local Setup

Models are expected in HuggingFace cache at `~/.cache/huggingface/hub/`. The scripts mount this path into the Docker container at `/root/.cache/huggingface`.
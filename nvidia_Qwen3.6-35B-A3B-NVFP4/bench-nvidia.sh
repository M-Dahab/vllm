#!/usr/bin/env bash
set -Eeuo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="nvidia/Qwen3.6-35B-A3B-NVFP4"
OUT_DIR="${OUT_DIR:-/home/mohammad/vllm/nvidia_Qwen3.6-35B-A3B-NVFP4/bench-results}"
PP="${PP:-512}"
TG="${TG:-128}"
RUNS="${RUNS:-1}"
CONCURRENCY="${CONCURRENCY:-1 2 4 6 8 10 12 16 26 32}"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_JSON="$OUT_DIR/qwen36-${TS}.json"
SUMMARY="$OUT_DIR/qwen36-${TS}-summary.txt"

echo "Benchmarking $MODEL at $BASE_URL"
echo "pp=$PP tg=$TG runs=$RUNS concurrency=[$CONCURRENCY]"
uvx llama-benchy \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --pp "$PP" \
  --tg "$TG" \
  --concurrency $CONCURRENCY \
  --runs "$RUNS" \
  --no-adapt-prompt \
  --skip-coherence \
  --format json \
  --save-result "$OUT_JSON"
python3 - "$OUT_JSON" > "$SUMMARY" <<'PY'
import json,sys
p=sys.argv[1]
data=json.load(open(p))
print('file:',p)
print('concurrency\tgeneration_tps\tpeak_tps\te2e_ttft_s')
best=None
for b in data.get('benchmarks',[]):
    c=b.get('concurrency')
    tg=(b.get('tg_throughput') or {}).get('mean')
    peak=(b.get('peak_throughput') or {}).get('mean')
    ttft=(b.get('e2e_ttft') or {}).get('mean')
    print(f'{c}\t{tg:.3f}\t{peak:.3f}\t{ttft:.3f}')
    if tg is not None and (best is None or tg > best[1]): best=(c,tg)
if best: print('\nBest generation throughput:', best[1], 'tok/s at concurrency', best[0])
PY
cat "$SUMMARY"
echo "Saved JSON: $OUT_JSON"

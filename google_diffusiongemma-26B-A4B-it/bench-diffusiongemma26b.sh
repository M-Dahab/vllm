#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
OUT_DIR="${OUT_DIR:-/home/mohammad/vllm/google_diffusiongemma-26B-A4B-it/bench-results}"
PP="${PP:-512}"
TG="${TG:-128}"
RUNS="${RUNS:-2}"
CONCURRENCY="${CONCURRENCY:-1 2 4}"

mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_JSON="$OUT_DIR/diffusiongemma26b-${TS}.json"
OUT_MD="$OUT_DIR/diffusiongemma26b-${TS}.md"
SUMMARY="$OUT_DIR/diffusiongemma26b-${TS}-summary.txt"

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
  --format json \
  --save-result "$OUT_JSON" | tee "$OUT_MD"

python3 - "$OUT_JSON" > "$SUMMARY" <<'PY'
import json, sys, math
p = sys.argv[1]
data = json.load(open(p))
rows = []
for b in data.get('benchmarks', []):
    c = b.get('concurrency')
    def g(path):
        x = b
        for k in path:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return x
    tg   = g(['tg_throughput','mean']) or g(['gen_tps','mean']) or g(['generation_throughput','mean'])
    peak = g(['peak_throughput','mean'])
    ttft = g(['e2e_ttft','mean']) or g(['ttft','mean'])
    reqps = g(['request_throughput','mean'])
    rows.append((c, tg, peak, ttft, reqps))

print('file:', p)
print('concurrency\tgeneration_tps\tpeak_tps\te2e_ttft_s\treq_per_s')
best = None
for r in rows:
    print('\t'.join('' if x is None else (f'{x:.3f}' if isinstance(x, (int, float)) else str(x)) for x in r))
    if r[1] is not None and (best is None or r[1] > best[1]):
        best = r
if best:
    print(f'\nBest generation throughput: {best[1]:.3f} tok/s at concurrency {best[0]}')
PY

cat "$SUMMARY"
echo "Saved JSON:    $OUT_JSON"
echo "Saved summary: $SUMMARY"

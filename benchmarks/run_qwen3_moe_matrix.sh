#!/usr/bin/env bash
set -euo pipefail

model=${MODEL:-/root/models/Qwen3-30B-A3B}
output=${OUTPUT:-benchmarks/results/qwen3-30b-a3b-grouped-a2a-4090x4.jsonl}
repeats=${REPEATS:-3}
num_pages=${NUM_PAGES:-64}

already_finished() {
  local run_name=$1
  [[ -f "$output" ]] && .venv/bin/python - "$output" "$run_name" <<'PY'
import json
import sys

path, name = sys.argv[1:]
with open(path) as source:
    if any(json.loads(line)["name"] == name for line in source if line.strip()):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_one() {
  local dispatch=$1
  local scenario=$2
  local concurrency=$3
  local prompt_tokens=$4
  local output_tokens=$5
  local repetition=$6
  local run_name="moe-tp4-${dispatch}-${scenario}-r${repetition}"
  if already_finished "$run_name"; then
    echo "SKIP $run_name"
    return
  fi
  echo "RUN $run_name"
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
    .venv/bin/torchrun --standalone --nproc-per-node=4 \
    benchmarks/benchmark_generation.py \
    --model "$model" --tensor-parallel-size 4 \
    --name "$run_name" --moe-dispatch "$dispatch" \
    --concurrency "$concurrency" --prompt-tokens "$prompt_tokens" \
    --max-new-tokens "$output_tokens" --num-pages "$num_pages" \
    --output "$output"
}

mkdir -p "$(dirname "$output")"
for repetition in $(seq 1 "$repeats"); do
  case "$repetition" in
    1) dispatches=(sorted grouped all_to_all) ;;
    2) dispatches=(all_to_all grouped sorted) ;;
    *) dispatches=(grouped sorted all_to_all) ;;
  esac
  for dispatch in "${dispatches[@]}"; do
    run_one "$dispatch" decode-latency 1 128 64 "$repetition"
    run_one "$dispatch" decode-throughput 8 128 64 "$repetition"
    run_one "$dispatch" prefill 4 1024 32 "$repetition"
  done
done

#!/usr/bin/env bash
# inspect_evals' AIME 2025 on the 8 GB RTX 4060, three stages, resumable.
#
#   setsid nohup inspect_aime/run_4060.sh >> logs/inspect/run.log 2>&1 &
#
# Inspect starts and stops its own vLLM server per stage. If a stage dies
# (power cut, OOM), re-running this script retries its last log, which keeps
# every sample already finished and runs only the rest.
#
# ponytail: no temperature watchdog here; run_aime_4060.sh has one if the card
# starts hitting 85C again.
set -u
cd "$(dirname "$0")/.."
LOGS=${LOGS:-logs/inspect}
GPU_MEM=${GPU_MEM:-0.90}     # 0.90 was the floor and ceiling on this card, see run_aime_4060.sh
CONN=${CONN:-8}              # requests in flight; vLLM queues what does not fit
mkdir -p "$LOGS"

stage() {  # name model thinking
  local name=$1 model=$2 thinking=$3 dir="$LOGS/$1"
  if [[ -f "$dir/DONE" ]]; then echo "== $name: done, skipping"; return; fi
  mkdir -p "$dir"
  local last; last=$(ls -t "$dir"/*.eval 2>/dev/null | head -1)
  echo "== $name: $(date '+%F %T') ${last:+retrying $last}"
  if [[ -n "$last" ]]; then
    inspect eval-retry "$last" --log-dir "$dir" --max-connections "$CONN" --display plain
  else
    inspect eval inspect_aime/task.py -T thinking="$thinking" \
      --model "vllm/$model" \
      -M gpu_memory_utilization="$GPU_MEM" -M enforce_eager=true -M max_model_len=34816 \
      --max-connections "$CONN" --log-dir "$dir" --display plain
  fi && touch "$dir/DONE"
}

stage minicpm5-think  openbmb/MiniCPM5-2B true  || exit 1
stage qwen-think      Qwen/Qwen3.5-2B    true  || exit 1
stage qwen-nothink    Qwen/Qwen3.5-2B    false || exit 1

python inspect_aime/analyze.py "$LOGS"/*/*.eval | tee "$LOGS/summary.txt"

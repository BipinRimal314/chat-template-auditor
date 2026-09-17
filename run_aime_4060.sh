#!/usr/bin/env bash
# AIME 2025 sweep on the 8 GB RTX 4060, detached and resumable.
#
#   setsid nohup ./run_aime_4060.sh >> results/vllm-4060/sweep.log 2>&1 &
#
# Re-running skips every generation already on disk, so a power cut or a
# watchdog stop costs at most one chunk. Results go to their own directory:
# results/aime2025_qwen3.5-2b_nothink.jsonl came from MLX, and a suite must not
# straddle backends, so nothing here appends to it.
#
# Settings forced by 8 GB (see backends.py): --eager, since CUDA graphs eat most
# of the KV cache, and 32k generations fit about one MiniCPM5 and 1.5 Qwen3.5
# sequences at a time. Chunks are small so a crash loses little work.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=${OUT:-results/vllm-4060}
N=${N:-4}
HOT=${HOT:-85}          # degrees C; card self-throttles near 83
mkdir -p "$OUT"
rm -f "$OUT/STOP"

# Watchdog: log temperature every 10 s (flushed, unlike nvidia-smi -f) and stop
# the run after three consecutive readings at or above $HOT.
watchdog() {
  local hot=0
  while :; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "$(date '+%F %T') temp=${t:-?} mem=${m:-?}" >> "$OUT/gpu.log"
    if [[ "${t:-0}" =~ ^[0-9]+$ ]] && (( t >= HOT )); then hot=$((hot+1)); else hot=0; fi
    if (( hot >= 3 )); then
      echo "$(date '+%F %T') WATCHDOG: ${t}C on 3 readings in a row, stopping" | tee -a "$OUT/gpu.log"
      touch "$OUT/STOP"
      [[ -f "$OUT/stage.pgid" ]] && kill -TERM -- "-$(cat "$OUT/stage.pgid")" 2>/dev/null
      return
    fi
    sleep 10
  done
}
watchdog & WD=$!
trap 'kill $WD 2>/dev/null; [[ -f "$OUT/stage.pgid" ]] && kill -TERM -- "-$(cat "$OUT/stage.pgid")" 2>/dev/null' EXIT

stage() {  # name, then eval_aime.py args
  local name=$1; shift
  if [[ -f "$OUT/STOP" ]]; then echo "skip $name: STOP present"; return 1; fi
  echo "=== $(date '+%F %T') start $name"
  setsid "$PY" eval_aime.py "$@" > "$OUT/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$OUT/stage.pgid"
  wait "$pid"; local rc=$?
  rm -f "$OUT/stage.pgid"
  echo "=== $(date '+%F %T') end $name exit=$rc"
  [[ -f "$OUT/STOP" ]] && return 1
  return 0
}

COMMON=(--year 2025 --n "$N" --backend vllm --eager --gpu-mem 0.92)

# Ablation first, as in sweep.sh: it can falsify the hypothesis fastest.
stage qwen_nothink  --model qwen3.5-2b  "${COMMON[@]}" --no-thinking --chunk-size 6 \
      --out "$OUT/aime2025_qwen3.5-2b_nothink.jsonl" || exit 1
stage qwen_think    --model qwen3.5-2b  "${COMMON[@]}"               --chunk-size 6 \
      --out "$OUT/aime2025_qwen3.5-2b.jsonl"         || exit 1
stage minicpm_think --model minicpm5-2b "${COMMON[@]}"               --chunk-size 4 \
      --out "$OUT/aime2025_minicpm5-2b.jsonl"        || exit 1

echo "=== $(date '+%F %T') sweep complete"
"$PY" eval_aime.py score "$OUT"/aime2025_*.jsonl

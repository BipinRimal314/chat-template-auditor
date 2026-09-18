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
# sequences at a time. --chunk-size is the most answers in flight; each answer
# is saved the moment it finishes, so a power cut loses only unfinished ones.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=${OUT:-results/vllm-4060}
# 2 attempts per problem, down from 4 on 18 Sep to halve the run. Problems
# already done keep their 4; scoring averages within a problem first, so every
# problem still counts once. 2 separates 30% from 80%, not close scores.
N=${N:-2}
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

# vLLM refuses to start unless util x total is free. The desktop's share moves:
# hyprlock alone holds ~170 MiB while the screen is locked. So pick the largest
# setting that fits now, capped at 0.92. Below 0.90 a 32k MiniCPM5 or Qwen3.5
# sequence no longer fits (measured: 0.88 fails for both), so wait instead.
gpu_mem() {
  local u
  while :; do
    u=$("$PY" -c 'import torch, math
f, t = torch.cuda.mem_get_info()
print(f"{min(0.92, math.floor((f - 96 * 2**20) / t * 100) / 100):.2f}")' 2>/dev/null)
    if [[ -n "$u" ]] && awk "BEGIN{exit !($u >= 0.90)}"; then echo "$u"; return; fi
    echo "$(date '+%F %T') waiting for GPU memory (would get ${u:-?}, need 0.90)" >&2
    [[ -f "$OUT/STOP" ]] && return 1
    sleep 60
  done
}

stage() {  # name, then eval_aime.py args
  local name=$1; shift
  if [[ -f "$OUT/STOP" ]]; then echo "skip $name: STOP present"; return 1; fi
  local gm; gm=$(gpu_mem) || return 1
  echo "=== $(date '+%F %T') start $name gpu-mem=$gm"
  setsid "$PY" eval_aime.py "$@" --gpu-mem "$gm" > "$OUT/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$OUT/stage.pgid"
  wait "$pid"; local rc=$?
  rm -f "$OUT/stage.pgid"
  echo "=== $(date '+%F %T') end $name exit=$rc"
  [[ -f "$OUT/STOP" ]] && return 1
  return 0
}

# --stop-loops: abort an answer once its last 3,000 characters are one block
# repeated exactly. Qwen3.5-2B with thinking off did this on over half its
# answers, each running to the 32k cap for ~12 minutes; continuing a periodic
# tail cannot change the extracted answer. Applied to every stage alike.
COMMON=(--year 2025 --n "$N" --backend vllm --eager --stop-loops)

# Ablation first, as in sweep.sh: it can falsify the hypothesis fastest.
stage qwen_nothink  --model qwen3.5-2b  "${COMMON[@]}" --no-thinking --chunk-size 6 \
      --out "$OUT/aime2025_qwen3.5-2b_nothink.jsonl" || exit 1
stage qwen_think    --model qwen3.5-2b  "${COMMON[@]}"               --chunk-size 6 \
      --out "$OUT/aime2025_qwen3.5-2b.jsonl"         || exit 1
stage minicpm_think --model minicpm5-2b "${COMMON[@]}"               --chunk-size 4 \
      --out "$OUT/aime2025_minicpm5-2b.jsonl"        || exit 1

echo "=== $(date '+%F %T') sweep complete"
"$PY" eval_aime.py score "$OUT"/aime2025_*.jsonl

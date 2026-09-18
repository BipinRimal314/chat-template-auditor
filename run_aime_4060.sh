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
PY=${PY:-.venv/bin/python}
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

# GPU memory share for vLLM. 0.90 is both the floor and, since 18 Sep, the
# ceiling: below it a 32k MiniCPM5 or Qwen3.5 sequence no longer fits (0.88
# fails for both), and above it there is no room for the desktop to grow. The
# desktop takes ~400-600 MiB, and hyprlock adds 169 MiB when the screen locks.
# A stage that started at 0.92 with the screen unlocked ran out of memory
# mid-answer 80 s after the screen locked; MiniCPM5 had already run 36 minutes
# at 0.90 while locked. vLLM's own startup refusal is still the authority and
# is handled in stage(). Override with GPU_MEM=... if the desktop changes.
gpu_mem() {
  echo "${GPU_MEM:-0.90}"
}

stage() {  # name, then eval_aime.py args
  local name=$1; shift
  if [[ -f "$OUT/STOP" ]]; then echo "skip $name: STOP present"; return 1; fi
  local gm rc pid ooms=0
  gm=$(gpu_mem) || return 1
  while :; do
    echo "=== $(date '+%F %T') start $name gpu-mem=$gm"
    setsid "$PY" eval_aime.py "$@" --gpu-mem "$gm" > "$OUT/$name.log" 2>&1 &
    pid=$!
    echo "$pid" > "$OUT/stage.pgid"
    wait "$pid"; rc=$?
    rm -f "$OUT/stage.pgid"
    echo "=== $(date '+%F %T') end $name exit=$rc"
    [[ -f "$OUT/STOP" ]] && return 1
    (( rc == 0 )) && return 0
    # The free-memory estimate can be optimistic: Qwen3.5's processor also
    # initialises CUDA in the parent before vLLM measures. On exactly that
    # startup refusal, step down; below 0.90, wait for memory to free up.
    if grep -q 'less than desired GPU memory utilization' "$OUT/$name.log"; then
      gm=$(awk "BEGIN{printf \"%.2f\", $gm - 0.01}")
      if awk "BEGIN{exit !($gm < 0.90)}"; then
        echo "$(date '+%F %T') vLLM refused even 0.90; waiting for GPU memory"
        sleep 60
        [[ -f "$OUT/STOP" ]] && return 1
        gm=$(gpu_mem)
      fi
      continue
    fi
    # Out of memory mid-run: something else on the GPU grew after vLLM had
    # reserved its share. Saved answers are safe, so wait and resume, but give
    # up after 5 in a stage rather than loop forever on a real problem.
    if grep -q 'CUDA out of memory' "$OUT/$name.log" && (( ++ooms <= 5 )); then
      cp "$OUT/$name.log" "$OUT/$name.oom$ooms.log"
      echo "$(date '+%F %T') $name ran out of GPU memory (${ooms}/5); resuming in 60s"
      sleep 60
      [[ -f "$OUT/STOP" ]] && return 1
      gm=$(gpu_mem)
      continue
    fi
    # Anything else is a real failure. Stop rather than run later stages on a
    # broken setup; finished answers are on disk and a re-run resumes.
    echo "=== $(date '+%F %T') $name failed (exit $rc); sweep stopped. See $OUT/$name.log"
    return 1
  done
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
# One answer at a time: with 4 in flight, two long MiniCPM5 answers outgrew the
# ~36k-token KV cache, vLLM preempted one and later rebuilt it in 8,192-token
# prefill chunks, and those 96-192 MiB activations hit fragmented memory and
# ran out three times on 18 Sep. At 32k only ~1 sequence fits anyway.
stage minicpm_think --model minicpm5-2b "${COMMON[@]}"               --chunk-size 1 \
      --out "$OUT/aime2025_minicpm5-2b.jsonl"        || exit 1

echo "=== $(date '+%F %T') sweep complete"
"$PY" eval_aime.py score "$OUT"/aime2025_*.jsonl

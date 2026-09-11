#!/usr/bin/env bash
# Full sweep. Every stage is resumable; re-running skips finished generations.
# Backend is auto-detected (mlx on Apple Silicon, vllm on CUDA) unless BACKEND is set.
set -u
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
B=${BACKEND:-auto}
N=${N:-4}

set -x
# Ablation first: it is the half of the hypothesis that can be falsified fastest.
# If this lands near the card's reported 29.6 for Qwen3.5-2B, the card ran defaults.
$PY eval_aime.py   --model qwen3.5-2b  --year 2025 --n "$N" --no-thinking --backend "$B"
$PY eval_code.py   --model qwen3.5-2b  --n 3                              --backend "$B"
$PY eval_aime.py   --model qwen3.5-2b  --year 2025 --n "$N"               --backend "$B"
$PY eval_aime.py   --model minicpm5-2b --year 2025 --n "$N"               --backend "$B"
$PY eval_nolima.py --model minicpm5-2b --lengths 2000,8000,32000          --backend "$B"
$PY eval_nolima.py --model qwen3.5-2b  --lengths 2000,8000,32000          --backend "$B"

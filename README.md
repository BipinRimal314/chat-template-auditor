# Small-model eval harness + chat-template auditor

Two tools built around one question: **was this benchmark table produced under
conditions that make its numbers comparable?**

The trigger was [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B),
whose card claims 2B-class SOTA with an average of 53.9 across 33 benchmarks,
including AIME 2025 at 86.5 against a Qwen3.5-2B baseline of 29.6.

## The finding

Chat templates decide whether a model is allowed to reason before answering, and
they do not agree on what happens when a harness calls `apply_chat_template()`
without passing `enable_thinking`.

| Model | Unset flag produces | Effect |
|---|---|---|
| MiniCPM5-2B | no reasoning prefill | model may open its own `<think>` |
| Qwen3.5-2B | `<think>\n\n</think>` | reasoning slot closed before it starts |
| Qwen3.5-4B | `<think>\n` | reasoning forced open |

Qwen3.5-2B and Qwen3.5-4B ship **inverted defaults in the same release line**.
The same inversion appears in downstream quantisations by unrelated parties
(`StationPC/Qwen3.5-0.8B-RKNN3` vs their 4B, `Vishva007/Qwen3.8-2B-Distill`
vs their 4B), because repackagers inherit the upstream template verbatim.

On the MiniCPM5 card's own baseline set, the eight compared models span **three
different reasoning states**: one suppressed, three forced-on, three left to the
model. That is enough to say the comparison is not controlled, without alleging
anything about intent.

### What this does not establish

An early ablation run trended *above* the card's reported 29.6, and the model
still emits ~8k tokens of visible step-by-step work when the `<think>` slot is
closed. Suppression blocks the reasoning channel, not the reasoning. The
template inversion is a real packaging defect; whether it explains the card's
baseline column is still open, and the AIME sweep is the test.

## auditor/ - chat-template auditor

Renders each template three times (flag unset, true, false) and compares what
lands after the assistant header. Rendering rather than pattern-matching is the
point: an earlier regex version of this tool got answers wrong in both
directions, including on models in this README.

```bash
python auditor/audit.py Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B openbmb/MiniCPM5-2B
python auditor/audit.py --file repos.txt --json out.json --token $HF_TOKEN
```

It reports per-model verdicts (`suppressed`, `forced-on`, `model-choice`) and
flags **family inversions**, where sibling models disagree and a family sweep is
therefore not self-consistent. Network and CPU only; no GPU, no model downloads.

Handles templates carried in `chat_template.jinja`, `chat_template.json` or
`tokenizer_config.json`, the `{% generation %}` extension transformers templates
use, and the `<think>` / `<seed:think>` / `<|channel>` reasoning conventions.
Gated repos are reported as gated rather than silently skipped.

## The evals

Both models run through one harness with the same precision, sampler, prompt and
reasoning flag. Three suites:

- **`eval_aime.py`** - MathArena AIME 2025 / 2026, 30 problems each, avg@n on
  exact integer match of the last `\boxed{}`. `--no-thinking` reproduces what a
  default harness would have done.
- **`eval_nolima.py`** - NoLiMa-style long context with a haystack generated at
  run time, so no model can have memorised it. Needles require a one-hop
  world-knowledge association and share no content words with the question.
- **`eval_code.py`** - 12 original single-file bug-repair tasks with hidden
  tests. A proxy for the card's coding-agent claims, **not** a reproduction of
  SWE-bench Verified, which needs per-repo containers.
  `python validate_tasks.py` proves every buggy version fails and every
  reference fix passes.

Everything is resumable. Results append per generation to `results/*.jsonl` and a
re-run skips what already landed.

## Setup

**Apple Silicon**

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-mlx.txt
.venv/bin/python smoke.py minicpm5-2b
```

**CUDA**

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-cuda.txt
.venv/bin/python smoke.py minicpm5-2b --backend vllm
```

The backend is chosen automatically by platform; `--backend` overrides. Prompt
construction goes through transformers' `AutoTokenizer` on both paths so the two
machines build byte-identical prompts.

## Choosing a machine

Single-stream speed is a wash between an M4 Pro (~273 GB/s) and a desktop
RTX 4060 (~272 GB/s), because generation at this size is bandwidth-bound. The
GPU's advantage is batching, and the limit on an 8 GB card is the KV cache.

| Model | Attention | KV cache |
|---|---|---|
| MiniCPM5-2B | 42 layers, all full | 42 KiB/token |
| Qwen3.5-2B | 6 full of 24, rest linear | 12 KiB/token |

Concurrent MiniCPM5 sequences that fit in 8 GB at bf16, worst case:

| Context | 8k | 16k | 32k | 64k |
|---|---|---|---|---|
| Sequences | 6 | 3 | 1 | 0 |

So: **AIME and bug repair belong on the GPU box** (4-6x via batching, and it is
not the machine you are typing on). **The 64k long-context run needs a
larger-memory machine** - MiniCPM5 at bf16 plus a 64k cache is about 7.8 GB and
will not fit in 8 GB. Quantising to make it fit would confound the measurement.

Do not mix backends within a single test. MLX and vLLM differ in sampling and
numerics, so if a suite moves machines, both models re-run there.

```bash
# GPU box
python eval_aime.py   --model minicpm5-2b --year 2025 --n 4 --backend vllm --chunk-size 24
python eval_code.py   --model qwen3.5-2b  --n 3          --backend vllm

# larger-memory machine
python eval_nolima.py --model minicpm5-2b --lengths 64000
```

## Status

| Suite | State |
|---|---|
| Template audit | complete, findings above |
| Bug repair, MiniCPM5-2B | 36/36 on MLX - **saturated**, confirms competence but ranks nothing |
| Bug repair, Qwen3.5-2B | invalid, first attempt hit a token cap only it reached; needs re-running |
| AIME ablation | 12 of 120, parked |
| AIME main runs | not started |
| NoLiMa | not started |

The 8k-cap bug-repair run is kept as `results/bugfix_qwen3.5-2b.CAPPED8k.jsonl`
as a worked example of the same harness mistake this project is about.

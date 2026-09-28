# Findings: the MiniCPM5-2B AIME baseline holds up

**Result, up front:** the MiniCPM5-2B card's AIME 2025 comparison against
Qwen3.5-2B reproduces on an independent harness. Two specific mechanisms that
could have made the card's Qwen baseline unfair were tested and both were
ruled out. The card's relative claim survives.

The useful finding is a methodological one that emerged on the way: **at the 2B
scale on AIME, an accuracy score is largely a measure of whether the model
terminates.** Across 259 attempts in four configurations, attempts that stopped
on their own were correct **69.5%** of the time (66 of 95); attempts that looped
or hit the token cap were correct **4.9%** of the time (8 of 164). For Qwen with
thinking on, under either sampler, the degenerate column is exactly zero: all 16
of its correct answers came from the 16 attempts that terminated.

Run 11-28 September 2026 on one RTX 4060 (8 GB), 17 hours of wall-clock
generation. All raw answers are in `results/vllm-4060/`.

---

## The question

[openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)'s card reports
AIME 2025 at **86.5** against a Qwen3.5-2B baseline of **29.6**. A 57-point win
over a same-size competitor is worth checking, because the baseline is the
easiest number on a card to produce carelessly.

The specific suspicion came from this repo's template auditor: Qwen3.5-2B's chat
template closes the reasoning slot (`<think>\n\n</think>`) when a harness calls
`apply_chat_template()` without `enable_thinking`, while MiniCPM5-2B's does not.
A harness that never sets the flag therefore benchmarks Qwen with its reasoning
channel shut and MiniCPM with it open. See `README.md` for the template survey,
including the inverted defaults between Qwen3.5-2B and Qwen3.5-4B.

**Hypothesis 1:** the card's 29.6 is Qwen measured with reasoning suppressed.
If so, running Qwen with thinking explicitly *on* should score materially higher.

## Method

Both models run through one harness: same 30 AIME 2025 problems
(`MathArena/aime_2025`), same prompt, same 32,768-token cap, same grader, same
loop detector, seeds fixed per attempt. The reasoning flag is set explicitly in
every stage rather than left to the template. vLLM on one 8 GB card,
`enforce_eager`, `gpu_memory_utilization` 0.90.

Scoring is **avg@n**: accuracy is averaged within a problem first, then across
problems, so each problem counts once regardless of how many attempts it has.
Confidence intervals are bootstrapped over the 30 problems, 20,000 resamples.

| Stage | Model | Thinking | Sampler |
|---|---|---|---|
| 1 | Qwen3.5-2B | off | shared |
| 2 | Qwen3.5-2B | on | shared |
| 3 | MiniCPM5-2B | on | shared |
| 4 | Qwen3.5-2B | on | Qwen's own |

Shared sampler: `temperature 0.6, top_p 0.95`, no top_k, no presence penalty,
applied identically to every model. Stage 4 is defined below.

## Results

| Stage | avg@n | 95% CI | pass@n | Card |
|---|---|---|---|---|
| Qwen3.5-2B, thinking **off** | 17.5% | [7.5, 29.2] | 26.7% | 29.6 |
| Qwen3.5-2B, thinking **on** | 11.7% | [3.3, 21.7] | 16.7% | 29.6 |
| **MiniCPM5-2B**, thinking on | **71.7%** | [56.7, 85.0] | 80.0% | 86.5 |
| Qwen3.5-2B, Qwen's sampler | 15.0% | [5.0, 26.7] | 20.0% | 29.6 |

**MiniCPM5 beats Qwen's best configuration by 54.2 points, 95% CI
[36.7, 70.0]**, against the card's claimed 57. Both our absolute numbers land
below the card's — Qwen's best by 12.1, MiniCPM by 14.8 — which reads as a
consistent harness offset rather than a discrepancy in either direction.
Qwen's interval reaches 29.2 against the card's 29.6, so the card's Qwen number
is **not** refuted by this data.

Per problem, MiniCPM solved 15 that Qwen never solved in any configuration;
Qwen solved **0** that MiniCPM missed; 6 went unsolved by everything. The
advantage is broad, not a favourable subset.

## Hypothesis 1: the chat template. Refuted.

Turning thinking on made Qwen **worse**, 17.5% → 11.7%, not better. On 30
problems those intervals overlap heavily, so the drop itself is not established;
what is established is that the predicted rescue does not happen. The template
inversion is a real packaging defect, but it does not explain the card's
baseline.

## What was actually going wrong

Most Qwen attempts never produced an answer at all. Splitting every attempt by
whether the model stopped on its own:

| Stage | terminated | capped | looped | acc if clean | acc if degenerate |
|---|---|---|---|---|---|
| Qwen, thinking off | 46.8% | 32.9% | 20.3% | 29.7% | 9.5% |
| Qwen, thinking on | 11.7% | 13.3% | **75.0%** | **100.0%** | 0.0% |
| MiniCPM5 | **70.0%** | 23.3% | 6.7% | 92.9% | 22.2% |
| Qwen, Qwen's sampler | 15.0% | 30.0% | 55.0% | **100.0%** | 0.0% |

*capped* = hit the 32,768-token limit. *looped* = stopped early because the last
3,000 characters had become one block repeated exactly.

Qwen with thinking on finished 7 of 60 attempts. **All 7 were correct.** The
other 53 produced nothing. The same held in stage 4: 9 clean, 9 correct, 51
degenerate, 0 correct.

The cap is not the binding constraint. With thinking off, Qwen's longest
answer that terminated was 17,751 tokens — 54% of the limit — with a median of
10,284. With thinking on the terminating answers are longer (median 20,688,
longest 26,400) but still finish inside the budget. Raising the cap would give
the looping attempts more room to loop; it would not convert them into answers.

So the headline scores for Qwen are, almost exactly, its termination rate —
11.7/11.7 and 15.0/15.0 are the same numbers twice.

## Hypothesis 2: the sampler. Also refuted.

Stages 1-3 gave every model one sampler. That is the defensible choice for a
controlled comparison, but it is not the sampler Qwen's own card evaluates the
2B with: it specifies `top_k 20` and `presence_penalty 1.5`, **and gives
loop-proneness as the reason**. The harness had therefore run Qwen in the exact
configuration its authors warn against and observed the exact failure they
predict.

**Hypothesis 2:** Qwen's score is an artifact of the shared sampler. If so, its
own settings should collapse the loop rate and move the score toward the ~100%
accuracy it shows on answers it finishes.

Stage 4 changes **only** the sampler — same model, prompt, thinking flag, seeds,
token cap and loop detector as stage 2, written to its own file.

| | shared sampler | Qwen's own |
|---|---|---|
| avg@n | 11.7% | 15.0% |
| looped | 45 | **33** |
| capped | 8 | **18** |
| terminated cleanly | 7 | 9 |
| degenerate overall | 88.3% | 85.0% |
| mean output tokens | 16,956 | 22,252 |

Paired over the same 30 problems: **+3.3 points, 95% CI [+0.0, +8.3]**. The
interval touches zero, and 3.3 points is ~6% of the 54-point gap.

The penalty worked on the symptom and not the disease. Loops fell 45 → 33, but
cut-offs rose 8 → 18 and mean output grew by 5,300 tokens: **suppressing
repetition made the model wander instead of repeat.** It converted loops into
truncations, not into finished answers.

## Conclusion

Qwen3.5-2B's poor AIME showing is not an artifact of the chat template and not
an artifact of the sampler. It is a failure to terminate, which persists under
the vendor's own anti-loop settings. The MiniCPM5 card's comparison is
substantively fair.

The transferable lesson is the one to carry into any model bake-off:

> Before comparing accuracy, compare termination rates. A model that loops or
> truncates scores near zero regardless of its reasoning quality, and a single
> accuracy column will silently attribute a formatting failure to capability.

This harness reports `truncated` and `looped` per attempt for that reason.

## Cost

| Stage | in flight | wall clock | correct | min/correct |
|---|---|---|---|---|
| Qwen, thinking off | 6 | 4h12m | 15 | 16.8 |
| Qwen, thinking on | 6 | 1h56m | 7 | 16.6 |
| MiniCPM5 | 1 | 7h11m | 43 | 10.0 |
| Qwen, Qwen's sampler | 6 | 3h40m | 9 | 24.5 |

Wall clock per stage, from `sweep.log`, excluding time the machine was off.
Do **not** derive this from the per-answer `seconds` field: with several answers
in flight that field is latency, not exclusive GPU time, and summing it
overstates a batched stage several-fold. The stages also differ in concurrency
(MiniCPM ran one answer at a time because two 32k sequences do not fit in 8 GB),
so these are the cost of obtaining these answers on this box, not a clean
throughput comparison between models.

## Limitations and disclosures

Everything here that could flatter or penalise a model, stated plainly.

- **One sampler for three of four stages**, which is not Qwen's recommended one.
  This was a deliberate choice for comparability, and stage 4 exists to measure
  what it cost. It is still the largest known bias in stages 1-3.
- **30 problems, 2 attempts each** (stage 1's first ten problems have 4, from a
  mid-run change). Intervals are wide; no point estimate here should be quoted
  without one.
- **Stage 1 is not uniform.** Attempts per problem dropped 4 → 2 and loop
  stopping was added, both at 39 answers in. See `LAB_NOTES.md`.
- **The grader was fixed mid-run** and saved answers re-graded: one answer went
  wrong → correct, and **4 are flagged `pred_uncertain`** because they were
  recovered from a truncated tail. All four were wrong either way, so they do
  not inflate Qwen. The originals are kept in
  `aime2025_qwen3.5-2b_nothink.before-rescore.jsonl`.
- **Loop stopping assumes a loop would not have recovered.** For a loop that
  repeats an already-emitted `\boxed{}` answer, stopping cannot change the
  grade. For a reasoning loop with no answer yet, it is an assumption — likely,
  but unverified. Every such answer is marked `looped` and can be re-run without
  the stop to measure the recovery rate. **This has not been done.**
- **Only the last 400 characters of each answer are stored**, so answers cannot
  be fully re-graded after the fact.
- **Absolute scores sit 10-15 points below the card's on both models.** The
  cause is not established. It is consistent across both models, so it does not
  affect the comparison, but it means these numbers do not reproduce the card's
  in absolute terms.
- Single machine, single run per configuration, no seed-variance study.

## Reproducing

```bash
cd chat-template-auditor
setsid nohup ./run_aime_4060.sh >> results/vllm-4060/sweep.log 2>&1 &
watch -n 30 .venv/bin/python progress.py
```

Re-running skips every answer already on disk, so it is safe to resume after an
interruption; each answer is fsynced as it finishes. `progress.py` reads only
saved files and never touches the GPU.

Stage 4 on its own:

```bash
.venv/bin/python eval_aime.py --model qwen3.5-2b --year 2025 --n 2 \
  --backend vllm --eager --stop-loops --chunk-size 6 \
  --top-k 20 --presence-penalty 1.5 \
  --out results/vllm-4060/aime2025_qwen3.5-2b_qwensampler.jsonl
```

### Files

| Path | Contents |
|---|---|
| `results/vllm-4060/aime2025_qwen3.5-2b_nothink.jsonl` | stage 1, 79 attempts |
| `results/vllm-4060/aime2025_qwen3.5-2b.jsonl` | stage 2, 60 attempts |
| `results/vllm-4060/aime2025_minicpm5-2b.jsonl` | stage 3, 60 attempts |
| `results/vllm-4060/aime2025_qwen3.5-2b_qwensampler.jsonl` | stage 4, 60 attempts |
| `results/vllm-4060/*.before-rescore.jsonl` | pre-grader-fix originals |
| `LAB_NOTES.md` | what went wrong running this, and what to do first next time |

Each row carries `idx`, `sample`, `gold`, `pred`, `correct`, `truncated`,
`looped`, `gen_tokens`, `seconds`, `gen_tps` and the answer's last 400
characters.

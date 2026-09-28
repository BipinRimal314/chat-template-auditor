# Does inspect_evals' AIME grader credit answers that never finished?

`inspect_evals` is the shared library of Inspect evaluations. Its AIME scorer
(`inspect_evals/utils/aime_common.py`) grades the **last substantive line** of a
completion. If generation stopped at the token cap in the middle of reasoning,
that line is whatever the model was saying at that moment, and any number at the
end of it is graded as the answer. `test_grader.py` pins this: a trace cut at
"...so the remainder is 1" is scored correct when the target is 1.

The earlier sweep (`../FINDINGS.md`) found that 2B models on AIME often don't
finish: Qwen3.5-2B with thinking on looped in 75% of attempts. So the question
is concrete. **On real runs, how often does the upstream grader give credit to
an attempt that never wrote its answer?** If it's never, that's worth knowing.
If it isn't never, it's an upstream fix with evidence attached.

This runs the upstream task **unchanged** (dataset, prompt, grader) and adds
one scorer, `termination`, which labels each attempt as terminated, capped or
looped, without touching the grade.

## Run it (Linux box, RTX 4060)

```bash
cd chat-template-auditor && git pull && git checkout inspect-aime
python3 -m venv .venv-inspect && . .venv-inspect/bin/activate
pip install -r inspect_aime/requirements.txt
python inspect_aime/test_grader.py          # prints ok
mkdir -p logs/inspect
setsid nohup inspect_aime/run_4060.sh >> logs/inspect/run.log 2>&1 &
tail -f logs/inspect/run.log
```

After a power cut, run the same `setsid` line again. Finished stages are
skipped, and an interrupted stage resumes from its log.

When it finishes, `logs/inspect/summary.txt` holds the result. The line to read is
`credit without ANSWER: line` for each stage. `inspect view --log-dir logs/inspect`
opens every attempt in a browser.

## Stages

| Stage | Model | Thinking |
|---|---|---|
| minicpm5-think | MiniCPM5-2B | on |
| qwen-think | Qwen3.5-2B | on |
| qwen-nothink | Qwen3.5-2B | off |

30 problems × 2 attempts each, 32,768-token cap, temperature 0.6, top_p 0.95,
the same as the earlier sweep. The prompt is upstream's (`ANSWER: $ANSWER`),
not the sweep's `\boxed{}`, so scores are not directly comparable with FINDINGS.md.

**Expect this to take longer than the earlier sweep.** That one stopped a looping
answer as soon as it was detected. Here every loop runs to the 32k cap, because the
point is to see what the upstream grader does with exactly those answers.

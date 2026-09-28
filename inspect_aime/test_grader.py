"""Pins the upstream grader behaviour this experiment measures, so a change in
inspect_evals shows up here before it silently changes the results.

    python inspect_aime/test_grader.py
"""
import asyncio

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_evals.utils.aime_common import aime_scorer

from task import ending


def grade(text, target, stop_reason="stop"):
    out = ModelOutput.from_content("m", text, stop_reason=stop_reason)
    st = TaskState(model="m", sample_id=1, epoch=1, input="q", messages=[], output=out)
    return asyncio.run(aime_scorer()(st, Target(target))), out


# A clean answer is graded correctly.
s, _ = grade("work...\nANSWER: 16", "16")
assert s.value == "C"

# The defect: reasoning cut at the cap, no ANSWER: line, trailing number = target.
s, out = grade("Try n = 3: then 3^2 + 7 = 16, so the remainder is 1", "1",
               stop_reason="max_tokens")
assert s.value == "C", "upstream no longer credits truncated reasoning; re-check the finding"
assert ending(out) == "capped"

# A loop repeating its answer, cut mid-token, is graded on the fragment.
s, _ = grade("ANSWER: 16\n" * 3 + "ANSWER: 1", "16", stop_reason="max_tokens")
assert s.value == "I" and s.answer == "1"

# Exact repetition is labelled looped even when the cap is what stopped it.
_, out = grade("wait, let me recheck. " * 200, "16", stop_reason="max_tokens")
assert ending(out) == "looped"

print("ok")

"""inspect_evals' own AIME 2025 task, run unchanged, with one extra scorer that
records how each answer ended.

The question is whether the upstream grader gives credit to answers that never
finished. aime_scorer grades the last substantive line of whatever came back,
so a completion cut off at the token cap is graded on its last number even when
it never wrote "ANSWER:". The termination scorer never changes a grade; it only
labels each attempt so analyze.py can split accuracy by how the attempt ended.

    inspect eval inspect_aime/task.py -T thinking=true --model vllm/openbmb/MiniCPM5-2B
"""
import sys
from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.model import ContentReasoning, GenerateConfig, ModelOutput
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from inspect_evals.aime2025 import aime2025

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import looping  # the exact-repetition detector used by the sweep


def full_text(output: ModelOutput) -> str:
    """Reasoning plus answer. Inspect moves <think>...</think> into a separate
    reasoning block, so .completion alone can hide a loop inside the reasoning."""
    if not output.choices:
        return ""
    content = output.choices[0].message.content
    if isinstance(content, str):
        return content
    return "".join(c.reasoning if isinstance(c, ContentReasoning)
                   else getattr(c, "text", "") for c in content)


def ending(output: ModelOutput) -> str:
    """terminated, looped or capped. A loop that ran into the cap counts as
    looped: the cap is where it stopped, not why."""
    if looping(full_text(output)):
        return "looped"
    if output.stop_reason == "max_tokens":
        return "capped"
    return "terminated"


@scorer(metrics={"terminated": [mean()], "capped": [mean()], "looped": [mean()]})
def termination():
    async def score(state: TaskState, target: Target) -> Score:
        end = ending(state.output)
        has_answer_line = "ANSWER:" in state.output.completion
        return Score(
            value={k: float(end == k) for k in ("terminated", "capped", "looped")},
            answer=end,
            metadata={"ending": end, "has_answer_line": has_answer_line,
                      "output_tokens": state.output.usage.output_tokens
                      if state.output.usage else None},
        )
    return score


@task
def aime2025_termination(thinking: bool = True, epochs: int = 2,
                         max_tokens: int = 32768) -> Task:
    """Upstream dataset, prompt and grader; the sweep's sampler and token cap."""
    base = aime2025()
    return Task(
        dataset=base.dataset,
        solver=base.solver,
        scorer=[*base.scorer, termination()],
        epochs=Epochs(epochs, ["mean"]),
        config=GenerateConfig(
            max_tokens=max_tokens, temperature=0.6, top_p=0.95,
            # No seed: vLLM applies one seed per request, which would make
            # both epochs of a problem the same text.
            # Set the reasoning flag explicitly: the two models' templates
            # disagree on the default, which is the confound this repo is about.
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        ),
        metadata={"thinking": thinking},
    )

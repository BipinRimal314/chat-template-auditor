"""Split each Inspect log's AIME accuracy by how the attempt ended.

    python inspect_aime/analyze.py logs/inspect/*.eval

The number that matters is "credit without ANSWER:": attempts the upstream
grader marked correct although the model never wrote its answer line, i.e. it
was graded on whatever number happened to be last when generation stopped.
"""
import sys

from inspect_ai.log import read_eval_log

ENDINGS = ("terminated", "capped", "looped")


def rows(path):
    log = read_eval_log(path)
    name = f"{log.eval.model} thinking={log.eval.task_args.get('thinking')}"
    out = []
    for s in log.samples or []:
        up = s.scores.get("aime_scorer")
        term = s.scores.get("termination")
        if up is None or term is None:
            continue           # sample errored or run was interrupted
        out.append({"id": s.id, "correct": up.value == "C",
                    "ending": term.metadata["ending"],
                    "answer_line": term.metadata["has_answer_line"],
                    "extracted": up.answer, "target": s.target})
    return name, out


def report(name, rs):
    if not rs:
        print(f"\n== {name}: no scored samples"); return
    n = len(rs)
    by = {}
    for r in rs:
        by.setdefault(r["id"], []).append(r["correct"])
    avg = 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)
    print(f"\n== {name}  ({len(by)} problems, {n} attempts)")
    print(f"upstream avg@n: {avg:.1f}%")
    for e in ENDINGS:
        g = [r for r in rs if r["ending"] == e]
        if g:
            acc = 100 * sum(r["correct"] for r in g) / len(g)
            print(f"  {e:11} {len(g):4} ({100*len(g)/n:5.1f}%)  correct {acc:5.1f}%")
    free = [r for r in rs if r["correct"] and not r["answer_line"]]
    print(f"credit without ANSWER: line: {len(free)} of {sum(r['correct'] for r in rs)} correct")
    for r in free:
        print(f"    problem {r['id']}: ended {r['ending']}, graded {r['extracted']!r} = target {r['target']!r}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(*rows(p))

"""Do MLX and vLLM agree on the same weights and the same prompts?

Every cross-machine comparison in this repo assumes they do. That assumption is
exactly the kind of thing this project exists to stop people taking on trust, so
it gets measured.

Decoding is greedy. MLX and vLLM use different RNGs, so a seeded sample can never
match across backends and only greedy decoding makes the comparison meaningful.
On identical weights the two should emit identical tokens until floating-point
differences accumulate, so the headline number is where they first diverge.

The probes are deliberately short, bounded tasks rather than competition maths.
Greedy decoding rambles, and a probe that truncates measures the token cap rather
than the backends. Every generation here must finish on its own.

  python check_backends.py run --model minicpm5-2b              # on each machine
  python check_backends.py compare results/agree_*_minicpm5-2b.jsonl
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (get_backend, build_prompt, strip_think, jsonl_append, jsonl_read)
from tasks.bugfix_tasks import TASKS

BUGFIX_TEMPLATE = """The function below has a bug.

Specification:
{spec}

Current code:
```python
{buggy}
```

Rewrite it so it satisfies the specification. Reply with one ```python code block
containing the complete corrected code and nothing else."""

# Short, bounded, and varied in shape: two plain generations, three code repairs.
# All observed well under 4k tokens on this model family.
EXTRA_PROBES = [
    ("arith", "What is 84 * 3 / 2? Answer with the number only."),
    ("list", "Name the first five prime numbers, comma separated, nothing else."),
]
CODE_PROBES = ["tally", "overlaps", "rotate_grid"]


def probes():
    out = list(EXTRA_PROBES)
    by_id = {t["id"]: t for t in TASKS}
    for tid in CODE_PROBES:
        t = by_id[tid]
        out.append((tid, BUGFIX_TEMPLATE.format(spec=t["spec"], buggy=t["buggy"].strip())))
    return out


def run(a):
    backend = get_backend(a.model, a.backend,
                          max_model_len=a.max_tokens + 2048,
                          gpu_memory_utilization=a.gpu_mem)
    out = a.out or f"results/agree_{backend.name}_{a.model}.jsonl"
    Path(out).unlink(missing_ok=True)

    items = [{"id": pid,
              "prompt": build_prompt(backend.tokenizer, text, thinking=True)}
             for pid, text in probes()]

    print(f"[agree] {a.model} backend={backend.name} greedy n={len(items)} -> {out}",
          flush=True)
    gens = backend.generate([i["prompt"] for i in items], a.max_tokens,
                            temp=0.0, top_p=1.0)
    recs, trunc = [], 0
    for it, g in zip(items, gens):
        trunc += bool(g["truncated"])
        recs.append({"id": it["id"], "gen_tokens": g["gen_tokens"],
                     "truncated": g["truncated"], "backend": backend.name,
                     "text": g["text"]})
        print(f"  {it['id']:<10} {g['gen_tokens']:>6} tok  "
              f"{'TRUNCATED' if g['truncated'] else 'complete'}", flush=True)
    jsonl_append(out, recs)

    if trunc:
        print(f"\nWARNING: {trunc} probe(s) hit the {a.max_tokens}-token cap. A truncated "
              f"probe measures the cap, not the backend. Re-run with a larger "
              f"--max-tokens before trusting the comparison.")
    print(f"\nwrote {out}. Copy it next to the other machine's file, then run:")
    print(f"  python check_backends.py compare results/agree_*_{a.model}.jsonl")


def divergence(a_text, b_text):
    """Character index where the two generations first differ, or None if equal."""
    n = min(len(a_text), len(b_text))
    for i in range(n):
        if a_text[i] != b_text[i]:
            return i
    return None if len(a_text) == len(b_text) else n


def compare(paths):
    files = {}
    for p in paths:
        rows = jsonl_read(p)
        if not rows:
            print(f"skip {p}: empty")
            continue
        files[rows[0]["backend"]] = {r["id"]: r for r in rows}
    if len(files) < 2:
        print(f"need two backends, found {list(files)}")
        return 1

    (na, A), (nb, B) = sorted(files.items())
    shared = [k for k in A if k in B]
    any_trunc = any(A[k]["truncated"] or B[k]["truncated"] for k in shared)
    print(f"\ncomparing {na} vs {nb} on {len(shared)} probes (greedy)\n")
    print(f"{'probe':<10} {'tok '+na:>9} {'tok '+nb:>9}  diverges at")
    identical = 0
    for k in shared:
        x, y = A[k], B[k]
        d = divergence(x["text"], y["text"])
        if d is None:
            identical += 1
        mark = "identical" if d is None else f"char {d} of {min(len(x['text']), len(y['text']))}"
        flag = "  [TRUNCATED]" if (x["truncated"] or y["truncated"]) else ""
        print(f"{k:<10} {x['gen_tokens']:>9} {y['gen_tokens']:>9}  {mark}{flag}")

    print(f"\nidentical output: {identical}/{len(shared)} probes")
    if any_trunc:
        print("\nSome probes truncated. Re-run with a larger --max-tokens; a truncated")
        print("probe measures the token cap rather than the backends.")
        return 1
    if identical == len(shared):
        print("\nBackends produce identical greedy output. Cross-machine results are")
        print("comparable, though each suite should still run entirely on one backend.")
    elif identical >= len(shared) * 0.6:
        print("\nBackends mostly agree, diverging late in generation. That is expected")
        print("floating-point drift. Treat scores as comparable, individual texts as not.")
    else:
        print("\nBackends DISAGREE early. Do not compare a result from one machine")
        print("against the other; re-run any affected suite on a single backend.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--backend", default="auto", choices=["auto", "mlx", "vllm"])
    r.add_argument("--max-tokens", type=int, default=6144)
    r.add_argument("--gpu-mem", type=float, default=0.92)
    r.add_argument("--out", default=None)

    c = sub.add_parser("compare")
    c.add_argument("paths", nargs="+")

    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        sys.exit(compare(a.paths))


if __name__ == "__main__":
    main()

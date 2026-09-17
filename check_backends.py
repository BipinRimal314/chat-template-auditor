"""Do MLX and vLLM agree on the same weights and the same prompts?

Every cross-machine comparison in this repo assumes they do. That assumption is
exactly the kind of thing this project exists to stop people taking on trust, so
it gets measured.

Decoding is greedy. MLX and vLLM use different RNGs, so a seeded sample can never
match across backends and only greedy decoding makes the comparison meaningful.
On identical weights the two should emit identical tokens until floating-point
differences accumulate, so the interesting number is where they first diverge.

  python check_backends.py run --model minicpm5-2b              # on each machine
  python check_backends.py compare results/agree_*_minicpm5-2b.jsonl
"""
import argparse, json, sys
from pathlib import Path

import datasets
from common import (get_backend, build_prompt, strip_think, extract_boxed,
                    as_aime_int, jsonl_append, jsonl_read)

PROMPT = ("{problem}\n\nPlease reason step by step, and put your final answer "
          "within \\boxed{{}}.")
PROBLEMS = 5   # deliberately small; this is a consistency probe, not a score


def run(a):
    ds = datasets.load_dataset("MathArena/aime_2025", split="train").select(range(PROBLEMS))
    backend = get_backend(a.model, a.backend,
                          max_model_len=a.max_tokens + 2048,
                          gpu_memory_utilization=a.gpu_mem,
                          enforce_eager=a.eager)
    out = a.out or f"results/agree_{backend.name}_{a.model}.jsonl"
    Path(out).unlink(missing_ok=True)

    items = [{"idx": str(r["problem_idx"]), "gold": as_aime_int(r["answer"]),
              "prompt": build_prompt(backend.tokenizer,
                                     PROMPT.format(problem=r["problem"]),
                                     thinking=True)} for r in ds]

    print(f"[agree] {a.model} backend={backend.name} greedy n={len(items)} -> {out}",
          flush=True)
    gens = backend.generate([i["prompt"] for i in items], a.max_tokens,
                            temp=0.0, top_p=1.0)
    recs = []
    for it, g in zip(items, gens):
        _, ans = strip_think(g["text"])
        pred = as_aime_int(extract_boxed(ans or g["text"]))
        recs.append({"idx": it["idx"], "gold": it["gold"], "pred": pred,
                     "correct": pred is not None and pred == it["gold"],
                     "gen_tokens": g["gen_tokens"], "truncated": g["truncated"],
                     "backend": backend.name, "text": g["text"]})
        print(f"  p{it['idx']:>2} pred={pred} gold={it['gold']} "
              f"{g['gen_tokens']}tok", flush=True)
    jsonl_append(out, recs)
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
        files[rows[0]["backend"]] = {r["idx"]: r for r in rows}
    if len(files) < 2:
        print(f"need two backends, found {list(files)}")
        return 1

    (na, A), (nb, B) = sorted(files.items())
    shared = sorted(set(A) & set(B), key=int)
    print(f"\ncomparing {na} vs {nb} on {len(shared)} problems (greedy)\n")
    print(f"{'prob':>5}  {'pred '+na:>14} {'pred '+nb:>14}  {'tok '+na:>9} {'tok '+nb:>9}  diverges at")
    same_pred = same_text = 0
    for k in shared:
        x, y = A[k], B[k]
        d = divergence(x["text"], y["text"])
        if x["pred"] == y["pred"]:
            same_pred += 1
        if d is None:
            same_text += 1
        print(f"{k:>5}  {str(x['pred']):>14} {str(y['pred']):>14}  "
              f"{x['gen_tokens']:>9} {y['gen_tokens']:>9}  "
              f"{'identical' if d is None else f'char {d}'}")

    print(f"\nsame final answer : {same_pred}/{len(shared)}")
    print(f"identical text    : {same_text}/{len(shared)}")
    if same_pred == len(shared):
        print("\nBackends agree on every answer. Cross-machine results are comparable,")
        print("though each suite should still run entirely on one backend.")
    else:
        print("\nBackends DISAGREE. Do not compare a result from one machine against")
        print("a result from the other; re-run any affected suite on a single backend.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--backend", default="auto", choices=["auto", "mlx", "vllm"])
    r.add_argument("--max-tokens", type=int, default=8192)
    r.add_argument("--gpu-mem", type=float, default=0.92)
    r.add_argument("--eager", action="store_true",
                   help="disable CUDA graphs; frees ~0.9 GiB of KV cache")
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

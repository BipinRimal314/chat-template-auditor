"""Bug-repair eval: model sees a spec plus buggy source, must return fixed code."""
import argparse, re, sys, time
from pathlib import Path

sys.path.insert(0, ".")
from tasks.bugfix_tasks import TASKS
from sandbox import run_tests
from common import (get_backend, build_prompt, strip_think,
                    jsonl_read, run_chunked)

TEMPLATE = """The function below has a bug.

Specification:
{spec}

Current code:
```python
{buggy}
```

Rewrite it so it satisfies the specification. Reply with one ```python code block
containing the complete corrected code and nothing else."""


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=3, help="samples per task")
    ap.add_argument("--max-tokens", type=int, default=24576,
                    help="must exceed what every model needs; a cap only one "
                         "model reaches silently handicaps that model")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "vllm"])
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--eager", action="store_true",
                    help="disable CUDA graphs; frees ~0.9 GiB of KV cache, "
                         "needed for long contexts on an 8 GB card")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or f"results/bugfix_{a.model}{'_nothink' if a.no_thinking else ''}.jsonl"
    done = {(r["id"], r["sample"]) for r in jsonl_read(out)}

    backend = get_backend(a.model, a.backend,
                          max_model_len=a.max_model_len or a.max_tokens + 2048,
                          gpu_memory_utilization=a.gpu_mem,
                          enforce_eager=a.eager)
    chunk = a.chunk_size or (1 if backend.name == "mlx" else 12)

    items = []
    for t in TASKS:
        for s in range(a.n):
            if (t["id"], s) in done:
                continue
            user = TEMPLATE.format(spec=t["spec"], buggy=t["buggy"].strip())
            items.append({
                "id": t["id"], "sample": s, "tests": t["tests"],
                "seed": (abs(hash(t["id"])) % 10000) + s,
                "prompt": build_prompt(backend.tokenizer, user,
                                       thinking=not a.no_thinking),
            })

    print(f"[bugfix] {a.model} backend={backend.name} n={a.n} chunk={chunk} "
          f"todo={len(items)} (skipped {len(done)}) -> {out}", flush=True)
    if not items:
        return score(out)

    def record(item, g):
        _, answer = strip_think(g["text"])
        code = extract_code(answer or g["text"])
        ok, detail = run_tests(code, item["tests"])
        return {"id": item["id"], "sample": item["sample"], "correct": ok,
                "truncated": g["truncated"], "gen_tokens": g["gen_tokens"],
                "seconds": g["seconds"], "detail": detail[:400], "code": code[:2000]}

    run_chunked(backend, items, out, chunk, a.max_tokens, a.temp, a.top_p,
                record, label=f"bugfix/{a.model}")
    score(out)


def score(path):
    rows = jsonl_read(path)
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    print(f"\n== {Path(path).name} ==")
    for k in sorted(by):
        v = by[k]
        print(f"  {k:<14} {sum(x['correct'] for x in v)}/{len(v)}")
    pass_at_1 = 100 * sum(r["correct"] for r in rows) / len(rows)
    solved = sum(any(x["correct"] for x in v) for v in by.values())
    print(f"  pass@1(avg over samples)={pass_at_1:.1f}%   "
          f"tasks solved at least once={solved}/{len(by)}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        for p in sys.argv[2:]:
            score(p)
    else:
        main()

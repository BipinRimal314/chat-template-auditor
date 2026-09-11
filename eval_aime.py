"""AIME 2025 / 2026 reproduction check.

The card claims MiniCPM5-2B = 86.5 on both exams and Qwen3.5-2B = 29.6 / 29.0.
Both models run through an identical harness with the reasoning flag set
explicitly, which is the specific confound the card's baseline numbers are
suspected of. --no-thinking reproduces what a default harness would have done.
"""
import argparse, sys
from pathlib import Path

import datasets
from common import (get_backend, build_prompt, strip_think, extract_boxed,
                    as_aime_int, jsonl_read, run_chunked)

PROMPT = ("{problem}\n\nPlease reason step by step, and put your final answer "
          "within \\boxed{{}}.")

DATASETS = {"2025": "MathArena/aime_2025", "2026": "MathArena/aime_2026"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--year", default="2025", choices=list(DATASETS))
    ap.add_argument("--n", type=int, default=4, help="samples per problem")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "vllm"])
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="prompts per batch; defaults to 1 on mlx, 24 on vllm")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or f"results/aime{a.year}_{a.model}{'_nothink' if a.no_thinking else ''}.jsonl"
    done = {(r["idx"], r["sample"]) for r in jsonl_read(out)}

    ds = datasets.load_dataset(DATASETS[a.year], split="train")
    if a.limit:
        ds = ds.select(range(a.limit))

    backend = get_backend(a.model, a.backend,
                          max_model_len=a.max_model_len or a.max_tokens + 2048,
                          gpu_memory_utilization=a.gpu_mem)
    chunk = a.chunk_size or (1 if backend.name == "mlx" else 24)

    items = []
    for row in ds:
        idx, gold = str(row["problem_idx"]), as_aime_int(row["answer"])
        for s in range(a.n):
            if (idx, s) in done:
                continue
            items.append({
                "idx": idx, "sample": s, "gold": gold,
                "seed": 1000 * s + int(idx),
                "prompt": build_prompt(backend.tokenizer,
                                       PROMPT.format(problem=row["problem"]),
                                       thinking=not a.no_thinking),
            })

    print(f"[aime{a.year}] {a.model} backend={backend.name} n={a.n} "
          f"chunk={chunk} todo={len(items)} (skipped {len(done)}) -> {out}", flush=True)
    if not items:
        return score(out)

    def record(item, g):
        _, answer_part = strip_think(g["text"])
        pred = as_aime_int(extract_boxed(answer_part or g["text"]))
        return {"idx": item["idx"], "sample": item["sample"], "gold": item["gold"],
                "pred": pred, "correct": pred is not None and pred == item["gold"],
                "truncated": g["truncated"], "gen_tokens": g["gen_tokens"],
                "seconds": g["seconds"], "gen_tps": g["gen_tps"],
                "tail": g["text"][-400:]}

    run_chunked(backend, items, out, chunk, a.max_tokens, a.temp, a.top_p,
                record, label=f"aime{a.year}/{a.model}")
    score(out)


def score(path):
    rows = jsonl_read(path)
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault(r["idx"], []).append(r)
    per = {k: sum(x["correct"] for x in v) / len(v) for k, v in by.items()}
    avg = 100 * sum(per.values()) / len(per)
    trunc = 100 * sum(r["truncated"] for r in rows) / len(rows)
    toks = sum(r["gen_tokens"] for r in rows) / len(rows)
    print(f"\n== {Path(path).name} ==")
    print(f"problems={len(per)}  gens={len(rows)}  avg@n={avg:.1f}  "
          f"truncated={trunc:.1f}%  mean_gen_tokens={toks:.0f}")
    return avg


if __name__ == "__main__":
    if sys.argv[1:2] == ["score"]:
        for p in sys.argv[2:]:
            score(p)
    else:
        main()

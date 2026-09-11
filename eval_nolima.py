"""NoLiMa-style long-context test with a fully synthetic haystack.

The card claims NoLiMa 68.1 for MiniCPM5-2B while every baseline sits near zero
(Qwen3.5-2B 17.1, LFM2.5-2.6B 0.7). Near-zero baselines usually mean a broken
harness rather than a real capability gap, so both models run the same generator.

The haystack is generated here rather than drawn from books, so no model can have
memorised it. Needles require a one-hop world-knowledge association and share no
content words with the question, which is the property NoLiMa isolates.
"""
import argparse, random, time
from pathlib import Path

from common import (get_backend, build_prompt, strip_think,
                    jsonl_read, run_chunked)

# (needle sentence, question, answer name) - the needle never contains the
# question's key term, so lexical matching alone cannot retrieve it.
NEEDLES = [
    ("Then Mira mentioned that she had spent the whole morning inside the Semperoper.",
     "Which character has been to Dresden?", "Mira"),
    ("Yusuf admitted that the Sagrada Familia had taken far longer to tour than he expected.",
     "Which character has been to Barcelona?", "Yusuf"),
    ("Elif kept a small bottle of insulin in her bag at all times.",
     "Which character has diabetes?", "Elif"),
    ("Tomas explained that he never eats anything containing gluten.",
     "Which character cannot eat bread made from wheat?", "Tomas"),
    ("Ravi had watched the sun set over the Bosphorus every evening that week.",
     "Which character was in Istanbul?", "Ravi"),
    ("Nadia's apartment looked out directly over the Charles Bridge.",
     "Which character lives in Prague?", "Nadia"),
    ("Ingrid complained that the altitude at Machu Picchu had left her breathless.",
     "Which character travelled to Peru?", "Ingrid"),
    ("Kofi practised his lines standing in front of the Globe Theatre.",
     "Which character was in London?", "Kofi"),
    ("Sasha said the Hermitage had taken two full days to walk through.",
     "Which character visited Saint Petersburg?", "Sasha"),
    ("Leila always carried an EpiPen with her because of peanuts.",
     "Which character has a severe food allergy?", "Leila"),
]

FILLER_NAMES = ["Arun", "Beatriz", "Caleb", "Dagny", "Emeka", "Farida", "Goran",
                "Hana", "Idris", "Junia", "Kwame", "Lorena", "Mateo", "Nils",
                "Oksana", "Pilar", "Quentin", "Rosa", "Stefan", "Tariq", "Ulla",
                "Viktor", "Wren", "Ximena", "Yara", "Zoltan"]
FILLER_ACTS = [
    "spent the afternoon reorganising the shelves in the back room",
    "brought a thermos of tea to the meeting and forgot to drink it",
    "argued that the schedule had been printed with the wrong dates",
    "left a note on the door explaining the delay",
    "counted the chairs twice and still came up one short",
    "borrowed a stapler and never returned it",
    "repainted the corridor a slightly different shade of grey",
    "wrote down every question but asked none of them",
    "kept checking whether the window had been latched properly",
    "offered to carry the boxes down the stairs instead of using the lift",
    "read the same paragraph four times without absorbing it",
    "suggested moving the whole thing to a Thursday",
    "found an old receipt folded inside a library book",
    "insisted the coffee machine had been making a new sound",
    "sorted the paperwork into piles that made sense only to them",
]


def make_haystack(target_tokens, tokenizer, needle, depth, seed):
    rng = random.Random(seed)
    lines = []
    n = 0
    # Build filler until slightly past the target, then splice the needle in.
    while n < target_tokens:
        s = f"{rng.choice(FILLER_NAMES)} {rng.choice(FILLER_ACTS)}."
        lines.append(s)
        n += len(tokenizer.encode(s))
    pos = max(0, min(len(lines) - 1, int(len(lines) * depth)))
    lines.insert(pos, needle)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="2000,8000,32000,64000")
    ap.add_argument("--depths", default="0.25,0.75")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "vllm"])
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lengths = [int(x) for x in a.lengths.split(",")]
    depths = [float(x) for x in a.depths.split(",")]
    out = a.out or f"results/nolima_{a.model}{'_nothink' if a.no_thinking else ''}.jsonl"
    done = {(r["length"], r["depth"], r["needle"]) for r in jsonl_read(out)}

    # Context has to cover the longest haystack plus the answer budget. On an 8 GB
    # card MiniCPM5 cannot hold a 64k cache at bf16 at all; run that length on a
    # larger-memory machine rather than quantising, which would confound the test.
    need = max(lengths) + a.max_tokens + 2048
    backend = get_backend(a.model, a.backend,
                          max_model_len=a.max_model_len or need,
                          gpu_memory_utilization=a.gpu_mem)
    chunk = a.chunk_size or (1 if backend.name == "mlx" else 8)

    items = []
    for L in lengths:
        for d in depths:
            for ni, (needle, question, gold) in enumerate(NEEDLES):
                if (L, d, ni) in done:
                    continue
                hay = make_haystack(L, backend.tokenizer, needle, d,
                                    seed=ni * 97 + L + int(d * 100))
                user = (f"Read the log below and answer the question.\n\n"
                        f"<log>\n{hay}\n</log>\n\n{question}\n"
                        f"Answer with the character's name only.")
                prompt = build_prompt(backend.tokenizer, user,
                                      thinking=not a.no_thinking)
                items.append({"length": L, "depth": d, "needle": ni, "gold": gold,
                              "seed": ni * 97 + L, "prompt": prompt,
                              "prompt_tokens": len(backend.tokenizer.encode(prompt))})

    print(f"[nolima] {a.model} backend={backend.name} lengths={lengths} "
          f"chunk={chunk} todo={len(items)} (skipped {len(done)}) -> {out}", flush=True)
    if not items:
        return score(out)

    def record(item, g):
        _, ans = strip_think(g["text"])
        final = (ans or g["text"]).strip()
        ok = item["gold"].lower() in final.lower()[-200:]
        return {"length": item["length"], "depth": item["depth"],
                "needle": item["needle"], "gold": item["gold"], "correct": ok,
                "prompt_tokens": item["prompt_tokens"],
                "gen_tokens": g["gen_tokens"], "truncated": g["truncated"],
                "seconds": g["seconds"], "answer": final[-300:]}

    run_chunked(backend, items, out, chunk, a.max_tokens, a.temp, a.top_p,
                record, label=f"nolima/{a.model}")
    score(out)


def score(path):
    rows = jsonl_read(path)
    if not rows:
        return
    print(f"\n== {Path(path).name} ==")
    lens = sorted({r["length"] for r in rows})
    for L in lens:
        sub = [r for r in rows if r["length"] == L]
        acc = 100 * sum(r["correct"] for r in sub) / len(sub)
        pt = sum(r["prompt_tokens"] for r in sub) / len(sub)
        print(f"  {L:>6} tokens (ctx~{pt:.0f}): {acc:5.1f}%  n={len(sub)}")
    print(f"  overall: {100*sum(r['correct'] for r in rows)/len(rows):.1f}%")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["score"]:
        for p in sys.argv[2:]:
            score(p)
    else:
        main()

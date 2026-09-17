"""Answer parsing, result storage, and the chunked work loop shared by every eval.

Work is dispatched in chunks rather than one prompt at a time so the vLLM backend
can batch, while results are still flushed to disk after every chunk. That keeps
a long run resumable: re-running skips whatever already landed in the JSONL.
"""
import json, re
from pathlib import Path

from backends import get_backend, build_prompt, MODELS  # re-exported for scripts


def strip_think(text):
    """Return (reasoning, answer_part). Handles a dangling <think> with no close tag."""
    if "</think>" in text:
        head, tail = text.split("</think>", 1)
        return head.replace("<think>", "").strip(), tail.strip()
    return text.strip(), ""


def extract_boxed(text):
    """Last \\boxed{...} with brace matching, falling back to a trailing integer."""
    idx = text.rfind("\\boxed")
    if idx != -1:
        i = text.find("{", idx)
        if i != -1:
            depth, j = 0, i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i + 1 : j].strip()
                j += 1
    m = re.findall(r"(-?\d+)", text)
    return m[-1] if m else None


def as_aime_int(s):
    """AIME answers are integers 0-999; normalise \\text{}, commas, latex noise."""
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}|\\!|\\,|\\ |\$|,", "", str(s)).strip()
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def jsonl_append(path, recs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Cut off a half-written final line before appending, or the next record
    # would be glued onto it and the corruption would stop being the last line.
    p = Path(path)
    if p.exists() and p.stat().st_size:
        data = p.read_bytes()
        if not data.endswith(b"\n"):
            with open(p, "r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)
    with open(path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def jsonl_read(path):
    """A power cut mid-append can leave a half-written final line. Drop it so the
    run resumes and regenerates that record; corruption anywhere else still raises."""
    p = Path(path)
    if not p.exists():
        return []
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    rows = []
    for i, l in enumerate(lines):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f"[jsonl] dropping truncated last line of {p.name}", flush=True)
    return rows


def run_chunked(backend, items, out_path, chunk_size, max_tokens, temp, top_p,
                make_record, label=""):
    """items: list of dicts each carrying at least `prompt` and `seed`.
    make_record(item, generation) -> the dict written to the JSONL."""
    import time
    total, done, t0 = len(items), 0, time.time()
    for i in range(0, total, chunk_size):
        chunk = items[i : i + chunk_size]
        gens = backend.generate([c["prompt"] for c in chunk], max_tokens,
                                temp=temp, top_p=top_p,
                                seeds=[c["seed"] for c in chunk])
        recs = [make_record(c, g) for c, g in zip(chunk, gens)]
        jsonl_append(out_path, recs)
        done += len(chunk)
        ok = sum(1 for r in recs if r.get("correct"))
        print(f"  [{label}] {done}/{total}  chunk_correct={ok}/{len(chunk)}  "
              f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)

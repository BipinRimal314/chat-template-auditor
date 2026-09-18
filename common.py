"""Answer parsing, result storage, and the chunked work loop shared by every eval.

Work is dispatched in chunks rather than one prompt at a time so the vLLM backend
can batch, while results are still flushed to disk after every chunk. That keeps
a long run resumable: re-running skips whatever already landed in the JSONL.
"""
import json, os, re
from pathlib import Path

from backends import get_backend, build_prompt, MODELS  # re-exported for scripts


def strip_think(text):
    """Return (reasoning, answer_part). Handles a dangling <think> with no close tag."""
    if "</think>" in text:
        head, tail = text.split("</think>", 1)
        return head.replace("<think>", "").strip(), tail.strip()
    return text.strip(), ""


def extract_boxed(text):
    """Last *complete* \\boxed{...} with brace matching, falling back to a
    trailing integer.

    An answer cut off at the token cap can end inside a box, e.g. a loop of
    "\\boxed{16}" stopped at "\\boxed{1". Taking the last box regardless graded
    that fragment as 1, so the unclosed tail is skipped and the last closed box
    wins; the integer fallback also ignores the unclosed fragment.
    """
    end = len(text)
    while True:
        idx = text.rfind("\\boxed", 0, end)
        if idx == -1:
            break
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
        if end == len(text):
            text = text[:idx]      # drop the unclosed fragment for the fallback
        end = idx
    m = re.findall(r"(-?\d+)", text)
    return m[-1] if m else None


def looping(text, window=3000, max_period=500):
    """True when the last `window` characters are one block of at most
    `max_period` characters repeated exactly, i.e. at least window/max_period = 6
    identical copies back to back. Deliberately strict: an answer that is merely
    long, or enumerating cases that differ, never matches, only exact repetition.
    Continuing a periodic tail only adds more copies of what is already there,
    so stopping it cannot change the answer that gets extracted.
    """
    if len(text) < window:
        return False
    tail = text[-window:]
    return any(tail[:-p] == tail[p:] for p in range(1, max_period + 1))


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
        # Force it to disk: after a power cut, unsynced appends came back as NUL
        # bytes in gpu.log, and a lost record means a regenerated answer.
        f.flush()
        os.fsync(f.fileno())


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
                make_record, label="", stop_loops=False):
    """items: list of dicts each carrying at least `prompt` and `seed`.
    make_record(item, generation) -> the dict written to the JSONL.

    Backends that can stream (vLLM) save every answer the moment it finishes,
    with at most `chunk_size` in flight, so a power cut costs only the answers
    still generating. Others (MLX) run in chunks and save after each chunk.
    """
    import time
    total, done, ok, t0 = len(items), 0, 0, time.time()

    def report(extra=""):
        print(f"  [{label}] {done}/{total}  correct={ok}/{done}  {extra}"
              f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    if hasattr(backend, "generate_stream"):
        stream = backend.generate_stream([c["prompt"] for c in items], max_tokens,
                                         temp=temp, top_p=top_p,
                                         seeds=[c["seed"] for c in items],
                                         window=chunk_size,
                                         stop_loops=stop_loops)
        for i, g in stream:
            rec = make_record(items[i], g)
            jsonl_append(out_path, [rec])
            done += 1
            ok += bool(rec.get("correct"))
            report(f"last={'ok' if rec.get('correct') else 'wrong'}"
                   f"{' CUT' if rec.get('truncated') else ''}"
                   f"{' LOOP' if rec.get('looped') else ''} {g['gen_tokens']}tok  ")
        return

    for i in range(0, total, chunk_size):
        chunk = items[i : i + chunk_size]
        gens = backend.generate([c["prompt"] for c in chunk], max_tokens,
                                temp=temp, top_p=top_p,
                                seeds=[c["seed"] for c in chunk])
        recs = [make_record(c, g) for c, g in zip(chunk, gens)]
        jsonl_append(out_path, recs)
        done += len(chunk)
        ok += sum(1 for r in recs if r.get("correct"))
        report()

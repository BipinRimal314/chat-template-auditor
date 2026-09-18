"""Live progress for the AIME sweep in results/vllm-4060.

  .venv/bin/python progress.py            # one snapshot
  watch -n 30 .venv/bin/python progress.py

Reads only what is already on disk, so it is safe to run at any time and never
touches the GPU. Accuracy is over finished attempts only; early numbers are noisy
and lean towards short answers, because short answers finish first.
"""
import json, os, re, subprocess, time
from datetime import datetime
from pathlib import Path

OUT = Path(os.environ.get("OUT") or Path(__file__).parent / "results" / "vllm-4060")
PROBLEMS = 30
N = int(os.environ.get("N", 2))  # attempts per problem; must match run_aime_4060.sh
CARD = {"qwen_nothink": 29.6, "qwen_think": 29.6, "minicpm_think": 86.5}
STAGES = [
    ("qwen_nothink",  "Qwen3.5-2B, thinking off", "aime2025_qwen3.5-2b_nothink.jsonl"),
    ("qwen_think",    "Qwen3.5-2B, thinking on",  "aime2025_qwen3.5-2b.jsonl"),
    ("minicpm_think", "MiniCPM5-2B, thinking on", "aime2025_minicpm5-2b.jsonl"),
]


def read_rows(p):
    rows = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def stage_times():
    """Per stage: (first start, last end or None, total running seconds).

    A stage can run in several sessions when the sweep is stopped and resumed.
    A session that ends without an "end" line (killed, power cut) is taken to
    have stopped at the last GPU log reading before the next start, so the
    overnight gap is not counted as running time.
    """
    events = []
    log = OUT / "sweep.log"
    if log.exists():
        for line in log.read_text().splitlines():
            m = re.match(r"=== (\S+ \S+) (start|end) (\w+)", line)
            if m:
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                events.append((t, m.group(2), m.group(3)))
    ticks = []
    gpu = OUT / "gpu.log"
    if gpu.exists():
        for line in gpu.read_text(errors="replace").splitlines():
            try:
                ticks.append(datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").timestamp())
            except ValueError:
                pass
    starts = [t for t, kind, _ in events if kind == "start"]
    out = {}
    for i, (t, kind, name) in enumerate(events):
        if kind != "start":
            continue
        end = next((u for u, k, n in events[i + 1:] if k == "end" and n == name), None)
        nxt = next((u for u in starts if u > t), None)
        if end is not None and (nxt is None or end <= nxt):
            stop, finished = end, True
        elif nxt is not None:
            stop = max([u for u in ticks if t <= u < nxt], default=t)
            finished = False
        else:
            stop, finished = time.time(), False
        first, _, total = out.get(name, (t, None, 0.0))
        out[name] = (first, stop if finished else None, total + (stop - t))
    return out


def in_flight(name):
    """The last tqdm counter vLLM printed for the chunk it is working on."""
    log = OUT / f"{name}.log"
    if not log.exists():
        return None
    tail = log.read_bytes()[-4000:].decode("utf-8", "replace")
    hits = re.findall(r"Processed prompts:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)", tail)
    return f"{hits[-1][0]}/{hits[-1][1]}" if hits else None


def hm(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m"


def gpu_live():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        util, temp = (int(x) for x in out.strip().splitlines()[0].split(","))
        return util, temp
    except Exception:
        return None, None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def runner_alive():
    r = subprocess.run(["pgrep", "-f", "^bash ./run_aime_4060.sh"], capture_output=True)
    return r.returncode == 0


def health(times, now):
    """Working, stuck, or stopped, from signals that do not depend on a result
    being saved. A batch of 32k-token answers can go most of an hour without
    saving anything, so a quiet results file alone proves nothing."""
    print("\nHEALTH")
    pgid = OUT / "stage.pgid"
    pid = int(pgid.read_text().strip()) if pgid.exists() and pgid.read_text().strip() else None
    util, temp = gpu_live()
    proc_ok = pid is not None and alive(pid)

    saves = [p.stat().st_mtime for p in OUT.glob("aime2025_*.jsonl")]
    logs = [p.stat().st_mtime for p in OUT.glob("*_think*.log")] + \
           [p.stat().st_mtime for p in OUT.glob("*_nothink.log")]
    since_save = f"saved {hm(now - max(saves))} ago" if saves else "none saved yet"
    since_log = hm(now - max(logs)) if logs else "-"

    print(f"  eval process   {'running (pid %d)' % pid if proc_ok else 'not running'}")
    print(f"  GPU busy       {'%d%%' % util if util is not None else 'unknown'}"
          f"{'   temp %dC' % temp if temp is not None else ''}")
    print(f"  last result    {since_save}")
    print(f"  last log line  {since_log} ago")

    done = "sweep complete" in (OUT / "sweep.log").read_text() if (OUT / "sweep.log").exists() else False
    if done:
        verdict = "FINISHED. Every stage completed."
    elif (OUT / "STOP").exists():
        verdict = "STOPPED by the heat watchdog. Let the card cool, then restart."
    elif not proc_ok and runner_alive():
        last = (OUT / "sweep.log").read_text().strip().splitlines()[-1:] or [""]
        if "waiting" in last[0] or "refused" in last[0]:
            verdict = ("WAITING. The runner is alive and waiting for GPU memory to free up, "
                       "usually because the screen is locked. It retries every minute.")
        else:
            verdict = "STARTING. The runner is alive and between stages or loading a model."
    elif not proc_ok:
        verdict = ("STOPPED. No eval process is running but the sweep did not finish. "
                   "Check the stage log for an error, then restart; finished work is kept.")
    elif util is not None and util < 20:
        verdict = ("CHECK. The process is alive but the GPU is idle. Brief dips happen while "
                   "a batch is saved or a model loads; if this persists past 10 minutes, it is stuck.")
    else:
        verdict = "WORKING. The process is alive and the GPU is busy generating."
    print(f"\n  {verdict}")


def main():
    times = stage_times()
    now = time.time()
    print(f"AIME 2025 sweep on RTX 4060   {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    head = f"{'stage':<26}{'done':>9}{'correct':>9}{'score':>8}{'card':>7}{'cut off':>9}{'looped':>8}{'avg tok':>9}{'elapsed':>9}{'eta':>9}"
    print(head)
    print("-" * len(head))
    for name, label, fname in STAGES:
        rows = read_rows(OUT / fname)
        n = len(rows)
        ok = sum(r["correct"] for r in rows)
        cut = sum(r["truncated"] for r in rows)
        loops = sum(r.get("looped", False) for r in rows)
        # Problems done before the switch to N=2 kept 4 attempts; count what
        # is actually planned, not 30 x N.
        have = {}
        for r in rows:
            have.setdefault(r["idx"], set()).add(r["sample"])
        TOTAL = n + sum(len(set(range(N)) - have.get(str(k), set()))
                        for k in range(1, PROBLEMS + 1))
        tok = sum(r["gen_tokens"] for r in rows) / n if n else 0
        score = f"{100 * ok / n:.1f}" if n else "-"
        if name in times:
            _, finished_at, el = times[name]
            status = "done" if finished_at else "running"
            eta = "-" if status == "done" or not n else hm(el / n * (TOTAL - n))
            el_s = hm(el)
        else:
            status, el_s, eta = "waiting", "-", "-"
        print(f"{label:<26}{n:>5}/{TOTAL:<3}{ok:>9}{score:>8}{CARD[name]:>7}{cut:>9}{loops:>8}{tok:>9.0f}{el_s:>9}{eta:>9}")
        if status == "running":
            f = in_flight(name)
            if f:
                print(f"{'':<26}current batch {f} finished")

    gpu = OUT / "gpu.log"
    if gpu.exists():
        # A power cut can leave NUL bytes or a half line at the end; keep only
        # well-formed readings.
        lines = [l for l in gpu.read_text(errors="replace").splitlines()
                 if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \S", l)]
        temps = [int(m.group(1)) for l in lines if (m := re.search(r"temp=(\d+)", l))]
        if temps:
            print(f"\nGPU  now {lines[-1].split(' ', 2)[2]}   peak temp {max(temps)}C   watchdog stops at 85C")
        if any("WATCHDOG" in l for l in lines):
            print("WATCHDOG STOPPED THE RUN. Restart with the command in run_aime_4060.sh.")
    health(times, now)
    print("\nscore = % of finished attempts correct. Early scores are noisy and favour short answers.")
    print("cut off = hit the 32,768-token limit. looped = stopped early for exact repetition.")
    print("score counts every attempt equally; problems done before the switch to 2 attempts have 4.")


if __name__ == "__main__":
    main()

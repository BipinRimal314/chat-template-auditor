"""chat-template-auditor - find models whose default template silently changes
what an evaluation measures.

A benchmark comparison is only meaningful if every model was asked the same
question under the same conditions. Chat templates decide one of those
conditions - whether the model is allowed to reason before answering - and they
do not agree on what happens when a harness does not pass the flag. This finds
the disagreements.
"""
import argparse, json, sys, concurrent.futures as cf, os, re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import fetch_template, model_meta
from render import audit as audit_template

# What the prefill state means for an evaluator.
VERDICT = {
    "OPEN":   ("forced-on",  "template pre-opens a reasoning block; the model must reason"),
    "CLOSED": ("suppressed", "template injects a closed empty reasoning block; reasoning is suppressed"),
    "NONE":   ("model-choice", "no reasoning prefill; the model decides"),
}


def family(repo):
    """Group by vendor plus model line, so Qwen3.5-2B and Qwen3.5-4B compare but
    Qwen3.5 and Qwen3 do not."""
    org, _, name = repo.partition("/")
    m = re.match(r"([A-Za-z][A-Za-z.\-]*?[0-9]+(?:\.[0-9]+)?)", name)
    return f"{org}/{m.group(1)}" if m else f"{org}/{name.split('-')[0]}"


def audit_repo(repo, token=None):
    tpl, src, status = fetch_template(repo, token)
    row = {"repo": repo, "family": family(repo), "status": status, "template_file": src}
    if status != "ok":
        return row
    try:
        a = audit_template(tpl)
    except Exception as e:
        row["status"] = "render-error"
        row["error"] = f"{type(e).__name__}: {e}"[:200]
        return row
    label, why = VERDICT.get(a["default"], ("unknown", ""))
    row.update({
        "default_state": a["default"],
        "verdict": label,
        "why": why,
        "honors_flag": a["honors_flag"],
        "unset_matches_thinking_on": a["unset_matches_thinking_on"],
        "states": a["states"],
        "convention": (a["detail"].get("true") or {}).get("convention"),
    })
    return row


def find_inversions(rows):
    """The high-value signal: two models from the same family that disagree on
    what an unset flag does. A harness sweeping that family produces numbers
    that are not comparable to each other."""
    byfam = defaultdict(list)
    for r in rows:
        if r.get("default_state"):
            byfam[r["family"]].append(r)
    out = []
    for fam, members in sorted(byfam.items()):
        states = {m["default_state"] for m in members}
        if len(states) > 1 and len(members) > 1:
            out.append({"family": fam, "states": sorted(states),
                        "members": sorted((m["repo"], m["default_state"]) for m in members)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="*", help="model repo ids, or - to read stdin")
    ap.add_argument("--file", help="file of repo ids, one per line")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", dest="json_out", help="write full results here")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token, to reach gated repos")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    repos = list(a.repos)
    if a.file:
        repos += [l.strip() for l in open(a.file) if l.strip() and not l.startswith("#")]
    if repos == ["-"] or (not repos and not sys.stdin.isatty()):
        repos = [l.strip() for l in sys.stdin if l.strip()]
    if not repos:
        ap.error("no repos given")

    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(audit_repo, r, a.token): r for r in repos}
        for f in cf.as_completed(futs):
            row = f.result()
            rows.append(row)
            if not a.quiet:
                v = row.get("verdict", row["status"])
                flag = "" if row.get("unset_matches_thinking_on", True) else "  <-- default != thinking-on"
                print(f"{v:<13} {row['repo']:<48}{flag}", flush=True)

    rows.sort(key=lambda r: (r.get("verdict", "zzz"), r["repo"]))
    report(rows)
    if a.json_out:
        json.dump({"rows": rows, "inversions": find_inversions(rows)},
                  open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")


def report(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r.get("verdict", r["status"])] += 1
    audited = [r for r in rows if r.get("default_state")]

    print("\n" + "=" * 66)
    print(f"audited {len(audited)} of {len(rows)} repos")
    for k in sorted(counts, key=lambda x: -counts[x]):
        print(f"  {k:<14} {counts[k]}")

    suppressed = [r for r in audited if r["default_state"] == "CLOSED"]
    if suppressed:
        print(f"\nreasoning suppressed by default ({len(suppressed)}):")
        print("  a harness that omits the flag measures these models with reasoning off")
        for r in suppressed:
            print(f"    {r['repo']}")

    inv = find_inversions(rows)
    if inv:
        print(f"\nfamily inversions ({len(inv)}):")
        print("  sibling models whose defaults disagree, so a family sweep is not self-consistent")
        for i in inv:
            print(f"    {i['family']}")
            for repo, st in i["members"]:
                print(f"        {st:<7} {repo}")
    print("=" * 66)


if __name__ == "__main__":
    main()

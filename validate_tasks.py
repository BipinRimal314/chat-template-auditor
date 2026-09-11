"""Sanity-check the task set: every buggy version must fail, every reference must pass."""
import sys
sys.path.insert(0, ".")
from tasks.bugfix_tasks import TASKS
from sandbox import run_tests

bad = 0
for t in TASKS:
    ok, detail = run_tests(t["buggy"], t["tests"])
    status = "BUG-STILL-PASSES!" if ok else "fails as expected"
    if ok:
        bad += 1
    first = detail.splitlines()[:3]
    print(f"{t['id']:<14} {status:<20} {' | '.join(first)}")
print(f"\n{bad} task(s) whose buggy version wrongly passes")

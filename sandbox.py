"""Run candidate code against hidden tests in a subprocess with a hard timeout."""
import subprocess, sys, tempfile, os
from pathlib import Path

RUNNER = '''
import sys, traceback
mod = {{}}
try:
    exec(compile(open({sol!r}).read(), "solution.py", "exec"), mod)
except Exception:
    traceback.print_exc(); print("IMPORT_FAIL"); sys.exit(2)
tests = {{}}
exec(compile(open({tst!r}).read(), "tests.py", "exec"), mod)
names = [k for k in list(mod) if k.startswith("test_")]
if not names:
    print("NO_TESTS"); sys.exit(3)
failed = []
for n in sorted(names):
    try:
        mod[n]()
    except Exception as e:
        failed.append(f"{{n}}: {{type(e).__name__}}: {{e}}")
print("PASSED" if not failed else "FAILED")
print(len(names) - len(failed), "/", len(names))
for f in failed: print("  -", f)
sys.exit(0 if not failed else 1)
'''


def run_tests(solution_code, test_code, timeout=15):
    """Returns (passed: bool, detail: str)."""
    with tempfile.TemporaryDirectory() as d:
        sol = Path(d) / "solution.py"
        tst = Path(d) / "tests.py"
        run = Path(d) / "run.py"
        sol.write_text(solution_code)
        tst.write_text(test_code)
        run.write_text(RUNNER.format(sol=str(sol), tst=str(tst)))
        try:
            p = subprocess.run([sys.executable, str(run)], capture_output=True,
                               text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT after {timeout}s"
        out = (p.stdout + p.stderr).strip()
        return p.returncode == 0, out[-1200:]

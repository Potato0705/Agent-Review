"""Line-coverage gate for ``agent_audit``, using only the standard library.

The package ships with no third-party dependencies, so coverage is measured
with :mod:`trace` rather than an external tool. The gate exists because most of
this project's safety claims live in refusal branches: code that only runs when
input is malformed. Those branches are easy to add and easy to leave untested,
and an untested refusal is indistinguishable from a missing one.

Run it directly, or through ``scripts/review.ps1``::

    python scripts/coverage_report.py
    python scripts/coverage_report.py --show-missing
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ratchets, not targets. Each floor sits just under the coverage that was
# actually measured when it was set, so deleting or weakening tests trips the
# gate while an ordinary refactor does not. Raise a floor once the matching
# tests exist; never lower one to turn a red run green.
OVERALL_FLOOR = 86.5
MODULE_FLOORS = {
    "audit": 98.8,
    "checkpoint": 83.2,
    "cli": 57.2,
    "comparison": 87.5,
    "html_report": 100.0,
    "io": 94.5,
    "models": 100.0,
    "provider": 76.3,
    "report": 100.0,
    "scoring": 91.3,
}

# Percentages are compared at the precision they are printed, so the table and
# the verdict can never contradict each other.
DISPLAY_PLACES = 1

RUNNER = """import sys, unittest
sys.path.insert(0, {root!r})
sys.path.insert(0, {tests!r})
suite = unittest.TestLoader().discover(start_dir={tests!r}, top_level_dir=None)
result = unittest.TextTestRunner(verbosity=0).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""


class ModuleCoverage:
    def __init__(self, name: str, executed: int, missing: list[tuple[int, str]]):
        self.name = name
        self.executed = executed
        self.missing = missing

    @property
    def total(self) -> int:
        return self.executed + len(self.missing)

    @property
    def percent(self) -> float:
        return 100.0 * self.executed / self.total if self.total else 100.0


def _run_traced_suite(cover_dir: Path, runner_path: Path) -> int:
    runner_path.write_text(
        RUNNER.format(
            root=str(PROJECT_ROOT), tests=str(PROJECT_ROOT / "tests")
        ),
        encoding="utf-8",
        newline="\n",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trace",
            "--count",
            "--missing",
            f"--coverdir={cover_dir}",
            f"--ignore-dir={sysconfig.get_paths()['stdlib']}",
            str(runner_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout or "")
        sys.stderr.write(completed.stderr or "")
    return completed.returncode


def _parse_cover_files(cover_dir: Path) -> list[ModuleCoverage]:
    modules: list[ModuleCoverage] = []
    for cover_file in sorted(cover_dir.glob("agent_audit.*.cover")):
        name = cover_file.stem.replace("agent_audit.", "")
        executed = 0
        missing: list[tuple[int, str]] = []
        text = cover_file.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.startswith(">>>>>>"):
                missing.append((number, line[7:].strip()))
            elif line.split(":", 1)[0].strip().isdigit():
                executed += 1
        modules.append(ModuleCoverage(name, executed, missing))
    return modules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print every line the test suite never executed.",
    )
    args = parser.parse_args(argv)

    work_dir = Path(tempfile.mkdtemp(prefix="agent-audit-coverage-"))
    try:
        cover_dir = work_dir / "cover"
        cover_dir.mkdir()
        suite_status = _run_traced_suite(cover_dir, work_dir / "run_suite.py")
        if suite_status != 0:
            print("Test suite failed under coverage tracing.", file=sys.stderr)
            return suite_status

        modules = _parse_cover_files(cover_dir)
        if not modules:
            print("No coverage data was produced.", file=sys.stderr)
            return 1

        executed = sum(module.executed for module in modules)
        total = sum(module.total for module in modules)
        overall = 100.0 * executed / total if total else 100.0

        print(f"{'module':<16}{'missed':>8}{'covered':>10}")
        failures: list[str] = []
        for module in modules:
            shown = round(module.percent, DISPLAY_PLACES)
            floor = MODULE_FLOORS.get(module.name)
            marker = ""
            if floor is not None and shown < floor:
                marker = f"  < floor {floor:.1f}%"
                failures.append(
                    f"{module.name}: {shown:.1f}% is below its {floor:.1f}% floor"
                )
            print(
                f"{module.name:<16}{len(module.missing):>8}{shown:>9.1f}%{marker}"
            )
        shown_overall = round(overall, DISPLAY_PLACES)
        print(f"{'OVERALL':<16}{total - executed:>8}{shown_overall:>9.1f}%")

        unknown = sorted(set(MODULE_FLOORS) - {module.name for module in modules})
        if unknown:
            failures.append(
                "floors name modules that produced no coverage data: "
                + ", ".join(unknown)
            )

        if shown_overall < OVERALL_FLOOR:
            failures.append(
                f"overall: {shown_overall:.1f}% is below the {OVERALL_FLOOR:.1f}% floor"
            )

        if args.show_missing:
            for module in modules:
                if not module.missing:
                    continue
                print(f"\n--- {module.name}: {len(module.missing)} uncovered ---")
                for number, source in module.missing:
                    print(f"  {number:>4}: {source}")

        if failures:
            print("\nCoverage gate failed:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1

        print("\nCoverage gate passed.")
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

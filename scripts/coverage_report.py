"""Line-coverage gate for ``agent_audit``, using only the standard library.

The package ships with no third-party dependencies, so coverage is measured
with :mod:`trace` rather than an external tool. The gate exists because most of
this project's safety claims live in refusal branches: code that only runs when
input is malformed. Those branches are easy to add and easy to leave untested,
and an untested refusal is indistinguishable from a missing one.

Run it directly, or through ``scripts/review.ps1``::

    python scripts/coverage_report.py
    python scripts/coverage_report.py --show-missing

Why this does not use ``python -m trace --ignore-dir``
------------------------------------------------------
``trace`` decides what to skip with :class:`trace._Ignore`, which is keyed by
the *bare file basename* and caches its answers. Once the standard library's
``io.py`` is skipped, the cache holds ``{"io": 1}``, so ``agent_audit/io.py``
is skipped too and silently vanishes from the report. Which file is seen first
depends on the platform, so the same command reported ten modules on Windows
and nine on Linux. This module therefore drives :class:`trace.Trace` directly
and decides what to record from the absolute path alone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import trace
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "agent_audit"
TESTS_DIR = PROJECT_ROOT / "tests"

# ``python -m agent_audit`` is a four-line shim. The unit suite imports the CLI
# directly, so the shim only runs in the packaging and offline-demo steps and
# would otherwise report as permanently uncovered.
EXCLUDED_MODULES = frozenset({"__main__"})

# Budgets are expressed as a maximum number of uncovered lines rather than a
# percentage. Interpreters disagree slightly about which lines carry bytecode —
# Python 3.10 and 3.12 differ by a line or two in several modules — and a
# percentage turns that into a swing whose size depends on how big the module
# is. A line budget says the thing we actually care about: how many executable
# lines no test ever touches.
#
# Measured on Windows/3.12, Ubuntu/3.12 and Ubuntu/3.10: all three agree on
# every count below, while the percentages differ by a tenth of a point. Each
# budget is that agreed count plus one line of slack for a future interpreter.
# Tighten a budget once the matching tests exist; never raise one to turn a red
# run green. Budgets must hold on the lowest supported interpreter, not just
# the one they were measured on.
MAX_UNCOVERED_LINES = {
    "__init__": 1,
    "audit": 3,
    "checkpoint": 2,
    "cli": 8,
    "comparison": 4,
    "html_report": 1,
    "io": 1,
    "models": 1,
    "provider": 2,
    "report": 1,
    "scoring": 1,
    "segmentation": 1,
}

# A coarse net underneath the per-module budgets, with headroom for the same
# interpreter differences.
OVERALL_FLOOR = 97.0

# Percentages are compared at the precision they are printed, so the table and
# the verdict can never contradict each other.
DISPLAY_PLACES = 1


class _PathScopedIgnore:
    """Record a frame only when its file lives inside the package.

    This deliberately ignores the module name that :mod:`trace` passes in.
    Deciding from the path alone is what keeps ``agent_audit/io.py`` from being
    mistaken for the standard library's ``io``.
    """

    def __init__(self, root: Path) -> None:
        self._root = os.path.normcase(str(root.resolve())) + os.sep

    def names(self, filename: str, modulename: str) -> int:
        try:
            resolved = os.path.normcase(os.path.abspath(filename))
        except (TypeError, ValueError):
            return 1
        return 0 if resolved.startswith(self._root) else 1


def module_name(source: Path) -> str:
    """Return a module's dotted name relative to the package directory.

    Discovery is recursive so that a future subpackage cannot slip past the
    gate: a module the tool never looks at reports no coverage at all, which
    is indistinguishable from having none.
    """

    relative = source.resolve().relative_to(PACKAGE_DIR.resolve()).with_suffix("")
    return ".".join(relative.parts)


def _executable_linenos(path: Path) -> set[int]:
    """Return every line number that carries bytecode."""

    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    linenos: set[int] = set()
    pending: list[types.CodeType] = [code]
    while pending:
        obj = pending.pop()
        for _, _, lineno in obj.co_lines():
            if lineno:
                linenos.add(lineno)
        pending.extend(
            const for const in obj.co_consts if isinstance(const, types.CodeType)
        )
    return linenos


def _collect(output_path: Path) -> int:
    """Run the suite under the tracer and record which package lines ran."""

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(TESTS_DIR))

    tracer = trace.Trace(count=1, trace=0)
    tracer.ignore = _PathScopedIgnore(PACKAGE_DIR)  # type: ignore[assignment]

    outcome: dict[str, bool] = {}

    def run_suite() -> None:
        # Discovery must happen inside the traced region: importing the test
        # modules is what executes the package's module-level code, its class
        # bodies and every `def` line. Discovering first would leave all of
        # that unrecorded and understate every module.
        suite = unittest.TestLoader().discover(start_dir=str(TESTS_DIR))
        outcome["ok"] = unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful()

    tracer.runfunc(run_suite)

    executed: dict[str, list[int]] = {}
    for filename, lineno in tracer.results().counts:
        executed.setdefault(os.path.abspath(filename), []).append(lineno)

    output_path.write_text(
        json.dumps({"ok": outcome.get("ok", False), "executed": executed}),
        encoding="utf-8",
        newline="\n",
    )
    return 0 if outcome.get("ok") else 1


class ModuleCoverage:
    def __init__(self, name: str, executed: set[int], executable: set[int]) -> None:
        self.name = name
        self.covered = executed & executable
        self.missing = sorted(executable - executed)
        self.total = len(executable)

    @property
    def percent(self) -> float:
        return 100.0 * len(self.covered) / self.total if self.total else 100.0


def _measure(executed_by_file: dict[str, list[int]]) -> list[ModuleCoverage]:
    normalised = {
        os.path.normcase(path): set(lines) for path, lines in executed_by_file.items()
    }
    modules: list[ModuleCoverage] = []
    for source in sorted(PACKAGE_DIR.rglob("*.py")):
        name = module_name(source)
        if name in EXCLUDED_MODULES:
            continue
        key = os.path.normcase(str(source.resolve()))
        modules.append(
            ModuleCoverage(name, normalised.get(key, set()), _executable_linenos(source))
        )
    return modules


def _report(modules: list[ModuleCoverage], show_missing: bool) -> int:
    covered = sum(len(module.covered) for module in modules)
    total = sum(module.total for module in modules)
    overall = round(100.0 * covered / total if total else 100.0, DISPLAY_PLACES)

    failures: list[str] = []
    print(f"{'module':<16}{'missed':>8}{'budget':>8}{'covered':>10}")
    for module in modules:
        shown = round(module.percent, DISPLAY_PLACES)
        budget = MAX_UNCOVERED_LINES.get(module.name)
        uncovered = len(module.missing)
        marker = ""
        if module.total and not module.covered:
            # A module with no recorded lines at all is almost always a
            # measurement failure rather than a real result.
            failures.append(f"{module.name}: no lines were recorded at all")
            marker = "  no data"
        elif budget is None:
            failures.append(f"{module.name}: has no budget in MAX_UNCOVERED_LINES")
            marker = "  no budget"
        elif uncovered > budget:
            failures.append(
                f"{module.name}: {uncovered} uncovered lines exceed its budget of {budget}"
            )
            marker = "  over budget"
        budget_text = "-" if budget is None else str(budget)
        print(
            f"{module.name:<16}{uncovered:>8}{budget_text:>8}{shown:>9.1f}%{marker}"
        )
    print(f"{'OVERALL':<16}{total - covered:>8}{'':>8}{overall:>9.1f}%")

    measured = {module.name for module in modules}
    unknown = sorted(set(MAX_UNCOVERED_LINES) - measured)
    if unknown:
        failures.append("budgets name modules that do not exist: " + ", ".join(unknown))

    if overall < OVERALL_FLOOR:
        failures.append(
            f"overall: {overall:.1f}% is below the {OVERALL_FLOOR:.1f}% floor"
        )

    if show_missing:
        for module in modules:
            if not module.missing:
                continue
            source = (
                PACKAGE_DIR.joinpath(*module.name.split("."))
                .with_suffix(".py")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            print(f"\n--- {module.name}: {len(module.missing)} uncovered ---")
            for lineno in module.missing:
                text = source[lineno - 1].strip() if lineno <= len(source) else ""
                print(f"  {lineno:>4}: {text}")

    if failures:
        print("\nCoverage gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nCoverage gate passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print every line the test suite never executed.",
    )
    parser.add_argument(
        "--collect",
        metavar="PATH",
        help=argparse.SUPPRESS,  # internal: the traced child writes JSON here
    )
    args = parser.parse_args(argv)

    if args.collect:
        return _collect(Path(args.collect))

    with tempfile.TemporaryDirectory(prefix="agent-audit-coverage-") as work:
        data_path = Path(work) / "coverage.json"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--collect", str(data_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        if not data_path.exists():
            sys.stderr.write(completed.stdout or "")
            sys.stderr.write(completed.stderr or "")
            print("The traced run produced no coverage data.", file=sys.stderr)
            return 1

        payload = json.loads(data_path.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            sys.stderr.write(completed.stdout or "")
            sys.stderr.write(completed.stderr or "")
            print("Test suite failed under coverage tracing.", file=sys.stderr)
            return 1

        return _report(_measure(payload["executed"]), args.show_missing)


if __name__ == "__main__":
    raise SystemExit(main())

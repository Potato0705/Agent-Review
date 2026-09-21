"""Guards for the coverage gate itself.

The gate is only worth having if it measures every module. It previously used
``python -m trace --ignore-dir``, whose skip decision is keyed by the bare file
basename and cached, so the standard library's ``io`` caused ``agent_audit/io``
to be dropped from the report on Linux while Windows still listed it. The
replacement decides from the absolute path, and these tests pin that contract.
"""

from __future__ import annotations

import importlib.util
import os
import sysconfig
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "agent_audit"


def _load_coverage_tool():
    """Import scripts/coverage_report.py without putting scripts/ on sys.path."""

    spec = importlib.util.spec_from_file_location(
        "agent_audit_coverage_report", ROOT / "scripts" / "coverage_report.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COVERAGE_TOOL = _load_coverage_tool()


class PathScopedIgnoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predicate = COVERAGE_TOOL._PathScopedIgnore(PACKAGE_DIR)
        self.stdlib_io = Path(sysconfig.get_paths()["stdlib"]) / "io.py"

    def _records(self, path: Path) -> bool:
        # trace calls this with the file and a module name; 0 means "record".
        return self.predicate.names(str(path), path.stem) == 0

    def test_records_every_module_in_the_package(self) -> None:
        sources = sorted(PACKAGE_DIR.glob("*.py"))

        self.assertGreater(len(sources), 5, "package layout looks wrong")
        for source in sources:
            with self.subTest(module=source.stem):
                self.assertTrue(self._records(source))

    def test_a_basename_shared_with_the_standard_library_is_still_recorded(self) -> None:
        """The exact defect that made CI disagree with the local run."""

        package_io = PACKAGE_DIR / "io.py"
        self.assertTrue(package_io.exists(), "this test assumes agent_audit/io.py")
        self.assertEqual(package_io.stem, self.stdlib_io.stem)

        # Ask about the standard library file first: a name-keyed cache would
        # poison the answer for ours.
        self.assertFalse(self._records(self.stdlib_io))
        self.assertTrue(self._records(package_io))

    def test_the_decision_does_not_depend_on_the_order_of_questions(self) -> None:
        package_io = PACKAGE_DIR / "io.py"

        forwards = COVERAGE_TOOL._PathScopedIgnore(PACKAGE_DIR)
        backwards = COVERAGE_TOOL._PathScopedIgnore(PACKAGE_DIR)

        forwards.names(str(self.stdlib_io), "io")
        first = forwards.names(str(package_io), "io")
        second = backwards.names(str(package_io), "io")

        self.assertEqual(first, second)

    def test_files_outside_the_package_are_not_recorded(self) -> None:
        for path in (self.stdlib_io, ROOT / "tests" / "test_coverage_tool.py", ROOT / "setup.py"):
            with self.subTest(path=path.name):
                self.assertFalse(self._records(path))

    def test_a_relative_path_is_resolved_before_the_decision(self) -> None:
        """trace passes whatever ``__file__`` holds, which may be relative."""

        previous = os.getcwd()
        os.chdir(ROOT)
        try:
            self.assertTrue(self._records(Path("agent_audit") / "io.py"))
        finally:
            os.chdir(previous)


class ExecutableLineTests(unittest.TestCase):
    def test_counts_lines_inside_nested_functions_and_classes(self) -> None:
        source = ROOT / "agent_audit" / "models.py"

        linenos = COVERAGE_TOOL._executable_linenos(source)

        text = source.read_text(encoding="utf-8").splitlines()
        # A dataclass field assignment lives in a class body, and a method body
        # lives in a nested code object; both must be counted.
        field_line = next(
            index for index, line in enumerate(text, 1) if "case_id: str" in line
        )
        self.assertIn(field_line, linenos)
        self.assertLess(len(linenos), len(text), "blank lines must not be counted")

    def test_ignores_comments_and_blank_lines(self) -> None:
        source = ROOT / "agent_audit" / "models.py"
        text = source.read_text(encoding="utf-8").splitlines()

        linenos = COVERAGE_TOOL._executable_linenos(source)

        for lineno in linenos:
            stripped = text[lineno - 1].strip()
            self.assertNotEqual(stripped, "")
            self.assertFalse(stripped.startswith("#"))


class ModuleBudgetTests(unittest.TestCase):
    def test_every_measured_module_has_a_budget(self) -> None:
        """A module without a budget can silently lose coverage."""

        measured = {
            source.stem
            for source in PACKAGE_DIR.glob("*.py")
            if source.stem not in COVERAGE_TOOL.EXCLUDED_MODULES
        }

        self.assertEqual(measured - set(COVERAGE_TOOL.MAX_UNCOVERED_LINES), set())
        self.assertEqual(set(COVERAGE_TOOL.MAX_UNCOVERED_LINES) - measured, set())

    def test_budgets_are_small_enough_to_notice_a_deleted_test_file(self) -> None:
        """A budget large enough to hide real regressions is not a gate."""

        for name, budget in COVERAGE_TOOL.MAX_UNCOVERED_LINES.items():
            with self.subTest(module=name):
                self.assertLessEqual(budget, 12)


if __name__ == "__main__":
    unittest.main()

"""Guards for the quickstart check.

The check reads its commands out of README.md. That is what keeps it honest —
and also what can make it silently blind: change the fence language or the
command prefix and it finds nothing to run, then reports success over an
empty list. These tests pin the extraction and the input-dependency rule.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load():
    name = "agent_audit_quickstart_check"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "quickstart_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load()


def _readme(body: str) -> Path:
    path = Path(tempfile.mkdtemp()) / "README.md"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


class ExtractionTests(unittest.TestCase):
    def test_a_backtick_continued_command_becomes_one_argv(self) -> None:
        commands = TOOL.readme_commands(
            _readme(
                "```powershell\npython -m agent_audit audit `\n"
                "  --input examples/demo_scores.csv `\n"
                "  --report outputs/demo.md\n```\n"
            )
        )

        self.assertEqual(
            commands,
            [[
                "python", "-m", "agent_audit", "audit",
                "--input", "examples/demo_scores.csv",
                "--report", "outputs/demo.md",
            ]],
        )

    def test_commands_needing_a_model_or_a_secret_are_left_out(self) -> None:
        commands = TOOL.readme_commands(
            _readme(
                "```powershell\n$env:OPENAI_API_KEY = \"k\"\n"
                "python -m agent_audit score --model YOUR_MODEL\n```\n"
            )
        )

        self.assertEqual(commands, [])

    def test_non_powershell_fences_are_ignored(self) -> None:
        commands = TOOL.readme_commands(
            _readme("```bash\npython -m agent_audit audit --input x\n```\n")
        )

        self.assertEqual(commands, [])

    def test_the_real_readme_still_yields_commands(self) -> None:
        """If this ever returns nothing, the check has gone blind."""

        self.assertTrue(TOOL.readme_commands(ROOT / "README.md"))


class InputDependencyTests(unittest.TestCase):
    def test_a_command_whose_input_is_absent_is_reported(self) -> None:
        work = Path(tempfile.mkdtemp())

        missing = TOOL.missing_inputs(
            ["python", "-m", "agent_audit", "audit", "--input", "outputs/live.csv"],
            work,
        )

        self.assertEqual(missing, ["outputs/live.csv"])

    def test_a_command_whose_input_exists_is_runnable(self) -> None:
        work = Path(tempfile.mkdtemp())
        (work / "cases.csv").write_text("x", encoding="utf-8")

        missing = TOOL.missing_inputs(
            ["python", "-m", "agent_audit", "audit", "--input", "cases.csv"], work
        )

        self.assertEqual(missing, [])

    def test_output_flags_are_not_treated_as_inputs(self) -> None:
        """`--output` naming a file that does not exist yet is the normal case."""

        work = Path(tempfile.mkdtemp())

        missing = TOOL.missing_inputs(
            ["python", "-m", "agent_audit", "generate", "--output", "outputs/new.csv"],
            work,
        )

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()

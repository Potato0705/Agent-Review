"""Guards for the mutation gate itself.

The gate edits production modules in place. Its one unacceptable outcome is
leaving a mutation in the working tree, where it can be committed and shipped.
A real Windows run hit ``OSError: [Errno 22]`` while restoring a module; the
write raised inside the ``finally`` block, so the restoration check never ran
and the process ended on a traceback with ``agent_audit/scoring.py`` still
mutated. These tests pin the contract that failure cannot do that again.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_mutation_tool():
    """Import scripts/mutation_check.py without putting scripts/ on sys.path.

    The module is registered under its own name first: ``@dataclass`` resolves
    annotations through ``sys.modules[cls.__module__]``, which does not exist
    yet while the module is still executing.
    """

    name = "agent_audit_mutation_check"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "mutation_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MUTATION_TOOL = _load_mutation_tool()


class WriteSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())

    def test_a_normal_write_succeeds(self) -> None:
        target = self.work / "module.py"

        self.assertTrue(MUTATION_TOOL.write_source(target, b"x = 1\n"))
        self.assertEqual(target.read_bytes(), b"x = 1\n")

    def test_a_failing_write_returns_false_instead_of_raising(self) -> None:
        """A raise here skips the restoration check and strands a mutation."""

        unwritable = self.work / "a-directory"
        unwritable.mkdir()

        with unittest.mock.patch.object(
            MUTATION_TOOL, "RESTORE_PAUSE_SECONDS", 0.0
        ):
            self.assertFalse(MUTATION_TOOL.write_source(unwritable, b"x = 1\n"))

    def test_a_transient_failure_is_retried(self) -> None:
        target = self.work / "module.py"
        attempts = {"count": 0}
        real_write = Path.write_bytes

        def flaky(self: Path, data: bytes) -> int:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise OSError(22, "Invalid argument")
            return real_write(self, data)

        with unittest.mock.patch.object(Path, "write_bytes", flaky):
            with unittest.mock.patch.object(
                MUTATION_TOOL, "RESTORE_PAUSE_SECONDS", 0.0
            ):
                self.assertTrue(MUTATION_TOOL.write_source(target, b"x = 1\n"))

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(target.read_bytes(), b"x = 1\n")

    def test_it_gives_up_rather_than_retrying_forever(self) -> None:
        target = self.work / "module.py"
        attempts = {"count": 0}

        def always_fails(self: Path, data: bytes) -> int:
            attempts["count"] += 1
            raise OSError(22, "Invalid argument")

        with unittest.mock.patch.object(Path, "write_bytes", always_fails):
            with unittest.mock.patch.object(
                MUTATION_TOOL, "RESTORE_PAUSE_SECONDS", 0.0
            ):
                self.assertFalse(MUTATION_TOOL.write_source(target, b"x = 1\n"))

        self.assertEqual(attempts["count"], MUTATION_TOOL.RESTORE_ATTEMPTS)


class CatalogueTests(unittest.TestCase):
    """Checks that hold while a module is mutated.

    Nothing here may read ``agent_audit`` source, because the harness runs
    this suite once per mutant with that source edited. An anchor-freshness
    test, for instance, would fail under every mutation and mark every mutant
    caught, turning the gate green no matter how weak the real tests are. The
    harness verifies its own anchors before it mutates anything.
    """

    def test_every_mutation_actually_changes_the_source(self) -> None:
        for mutant in MUTATION_TOOL.MUTANTS:
            with self.subTest(mutant=f"{mutant.module}: {mutant.label}"):
                self.assertNotEqual(mutant.original, mutant.mutated)

    def test_labels_are_unique_so_a_survivor_can_be_identified(self) -> None:
        labels = [f"{m.module}: {m.label}" for m in MUTATION_TOOL.MUTANTS]

        self.assertEqual(len(labels), len(set(labels)))


if __name__ == "__main__":
    unittest.main()

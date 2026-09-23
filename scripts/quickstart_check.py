"""Run the README's own commands against a fresh clone in a clean virtualenv.

The test suite imports the package directly from the working tree. It can
therefore pass while the thing a new user actually does — clone, install,
paste the first command from the README — is broken: a missing entry point, a
packaging mistake, a command that was renamed in the code and not in the
docs. None of that is visible from inside the tree.

The commands are *extracted from the README*, never copied here. A check that
keeps its own copy of the instructions drifts away from them silently, and
then it is verifying a document nobody is reading.

Blocks that need a model or a secret are skipped: they name a placeholder
(``YOUR_...``) or set an environment variable, and neither belongs in an
offline check.

Run it directly::

    python scripts/quickstart_check.py
    python scripts/quickstart_check.py --keep   # leave the sandbox for a look
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"```powershell\n(.*?)```", re.DOTALL)
NEEDS_A_MODEL = ("YOUR_", "$env:")
# Flags whose value is a file the command reads. A README command whose input
# only exists after a model has been called cannot run offline; that is the
# document being honest about ordering, not the document being broken.
INPUT_FLAGS = (
    "--input", "--reference", "--candidate", "--rubric-file",
    "--paraphrase-review", "--paraphrase-manifest", "--generation-manifest",
)


def missing_inputs(command: list[str], cwd: Path) -> list[str]:
    """Return the input paths this command needs that are not there yet."""

    missing = []
    for flag, value in zip(command, command[1:]):
        if flag in INPUT_FLAGS and not (cwd / value).exists():
            missing.append(value)
    return missing


def readme_commands(readme: Path) -> list[list[str]]:
    """Return the offline `python -m agent_audit` commands the README shows."""

    commands: list[list[str]] = []
    for block in FENCE.findall(readme.read_text(encoding="utf-8")):
        # PowerShell continues a line with a trailing backtick.
        joined = block.replace("`\n", " ").strip()
        for line in joined.splitlines():
            line = line.strip()
            if not line.startswith("python -m agent_audit"):
                continue
            if any(marker in line for marker in NEEDS_A_MODEL):
                continue
            commands.append(line.split())
    return commands


def _run(argv: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=cwd, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def _report(step: str, completed: subprocess.CompletedProcess) -> bool:
    if completed.returncode == 0:
        print(f"  ok    {step}")
        return True
    sys.stdout.flush()
    print(f"  FAIL  {step}", file=sys.stderr)
    for stream in (completed.stdout, completed.stderr):
        tail = (stream or "").strip().splitlines()[-12:]
        for line in tail:
            print(f"        {line}", file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true", help="Leave the sandbox directory in place."
    )
    args = parser.parse_args(argv)

    sandbox = Path(tempfile.mkdtemp(prefix="agent-review-quickstart-"))
    clone = sandbox / "Agent-Review"
    venv = sandbox / "venv"
    failures: list[str] = []

    try:
        print(f"Sandbox: {sandbox}")
        dirty = _run(["git", "status", "--porcelain"], PROJECT_ROOT, dict(os.environ))
        if dirty.stdout.strip():
            print(
                "  note  the working tree has uncommitted changes; this check "
                "clones HEAD and will not see them.",
            )
        # Clone rather than copy: a new user gets what is committed, not what
        # happens to be sitting in the working tree.
        if not _report(
            "git clone (committed tree only)",
            _run(
                ["git", "clone", "--quiet", str(PROJECT_ROOT), str(clone)],
                cwd=sandbox,
                env=dict(os.environ),
            ),
        ):
            return 1

        if not _report(
            "create virtualenv",
            _run([sys.executable, "-m", "venv", str(venv)], cwd=sandbox, env=dict(os.environ)),
        ):
            return 1

        bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        env = {**os.environ, "VIRTUAL_ENV": str(venv), "PYTHONUTF8": "1"}
        env.pop("PYTHONHOME", None)

        if not _report(
            "pip install -e .",
            _run([str(python), "-m", "pip", "install", "--quiet", "-e", "."], clone, env),
        ):
            return 1

        commands = readme_commands(clone / "README.md")
        if not commands:
            print(
                "No runnable commands found in README.md. Either the quickstart "
                "moved or the fence language changed; this check is now blind.",
                file=sys.stderr,
            )
            return 1
        print(f"Found {len(commands)} offline command(s) in README.md")

        skipped = 0
        for command in commands:
            label = " ".join(command[:5]) + (" ..." if len(command) > 5 else "")
            absent = missing_inputs(command, clone)
            if absent:
                print(f"  skip  {label}  (needs {', '.join(absent)})")
                skipped += 1
                continue
            if not _report(label, _run([str(python), *command[1:]], clone, env)):
                failures.append(label)
        if skipped:
            print(f"  {skipped} command(s) skipped: their inputs come from a model run.")

        console = bin_dir / ("agent-audit.exe" if os.name == "nt" else "agent-audit")
        if not console.exists():
            print(f"  FAIL  console entry point missing: {console}", file=sys.stderr)
            failures.append("console entry point")
        elif not _report(
            "agent-audit --help", _run([str(console), "--help"], clone, env)
        ):
            failures.append("agent-audit --help")
    finally:
        if args.keep:
            print(f"Sandbox kept at {sandbox}")
        else:
            shutil.rmtree(sandbox, ignore_errors=True)

    if failures:
        print(
            f"\nQuickstart check failed on {len(failures)} step(s): "
            + ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    print("\nQuickstart check passed: a fresh clone installs and the README runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The command-line entry point: a registry, not a switchboard.

Each subcommand owns its own module and contributes two things — the parser
section that defines its options and the function that runs it. Adding a
seventh command means adding a file and one line here, rather than growing
this one past the point where anybody can hold it in their head.

Each command module also declares `USAGE_ERRORS`: the exceptions that mean
"the operator asked for something impossible" rather than "the tool broke".
Those become an argparse usage error with the message intact; anything else
propagates as a traceback, because a bug should look like a bug.
"""

from __future__ import annotations

import argparse
from types import ModuleType

from . import (
    cli_audit,
    cli_compare,
    cli_generate,
    cli_paraphrase,
    cli_score,
    cli_trajectory,
)


COMMANDS: dict[str, ModuleType] = {
    "audit": cli_audit,
    "score": cli_score,
    "compare": cli_compare,
    "generate": cli_generate,
    "paraphrase": cli_paraphrase,
    "trajectory": cli_trajectory,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-audit",
        description="Audit LLM grader scores for gaming, degradation and paraphrase sensitivity.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS.values():
        command.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = COMMANDS[args.command]
    try:
        return command.run(args)
    except command.USAGE_ERRORS as exc:
        # `parser.error` exits, so nothing follows it.
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

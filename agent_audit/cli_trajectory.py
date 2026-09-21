"""The `trajectory` subcommand: variants of annotated agent runs.

What is audited is the grader that scores agent runs, not the agent. The
output is ordinary `score` input, so everything downstream is unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .cli_common import (
    describe_merge,
    merge_summary_of,
    refuse_to_overwrite_cases,
    row_fingerprint_payload,
    strategy_list,
)
from .io import load_scoring_cases, stable_hash, write_scoring_cases
from .scoring import write_json
from .segmentation import LANGUAGES
from .trajectory import load_trajectory_cases
from .trajectory_variants import (
    DEGRADATION_STRATEGIES as TRAJECTORY_DEGRADATION,
    GAMING_STRATEGIES as TRAJECTORY_GAMING,
    TrajectoryPostconditionError,
    generate_trajectory_variants,
)
from .variants import GeneratedRow, merge_into_existing


USAGE_ERRORS: tuple[type[Exception], ...] = (ValueError, TrajectoryPostconditionError)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "trajectory",
        help="Generate variants of annotated agent trajectories.",
    )
    command.add_argument(
        "--input", required=True, help="Annotated trajectory JSONL."
    )
    command.add_argument(
        "--output", required=True, help="Scoring case CSV to write."
    )
    command.add_argument(
        "--gaming",
        default=",".join(TRAJECTORY_GAMING),
        help=f"Comma-separated gaming strategies from {list(TRAJECTORY_GAMING)}.",
    )
    command.add_argument(
        "--degradation",
        default=",".join(TRAJECTORY_DEGRADATION),
        help=(
            "Comma-separated degradation strategies from "
            f"{list(TRAJECTORY_DEGRADATION)}."
        ),
    )
    command.add_argument(
        "--paraphrase",
        help=(
            "Optional paraphrase strategies. Off by default so a run without "
            "annotated independent steps is not refused for a variant family "
            "nobody asked for."
        ),
    )
    command.add_argument(
        "--seed", type=int, default=0, help="Seed for choosing repeated calls."
    )
    command.add_argument(
        "--language",
        choices=sorted(LANGUAGES),
        default="chinese",
        help="Transcript labels and padding language. Never auto-detected.",
    )
    command.add_argument(
        "--show-steps",
        action="store_true",
        help=(
            "Print the numbered steps and annotations for each trajectory and "
            "stop, so annotations can be checked against what the tool sees."
        ),
    )
    command.add_argument(
        "--append",
        action="store_true",
        help=(
            "Merge into an existing case file: keep every row already there, "
            "including hand edits, and add only the missing ones."
        ),
    )
    command.add_argument(
        "--manifest", help="Run manifest path; defaults next to the output CSV."
    )


def run(args: argparse.Namespace) -> int:
    language = LANGUAGES[getattr(args, "language", "chinese")]
    cases = load_trajectory_cases(args.input)

    if getattr(args, "show_steps", False):
        for case in cases:
            print(f"[{case.case_id}] {language.name}")
            for step in case.steps:
                print(f"  {step.index}. {step.tool} {step.args} -> {step.result}")
            print(
                "  load-bearing: "
                + ", ".join(str(index) for index in case.load_bearing_steps)
            )
            groups = case.independent_steps
            print(
                "  independent: "
                + (
                    "; ".join(
                        ", ".join(str(index) for index in group) for group in groups
                    )
                    if groups
                    else "none"
                )
            )
        return 0

    output_path = Path(args.output)
    refuse_to_overwrite_cases(output_path, getattr(args, "append", False))

    run = generate_trajectory_variants(
        cases,
        gaming=strategy_list(args.gaming),
        degradation=strategy_list(args.degradation),
        paraphrase=strategy_list(getattr(args, "paraphrase", None)),
        language=language,
        seed=args.seed,
    )
    generated = tuple(
        GeneratedRow(
            case_id=row.case_id,
            variant_id=row.variant_id,
            variant_type=row.variant_type,
            text=row.text,
            notes=row.notes,
        )
        for row in run.rows
    )

    rows = generated
    merge_summary: dict[str, Any] = merge_summary_of(rows)
    set_origin = "machine-generated"
    if getattr(args, "append", False):
        if not output_path.exists():
            raise ValueError(
                f"--append needs an existing case file, but {output_path} does not "
                "exist. Run without --append to create it."
            )
        outcome = merge_into_existing(generated, load_scoring_cases(output_path))
        rows = outcome.rows
        set_origin = outcome.set_origin
        merge_summary = describe_merge(outcome)

    output_path = write_scoring_cases(output_path, rows)
    manifest_path = (
        Path(args.manifest)
        if getattr(args, "manifest", None)
        else output_path.with_suffix(".manifest.json")
    )
    manifest = {
        **run.manifest,
        "output_sha256": stable_hash(row_fingerprint_payload(rows)),
        "row_count": len(rows),
        "set_origin": set_origin,
        "merge": merge_summary,
        "input_path": str(Path(args.input).resolve()),
        "input_sha256": stable_hash(
            [
                {
                    "case_id": case.case_id,
                    "task": case.task,
                    "steps": [list(step.identity) for step in case.steps],
                    "final_answer": case.final_answer,
                    "load_bearing_steps": list(case.load_bearing_steps),
                    "independent_steps": [list(g) for g in case.independent_steps],
                }
                for case in cases
            ]
        ),
    }
    write_json(manifest_path, manifest)

    print(f"Cases written to: {output_path.resolve()}")
    print(f"Manifest written to: {manifest_path.resolve()}")
    print(f"Generated {len(rows)} rows from {len(cases)} trajectories.")
    if set_origin == "mixed":
        print(
            "This set contains hand-written or hand-edited rows, so it is mixed, "
            "not machine-generated. Audit it with --variant-origin mixed.",
            file=sys.stderr,
        )
    else:
        print(
            "Variants are machine-generated: review each one before delivery, "
            "then audit with --variant-origin machine-generated and this manifest."
        )
    return 0



"""The `generate` subcommand: text variants, and ratified rewrites merged in.

A model may draft a paraphrase but only a person may admit one to the case
set, so the merge here is deliberately fail-closed: an unrecognised status or
a baseline that changed since drafting stops the run and names the case.
"""

from __future__ import annotations

import argparse
import json
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
from .io import (
    load_baseline_cases,
    load_paraphrase_review,
    load_scoring_cases,
    stable_hash,
    write_scoring_cases,
)
from .scoring import write_json
from .segmentation import LANGUAGES
from .variants import (
    DEGRADATION_STRATEGIES,
    GAMING_STRATEGIES,
    GeneratedRow,
    generate_variants,
    merge_into_existing,
)


USAGE_ERRORS: tuple[type[Exception], ...] = (ValueError,)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "generate",
        help="Build gaming and degradation variants from annotated baselines.",
    )
    command.add_argument("--input", required=True, help="Annotated baseline CSV.")
    command.add_argument("--output", required=True, help="Output case CSV.")
    command.add_argument(
        "--manifest", help="Generation manifest path; defaults next to the output CSV."
    )
    command.add_argument(
        "--gaming",
        default=",".join(GAMING_STRATEGIES),
        help=f"Comma-separated gaming strategies from {list(GAMING_STRATEGIES)}.",
    )
    command.add_argument(
        "--degradation",
        default=",".join(DEGRADATION_STRATEGIES),
        help=(
            "Comma-separated degradation strategies from "
            f"{list(DEGRADATION_STRATEGIES)}."
        ),
    )
    command.add_argument(
        "--paraphrase",
        help=(
            "Optional paraphrase strategies. Off by default: the conservative "
            "rewrite almost always scores the same and would dilute the "
            "violation rate."
        ),
    )
    command.add_argument(
        "--seed", type=int, default=0, help="Seed for corpus selection."
    )
    command.add_argument(
        "--language",
        choices=sorted(LANGUAGES),
        default="chinese",
        help="Sentence splitting and corpus language. Never auto-detected.",
    )
    command.add_argument(
        "--paraphrase-review",
        help=(
            "Reviewed paraphrase file; approved rows are added as paraphrase "
            "variants. Requires --append."
        ),
    )
    command.add_argument(
        "--paraphrase-manifest",
        help=(
            "Manifest of the drafting run; defaults next to the review file. "
            "It names the model that wrote the rewrites."
        ),
    )
    command.add_argument(
        "--show-sentences",
        action="store_true",
        help=(
            "Print the numbered sentence split for each baseline and stop, so "
            "annotations can be checked against what the splitter sees."
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



def run(args: argparse.Namespace) -> int:
    language = LANGUAGES[getattr(args, "language", "chinese")]
    if getattr(args, "show_sentences", False):
        for case in load_baseline_cases(args.input, language=language):
            print(f"[{case.case_id}] {language.name}")
            for index, sentence in enumerate(language.split(case.text), start=1):
                print(f"  {index}. {sentence.strip()}")
            print(f"  annotated: {', '.join(str(i) for i in case.evidence_sentences)}")
        return 0

    output_path = Path(args.output)
    refuse_to_overwrite_cases(output_path, getattr(args, "append", False))
    cases = load_baseline_cases(args.input, language=language)
    run = generate_variants(
        cases,
        seed=args.seed,
        gaming=strategy_list(args.gaming),
        degradation=strategy_list(args.degradation),
        paraphrase=strategy_list(getattr(args, "paraphrase", None)),
        language=language,
    )

    review_argument = getattr(args, "paraphrase_review", None)
    review_counts: dict[str, Any] | None = None
    if review_argument and not getattr(args, "append", False):
        raise ValueError(
            "--paraphrase-review requires --append: ratified rewrites are merged "
            "into an existing case file, not generated from scratch."
        )

    rows = run.rows
    merge_summary: dict[str, Any] = merge_summary_of(rows)
    set_origin = "machine-generated"
    if getattr(args, "append", False):
        if not output_path.exists():
            raise ValueError(
                f"--append needs an existing case file, but {output_path} does not "
                "exist. Run without --append to create it."
            )
        outcome = merge_into_existing(run.rows, load_scoring_cases(output_path))
        rows = outcome.rows
        set_origin = outcome.set_origin
        merge_summary = describe_merge(outcome)

    paraphrase_model: str | None = None
    if review_argument:
        # The review file is read first so that its own absence is what gets
        # reported; complaining about a missing manifest sends the operator
        # looking for the wrong file.
        ratified, review_counts = _approved_paraphrase_rows(
            Path(review_argument), cases
        )
        paraphrase_model = _paraphrase_model(
            Path(review_argument), getattr(args, "paraphrase_manifest", None)
        )
        # Ratified rows are added after the merge, never as part of the
        # generator's target set: their equivalence rests on human judgement,
        # so counting them as generated would be the laundering this design
        # exists to prevent.
        present = {(row.case_id, row.variant_id) for row in rows}
        fresh = tuple(
            row for row in ratified if (row.case_id, row.variant_id) not in present
        )
        rows = rows + fresh
        merge_summary["ratified_paraphrase"] = [
            [row.case_id, row.variant_id] for row in ratified
        ]
        if ratified:
            set_origin = "mixed"

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
        "paraphrase_review": review_counts,
        "paraphrase_model": paraphrase_model,
        "input_path": str(Path(args.input).resolve()),
        "input_sha256": stable_hash(
            [
                {
                    "case_id": case.case_id,
                    "text": case.text,
                    "evidence_sentences": list(case.evidence_sentences),
                }
                for case in cases
            ]
        ),
    }
    write_json(manifest_path, manifest)

    print(f"Cases written to: {output_path.resolve()}")
    print(f"Manifest written to: {manifest_path.resolve()}")
    print(f"Generated {len(rows)} rows from {len(cases)} baselines.")
    if review_counts is not None:
        summary = ", ".join(f"{key}={value}" for key, value in sorted(review_counts.items()))
        print(f"Paraphrase review rows: {summary}.")
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



def _paraphrase_model(review_path: Path, manifest_argument: str | None) -> str:
    """Return the model that drafted these rewrites.

    The audit compares this name against the scoring model, because a set
    whose rewrites were written and graded by the same model tests nothing:
    the system under test also supplied the definition of "same meaning".
    Guessing the name would defeat that check, so a missing manifest is an
    error rather than an unknown.
    """

    manifest_path = (
        Path(manifest_argument)
        if manifest_argument
        else review_path.with_suffix(".manifest.json")
    )
    if not manifest_path.exists():
        raise ValueError(
            f"Paraphrase manifest does not exist: {manifest_path}. It names the "
            "model that drafted these rewrites, which the audit needs to rule "
            "out a rewriter and a grader that are the same model. Name it with "
            "--paraphrase-manifest if it lives elsewhere."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Paraphrase manifest is malformed: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("Paraphrase manifest must contain a JSON object.")
    model = manifest.get("paraphrase_model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(
            "Paraphrase manifest field 'paraphrase_model' must be a non-empty "
            "string."
        )
    return model.strip()


def _approved_paraphrase_rows(
    review_path: Path, cases: list[Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Return approved rows, refusing any that no longer match their baseline."""

    rows = load_paraphrase_review(review_path)
    counts: dict[str, int] = {status: 0 for status in ("pending", "blocked", "approved", "rejected")}
    for row in rows:
        counts[row.status] += 1

    expected = {case.case_id: stable_hash(case.text) for case in cases}
    approved: list[Any] = []
    for row in rows:
        if row.status != "approved":
            continue
        if row.case_id not in expected:
            raise ValueError(
                f"Review row names case {row.case_id!r}, which is not in the "
                "baseline file."
            )
        if row.baseline_sha256 != expected[row.case_id]:
            raise ValueError(
                f"Case {row.case_id!r} was edited after this paraphrase was "
                "drafted, so the draft rewrites text that no longer exists. "
                "Redraft it instead of merging a stale rewrite."
            )
        notes = ["generated=human_ratified_paraphrase"]
        if row.reviewer_note:
            notes.append(f"reviewer_note={row.reviewer_note}")
        approved.append(
            GeneratedRow(
                case_id=row.case_id,
                variant_id="paraphrase_ratified",
                variant_type="paraphrase",
                text=row.draft_text,
                notes=" | ".join(notes),
            )
        )
    return approved, counts



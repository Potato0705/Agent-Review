from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlsplit

from .audit import AuditConfig, audit_records
from .comparison import compare_audits, load_audit_result, render_comparison_report
from .html_report import render_audit_html, render_comparison_html
from .io import (
    load_baseline_cases,
    load_paraphrase_review,
    load_score_records,
    load_scoring_cases,
    stable_hash,
    write_paraphrase_review,
    write_score_records,
    write_scoring_cases,
    write_text,
)
from .provider import OpenAICompatibleConfig, OpenAICompatibleScorer, ProviderError
from .report import render_markdown_report
from .scoring import run_scoring, write_json, write_jsonl
from .paraphrase import draft_paraphrases
from .segmentation import LANGUAGES
from .variants import (
    DEGRADATION_STRATEGIES,
    GAMING_STRATEGIES,
    generate_variants,
    merge_into_existing,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-audit",
        description="Audit LLM grader scores for gaming, degradation and paraphrase sensitivity.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="Audit a score CSV and create a report.")
    audit_parser.add_argument("--input", required=True, help="Input score CSV.")
    audit_parser.add_argument("--report", required=True, help="Output Markdown report.")
    audit_parser.add_argument("--json", dest="json_output", help="Optional JSON result path.")
    audit_parser.add_argument(
        "--html", dest="html_output", help="Optional self-contained HTML report path."
    )
    audit_parser.add_argument(
        "--manifest",
        help="Optional scoring manifest; auto-detected next to the input CSV when present.",
    )
    audit_parser.add_argument("--invariance-tolerance", type=float, default=0.5)
    audit_parser.add_argument("--min-degradation-drop", type=float, default=1.0)
    audit_parser.add_argument("--gaming-tolerance", type=float, default=0.0)
    audit_parser.add_argument("--score-min", type=float)
    audit_parser.add_argument("--score-max", type=float)
    audit_parser.add_argument(
        "--data-provenance",
        choices=["synthetic", "public-demo", "authorized-private", "unspecified"],
        default="unspecified",
        help="Declared origin of the evaluated data.",
    )
    audit_parser.add_argument(
        "--generation-manifest",
        help=(
            "Generation manifest proving the scored cases are the generated "
            "ones. Required by --variant-origin machine-generated."
        ),
    )
    audit_parser.add_argument(
        "--variant-origin",
        choices=["human-authored", "machine-generated", "mixed", "unspecified"],
        default="unspecified",
        help="Whether the variants were written by hand or generated.",
    )

    score_parser = subparsers.add_parser(
        "score", help="Score case texts with an OpenAI-compatible chat API."
    )
    score_parser.add_argument("--input", required=True, help="Input case CSV.")
    score_parser.add_argument("--output", required=True, help="Output score CSV.")
    score_parser.add_argument("--rubric-file", required=True, help="UTF-8 rubric file.")
    score_parser.add_argument("--model", required=True)
    score_parser.add_argument(
        "--base-url", default="https://api.openai.com/v1", help="API base URL."
    )
    score_parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    score_parser.add_argument("--system-name", help="Name stored in the score CSV.")
    score_parser.add_argument("--score-min", type=float, default=0.0)
    score_parser.add_argument("--score-max", type=float, default=10.0)
    score_parser.add_argument("--temperature", type=float, default=0.0)
    score_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Independent scoring calls per input (1-100). This multiplies API usage.",
    )
    score_parser.add_argument("--timeout", type=float, default=60.0)
    score_parser.add_argument("--max-retries", type=int, default=2)
    score_parser.add_argument(
        "--manifest", help="Run manifest path; defaults next to the output CSV."
    )
    score_parser.add_argument(
        "--raw-output",
        help="Optional JSONL trace path. May contain model-generated sensitive text.",
    )
    score_parser.add_argument(
        "--checkpoint",
        help=(
            "Optional per-sample JSONL checkpoint for crash recovery. "
            "Contains full model replies and must be stored securely."
        ),
    )
    score_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching existing --checkpoint without repeating saved calls.",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two compatible audit JSON results."
    )
    compare_parser.add_argument("--reference", required=True, help="Reference audit JSON.")
    compare_parser.add_argument("--candidate", required=True, help="Candidate audit JSON.")
    compare_parser.add_argument("--report", required=True, help="Output Markdown report.")
    compare_parser.add_argument("--json", dest="json_output", help="Optional JSON result path.")
    compare_parser.add_argument(
        "--html", dest="html_output", help="Optional self-contained HTML report path."
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Build gaming and degradation variants from annotated baselines.",
    )
    generate_parser.add_argument("--input", required=True, help="Annotated baseline CSV.")
    generate_parser.add_argument("--output", required=True, help="Output case CSV.")
    generate_parser.add_argument(
        "--manifest", help="Generation manifest path; defaults next to the output CSV."
    )
    generate_parser.add_argument(
        "--gaming",
        default=",".join(GAMING_STRATEGIES),
        help=f"Comma-separated gaming strategies from {list(GAMING_STRATEGIES)}.",
    )
    generate_parser.add_argument(
        "--degradation",
        default=",".join(DEGRADATION_STRATEGIES),
        help=(
            "Comma-separated degradation strategies from "
            f"{list(DEGRADATION_STRATEGIES)}."
        ),
    )
    generate_parser.add_argument(
        "--paraphrase",
        help=(
            "Optional paraphrase strategies. Off by default: the conservative "
            "rewrite almost always scores the same and would dilute the "
            "violation rate."
        ),
    )
    generate_parser.add_argument(
        "--seed", type=int, default=0, help="Seed for corpus selection."
    )
    generate_parser.add_argument(
        "--language",
        choices=sorted(LANGUAGES),
        default="chinese",
        help="Sentence splitting and corpus language. Never auto-detected.",
    )
    generate_parser.add_argument(
        "--paraphrase-review",
        help=(
            "Reviewed paraphrase file; approved rows are added as paraphrase "
            "variants. Requires --append."
        ),
    )
    generate_parser.add_argument(
        "--paraphrase-manifest",
        help=(
            "Manifest of the drafting run; defaults next to the review file. "
            "It names the model that wrote the rewrites."
        ),
    )
    generate_parser.add_argument(
        "--show-sentences",
        action="store_true",
        help=(
            "Print the numbered sentence split for each baseline and stop, so "
            "annotations can be checked against what the splitter sees."
        ),
    )
    generate_parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Merge into an existing case file: keep every row already there, "
            "including hand edits, and add only the missing ones."
        ),
    )

    paraphrase_parser = subparsers.add_parser(
        "paraphrase",
        help="Draft paraphrase candidates for human review. Calls a model.",
    )
    paraphrase_parser.add_argument("--input", required=True, help="Annotated baseline CSV.")
    paraphrase_parser.add_argument("--output", required=True, help="Review CSV path.")
    paraphrase_parser.add_argument("--model", required=True)
    paraphrase_parser.add_argument(
        "--base-url", default="https://api.openai.com/v1", help="API base URL."
    )
    paraphrase_parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    paraphrase_parser.add_argument(
        "--language", choices=sorted(LANGUAGES), default="chinese"
    )
    paraphrase_parser.add_argument("--temperature", type=float, default=0.0)
    paraphrase_parser.add_argument("--timeout", type=float, default=60.0)
    paraphrase_parser.add_argument("--max-retries", type=int, default=2)
    paraphrase_parser.add_argument(
        "--manifest", help="Run manifest path; defaults next to the review CSV."
    )
    return parser


def run_audit(args: argparse.Namespace) -> int:
    config = AuditConfig(
        invariance_tolerance=args.invariance_tolerance,
        min_degradation_drop=args.min_degradation_drop,
        gaming_tolerance=args.gaming_tolerance,
        score_min=args.score_min,
        score_max=args.score_max,
        data_provenance=args.data_provenance,
        variant_origin=args.variant_origin,
    )
    records = load_score_records(args.input)
    result = audit_records(records, config)
    manifest_argument = getattr(args, "manifest", None)
    manifest_path = (
        Path(manifest_argument)
        if manifest_argument
        else Path(args.input).with_suffix(".manifest.json")
    )
    comparison_context = None
    if manifest_argument or manifest_path.exists():
        comparison_context = _load_comparison_context(
            manifest_path, expected_record_count=len(records), config=config
        )

    generation_argument = getattr(args, "generation_manifest", None)
    if config.variant_origin == "machine-generated" and not generation_argument:
        raise ValueError(
            "--variant-origin machine-generated requires --generation-manifest so "
            "the claim can be verified; use mixed if the cases were edited."
        )
    if generation_argument:
        if comparison_context is None:
            raise ValueError(
                "--generation-manifest needs the scoring manifest as well; without "
                "it there is no scored-input fingerprint to verify against."
            )
        generation_sha256, paraphrase_model = _load_generation_context(
            Path(generation_argument),
            scored_input_sha256=str(comparison_context["input_sha256"]),
            declared_origin=config.variant_origin,
        )
        comparison_context["generation_sha256"] = generation_sha256
        comparison_context["paraphrase_model"] = paraphrase_model
        scoring_model = str(comparison_context["model"])
        if (
            paraphrase_model
            and paraphrase_model.casefold() == scoring_model.casefold()
        ):
            raise ValueError(
                f"The paraphrases were drafted by {paraphrase_model!r} and "
                "scored by the same model, so the system under test also "
                "supplied the definition of 'same meaning'. Redraft them with "
                "a different model."
            )
    elif comparison_context is not None:
        comparison_context["generation_sha256"] = None
        comparison_context["paraphrase_model"] = None
    models = None
    if comparison_context and comparison_context.get("paraphrase_model"):
        models = {
            "paraphrase": str(comparison_context["paraphrase_model"]),
            "scoring": str(comparison_context["model"]),
        }
    report_path = write_text(args.report, render_markdown_report(result, models=models))
    print(f"Report written to: {report_path.resolve()}")
    risk_status = "provisional" if result.risk_is_provisional else "screening"
    print(
        f"Risk level: {result.risk_level} ({risk_status}); "
        f"violation rate: {result.violation_rate:.1%}"
    )

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        payload["comparison_context"] = comparison_context
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        print(f"JSON written to: {json_path.resolve()}")
    html_output = getattr(args, "html_output", None)
    if html_output:
        html_path = write_text(html_output, render_audit_html(result, models=models))
        print(f"HTML written to: {html_path.resolve()}")
    return 0


def _load_comparison_context(
    manifest_path: Path, *, expected_record_count: int, config: AuditConfig
) -> dict[str, object]:
    if not manifest_path.exists():
        raise ValueError(f"Scoring manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scoring manifest is malformed: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Scoring manifest must contain a JSON object.")

    def validated_hash(key: str) -> str:
        value = manifest.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError(f"Scoring manifest field {key!r} must be a SHA-256 hash.")
        return value.lower()

    record_count = manifest.get("record_count")
    repeats = manifest.get("repeats", 1)
    temperature = manifest.get("temperature")
    model = manifest.get("model")
    if isinstance(record_count, bool) or not isinstance(record_count, int):
        raise ValueError("Scoring manifest record_count must be an integer.")
    if record_count != expected_record_count:
        raise ValueError("Scoring manifest record_count does not match the score CSV.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Scoring manifest model must be a non-empty string.")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Scoring manifest repeats must be a positive integer.")
    if isinstance(temperature, bool):
        raise ValueError("Scoring manifest temperature must be numeric.")
    try:
        parsed_temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("Scoring manifest temperature must be numeric.") from exc
    if not math.isfinite(parsed_temperature):
        raise ValueError("Scoring manifest temperature must be finite.")

    manifest_range = manifest.get("score_range")
    if config.score_min is not None and config.score_max is not None:
        if (
            not isinstance(manifest_range, list)
            or len(manifest_range) != 2
            or any(isinstance(value, bool) for value in manifest_range)
        ):
            raise ValueError("Scoring manifest score_range must contain two numbers.")
        try:
            parsed_range = [float(value) for value in manifest_range]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Scoring manifest score_range must contain two numbers."
            ) from exc
        if not all(math.isfinite(value) for value in parsed_range) or not all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(
                parsed_range, [config.score_min, config.score_max], strict=True
            )
        ):
            raise ValueError("Scoring manifest score_range does not match audit config.")

    return {
        "input_sha256": validated_hash("input_sha256"),
        "rubric_sha256": validated_hash("rubric_sha256"),
        "temperature": parsed_temperature,
        "repeats": repeats,
        "model": model.strip(),
    }


GENERATION_IDENTITY_FIELDS = (
    "generator",
    "language",
    "seed",
    "input_sha256",
    "gaming_strategies",
    "degradation_strategies",
    "paraphrase_strategies",
)


def _load_generation_context(
    manifest_path: Path, *, scored_input_sha256: str, declared_origin: str
) -> tuple[str, str | None]:
    """Prove the scored cases are exactly what the generator produced.

    The generator fingerprints its output and the scorer fingerprints its
    input. If those disagree the cases were edited after generation, so a
    `machine-generated` label would be false and `mixed` is the honest one.

    A mixed set may present a manifest too. The fingerprint still proves which
    run produced the file, and the manifest carries the one fact a mixed set
    needs most: which model drafted the ratified rewrites. Only the
    machine-generated claim itself is checked against `set_origin`.
    """

    if not manifest_path.exists():
        raise ValueError(f"Generation manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Generation manifest is malformed: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("Generation manifest must contain a JSON object.")

    def validated_hash(key: str) -> str:
        value = manifest.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError(
                f"Generation manifest field {key!r} must be a SHA-256 hash."
            )
        return value.lower()

    output_sha256 = validated_hash("output_sha256")
    baseline_sha256 = validated_hash("input_sha256")
    if output_sha256 != scored_input_sha256.lower():
        raise ValueError(
            "The scored cases are not the ones this generation manifest "
            "produced; they were edited after generation. Declare "
            "--variant-origin mixed instead of machine-generated."
        )

    # Manifests written before append mode existed described a full generation
    # run, so a missing field means the set was entirely machine-generated.
    set_origin = manifest.get("set_origin", "machine-generated")
    if declared_origin == "machine-generated" and set_origin != "machine-generated":
        raise ValueError(
            "The generation manifest says this set contains hand-written or "
            "hand-edited rows, so it is mixed, not machine-generated. Declare "
            "--variant-origin mixed."
        )
    if declared_origin == "human-authored":
        raise ValueError(
            "The fingerprint above proves a generator produced these cases, so "
            "--variant-origin human-authored is false. Declare "
            "machine-generated or mixed."
        )

    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Generation manifest seed must be an integer.")
    for key in ("generator", "language"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(
                f"Generation manifest field {key!r} must be a non-empty string."
            )
    for key in (
        "gaming_strategies",
        "degradation_strategies",
        "paraphrase_strategies",
    ):
        value = manifest.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(
                f"Generation manifest field {key!r} must be a list of strings."
            )

    paraphrase_model = manifest.get("paraphrase_model")
    if paraphrase_model is not None and (
        not isinstance(paraphrase_model, str) or not paraphrase_model.strip()
    ):
        raise ValueError(
            "Generation manifest field 'paraphrase_model' must be a non-empty "
            "string when present."
        )

    identity = {key: manifest[key] for key in GENERATION_IDENTITY_FIELDS}
    identity["input_sha256"] = baseline_sha256
    return stable_hash(identity), (
        paraphrase_model.strip() if isinstance(paraphrase_model, str) else None
    )


def run_score(args: argparse.Namespace) -> int:
    checkpoint = getattr(args, "checkpoint", None)
    resume = bool(getattr(args, "resume", False))
    if resume and not checkpoint:
        raise ValueError("--resume requires --checkpoint.")
    if args.repeats > 1 and not args.raw_output:
        print(
            "Warning: repeated scoring without --raw-output keeps only the first "
            "sample reason in the aggregate CSV.",
            file=sys.stderr,
        )
    if checkpoint:
        print(
            "Warning: checkpoint files contain full model replies and must be "
            "handled as sensitive data.",
            file=sys.stderr,
        )
    api_key = os.environ.get(args.api_key_env, "")
    hostname = urlsplit(args.base_url).hostname
    if not api_key and hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            f"Environment variable {args.api_key_env!r} is not set. "
            "API keys are accepted only through environment variables."
        )
    if not api_key:
        api_key = "local-provider"

    rubric_path = Path(args.rubric_file)
    if not rubric_path.exists():
        raise ValueError(f"Rubric file does not exist: {rubric_path}")
    rubric = rubric_path.read_text(encoding="utf-8").strip()
    cases = load_scoring_cases(args.input)

    provider_config = OpenAICompatibleConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        score_min=args.score_min,
        score_max=args.score_max,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    scorer = OpenAICompatibleScorer(provider_config)
    system_name = args.system_name or args.model
    scoring_run = run_scoring(
        cases,
        scorer,
        rubric,
        system_name=system_name,
        provider_name="openai-compatible",
        model=args.model,
        base_url=args.base_url,
        score_min=args.score_min,
        score_max=args.score_max,
        temperature=args.temperature,
        repeats=args.repeats,
        checkpoint_path=checkpoint,
        resume=resume,
    )

    output_path = write_score_records(args.output, list(scoring_run.records))
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(
        ".manifest.json"
    )
    write_json(manifest_path, scoring_run.manifest)
    print(f"Scores written to: {output_path.resolve()}")
    print(f"Manifest written to: {manifest_path.resolve()}")
    if checkpoint:
        print(
            "Checkpoint samples: "
            f"{scoring_run.manifest['resumed_sample_count']} resumed, "
            f"{scoring_run.manifest['new_sample_count']} new."
        )
    if args.raw_output:
        raw_path = write_jsonl(args.raw_output, scoring_run.traces)
        print(f"Raw traces written to: {raw_path.resolve()}")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    result = compare_audits(
        load_audit_result(args.reference), load_audit_result(args.candidate)
    )
    report_path = write_text(args.report, render_comparison_report(result))
    print(f"Comparison report written to: {report_path.resolve()}")
    print(
        f"Threshold regressions: {result.regression_count}; "
        f"improvements: {result.improvement_count}"
    )
    if args.json_output:
        json_path = write_json(args.json_output, result.to_dict())
        print(f"Comparison JSON written to: {json_path.resolve()}")
    html_output = getattr(args, "html_output", None)
    if html_output:
        html_path = write_text(html_output, render_comparison_html(result))
        print(f"Comparison HTML written to: {html_path.resolve()}")
    return 0


def _strategy_list(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def run_generate(args: argparse.Namespace) -> int:
    language = LANGUAGES[getattr(args, "language", "chinese")]
    if getattr(args, "show_sentences", False):
        for case in load_baseline_cases(args.input, language=language):
            print(f"[{case.case_id}] {language.name}")
            for index, sentence in enumerate(language.split(case.text), start=1):
                print(f"  {index}. {sentence.strip()}")
            print(f"  annotated: {', '.join(str(i) for i in case.evidence_sentences)}")
        return 0

    output_path = Path(args.output)
    if output_path.exists() and not getattr(args, "append", False):
        raise ValueError(
            f"Case file already exists: {output_path}. Hand edits live in this "
            "file, so it is never overwritten; use --append to merge or choose "
            "a new path."
        )
    cases = load_baseline_cases(args.input, language=language)
    run = generate_variants(
        cases,
        seed=args.seed,
        gaming=_strategy_list(args.gaming),
        degradation=_strategy_list(args.degradation),
        paraphrase=_strategy_list(getattr(args, "paraphrase", None)),
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
    merge_summary: dict[str, Any] = {
        "appended": [],
        "preserved": [[row.case_id, row.variant_id] for row in rows],
        "edited": [],
        "foreign": [],
    }
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
        merge_summary = {
            "appended": [list(item) for item in outcome.appended],
            "preserved": [list(item) for item in outcome.preserved],
            "edited": [list(item) for item in outcome.edited],
            "foreign": [list(item) for item in outcome.foreign],
        }

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
        "output_sha256": stable_hash(
            [
                {
                    "case_id": row.case_id,
                    "variant_id": row.variant_id,
                    "variant_type": row.variant_type,
                    "text": row.text,
                    "notes": row.notes,
                }
                for row in rows
            ]
        ),
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

    from .variants import GeneratedRow

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


def run_paraphrase(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    if output_path.exists():
        raise ValueError(
            f"Review file already exists: {output_path}. Reviewer decisions live "
            "in this file, so it is never overwritten; choose a new path."
        )

    language = LANGUAGES[args.language]
    cases = load_baseline_cases(args.input, language=language)

    api_key = os.environ.get(args.api_key_env, "")
    hostname = urlsplit(args.base_url).hostname
    if not api_key and hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            f"Environment variable {args.api_key_env!r} is not set. "
            "API keys are accepted only through environment variables."
        )
    if not api_key:
        api_key = "local-provider"

    print(
        "Warning: baseline texts are being sent to the model provider. Confirm "
        "you are authorised to share this material before continuing.",
        file=sys.stderr,
    )

    config = OpenAICompatibleConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    config.validate()
    drafts = draft_paraphrases(cases, OpenAICompatibleScorer(config), language)

    review_path = write_paraphrase_review(output_path, drafts)
    manifest_path = (
        Path(args.manifest)
        if getattr(args, "manifest", None)
        else review_path.with_suffix(".manifest.json")
    )
    blocked = sum(1 for draft in drafts if draft.status == "blocked")
    write_json(
        manifest_path,
        {
            "generator": "agent-review",
            "paraphrase_model": args.model,
            "base_url": args.base_url,
            "language": language.name,
            "temperature": args.temperature,
            "case_count": len(cases),
            "blocked_count": blocked,
            "requires_human_review": True,
            "input_path": str(Path(args.input).resolve()),
            "drafts": [
                {
                    "case_id": draft.case_id,
                    "baseline_sha256": draft.baseline_sha256,
                    "status": draft.status,
                    "blocking_checks": list(draft.blocking_checks),
                    "review_notes": list(draft.review_notes),
                    "latency_seconds": round(draft.latency_seconds, 4),
                    "raw_content": draft.raw_content,
                }
                for draft in drafts
            ],
        },
    )

    print(f"Review file written to: {review_path.resolve()}")
    print(f"Manifest written to: {manifest_path.resolve()}")
    print(
        f"Drafted {len(drafts)} paraphrases; {blocked} blocked by checks. "
        "Nothing enters the case set until you set status=approved: no machine "
        "check can prove two texts mean the same thing."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "audit":
        try:
            return run_audit(args)
        except (ValueError, ProviderError) as exc:
            parser.error(str(exc))
    if args.command == "score":
        try:
            return run_score(args)
        except (ValueError, ProviderError) as exc:
            parser.error(str(exc))
    if args.command == "compare":
        try:
            return run_compare(args)
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "generate":
        try:
            return run_generate(args)
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "paraphrase":
        try:
            return run_paraphrase(args)
        except (ValueError, ProviderError) as exc:
            parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

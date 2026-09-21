from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from .audit import AuditConfig, audit_records
from .comparison import compare_audits, load_audit_result, render_comparison_report
from .html_report import render_audit_html, render_comparison_html
from .io import (
    load_score_records,
    load_scoring_cases,
    write_score_records,
    write_text,
)
from .provider import OpenAICompatibleConfig, OpenAICompatibleScorer, ProviderError
from .report import render_markdown_report
from .scoring import run_scoring, write_json, write_jsonl


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
    return parser


def run_audit(args: argparse.Namespace) -> int:
    config = AuditConfig(
        invariance_tolerance=args.invariance_tolerance,
        min_degradation_drop=args.min_degradation_drop,
        gaming_tolerance=args.gaming_tolerance,
        score_min=args.score_min,
        score_max=args.score_max,
        data_provenance=args.data_provenance,
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
    report_path = write_text(args.report, render_markdown_report(result))
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
        html_path = write_text(html_output, render_audit_html(result))
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from .audit import AuditConfig, audit_records
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
    result = audit_records(load_score_records(args.input), config)
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
        json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON written to: {json_path.resolve()}")
    return 0


def run_score(args: argparse.Namespace) -> int:
    if args.repeats > 1 and not args.raw_output:
        print(
            "Warning: repeated scoring without --raw-output keeps only the first "
            "sample reason in the aggregate CSV.",
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
    )

    output_path = write_score_records(args.output, list(scoring_run.records))
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(
        ".manifest.json"
    )
    write_json(manifest_path, scoring_run.manifest)
    print(f"Scores written to: {output_path.resolve()}")
    print(f"Manifest written to: {manifest_path.resolve()}")
    if args.raw_output:
        raw_path = write_jsonl(args.raw_output, scoring_run.traces)
        print(f"Raw traces written to: {raw_path.resolve()}")
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

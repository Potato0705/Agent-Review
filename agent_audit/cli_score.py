"""The `score` subcommand: call a model and record what it said."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from .io import load_scoring_cases, write_score_records
from .provider import OpenAICompatibleConfig, OpenAICompatibleScorer, ProviderError
from .scoring import run_scoring, write_json, write_jsonl


USAGE_ERRORS: tuple[type[Exception], ...] = (ValueError, ProviderError)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "score", help="Score case texts with an OpenAI-compatible chat API."
    )
    command.add_argument("--input", required=True, help="Input case CSV.")
    command.add_argument("--output", required=True, help="Output score CSV.")
    command.add_argument("--rubric-file", required=True, help="UTF-8 rubric file.")
    command.add_argument("--model", required=True)
    command.add_argument(
        "--base-url", default="https://api.openai.com/v1", help="API base URL."
    )
    command.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    command.add_argument("--system-name", help="Name stored in the score CSV.")
    command.add_argument("--score-min", type=float, default=0.0)
    command.add_argument("--score-max", type=float, default=10.0)
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Independent scoring calls per input (1-100). This multiplies API usage.",
    )
    command.add_argument("--timeout", type=float, default=60.0)
    command.add_argument("--max-retries", type=int, default=2)
    command.add_argument(
        "--manifest", help="Run manifest path; defaults next to the output CSV."
    )
    command.add_argument(
        "--raw-output",
        help="Optional JSONL trace path. May contain model-generated sensitive text.",
    )
    command.add_argument(
        "--checkpoint",
        help=(
            "Optional per-sample JSONL checkpoint for crash recovery. "
            "Contains full model replies and must be stored securely."
        ),
    )
    command.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching existing --checkpoint without repeating saved calls.",
    )



def run(args: argparse.Namespace) -> int:
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



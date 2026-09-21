"""The `paraphrase` subcommand: draft rewrites for a person to ratify.

The command writes a review file and never touches the case set. No machine
check can prove two texts mean the same thing, so the tool does not claim one
does; it produces candidates and a focused list of things to look at.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from .io import load_baseline_cases, write_paraphrase_review
from .paraphrase import draft_paraphrases
from .provider import OpenAICompatibleConfig, OpenAICompatibleScorer, ProviderError
from .scoring import write_json
from .segmentation import LANGUAGES


USAGE_ERRORS: tuple[type[Exception], ...] = (ValueError, ProviderError)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "paraphrase",
        help="Draft paraphrase candidates for human review. Calls a model.",
    )
    command.add_argument("--input", required=True, help="Annotated baseline CSV.")
    command.add_argument("--output", required=True, help="Review CSV path.")
    command.add_argument("--model", required=True)
    command.add_argument(
        "--base-url", default="https://api.openai.com/v1", help="API base URL."
    )
    command.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    command.add_argument(
        "--language", choices=sorted(LANGUAGES), default="chinese"
    )
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--timeout", type=float, default=60.0)
    command.add_argument("--max-retries", type=int, default=2)
    command.add_argument(
        "--manifest", help="Run manifest path; defaults next to the review CSV."
    )



def run(args: argparse.Namespace) -> int:
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



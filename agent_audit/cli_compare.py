"""The `compare` subcommand: two audits of the same cases, side by side."""

from __future__ import annotations

import argparse

from .comparison import compare_audits, load_audit_result, render_comparison_report
from .html_report import render_comparison_html
from .io import write_text
from .scoring import write_json


USAGE_ERRORS: tuple[type[Exception], ...] = (ValueError,)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "compare", help="Compare two compatible audit JSON results."
    )
    command.add_argument("--reference", required=True, help="Reference audit JSON.")
    command.add_argument("--candidate", required=True, help="Candidate audit JSON.")
    command.add_argument("--report", required=True, help="Output Markdown report.")
    command.add_argument(
        "--json", dest="json_output", help="Optional JSON result path."
    )
    command.add_argument(
        "--html", dest="html_output", help="Optional self-contained HTML report path."
    )


def run(args: argparse.Namespace) -> int:
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

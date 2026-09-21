"""The `audit` subcommand and the evidence chain it verifies.

The scoring manifest fingerprints what was scored; the generation manifest
fingerprints what was generated. Matching them is what turns a variant-origin
declaration from something a user typed into something the tool can check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .audit import AuditConfig, audit_records
from .html_report import render_audit_html
from .io import load_score_records, stable_hash, write_text
from .report import render_markdown_report


USAGE_ERRORS: tuple[type[Exception], ...] = (ValueError,)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser("audit", help="Audit a score CSV and create a report.")
    command.add_argument("--input", required=True, help="Input score CSV.")
    command.add_argument("--report", required=True, help="Output Markdown report.")
    command.add_argument("--json", dest="json_output", help="Optional JSON result path.")
    command.add_argument(
        "--html", dest="html_output", help="Optional self-contained HTML report path."
    )
    command.add_argument(
        "--manifest",
        help="Optional scoring manifest; auto-detected next to the input CSV when present.",
    )
    command.add_argument("--invariance-tolerance", type=float, default=0.5)
    command.add_argument("--min-degradation-drop", type=float, default=1.0)
    command.add_argument("--gaming-tolerance", type=float, default=0.0)
    command.add_argument("--score-min", type=float)
    command.add_argument("--score-max", type=float)
    command.add_argument(
        "--data-provenance",
        choices=["synthetic", "public-demo", "authorized-private", "unspecified"],
        default="unspecified",
        help="Declared origin of the evaluated data.",
    )
    command.add_argument(
        "--generation-manifest",
        help=(
            "Generation manifest proving the scored cases are the generated "
            "ones. Required by --variant-origin machine-generated."
        ),
    )
    command.add_argument(
        "--variant-origin",
        choices=["human-authored", "machine-generated", "mixed", "unspecified"],
        default="unspecified",
        help="Whether the variants were written by hand or generated.",
    )



def run(args: argparse.Namespace) -> int:
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



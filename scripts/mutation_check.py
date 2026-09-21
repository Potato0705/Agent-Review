"""Mutation gate: prove the test suite actually defends the decision logic.

Line coverage says a line ran. It does not say anything would fail if that line
were wrong. This project publishes a measurement, so a silently wrong threshold
or a flipped sign is worse than a crash: it produces a confident, wrong client
report. Each mutant below is a defect this project could plausibly ship. A
mutant that survives means no test distinguishes correct behaviour from that
defect, and the gate fails.

Run it directly::

    python scripts/mutation_check.py
    python scripts/mutation_check.py --list

Adding decision logic means adding a mutant here. Removing one is only
legitimate when the code it targets is gone.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mutant:
    module: str
    label: str
    original: str
    mutated: str

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / "agent_audit" / f"{self.module}.py"


MUTANTS: tuple[Mutant, ...] = (
    # --- fail-closed input validation ---------------------------------------
    Mutant(
        "audit",
        "accept non-finite scores",
        "        if not math.isfinite(row.score):",
        "        if False:",
    ),
    Mutant(
        "provider",
        "accept boolean scores",
        "        if isinstance(raw_score, bool):",
        "        if False:",
    ),
    # --- uncertainty --------------------------------------------------------
    Mutant(
        "audit",
        "use the normal approximation past the t table",
        "    return t_critical_95[-1]",
        "    return 1.96",
    ),
    Mutant(
        "audit",
        "never mark a result provisional",
        "            uncertainty_evaluable_count < variant_count or uncertain_count > 0",
        "            False",
    ),
    # --- the published validity-margin formula ------------------------------
    Mutant(
        "audit",
        "drop the positive-gain clamp",
        "rewarded_gaming = max(0.0, max(case_gaming_gains))",
        "rewarded_gaming = max(case_gaming_gains)",
    ),
    Mutant(
        "audit",
        "use the worst rather than the mean degradation drop",
        "degradation_sensitivity = fmean(case_degradation_drops)",
        "degradation_sensitivity = max(case_degradation_drops)",
    ),
    # --- decision thresholds ------------------------------------------------
    Mutant(
        "audit",
        "treat a gaming gain exactly at tolerance as a violation",
        "                threshold_contrast = gain - cfg.gaming_tolerance\n"
        "                violated = threshold_contrast > 0",
        "                threshold_contrast = gain - cfg.gaming_tolerance\n"
        "                violated = threshold_contrast >= 0",
    ),
    Mutant(
        "audit",
        "treat a degradation drop exactly at the minimum as a violation",
        "                threshold_contrast = drop - cfg.min_degradation_drop\n"
        "                violated = threshold_contrast < 0",
        "                threshold_contrast = drop - cfg.min_degradation_drop\n"
        "                violated = threshold_contrast <= 0",
    ),
    Mutant(
        "audit",
        "treat paraphrase drift exactly at tolerance as a violation",
        "                threshold_contrast = absolute_delta - cfg.invariance_tolerance\n"
        "                violated = threshold_contrast > 0",
        "                threshold_contrast = absolute_delta - cfg.invariance_tolerance\n"
        "                violated = threshold_contrast >= 0",
    ),
    # --- risk classification ------------------------------------------------
    Mutant(
        "audit",
        "disable the critical gaming-gain trigger",
        "        worst_gaming_gain is not None\n"
        "        and worst_gaming_gain >= critical_gaming_gain",
        "        worst_gaming_gain is not None\n"
        "        and worst_gaming_gain >= critical_gaming_gain * 100",
    ),
    Mutant(
        "audit",
        "ignore a non-positive validity margin when ranking risk",
        "    if violation_rate >= 0.5 or (\n"
        "        mean_validity_margin is not None and mean_validity_margin <= 0.0\n"
        "    ):",
        "    if violation_rate >= 0.5:",
    ),
    # --- evidence grading ---------------------------------------------------
    Mutant(
        "report",
        "shift the demonstration-grade boundary",
        "    if result.case_count < 5:",
        "    if result.case_count < 2:",
    ),
    Mutant(
        "report",
        "shift the exploratory-grade boundary",
        "    if result.case_count < 30:",
        "    if result.case_count < 300:",
    ),
    Mutant(
        "html_report",
        "shift the demonstration-grade boundary in HTML",
        "    if case_count < 5:",
        "    if case_count < 2:",
    ),
    Mutant(
        "html_report",
        "shift the exploratory-grade boundary in HTML",
        "    if case_count < 30:",
        "    if case_count < 300:",
    ),
    # --- comparison direction and comparability -----------------------------
    Mutant(
        "comparison",
        "flip the degradation severity direction",
        '        return delta + float(config["min_degradation_drop"])',
        '        return -delta + float(config["min_degradation_drop"])',
    ),
    Mutant(
        "comparison",
        "ignore paraphrase drift direction",
        '    return abs(delta) - float(config["invariance_tolerance"])',
        '    return delta - float(config["invariance_tolerance"])',
    ),
    Mutant(
        "comparison",
        "allow audits with different scoring contexts",
        "    if context_mismatches:",
        "    if False:",
    ),
    Mutant(
        "comparison",
        "allow audits with different decision thresholds",
        "    if mismatched_fields:",
        "    if False:",
    ),
    Mutant(
        "comparison",
        "allow audits with different variant identities",
        "    if set(reference_variants) != set(candidate_variants):",
        "    if False:",
    ),
    # --- long-run checkpoint integrity ---------------------------------------
    Mutant(
        "scoring",
        "resume samples that belong to another run",
        "    if unexpected_samples:",
        "    if False:",
    ),
    Mutant(
        "scoring",
        "allow any repeat count",
        "    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 100:",
        "    if False:",
    ),
    Mutant(
        "checkpoint",
        "resume a checkpoint recorded under a different context",
        "    if context != expected_context:",
        "    if False:",
    ),
    Mutant(
        "checkpoint",
        "accept duplicate samples in a checkpoint",
        "        if sample.identity in samples:",
        "        if False:",
    ),
    # --- manifest evidence chain ---------------------------------------------
    Mutant(
        "cli",
        "accept a manifest whose record count contradicts the CSV",
        "    if record_count != expected_record_count:",
        "    if False:",
    ),
    # --- variant origin declaration -------------------------------------------
    Mutant(
        "comparison",
        "allow audits with different variant origins",
        '    if reference_config.get("variant_origin") != candidate_config.get(\n'
        '        "variant_origin"\n'
        "    ):",
        "    if False:",
    ),
    Mutant(
        "report",
        "drop the floor-test warning for generated variants",
        '            "变体由工具生成，属于下限测试；"\n'
        '            "未通过是确凿证据，通过不能证明系统可靠。"',
        '            "变体由工具生成。"',
    ),
    Mutant(
        "html_report",
        "drop the floor-test warning from the delivered page",
        '            "变体由工具生成，属于下限测试；"\n'
        '            "未通过是确凿证据，通过不能证明系统可靠。"',
        '            "变体由工具生成。"',
    ),
    # --- English splitting and joining ----------------------------------------
    Mutant(
        "segmentation",
        "split on every period, abbreviations and decimals included",
        "        if not _ends_a_sentence(text, index, strategy):",
        "        if False:",
    ),
    Mutant(
        "segmentation",
        "treat an abbreviation as the end of a sentence",
        "        if letters.lower() in language.abbreviations:",
        "        if False:",
    ),
    Mutant(
        "variants",
        "replace a connective inside a longer English word",
        "                if require_word_boundaries:",
        "                if False:",
    ),
    Mutant(
        "io",
        "ignore a declared sentence count that disagrees with the splitter",
        "                if int(declared) != len(sentences):",
        "                if False:",
    ),
    # --- appending must not destroy or launder hand-written work --------------
    Mutant(
        "cli",
        "overwrite an existing case file and its hand edits",
        "    if output_path.exists() and not append:",
        "    if False:",
    ),
    Mutant(
        "cli",
        "accept a mixed set as machine-generated",
        '    if declared_origin == "machine-generated" and set_origin != "machine-generated":',
        "    if False:",
    ),
    Mutant(
        "variants",
        "call an edited set machine-generated anyway",
        '        return "mixed" if self.edited or self.foreign else "machine-generated"',
        '        return "machine-generated"',
    ),
    Mutant(
        "variants",
        "overwrite an existing row with the freshly generated one",
        "        if current is None:\n"
        "            rows.append(row)\n"
        "            appended.append(identity)\n"
        "            continue",
        "        if True:\n"
        "            rows.append(row)\n"
        "            appended.append(identity)\n"
        "            continue",
    ),
    # --- a machine-generated claim must be provable ----------------------------
    Mutant(
        "cli",
        "accept scored cases the generator never produced",
        "    if output_sha256 != scored_input_sha256.lower():",
        "    if False:",
    ),
    Mutant(
        "cli",
        "let a machine-generated claim go unproven",
        '    if config.variant_origin == "machine-generated" and not generation_argument:',
        "    if False:",
    ),
    Mutant(
        "comparison",
        "compare audits from different generation runs",
        '    if (reference_generation is None) != (candidate_generation is None) or (\n'
        "        reference_generation is not None\n"
        '        and str(reference_generation).lower() != str(candidate_generation).lower()\n'
        "    ):",
        "    if False:",
    ),
    # --- generated variants must match their own label ------------------------
    Mutant(
        "variants",
        "accept a gaming variant that removed content",
        "    if not produced.startswith(baseline) or len(produced) <= len(baseline):",
        "    if False:",
    ),
    Mutant(
        "variants",
        "accept a degradation variant that kept the evidence",
        "        if annotated in produced:",
        "        if False:",
    ),
    Mutant(
        "variants",
        "substitute connectives anywhere instead of at clause boundaries",
        "        at_boundary = index == 0 or text[index - 1] in CLAUSE_BOUNDARIES",
        "        at_boundary = True",
    ),
    Mutant(
        "variants",
        "allow a strategy family to be empty",
        "    if not selected:",
        "    if False:",
    ),
    Mutant(
        "io",
        "accept an evidence index past the last sentence",
        "            if indices[-1] > len(sentences):",
        "            if False:",
    ),
    Mutant(
        "io",
        "accept an annotation that covers every sentence",
        "            if len(indices) == len(sentences):",
        "            if False:",
    ),
    # --- a model rewrite is a draft until a human ratifies it -----------------
    Mutant(
        "cli",
        "merge rewrites nobody approved",
        '        if row.status != "approved":',
        "        if False:",
    ),
    Mutant(
        "cli",
        "merge a rewrite drafted against a since-edited baseline",
        "        if row.baseline_sha256 != expected[row.case_id]:",
        "        if False:",
    ),
    Mutant(
        "cli",
        "keep a human-ratified set labelled machine-generated",
        "        if ratified:\n"
        '            set_origin = "mixed"',
        "        if False:\n"
        '            set_origin = "mixed"',
    ),
    Mutant(
        "cli",
        "re-append ratified rewrites that are already in the file",
        "        present = {(row.case_id, row.variant_id) for row in rows}",
        "        present = set()",
    ),
    Mutant(
        "cli",
        "let a paraphrase review bypass --append and drop the existing set",
        '    if review_argument and not getattr(args, "append", False):',
        "    if False:",
    ),
    Mutant(
        "cli",
        "accept a generated set declared as human-authored",
        '    if declared_origin == "human-authored":',
        "    if False:",
    ),
    Mutant(
        "cli",
        "grade rewrites with the model that wrote them",
        "            and paraphrase_model.casefold() == scoring_model.casefold()",
        "            and False",
    ),
    Mutant(
        "cli",
        "guess at the drafting model instead of refusing",
        "    if not manifest_path.exists():\n"
        "        raise ValueError(\n"
        '            f"Paraphrase manifest does not exist: {manifest_path}. It names the "',
        "    if False:\n"
        "        raise ValueError(\n"
        '            f"Paraphrase manifest does not exist: {manifest_path}. It names the "',
    ),
    Mutant(
        "io",
        "silently skip a misspelled review status",
        '            if values["status"] not in REVIEW_STATUSES:',
        "            if False:",
    ),
    Mutant(
        "paraphrase",
        "approve a draft that just echoes the baseline",
        "    elif stripped_baseline and stripped_baseline in stripped_draft:",
        "    elif False:",
    ),
    Mutant(
        "paraphrase",
        "approve a truncated or runaway draft",
        "        if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:",
        "        if False:",
    ),
    Mutant(
        "paraphrase",
        "promote the number check back into a gate",
        '        notes.append(\n            "numbers in the baseline not found in the draft: "',
        '        blocking.append(\n            "numbers in the baseline not found in the draft: "',
    ),
    Mutant(
        "paraphrase",
        "send a draft to review without flagging its failed checks",
        '                status="blocked" if blocking else "pending",',
        '                status="pending",',
    ),
    # --- trajectory variants must match their own label -----------------------
    Mutant(
        "trajectory_variants",
        "let a gaming variant invent a call the agent never made",
        "        if identity not in allowed:",
        "        if False:",
    ),
    Mutant(
        "trajectory_variants",
        "let a gaming variant rewrite the final answer",
        "    if produced.final_answer != baseline.final_answer:\n"
        "        raise TrajectoryPostconditionError(\n"
        '            "redundant_tool_calls must leave the final answer untouched."',
        "    if False:\n"
        "        raise TrajectoryPostconditionError(\n"
        '            "redundant_tool_calls must leave the final answer untouched."',
    ),
    Mutant(
        "trajectory_variants",
        "let padding change what the agent actually did",
        "    if _identities(produced.steps) != _identities(baseline.steps):",
        "    if False:",
    ),
    Mutant(
        "trajectory_variants",
        "drop the final answer from a degradation variant",
        "    if produced.final_answer != baseline.final_answer:\n"
        "        raise TrajectoryPostconditionError(\n"
        '            "remove_load_bearing_step must keep the final answer: changing the "',
        "    if False:\n"
        "        raise TrajectoryPostconditionError(\n"
        '            "remove_load_bearing_step must keep the final answer: changing the "',
    ),
    Mutant(
        "trajectory_variants",
        "keep the remaining steps in any order at all",
        "    if _identities(produced.steps) != _identities(tuple(kept)):",
        "    if False:",
    ),
    Mutant(
        "trajectory_variants",
        "hollow out a step nobody annotated",
        "        elif before.result != after.result:",
        "        elif False:",
    ),
    Mutant(
        "trajectory_variants",
        "call an already-empty result hollowed out",
        "            if not before.result.strip():",
        "            if False:",
    ),
    Mutant(
        "trajectory_variants",
        "reorder steps the reviewer never called independent",
        "    if not case.independent_steps:",
        "    if False:",
    ),
    Mutant(
        "trajectory_variants",
        "ship the baseline order as a paraphrase variant",
        "    if _identities(produced.steps) == _identities(baseline.steps):",
        "    if False:",
    ),
    Mutant(
        "trajectory",
        "infer that unannotated steps are load-bearing",
        "    if not indices:",
        "    if False:",
    ),
    Mutant(
        "trajectory",
        "accept an annotation covering every step",
        "    if len(indices) == step_count:",
        "    if False:",
    ),
    Mutant(
        "trajectory",
        "accept a step in two independent groups",
        "        if overlap:",
        "        if False:",
    ),
    Mutant(
        "trajectory",
        "accept an out-of-range load-bearing index",
        "    if any(index < 1 or index > step_count for index in indices):",
        "    if False:",
    ),
    Mutant(
        "trajectory",
        "serialise call arguments in whatever order they arrived",
        "        return json.dumps(raw, ensure_ascii=False, sort_keys=True)",
        "        return json.dumps(raw, ensure_ascii=False)",
    ),
    # --- untrusted input ------------------------------------------------------
    Mutant(
        "io",
        "accept a non-finite score from a CSV",
        "            if not math.isfinite(score):",
        "            if False:",
    ),
    Mutant(
        "provider",
        "retry a terminal client error",
        "                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}",
        "                retryable = True",
    ),
)


RESTORE_ATTEMPTS = 5
RESTORE_PAUSE_SECONDS = 0.2


def write_source(path: Path, data: bytes) -> bool:
    """Write ``data`` to ``path``, retrying, and never raising.

    A real run on Windows died here with ``OSError: [Errno 22]`` while putting
    a module back, and because the write raised inside the ``finally`` block
    the restoration check below never ran: the process ended on a traceback
    with a mutated module still in the working tree. That is the one outcome
    this script must never produce, so every write goes through here and
    reports failure as a value the caller has to handle.
    """

    for attempt in range(RESTORE_ATTEMPTS):
        try:
            path.write_bytes(data)
            return True
        except OSError:
            if attempt + 1 == RESTORE_ATTEMPTS:
                return False
            time.sleep(RESTORE_PAUSE_SECONDS)
    return False


def _run_suite(env: dict[str, str]) -> bool:
    """Return True when the suite passes."""

    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="Print the catalogue and exit."
    )
    args = parser.parse_args(argv)

    if args.list:
        for mutant in MUTANTS:
            print(f"{mutant.module:<12} {mutant.label}")
        print(f"\n{len(MUTANTS)} mutants.")
        return 0

    # Bytecode caching is disabled for every child run. A mutation can be the
    # same byte length as the original, and on a coarse filesystem clock the
    # restored file can reuse a cached .pyc from the mutated source.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}

    targets = {mutant.path for mutant in MUTANTS}
    originals = {path: path.read_bytes() for path in targets}

    for mutant in MUTANTS:
        source = originals[mutant.path].decode("utf-8")
        occurrences = source.count(mutant.original)
        if occurrences != 1:
            print(
                f"Mutant anchor is stale: {mutant.module} / {mutant.label} "
                f"matched {occurrences} times. Update scripts/mutation_check.py.",
                file=sys.stderr,
            )
            return 1

    print("Verifying the suite is green before mutating...")
    if not _run_suite(env):
        print("The test suite already fails; fix that first.", file=sys.stderr)
        return 1

    survivors: list[Mutant] = []
    aborted = ""
    try:
        for index, mutant in enumerate(MUTANTS, start=1):
            source = originals[mutant.path].decode("utf-8")
            mutated = source.replace(mutant.original, mutant.mutated).encode("utf-8")
            if not write_source(mutant.path, mutated):
                aborted = f"could not write {mutant.path}"
                break
            caught = not _run_suite(env)
            if not write_source(mutant.path, originals[mutant.path]):
                aborted = f"could not restore {mutant.path}"
                break
            status = "caught  " if caught else "SURVIVED"
            print(f"  [{index:>2}/{len(MUTANTS)}] {status}  {mutant.module}: {mutant.label}")
            if not caught:
                survivors.append(mutant)
    finally:
        for path, data in originals.items():
            write_source(path, data)

    damaged = [
        path
        for path, data in originals.items()
        if hashlib.sha256(path.read_bytes()).hexdigest()
        != hashlib.sha256(data).hexdigest()
    ]
    if damaged:
        print(
            "FATAL: these files still hold a mutation and could be committed. "
            "Restore them before doing anything else:",
            file=sys.stderr,
        )
        for path in damaged:
            print(f"  git checkout -- {path.relative_to(PROJECT_ROOT)}", file=sys.stderr)
        return 2

    if aborted:
        print(
            f"Mutation run aborted: {aborted}. Every source file was restored, "
            "so the working tree is clean; rerun the gate.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants caught; "
        "all source files restored."
    )
    if survivors:
        print("\nMutation gate failed. No test distinguishes these defects:", file=sys.stderr)
        for mutant in survivors:
            print(f"  - {mutant.module}: {mutant.label}", file=sys.stderr)
        return 1

    print("Mutation gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

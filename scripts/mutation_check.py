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
)


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
    try:
        for index, mutant in enumerate(MUTANTS, start=1):
            source = originals[mutant.path].decode("utf-8")
            mutant.path.write_bytes(
                source.replace(mutant.original, mutant.mutated).encode("utf-8")
            )
            caught = not _run_suite(env)
            mutant.path.write_bytes(originals[mutant.path])
            status = "caught  " if caught else "SURVIVED"
            print(f"  [{index:>2}/{len(MUTANTS)}] {status}  {mutant.module}: {mutant.label}")
            if not caught:
                survivors.append(mutant)
    finally:
        for path, data in originals.items():
            path.write_bytes(data)

    for path, data in originals.items():
        restored = path.read_bytes()
        if hashlib.sha256(restored).hexdigest() != hashlib.sha256(data).hexdigest():
            print(
                f"FATAL: {path} was not restored. Restore it from Git before "
                "continuing.",
                file=sys.stderr,
            )
            return 2

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

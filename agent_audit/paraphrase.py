"""Model-drafted paraphrase candidates and the checks a reviewer gets with them.

No machine check can prove two texts mean the same thing, so this module never
claims one does. It produces drafts for a human to ratify, blocks only the
cases that are unambiguously wrong, and hands everything else to the reviewer
as a focused list of things to look at.

The split between gate and note is measured, not guessed. Number and negation
checks reject 2 of the 5 hand-written paraphrases in ``examples``: the
"missing number" in one is the 一 inside 统一, and the "lost negation" in
another is 不足 rewritten as 缺乏. A gate that rejects 40% of genuine work
trains reviewers to ignore it, so those two are notes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from .segmentation import LanguageStrategy


MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 2.5

NUMBER_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)?")

REVIEW_STATUSES = ("pending", "blocked", "approved", "rejected")


class Rewriter(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        ...


@dataclass(frozen=True)
class ParaphraseDraft:
    case_id: str
    baseline_sha256: str
    baseline_text: str
    draft_text: str
    status: str
    blocking_checks: tuple[str, ...]
    review_notes: tuple[str, ...]
    raw_content: str
    latency_seconds: float


def _numbers(text: str, language: LanguageStrategy) -> list[str]:
    """Return number tokens worth drawing a reviewer's eye to.

    A Chinese numeral only counts when a measure word follows it. Matching
    bare numerals would report the 一 inside 统一 and 一起 as missing numbers,
    which is exactly the false positive that made this a note rather than a
    gate in the first place.
    """

    found = NUMBER_PATTERN.findall(text)
    if language.numeral_characters and language.measure_words:
        units = "|".join(re.escape(word) for word in language.measure_words)
        pattern = re.compile(f"[{language.numeral_characters}]+(?:{units})")
        found.extend(pattern.findall(text))
    return found


def _negation_count(text: str, language: LanguageStrategy) -> int:
    if not language.negation_markers:
        return 0
    if language.requires_word_boundaries:
        words = re.findall(r"[A-Za-z']+", text.lower())
        return sum(1 for word in words if word in language.negation_markers)
    return sum(text.count(marker) for marker in language.negation_markers)


def review_checks(
    baseline: str, draft: str, language: LanguageStrategy
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(blocking, notes)`` for one draft.

    ``blocking`` names checks that are unambiguously wrong and must be fixed
    before approval. ``notes`` are for the reviewer's attention only and never
    change the outcome.
    """

    stripped_baseline = baseline.strip()
    stripped_draft = draft.strip()

    blocking: list[str] = []
    if not stripped_draft:
        blocking.append("empty")
        return tuple(blocking), ()
    if stripped_draft == stripped_baseline:
        blocking.append("identical")
    elif stripped_baseline and stripped_baseline in stripped_draft:
        blocking.append("echoes_baseline")
    if stripped_baseline:
        ratio = len(stripped_draft) / len(stripped_baseline)
        if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
            blocking.append("length_out_of_band")

    notes: list[str] = []
    baseline_numbers = _numbers(stripped_baseline, language)
    missing = [
        number for number in baseline_numbers if number not in stripped_draft
    ]
    if missing:
        notes.append(
            "numbers in the baseline not found in the draft: "
            + ", ".join(sorted(set(missing)))
        )
    baseline_negations = _negation_count(stripped_baseline, language)
    draft_negations = _negation_count(stripped_draft, language)
    if baseline_negations != draft_negations:
        notes.append(
            f"negation markers changed from {baseline_negations} to {draft_negations}"
        )

    return tuple(blocking), tuple(notes)


def build_messages(text: str, language: LanguageStrategy) -> list[dict[str, str]]:
    """Ask for a faithful rewrite and nothing else.

    The rubric is deliberately absent: a rewrite optimised against the very
    criteria under test would be circular. So is any hint that the output will
    be used to probe a grader, which invites adversarial rather than faithful
    rewriting.
    """

    return [
        {"role": "system", "content": language.paraphrase_instruction},
        {"role": "user", "content": text},
    ]

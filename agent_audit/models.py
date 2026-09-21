from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


VariantType = Literal["baseline", "gaming", "degradation", "paraphrase"]


@dataclass(frozen=True)
class ScoringCase:
    case_id: str
    variant_id: str
    variant_type: VariantType
    text: str
    notes: str = ""


@dataclass(frozen=True)
class BaselineCase:
    """A baseline text plus the sentences a degradation variant must remove.

    ``evidence_sentences`` holds 1-based indices into
    ``segmentation.split_sentences(text)``. The generator never infers which
    sentences carry the argument; a wrong guess would produce a variant that
    claims to be degraded without being degraded.
    """

    case_id: str
    text: str
    evidence_sentences: tuple[int, ...]
    notes: str = ""


@dataclass(frozen=True)
class ParaphraseReviewRow:
    """One line of the file a reviewer ratifies paraphrase drafts in."""

    case_id: str
    baseline_sha256: str
    draft_text: str
    status: str
    reviewer_note: str = ""


@dataclass(frozen=True)
class ScoreRecord:
    system_name: str
    case_id: str
    variant_id: str
    variant_type: VariantType
    score: float
    notes: str = ""
    score_stddev: float | None = None
    sample_count: int = 1


@dataclass(frozen=True)
class ProviderScore:
    score: float
    reason: str
    raw_content: str
    response_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class VariantFinding:
    case_id: str
    variant_id: str
    variant_type: VariantType
    baseline_score: float
    variant_score: float
    delta: float
    violated: bool
    uncertain: bool
    delta_standard_error: float | None
    variant_score_stddev: float | None
    variant_sample_count: int
    message: str


@dataclass(frozen=True)
class CaseFinding:
    case_id: str
    baseline_score: float
    baseline_score_stddev: float | None
    baseline_sample_count: int
    variants: tuple[VariantFinding, ...]
    validity_margin: float | None

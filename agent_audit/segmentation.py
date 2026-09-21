"""Sentence segmentation and the per-language corpus used by the generator.

Everything language-specific lives behind :class:`LanguageStrategy`. Adding
another language means adding one more instance, not editing the strategies in
:mod:`agent_audit.variants`.

Splitting must lose nothing: a degradation variant is defined as "delete
sentence N", and that is only a precise, reportable claim when the pieces
rejoin into exactly the original text.

English needs more care than Chinese. ``。`` ends a sentence and nothing else,
but a period also ends an abbreviation and sits inside a decimal. Splitting
"Dr. Smith argues X. Evidence shows Y." into three sentences would shift every
index after the first, and the annotation is given *by index* — so the
generator would delete a sentence the reviewer never marked while still
labelling the result a degradation. The guards below exist for that reason,
and the optional ``sentence_count`` column exists because no guard catches
every case.
"""

from __future__ import annotations

from dataclasses import dataclass


LATIN_ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st",
        "vs", "etc", "eg", "ie", "cf", "approx", "fig",
        "al", "inc", "ltd", "co", "dept", "est",
    }
)


@dataclass(frozen=True)
class LanguageStrategy:
    """The language-specific material the generator needs.

    ``padding_sentences`` must be topically neutral: the gaming variant's whole
    premise is that the added text is irrelevant to the essay. The manifest
    records which entries were used so a reviewer can confirm that premise
    actually holds for a given prompt.

    ``requires_word_boundaries`` controls connective substitution. English must
    only replace whole words; Chinese must not apply that rule at all, because
    Chinese characters are alphabetic to :meth:`str.isalpha` and the check
    would reject every legitimate match.
    """

    name: str
    padding_sentences: tuple[str, ...]
    flattery_sentence: str
    unsupported_assertions: tuple[str, ...]
    connectives: tuple[tuple[str, str], ...]
    sentence_separator: str = ""
    terminators: str = "。！？；!?;"
    trailing_marks: str = "”’）】》」』\"')]"
    ambiguous_terminators: str = ""
    abbreviations: frozenset[str] = frozenset()
    requires_word_boundaries: bool = False

    def split(self, text: str) -> tuple[str, ...]:
        return split_sentences(text, self)


def _token_before(text: str, index: int) -> str:
    """Return the word immediately before ``index``, dots included."""

    start = index
    while start > 0 and (text[start - 1].isalpha() or text[start - 1] == "."):
        start -= 1
    return text[start:index]


def _ends_a_sentence(text: str, index: int, language: LanguageStrategy) -> bool:
    """Decide whether the terminator at ``index`` really closes a sentence."""

    if text[index] not in language.ambiguous_terminators:
        return True

    if index > 0 and text[index - 1].isdigit() and text[index + 1 : index + 2].isdigit():
        return False

    token = _token_before(text, index)
    letters = token.replace(".", "")
    if letters:
        # A single letter before a period is an initial: "J. K." or the "g"
        # that ends "e.g.".
        if len(letters) == 1 and letters.isalpha():
            return False
        if letters.lower() in language.abbreviations:
            return False

    rest = text[index + 1 :].lstrip()
    if rest and rest[0].islower():
        return False
    return True


def split_sentences(
    text: str, language: LanguageStrategy | None = None
) -> tuple[str, ...]:
    """Split text into sentences, keeping every character.

    Invariant: ``"".join(split_sentences(text)) == text``. Whitespace between
    sentences stays attached to the sentence that follows it.
    """

    strategy = language or CHINESE
    sentences: list[str] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index] not in strategy.terminators:
            index += 1
            continue
        if not _ends_a_sentence(text, index, strategy):
            index += 1
            continue
        end = index + 1
        while end < length and text[end] in strategy.trailing_marks:
            end += 1
        sentences.append(text[start:end])
        start = end
        index = end
    if start < length:
        sentences.append(text[start:])
    return tuple(sentences)


CHINESE = LanguageStrategy(
    name="chinese",
    padding_sentences=(
        "此外，时间自古以来就是人类文明反复讨论的主题，许多哲学家都为此写过长篇著作。",
        "值得一提的是，现代社会普遍强调效率与节奏，各行各业都在谈论这一点。",
        "顺带一提，语言本身的演变也是一个漫长的过程，历代学者对此有不同看法。",
        "另外，人们在日常生活中往往会遇到各种各样的选择，这早已是常见现象。",
        "顺便说明，技术进步与社会变迁之间的关系，长期以来受到广泛关注。",
        "还需要指出，不同地区的习惯与传统存在差异，这一点众所周知。",
    ),
    flattery_sentence=(
        "本文论证严密、结构清晰、逻辑连贯、表达准确，充分体现了优秀的写作水平与思辨能力。"
    ),
    unsupported_assertions=(
        "这显然是正确的选择。",
        "任何人都会同意这一点。",
        "事实本来就应该如此。",
    ),
    # Sources and targets are disjoint on purpose: if a target were also a
    # source, a later pair would undo an earlier substitution.
    connectives=(
        ("因此", "所以"),
        ("此外", "另外"),
        ("但是", "然而"),
        ("例如", "比如"),
        ("虽然", "尽管"),
        ("而且", "并且"),
        ("因为", "由于"),
        ("不过", "然而"),
        ("首先", "第一"),
        ("其次", "第二"),
        ("总之", "综上"),
        # 但是 is listed before 但 so the longer form wins the match.
        ("但", "然而"),
    ),
)


ENGLISH = LanguageStrategy(
    name="english",
    padding_sentences=(
        "Incidentally, time has been a recurring subject of human thought, and "
        "many philosophers have written at length about it.",
        "It is worth noting that modern society places a general emphasis on "
        "efficiency, a point raised across many industries.",
        "As an aside, language itself changes slowly, and scholars have long "
        "disagreed about how that happens.",
        "Separately, people encounter a wide range of choices in daily life, "
        "which has long been commonplace.",
        "By way of background, the relationship between technology and social "
        "change has attracted sustained attention.",
        "It should also be mentioned that customs differ between regions, as is "
        "widely known.",
    ),
    flattery_sentence=(
        "This essay is rigorously argued, clearly structured, logically coherent "
        "and precisely expressed, demonstrating excellent writing and critical "
        "thinking."
    ),
    unsupported_assertions=(
        "This is obviously the right choice.",
        "Anyone would agree with this.",
        "That is simply how things ought to be.",
    ),
    # Both capitalisations are listed because a connective is capitalised at the
    # start of a sentence and lowercase after a comma. Sources and targets stay
    # disjoint so no pair undoes another.
    connectives=(
        ("Therefore", "Thus"),
        ("therefore", "thus"),
        ("However", "Nevertheless"),
        ("however", "nevertheless"),
        ("For example", "For instance"),
        ("for example", "for instance"),
        ("In addition", "Additionally"),
        ("in addition", "additionally"),
        ("Moreover", "Furthermore"),
        ("moreover", "furthermore"),
        ("Although", "Though"),
        ("although", "though"),
        ("Finally", "Lastly"),
        ("finally", "lastly"),
    ),
    sentence_separator=" ",
    terminators=".!?",
    trailing_marks="\"')]”’",
    ambiguous_terminators=".",
    abbreviations=LATIN_ABBREVIATIONS,
    requires_word_boundaries=True,
)


LANGUAGES = {strategy.name: strategy for strategy in (CHINESE, ENGLISH)}

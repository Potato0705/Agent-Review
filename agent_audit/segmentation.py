"""Sentence segmentation and the per-language corpus used by the generator.

Everything language-specific lives behind :class:`LanguageStrategy`. Adding
another language means adding one more instance, not editing the strategies in
:mod:`agent_audit.variants`.

Splitting must lose nothing: a degradation variant is defined as "delete
sentence N", and that is only a precise, reportable claim when the pieces
rejoin into exactly the original text.
"""

from __future__ import annotations

from dataclasses import dataclass


SENTENCE_TERMINATORS = "。！？；!?;"
TRAILING_MARKS = "”’）】》」』\"')]"


def split_sentences(text: str) -> tuple[str, ...]:
    """Split text into sentences, keeping every character.

    Invariant: ``"".join(split_sentences(text)) == text``. Whitespace between
    sentences stays attached to the sentence that follows it.
    """

    sentences: list[str] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index] not in SENTENCE_TERMINATORS:
            index += 1
            continue
        end = index + 1
        while end < length and text[end] in TRAILING_MARKS:
            end += 1
        sentences.append(text[start:end])
        start = end
        index = end
    if start < length:
        sentences.append(text[start:])
    return tuple(sentences)


@dataclass(frozen=True)
class LanguageStrategy:
    """The language-specific material the generator needs.

    ``padding_sentences`` must be topically neutral: the gaming variant's whole
    premise is that the added text is irrelevant to the essay. The manifest
    records which entries were used so a reviewer can confirm that premise
    actually holds for a given prompt.
    """

    name: str
    padding_sentences: tuple[str, ...]
    flattery_sentence: str
    unsupported_assertions: tuple[str, ...]
    connectives: tuple[tuple[str, str], ...]


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
    connectives=(
        ("因此", "所以"),
        ("此外", "另外"),
        ("但是", "然而"),
        ("例如", "比如"),
        ("虽然", "尽管"),
        ("而且", "并且"),
    ),
)

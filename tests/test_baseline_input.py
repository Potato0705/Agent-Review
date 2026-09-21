"""Fail-closed tests for the annotated baseline input.

The annotation is the only thing that tells the generator which sentences
carry the argument. A wrong index produces a variant that is labelled
"degradation" but is not degraded, so every way of getting it wrong is
rejected at load time, where the row number is still available.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_audit.io import load_baseline_cases


HEADER = "case_id,text,evidence_sentences,notes"
GOOD_TEXT = "学校应推迟上课时间。一项调查显示睡眠充足的学生成绩更稳定。可以先试行一个学期。"


def _write(*rows: str, header: str = HEADER) -> Path:
    path = Path(tempfile.mkdtemp()) / "baselines.csv"
    path.write_text("\n".join((header, *rows)) + "\n", encoding="utf-8", newline="\n")
    return path


def _row(case_id: str = "c1", text: str = GOOD_TEXT, evidence: str = "2", notes: str = "") -> str:
    return f'{case_id},"{text}",{evidence},{notes}'


class BaselineLoadingTests(unittest.TestCase):
    def test_loads_a_well_formed_file(self) -> None:
        cases = load_baseline_cases(_write(_row(notes="原始短文")))

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "c1")
        self.assertEqual(cases[0].text, GOOD_TEXT)
        self.assertEqual(cases[0].evidence_sentences, (2,))
        self.assertEqual(cases[0].notes, "原始短文")

    def test_accepts_several_indices_in_any_order_with_spaces(self) -> None:
        cases = load_baseline_cases(_write(_row(evidence='"3, 2"')))

        self.assertEqual(cases[0].evidence_sentences, (2, 3))

    def test_rejects_a_missing_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_baseline_cases(Path(tempfile.mkdtemp()) / "absent.csv")

    def test_rejects_a_file_without_a_header(self) -> None:
        path = Path(tempfile.mkdtemp()) / "baselines.csv"
        path.write_text("", encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "no header row"):
            load_baseline_cases(path)

    def test_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            load_baseline_cases(_write("c1,text", header="case_id,text"))

    def test_rejects_an_empty_case_id_or_text(self) -> None:
        for row in (_row(case_id=" "), _row(text=" ")):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "empty required values"):
                    load_baseline_cases(_write(row))

    def test_rejects_a_duplicate_case_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            load_baseline_cases(_write(_row(), _row()))

    def test_rejects_a_header_only_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "no baseline cases"):
            load_baseline_cases(_write())

    def test_requires_an_annotation(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence_sentences must not be empty"):
            load_baseline_cases(_write(_row(evidence="")))

    def test_rejects_a_non_integer_index(self) -> None:
        for value in ("two", "1.5", '"1,x"'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "evidence_sentences must be positive integers"
                ):
                    load_baseline_cases(_write(_row(evidence=value)))

    def test_rejects_a_non_positive_index(self) -> None:
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "evidence_sentences must be positive integers"
                ):
                    load_baseline_cases(_write(_row(evidence=value)))

    def test_rejects_a_duplicate_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate evidence_sentences"):
            load_baseline_cases(_write(_row(evidence='"2,2"')))

    def test_rejects_an_index_past_the_last_sentence(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 3 sentences"):
            load_baseline_cases(_write(_row(evidence="4")))

    def test_rejects_a_baseline_with_a_single_sentence(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two sentences"):
            load_baseline_cases(_write(_row(text="只有一句话。", evidence="1")))

    def test_rejects_an_annotation_covering_every_sentence(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot annotate every sentence"):
            load_baseline_cases(_write(_row(evidence='"1,2,3"')))


if __name__ == "__main__":
    unittest.main()

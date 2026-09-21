from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from agent_audit.audit import AuditConfig, audit_records
from agent_audit.html_report import render_audit_html
from agent_audit.io import load_score_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
DOCS_DIR = PROJECT_ROOT / "docs"
ISSUE_TEMPLATE_DIR = PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE"


def _markdown_targets(document: Path) -> list[str]:
    text = document.read_text(encoding="utf-8")
    return [target.strip().strip("<>") for target in MARKDOWN_LINK.findall(text)]


def _public_markdown_files() -> list[Path]:
    return [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README_EN.md",
        *sorted(DOCS_DIR.rglob("*.md")),
    ]


class PublicAssetTests(unittest.TestCase):
    def test_committed_html_demo_matches_current_renderer(self) -> None:
        records = load_score_records(PROJECT_ROOT / "examples" / "demo_scores.csv")
        result = audit_records(
            records,
            AuditConfig(
                score_min=0,
                score_max=10,
                data_provenance="synthetic",
            ),
        )

        expected = render_audit_html(result)
        actual = (
            PROJECT_ROOT / "docs" / "examples" / "example_audit_report.html"
        ).read_text(encoding="utf-8")

        self.assertEqual(actual, expected)

    def test_relative_markdown_links_resolve_to_existing_files(self) -> None:
        missing: list[str] = []

        for document in _public_markdown_files():
            for target in _markdown_targets(document):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_text = unquote(target.split("#", 1)[0])
                if not path_text:
                    continue
                resolved = (document.parent / path_text).resolve()
                if not resolved.exists():
                    missing.append(f"{document.relative_to(PROJECT_ROOT)} -> {target}")

        self.assertEqual(missing, [], "Broken relative Markdown links were found")

    def test_issue_template_links_point_at_existing_templates(self) -> None:
        """Public intake links are the only entry point clients see.

        The relative-link test skips absolute URLs, so a renamed template would
        otherwise break three published links without failing any gate.
        """

        referenced: list[str] = []
        missing: list[str] = []

        for document in _public_markdown_files():
            for target in _markdown_targets(document):
                query = parse_qs(urlparse(target).query)
                for template_name in query.get("template", []):
                    referenced.append(template_name)
                    if not (ISSUE_TEMPLATE_DIR / template_name).is_file():
                        missing.append(
                            f"{document.relative_to(PROJECT_ROOT)} -> {template_name}"
                        )

        self.assertEqual(missing, [], "Issue template links have no matching file")
        self.assertNotEqual(referenced, [], "No issue template link was checked")

    def test_docs_navigation_lists_every_classified_document(self) -> None:
        """Every doc must be reachable from the navigation page.

        Without this, a new file can sit in a category directory that nobody
        links to, which is how a documentation tree drifts out of order.
        """

        navigation = DOCS_DIR / "README.md"
        linked = {
            (navigation.parent / unquote(target.split("#", 1)[0])).resolve()
            for target in _markdown_targets(navigation)
            if not target.startswith(("http://", "https://", "mailto:", "#"))
        }
        classified = {
            path.resolve()
            for path in DOCS_DIR.rglob("*")
            if path.is_file() and path != navigation
        }

        unlisted = sorted(
            str(path.relative_to(PROJECT_ROOT)) for path in classified - linked
        )

        self.assertEqual(unlisted, [], "Documents are missing from docs/README.md")


if __name__ == "__main__":
    unittest.main()

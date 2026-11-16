"""README.md and docs/LEGACY_RESULTS.md: no em dash, resolvable relative links, retired sections moved out of the README."""
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FILES = [REPO_ROOT / "README.md", REPO_ROOT / "docs" / "LEGACY_RESULTS.md"]
LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_em_dash(path):
    assert "\u2014" not in path.read_text()


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_relative_links_resolve(path):
    broken = []
    for target in LINK.findall(path.read_text()):
        if re.match(r"^(https?:|mailto:|#)", target):
            continue
        rel = target.split("#")[0]
        if rel and not (path.parent / rel).exists():
            broken.append(target)
    assert not broken, broken


def test_retired_sections_are_not_in_the_readme():
    text = (REPO_ROOT / "README.md").read_text()
    for heading in ("## Key Findings", "### Baseline Accuracy", "### Attention Analysis - Transformer"):
        assert heading not in text
    assert (REPO_ROOT / "docs" / "LEGACY_RESULTS.md").read_text().startswith("# Retired pre-v2 results\n\n")


def test_notice_above_only_below_the_notice():
    """'notice above' may appear only below the notice it names.

    That notice is the "Pre-v2 Results Invalidated" heading at the top of
    docs/LEGACY_RESULTS.md, so that file's own references to it are correct.
    In any other tracked Markdown file the phrase points at nothing.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    legacy = "docs/LEGACY_RESULTS.md"
    assert legacy in tracked
    for rel in tracked:
        text = (REPO_ROOT / rel).read_text()
        hits = [m.start() for m in re.finditer("notice above", text)]
        if rel == legacy:
            notice = text.index("Pre-v2 Results Invalidated")
            assert all(h > notice for h in hits), rel
        else:
            assert not hits, rel


def test_citation_does_not_claim_an_unposted_preprint():
    """The BibTeX may call the paper an arXiv preprint only once the README stops saying the arXiv ID is pending."""
    text = (REPO_ROOT / "README.md").read_text()
    start = text.index("@misc{jamanjyothi2026partial")
    bib = text[start:]
    bib = bib[: bib.index("\n}") + 2]
    if "arXiv ID pending" in text:
        assert "arXiv preprint" not in bib, bib

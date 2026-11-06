"""README.md and docs/LEGACY_RESULTS.md: no em dash, resolvable relative links, retired sections moved out of the README."""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FILES = [REPO_ROOT / "README.md", REPO_ROOT / "docs" / "LEGACY_RESULTS.md"]
LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_em_dash(path):
    assert "—" not in path.read_text()


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

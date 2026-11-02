"""CHECKPOINTS.md sweep section: the four notes and the 13 table rows (experiments/scripts/pos_sweep_registry.py)."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pos_sweep_registry", REPO_ROOT / "experiments/scripts/pos_sweep_registry.py")
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)

# sha256 of the 13 table rows as they stood on main before the notes were added
ROWS_SHA256 = "94d7db87347ca0d73c32f3b8f0e4940a07256db02afc7391a0c62d41c43d2469"


def committed_section():
    text = (REPO_ROOT / "CHECKPOINTS.md").read_text()
    return text[text.index(reg.TITLE):]


def test_committed_section_ends_with_the_four_notes():
    lines = committed_section().rstrip("\n").splitlines()
    assert lines[-len(reg.NOTES) - 2:-len(reg.NOTES)] == ["Notes:", ""]
    assert lines[-len(reg.NOTES):] == reg.NOTES and len(reg.NOTES) == 4
    assert all("—" not in n for n in reg.NOTES)


def test_table_rows_unchanged_from_main():
    rows = [l for l in committed_section().splitlines() if l.startswith("| `results_v2/pos_sweep/")]
    assert len(rows) == 13
    assert hashlib.sha256("\n".join(rows).encode()).hexdigest() == ROWS_SHA256


@pytest.mark.skipif(len(reg.run_dirs()) != 13, reason="sweep checkpoints absent")
def test_generator_output_matches_committed_section_and_ends_with_notes():
    sec = reg.section()
    assert sec == committed_section()
    assert sec.rstrip("\n").splitlines()[-4:] == reg.NOTES

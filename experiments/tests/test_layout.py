"""Repository layout: pre-v2 files live under legacy/, the root holds only current entries, and no tracked text file contains the em dash."""
import ast
import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_ALLOWED = {
    ".gitignore", "CHECKPOINTS.md", "CITATION.cff", "LICENSE", "README.md", "requirements.txt", "train_v2.py",
    "data", "datasets_v2", "docs", "experiments", "legacy", "models", "results_v2", "scripts", "utils",
}


def tracked():
    return subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()


def test_root_holds_only_current_entries():
    top = {p.split("/")[0] for p in tracked()}
    assert top <= ROOT_ALLOWED, sorted(top - ROOT_ALLOWED)


def test_pre_v2_items_moved_under_legacy():
    files = set(tracked())
    for old in ("main.py", "datasets/study1/train.json", "results/transformer/evaluation_results.json",
                "figures/results_summary.csv", "experiments/results/head_ablation/cp3_ablation_chart.png"):
        assert old not in files and f"legacy/{old}" in files, old


def test_current_code_does_not_import_legacy():
    """Parsed import nodes, not text, so no comment, docstring or string literal can trip it."""
    for p in tracked():
        if p.endswith(".py") and not p.startswith("legacy/"):
            tree = ast.parse((REPO_ROOT / p).read_text(encoding="utf-8"), filename=p)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                assert not any(n == "legacy" or n.startswith("legacy.") for n in names), p


def test_matrix_summary_lives_in_results_v2(monkeypatch):
    assert (REPO_ROOT / "results_v2" / "matrix_summary.txt").exists()
    spec = importlib.util.spec_from_file_location("pss", REPO_ROOT / "scripts/analysis/per_seed_and_paired_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(REPO_ROOT)
    assert len(mod.matrix_summary()) == 12


def test_license_and_citation():
    lic = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in lic and "Version 2.0, January 2004" in lic
    cff = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for key in ("cff-version:", "title:", "authors:", "license: Apache-2.0", "preferred-citation:"):
        assert key in cff, key
    assert "arXiv preprint" not in cff


def test_no_em_dash_in_any_tracked_text_file():
    bad = []
    for p in tracked():
        try:
            text = (REPO_ROOT / p).read_text(encoding="utf-8")
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        if "\u2014" in text:
            bad.append(p)
    assert not bad, bad

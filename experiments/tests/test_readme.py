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


README_ROOTS = ("results_v2/", "experiments/", "scripts/", "datasets_v2/", "data/", "docs/", "legacy/", "models/", "utils/")


def test_readme_backticked_paths_exist():
    """Every repository path the README names in backticks exists, so the number-to-file table cannot point at nothing."""
    text = (REPO_ROOT / "README.md").read_text()
    missing = [t for t in re.findall("`([^` ]+)`", text)
               if t.startswith(README_ROOTS) and not any(c in t for c in "<>{}*") and not (REPO_ROOT / t).exists()]
    assert not missing, missing


def test_readme_commands_point_at_existing_scripts():
    """Every command the README gives runs a script that exists."""
    text = (REPO_ROOT / "README.md").read_text()
    for mod in re.findall("python -m (scripts[.][A-Za-z0-9_.]+)", text):
        assert (REPO_ROOT / (mod.replace(".", "/") + ".py")).exists(), mod
    for path in re.findall("python ((?:experiments/)?[A-Za-z0-9_/]+[.]py)", text):
        assert (REPO_ROOT / path).exists(), path


def test_readme_runs_no_retired_entry_point():
    text = (REPO_ROOT / "README.md").read_text()
    assert "python main.py" not in text and "main.py --mode" not in text


def test_readme_states_the_python_floor():
    text = (REPO_ROOT / "README.md").read_text()
    assert "Python 3.9 or later" in text and "Python 3.8" not in text


def test_readme_training_paragraph_matches_train_v2():
    """The README's training paragraph and TRAINING_CONFIG_V2 state the same values. The pre-rebuild README had drifted to v1 values."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    import train_v2
    cfg = train_v2.TRAINING_CONFIG_V2
    assert (cfg["learning_rate"], cfg["warmup_steps"], cfg["min_lr"], cfg["batch_size"], cfg["num_epochs"],
            cfg["early_stopping_patience"], cfg["max_input_len"], cfg["max_output_len"]) == (
        {"lstm": 0.001, "transformer": 3e-4}, 400, 1e-5, 64, 200, 50, 64, 12)
    text = (REPO_ROOT / "README.md").read_text()
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("**Training, both models.**"))
    para = []
    for line in lines[start:]:
        if not line.strip():
            break
        para.append(line)
    para = " ".join(para)  # the whole paragraph, so rewrapping it cannot break the test
    for s in ("0.001 (LSTM)", "3e-4 (Transformer)", "400-step", "to 1e-5", "batch size 64",
              "200 epochs", "patience of 50", "64 and 12 tokens"):
        assert s in para, s


def test_readme_results_match_the_result_files():
    """Every cell of the two Results tables, and the eight numbers the Abstract quotes, recomputed from the committed result files."""
    import json
    import statistics
    rv = REPO_ROOT / "results_v2"
    val = {}
    for line in (rv / "matrix_summary.txt").read_text().splitlines():
        run, acc = line.split()[:2]
        val[run] = float(acc.split("=")[1].rstrip("%"))
    test = json.loads((rv / "test_set_results.json").read_text())

    def span(xs):
        return f"{statistics.mean(xs):.1f}% [{min(xs):.1f}-{max(xs):.1f}]"

    text = (REPO_ROOT / "README.md").read_text()
    blocks = {1: text[text.index("**Study 1 ("):text.index("**Study 2 (")], 2: text[text.index("**Study 2 ("):]}
    ood_keys = {1: [f"study1/ood_ops{k}.json" for k in (4, 5, 6, 7)], 2: ["study2/ood.json"]}
    means = {}
    for model, label in (("lstm", "LSTM"), ("transformer", "Transformer")):
        for study in (1, 2):
            runs = [f"{model}_study{study}_seed{s}" for s in (0, 1, 2)]
            oods = [json.loads((rv / model / f"study{study}" / f"seed{s}" / "ood_results.json").read_text()) for s in (0, 1, 2)]
            t = [test[r]["test_accuracy"] for r in runs]
            cells = [span([val[r] for r in runs]), f"{statistics.mean(t):.2f}% ± {statistics.stdev(t):.2f}"]
            cells += [span([o[k] for o in oods]) for k in ood_keys[study]]
            row = "| " + label + " | " + " | ".join(cells) + " |"
            assert row in blocks[study], row
            means[model, study, "test"] = statistics.mean(t)
            for k in ood_keys[study]:
                means[model, study, k] = statistics.mean(o[k] for o in oods)
    abstract = text[text.index("## Abstract"):text.index("## Results")]
    quoted = [means["lstm", 1, "test"], means["lstm", 2, "test"], means["lstm", 1, "study1/ood_ops4.json"],
              means["lstm", 1, "study1/ood_ops7.json"], means["lstm", 2, "study2/ood.json"],
              means["transformer", 1, "test"], means["transformer", 2, "test"],
              max(v for (m, s, k), v in means.items() if m == "transformer" and k != "test")]
    for x in quoted:
        assert f" {x:.1f}%" in abstract, x

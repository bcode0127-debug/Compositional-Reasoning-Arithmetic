"""Tests for the sinusoidal hardware-control table (experiments/scripts/pos_sweep_hardware_control.py)."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hc = _load("pos_sweep_hardware_control", "experiments/scripts/pos_sweep_hardware_control.py")
pb = hc.pb
N_EXPR = 150


def stats(acc, rng, n=10):
    return {"n": [float(n)] * N_EXPR, "correct": rng.binomial(n, acc, size=N_EXPR).astype(float).tolist()}


def side(acc, epoch, test, rng):
    st = stats(acc, rng)
    return {"selected_epoch": epoch, "selected_val_accuracy": 7.5, "test_accuracy": test,
            "layer3_position": sum(st["correct"]) / sum(st["n"]), "layer3_stats": st}


def synth_doc(rng=None):
    rng = rng or np.random.default_rng(0)
    envs = [{"name": n, "gpu": "x", "environment": "y"} for n in
            ("paper checkpoints", "probe grid", "arms and NoPE", "sinusoidal retrain")]
    doc = {"header": hc.HEADER, "not_used_by_any_rule": True, "environments": envs, "seeds": {}}
    for k in (0, 1, 2):
        paper, retrain = side(0.19, 114, 7.9, rng), side(0.21, 120, 8.4, rng)
        learned = stats(0.23, rng)
        doc["seeds"][str(k)] = {
            "paper": paper, "retrain": retrain, "hardware_delta": hc.hardware_delta(paper, retrain),
            "descriptive_learned_minus_retrain": hc.learned_minus_retrain(
                k, learned, retrain["layer3_stats"], sum(learned["correct"]) / sum(learned["n"]),
                retrain["layer3_position"]),
            "not_used_by_any_rule": True}
    return doc


def test_hardware_delta_is_retrain_minus_paper():
    rng = np.random.default_rng(1)
    paper, retrain = side(0.19, 160, 7.7, rng), side(0.25, 150, 9.1, rng)
    d = hc.hardware_delta(paper, retrain)
    assert d["test_accuracy_points"] == pytest.approx(9.1 - 7.7)
    assert d["layer3_position"] == pytest.approx(retrain["layer3_position"] - paper["layer3_position"])
    assert d["selected_epoch"] == -10 and d["not_used_by_any_rule"] is True


def test_learned_minus_retrain_is_seed_matched_paired_bootstrap():
    rng = np.random.default_rng(2)
    learned, retrain = stats(0.30, rng), stats(0.20, rng)
    la, ra = sum(learned["correct"]) / sum(learned["n"]), sum(retrain["correct"]) / sum(retrain["n"])
    c = hc.learned_minus_retrain(1, learned, retrain, la, ra)
    assert c["diff"] == pytest.approx(la - ra) and c["boot_seed"] == pb.npr.BOOT_SEED + 1
    assert c["ci"][0] < c["diff"] < c["ci"][1] and c["ci"][0] > 0
    sel = pb.boot_sel(pb.npr.BOOT_SEED + 1, N_EXPR)
    manual = pb.interval(pb.acc_boot(pb.as_stats(learned), sel) - pb.acc_boot(pb.as_stats(retrain), sel))
    assert c["ci"] == manual
    assert hc.learned_minus_retrain(2, learned, retrain, la, ra)["ci"] != c["ci"]  # a different seed draws differently


def test_environments_four_rows_and_consistency(tmp_path):
    for k in (0, 1, 2):
        d = tmp_path / f"seed{k}"
        d.mkdir()
        (d / "environment.json").write_text(json.dumps({
            "gpu": "NVIDIA RTX 4000 Ada Generation", "driver_version": "595.91.07", "torch": "2.8.0+cu128",
            "numpy": "2.5.2", "python": "3.12.3", "pod_image": "img", "git_commit": "abc"}))
    envs = hc.environments(tmp_path)
    assert [e["name"] for e in envs] == ["paper checkpoints", "probe grid", "arms and NoPE", "sinusoidal retrain"]
    assert "unrecorded" in envs[0]["gpu"] and "L4" in envs[1]["gpu"] and "scikit-learn 1.9.1" in envs[1]["environment"]
    assert envs[2]["driver_version"] == "595.91.07" and envs[3]["git_commit"] == "abc"
    bad = json.loads((tmp_path / "seed1" / "environment.json").read_text())
    bad["git_commit"] = "def"
    (tmp_path / "seed1" / "environment.json").write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        hc.environments(tmp_path)


def test_validate_accepts_and_rejects():
    doc = synth_doc()
    hc.validate(doc)
    for mutate in (lambda d: d.update(not_used_by_any_rule=False),
                   lambda d: d["seeds"]["0"].pop("hardware_delta"),
                   lambda d: d["seeds"]["1"].update(not_used_by_any_rule=False),
                   lambda d: d["seeds"]["2"]["paper"]["layer3_stats"]["n"].pop(),
                   lambda d: d["environments"].pop()):
        bad = copy.deepcopy(doc)
        mutate(bad)
        with pytest.raises((AssertionError, KeyError)):
            hc.validate(bad)


def test_markdown_labels_every_row():
    md = hc.md_report(synth_doc())
    rows = [l for l in md.splitlines() if l.startswith("| ") and not l.startswith("| Seed") and not l.startswith("| Checkpoints")]
    seed_rows = [l for l in rows if l.split("|")[1].strip() in ("0", "1", "2")]
    assert len(seed_rows) == 6 and all(hc.NOT_USED in l for l in seed_rows)
    assert hc.NOT_USED in md.splitlines()[2]
    env_rows = [l for l in md.split("## Environments")[1].splitlines() if l.startswith("| ") and "---" not in l][1:]
    assert len(env_rows) == 4


OUT = hc.OUT


@pytest.mark.skipif(not (OUT / "hardware_control.json").exists(), reason="hardware-control outputs not generated yet")
def test_committed_outputs_schema():
    doc = json.loads((OUT / "hardware_control.json").read_text())
    hc.validate(doc)
    assert set(doc["seeds"]) == {"0", "1", "2"}
    md = (OUT / "hardware_control.md").read_text()
    assert hc.NOT_USED in md
    assert doc["environments"][3]["git_commit"].startswith("2030fd5")
    assert [doc["seeds"][k]["paper"]["selected_epoch"] for k in "012"] == [114, 160, 160]


@pytest.mark.skipif(not (hc.RETRAIN_ROOT / "seed0" / "best_model.pt").exists()
                    or not (hc.RETRAIN_ROOT / "seed2" / "best_model.pt").exists(),
                    reason="retrained checkpoints absent")
def test_end_to_end_one_seed(tmp_path):
    out = tmp_path / "hc"
    r = subprocess.run([sys.executable, "-u", str(REPO_ROOT / "experiments/scripts/pos_sweep_hardware_control.py"),
                        "--out-dir", str(out), "--seeds", "0"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    doc = json.loads((out / "hardware_control.json").read_text())
    hc.validate(doc)
    assert set(doc["seeds"]) == {"0"} and (out / "hardware_control.md").exists()

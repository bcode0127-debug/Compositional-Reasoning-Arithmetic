"""Tests for the positional-encoding sweep probe battery (experiments/scripts/pos_sweep_battery.py).

Unit tests use synthetic per-expression statistics; the integration tests use the paper's Hub checkpoint and the
sweep's local run directories (skipped when the run directories are absent).
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load("pos_sweep_battery", "experiments/scripts/pos_sweep_battery.py")
npr = pb.npr
N_EXPR = 150


# ---------------------------------------------------------------------------
# Loader and probe (STEP 1, STEP 2)
# ---------------------------------------------------------------------------

def test_load_model_run_dir_matches_hub(tmp_path):
    """The paper's transformer study1 seed 0, loaded from the Hub and from a local run dir holding the same file,
    gives identical weights; the run_dir path skips FINGERPRINTS and requires dataset_version."""
    npr.configure_device("cpu")
    hub = npr.hf_file(npr.MODEL_REPO, "transformer/study1/seed0/best_model.pt", "model")
    ckpt = torch.load(hub, map_location="cpu", weights_only=False)
    d = tmp_path / "run"
    d.mkdir()
    torch.save(ckpt, d / "best_model.pt")
    m_hub = npr.load_model("transformer", "study1", 0)
    m_dir = npr.load_model("transformer", "study1", 0, run_dir=d)
    sd_h, sd_d = m_hub.state_dict(), m_dir.state_dict()
    assert set(sd_h) == set(sd_d) and all(torch.equal(sd_h[k], sd_d[k]) for k in sd_h)

    other = copy.copy(ckpt)
    other["val_accuracy"] = 1.0  # would trip FINGERPRINTS on the Hub path
    d2 = tmp_path / "run2"
    d2.mkdir()
    torch.save(other, d2 / "best_model.pt")
    npr.load_model("transformer", "study1", 0, run_dir=d2)

    bad = {k: v for k, v in ckpt.items() if k != "dataset_version"}
    d3 = tmp_path / "run3"
    d3.mkdir()
    torch.save(bad, d3 / "best_model.pt")
    with pytest.raises(AssertionError):
        npr.load_model("transformer", "study1", 0, run_dir=d3)


@pytest.fixture(scope="module")
def paper_seed0_feats():
    npr.configure_device("cpu")
    torch.use_deterministic_algorithms(False)
    sitems, _ = npr.stratified_split_items()
    model = npr.load_model("transformer", "study1", 0)
    feats = npr.layerwise_acts(model, sitems, features=("layer3",), with_ids=True)["layer3"]
    return sitems, feats


def test_position_probe_equals_identity_probe(paper_seed0_feats):
    _, feats = paper_seed0_feats
    a = npr.identity_probe(feats["train"], feats[npr.STRAT_SPLIT], "position")
    b = npr.position_probe(feats["train"], feats[npr.STRAT_SPLIT], "position")
    st = b.pop("stats")
    assert a == b
    assert st["correct"].sum() / st["n"].sum() == pytest.approx(a["accuracy"], abs=1e-12)
    assert len(st["n"]) == N_EXPR and st["n"].sum() == len(feats[npr.STRAT_SPLIT]["position"])


# ---------------------------------------------------------------------------
# Anchor (STEP 3)
# ---------------------------------------------------------------------------

def test_anchor_assertion_fails_loudly(monkeypatch):
    out = {k: {"accuracy": v} for k, v in pb.EXPECTED_BASELINES.items()}
    pb.assert_expected_baselines(out)
    out["B2"]["accuracy"] = 0.1894 + 0.0003
    with pytest.raises(AssertionError, match="B2"):
        pb.assert_expected_baselines(out)


def test_baseline_rows_align_with_activations(paper_seed0_feats):
    sitems, feats = paper_seed0_feats
    for kind in ("B1", "B2", "B3"):
        b = pb.diag.baseline_features(sitems[npr.STRAT_SPLIT], kind)
        assert np.array_equal(b["position"], feats[npr.STRAT_SPLIT]["position"])


def test_anchor_is_highest_baseline_mechanically(paper_seed0_feats):
    sitems, _ = paper_seed0_feats
    anchor, _ = pb.compute_anchor(sitems, "skip")
    accs = {k: v["accuracy"] for k, v in anchor["baselines"].items()}
    assert anchor["anchor_name"] == max(accs, key=accs.get) and anchor["anchor_accuracy"] == max(accs.values())
    pb.validate_anchor(anchor)
    print("\nanchor (this environment):", {k: round(v, 4) for k, v in accs.items()}, anchor["anchor_name"])


# ---------------------------------------------------------------------------
# Rules on synthetic statistics
# ---------------------------------------------------------------------------

def synth_row(acc, anchor_stats, seed, rng, n_pos=10):
    """A position row whose per-expression correctness has mean ~acc; CIs are computed as the battery computes them."""
    correct = rng.binomial(n_pos, acc, size=N_EXPR).astype(float)
    st = {"n": np.full(N_EXPR, float(n_pos)), "correct": correct}
    sel = pb.boot_sel(npr.BOOT_SEED + seed, N_EXPR)
    a = float(correct.sum() / st["n"].sum())
    va, vp = pb.row_vs_references(a, st, sel, float(anchor_stats["correct"].sum() / anchor_stats["n"].sum()), anchor_stats)
    return {"accuracy": a, "majority": 0.0525, "n_classes_train": 23, "convergence_warning": False, "dim": 256,
            "stats": pb.stats_json(st), "vs_anchor": va, "vs_prereg": vp}


def synth_arms(accs, anchor_acc=0.19, seed=7):
    """accs: {arm: {layer: accuracy}}; same accuracy for every seed."""
    rng = np.random.default_rng(seed)
    anchor_stats = {"n": np.full(N_EXPR, 10.0), "correct": rng.binomial(10, anchor_acc, size=N_EXPR).astype(float)}
    arms = {}
    for arm, layers in accs.items():
        arms[arm] = {"arm": arm, "seeds": {}}
        for k in pb.SEEDS:
            pos = {f: synth_row(layers.get(f, layers["layer3"]), anchor_stats, k, rng) for f in pb.FEATURES}
            arms[arm]["seeds"][str(k)] = {"position": pos}
    anchor = {"anchor_name": "B2", "anchor_accuracy": float(anchor_stats["correct"].sum() / anchor_stats["n"].sum()),
              "baselines": {"B2": {"stats": pb.stats_json(anchor_stats)}}}
    return arms, anchor


def test_injection_and_survives_use_anchor_lower_bound():
    arms, _ = synth_arms({"rope": {"layer1": 0.55, "layer3": 0.55}, "alibi": {"layer1": 0.21, "layer3": 0.21}})
    assert pb.rule_injection(arms, "rope", "anchor")["injected"] is True
    assert pb.rule_injection(arms, "alibi", "anchor")["injected"] is False
    # 0.30 clears the pre-registered 0.053 + 0.10 but not ANCHOR + 0.10
    arms, _ = synth_arms({"rope": {"layer3": 0.30}})
    assert pb.rule_survives(arms, "rope", "prereg")["survives"] is True
    assert pb.rule_survives(arms, "rope", "anchor")["survives"] is False


def test_p2_and_trigger_min_max_inside_resample():
    arms, _ = synth_arms({"sinusoidal": {"layer3": 0.19}, "learned": {"layer3": 0.20},
                          "rope": {"layer3": 0.60}, "alibi": {"layer3": 0.55}})
    r = pb.rule_p2_trigger(arms, ["rope", "alibi"])
    assert r["evaluated"] and r["p2_holds"] and r["trigger_fires"]
    # a re-injecting arm no better than learned: P2 does not hold
    arms, _ = synth_arms({"sinusoidal": {"layer3": 0.19}, "learned": {"layer3": 0.30},
                          "rope": {"layer3": 0.60}, "alibi": {"layer3": 0.28}})
    r = pb.rule_p2_trigger(arms, ["rope", "alibi"])
    assert r["p2_holds"] is False and r["trigger_fires"] is False
    # the statistic equals the manual min/max inside one resample
    k = 1
    names = ["rope", "alibi", "sinusoidal", "learned"]
    sel = pb.boot_sel(npr.BOOT_SEED + k, N_EXPR)
    acc = {a: pb.acc_boot(pb.seed_stats(arms, a, k), sel) for a in names}
    manual = pb.interval(np.minimum(acc["rope"], acc["alibi"]) - np.maximum(acc["sinusoidal"], acc["learned"]))
    assert pb.cross_arm_stat(arms, ["rope", "alibi"], k) == manual
    # nothing injected: not evaluated
    assert pb.rule_p2_trigger(arms, [])["evaluated"] is False


def test_p1_includes_zero_on_two_of_three():
    arms, _ = synth_arms({"sinusoidal": {"layer3": 0.19}, "learned": {"layer3": 0.19}})
    assert pb.rule_p1(arms)["holds"] is True
    arms, _ = synth_arms({"sinusoidal": {"layer3": 0.19}, "learned": {"layer3": 0.45}})
    assert pb.rule_p1(arms)["holds"] is False


def test_id_gap_direction_and_pairing(tmp_path):
    rng = np.random.default_rng(3)
    for k in pb.SEEDS:
        for name, p in (("arm", 0.30), ("sin", 0.07)):
            flags = rng.binomial(1, p, size=1000).tolist()
            d = (tmp_path / "armrun" / f"seed{k}") if name == "arm" else (tmp_path / "sinusoidal_reeval" / f"seed{k}")
            d.mkdir(parents=True)
            (d / "test_results.json").write_text(json.dumps({"n": 1000, "per_example_correct": flags}))
    orig = pb.run_dir_for
    pb.run_dir_for = lambda arm, k: tmp_path / "armrun" / f"seed{k}"
    try:
        r = pb.rule_id_gap(tmp_path, "rope")
        assert r["gap_moves"] is True and r["direction"] == "up"
        short = tmp_path / "sinusoidal_reeval" / "seed0" / "test_results.json"
        short.write_text(json.dumps({"n": 999, "per_example_correct": [0] * 999}))
        with pytest.raises(pb.NotEvaluable):
            pb.rule_id_gap(tmp_path, "rope")
    finally:
        pb.run_dir_for = orig


def test_kill_gate_cases():
    assert "architectural" in pb.kill_gate({a: False for a in pb.ARMS}, {})["verdict"]
    assert "does not move" in pb.kill_gate({"rope": True, "alibi": False}, {"rope": False})["verdict"]
    assert "ID gap moves" in pb.kill_gate({"rope": True}, {"rope": True})["verdict"]
    assert "reference arm" in pb.kill_gate({"sinusoidal": True}, {})["verdict"]


def test_missing_data_is_reported_not_evaluable():
    arms, anchor = synth_arms({"sinusoidal": {"layer3": 0.19}})
    floor = json.loads(pb.FLOOR_DIAG.read_text())
    rules = pb.evaluate_rules(arms, anchor, floor, Path("/nonexistent"))
    assert rules["floor_a"]["evaluable"] is False
    assert rules["p1_d"]["evaluable"] is False
    assert rules["kill_gate_i"]["anchor"]["evaluable"] is False
    pb.validate_rules(rules)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def test_schema_validators_accept_synthetic_and_reject_broken():
    arms, anchor = synth_arms({"learned": {"layer3": 0.2}})
    doc = arms["learned"]
    doc.update({"positional_encoding": "learned", "split": "stratified_test", "features": list(pb.FEATURES)})
    for s in doc["seeds"].values():
        s["checkpoint"] = {"source": "local", "sha256": "0" * 64}
    pb.validate_arm_doc(doc)
    broken = copy.deepcopy(doc)
    del broken["seeds"]["0"]["position"]["layer3"]["vs_anchor"]
    with pytest.raises(AssertionError):
        pb.validate_arm_doc(broken)
    broken = copy.deepcopy(doc)
    broken["seeds"]["1"]["position"]["layer1"]["stats"]["n"] = broken["seeds"]["1"]["position"]["layer1"]["stats"]["n"][:-1]
    with pytest.raises(AssertionError):
        pb.validate_arm_doc(broken)


OUT = pb.DEFAULT_OUT


@pytest.mark.skipif(not (OUT / "decision_rules.json").exists(), reason="battery outputs not generated yet")
def test_committed_outputs_schema():
    for arm in pb.ARMS:
        pb.validate_arm_doc(json.loads((OUT / f"{arm}.json").read_text()))
    nope = json.loads((OUT / "nope.json").read_text())
    pb.validate_arm_doc(nope)
    pb.validate_anchor(json.loads((OUT / "anchor.json").read_text()))
    pb.validate_rules(json.loads((OUT / "decision_rules.json").read_text()))
    env = json.loads((OUT / "environment.json").read_text())
    pb.validate_environment(env)
    summary = (OUT / "summary.md").read_text()
    assert summary.splitlines()[0].startswith("Design:") and "9365283" in summary.splitlines()[0]
    assert "2026-10-28" in summary.splitlines()[1]


# ---------------------------------------------------------------------------
# End-to-end on one local checkpoint (no depth), under a tmp output directory
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (REPO_ROOT / "results_v2/pos_sweep/learned/study1/seed0/best_model.pt").exists(),
                    reason="sweep run directories absent")
def test_probe_stage_and_rules_end_to_end(tmp_path):
    out = tmp_path / "out"
    cmd = [sys.executable, "-u", str(REPO_ROOT / "experiments/scripts/pos_sweep_battery.py"), "--stage", "probes",
           "--out-dir", str(out), "--arms", "learned", "--seeds", "0", "--no-depth", "--no-nope",
           "--anchor-check", "skip"]
    import subprocess
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    doc = json.loads((out / "learned.json").read_text())
    pb.validate_arm_doc(doc)
    assert doc["positional_encoding"] == "learned" and doc["seeds"]["0"]["checkpoint"]["source"] == "local"
    pb.validate_anchor(json.loads((out / "anchor.json").read_text()))
    pb.validate_environment(json.loads((out / "environment.json").read_text()))
    rules_cmd = cmd[:4] + ["rules", "--out-dir", str(out)]
    r = subprocess.run(rules_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    pb.validate_rules(json.loads((out / "decision_rules.json").read_text()))
    assert "not evaluable" in (out / "summary.md").read_text()
    # the committed output directory is never reachable with the anchor check skipped
    bad = subprocess.run([sys.executable, str(REPO_ROOT / "experiments/scripts/pos_sweep_battery.py"), "--stage", "rules",
                          "--anchor-check", "skip"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert bad.returncode != 0

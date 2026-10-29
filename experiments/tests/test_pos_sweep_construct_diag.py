"""Tests for the construct diagnostic (experiments/scripts/pos_sweep_construct_diag.py).

The machinery tests use cases whose answer is known by construction (NoPE is reversal-equivariant, the sinusoidal
encoder is not); the hypotheses themselves (ALiBi, RoPE) are measured by the script, not asserted here.
"""
import importlib.util
import json
import subprocess
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


cd = _load("pos_sweep_construct_diag", "experiments/scripts/pos_sweep_construct_diag.py")
pb, npr = cd.pb, cd.npr
HAVE_RUNS = (REPO_ROOT / "results_v2/pos_sweep/nope/study1/seed0/best_model.pt").exists()


@pytest.fixture(scope="module")
def sitems():
    npr.configure_device("cpu")
    torch.use_deterministic_algorithms(False)
    return npr.stratified_split_items()[0]


# ---------------------------------------------------------------------------
# Pairs and features
# ---------------------------------------------------------------------------

def test_pairs_properties_and_determinism(sitems):
    pairs = cd.make_pairs(sitems)
    again = cd.make_pairs(sitems)
    for split in ("train", npr.STRAT_SPLIT):
        ii, jj, off = pairs[split]
        assert all(np.array_equal(a, b) for a, b in zip(pairs[split], again[split]))
        assert len(off) == len(sitems[split]) + 1 and off[-1] == len(ii) == len(jj)
        d = np.abs(jj - ii)
        assert d.min() >= 1 and d.max() <= cd.MAX_OFFSET
        counts = np.diff(off)
        assert counts.max() <= cd.MAX_PAIRS and counts.min() >= 1
        lengths = np.array([len(it["input"]) for it in sitems[split]])
        e = np.repeat(np.arange(len(lengths)), counts)
        assert (ii >= 0).all() and (ii < lengths[e]).all() and (jj >= 0).all() and (jj < lengths[e]).all()
    t = cd.pair_targets(*pairs[npr.STRAT_SPLIT][:2])
    assert set(t["magnitude"]) <= set(range(1, 9)) and set(t["sign"]) == {0, 1}


def test_pair_feature_rows_alignment():
    # two expressions of length 3 and 2: X rows are labelled by (expression, position)
    X = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], dtype=float)
    row_off = np.array([0, 3, 5])
    pairs = (np.array([0, 2, 1]), np.array([2, 0, 0]), np.array([0, 2, 3]))  # expr0: (0,2) (2,0); expr1: (1,0)
    F = cd.pair_feature_rows(X, row_off, pairs)
    assert F.tolist() == [[0, 0, 0, 2], [0, 2, 0, 0], [1, 1, 1, 0]]


def test_baseline_pair_features(sitems):
    pairs = cd.make_pairs(sitems)
    F = cd.baseline_pair_features(sitems, npr.STRAT_SPLIT, pairs[npr.STRAT_SPLIT])
    V, L = npr.tokenizer.vocab_size, npr.MAX_INPUT_LEN
    assert F.shape == (pairs[npr.STRAT_SPLIT][2][-1], 2 * V + L + 1)
    assert (F[:, :V].sum(1) == 1).all() and (F[:, V:2 * V].sum(1) == 1).all() and (F[:, 2 * V:].sum(1) == 1).all()
    ii, jj, off = pairs[npr.STRAT_SPLIT]
    item = sitems[npr.STRAT_SPLIT][0]
    ids = npr.encode_checked(item["input"], L)
    assert F[0, ids[ii[0]]] == 1 and F[0, V + ids[jj[0]]] == 1 and F[0, 2 * V + len(item["input"])] == 1


# ---------------------------------------------------------------------------
# Equivariance machinery on cases known by construction
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_RUNS, reason="sweep run directories absent")
def test_reversal_nope_passes_sinusoidal_fails(sitems):
    items = sitems[npr.STRAT_SPLIT][:3]
    nope = cd.verdict_equiv(cd.reversal_test(cd.load_ckpt("nope", 0), items))
    assert nope["pass"] and nope["max_over_checked"] <= 1e-5
    sin = cd.verdict_equiv(cd.reversal_test(cd.load_ckpt("sinusoidal", 0), items))
    assert not sin["pass"]
    # the permutation test of the NoPE diagnostic agrees on NoPE
    worst, _ = cd.diag.equivariance(cd.load_ckpt("nope", 0), items)
    assert max(worst[k] for k in cd.CHECKED) <= 1e-5


def test_reversal_keeps_padding_fixed(sitems):
    item = sitems[npr.STRAT_SPLIT][0]
    src = npr.encode_input(item["input"])
    real = np.flatnonzero(src.numpy() != npr.pad_idx)
    src_r = src.clone()
    src_r[real] = src[torch.from_numpy(real[::-1].copy())]
    assert np.array_equal(src_r.numpy()[len(real):], src.numpy()[len(real):])
    assert src_r.numpy()[:len(real)].tolist() == src.numpy()[:len(real)][::-1].tolist()


# ---------------------------------------------------------------------------
# Readings and predictions on synthetic statistics
# ---------------------------------------------------------------------------

def synth_stats(acc, rng, n_expr=150, n_pairs=20):
    return {"n": [float(n_pairs)] * n_expr, "correct": rng.binomial(n_pairs, acc, size=n_expr).astype(float).tolist()}


def test_reading_requires_margin_over_the_max_reference_on_all_seeds():
    rng = np.random.default_rng(0)
    nope, base = synth_stats(0.30, rng), synth_stats(0.50, rng)  # the baseline is the higher reference
    high = {s: synth_stats(0.80, rng) for s in (0, 1, 2)}
    assert cd.reading(high, nope, base)["carries_relative_position"] is True
    # clears NoPE by a wide margin but not the (higher) baseline by +0.10
    mid = {s: synth_stats(0.58, rng) for s in (0, 1, 2)}
    r = cd.reading(mid, nope, base)
    assert r["carries_relative_position"] is False
    assert r["per_seed"]["0"]["ci_vs_nope"][0] > 0.10 and r["per_seed"]["0"]["ci_vs_max_reference"][0] < 0.10
    # one seed below the margin is enough to fail
    mixed = {0: synth_stats(0.80, rng), 1: synth_stats(0.80, rng), 2: synth_stats(0.55, rng)}
    assert cd.reading(mixed, nope, base)["carries_relative_position"] is False


def test_predictions_marks():
    def readings(rope_mag, rope_sign, alibi_mag, alibi_sign):
        vals = {"rope": {"magnitude": rope_mag, "sign": rope_sign}, "alibi": {"magnitude": alibi_mag, "sign": alibi_sign}}
        return {"readings": {arm: {l: {t: {"carries_relative_position": vals[arm][t]} for t in cd.TARGETS}
                                    for l in cd.H2_LAYERS} for arm in vals}}
    h1, h3 = {"held": True}, {"held": False}
    p = cd.predictions(h1, h3, readings(True, True, True, False))
    assert p["H2: RoPE carries magnitude and sign (every layer 1-3)"] is True
    assert p["H2: ALiBi carries magnitude and its sign stays near the references (every layer 1-3)"] is True
    assert p["Construct reading fails (RoPE and ALiBi clear no reference at any layer or target)"] is False
    assert p["H3: RoPE fails both reversal and permutation by orders of magnitude (all seeds)"] is False
    p = cd.predictions(h1, h3, readings(False, False, False, False))
    assert p["Construct reading fails (RoPE and ALiBi clear no reference at any layer or target)"] is True
    p = cd.predictions(h1, h3, readings(True, True, True, True))
    assert p["H2: ALiBi carries magnitude and its sign stays near the references (every layer 1-3)"] is False


def test_verdict_equiv_thresholds():
    ok = {k: 1e-7 for k in cd.EQUIV_FEATURES}
    assert cd.verdict_equiv(ok)["pass"] and not cd.verdict_equiv(ok)["fails_by_orders_of_magnitude"]
    mid = dict(ok, layer2=5e-5)  # fails the tolerance, but by less than two orders of magnitude
    v = cd.verdict_equiv(mid)
    assert not v["pass"] and not v["fails_by_orders_of_magnitude"]
    big = dict(ok, layer3=2e-1)
    assert cd.verdict_equiv(big)["fails_by_orders_of_magnitude"]
    # encoder_input is excluded from the rule
    assert cd.verdict_equiv(dict(ok, encoder_input=1.0))["pass"]


# ---------------------------------------------------------------------------
# End to end on one ALiBi checkpoint, under a tmp output directory, and schemas
# ---------------------------------------------------------------------------

OUT = cd.OUT


@pytest.mark.skipif(not (OUT / "construct_diag.md").exists(), reason="diagnostic outputs not generated yet")
def test_committed_outputs_schema():
    cd.validate_h1(json.loads((OUT / "h1_alibi_reversal.json").read_text()))
    cd.validate_h3(json.loads((OUT / "h3_rope.json").read_text()))
    cd.validate_h2(json.loads((OUT / "h2_relative_offset.json").read_text()))
    pb.validate_environment(json.loads((OUT / "environment.json").read_text()))
    md = (OUT / "construct_diag.md").read_text()
    assert "diagnostic" in md.splitlines()[0] and "unchanged" in md.splitlines()[0]
    assert "smoke run" in md.splitlines()[2]


@pytest.mark.skipif(not (REPO_ROOT / "results_v2/pos_sweep/alibi/study1/seed0/best_model.pt").exists(),
                    reason="sweep run directories absent")
def test_end_to_end_one_seed(tmp_path):
    out = tmp_path / "cd"
    r = subprocess.run([sys.executable, "-u", str(REPO_ROOT / "experiments/scripts/pos_sweep_construct_diag.py"),
                        "--parts", "all", "--out-dir", str(out), "--arms", "alibi", "--seeds", "0"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    cd.validate_h1(json.loads((out / "h1_alibi_reversal.json").read_text()))
    cd.validate_h3(json.loads((out / "h3_rope.json").read_text()))
    h2 = json.loads((out / "h2_relative_offset.json").read_text())
    cd.validate_h2(h2)
    assert set(h2["checkpoints"]) == {"alibi_seed0", "nope_seed0"} and set(h2["readings"]) == {"alibi"}
    pb.validate_environment(json.loads((out / "environment.json").read_text()))
    assert "Pre-stated predictions" in (out / "construct_diag.md").read_text()
    print("\n", (out / "construct_diag.md").read_text()[-1500:])

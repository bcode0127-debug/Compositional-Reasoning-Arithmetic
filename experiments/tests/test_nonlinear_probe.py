"""Tests for experiments/scripts/nonlinear_probe.py (Study 1 seed 0 only).

Run: pytest -s experiments/tests/test_nonlinear_probe.py
Activations are cached in /tmp/nonlinear_probe_test_cache.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "nonlinear_probe", REPO_ROOT / "experiments" / "scripts" / "nonlinear_probe.py")
npr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(npr)

CACHE = Path("/tmp/nonlinear_probe_test_cache")
PHASE3 = REPO_ROOT / "experiments" / "results" / "analysis_v2"
TOL = 0.02
SPLITS = npr.eval_split_names("study1")

npr.configure_device("auto")


@pytest.fixture(scope="session")
def models():
    return {"lstm": npr.load_model("lstm", "study1", 0),
            "transformer": npr.load_model("transformer", "study1", 0)}


@pytest.fixture(scope="session")
def acts(models):
    out = {}
    for model_name, state in (("lstm", "h"), ("lstm", "c"), ("transformer", "pooled"),
                              ("transformer", "pooled_unmasked")):
        out[(model_name, state)] = npr.get_activations(model_name, "study1", 0, state, models[model_name], CACHE)
    return out


@pytest.fixture(scope="session")
def lstm_h_depth_block(acts):
    return npr.build_output_block("lstm", "study1", 0, "depth", acts[("lstm", "h")])


def test_shapes(acts):
    items = npr.load_split_items("study1")
    dims = {("lstm", "h"): 512, ("lstm", "c"): 512, ("transformer", "pooled"): 256,
            ("transformer", "pooled_unmasked"): 256}
    for key, d in acts.items():
        for split, a in d.items():
            lens = [len(it["input"]) for it in items[split]]
            assert a["X"].shape == (sum(lens), dims[key]), (key, split, a["X"].shape)
            assert np.diff(a["offsets"]).tolist() == lens
            assert len(a["depth"]) == len(a["value"]) == sum(lens)
    assert len(acts[("lstm", "h")]["ID_val_heldout"]["depth"]) == 2945


def test_extraction_matches_model(models):
    """LSTM manual unroll h == nn.LSTM outputs; masked Transformer extraction ==
    the model's own encoder layers run with src_key_padding_mask, at real positions."""
    expr = npr.load_split_items("study1")["train"][0]["input"]
    src = npr.encode_input(expr)
    lstm = models["lstm"]
    with torch.no_grad():
        emb = lstm.encoder.dropout(lstm.encoder.embedding(src.unsqueeze(0).to(npr.DEVICE)))
        h_f, _, h_b, _ = npr.manual_lstm_unroll(lstm.encoder.lstm, emb)
    h_manual = torch.cat([h_f, h_b], -1)[0].cpu().numpy()
    assert np.abs(h_manual - npr.lstm_encoder_states(lstm, src)).max() < 1e-5

    tr = models["transformer"]
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(npr.DEVICE)
        mask = src_in == tr.pad_idx
        x = tr.pos_encoder(tr.encoder_embedding(src_in) * (tr.d_model ** 0.5))
        outs = []
        for layer in tr.transformer.encoder.layers:
            x = layer(x, src_key_padding_mask=mask)
            outs.append(x)
        ref = torch.stack(outs).mean(0)[0].cpu().numpy()
    got = npr.transformer_encoder_states_mean_pooled_masked(tr, src)
    n = len(expr)
    diff = np.abs(got[:n] - ref[:n]).max()
    print(f"\nmasked extraction vs model encoder layers, max|diff| over real positions: {diff:.2e}")
    assert diff < 1e-4
    unmasked = npr.transformer_encoder_states_mean_pooled(tr, src)
    print(f"masked vs unmasked extraction, max|diff| over real positions: {np.abs(got[:n] - unmasked[:n]).max():.3f}")


def test_majority_baseline(acts):
    a = acts[("lstm", "h")]
    expected = {"ID_val_heldout": 0.729, "ops4": 0.527, "ops5": 0.471, "ops6": 0.594, "ops7": 0.683}
    for s, v in expected.items():
        got = npr.majority_baseline(a[s]["depth"])
        print(f"\nmajority {s}: {got:.4f} (expected {v})")
        assert abs(got - v) < 5e-4


def test_linear_reproduction(acts):
    stored = json.loads((PHASE3 / "phase3_probe_results.json").read_text())
    stored_c = json.loads((PHASE3 / "phase3_addendum_celltstate_results.json").read_text())
    refs = {("lstm", "h"): stored["lstm"], ("lstm", "c"): stored_c,
            ("transformer", "pooled_unmasked"): stored["transformer"]}
    failures = []
    print()
    for key, ref in refs.items():
        a = acts[key]
        evals = {s: a[s] for s in SPLITS}
        lin_v = npr.linear_probe("value", a["train"], evals)
        lin_d = npr.linear_probe("depth", a["train"], evals)
        for s in SPLITS:
            dv = lin_v[s] - ref[s]["value_r2"]
            dd = lin_d[s] - ref[s]["depth_acc"]
            print(f"linear {key[0]} {key[1]:16s} {s:15s} value {lin_v[s]:+.4f} (stored {ref[s]['value_r2']:+.4f}, "
                  f"delta {dv:+.5f})  depth {lin_d[s]:.4f} (stored {ref[s]['depth_acc']:.4f}, delta {dd:+.5f})")
            if s == "ID_val_heldout" and (abs(dv) > TOL or abs(dd) > TOL):
                failures.append((key, s, dv, dd))
        if "_warnings" in lin_d:
            print(f"  LogisticRegression warnings: {lin_d['_warnings']}")
    assert not failures, failures


def test_determinism(acts):
    a = acts[("lstm", "h")]
    run = lambda: npr.run_mlp_protocol("depth", a, SPLITS, npr.HIDDEN, probe_seeds=(0,), control=False)
    r1, r2 = run(), run()
    m1, m2 = r1["mlp"][0], r2["mlp"][0]
    print(f"\n[{npr.DEVICE}] run1 {m1['heldout']} epochs {m1['epochs']}\n[{npr.DEVICE}] run2 {m2['heldout']}")
    assert m1["heldout"] == m2["heldout"] and m1["heldout_240"] == m2["heldout_240"]
    assert m1["epochs"] == m2["epochs"]
    if npr.DEVICE == "cuda":
        npr.DEVICE = "cpu"
        try:
            rc = run()["mlp"][0]
        finally:
            npr.DEVICE = "cuda"
        print(f"[cpu]  {rc['heldout']} epochs {rc['epochs']}")
        for s in SPLITS:
            assert abs(rc["heldout"][s] - m1["heldout"][s]) <= 0.005, (s, rc["heldout"][s], m1["heldout"][s])


def test_schema(lstm_h_depth_block, tmp_path):
    ref_keys = set(json.loads((PHASE3 / "phase3_probe_results.json").read_text()).keys())
    split_fields = {"linear", "mlp_mean", "mlp_std", "mlp_per_seed", "mlp_240_trained", "control",
                    "control_train", "majority", "selectivity", "bootstrap_ci", "n_examples", "n_positions",
                    "verdict", "attention_mask", "study_label", "fit_status"}
    p = tmp_path / npr.out_name("lstm", "study1", 0, "h", "depth")
    p.write_text(json.dumps(lstm_h_depth_block))
    files = [p] + sorted(npr.DEFAULT_OUT.glob("*_seed[012]_*.json"))
    for f in files:
        d = json.loads(f.read_text())
        assert set(d.keys()) <= ref_keys and len(d) == 1, (f.name, d.keys())
        block = next(iter(d.values()))
        study = "study2" if "_study2_" in f.name else "study1"
        assert tuple(block.keys()) == npr.eval_split_names(study), (f.name, block.keys())
        for s, e in block.items():
            missing = split_fields - e.keys()
            assert not missing, (f.name, s, missing)
            assert set(e["bootstrap_ci"]) == {"seed0", "seed1", "seed2"}
    print(f"\nschema OK on {len(files)} file(s)")


def test_control_sanity(lstm_h_depth_block):
    print()
    for s, e in lstm_h_depth_block["lstm"].items():
        print(f"control {s}: held-out {e['control']:.4f} per seed {[round(x, 4) for x in e['control_per_seed']]} "
              f"majority {e['majority']:.4f} train {e['control_train']:.4f} | mlp {e['mlp_mean']:.4f}")
    over = {s: e for s, e in lstm_h_depth_block["lstm"].items() if max(e["control_per_seed"]) > e["majority"] + 0.02}
    assert not over, list(over)

"""Tests for experiments/scripts/nonlinear_probe.py (Study 1 seed 0 only).

Run: pytest -s experiments/tests/test_nonlinear_probe.py
Activations are cached in /tmp/nonlinear_probe_test_cache.
"""
import importlib.util
import json
from collections import Counter
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
    files = [p] + sorted(f for f in npr.DEFAULT_OUT.glob("*_seed[012]_*.json") if "bridge" not in f.name)
    bridge = npr.DEFAULT_OUT / "transformer_study1_seed0_pooled_unmasked_bridge.json"
    if bridge.exists():  # linear-only by design: check top level and splits
        b = json.loads(bridge.read_text())
        assert set(b.keys()) == {"transformer"} and tuple(b["transformer"]) == SPLITS
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


def test_input_baseline_features():
    items = npr.load_split_items("study1")["ID_val_heldout"]
    a = npr.input_baseline_features(items)
    V, L = npr.tokenizer.vocab_size, npr.MAX_INPUT_LEN
    assert a["X"].shape == (2945, V + 2 * L + 1)
    assert (a["X"].sum(1) == 3).all()  # exactly one char, one position, one length bit
    first = items[0]["input"]
    assert a["X"][0, npr.tokenizer.encode(first)[0]] == 1 and a["X"][0, V] == 1 and a["X"][0, V + L + len(first)] == 1
    assert np.array_equal(a["depth"][:len(first)], npr.compute_position_targets(first)[0])


def test_label_space_ceilings():
    c = npr.label_space_ceilings("study1")
    assert c["training_classes"] == [1, 2]
    assert abs(c["training_majority"] - 0.554) < 5e-4
    expected = {"ID_val_heldout": (1.0, 0.779), "ops4": (0.709, 0.577), "ops5": (0.529, 0.521),
                "ops6": (0.406, 0.644), "ops7": (0.317, 0.733)}
    print()
    for s, (ceil, bar) in expected.items():
        print(f"ceiling {s}: {c[s]['ceiling']:.4f} (expected {ceil}) bar {c[s]['bar']:.4f} (expected {bar})")
        assert abs(c[s]["ceiling"] - ceil) < 5e-4 and abs(c[s]["bar"] - bar) < 5e-4


def test_ordinal_baseline_features():
    items = npr.load_split_items("study1")["ID_val_heldout"]
    a = npr.ordinal_baseline_features(items)
    V = npr.tokenizer.vocab_size
    assert a["X"].shape == (2945, V + 3)
    assert (a["X"][:, :V].sum(1) == 1).all()
    first = items[0]["input"]
    n = len(first)
    assert a["X"][n - 1, V] == n - 1 and a["X"][n - 1, V + 1] == n
    assert abs(a["X"][n - 1, V + 2] - (n - 1) / n) < 1e-6


def test_stratified_split():
    items, idx = npr.stratified_split_items()
    tr, te = items["train"], items[npr.STRAT_SPLIT]
    assert len(tr) == 300 and len(te) == 150
    assert not set(idx["train_indices"]) & set(idx["test_indices"])
    ops = lambda xs: sorted(Counter(x["num_operations"] for x in xs).items())
    assert ops(tr) == [(2, 100), (3, 200)] and ops(te) == [(2, 50), (3, 100)]
    assert ops(tr[:240]) == [(2, 80), (3, 160)] and ops(tr[240:]) == [(2, 20), (3, 40)]
    assert npr.stratified_split_items()[1] == idx  # seeded: identical on repeat


def test_transformer_layerwise_matches_pooled(models):
    expr = npr.load_split_items("study1")["train"][0]["input"]
    src = npr.encode_input(expr)
    lw = npr.transformer_layerwise_masked(models["transformer"], src)
    pooled = npr.transformer_encoder_states_mean_pooled_masked(models["transformer"], src)
    assert np.array_equal(lw["mean_pool"], pooled)
    assert np.allclose((lw["layer1"] + lw["layer2"] + lw["layer3"]) / 3, pooled, atol=1e-5)
    assert lw["encoder_input"].shape == pooled.shape == (npr.MAX_INPUT_LEN, 256)


def test_char_position_features():
    items = npr.load_split_items("study1")["ID_val_heldout"]
    a = npr.char_position_features(items)
    assert a["X"].shape == (2945, npr.tokenizer.vocab_size + npr.MAX_INPUT_LEN)
    assert (a["X"].sum(1) == 2).all()


def test_transformer_memory_matches_encoder_output(models):
    """The 'memory' feature equals nn.Transformer's own encoder output (layers + final norm) at real positions."""
    tr = models["transformer"]
    expr = npr.load_split_items("study1")["train"][0]["input"]
    src = npr.encode_input(expr)
    lw = npr.transformer_layerwise_masked(tr, src)
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(npr.DEVICE)
        x = tr.pos_encoder(tr.encoder_embedding(src_in) * (tr.d_model ** 0.5))
        ref = tr.transformer.encoder(x, src_key_padding_mask=(src_in == tr.pad_idx))[0].cpu().numpy()
    n = len(expr)
    diff = np.abs(lw["memory"][:n] - ref[:n]).max()
    print(f"\nmemory vs nn.Transformer encoder output, max|diff| over real positions: {diff:.2e}")
    assert diff < 1e-4


def test_final_position_extraction_shape(models):
    """Final-position final-answer extraction (cell 32): one vector per example, taken at len(expr) - 1."""
    items = {"train": npr.load_split_items("study1")["train"][:5]}
    for (m, state), dim in ((("lstm", "h"), 512), (("lstm", "c"), 512), (("transformer", "pooled"), 256),
                            (("transformer", "pooled_unmasked"), 256)):
        fn = npr.STATE_FNS[(m, state)]
        a = npr.final_answer_acts(lambda s: fn(models[m], s), items)["train"]
        assert a["X"].shape == (5, dim) and a["offsets"].tolist() == list(range(6))
        assert a["value"].tolist() == [int(it["output"]) for it in items["train"]]
        expr = items["train"][0]["input"]
        assert np.array_equal(a["X"][0], fn(models[m], npr.encode_input(expr))[len(expr) - 1])
    b = npr.final_length_features(items)["train"]
    assert b["X"].shape == (5, npr.MAX_INPUT_LEN + 1) and (b["X"].sum(1) == 1).all()


def test_lstm_final_position_reproduction():
    """Phase 3 addendum (cell 32): LSTM seed-0 final-position Ridge R² reproduces the stored rows within ±0.02."""
    block = npr.matched_lstm_final_position(CACHE)
    print()
    for state in ("h", "c"):
        e = block[state]
        print(f"LSTM {state} final position, train R² {e['train_r2']:.3f}: " + " ".join(
            f"{s} {e[s]['rerun']:+.3f} (stored {e[s]['stored']:+.3f})" for s in SPLITS))
        assert e["within_tolerance"], (state, e["max_abs_delta"])

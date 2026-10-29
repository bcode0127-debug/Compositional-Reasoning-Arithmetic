"""Tests for the extraction path x device check (experiments/scripts/extraction_path_check.py)."""
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


ec = _load("extraction_path_check", "experiments/scripts/extraction_path_check.py")
pb, npr = ec.pb, ec.npr
STRAT = npr.STRAT_SPLIT


def test_differences_arithmetic():
    d = ec.differences(0.1810, 0.1821, 0.1803, 0.1807)
    assert d["path_at_fixed_device"] == pytest.approx(0.1803 - 0.1807)
    assert d["architecture_for_hooked"] == pytest.approx(0.1821 - 0.1803)
    assert d["device_and_environment_for_fused"] == pytest.approx(0.1810 - 0.1807)


@pytest.fixture(scope="module")
def paper0():
    npr.configure_device("cpu")
    torch.use_deterministic_algorithms(False)
    sitems, _ = npr.stratified_split_items()
    return npr.load_model("transformer", "study1", 0), sitems


def test_kernel_counters_show_which_kernel_each_path_takes(paper0):
    model, sitems = paper0
    n = 2
    ev = ec.kernel_evidence(model, sitems[STRAT][:n])
    L = ev["encoder_layers"]
    # only the per-layer call without hooks takes the layer-level fused kernel
    assert ev["fused_layer"]["_transformer_encoder_layer_fwd"] == n * L
    assert ev["hooked"]["_transformer_encoder_layer_fwd"] == 0 and ev["legacy"]["_transformer_encoder_layer_fwd"] == 0
    # the counters are removed afterwards
    assert not hasattr(torch._native_multi_head_attention, "__wrapped__")
    with ec.kernel_counters() as c:
        pass
    assert sum(c.values()) == 0


def test_extraction_paths_agree_as_documented(paper0):
    model, sitems = paper0
    small = {s: its[:5] for s, its in sitems.items()}
    acts = {p: ec.layer3_acts(model, small, p) for p in ec.PATHS}
    for s in small:
        for p in ("legacy", "fused_layer"):
            assert np.array_equal(acts[p][s]["position"], acts["hooked"][s]["position"])
            assert np.array_equal(acts[p][s]["offsets"], acts["hooked"][s]["offsets"])
    # the committed extraction equals the layer-level fused path (the existing sweep test measures 0.0 on CPU)
    assert ec.max_abs_diff(acts["legacy"], acts["fused_layer"]) <= 1e-6
    # the hooked path differs from both by rounding only
    d = ec.max_abs_diff(acts["hooked"], acts["legacy"])
    assert 0 < d < ec.ENTRY_ACTIVATION_BOUND
    assert npr.LEGACY_EXTRACTION is False  # the switch is restored


def test_validate_accepts_and_rejects():
    rng = np.random.default_rng(0)
    st = {"n": [10.0] * 150, "correct": rng.binomial(10, 0.2, 150).astype(float).tolist()}
    probes = {p: {"accuracy": 0.2, "stats": st} for p in ec.PATHS}
    ke = {p: {"_native_multi_head_attention": 9, "_transformer_encoder_layer_fwd": 0} for p in ec.PATHS}
    ke.update(n_expressions=3, encoder_layers=3)
    seed = {"probe_grid_committed": 0.1936, "battery_committed": 0.1936, "mac_hooked": 0.2, "mac_fused": 0.2,
            "mac_fused_layer": 0.2, "differences": ec.differences(0.1936, 0.1936, 0.2, 0.2),
            "activation_max_abs_diff": dict.fromkeys(ec.ACTIVATION_KEYS, 1e-5), "probes": probes, "kernel_evidence": ke, "n_rows": {"train": 1}}
    doc = {"header": ec.HEADER, "not_used_by_any_rule": True, "paths": ec.PATHS, "entry_activation_bound": 5.4e-4,
           "seeds": {"0": seed}}
    ec.validate(doc)
    for mutate in (lambda d: d.update(header="x"),
                   lambda d: d["seeds"]["0"]["differences"].update(path_at_fixed_device=0.5),
                   lambda d: d["seeds"]["0"]["probes"]["hooked"].update(accuracy=0.3),
                   lambda d: d["seeds"]["0"]["kernel_evidence"]["legacy"].pop("_native_multi_head_attention")):
        bad = copy.deepcopy(doc)
        mutate(bad)
        with pytest.raises((AssertionError, KeyError)):
            ec.validate(bad)
    assert ec.md_report(doc).splitlines()[0] == "Reconciliation check; not used by any rule."


OUT = ec.OUT


@pytest.mark.skipif(not (OUT / "extraction_check.json").exists(), reason="extraction-check outputs not generated yet")
def test_committed_outputs_schema():
    doc = json.loads((OUT / "extraction_check.json").read_text())
    ec.validate(doc)
    assert set(doc["seeds"]) == {"0", "1", "2"}
    assert (OUT / "extraction_check.md").read_text().splitlines()[0] == "Reconciliation check; not used by any rule."
    env = json.loads((OUT / "environment.json").read_text())
    pb.validate_environment(env)
    for k in ("cpu_arch", "cpu_brand", "torch_threads", "python_executable"):
        assert k in env
    # committed reference numbers are the ones read from the committed files
    assert [round(doc["seeds"][k]["probe_grid_committed"], 4) for k in "012"] == [0.1936, 0.1775, 0.1810]
    assert [round(doc["seeds"][k]["battery_committed"], 4) for k in "012"] == [0.1936, 0.1779, 0.1821]

"""Extraction path x device check for the sinusoidal layer-3 position numbers.

Reconciliation check; not used by any rule. The paper's probe grid (fused eval path, L4 GPU, committed) and the
battery (hooked path, pod CPU, committed) report layer-3 linear position accuracy for the paper's sinusoidal
Transformer checkpoints that differ by up to 0.0011. This script measures the path effect with the device held
fixed: the same checkpoints (Hub path, fingerprint check on), the same stratified split, extracted two ways on
this machine's CPU, and the battery's position probe on each.

  (i)   hooked: the extraction the battery uses (nonlinear_probe.transformer_states_hooked).
  (ii)  legacy: the extraction of the committed probe grid at 185b127, through the current code's switch
        (nonlinear_probe.LEGACY_EXTRACTION = True, transformer_layerwise_masked).
  (iii) fused layer (supplementary): each encoder layer called without hooks, which takes PyTorch's layer-level
        fused kernel (torch._transformer_encoder_layer_fwd).

Kernel evidence is measured, not assumed: the torch kernels are wrapped with call counters and each path runs the
same expressions.

  python experiments/scripts/extraction_path_check.py
"""
import argparse
import contextlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
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
STRAT = npr.STRAT_SPLIT
SEEDS = pb.SEEDS
BATTERY = REPO_ROOT / "experiments" / "results" / "analysis_v2" / "pos_sweep"
OUT = BATTERY / "extraction_check"
GRID_FILE = REPO_ROOT / "experiments/results/analysis_v2/nonlinear_probe/step4e_addenda.json"
HEADER = "Reconciliation check; not used by any rule."
KERNELS = ("_native_multi_head_attention", "_transformer_encoder_layer_fwd")
N_KERNEL_EXPRESSIONS = 3
ACTIVATION_KEYS = ("hooked_vs_legacy_all_rows", "hooked_vs_legacy_test_rows", "legacy_vs_fused_layer_all_rows",
                   "hooked_vs_fused_layer_all_rows")
ENTRY_ACTIVATION_BOUND = 5.4e-4  # Deviation 2026-10-19, "Probe extraction path"
PATHS = {
    "hooked": "nonlinear_probe.transformer_states_hooked (the battery's extraction)",
    "legacy": "nonlinear_probe.transformer_layerwise_masked via LEGACY_EXTRACTION = True (the extraction of the "
              "committed probe grid at 185b127; the current code's switch for it)",
    "fused_layer": "each encoder layer called without hooks: torch._transformer_encoder_layer_fwd",
}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def layer3_acts(model, sitems, path):
    """{split: {"X", "position", "offsets"}} for layer 3, rows in layerwise_acts order."""
    if path in ("hooked", "legacy"):
        previous = npr.LEGACY_EXTRACTION
        npr.LEGACY_EXTRACTION = (path == "legacy")
        try:
            feats = npr.layerwise_acts(model, sitems, features=("layer3",), with_ids=True)["layer3"]
        finally:
            npr.LEGACY_EXTRACTION = previous
        return {s: {k: feats[s][k] for k in ("X", "position", "offsets")} for s in feats}
    assert path == "fused_layer"
    out = {}
    for split, items in sitems.items():
        rows, pos, lengths = [], [], []
        for item in items:
            n = len(item["input"])
            rows.append(fused_layer3_state(model, npr.encode_input(item["input"]))[:n])
            pos.extend(range(n))
            lengths.append(n)
        out[split] = {"X": np.concatenate(rows), "position": np.array(pos),
                      "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)}
    return out


def fused_layer3_state(model, src):
    with torch.no_grad():
        src_in = src.unsqueeze(0)
        mask = src_in == model.pad_idx
        x = model.pos_encoder(model.encoder_embedding(src_in) * math.sqrt(model.d_model))
        for layer in model.transformer.encoder.layers:
            x = layer(x, src_key_padding_mask=mask)  # no hooks attached: the layer-level fast path
        # x is now layer 3 (before the encoder's final LayerNorm)
    return x[0].numpy()


@contextlib.contextmanager
def kernel_counters():
    counts, originals = Counter(), {}
    for name in KERNELS:
        orig = getattr(torch, name)
        originals[name] = orig

        def make(name, orig):
            def wrapped(*a, **k):
                counts[name] += 1
                return orig(*a, **k)
            return wrapped
        setattr(torch, name, make(name, orig))
    try:
        yield counts
    finally:
        for name, orig in originals.items():
            setattr(torch, name, orig)


def kernel_evidence(model, items):
    """Kernel calls per path over the same expressions (one forward per expression). The hooked path runs the full
    encoder-decoder forward; the decoder adds no calls to either kernel."""
    fns = {"hooked": lambda s: npr.transformer_states_hooked(model, s),
           "legacy": lambda s: npr.transformer_layerwise_masked(model, s),
           "fused_layer": lambda s: fused_layer3_state(model, s)}
    out = {}
    for path, fn in fns.items():
        with kernel_counters() as counts:
            for it in items:
                fn(npr.encode_input(it["input"]))
        out[path] = {k: int(counts.get(k, 0)) for k in KERNELS}
    out["n_expressions"] = len(items)
    out["encoder_layers"] = len(model.transformer.encoder.layers)
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def max_abs_diff(a, b, splits=None):
    splits = splits or list(a)
    return float(max(np.abs(a[s]["X"] - b[s]["X"]).max() for s in splits))


def differences(grid, battery, hooked, fused):
    return {"path_at_fixed_device": hooked - fused,
            "architecture_for_hooked": battery - hooked,
            "device_and_environment_for_fused": grid - fused}


def committed_numbers(seed):
    grid = json.loads(GRID_FILE.read_text())["identity"][f"seed{seed}"]["layer3"]["position"]["accuracy"]
    battery = json.loads((BATTERY / "sinusoidal.json").read_text())["seeds"][str(seed)]["position"]["layer3"]["accuracy"]
    return grid, battery


def build(seeds=SEEDS):
    npr.configure_device("cpu")
    torch.use_deterministic_algorithms(False)  # as the reproduction gate and the battery
    sitems, _ = npr.stratified_split_items()
    doc = {"header": HEADER, "not_used_by_any_rule": True, "paths": PATHS, "entry_activation_bound": ENTRY_ACTIVATION_BOUND,
           "seeds": {}}
    for seed in seeds:
        t0 = time.time()
        model = npr.load_model("transformer", "study1", seed)  # Hub path, fingerprint check on
        acts = {p: layer3_acts(model, sitems, p) for p in PATHS}
        for p in ("legacy", "fused_layer"):
            for s in sitems:
                assert np.array_equal(acts[p][s]["position"], acts["hooked"][s]["position"]), f"{p} {s}: rows misaligned"
        probes = {}
        for p in PATHS:
            r = npr.position_probe(acts[p]["train"], acts[p][STRAT], "position")
            probes[p] = {"accuracy": r["accuracy"], "convergence_warning": r["convergence_warning"],
                         "stats": pb.stats_json(r["stats"])}
        grid, battery = committed_numbers(seed)
        hooked, fused = probes["hooked"]["accuracy"], probes["legacy"]["accuracy"]
        doc["seeds"][str(seed)] = {
            "probe_grid_committed": grid, "battery_committed": battery,
            "mac_hooked": hooked, "mac_fused": fused, "mac_fused_layer": probes["fused_layer"]["accuracy"],
            "differences": differences(grid, battery, hooked, fused),
            "activation_max_abs_diff": {
                "hooked_vs_legacy_all_rows": max_abs_diff(acts["hooked"], acts["legacy"]),
                "hooked_vs_legacy_test_rows": max_abs_diff(acts["hooked"], acts["legacy"], [STRAT]),
                "legacy_vs_fused_layer_all_rows": max_abs_diff(acts["legacy"], acts["fused_layer"]),
                "hooked_vs_fused_layer_all_rows": max_abs_diff(acts["hooked"], acts["fused_layer"])},
            "n_rows": {s: int(len(acts["hooked"][s]["position"])) for s in sitems},
            "probes": probes,
            "kernel_evidence": kernel_evidence(model, sitems[STRAT][:N_KERNEL_EXPRESSIONS]),
            "checkpoint_sha256": npr.FILE_RECORDS[f"model:transformer/study1/seed{seed}/best_model.pt"]["sha256"],
            "fingerprint_checked": seed == 0,
            "not_used_by_any_rule": True}
        s = doc["seeds"][str(seed)]
        print(f"[seed{seed}] grid {grid:.4f} battery {battery:.4f} hooked {hooked:.4f} fused(legacy) {fused:.4f} "
              f"fused-layer {s['mac_fused_layer']:.4f}; act diff hooked-legacy {s['activation_max_abs_diff']['hooked_vs_legacy_all_rows']:.2e} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del model
    return doc


def md_report(doc):
    f = "{:.4f}".format
    L = [HEADER, "",
         "# Extraction path x device check, sinusoidal layer-3 linear position",
         "",
         "Descriptive only; no reading beyond the differences. Same paper checkpoints (Hub path), same stratified split, "
         "same position probe (LogisticRegression, max_iter 1000, unscaled).", "",
         "## Layer-3 position accuracy per seed", "",
         "| Seed | Probe grid (fused, L4 GPU, committed) | Battery (hooked, pod CPU, committed) | Mac hooked CPU | Mac fused CPU |",
         "|---|---|---|---|---|"]
    for k, s in doc["seeds"].items():
        L.append(f"| {k} | {f(s['probe_grid_committed'])} | {f(s['battery_committed'])} | {f(s['mac_hooked'])} | "
                 f"{f(s['mac_fused'])} |")
    L += ["", "## Differences", "",
          "| Seed | Path at fixed device (Mac hooked - Mac fused) | Architecture for hooked (battery - Mac hooked) | "
          "Device and environment for fused (probe grid - Mac fused) |", "|---|---|---|---|"]
    for k, s in doc["seeds"].items():
        d = s["differences"]
        L.append(f"| {k} | {d['path_at_fixed_device']:+.4f} | {d['architecture_for_hooked']:+.4f} | "
                 f"{d['device_and_environment_for_fused']:+.4f} |")
    L += ["", "## Layer-3 activation max abs difference between extractions (Mac CPU)", "",
          f"The Deviation entry's bound for the path difference is {doc['entry_activation_bound']:.1e}.", "",
          "| Seed | Hooked vs legacy (all rows) | Hooked vs legacy (test rows) | Legacy vs fused layer | Hooked vs fused layer |",
          "|---|---|---|---|---|"]
    for k, s in doc["seeds"].items():
        a = s["activation_max_abs_diff"]
        L.append(f"| {k} | {a['hooked_vs_legacy_all_rows']:.2e} | {a['hooked_vs_legacy_test_rows']:.2e} | "
                 f"{a['legacy_vs_fused_layer_all_rows']:.2e} | {a['hooked_vs_fused_layer_all_rows']:.2e} |")
    L += ["", "## Which fused path", "",
          f"Mac fused CPU is path (ii): {PATHS['legacy']}.", "",
          "Supplementary, not in the table above: path (iii), " + PATHS["fused_layer"] + ". Its layer-3 position "
          "accuracy per seed: " + ", ".join(f"seed {k} {f(s['mac_fused_layer'])}" for k, s in doc["seeds"].items()) + ".", "",
          "Kernel calls per path (the same expressions, counters on the torch kernels; seed 0):", "",
          "| Path | torch._native_multi_head_attention | torch._transformer_encoder_layer_fwd |", "|---|---|---|"]
    ke = doc["seeds"][next(iter(doc["seeds"]))]["kernel_evidence"]
    for p in PATHS:
        L.append(f"| {p} | {ke[p]['_native_multi_head_attention']} | {ke[p]['_transformer_encoder_layer_fwd']} |")
    L += ["", f"{ke['n_expressions']} expressions through {ke['encoder_layers']} encoder layers each, so one call per "
          "expression and layer where a kernel is used."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Schema, environment, driver
# ---------------------------------------------------------------------------

def validate(doc):
    pb._has(doc, ("header", "not_used_by_any_rule", "paths", "entry_activation_bound", "seeds"), "extraction_check")
    assert doc["header"] == HEADER and doc["not_used_by_any_rule"] is True
    for k, s in doc["seeds"].items():
        pb._has(s, ("probe_grid_committed", "battery_committed", "mac_hooked", "mac_fused", "mac_fused_layer",
                    "differences", "activation_max_abs_diff", "probes", "kernel_evidence", "n_rows"), f"seed{k}")
        d = s["differences"]
        assert abs(d["path_at_fixed_device"] - (s["mac_hooked"] - s["mac_fused"])) < 1e-12
        assert abs(d["architecture_for_hooked"] - (s["battery_committed"] - s["mac_hooked"])) < 1e-12
        assert abs(d["device_and_environment_for_fused"] - (s["probe_grid_committed"] - s["mac_fused"])) < 1e-12
        for p in PATHS:
            pb._has(s["probes"][p], ("accuracy", "stats"), f"seed{k} {p}")
            pb.validate_stats(s["probes"][p]["stats"], f"seed{k} {p}")
            assert set(s["kernel_evidence"][p]) == set(KERNELS)
        assert abs(s["probes"]["hooked"]["accuracy"] - s["mac_hooked"]) < 1e-12
        assert abs(s["probes"]["legacy"]["accuracy"] - s["mac_fused"]) < 1e-12
        assert set(s["activation_max_abs_diff"]) == set(ACTIVATION_KEYS)
        assert all(v >= 0 for v in s["activation_max_abs_diff"].values())


def environment_doc(argv):
    import sklearn
    env = pb.environment_doc("extraction_check", "none", argv)
    brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    env.update({"sklearn": sklearn.__version__, "cpu_arch": platform.machine(), "cpu_brand": brand or platform.processor(),
                "logical_cpus": __import__("os").cpu_count(), "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "python_executable": sys.executable,
                "role": "extraction path x device check (Mac CPU, not the sweep pod)"})
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()
    t0 = time.time()
    env = environment_doc(sys.argv[1:])
    pb.write_json(args.out_dir / "environment.json", env)  # written at the start of the run
    doc = build(args.seeds)
    validate(doc)
    pb.write_json(args.out_dir / "extraction_check.json", doc)
    (args.out_dir / "extraction_check.md").write_text(md_report(doc))
    env["finished_at_utc"] = pb.now_utc()
    env["wall_seconds"] = round(time.time() - t0, 1)
    pb.write_json(args.out_dir / "environment.json", env)
    print(f"wrote extraction_check.json, extraction_check.md and environment.json in {args.out_dir}; "
          f"{env['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()

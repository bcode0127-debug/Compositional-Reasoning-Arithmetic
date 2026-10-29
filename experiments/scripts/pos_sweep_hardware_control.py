"""Hardware-control table for the positional-encoding sweep (Deviation 2026-10-29, supplementary retrain).

Not used by any rule. Puts the paper's sinusoidal checkpoints (GPU and environment unrecorded) beside sinusoidal
checkpoints retrained with the design's protocol on the sweep's RTX 4000 Ada pod, per seed: selected epoch, test
accuracy, layer-3 linear position accuracy, the hardware delta (retrain minus paper), and, descriptively, the paired
bootstrap interval on learned minus retrained sinusoidal layer-3 position. Reads the battery outputs; writes only
under the output directory.

  python experiments/scripts/pos_sweep_hardware_control.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import importlib.util

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
RETRAIN_ROOT = REPO_ROOT / "results_v2" / "pos_sweep" / "sinusoidal_retrain" / "study1"
OUT = BATTERY / "hardware_control"
NOT_USED = "not used by any rule"
HEADER = (f"Hardware control for the sinusoidal arm: {NOT_USED}. The pre-registered comparison against the paper's "
          "checkpoints stays primary; nothing here is recomputed into a verdict.")


def paper_row(seed, battery=BATTERY):
    sin = json.loads((Path(battery) / "sinusoidal.json").read_text())["seeds"][str(seed)]
    re = json.loads((Path(battery) / "sinusoidal_reeval" / f"seed{seed}" / "test_results.json").read_text())
    l3 = sin["position"]["layer3"]
    return {"selected_epoch": sin["checkpoint"]["epoch"], "selected_val_accuracy": sin["checkpoint"]["val_accuracy"],
            "test_accuracy": re["test_accuracy"], "layer3_position": l3["accuracy"], "layer3_stats": l3["stats"],
            "source": "battery sinusoidal.json (layer 3) and sinusoidal_reeval test_results.json (test accuracy)"}


def retrain_row(seed, sitems, retrain_root=RETRAIN_ROOT):
    run_dir = Path(retrain_root) / f"seed{seed}"
    model = npr.load_model("transformer", "study1", seed, run_dir=run_dir)
    meta = npr.CKPT_META[str(run_dir)]
    test = json.loads((run_dir / "test_results.json").read_text())
    assert test["n"] == 1000 and test["positional_encoding"] == "sinusoidal" and test["epoch"] == meta["epoch"]
    feats = npr.layerwise_acts(model, sitems, features=("layer3",), with_ids=True)["layer3"]
    r = npr.position_probe(feats["train"], feats[STRAT], "position")
    return {"selected_epoch": meta["epoch"], "selected_val_accuracy": meta["val_accuracy"],
            "test_accuracy": test["test_accuracy"], "layer3_position": r["accuracy"],
            "layer3_stats": pb.stats_json(r["stats"]), "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "checkpoint_sha256": npr.sha256_file(run_dir / "best_model.pt"),
            "source": "run dir test_results.json (test accuracy); nonlinear_probe.position_probe on load_model(run_dir=)"}


def learned_minus_retrain(seed, learned_stats, retrain_stats, learned_acc, retrain_acc):
    """Descriptive: paired bootstrap interval, by expression (N_BOOT resamples, BOOT_SEED + seed), seed-matched."""
    a, b = pb.as_stats(learned_stats), pb.as_stats(retrain_stats)
    sel = pb.boot_sel(npr.BOOT_SEED + seed, len(a["n"]))
    return {"learned_layer3": learned_acc, "retrain_layer3": retrain_acc, "diff": learned_acc - retrain_acc,
            "ci": pb.interval(pb.acc_boot(a, sel) - pb.acc_boot(b, sel)), "n_boot": npr.N_BOOT,
            "boot_seed": npr.BOOT_SEED + seed, "not_used_by_any_rule": True}


def hardware_delta(paper, retrain):
    return {"test_accuracy_points": retrain["test_accuracy"] - paper["test_accuracy"],
            "layer3_position": retrain["layer3_position"] - paper["layer3_position"],
            "selected_epoch": retrain["selected_epoch"] - paper["selected_epoch"], "not_used_by_any_rule": True}


def environments(retrain_root=RETRAIN_ROOT):
    envs = [json.loads((Path(retrain_root) / f"seed{k}" / "environment.json").read_text()) for k in SEEDS]
    keys = ("gpu", "driver_version", "torch", "numpy", "python", "pod_image", "git_commit")
    assert all(all(e[k] == envs[0][k] for k in keys) for e in envs), "retrain runs differ in environment"
    e = envs[0]
    return [
        {"name": "paper checkpoints", "gpu": "unrecorded", "environment": "unrecorded"},
        {"name": "probe grid", "gpu": "L4", "environment": "torch 2.8.0+cu128, scikit-learn 1.9.1"},
        {"name": "arms and NoPE", "gpu": "RTX 4000 Ada", "driver_version": "595.91.07",
         "environment": "torch 2.8.0+cu128, numpy 2.5.2"},
        {"name": "sinusoidal retrain", "gpu": e["gpu"], "driver_version": e["driver_version"],
         "environment": f"torch {e['torch']}, numpy {e['numpy']}, python {e['python']}",
         "pod_image": e["pod_image"], "git_commit": e["git_commit"]},
    ]


def build(seeds=SEEDS, battery=BATTERY, retrain_root=RETRAIN_ROOT):
    npr.configure_device("cpu")
    import torch
    torch.use_deterministic_algorithms(False)  # as the reproduction gate and the battery
    sitems, _ = npr.stratified_split_items()
    learned = json.loads((Path(battery) / "learned.json").read_text())
    doc = {"header": HEADER, "not_used_by_any_rule": True, "environments": environments(retrain_root), "seeds": {}}
    for k in seeds:
        t0 = time.time()
        paper, retrain = paper_row(k, battery), retrain_row(k, sitems, retrain_root)
        l3 = learned["seeds"][str(k)]["position"]["layer3"]
        doc["seeds"][str(k)] = {
            "paper": paper, "retrain": retrain, "hardware_delta": hardware_delta(paper, retrain),
            "descriptive_learned_minus_retrain": learned_minus_retrain(
                k, l3["stats"], retrain["layer3_stats"], l3["accuracy"], retrain["layer3_position"]),
            "not_used_by_any_rule": True}
        print(f"[seed{k}] retrain epoch {retrain['selected_epoch']} test {retrain['test_accuracy']:.2f} layer3 "
              f"{retrain['layer3_position']:.4f}; paper epoch {paper['selected_epoch']} test {paper['test_accuracy']:.2f} "
              f"layer3 {paper['layer3_position']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return doc


def md_report(doc):
    L = ["# Sinusoidal hardware control", "", doc["header"], "",
         "Every row below is " + NOT_USED + ".", "",
         "## Per seed (paper checkpoint, retrain, hardware delta = retrain minus paper)", "",
         f"| Seed | Selected epoch (paper / retrain) | Test accuracy % (paper / retrain / delta) | Layer-3 position "
         f"(paper / retrain / delta) | Note |", "|---|---|---|---|---|"]
    for k, s in doc["seeds"].items():
        p, r, d = s["paper"], s["retrain"], s["hardware_delta"]
        L.append(f"| {k} | {p['selected_epoch']} / {r['selected_epoch']} | {p['test_accuracy']:.2f} / "
                 f"{r['test_accuracy']:.2f} / {d['test_accuracy_points']:+.2f} | {p['layer3_position']:.4f} / "
                 f"{r['layer3_position']:.4f} / {d['layer3_position']:+.4f} | {NOT_USED} |")
    L += ["", "## Descriptive: learned minus retrained sinusoidal, layer-3 position (seed-matched, paired bootstrap by "
          f"expression, {next(iter(doc['seeds'].values()))['descriptive_learned_minus_retrain']['n_boot']} resamples)", "",
          "| Seed | Learned | Retrained sinusoidal | Difference | 95% interval | Note |", "|---|---|---|---|---|---|"]
    for k, s in doc["seeds"].items():
        c = s["descriptive_learned_minus_retrain"]
        L.append(f"| {k} | {c['learned_layer3']:.4f} | {c['retrain_layer3']:.4f} | {c['diff']:+.4f} | "
                 f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] | {NOT_USED} |")
    L += ["", "## Environments", "", "| Checkpoints | GPU | Environment |", "|---|---|---|"]
    for e in doc["environments"]:
        extra = ""
        if "driver_version" in e:
            extra += f", driver {e['driver_version']}"
        if "git_commit" in e:
            extra += f", commit `{e['git_commit']}`"
        L.append(f"| {e['name']} | {e['gpu']} | {e['environment']}{extra} |")
    return "\n".join(L) + "\n"


def validate(doc):
    pb._has(doc, ("header", "not_used_by_any_rule", "environments", "seeds"), "hardware_control")
    assert doc["not_used_by_any_rule"] is True and NOT_USED in doc["header"]
    assert len(doc["environments"]) == 4
    assert [e["name"] for e in doc["environments"]] == ["paper checkpoints", "probe grid", "arms and NoPE",
                                                        "sinusoidal retrain"]
    for k, s in doc["seeds"].items():
        pb._has(s, ("paper", "retrain", "hardware_delta", "descriptive_learned_minus_retrain", "not_used_by_any_rule"),
                f"seed{k}")
        assert s["not_used_by_any_rule"] is True and s["hardware_delta"]["not_used_by_any_rule"] is True
        assert s["descriptive_learned_minus_retrain"]["not_used_by_any_rule"] is True
        for side in ("paper", "retrain"):
            pb._has(s[side], ("selected_epoch", "test_accuracy", "layer3_position", "layer3_stats"), f"seed{k} {side}")
            pb.validate_stats(s[side]["layer3_stats"], f"seed{k} {side}")
        pb._has(s["hardware_delta"], ("test_accuracy_points", "layer3_position", "selected_epoch"), f"seed{k} delta")
        assert abs(s["hardware_delta"]["layer3_position"]
                   - (s["retrain"]["layer3_position"] - s["paper"]["layer3_position"])) < 1e-12
        ci = s["descriptive_learned_minus_retrain"]["ci"]
        assert len(ci) == 2 and ci[0] <= ci[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--battery-dir", type=Path, default=BATTERY)
    ap.add_argument("--retrain-root", type=Path, default=RETRAIN_ROOT)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()
    t0 = time.time()
    doc = build(args.seeds, args.battery_dir, args.retrain_root)
    doc["environment"] = pb.environment_doc("hardware_control", "none", sys.argv[1:])
    doc["wall_seconds"] = round(time.time() - t0, 1)
    validate(doc)
    pb.write_json(args.out_dir / "hardware_control.json", doc)
    (args.out_dir / "hardware_control.md").write_text(md_report(doc))
    print(f"wrote hardware_control.json and hardware_control.md in {args.out_dir}; {doc['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()

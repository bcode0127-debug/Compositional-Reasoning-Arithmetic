"""Readings for the tuning control, implementing docs/design/tuning-control.md (pre-registered at TUNE_COMMIT).

Reads results_v2/tuning_control/<cell>/study1/seed<k>/ (history.json, test_results.json, ood_results.json,
environment.json), the descriptive RTX 4000 Ada runs in results_v2/pos_sweep, and the LSTM paper checkpoints
(pinned Hub commit, re-evaluated per expression with eval_test_splits.py --run-dir on CPU). Writes only under the
output directory.

  python experiments/scripts/tuning_control_readings.py
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load("pos_sweep_battery", "experiments/scripts/pos_sweep_battery.py")
npr = pb.npr
SEEDS = (0, 1, 2)
DESIGN = "docs/design/tuning-control.md"
TUNE_COMMIT = "a280bfc98b14b65c42a80025f14998095850d260"
FIRST_LINE = "Tuning control per docs/design/tuning-control.md (pre-registered at TUNE_COMMIT)."
RUN_ROOT = REPO_ROOT / "results_v2" / "tuning_control"
SWEEP_ROOT = REPO_ROOT / "results_v2" / "pos_sweep"
DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "analysis_v2" / "tuning_control"

# cell -> (encoding, learning rate, weight decay), as in the design's Cells table
CELLS = {"R1": ("sinusoidal", 3e-4, 0.0), "E1a": ("sinusoidal", 1e-4, 0.0), "E1b": ("sinusoidal", 1e-3, 0.0),
         "E3a": ("sinusoidal", 3e-4, 0.01), "R2": ("rope", 3e-4, 0.0), "E2a": ("rope", 1e-4, 0.0),
         "E2b": ("rope", 1e-3, 0.0), "E3b": ("sinusoidal", 3e-4, 0.1)}
ALL_CELLS = {"sinusoidal": ("E1a", "R1", "E1b", "E3a", "E3b"), "rope": ("E2a", "R2", "E2b")}  # every cell of the encoding
REFERENCE = {"sinusoidal": "R1", "rope": "R2"}
GRID = {"sinusoidal": ("E1a", "R1", "E1b"), "rope": ("E2a", "R2", "E2b")}  # weight decay 0: the learning-rate grid
LR_READING = ("E1a", "E1b", "E2a", "E2b")  # reading 1
WD_READING = ("E3a", "E3b")  # reading 5
RULES = ("max_train", "selected_epoch")  # primary first (Deviation 2026-11-01); the pre-registered rule second
RULE_TEXT = {"max_train": "maximum training accuracy over the run at least 80% (primary, Deviation 2026-11-01)",
             "selected_epoch": "training accuracy at the selected epoch at least 80% (as pre-registered)"}
SPIKE_POINTS = 10.0  # a run whose maximum training accuracy exceeds its final value by more than this is flagged
FIT_MIN, BAND_LO, BAND_HI, CONSTANT_BASELINE, EPOCH_CAP = 80.0, 70.0, 90.0, 1.50, 200
# descriptive: RTX 4000 Ada runs of the same configuration, reported beside the references
DESC_RUNS = {"sinusoidal_retrain": ("sinusoidal", "R1"), "rope": ("rope", "R2")}
LSTM_VAL_FINGERPRINTS = {0: 35.1, 1: 35.9, 2: 36.1}  # CHECKPOINTS.md, Study 1 LSTM checkpoints
POD_THREADS = {"nproc_shim": 6, "omp_num_threads_per_run": 3, "mkl_num_threads_per_run": 3, "max_concurrent_runs": 2,
               "pod_cpu_quota": "about 6.8 CPUs (cgroup cpu.max 680000 100000) while nproc reports 256",
               "note": "a pod-side /root/shim/nproc printing 6 was put first on PATH so run_tuning_control.sh set max(1, 6/2) = 3 "
                       "threads per run; no repository file changed. Recorded from the pod setup, not read from the runs."}
OOD_FILES = ("study1/ood_ops4.json", "study1/ood_ops5.json", "study1/ood_ops6.json", "study1/ood_ops7.json")


def run_dir(root, cell, seed):
    return Path(root) / cell / "study1" / f"seed{seed}"


# ---------------------------------------------------------------------------
# Per-run record and flags
# ---------------------------------------------------------------------------

def load_run(d, label, expect=None):
    """One run directory -> record with its per-expression test flags and the pre-registered per-run flags."""
    d = Path(d)
    hist = json.loads((d / "history.json").read_text())
    test = json.loads((d / "test_results.json").read_text())
    env = json.loads((d / "environment.json").read_text())
    ood = json.loads((d / "ood_results.json").read_text())
    flags = np.asarray(test["per_example_correct"], dtype=np.float64)
    assert test["n"] == 1000 and len(flags) == 1000, f"{d}: {len(flags)} test flags"
    if expect is not None:
        enc, lr, wd = expect
        assert test["positional_encoding"] == enc, f"{d}: encoding {test['positional_encoding']}"
        assert (env["learning_rate"], env["weight_decay"]) == (lr, wd), f"{d}: {env['learning_rate']}, {env['weight_decay']}"
        assert env["optimizer"] == ("AdamW" if wd > 0 else "Adam"), f"{d}: optimizer {env['optimizer']}"
    va = hist["val_accuracies"]
    e = va.index(max(va))  # selected epoch: the first index of the maximum of val_accuracies; train accuracy read at the same index
    assert e == test["epoch"], f"{d}: first argmax of val_accuracies {e} != checkpoint epoch {test['epoch']}"
    val_sel, train_sel = va[e], hist["train_accuracies"][e]
    train_max = max(hist["train_accuracies"])
    train_final = hist["train_accuracies"][-1]
    assert abs(val_sel - test["val_accuracy_fingerprint"]) < 1e-9, f"{d}: val at epoch {e}"
    losses = list(hist["train_losses"]) + list(hist["val_losses"])
    epochs_run = len(hist["val_accuracies"])
    diverged = (not all(math.isfinite(x) for x in losses)) or max(hist["val_accuracies"]) <= CONSTANT_BASELINE
    return {"label": label, "dir": str(d), "selected_epoch": e, "epochs_run": epochs_run,
            "train_accuracy_at_selected": train_sel, "max_train_accuracy": train_max, "final_train_accuracy": train_final,
            "max_minus_final": train_max - train_final, "spike_flag": bool(round(train_max - train_final, 6) > SPIKE_POINTS),
            "val_accuracy": val_sel,
            "test_accuracy": test["test_accuracy"],
            "fits_by_rule": {"max_train": bool(train_max >= FIT_MIN), "selected_epoch": bool(train_sel >= FIT_MIN)},
            "band_by_rule": {"max_train": bool(BAND_LO <= train_max < BAND_HI),
                             "selected_epoch": bool(BAND_LO <= train_sel < BAND_HI)},
            "diverged": bool(diverged), "epoch_cap": bool(epochs_run >= EPOCH_CAP),
            "wall_time_seconds": hist.get("wall_time_seconds"), "gpu": env.get("gpu"),
            "git_commit": env.get("git_commit"), "learning_rate": env.get("learning_rate"),
            "weight_decay": env.get("weight_decay"), "optimizer": env.get("optimizer"),
            "ood": {k: ood[k] for k in OOD_FILES}, "flags": flags}


def public(rec):
    return {k: v for k, v in rec.items() if k != "flags"}


def under_rule(runs, rule):
    """Copy of {cell: {seed: record}} (or {seed: record}) with fits and band set by one fit rule."""
    def one(r):
        if "fits_by_rule" not in r:  # LSTM paper checkpoints count as fitting under either rule
            return r
        return {**r, "fits": r["fits_by_rule"][rule], "band": r["band_by_rule"][rule]}
    first = next(iter(runs.values()))
    if isinstance(first, dict) and "flags" not in first:
        return {k: {s: one(r) for s, r in v.items()} for k, v in runs.items()}
    return {k: one(r) for k, r in runs.items()}


def load_cells(root=RUN_ROOT):
    return {c: {k: load_run(run_dir(root, c, k), f"{c}/seed{k}", CELLS[c]) for k in SEEDS} for c in CELLS}


def load_descriptive(sweep_root=SWEEP_ROOT):
    return {a: {k: load_run(run_dir(sweep_root, a, k), f"{a}/seed{k}") for k in SEEDS} for a in DESC_RUNS}


def non_fit_kind(r):
    """Why a non-fitting seed did not fit: it reached the epoch cap (a budget result), or it stopped before the cap
    (a rate result), with 'diverged' added when that flag is set. None for a fitting seed."""
    if r["fits"]:
        return None
    if r["epoch_cap"]:
        return "epoch cap"
    return "stopped before cap" + (", diverged" if r["diverged"] else "")


def cell_status(runs):
    n_fit = sum(r["fits"] for r in runs.values())
    return {"non_fit_kind": {str(k): non_fit_kind(r) for k, r in runs.items()}, "n_fitting_seeds": n_fit, "uninformative": bool(n_fit < 2),
            "fits": {str(k): r["fits"] for k, r in runs.items()},
            "diverged": {str(k): r["diverged"] for k, r in runs.items()},
            "epoch_cap": {str(k): r["epoch_cap"] for k, r in runs.items()},
            "band_70_90": {str(k): r["band"] for k, r in runs.items()}}


# ---------------------------------------------------------------------------
# LSTM paper checkpoints (Hub path, fingerprints on, per-expression test flags)
# ---------------------------------------------------------------------------

def lstm_reeval(out_dir, device="cpu"):
    """Re-evaluate the three Study 1 LSTM paper checkpoints per expression, as the battery does for the
    sinusoidal arm (sinusoidal_reeval): Hub file at the pinned commit -> eval_test_splits.py --run-dir."""
    import os
    npr.configure_device("cpu")
    for k in SEEDS:  # fingerprints on: load_model checks FINGERPRINTS where the checkpoint has one (LSTM seed 0)
        npr.load_model("lstm", "study1", k)
    for k in SEEDS:
        dst = Path(out_dir) / "lstm_reeval" / f"seed{k}"
        dst.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copyfile(npr.hf_file(npr.MODEL_REPO, f"lstm/study1/seed{k}/best_model.pt", "model"),
                            Path(tmp) / "best_model.pt")
            subprocess.run([sys.executable, "-u", "scripts/eval/eval_test_splits.py", "--run-dir", tmp, "--study",
                            "study1", "--device", device], cwd=REPO_ROOT, check=True,
                           env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
            shutil.copyfile(Path(tmp) / "test_results.json", dst / "test_results.json")


def load_lstm(out_dir):
    out = {}
    for k in SEEDS:
        t = json.loads((Path(out_dir) / "lstm_reeval" / f"seed{k}" / "test_results.json").read_text())
        assert t["model_type"] == "lstm" and t["n"] == 1000 and len(t["per_example_correct"]) == 1000
        assert round(t["val_accuracy_fingerprint"], 1) == LSTM_VAL_FINGERPRINTS[k], f"LSTM seed{k} fingerprint"
        ood = json.loads((REPO_ROOT / "results_v2" / "lstm" / "study1" / f"seed{k}" / "ood_results.json").read_text())
        out[k] = {"label": f"lstm/seed{k}", "test_accuracy": t["test_accuracy"], "val_accuracy": t["val_accuracy_fingerprint"],
                  "selected_epoch": t["epoch"], "fits": True, "band": False, "diverged": False, "epoch_cap": False,
                  "ood": {f: ood[f] for f in OOD_FILES}, "flags": np.asarray(t["per_example_correct"], dtype=np.float64)}
    return out


# ---------------------------------------------------------------------------
# Verification: Mac CPU recomputation against the recorded per-expression test results
# ---------------------------------------------------------------------------

class VerificationMismatch(Exception):
    pass


def recompute_flags(src_dir, device="cpu"):
    """Re-evaluate <src_dir>/best_model.pt per expression in a scratch copy, so the recorded test_results.json is untouched."""
    import os
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(Path(src_dir) / "best_model.pt", Path(tmp) / "best_model.pt")
        subprocess.run([sys.executable, "-u", "scripts/eval/eval_test_splits.py", "--run-dir", tmp, "--study", "study1",
                        "--device", device], cwd=REPO_ROOT, check=True, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
                       stdout=subprocess.DEVNULL)
        return json.loads((Path(tmp) / "test_results.json").read_text())


def compare_flags(name, recomputed, recorded_flags, recorded_acc):
    a, b = list(recomputed["per_example_correct"]), list(recorded_flags)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {"run": name, "recomputed_test_accuracy": recomputed["test_accuracy"], "recorded_test_accuracy": recorded_acc,
            "n_expressions": len(a), "n_flag_differences": len(diff) + abs(len(a) - len(b)), "first_differences": diff[:10],
            "match": bool(a == b and recomputed["test_accuracy"] == recorded_acc)}


def verify(out_dir, run_root=RUN_ROOT, lstm_done=None, device="cpu"):
    """Every tuning run: Mac recomputation against the pod's test_results.json. LSTM: the Hub re-evaluation against the
    committed paper values (results_v2/test_set_results.json). Raises VerificationMismatch before any reading is computed."""
    rows = []
    for c in CELLS:
        for k in SEEDS:
            d = run_dir(run_root, c, k)
            rec = json.loads((d / "test_results.json").read_text())
            rows.append({"cell": c, "seed": k, **compare_flags(f"{c}/seed{k}", recompute_flags(d, device),
                                                              rec["per_example_correct"], rec["test_accuracy"])})
            print(f"verify {c}/seed{k}: match={rows[-1]['match']}", flush=True)
    committed = json.loads((REPO_ROOT / "results_v2" / "test_set_results.json").read_text())
    for k in SEEDS:
        re = json.loads((Path(out_dir) / "lstm_reeval" / f"seed{k}" / "test_results.json").read_text())
        c = committed[f"lstm_study1_seed{k}"]
        rows.append({"cell": "lstm", "seed": k, **compare_flags(f"lstm/seed{k}", re, c["per_example_correct"], c["test_accuracy"])})
    bad = [r for r in rows if not r["match"]]
    doc = {"n_runs_checked": len(rows), "n_mismatches": len(bad), "device": device, "rows": rows}
    pb.write_json(Path(out_dir) / "verification.json", doc)
    if bad:
        raise VerificationMismatch("; ".join(f"{r['run']}: {r['n_flag_differences']} differing expressions, recomputed "
                                             f"{r['recomputed_test_accuracy']} vs recorded {r['recorded_test_accuracy']}" for r in bad))
    return doc


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------

def diff_ci(a, b, seed):
    sel = pb.boot_sel(npr.BOOT_SEED + seed, 1000)
    return pb.interval(a[sel].mean(-1) - b[sel].mean(-1))


def compare(a_runs, b_runs, drop_band=False):
    """a minus b, per seed, paired by expression and seed-matched. Eligible seeds: both runs fit (and neither is in
    the 70-90% band when drop_band). Holds if the interval excludes 0 on every eligible seed, same direction, with
    at least 2 eligible seeds."""
    per, elig = {}, []
    for k in SEEDS:
        a, b = a_runs[k], b_runs[k]
        ok = a["fits"] and b["fits"] and not (drop_band and (a["band"] or b["band"]))
        lo, hi = diff_ci(a["flags"], b["flags"], k)
        per[str(k)] = {"a_test": a["test_accuracy"], "b_test": b["test_accuracy"], "a_val": a["val_accuracy"],
                       "b_val": b["val_accuracy"], "diff_points": a["test_accuracy"] - b["test_accuracy"],
                       "ci_fraction": [lo, hi], "eligible": bool(ok)}
        if ok:
            elig.append(k)
    up = [k for k in elig if per[str(k)]["ci_fraction"][0] > 0]
    down = [k for k in elig if per[str(k)]["ci_fraction"][1] < 0]
    n = len(elig)
    if n < 2:
        verdict, direction = "not evaluable (fewer than 2 eligible seeds)", None
    elif len(up) == n:
        verdict, direction = "up", "up"
    elif len(down) == n:
        verdict, direction = "down", "down"
    else:
        verdict, direction = "no", None
    return {"per_seed": per, "n_eligible": n, "n_excluding_zero_up": len(up), "n_excluding_zero_down": len(down),
            "direction": direction, "holds": direction is not None, "verdict": verdict,
            "count": f"{max(len(up), len(down))}/{n} eligible seeds exclude 0 in one direction"}


def both_variants(a_runs, b_runs):
    return {"all_fitting_seeds": compare(a_runs, b_runs), "excluding_band_70_90": compare(a_runs, b_runs, drop_band=True)}


def mean(xs):
    return float(np.mean(xs))


def cell_summary(runs):
    return {"mean_test": mean([r["test_accuracy"] for r in runs.values()]),
            "mean_val": mean([r["val_accuracy"] for r in runs.values()]),
            "test_by_seed": {str(k): r["test_accuracy"] for k, r in runs.items()},
            "val_by_seed": {str(k): r["val_accuracy"] for k, r in runs.items()}}


def best_by_val(cells, enc, pool=GRID):
    """Selected on mean validation accuracy over seeds 0, 1, 2 (never on test). pool=GRID: the learning-rate grid (readings 3
    and 4); pool=ALL_CELLS: every cell of the encoding (the named best configuration)."""
    return max(pool[enc], key=lambda c: mean([r["val_accuracy"] for r in cells[c].values()]))


def evp(vals):
    """Expected maximum validation accuracy after n configurations drawn with replacement from the observed runs
    (Dodge et al. 2019), n = 1..len(vals)."""
    v = np.sort(np.asarray(vals, dtype=np.float64))
    N = len(v)
    return {str(n): float(sum(v[i] * (((i + 1) / N) ** n - (i / N) ** n) for i in range(N))) for n in range(1, N + 1)}


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------

def build(cells, desc, lstm, summary_lines=None, rule="max_train"):
    cells, desc = under_rule(cells, rule), under_rule(desc, rule)
    status = {c: cell_status(r) for c, r in cells.items()}
    best = {enc: best_by_val(cells, enc) for enc in GRID}
    best_all = {enc: best_by_val(cells, enc, ALL_CELLS) for enc in GRID}
    doc = {"first_line": FIRST_LINE, "design": DESIGN, "tune_commit": TUNE_COMMIT, "fit_rule": rule,
           "fit_rule_text": RULE_TEXT[rule],
           "cells": {c: {"encoding": CELLS[c][0], "learning_rate": CELLS[c][1], "weight_decay": CELLS[c][2],
                         "status": status[c], "summary": cell_summary(cells[c]),
                         "runs": {str(k): public(r) for k, r in cells[c].items()}} for c in CELLS}}

    def reading1(cs):
        out = {}
        for c in cs:
            ref = REFERENCE[CELLS[c][0]]
            if status[c]["uninformative"] or status[ref]["uninformative"]:
                out[c] = {"reference": ref, "uninformative": True,
                          "note": "fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result"}
            else:
                out[c] = {"reference": ref, "uninformative": False, **both_variants(cells[c], cells[ref])}
        return out

    doc["reading1_tuning_effect"] = reading1(LR_READING)
    r2 = {}
    for enc in GRID:
        rows = {c: cell_summary(cells[c]) for c in GRID[enc]}
        every = [(cells[c][k]["test_accuracy"], c, k) for c in GRID[enc] for k in SEEDS]
        top = max(every)
        r2[enc] = {"per_rate_means_over_seeds_0_1_2": rows,
                   "best_over_grid_maximum": {"test_accuracy": top[0], "cell": top[1], "seed": top[2]},
                   "best_learning_rate_cell_by_mean_validation": best[enc],
                   "best_configuration_by_mean_validation": best_all[enc],
                   "configurations_covered": list(ALL_CELLS[enc]),
                   "expected_validation_performance": evp([cells[c][k]["val_accuracy"] for c in ALL_CELLS[enc] for k in SEEDS])}
    doc["reading2_best_over_grid"] = r2
    sin_best, rope_best = best["sinusoidal"], best["rope"]
    def pool(enc):
        return {"cells_mean_validation_over_seeds_0_1_2": {c: mean([r["val_accuracy"] for r in cells[c].values()]) for c in GRID[enc]},
                "selected_cell": best[enc]}

    doc["reading3_architecture_gap"] = {
        "selection_pool": {"sinusoidal": pool("sinusoidal")},
        "sinusoidal_cell": sin_best, "lstm_minus_transformer": both_variants(lstm, cells[sin_best]),
        "descriptive_best_configuration_overall": {
            "label": "descriptive", "sinusoidal_cell": best_all["sinusoidal"],
            "lstm_minus_transformer": both_variants(lstm, cells[best_all["sinusoidal"]])}}
    doc["reading4_rope_gain"] = {"selection_pool": {"rope": pool("rope"), "sinusoidal": pool("sinusoidal")}, "rope_cell": rope_best, "sinusoidal_cell": sin_best,
                                 "rope_minus_sinusoidal": both_variants(cells[rope_best], cells[sin_best])}
    doc["reading5_weight_decay"] = reading1(WD_READING)
    doc["lstm"] = {"summary": cell_summary(lstm), "runs": {str(k): public(r) for k, r in lstm.items()}}
    doc["descriptive_rtx4000ada"] = {
        a: {"beside": ref, "summary": cell_summary(desc[a]), "runs": {str(k): public(r) for k, r in desc[a].items()},
            "diff_vs_pod_reference_not_a_reference": compare(desc[a], cells[ref]),
            "note": "RTX 4000 Ada; reported beside the same-GPU reference, descriptively, never as a reference"}
        for a, (_, ref) in DESC_RUNS.items()}
    ood = {}
    for c in CELLS:
        ood[c] = {f: mean([cells[c][k]["ood"][f] for k in SEEDS]) for f in OOD_FILES}
    ood["lstm"] = {f: mean([lstm[k]["ood"][f] for k in SEEDS]) for f in OOD_FILES}
    doc["ood_ops4_7_descriptive"] = ood
    walls = {c: [cells[c][k]["wall_time_seconds"] for k in SEEDS] for c in CELLS}
    doc["compute"] = {"train_seconds_by_cell": walls,
                      "total_train_hours": sum(sum(w) for w in walls.values()) / 3600,
                      "reference_train_hours": sum(sum(walls[c]) for c in ("R1", "R2")) / 3600,
                      "summary_txt_run_lines": summary_lines,
                      "note": "sum of per-run training wall time; two runs shared the pod, so pod wall-clock is shorter"}
    return doc


def read_summary(root):
    p = Path(root) / "summary.txt"
    return p.read_text().splitlines() if p.exists() else None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def f2(x):
    return f"{x:.2f}"


def ci_t(ci):
    return f"[{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]"


def cmp_table(label_a, label_b, cmp):
    L = [f"| Seed | {label_a} test % (val %) | {label_b} test % (val %) | Difference (points) | 95% interval (points) | Eligible |",
         "|---|---|---|---|---|---|"]
    for k, s in cmp["per_seed"].items():
        L.append(f"| {k} | {f2(s['a_test'])} ({f2(s['a_val'])}) | {f2(s['b_test'])} ({f2(s['b_val'])}) | "
                 f"{s['diff_points']:+.2f} | {ci_t(s['ci_fraction'])} | {'yes' if s['eligible'] else 'no'} |")
    return L


def variant_block(label_a, label_b, v, word):
    L = []
    for name, c in (("all fitting seeds", v["all_fitting_seeds"]), ("excluding runs with 70-90% training accuracy", v["excluding_band_70_90"])):
        verdict = c["verdict"] if c["direction"] is None else f"{word} {c['direction']}"
        L += [f"*{name}:* {verdict} ({c['count']}).", ""] + cmp_table(label_a, label_b, c) + [""]
    return L


def pool_lines(pools):
    L = []
    for enc, p in pools.items():
        L.append(f"Selection pool, {enc} (mean validation accuracy over seeds 0, 1 and 2, whatever their fit): " +
                 ", ".join(f"{c} {f2(v)}%" for c, v in p["cells_mean_validation_over_seeds_0_1_2"].items()) +
                 f". Selected: {p['selected_cell']}.")
    return L + [""]


def md_report(doc):
    L = [doc["first_line"], "", f"Fit rule for this section: {doc['fit_rule_text']}.", "", "Test accuracy is the endpoint; every test number is shown beside its validation number. "
         "Paired bootstraps resample the 1000 test expressions (N_BOOT 1000), seed-matched. Learning rates are selected on mean "
         "validation accuracy over seeds 0, 1 and 2, never on test.", "",
         "Interpretation notes: the learning-rate grid for readings 2 to 4 is the weight-decay-0 cells of each encoding (sinusoidal: "
         "E1a, R1, E1b; RoPE: E2a, R2, E2b); E3a and E3b enter reading 5 only. A seed is eligible for a comparison when both runs fit "
         "under this section's fit rule (stated above); the LSTM paper checkpoints count as fitting.", "",
         "## Per-run table", "",
         "| Cell | Seed | Selected epoch | Train acc % at selected epoch | Max train acc % | Final train acc % | Max minus final | Spike flag | Val acc % | Test acc % | Fits | Diverged | Epoch cap | 70-90% band |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, cd in doc["cells"].items():
        for k, r in cd["runs"].items():
            L.append(f"| {c} | {k} | {r['selected_epoch']} | {f2(r['train_accuracy_at_selected'])} | {f2(r['max_train_accuracy'])} | {f2(r['final_train_accuracy'])} | "
                     f"{f2(r['max_minus_final'])} | {r['spike_flag']} | {f2(r['val_accuracy'])} | "
                     f"{f2(r['test_accuracy'])} | {r['fits']} | {r['diverged']} | {r['epoch_cap']} | {r['band']} |")

    def r1(title, block):
        out = ["", title, ""]
        for c, d in block.items():
            out += [f"### {c} against {d['reference']}", ""]
            if d["uninformative"]:
                out += [d["note"], ""]
            else:
                out += variant_block(c, d["reference"], d, "moves")
        return out

    L += r1("## Reading 1: tuning effect per cell (cell minus its reference)", doc["reading1_tuning_effect"])
    L += ["", "## Reading 2: best over the grid", "", "Maxima and per-rate means over seeds 0, 1 and 2 are labelled as such.", ""]
    for enc, d in doc["reading2_best_over_grid"].items():
        m = d["best_over_grid_maximum"]
        L += [f"### {enc}", "", f"Maximum test accuracy over its learning rates and seeds: **{f2(m['test_accuracy'])}%** (cell {m['cell']}, seed {m['seed']}). "
              f"Best configuration by mean validation accuracy over all of the encoding's cells ({', '.join(d['configurations_covered'])}): "
              f"**{d['best_configuration_by_mean_validation']}**; best learning-rate-grid cell: **{d['best_learning_rate_cell_by_mean_validation']}**.", "",
              "| Cell | LR | Mean val % (mean over seeds) | Mean test % (mean over seeds) |", "|---|---|---|---|"]
        for c, s in d["per_rate_means_over_seeds_0_1_2"].items():
            L.append(f"| {c} | {CELLS[c][1]:g} | {f2(s['mean_val'])} | {f2(s['mean_test'])} |")
        e = d["expected_validation_performance"]
        L += ["", "Expected maximum validation accuracy after n configurations (Dodge et al.), descriptive: " +
              ", ".join(f"n={n}: {f2(v)}" for n, v in e.items()), ""]
    r3 = doc["reading3_architecture_gap"]
    L += ["## Reading 3: the architecture gap under tuning", "",
          f"Sinusoidal learning rate with the highest mean validation accuracy: cell {r3['sinusoidal_cell']}. LSTM minus Transformer.", ""]
    L += pool_lines(r3["selection_pool"])
    L += variant_block("LSTM", r3["sinusoidal_cell"], r3["lstm_minus_transformer"], "gap holds")
    dsc = r3["descriptive_best_configuration_overall"]
    L += [f"### Descriptive: the gap at the best configuration overall (cell {dsc['sinusoidal_cell']}, all sinusoidal cells including weight decay)", "",
          "Not part of the pre-registered reading 3.", ""]
    L += variant_block("LSTM", dsc["sinusoidal_cell"], dsc["lstm_minus_transformer"], "gap holds")
    r4 = doc["reading4_rope_gain"]
    L += ["## Reading 4: RoPE's gain under tuning", "",
          f"Best-validation cells: RoPE {r4['rope_cell']}, sinusoidal {r4['sinusoidal_cell']}. RoPE minus sinusoidal.", ""]
    L += pool_lines(r4["selection_pool"])
    L += variant_block("RoPE", "sinusoidal", r4["rope_minus_sinusoidal"], "gain holds")
    L += r1("## Reading 5: weight decay (reading 1 applied to E3a and E3b)", doc["reading5_weight_decay"])
    L += ["", "## Descriptive: RTX 4000 Ada runs beside the same-GPU references (never references)", ""]
    for a, d in doc["descriptive_rtx4000ada"].items():
        L += [f"### {a} beside {d['beside']}", "", f"Mean val {f2(d['summary']['mean_val'])}%, mean test {f2(d['summary']['mean_test'])}% "
              f"(RTX 4000 Ada) against the pod reference. Difference RTX 4000 Ada minus pod:", ""]
        L += cmp_table(a, d["beside"], d["diff_vs_pod_reference_not_a_reference"]) + [""]
    L += ["## Out-of-distribution accuracy, operations 4-7 (descriptive, means over seeds 0, 1 and 2)", "",
          "| Cell | ops4 | ops5 | ops6 | ops7 |", "|---|---|---|---|---|"]
    for c, o in doc["ood_ops4_7_descriptive"].items():
        L.append(f"| {c} | " + " | ".join(f2(o[f]) for f in OOD_FILES) + " |")
    cp = doc["compute"]
    L += ["", "## Compute", "", f"Total training time over all 24 runs: {cp['total_train_hours']:.2f} run-hours; reference runs (R1, R2): "
          f"{cp['reference_train_hours']:.2f} run-hours. {cp['note']}.", ""]
    return "\n".join(L)


SAME_LINE = "All other cells: same classification under both rules."


def verdict_of(entry):
    if entry["uninformative"]:
        return "uninformative"
    c = entry["all_fitting_seeds"]
    return c["direction"] if c["direction"] is not None else c["verdict"]


def verdict_map(doc):
    out = {}
    for key, label in (("reading1_tuning_effect", "reading 1"), ("reading5_weight_decay", "reading 5")):
        for c, d in doc[key].items():
            out[f"{label} {c}"] = verdict_of(d)
    r3, r4 = doc["reading3_architecture_gap"], doc["reading4_rope_gain"]
    out["reading 3"] = r3["lstm_minus_transformer"]["all_fitting_seeds"]["direction"] or r3["lstm_minus_transformer"]["all_fitting_seeds"]["verdict"]
    out["reading 4"] = r4["rope_minus_sinusoidal"]["all_fitting_seeds"]["direction"] or r4["rope_minus_sinusoidal"]["all_fitting_seeds"]["verdict"]
    return out


def differences(full):
    """The two cross-rule lines: cells whose informative/uninformative classification differs, readings whose verdict differs."""
    a, b = full[RULES[0]], full[RULES[1]]
    cls = [f"{c} ({'uninformative' if a['cells'][c]['status']['uninformative'] else 'informative'} under max_train, "
           f"{'uninformative' if b['cells'][c]['status']['uninformative'] else 'informative'} under selected_epoch)"
           for c in CELLS if a["cells"][c]["status"]["uninformative"] != b["cells"][c]["status"]["uninformative"]]
    va, vb = verdict_map(a), verdict_map(b)
    ver = [f"{k} ({va[k]} under max_train, {vb[k]} under selected_epoch)" for k in va if va[k] != vb[k]]
    return ("Classification differs between rules: " + ("; ".join(cls) if cls else "none"),
            "Verdict differs between rules: " + ("; ".join(ver) if ver else "none"))


def cells_table(full):
    a, b = full[RULES[0]], full[RULES[1]]
    L = ["## Cells: fit count and classification under each rule", "",
         "| Cell | Encoding | LR | WD | Fitting seeds, max_train (primary) | Classification, max_train | Fitting seeds, selected_epoch (pre-registered) | "
         "Classification, selected_epoch | Non-fit kind, max_train | Non-fit kind, selected_epoch | Mean val % | Mean test % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, cd in a["cells"].items():
        sa, sb = cd["status"], b["cells"][c]["status"]
        cl = lambda st: "uninformative" if st["uninformative"] else "informative"
        kinds = lambda st: "; ".join(f"seed{k}: {v}" for k, v in st["non_fit_kind"].items() if v) or "none"
        L.append(f"| {c} | {cd['encoding']} | {cd['learning_rate']:g} | {cd['weight_decay']:g} | {sa['n_fitting_seeds']}/3 | {cl(sa)} | "
                 f"{sb['n_fitting_seeds']}/3 | {cl(sb)} | {kinds(sa)} | {kinds(sb)} | {f2(cd['summary']['mean_val'])} | {f2(cd['summary']['mean_test'])} |")
    return L


def md_full(full):
    first, second = md_report(full[RULES[0]]).splitlines(), md_report(full[RULES[1]]).splitlines()
    cls, ver = differences(full)
    return "\n".join([first[0], ""] + cells_table(full) + ["", cls, SAME_LINE, "", ver, ""] + first[1:] +
                      ["", "---", "", "## Under the pre-registered fit rule (selected-epoch training accuracy)", ""] + second[1:]) + "\n"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def validate(full):
    pb._has(full, ("first_line", "fit_rules", "primary_rule", "pre_registered_rule", *RULES), "readings")
    assert full["first_line"] == FIRST_LINE and full["primary_rule"] == "max_train" and full["pre_registered_rule"] == "selected_epoch"
    for rule in RULES:
        assert full[rule]["fit_rule"] == rule
        validate_rule(full[rule])


def validate_rule(doc):
    pb._has(doc, ("first_line", "design", "tune_commit", "cells", "reading1_tuning_effect", "reading2_best_over_grid",
                  "reading3_architecture_gap", "reading4_rope_gain", "reading5_weight_decay", "lstm",
                  "descriptive_rtx4000ada", "ood_ops4_7_descriptive", "compute", "fit_rule"), "readings")
    assert doc["first_line"] == FIRST_LINE and set(doc["cells"]) == set(CELLS)
    for c, cd in doc["cells"].items():
        pb._has(cd, ("encoding", "learning_rate", "weight_decay", "status", "summary", "runs"), c)
        assert set(cd["runs"]) == {"0", "1", "2"}
        pb._has(cd["status"], ("n_fitting_seeds", "uninformative", "fits", "diverged", "epoch_cap", "band_70_90"), c)
        for r in cd["runs"].values():
            pb._has(r, ("selected_epoch", "train_accuracy_at_selected", "max_train_accuracy", "final_train_accuracy", "max_minus_final", "spike_flag", "val_accuracy", "test_accuracy", "fits", "diverged",
                        "epoch_cap", "band", "ood"), c)

    def check_cmp(c, where):
        pb._has(c, ("per_seed", "n_eligible", "direction", "holds", "verdict", "count"), where)
        assert set(c["per_seed"]) == {"0", "1", "2"}
        for s in c["per_seed"].values():
            assert len(s["ci_fraction"]) == 2 and s["ci_fraction"][0] <= s["ci_fraction"][1]

    def check_variants(v, where):
        pb._has(v, ("all_fitting_seeds", "excluding_band_70_90"), where)
        for n in ("all_fitting_seeds", "excluding_band_70_90"):
            check_cmp(v[n], f"{where} {n}")

    for rd in ("reading3_architecture_gap", "reading4_rope_gain"):
        assert all(p["selected_cell"] in p["cells_mean_validation_over_seeds_0_1_2"] for p in doc[rd]["selection_pool"].values())
    for c in CELLS:
        assert set(doc["cells"][c]["status"]["non_fit_kind"]) == {"0", "1", "2"}
    for key, cs in (("reading1_tuning_effect", LR_READING), ("reading5_weight_decay", WD_READING)):
        assert set(doc[key]) == set(cs)
        for c, d in doc[key].items():
            assert d["reference"] == REFERENCE[CELLS[c][0]]
            if not d["uninformative"]:
                check_variants(d, f"{key} {c}")
    assert set(doc["reading2_best_over_grid"]) == {"sinusoidal", "rope"}
    for enc, d in doc["reading2_best_over_grid"].items():
        pb._has(d, ("per_rate_means_over_seeds_0_1_2", "best_over_grid_maximum", "best_configuration_by_mean_validation",
                    "expected_validation_performance", "best_learning_rate_cell_by_mean_validation", "configurations_covered"), enc)
        assert d["best_configuration_by_mean_validation"] in ALL_CELLS[enc] and d["best_learning_rate_cell_by_mean_validation"] in GRID[enc]
        assert len(d["expected_validation_performance"]) == 3 * len(ALL_CELLS[enc])
    check_variants(doc["reading3_architecture_gap"]["lstm_minus_transformer"], "reading3")
    dsc = doc["reading3_architecture_gap"]["descriptive_best_configuration_overall"]
    assert dsc["label"] == "descriptive" and dsc["sinusoidal_cell"] in ALL_CELLS["sinusoidal"]
    check_variants(dsc["lstm_minus_transformer"], "reading3 descriptive")
    check_variants(doc["reading4_rope_gain"]["rope_minus_sinusoidal"], "reading4")
    assert set(doc["descriptive_rtx4000ada"]) == set(DESC_RUNS)
    assert set(doc["ood_ops4_7_descriptive"]) == set(CELLS) | {"lstm"}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT)
    ap.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    ap.add_argument("--skip-lstm-reeval", action="store_true", help="reuse out-dir/lstm_reeval")
    ap.add_argument("--skip-verify", action="store_true", help="tests only; the real run always verifies first")
    args = ap.parse_args()
    t0 = datetime.datetime.now()
    npr.configure_device("cpu")
    if not args.skip_lstm_reeval:
        lstm_reeval(args.out_dir)
    if not args.skip_verify:
        try:
            v = verify(args.out_dir, args.run_root)
        except VerificationMismatch as e:
            print(f"STOP: verification mismatch, no reading computed: {e}", file=sys.stderr)
            sys.exit(2)
        print(f"verification: {v['n_runs_checked']} runs, {v['n_mismatches']} mismatches")
    cells, desc, lstm = load_cells(args.run_root), load_descriptive(args.sweep_root), load_lstm(args.out_dir)
    summary = read_summary(args.run_root)
    doc = {"first_line": FIRST_LINE, "fit_rules": RULE_TEXT, "primary_rule": RULES[0], "pre_registered_rule": RULES[1],
           **{r: build(cells, desc, lstm, summary, r) for r in RULES}}
    validate(doc)
    pb.write_json(args.out_dir / "readings.json", doc)
    (args.out_dir / "readings.md").write_text(md_full(doc))
    env = pb.environment_doc("tuning_control_readings", "cpu", sys.argv[1:])
    env.update({"pod_thread_setting": POD_THREADS, "tune_commit": TUNE_COMMIT, "design_sha256": sha256_file(REPO_ROOT / DESIGN),
                "wall_seconds": round((datetime.datetime.now() - t0).total_seconds(), 1)})
    pb.write_json(args.out_dir / "environment.json", env)
    print(f"wrote readings.json, readings.md, environment.json in {args.out_dir}")


if __name__ == "__main__":
    main()

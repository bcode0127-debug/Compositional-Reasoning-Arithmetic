"""Per-arm probe battery for the positional-encoding sweep.

Design: docs/design/pos-encoding-sweep.md (pre-registration 9365283) as amended by its Deviations entries
(2026-10-19 x2, 2026-10-28). Rules are evaluated from per-expression statistics written by the probe stage, so
`--stage rules` can be rerun without re-probing.

Stages (default all):
  probes  anchor + position/depth rows for sinusoidal (Hub), learned/rope/alibi (local), nope (local, floor only)
  reeval  re-evaluate the three paper checkpoints on the test split in this environment (sinusoidal_reeval/)
  rules   decision_rules.json + summary.md from the files above

  python experiments/scripts/pos_sweep_battery.py --stage all --eval-device cuda
"""
import argparse
import datetime
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
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


npr = _load("nonlinear_probe", "experiments/scripts/nonlinear_probe.py")
diag = _load("nope_floor_diag", "experiments/scripts/nope_floor_diag.py")

DESIGN = "docs/design/pos-encoding-sweep.md"
DESIGN_COMMIT = "9365283"
DEVIATIONS_APPLIED = ["2026-10-19: Specifications added before the first run", "2026-10-19: Probe extraction path",
                      "2026-10-28: NoPE floor failed; the position reference is replaced by an order-blind baseline"]
ARMS = ("sinusoidal", "learned", "rope", "alibi")
SEEDS = (0, 1, 2)
FEATURES = npr.LAYERWISE_FEATURES_4E  # encoder_input, layer1, layer2, layer3, mean_pool, memory
STRAT = npr.STRAT_SPLIT
PREREG_REF = 0.053
MARGIN = 0.10
EQUIV_TOL = 1e-5
FLOOR_BAND = 0.02
EXPECTED_BASELINES = {"B1": 0.1621, "B2": 0.1894, "B3": 0.1880}  # Deviation 2026-10-28, 3b
REFS = {"anchor": "vs_anchor", "prereg": "vs_prereg"}
REF_TEXT = {"anchor": "ANCHOR (order-blind baseline, primary)",
            "prereg": "the design's 0.053, superseded before any arm was probed by order-blind baseline B2 at "
                      "0.1894, on the D1 and D2 evidence"}
RUN_ROOT = REPO_ROOT / "results_v2" / "pos_sweep"
DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "analysis_v2" / "pos_sweep"
FLOOR_DIAG = DEFAULT_OUT / "nope_floor" / "nope_floor_diag.json"
GATE_REF_FILE = REPO_ROOT / "experiments/results/analysis_v2/nonlinear_probe/step4e_addenda.json"


class NotEvaluable(Exception):
    pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def run_dir_for(arm, seed):
    return RUN_ROOT / arm / "study1" / f"seed{seed}"


def boot_sel(seed, n):
    """Resample indices, the same draw paired_bootstrap makes for this seed (N_BOOT x n examples)."""
    return np.random.default_rng(seed).integers(0, n, size=(npr.N_BOOT, n))


def interval(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def as_stats(d):
    return {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}


def stats_json(st):
    return {k: [float(x) for x in v] for k, v in st.items()}


def acc_boot(st, sel):
    return npr.stat_metric("depth", st, sel)


def write_json(path, doc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2))


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Probe stage
# ---------------------------------------------------------------------------

def assert_expected_baselines(out):
    """The Deviation entry's recorded B1-B3 accuracies, to four decimals: a different split, feature build or
    library stack fails here, loudly, before any arm is probed."""
    for kind, want in EXPECTED_BASELINES.items():
        got = round(out[kind]["accuracy"], 4)
        assert got == want, f"ANCHOR check: {kind} accuracy {got} != {want} (Deviation 2026-10-28 3b)"


def compute_anchor(sitems, check):
    """B1-B3 through the sweep's position probe on the stratified split; ANCHOR is the highest of the three."""
    te = sitems[STRAT]
    offsets = np.concatenate([[0], np.cumsum([len(it["input"]) for it in te])]).astype(np.int64)
    out, positions = {}, None
    for kind in ("B1", "B2", "B3"):
        b = {s: diag.baseline_features(sitems[s], kind) for s in ("train", STRAT)}
        r = npr.position_probe({"X": b["train"]["X"], "position": b["train"]["position"]},
                               {"X": b[STRAT]["X"], "position": b[STRAT]["position"], "offsets": offsets})
        stats = r.pop("stats")
        r["stats"] = stats_json(stats)
        r["dim"] = int(b["train"]["X"].shape[1])
        out[kind] = r
        positions = b[STRAT]["position"]
    if check == "assert":
        assert_expected_baselines(out)
    best = max(out, key=lambda k: out[k]["accuracy"])  # first of B1, B2, B3 on a tie
    anchor = {"baselines": out, "anchor_name": best, "anchor_accuracy": out[best]["accuracy"],
              "expected_baselines": EXPECTED_BASELINES, "expected_check": check,
              "majority": out[best]["majority"], "prereg_reference": PREREG_REF,
              "n_expressions": len(te), "split": STRAT}
    return anchor, positions


def row_vs_references(acc, st, sel, anchor_acc, anchor_stats):
    ab = acc_boot(st, sel)
    nb = acc_boot(anchor_stats, sel)
    return ({"diff": float(acc - anchor_acc), "ci": interval(ab - nb)},
            {"diff": float(acc - PREREG_REF), "ci": interval(ab - PREREG_REF)})


def depth_memory_row(a, cp_block, prot_cp):
    """Depth at the memory under the paper's protocol: linear + MLP, control, bootstraps vs control, vs the
    split's majority, vs the one-hot(char, position) baseline (prot_cp rebuilt in this run)."""
    lin = npr.linear_probe("depth", a["train"], {STRAT: a[STRAT]})
    prot = npr.run_mlp_protocol("depth", a, (STRAT,), npr.HIDDEN)
    e = npr.assemble_split("depth", STRAT, a, lin, prot)
    n_ex = len(a[STRAT]["offsets"]) - 1
    ci_cp = npr.paired_cis(prot, prot_cp, STRAT, n_ex)
    base_mlp = cp_block[STRAT]["mlp_mean"]
    keep = ("linear", "mlp_mean", "mlp_std", "mlp_per_seed", "control", "control_train", "majority", "majority_class",
            "bootstrap_ci", "bootstrap_ci_vs_majority", "n_examples", "n_positions", "verdict")
    row = {k: e[k] for k in keep}
    row["verdict_rule_base"] = row.pop("verdict")
    row["dim"] = int(a[STRAT]["X"].shape[1])
    row["char_position_baseline_mlp"] = base_mlp
    row["bootstrap_ci_vs_char_position_baseline"] = ci_cp
    row["depth_code_appears"] = bool(
        row["verdict_rule_base"] == "yes"
        and all(c[0] > 0 for c in row["bootstrap_ci_vs_majority"].values())
        and all(c[0] > 0 for c in ci_cp.values())
        and row["mlp_mean"] > base_mlp)
    return row


def probe_checkpoint(model, sitems, anchor, anchor_positions, seed, depth_ctx):
    feats = npr.layerwise_acts(model, sitems, features=FEATURES, with_ids=True)
    anchor_stats = as_stats(anchor["baselines"][anchor["anchor_name"]]["stats"])
    n_ex = len(feats["layer3"][STRAT]["offsets"]) - 1
    sel = boot_sel(npr.BOOT_SEED + seed, n_ex)
    position = {}
    for f in FEATURES:
        assert np.array_equal(feats[f][STRAT]["position"], anchor_positions), f"{f}: rows misaligned with the anchor"
        r = npr.position_probe(feats[f]["train"], feats[f][STRAT], "position")
        st = r.pop("stats")
        va, vp = row_vs_references(r["accuracy"], st, sel, anchor["anchor_accuracy"], anchor_stats)
        r.update({"dim": int(feats[f]["train"]["X"].shape[1]), "stats": stats_json(st),
                  "vs_anchor": va, "vs_prereg": vp})
        position[f] = r
    out = {"position": position}
    if depth_ctx is not None:
        out["depth_memory"] = depth_memory_row(feats["memory"], *depth_ctx)
    return out


def checkpoint_meta(arm, seed, run_dir):
    if run_dir is None:
        rec = npr.FILE_RECORDS[f"model:transformer/study1/seed{seed}/best_model.pt"]
        meta = dict(npr.CKPT_META[f"transformer/study1/seed{seed}"])
        meta.update({"source": "hub", "hf_commit": rec["hf_commit"], "sha256": rec["sha256"],
                     "filename": rec["filename"]})
    else:
        meta = dict(npr.CKPT_META[str(run_dir)])
        meta.update({"source": "local", "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                     "sha256": npr.sha256_file(run_dir / "best_model.pt")})
    return meta


def probe_arm(arm, seeds, sitems, anchor, anchor_positions, depth_ctx, out_dir):
    doc = {"arm": arm, "split": STRAT, "features": list(FEATURES), "seeds": {}}
    for seed in seeds:
        t0 = time.time()
        run_dir = None if arm == "sinusoidal" else run_dir_for(arm, seed)
        model = npr.load_model("transformer", "study1", seed, run_dir=run_dir)
        meta = checkpoint_meta(arm, seed, run_dir)
        doc["positional_encoding"] = meta.get("positional_encoding") or "sinusoidal"
        res = probe_checkpoint(model, sitems, anchor, anchor_positions, seed, depth_ctx)
        res["checkpoint"] = meta
        doc["seeds"][str(seed)] = res
        l3 = res["position"]["layer3"]
        print(f"[{arm} seed{seed}] layer3 position {l3['accuracy']:.4f} (vs ANCHOR {l3['vs_anchor']['diff']:+.4f} "
              f"CI {l3['vs_anchor']['ci'][0]:+.4f}..{l3['vs_anchor']['ci'][1]:+.4f}); "
              f"{time.time() - t0:.0f}s", flush=True)
        write_json(out_dir / f"{arm}.json", doc)  # incremental: a crash keeps finished seeds
        del model
    return doc


def gate_reference():
    """Committed layer-3 position for the paper's sinusoidal seed 0 (the design's reproduction gate)."""
    ref = json.loads(GATE_REF_FILE.read_text())
    return ref["identity"]["seed0"]["layer3"]["position"]["accuracy"]


def environment_doc(stage, eval_device, argv):
    import sklearn
    import train_v2
    env = train_v2.run_environment("cpu")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                           capture_output=True, text=True).stdout.strip()
    env.update({"sklearn": sklearn.__version__, "probe_device": "cpu", "eval_device": eval_device,
                "stage": stage, "argv": argv, "git_dirty_tracked": bool(dirty),
                "role": "positional-encoding sweep probe battery (one run, one environment, every checkpoint)"})
    return env


def run_probes(args, out_dir):
    npr.configure_device("cpu")
    torch.use_deterministic_algorithms(False)  # as the reproduction gate and the NoPE diagnostic
    sitems, _ = npr.stratified_split_items()
    anchor, anchor_positions = compute_anchor(sitems, args.anchor_check)
    write_json(out_dir / "anchor.json", anchor)
    print(f"ANCHOR {anchor['anchor_name']} = {anchor['anchor_accuracy']:.4f}  " + "  ".join(
        f"{k} {v['accuracy']:.4f}" for k, v in anchor["baselines"].items()), flush=True)

    depth_ctx = None
    if not args.no_depth:
        acts_cp = {k: npr.char_position_features(v) for k, v in sitems.items()}
        cp_block, prot_cp = npr.baseline_block(acts_cp, (STRAT,))
        depth_ctx = (cp_block, prot_cp)
        print(f"one-hot(char, position) baseline MLP {cp_block[STRAT]['mlp_mean']:.4f}", flush=True)

    for arm in args.arms:
        seeds = args.seeds
        doc = probe_arm(arm, seeds, sitems, anchor, anchor_positions, depth_ctx, out_dir)
        if arm == "sinusoidal" and 0 in seeds:
            got = doc["seeds"]["0"]["position"]["layer3"]["accuracy"]
            ref = gate_reference()
            assert abs(got - ref) <= 0.02, f"reproduction gate: sinusoidal seed 0 layer 3 {got:.4f} vs {ref:.4f}"
            print(f"reproduction gate (position, layer 3): {got:.4f} vs committed {ref:.4f}", flush=True)
    if not args.no_nope:
        probe_arm("nope", (0,), sitems, anchor, anchor_positions, None, out_dir)


# ---------------------------------------------------------------------------
# Re-evaluation of the paper's sinusoidal checkpoints (shared environment and device with the arm runs)
# ---------------------------------------------------------------------------

def run_reeval(out_dir, eval_device):
    for seed in SEEDS:
        d = out_dir / "sinusoidal_reeval" / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        src = npr.hf_file(npr.MODEL_REPO, f"transformer/study1/seed{seed}/best_model.pt", "model")
        shutil.copyfile(src, d / "best_model.pt")  # *.pt is gitignored; not committed
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        subprocess.run([sys.executable, "-u", "scripts/eval/eval_test_splits.py", "--run-dir", str(d),
                        "--study", "study1", "--device", eval_device], cwd=REPO_ROOT, check=True, env=env)


# ---------------------------------------------------------------------------
# Rules (pure functions of the written files)
# ---------------------------------------------------------------------------

def load_arms(out_dir):
    arms = {}
    for arm in ARMS + ("nope",):
        f = Path(out_dir) / f"{arm}.json"
        if f.exists():
            arms[arm] = json.loads(f.read_text())
    return arms


def row(arms, arm, seed, feat):
    try:
        return arms[arm]["seeds"][str(seed)]["position"][feat]
    except KeyError:
        raise NotEvaluable(f"missing {arm} seed{seed} {feat}")


def need_arms(arms, names, seeds=SEEDS):
    for a in names:
        for k in seeds:
            row(arms, a, k, "layer3")


def rule_floor(arms, anchor, floor_diag):
    nope = row(arms, "nope", 0, "layer3")
    d1 = floor_diag["D1"]
    diffs = d1["max_abs_diff"]
    equiv = bool(d1["pass"]) and all(diffs[k] <= EQUIV_TOL for k in ("layer1", "layer2", "layer3", "memory"))
    bound = anchor["anchor_accuracy"] + FLOOR_BAND
    return {"equivariance_max_abs_diff": diffs, "equivariance_pass": equiv,
            "nope_layer3_accuracy": nope["accuracy"], "anchor_accuracy": anchor["anchor_accuracy"],
            "bound": bound, "accuracy_pass": bool(nope["accuracy"] <= bound), "pass": bool(equiv and nope["accuracy"] <= bound),
            "as_pre_registered_0.053": {"pass": bool(abs(nope["accuracy"] - PREREG_REF) <= FLOOR_BAND),
                                        "delta": nope["accuracy"] - PREREG_REF}}


def rule_injection(arms, arm, ref):
    need_arms(arms, [arm])
    lbs = {str(k): row(arms, arm, k, "layer1")[REFS[ref]]["ci"][0] for k in SEEDS}
    return {"lower_bounds": lbs, "injected": bool(all(v > MARGIN for v in lbs.values()))}


def rule_survives(arms, arm, ref):
    need_arms(arms, [arm])
    lbs = {str(k): row(arms, arm, k, "layer3")[REFS[ref]]["ci"][0] for k in SEEDS}
    return {"lower_bounds": lbs, "survives": bool(all(v > MARGIN for v in lbs.values()))}


def seed_stats(arms, arm, k):
    return as_stats(row(arms, arm, k, "layer3")["stats"])


def rule_p1(arms):
    need_arms(arms, ("learned", "sinusoidal"))
    per = {}
    for k in SEEDS:
        sel = boot_sel(npr.BOOT_SEED + k, len(seed_stats(arms, "learned", k)["n"]))
        diff = acc_boot(seed_stats(arms, "learned", k), sel) - acc_boot(seed_stats(arms, "sinusoidal", k), sel)
        lo, hi = interval(diff)
        per[str(k)] = {"ci": [lo, hi], "includes_zero": bool(lo <= 0 <= hi)}
    return {"per_seed": per, "holds": bool(sum(v["includes_zero"] for v in per.values()) >= 2)}


def cross_arm_stat(arms, reinjecting, k):
    """min over the re-injecting arms minus max over {sinusoidal, learned}, min and max taken inside each resample.
    This specifies what the design left open (it names the statistic but not the resampling scheme); it is not a
    quotation of the design."""
    names = list(reinjecting) + ["sinusoidal", "learned"]
    need_arms(arms, names, (k,))
    n = len(seed_stats(arms, "sinusoidal", k)["n"])
    sel = boot_sel(npr.BOOT_SEED + k, n)
    acc = {a: acc_boot(seed_stats(arms, a, k), sel) for a in names}
    lo_re = np.minimum.reduce([acc[a] for a in reinjecting])
    hi_once = np.maximum(acc["sinusoidal"], acc["learned"])
    return interval(lo_re - hi_once)


def rule_p2_trigger(arms, injected):
    if not injected:
        return {"evaluated": False, "reason": "no re-injecting arm injected"}
    per = {str(k): cross_arm_stat(arms, injected, k) for k in SEEDS}
    return {"evaluated": True, "reinjecting_arms_used": list(injected), "per_seed_ci": per,
            "p2_holds": bool(all(v[0] > 0 for v in per.values())),
            "trigger_fires": bool(all(v[0] > MARGIN for v in per.values()))}


def load_test_flags(path):
    d = json.loads(Path(path).read_text())
    flags = d["per_example_correct"]
    assert len(flags) == 1000 and d["n"] == 1000, f"{path}: {len(flags)} entries"
    return d, np.asarray(flags, dtype=np.float64)


def rule_id_gap(out_dir, arm):
    per = {}
    for k in SEEDS:
        try:
            _, a = load_test_flags(run_dir_for(arm, k) / "test_results.json")
            _, b = load_test_flags(Path(out_dir) / "sinusoidal_reeval" / f"seed{k}" / "test_results.json")
        except (FileNotFoundError, AssertionError) as e:
            raise NotEvaluable(f"{arm} seed{k}: {e}")
        sel = boot_sel(npr.BOOT_SEED + k, 1000)
        diff = a[sel].mean(-1) - b[sel].mean(-1)
        lo, hi = interval(diff)
        per[str(k)] = {"arm_acc": float(a.mean() * 100), "sinusoidal_reeval_acc": float(b.mean() * 100),
                       "diff_points": float((a.mean() - b.mean()) * 100), "ci": [lo, hi]}
    up = all(v["ci"][0] > 0 for v in per.values())
    down = all(v["ci"][1] < 0 for v in per.values())
    return {"per_seed": per, "gap_moves": bool(up or down), "direction": "up" if up else ("down" if down else None)}


def rule_depth(arms, arm):
    flags = {}
    for k in SEEDS:
        try:
            flags[str(k)] = arms[arm]["seeds"][str(k)]["depth_memory"]["depth_code_appears"]
        except KeyError:
            raise NotEvaluable(f"missing {arm} seed{k} depth_memory")
    return {"per_seed": flags, "depth_code_appears": bool(all(flags.values()))}


def kill_gate(survive, gap):
    """survive: {arm: bool}; gap: {arm: bool or None}."""
    survivors = [a for a, v in survive.items() if v]
    if not survivors:
        return {"verdict": "no arm satisfies 'position survives': the loss is architectural, not encoding-specific",
                "survivors": []}
    parts = []
    for a in survivors:
        g = gap.get(a)
        if a == "sinusoidal":
            parts.append("sinusoidal: position survives (the reference arm for the ID gap)")
        elif g is None:
            parts.append(f"{a}: position survives; ID gap not evaluable")
        elif g:
            parts.append(f"{a}: position survives and the ID gap moves")
        else:
            parts.append(f"{a}: position survives but the ID gap does not move: the mechanism is encoding-specific "
                         "and the behavioral gap is not")
    return {"verdict": "; ".join(parts), "survivors": survivors}


def guarded(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except NotEvaluable as e:
        return {"evaluable": False, "reason": str(e)}


def evaluate_rules(arms, anchor, floor_diag, out_dir):
    rules = {"anchor": {"name": anchor["anchor_name"], "accuracy": anchor["anchor_accuracy"]}}
    rules["floor_a"] = guarded(rule_floor, arms, anchor, floor_diag)
    inj, surv, p2 = {}, {}, {}
    for ref in REFS:
        inj[ref] = {a: guarded(rule_injection, arms, a, ref) for a in ("rope", "alibi")}
        surv[ref] = {a: guarded(rule_survives, arms, a, ref) for a in ARMS}
        injected = [a for a in ("rope", "alibi") if inj[ref][a].get("injected")]
        evaluable = all("injected" in inj[ref][a] for a in ("rope", "alibi"))
        p2[ref] = (guarded(rule_p2_trigger, arms, injected) if evaluable
                   else {"evaluated": False, "reason": "injection not evaluable"})
    rules["injection_b"], rules["survives_c"], rules["p2_trigger_ef"] = inj, surv, p2
    rules["p1_d"] = guarded(rule_p1, arms)
    gap = {a: guarded(rule_id_gap, out_dir, a) for a in ("learned", "rope", "alibi")}
    rules["id_gap_g"] = gap
    rules["depth_h"] = {a: guarded(rule_depth, arms, a) for a in ARMS}
    kg = {}
    for ref in REFS:
        survive = {a: surv[ref][a].get("survives") for a in ARMS}
        if any(v is None for v in survive.values()):
            kg[ref] = {"evaluable": False, "reason": "survives not evaluable for every arm"}
        else:
            kg[ref] = kill_gate(survive, {a: gap[a].get("gap_moves") for a in gap})
    rules["kill_gate_i"] = kg
    rules["reference_text"] = REF_TEXT
    return rules


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------

def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def ci_text(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def verdict_word(d, key):
    return "not evaluable (" + d["reason"] + ")" if d.get("evaluable") is False else str(d[key])


def build_summary(arms, anchor, rules, out_dir, meta):
    L = []
    L += [f"Design: `{DESIGN}` (pre-registration {DESIGN_COMMIT}).",
          "Deviations entries applied: " + "; ".join(f"`{d}`" for d in DEVIATIONS_APPLIED) + ".", "",
          "# Positional-encoding sweep: probe battery", "",
          f"Run: {meta.get('started_at_utc')} on {meta.get('gpu')} pod image `{meta.get('pod_image')}`, torch "
          f"{meta.get('torch')}, scikit-learn {meta.get('sklearn')}, probes on {meta.get('probe_device')}, test "
          f"re-evaluation on {meta.get('eval_device')}, code commit `{meta.get('git_commit')}`.", "",
          "The committed paper numbers (experiments/results/analysis_v2/nonlinear_probe/, "
          "results_v2/test_set_results.json) are unchanged by this run.", "",
          "## ANCHOR (Deviation 2026-10-28)", "",
          "| Baseline | Position accuracy |", "|---|---|"]
    for k, v in anchor["baselines"].items():
        L.append(f"| {k} | {fmt(v['accuracy'])} |")
    L += ["", f"ANCHOR = {anchor['anchor_name']} = {fmt(anchor['anchor_accuracy'])} (the highest of the three, chosen "
          f"mechanically). Majority {fmt(anchor['majority'])}. The 0.053 appears only in the \"as pre-registered\" "
          "columns.", ""]

    L += ["## Decision rules", "",
          "P2 specification note: the design names the P2 and trigger statistic (min over the re-injecting arms minus "
          "max over {sinusoidal, learned}) but not how min and max meet the resampling. This run takes min and max "
          "inside each resample (same 150 expressions drawn for every arm). That specifies what the design left "
          "open; it is not a quotation of the design.", ""]
    fl = rules["floor_a"]
    if fl.get("evaluable") is False:
        L.append(f"- **a) Floor:** not evaluable ({fl['reason']}).")
    else:
        L.append(f"- **a) Floor (check):** equivariance max diff {max(fl['equivariance_max_abs_diff'].values()):.2e} "
                 f"(pass {fl['equivariance_pass']}); NoPE layer-3 {fmt(fl['nope_layer3_accuracy'])} <= ANCHOR + "
                 f"{FLOOR_BAND} = {fmt(fl['bound'])}: {fl['accuracy_pass']}. **{'PASS' if fl['pass'] else 'FAIL'}.** "
                 f"As pre-registered (0.053 +/- {FLOOR_BAND}): {'pass' if fl['as_pre_registered_0.053']['pass'] else 'fail'} "
                 f"(delta {fl['as_pre_registered_0.053']['delta']:+.4f}).")
    for ref, label in (("anchor", "ANCHOR"), ("prereg", "as pre-registered (0.053)")):
        L += ["", f"### Under {label}", ""]
        for a in ("rope", "alibi"):
            d = rules["injection_b"][ref][a]
            L.append(f"- **b) Injection, {a}, layer 1:** " + (verdict_word(d, "injected") if d.get("evaluable") is False
                     else f"lower bounds {[round(v, 4) for v in d['lower_bounds'].values()]} -> "
                          f"{'injected' if d['injected'] else 'not injected'}"))
        for a in ARMS:
            d = rules["survives_c"][ref][a]
            L.append(f"- **c) Survives, {a}, layer 3:** " + (verdict_word(d, "survives") if d.get("evaluable") is False
                     else f"lower bounds {[round(v, 4) for v in d['lower_bounds'].values()]} -> {d['survives']}"))
        d = rules["p2_trigger_ef"][ref]
        if d.get("evaluated") is False or d.get("evaluable") is False:
            L.append(f"- **e/f) P2 and trigger:** not evaluated ({d.get('reason')}).")
        else:
            L.append(f"- **e) P2** (re-injecting arms used: {d['reinjecting_arms_used']}): per-seed CIs "
                     f"{[ci_text(v) for v in d['per_seed_ci'].values()]} -> {d['p2_holds']}")
            L.append(f"- **f) Trigger** (lower bound > +{MARGIN} on all seeds): {d['trigger_fires']}")
        kg = rules["kill_gate_i"][ref]
        L.append("- **i) Kill gate:** " + (verdict_word(kg, "verdict") if kg.get("evaluable") is False else kg["verdict"]))
    L += ["", "### Reference-free rules", ""]
    d = rules["p1_d"]
    L.append("- **d) P1:** " + (verdict_word(d, "holds") if d.get("evaluable") is False else
             f"learned - sinusoidal layer-3 CIs {[ci_text(v['ci']) for v in d['per_seed'].values()]}; includes zero on "
             f"{sum(v['includes_zero'] for v in d['per_seed'].values())} of 3 seeds -> {d['holds']}"))
    for a in ("learned", "rope", "alibi"):
        d = rules["id_gap_g"][a]
        L.append(f"- **g) ID gap, {a}:** " + (verdict_word(d, "gap_moves") if d.get("evaluable") is False else
                 f"diff (points) {[round(v['diff_points'], 2) for v in d['per_seed'].values()]}, CIs "
                 f"{[ci_text(v['ci']) for v in d['per_seed'].values()]} -> gap moves {d['gap_moves']}"
                 f"{' (' + d['direction'] + ')' if d['direction'] else ''}"))
    for a in ARMS:
        d = rules["depth_h"][a]
        L.append(f"- **h) Depth code at memory, {a}:** " + (verdict_word(d, "depth_code_appears")
                 if d.get("evaluable") is False else f"per seed {list(d['per_seed'].values())} -> {d['depth_code_appears']}"))

    L += ["", "## Per-seed position tables (accuracy; vs ANCHOR diff [95% CI]; vs 0.053 diff)", ""]
    for arm in ARMS + ("nope",):
        if arm not in arms:
            continue
        for feat in FEATURES:
            L += [f"### {arm}, {feat}" + (" (construction check only)" if feat == "encoder_input" else ""), "",
                  "| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |", "|---|---|---|---|---|---|"]
            for k in sorted(arms[arm]["seeds"], key=int):
                r = arms[arm]["seeds"][k]["position"][feat]
                L.append(f"| {k} | {fmt(r['accuracy'])} | {r['vs_anchor']['diff']:+.4f} | {ci_text(r['vs_anchor']['ci'])} | "
                         f"{r['vs_prereg']['diff']:+.4f} | {ci_text(r['vs_prereg']['ci'])} |")
            L.append("")
    L += ["## Depth at the memory", "",
          "| Arm | Seed | Linear | MLP mean | Control | Majority | vs char+position baseline (lower bound) | Rule base | Depth code |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for k in sorted(arms.get(arm, {}).get("seeds", {}), key=int):
            r = arms[arm]["seeds"][k].get("depth_memory")
            if r:
                lb = min(c[0] for c in r["bootstrap_ci_vs_char_position_baseline"].values())
                L.append(f"| {arm} | {k} | {fmt(r['linear'], 3)} | {fmt(r['mlp_mean'], 3)} | {fmt(r['control'], 3)} | "
                         f"{fmt(r['majority'], 3)} | {lb:+.3f} | {r['verdict_rule_base']} | {r['depth_code_appears']} |")
    L += ["", "## In-distribution test accuracy (Study 1, 1,000 examples, autoregressive greedy, exact match)", ""]
    L += test_table(out_dir)
    L += ["", "## For the paper sentence: sinusoidal layer-3 position minus ANCHOR", "",
          "| Seed | Layer-3 accuracy | minus ANCHOR | 95% CI |", "|---|---|---|---|"]
    for k in sorted(arms.get("sinusoidal", {}).get("seeds", {}), key=int):
        r = arms["sinusoidal"]["seeds"][k]["position"]["layer3"]
        L.append(f"| {k} | {fmt(r['accuracy'])} | {r['vs_anchor']['diff']:+.4f} | {ci_text(r['vs_anchor']['ci'])} |")
    return "\n".join(L) + "\n"


def test_table(out_dir):
    L = ["| Arm | Seed | Test accuracy (%) | Source |", "|---|---|---|---|"]
    committed = json.loads((REPO_ROOT / "results_v2/test_set_results.json").read_text())
    for k in SEEDS:
        p = Path(out_dir) / "sinusoidal_reeval" / f"seed{k}" / "test_results.json"
        re_acc = json.loads(p.read_text())["test_accuracy"] if p.exists() else None
        c = committed[f"transformer_study1_seed{k}"]["test_accuracy"]
        L.append(f"| sinusoidal (re-evaluated, used by the rule) | {k} | {fmt(re_acc, 2)} | this run, pod, cuda |")
        L.append(f"| sinusoidal (committed) | {k} | {fmt(c, 2)} | results_v2/test_set_results.json (torch 2.13.0, CPU); "
                 "shown only, the ID-gap rule used the re-evaluated value |")
    for arm in ("learned", "rope", "alibi"):
        for k in SEEDS:
            p = run_dir_for(arm, k) / "test_results.json"
            acc = json.loads(p.read_text())["test_accuracy"] if p.exists() else None
            L.append(f"| {arm} | {k} | {fmt(acc, 2)} | run dir test_results.json (runner, cuda) |")
    return L


# ---------------------------------------------------------------------------
# Schema checks (used by the tests on every new JSON)
# ---------------------------------------------------------------------------

def _has(d, keys, where):
    missing = [k for k in keys if k not in d]
    assert not missing, f"{where}: missing {missing}"


def validate_stats(st, where, n_expr=150):
    _has(st, ("n", "correct"), where)
    assert len(st["n"]) == len(st["correct"]) == n_expr, where
    assert all(0 <= c <= n for c, n in zip(st["correct"], st["n"])), where


def validate_position_row(r, where, n_expr=150):
    _has(r, ("accuracy", "majority", "n_classes_train", "convergence_warning", "dim", "stats", "vs_anchor", "vs_prereg"), where)
    assert 0 <= r["accuracy"] <= 1
    validate_stats(r["stats"], where, n_expr)
    for ref in ("vs_anchor", "vs_prereg"):
        _has(r[ref], ("diff", "ci"), where)
        assert len(r[ref]["ci"]) == 2 and r[ref]["ci"][0] <= r[ref]["ci"][1], where


def validate_arm_doc(doc, n_expr=150):
    _has(doc, ("arm", "positional_encoding", "split", "features", "seeds"), "arm doc")
    assert tuple(doc["features"]) == tuple(FEATURES)
    for k, s in doc["seeds"].items():
        _has(s, ("position", "checkpoint"), f"{doc['arm']} seed{k}")
        _has(s["checkpoint"], ("source", "sha256"), f"{doc['arm']} seed{k} checkpoint")
        for f in FEATURES:
            validate_position_row(s["position"][f], f"{doc['arm']} seed{k} {f}", n_expr)
        if "depth_memory" in s:
            d = s["depth_memory"]
            _has(d, ("linear", "mlp_mean", "mlp_std", "control", "majority", "bootstrap_ci", "bootstrap_ci_vs_majority",
                     "bootstrap_ci_vs_char_position_baseline", "depth_code_appears", "verdict_rule_base"),
                 f"{doc['arm']} seed{k} depth_memory")


def validate_anchor(doc, n_expr=150):
    _has(doc, ("baselines", "anchor_name", "anchor_accuracy", "expected_baselines", "majority", "n_expressions"), "anchor")
    assert set(doc["baselines"]) == {"B1", "B2", "B3"}
    for k, b in doc["baselines"].items():
        _has(b, ("accuracy", "majority", "dim", "stats"), f"anchor {k}")
        validate_stats(b["stats"], f"anchor {k}", n_expr)
    assert doc["anchor_accuracy"] == max(b["accuracy"] for b in doc["baselines"].values())


def validate_rules(doc):
    _has(doc, ("anchor", "floor_a", "injection_b", "survives_c", "p2_trigger_ef", "p1_d", "id_gap_g", "depth_h",
               "kill_gate_i", "reference_text"), "decision_rules")
    for ref in REFS:
        _has(doc["injection_b"], (ref,), "injection_b")
        _has(doc["kill_gate_i"], (ref,), "kill_gate_i")


def validate_environment(doc):
    _has(doc, ("python", "torch", "numpy", "sklearn", "driver_version", "pod_image", "git_commit", "started_at_utc",
               "probe_device", "eval_device"), "environment")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("all", "probes", "reeval", "rules"), default="all")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--eval-device", default="cuda")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-nope", action="store_true")
    ap.add_argument("--anchor-check", choices=("assert", "skip"), default="assert",
                    help="skip exists for smoke runs under a different scikit-learn; refused for the committed out dir")
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()
    if args.anchor_check == "skip" and out_dir == DEFAULT_OUT.resolve():
        ap.error("--anchor-check skip is refused for the committed output directory")
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    env_file = out_dir / "environment.json"
    if args.stage == "rules" and env_file.exists():
        env = json.loads(env_file.read_text())  # keep the probe run's record; this stage only reads files
    else:
        env = environment_doc(args.stage, args.eval_device, sys.argv[1:])
        write_json(env_file, env)  # written at the start of the run

    if args.stage in ("all", "probes"):
        run_probes(args, out_dir)
    if args.stage in ("all", "reeval"):
        run_reeval(out_dir, args.eval_device)
    if args.stage in ("all", "rules"):
        arms = load_arms(out_dir)
        anchor = json.loads((out_dir / "anchor.json").read_text())
        floor_diag = json.loads(FLOOR_DIAG.read_text())
        rules = evaluate_rules(arms, anchor, floor_diag, out_dir)
        write_json(out_dir / "decision_rules.json", rules)
        (out_dir / "summary.md").write_text(build_summary(arms, anchor, rules, out_dir, env))
        print(f"wrote decision_rules.json and summary.md in {out_dir}", flush=True)
    env["finished_at_utc"] = now_utc()
    env["wall_seconds"] = round(time.time() - t0, 1)
    write_json(out_dir / "environment.json", env)
    print(f"wall time {env['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()

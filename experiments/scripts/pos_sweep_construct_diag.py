"""Construct diagnostic H1-H3 for the positional-encoding sweep.

A diagnostic beside the pre-registered verdicts (summary.md, decision_rules.json and every battery output are
untouched). Tests and readings were fixed before this script was written.

  H1  reversal equivariance, ALiBi seeds 0-2: reverse the real-token block (key padding mask fixed), run both
      orders through the probe hooks, un-reverse the second output; max abs diff per feature, pass <= 1e-5 at
      layers 1-3 and memory. Controls: NoPE (equivariant by construction) and the paper's sinusoidal seed 0.
  H3  RoPE seeds 0-2: the same reversal test plus the NoPE diagnostic's random-permutation test (seed 0).
  H2  relative-offset probe on every checkpoint, layers 1-3: pairs (i, j) inside an expression with
      1 <= |j - i| <= 8, at most 40 per expression (seed 0), the same pairs for every checkpoint; features
      concat(h_i, h_j); targets |j - i| (8 classes) and sign (2 classes); LogisticRegression(max_iter=1000),
      unscaled. References under the same probe and pairs: the NoPE checkpoint (same layer) and an order-blind
      feature baseline one-hot(char_i) + one-hot(char_j) + one-hot(expression length).

  python experiments/scripts/pos_sweep_construct_diag.py --parts all
"""
import argparse
import importlib.util
import json
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


pb = _load("pos_sweep_battery", "experiments/scripts/pos_sweep_battery.py")
npr, diag = pb.npr, pb.diag

OUT = REPO_ROOT / "experiments" / "results" / "analysis_v2" / "pos_sweep" / "construct_diag"
STRAT = npr.STRAT_SPLIT
EQUIV_TOL = 1e-5
EQUIV_FEATURES = diag.EQUIV_FEATURES            # encoder_input, layer1, layer2, layer3, memory
CHECKED = ("layer1", "layer2", "layer3", "memory")  # the features the pass rule applies to
ORDERS_OF_MAGNITUDE = 100.0                      # "fails by orders of magnitude": >= 100 x the tolerance
N_EXPR = diag.N_EQUIV                            # 20
MAX_OFFSET = 8
MAX_PAIRS = 40
PAIR_SEED = 0
H2_LAYERS = ("layer1", "layer2", "layer3")
TARGETS = ("magnitude", "sign")
CARRY_MARGIN = 0.10
H2_ARMS = pb.ARMS  # sinusoidal, learned, rope, alibi (three seeds each); nope (seed 0) is the reference


def ckpt_run_dir(arm, seed):
    return None if arm == "sinusoidal" else pb.run_dir_for(arm, seed)


def load_ckpt(arm, seed):
    return npr.load_model("transformer", "study1", seed, run_dir=ckpt_run_dir(arm, seed))


# ---------------------------------------------------------------------------
# H1 and H3: equivariance
# ---------------------------------------------------------------------------

def reversal_test(model, items):
    """Max abs diff per feature between f(x) and R^-1 f(R x), R reversing the real-token block in place; the
    padding positions (and so the key padding mask) are unchanged."""
    worst = {k: 0.0 for k in EQUIV_FEATURES}
    for item in items:
        src = npr.encode_input(item["input"])
        real = np.flatnonzero(src.numpy() != npr.pad_idx)
        assert np.array_equal(real, np.arange(len(real))), "real tokens are not a prefix"
        src_r = src.clone()
        src_r[real] = src[torch.from_numpy(real[::-1].copy())]
        a = npr.transformer_states_hooked(model, src)
        b = npr.transformer_states_hooked(model, src_r)
        for k in EQUIV_FEATURES:
            worst[k] = max(worst[k], float(np.abs(a[k][real] - b[k][real][::-1]).max()))
    return worst


def verdict_equiv(worst):
    m = max(worst[k] for k in CHECKED)
    return {"max_abs_diff": worst, "max_over_checked": m, "pass": bool(m <= EQUIV_TOL),
            "fails_by_orders_of_magnitude": bool(m >= ORDERS_OF_MAGNITUDE * EQUIV_TOL)}


def run_h1(items, seeds, out_dir):
    doc = {"test": "reversal equivariance", "arm": "alibi", "n_expressions": len(items), "tolerance": EQUIV_TOL,
           "checked_features": list(CHECKED), "expected": "pass", "seeds": {}, "controls": {}}
    for seed in seeds:
        doc["seeds"][str(seed)] = verdict_equiv(reversal_test(load_ckpt("alibi", seed), items))
        print(f"[H1 alibi seed{seed}] max diff {doc['seeds'][str(seed)]['max_over_checked']:.2e} "
              f"pass {doc['seeds'][str(seed)]['pass']}", flush=True)
    for arm, expected in (("nope", "pass"), ("sinusoidal", "fail")):
        v = verdict_equiv(reversal_test(load_ckpt(arm, 0), items))
        v["expected"] = expected
        doc["controls"][f"{arm}_seed0"] = v
        print(f"[H1 control {arm} seed0] max diff {v['max_over_checked']:.2e} (expected {expected})", flush=True)
    doc["held"] = bool(all(v["pass"] for v in doc["seeds"].values()))
    pb.write_json(out_dir / "h1_alibi_reversal.json", doc)
    return doc


def run_h3(items, seeds, out_dir):
    doc = {"arm": "rope", "n_expressions": len(items), "tolerance": EQUIV_TOL, "checked_features": list(CHECKED),
           "orders_of_magnitude_rule": f"max diff >= {ORDERS_OF_MAGNITUDE:g} x tolerance",
           "expected": "encoder_input passes (no position at the input); layers 1-3 and memory fail, on both tests",
           "seeds": {}}
    for seed in seeds:
        model = load_ckpt("rope", seed)
        rev = verdict_equiv(reversal_test(model, items))
        worst, n_identity = diag.equivariance(model, items)
        perm = verdict_equiv(worst)
        perm["n_identity_permutations"] = n_identity
        doc["seeds"][str(seed)] = {"reversal": rev, "permutation": perm}
        print(f"[H3 rope seed{seed}] reversal {rev['max_over_checked']:.2e} permutation "
              f"{perm['max_over_checked']:.2e}", flush=True)
    doc["held"] = h3_held(doc)
    pb.write_json(out_dir / "h3_rope.json", doc)
    return doc


# ---------------------------------------------------------------------------
# H2: relative-offset probe
# ---------------------------------------------------------------------------

def make_pairs(sitems):
    """{split: (i, j, expression offsets)}: all (i, j) with 1 <= |j - i| <= MAX_OFFSET inside each expression, at
    most MAX_PAIRS per expression sampled with PAIR_SEED, train then test in order. Depends only on the
    expressions' lengths, so every checkpoint sees the same pairs."""
    rng = np.random.default_rng(PAIR_SEED)
    out = {}
    for split in ("train", STRAT):
        ii, jj, counts = [], [], []
        for item in sitems[split]:
            L = len(item["input"])
            pairs = [(i, j) for i in range(L) for j in range(L) if 1 <= abs(j - i) <= MAX_OFFSET]
            if len(pairs) > MAX_PAIRS:
                pairs = [pairs[k] for k in np.sort(rng.choice(len(pairs), MAX_PAIRS, replace=False))]
            ii += [p[0] for p in pairs]
            jj += [p[1] for p in pairs]
            counts.append(len(pairs))
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        out[split] = (np.asarray(ii), np.asarray(jj), offsets)
    return out


def pair_targets(ii, jj):
    return {"magnitude": np.abs(jj - ii), "sign": (jj > ii).astype(np.int64)}


def expression_rows(sitems, split):
    """Row offset of each expression in layerwise_acts order and the expression index of each pair row."""
    lengths = [len(it["input"]) for it in sitems[split]]
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def pair_feature_rows(X, row_off, pairs):
    ii, jj, poff = pairs
    expr = np.repeat(np.arange(len(poff) - 1), np.diff(poff))
    return np.concatenate([X[row_off[expr] + ii], X[row_off[expr] + jj]], axis=1)


BASELINE_KINDS = ("B1p", "B2p", "B3p")


def baseline_pair_features(sitems, split, pairs, kind="B2p"):
    """Order-blind pair features: B1p one-hot(char_i) + one-hot(char_j); B2p B1p + one-hot(expression length);
    B3p B2p + counts of each vocabulary character in the expression."""
    V, L = npr.tokenizer.vocab_size, npr.MAX_INPUT_LEN
    ii, jj, poff = pairs
    dim = {"B1p": 2 * V, "B2p": 2 * V + L + 1, "B3p": 2 * V + L + 1 + V}[kind]
    rows = []
    for e, item in enumerate(sitems[split]):
        ids = npr.encode_checked(item["input"], L)
        counts = np.bincount(ids, minlength=V).astype(np.float32)
        for k in range(poff[e], poff[e + 1]):
            v = np.zeros(dim, dtype=np.float32)
            v[ids[ii[k]]] = 1.0
            v[V + ids[jj[k]]] = 1.0
            if kind in ("B2p", "B3p"):
                v[2 * V + len(item["input"])] = 1.0
            if kind == "B3p":
                v[2 * V + L + 1:] = counts
            rows.append(v)
    return np.array(rows)


def fit_probe(Xtr, ytr, Xte, yte, poff_te):
    from sklearn.linear_model import LogisticRegression
    import warnings
    probe = LogisticRegression(max_iter=1000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        probe.fit(Xtr, ytr)
    yhat = probe.predict(Xte)
    st = npr.example_stats("depth", yte, yhat, poff_te)
    return {"accuracy": float(np.mean(yhat == yte)), "majority": npr.majority_baseline(yte),
            "n_classes_train": int(len(set(ytr.tolist()))),
            "convergence_warning": any("converge" in str(x.message).lower() for x in w),
            "stats": pb.stats_json(st)}


def probe_checkpoint(model, sitems, pairs):
    feats = npr.layerwise_acts(model, sitems, features=H2_LAYERS, with_ids=False)
    roff = {s: expression_rows(sitems, s) for s in ("train", STRAT)}
    ytr, yte = pair_targets(*pairs["train"][:2]), pair_targets(*pairs[STRAT][:2])
    out = {}
    for layer in H2_LAYERS:
        Xtr = pair_feature_rows(feats[layer]["train"]["X"], roff["train"], pairs["train"])
        Xte = pair_feature_rows(feats[layer][STRAT]["X"], roff[STRAT], pairs[STRAT])
        out[layer] = {t: fit_probe(Xtr, ytr[t], Xte, yte[t], pairs[STRAT][2]) for t in TARGETS}
        out[layer]["dim"] = int(Xtr.shape[1])
    return out


def feature_baselines(sitems, pairs):
    ytr, yte = pair_targets(*pairs["train"][:2]), pair_targets(*pairs[STRAT][:2])
    out = {}
    for kind in BASELINE_KINDS:
        Xtr = baseline_pair_features(sitems, "train", pairs["train"], kind)
        Xte = baseline_pair_features(sitems, STRAT, pairs[STRAT], kind)
        out[kind] = {t: fit_probe(Xtr, ytr[t], Xte, yte[t], pairs[STRAT][2]) for t in TARGETS}
        out[kind]["dim"] = int(Xtr.shape[1])
    return out


REFERENCE_ORDER = ("NoPE",) + BASELINE_KINDS


def reading(arm_stats_by_seed, refs):
    """refs: {name: {"accuracy", "stats"}} for NoPE (same layer) and B1p-B3p. The reference is the one with the
    highest point accuracy, chosen mechanically (first in REFERENCE_ORDER on a tie). "Carries relative position"
    if the bootstrap lower bound on (arm - chosen reference), paired by expression, exceeds +0.10 on all seeds.
    Intervals against every reference are reported beside it."""
    chosen = max(REFERENCE_ORDER, key=lambda k: refs[k]["accuracy"])
    per = {}
    for seed, st in arm_stats_by_seed.items():
        st = pb.as_stats(st)
        sel = pb.boot_sel(npr.BOOT_SEED + int(seed), len(st["n"]))
        a = pb.acc_boot(st, sel)
        cis = {k: pb.interval(a - pb.acc_boot(pb.as_stats(refs[k]["stats"]), sel)) for k in REFERENCE_ORDER}
        per[str(seed)] = {"ci_vs_chosen": cis[chosen], "ci_vs_each": cis}
    lbs = [v["ci_vs_chosen"][0] for v in per.values()]
    return {"chosen_reference": chosen, "reference_accuracies": {k: refs[k]["accuracy"] for k in REFERENCE_ORDER},
            "per_seed": per, "carries_relative_position": bool(all(lb > CARRY_MARGIN for lb in lbs))}


def run_h2(sitems, arms, seeds, out_dir):
    pairs = make_pairs(sitems)
    doc = {"definition": {"pairs": f"1 <= |j - i| <= {MAX_OFFSET}, at most {MAX_PAIRS} per expression, seed {PAIR_SEED}, "
                                   "same pairs for every checkpoint",
                          "n_pairs": {s: int(pairs[s][2][-1]) for s in ("train", STRAT)},
                          "targets": {"magnitude": "|j - i| (8 classes)", "sign": "sign of j - i (2 classes)"},
                          "probe": "LogisticRegression(max_iter=1000), unscaled",
                          "split": "pairs never cross expressions: training pairs come from the 300 training "
                                   "expressions of the stratified split, test pairs from its 150 test expressions; "
                                   "the bootstrap resamples test expressions",
                          "carries_rule": CARRIES_RULE},
           "feature_baselines": None, "checkpoints": {}, "readings": {}}
    doc["feature_baselines"] = feature_baselines(sitems, pairs)
    todo = [(a, s) for a in arms for s in seeds] + [("nope", 0)]
    for arm, seed in todo:
        t0 = time.time()
        model = load_ckpt(arm, seed)
        doc["checkpoints"][f"{arm}_seed{seed}"] = probe_checkpoint(model, sitems, pairs)
        r = doc["checkpoints"][f"{arm}_seed{seed}"]
        print(f"[H2 {arm} seed{seed}] " + "  ".join(f"{l}: mag {r[l]['magnitude']['accuracy']:.3f} sign "
              f"{r[l]['sign']['accuracy']:.3f}" for l in H2_LAYERS) + f"  ({time.time() - t0:.0f}s)", flush=True)
        pb.write_json(out_dir / "h2_relative_offset.json", doc)
    compute_readings(doc, arms)
    pb.write_json(out_dir / "h2_relative_offset.json", doc)
    return doc


CARRIES_RULE = (f"reference = the highest point accuracy among NoPE (same layer; one seed, the reference for every "
                f"arm seed), B1p, B2p and B3p, chosen mechanically; carries if the lower bound of (arm - reference), "
                f"paired by expression, exceeds +{CARRY_MARGIN} on all three seeds")


def compute_readings(doc, arms):
    for arm in arms:
        doc["readings"][arm] = {}
        for layer in H2_LAYERS:
            doc["readings"][arm][layer] = {}
            for t in TARGETS:
                seeds = {k.split("seed")[1]: v[layer][t]["stats"] for k, v in doc["checkpoints"].items()
                         if k.startswith(arm + "_seed")}
                refs = {"NoPE": doc["checkpoints"]["nope_seed0"][layer][t]}
                refs.update({k: doc["feature_baselines"][k][t] for k in BASELINE_KINDS})
                doc["readings"][arm][layer][t] = reading(seeds, refs)


def amend_h2(doc, sitems, arms):
    """Amended reference (all of NoPE, B1p, B2p, B3p) applied to an H2 run made with the original reference
    (NoPE and B2p only). The checkpoint probes and their per-expression statistics are reused unchanged; the
    order-blind baselines are recomputed on the same pairs and the readings recomputed."""
    pairs = make_pairs(sitems)
    assert int(pairs[STRAT][2][-1]) == doc["definition"]["n_pairs"][STRAT], "pairs differ from the probe run"
    original = doc.pop("feature_baseline", None)
    doc.pop("feature_baselines", None)
    doc["feature_baselines"] = feature_baselines(sitems, pairs)
    if original is not None:
        same = all(original[t]["accuracy"] == doc["feature_baselines"]["B2p"][t]["accuracy"] for t in TARGETS)
        assert same, "B2p recomputed differs from the probe run's feature baseline"
    doc["definition"]["carries_rule"] = CARRIES_RULE
    doc["definition"]["split"] = ("pairs never cross expressions: training pairs come from the 300 training "
                                  "expressions of the stratified split, test pairs from its 150 test expressions; "
                                  "the bootstrap resamples test expressions")
    doc["readings"] = {}
    compute_readings(doc, [a for a in arms if any(k.startswith(a + "_seed") for k in doc["checkpoints"])])
    doc["amendment"] = {"applied_at_utc": pb.now_utc(), "probes_reused": original is not None,
                        "text": "H2 order-blind reference = the highest of NoPE, B1p, B2p, B3p, chosen mechanically, all "
                                "on the same pairs; received after the probe run started and before its readings "
                                "were computed; readings recomputed from the run's per-expression statistics"}
    return doc


def h3_held(doc):
    """Amended H3 reading: encoder_input passes (no position at the input); each of layers 1-3 and memory fails,
    on both tests and every seed."""
    ok = True
    for s in doc["seeds"].values():
        for name in ("reversal", "permutation"):
            d = s[name]["max_abs_diff"]
            ok &= d["encoder_input"] <= EQUIV_TOL and all(d[k] > EQUIV_TOL for k in CHECKED)
    return bool(ok)


# ---------------------------------------------------------------------------
# Pre-stated predictions and the report
# ---------------------------------------------------------------------------

def carries(h2, arm, layer, target):
    return h2["readings"][arm][layer][target]["carries_relative_position"]


def predictions(h1, h3, h2):
    """Mechanical marks. Per-layer readings are in the tables; the aggregation across layers (a prediction holds
    only if it holds at every layer 1-3) is a choice made when this script was written, before the pod run."""
    p = {}
    p["H1: ALiBi is reversal-equivariant (pass on all seeds)"] = h1["held"]
    p["H3: RoPE encoder_input passes; layers 1-3 and memory fail, reversal and permutation (all seeds)"] = h3["held"]
    if "rope" in h2["readings"]:
        p["H2: RoPE carries magnitude and sign (every layer 1-3)"] = bool(all(
            carries(h2, "rope", l, t) for l in H2_LAYERS for t in TARGETS))
    if "alibi" in h2["readings"]:
        p["H2: ALiBi carries magnitude and its sign stays near the references (every layer 1-3)"] = bool(all(
            carries(h2, "alibi", l, "magnitude") and not carries(h2, "alibi", l, "sign") for l in H2_LAYERS))
    return p


def h2_finding(h2):
    """Amendment 3: if neither RoPE nor ALiBi clears the reference at any layer or target, that is a finding (no
    linearly decodable relative offset), not a failure of the construct reading."""
    if not ("rope" in h2["readings"] and "alibi" in h2["readings"]):
        return None
    clears = [(a, l, t) for a in ("rope", "alibi") for l in H2_LAYERS for t in TARGETS if carries(h2, a, l, t)]
    if not clears:
        return ("Finding: neither RoPE nor ALiBi carries a linearly decodable relative offset at layers 1-3 "
                "(no reading clears the reference).")
    return "RoPE or ALiBi clears the reference at: " + ", ".join(f"{a} {l} {t}" for a, l, t in clears) + "."


def md_report(h1, h3, h2, env):
    L = ["This is a construct diagnostic for the positional-encoding sweep (hypotheses H1 to H3). It sits beside the "
         "pre-registered verdicts: summary.md, decision_rules.json and every battery output are unchanged by it.",
         "",
         "The hypotheses came from a Mac smoke run of the battery code (scikit-learn 1.6.1, scratch output "
         "directory, anchor check skipped), whose layer-1 injection result for RoPE and ALiBi was reported before "
         "the pod battery ran; the pod battery then gave the same result. The tests and readings below were fixed "
         "before this script was written.", "",
         f"Run: {env.get('started_at_utc')} on {env.get('gpu')} pod image `{env.get('pod_image')}`, torch "
         f"{env.get('torch')}, scikit-learn {env.get('sklearn')}, probes on {env.get('probe_device')}, code commit "
         f"`{env.get('git_commit')}`.", "",
         "## H1: ALiBi, reversal equivariance (20 stratified test expressions)", "",
         "| Checkpoint | encoder_input | layer1 | layer2 | layer3 | memory | Max (layers 1-3, memory) | Pass (<= 1e-5) |",
         "|---|---|---|---|---|---|---|---|"]

    def row(name, v):
        d = v["max_abs_diff"]
        return (f"| {name} | " + " | ".join(f"{d[k]:.2e}" for k in EQUIV_FEATURES) +
                f" | {v['max_over_checked']:.2e} | {v['pass']} |")

    for k, v in h1["seeds"].items():
        L.append(row(f"alibi seed{k}", v))
    for k, v in h1["controls"].items():
        L.append(row(f"control {k} (expected {v['expected']})", v))
    L += ["", "## H3: RoPE, reversal and random permutation", "",
          "Expected (amended reading): encoder_input passes, since RoPE adds no position at the input; layers 1-3 and "
          "memory fail. \"Fails by orders of magnitude\" means the largest of layers 1-3 and memory is at least "
          f"{ORDERS_OF_MAGNITUDE:g} times the tolerance.", "",
          "| Checkpoint | Test | encoder_input | layer1 | layer2 | layer3 | memory | Fails by orders of magnitude |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in h3["seeds"].items():
        for name in ("reversal", "permutation"):
            v = s[name]
            d = v["max_abs_diff"]
            L.append(f"| rope seed{k} | {name} | {d['encoder_input']:.2e} | {d['layer1']:.2e} | {d['layer2']:.2e} | "
                     f"{d['layer3']:.2e} | {d['memory']:.2e} | {v['fails_by_orders_of_magnitude']} |")
    L += ["", f"## H2: relative-offset probe (pairs: {h2['definition']['n_pairs']})", "",
          "Split: " + h2["definition"]["split"] + ".", "",
          "Reference (amended, applied before the readings were computed): the highest point accuracy among NoPE at "
          "the same layer and the order-blind pair baselines B1p (one-hot char_i + one-hot char_j), B2p (B1p + one-hot "
          "expression length) and B3p (B2p + character counts), chosen mechanically. The probe run itself used NoPE "
          "and B2p only; its checkpoint statistics are reused unchanged and the readings were recomputed. Before the "
          "amendment was received, H2 point accuracies for sinusoidal seeds 0-2 and learned seed 0 had been printed "
          "in the pod run log, and a Mac test run (scikit-learn 1.6.1) had printed ALiBi seed 0's table.", "",
          f"A reading is \"carries relative position\" if the lower bound of (arm - chosen reference), paired by "
          f"expression, exceeds +{CARRY_MARGIN} on all three seeds.", "",
          "| Arm | Layer | Target | Accuracy by seed | NoPE | B1p | B2p | B3p | Chosen | Lower bound by seed | Carries |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in h2["readings"]:
        for layer in H2_LAYERS:
            for t in TARGETS:
                accs = [h2["checkpoints"][f"{arm}_seed{s}"][layer][t]["accuracy"] for s in (0, 1, 2)
                        if f"{arm}_seed{s}" in h2["checkpoints"]]
                r = h2["readings"][arm][layer][t]
                ra = r["reference_accuracies"]
                lbs = [v["ci_vs_chosen"][0] for v in r["per_seed"].values()]
                L.append(f"| {arm} | {layer} | {t} | {', '.join(f'{a:.3f}' for a in accs)} | " +
                         " | ".join(f"{ra[k]:.3f}" for k in REFERENCE_ORDER) +
                         f" | {r['chosen_reference']} | {', '.join(f'{x:+.3f}' for x in lbs)} | "
                         f"{r['carries_relative_position']} |")
    maj = {t: h2['checkpoints']['nope_seed0']['layer1'][t]['majority'] for t in TARGETS}
    L += ["", f"Majority rate in the test pairs: magnitude {maj['magnitude']:.3f}, sign {maj['sign']:.3f}.", ""]
    finding = h2_finding(h2)
    if finding:
        L += [finding, ""]
    L += ["## Pre-stated predictions", "",
          "A prediction holds only if it holds at every layer 1 to 3 (an aggregation fixed in the script before the "
          "run); the per-layer readings are in the H2 table. If neither RoPE nor ALiBi clears the reference, that is "
          "reported above as a finding (no linearly decodable relative offset), not as a failure.", "",
          "| Prediction | Held |", "|---|---|"]
    for k, v in predictions(h1, h3, h2).items():
        L.append(f"| {k} | {'held' if v else 'not held'} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def validate_h1(doc):
    pb._has(doc, ("test", "arm", "n_expressions", "tolerance", "seeds", "controls", "held"), "h1")
    for k, v in list(doc["seeds"].items()) + list(doc["controls"].items()):
        pb._has(v, ("max_abs_diff", "max_over_checked", "pass", "fails_by_orders_of_magnitude"), f"h1 {k}")
        assert set(v["max_abs_diff"]) == set(EQUIV_FEATURES)


def validate_h3(doc):
    pb._has(doc, ("arm", "n_expressions", "seeds", "held"), "h3")
    for k, s in doc["seeds"].items():
        for name in ("reversal", "permutation"):
            pb._has(s[name], ("max_abs_diff", "max_over_checked", "pass", "fails_by_orders_of_magnitude"), f"h3 {k} {name}")
        pb._has(s["permutation"], ("n_identity_permutations",), f"h3 {k}")


def validate_h2(doc, n_expr=150):
    pb._has(doc, ("definition", "feature_baselines", "checkpoints", "readings"), "h2")
    pb._has(doc["definition"], ("pairs", "n_pairs", "split", "carries_rule"), "h2 definition")
    assert "nope_seed0" in doc["checkpoints"] and "feature_baseline" not in doc
    for kind in BASELINE_KINDS:
        for t in TARGETS:
            pb._has(doc["feature_baselines"][kind][t], ("accuracy", "majority", "stats"), f"h2 {kind} {t}")
            pb.validate_stats(doc["feature_baselines"][kind][t]["stats"], f"h2 {kind} {t}", n_expr)
    for name, ck in doc["checkpoints"].items():
        for layer in H2_LAYERS:
            for t in TARGETS:
                pb._has(ck[layer][t], ("accuracy", "majority", "n_classes_train", "stats"), f"h2 {name} {layer} {t}")
                pb.validate_stats(ck[layer][t]["stats"], f"h2 {name} {layer} {t}", n_expr)
    for arm, layers in doc["readings"].items():
        for layer in H2_LAYERS:
            for t in TARGETS:
                r = layers[layer][t]
                pb._has(r, ("chosen_reference", "reference_accuracies", "per_seed", "carries_relative_position"),
                        f"h2 reading {arm} {layer} {t}")
                assert r["chosen_reference"] == max(REFERENCE_ORDER, key=lambda k: r["reference_accuracies"][k])
                for v in r["per_seed"].values():
                    pb._has(v, ("ci_vs_chosen", "ci_vs_each"), f"h2 reading {arm}")
                    assert set(v["ci_vs_each"]) == set(REFERENCE_ORDER)
                    assert v["ci_vs_chosen"] == v["ci_vs_each"][r["chosen_reference"]]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", choices=("all", "h1", "h3", "h2", "amend", "report"), default=["all"],
                    help="amend: apply the amended H2 reference and H3 reading to existing outputs, without re-probing")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--arms", nargs="+", default=list(H2_ARMS), choices=H2_ARMS, help="arms for H2")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(pb.SEEDS))
    args = ap.parse_args()
    parts = {"h1", "h3", "h2", "report"} if "all" in args.parts else set(args.parts)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env_file = out_dir / "environment.json"
    if parts <= {"amend", "report"} and env_file.exists():
        env = json.loads(env_file.read_text())  # the probe run's record
        if "amend" in parts:
            amend_env = pb.environment_doc("construct_diag amend", "none", sys.argv[1:])
            env["amendment_run"] = {k: amend_env[k] for k in ("git_commit", "started_at_utc", "sklearn", "torch", "argv")}
            pb.write_json(env_file, env)
    else:
        env = pb.environment_doc("construct_diag", "none", sys.argv[1:])
        pb.write_json(env_file, env)  # written at the start of the run

    npr.configure_device("cpu")
    torch.use_deterministic_algorithms(False)  # as the reproduction gate and the NoPE diagnostic
    sitems, _ = npr.stratified_split_items()
    items = sitems[STRAT][:N_EXPR]  # the NoPE diagnostic's 20 expressions
    if "h1" in parts:
        run_h1(items, args.seeds, out_dir)
    if "h3" in parts:
        run_h3(items, args.seeds, out_dir)
    if "h2" in parts:
        run_h2(sitems, args.arms, args.seeds, out_dir)
    if "amend" in parts:
        h2_file, h3_file = out_dir / "h2_relative_offset.json", out_dir / "h3_rope.json"
        h2 = amend_h2(json.loads(h2_file.read_text()), sitems, list(H2_ARMS))
        pb.write_json(h2_file, h2)
        h3 = json.loads(h3_file.read_text())
        h3["expected"] = "encoder_input passes (no position at the input); layers 1-3 and memory fail, on both tests"
        h3["held"] = h3_held(h3)
        pb.write_json(h3_file, h3)
        print("amended: H2 readings recomputed with the NoPE/B1p/B2p/B3p reference; H3 reading updated", flush=True)
    if "report" in parts:
        h1, h3, h2 = (json.loads((out_dir / f).read_text())
                      for f in ("h1_alibi_reversal.json", "h3_rope.json", "h2_relative_offset.json"))
        (out_dir / "construct_diag.md").write_text(md_report(h1, h3, h2, env))
    env["finished_at_utc"] = pb.now_utc()
    env["wall_seconds"] = round(time.time() - t0, 1)
    if not parts <= {"amend", "report"}:
        pb.write_json(env_file, env)
    print(f"wall time {env['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()

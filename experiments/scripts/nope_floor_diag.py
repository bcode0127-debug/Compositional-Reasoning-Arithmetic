"""Diagnostics for the NoPE floor failure (positional-encoding sweep, nope study1 seed 0).

D1  Permutation equivariance of the trained NoPE encoder: for 20 stratified test expressions, permute
    the real (non-pad) encoder tokens (numpy seed 0), read both orders through the probes' hooks
    (transformer_states_hooked), un-permute the second, report max abs diff per feature. Pass <= 1e-5.
D2  Order-blind baselines on the same stratified split and positions, LogisticRegression(max_iter=1000)
    unscaled, position index as class (as identity_probe):
      B1 one-hot(char at the position)
      B2 B1 + one-hot(expression length)
      B3 B2 + bag-of-chars counts over the expression
    The NoPE layer-3 probe is recomputed on the same rows for reference.

  python experiments/scripts/nope_floor_diag.py --run-dir results_v2/pos_sweep/nope/study1/seed0
Writes nope_floor_diag.json into the run dir.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
N_EQUIV = 20
EQUIV_SEED = 0
EQUIV_TOL = 1e-5
EQUIV_FEATURES = ("encoder_input", "layer1", "layer2", "layer3", "memory")

_spec = importlib.util.spec_from_file_location("nonlinear_probe", REPO_ROOT / "experiments/scripts/nonlinear_probe.py")
npr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(npr)


def load_run_model(run_dir: Path):
    """Build the model from the checkpoint's positional_encoding, as nonlinear_probe.load_model does."""
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    assert ckpt["dataset_version"] == "datasets_v2", f"dataset_version={ckpt['dataset_version']!r}"
    model = npr.create_transformer_model(
        vocab_size=npr.tokenizer.vocab_size, d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3, pad_idx=npr.pad_idx,
        positional_encoding=ckpt.get("positional_encoding", "sinusoidal"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(npr.DEVICE)
    model.eval()
    return model, ckpt


def equivariance(model, items):
    """Max abs diff per feature between f(x) and P^-1 f(P x), over the real positions."""
    rng = np.random.default_rng(EQUIV_SEED)
    worst = {k: 0.0 for k in EQUIV_FEATURES}
    n_identity = 0
    for item in items:
        src = npr.encode_input(item["input"])
        real = np.flatnonzero(src.numpy() != npr.pad_idx)
        perm = rng.permutation(len(real))
        n_identity += int(np.all(perm == np.arange(len(real))))
        src_p = src.clone()
        src_p[real] = src[real[perm]]
        a = npr.transformer_states_hooked(model, src)
        b = npr.transformer_states_hooked(model, src_p)
        for k in EQUIV_FEATURES:
            b_un = np.empty_like(b[k][real])
            b_un[perm] = b[k][real]  # position real[j] of the permuted input holds original token real[perm[j]]
            worst[k] = max(worst[k], float(np.abs(a[k][real] - b_un).max()))
    return worst, n_identity


def baseline_features(items, kind):
    """Per-position rows in layerwise_acts order (expression, then position 0..len-1)."""
    V, L = npr.tokenizer.vocab_size, npr.MAX_INPUT_LEN
    X, pos = [], []
    for item in items:
        expr = item["input"]
        ids = npr.encode_checked(expr, L)
        counts = np.bincount(ids, minlength=V).astype(np.float32)
        for p in range(len(expr)):
            parts = [np.eye(V, dtype=np.float32)[ids[p]]]
            if kind in ("B2", "B3"):
                parts.append(np.eye(L + 1, dtype=np.float32)[len(expr)])
            if kind == "B3":
                parts.append(counts)
            X.append(np.concatenate(parts))
            pos.append(p)
    return {"X": np.array(X), "position": np.array(pos)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    npr.configure_device("cpu")
    torch.use_deterministic_algorithms(False)  # as the reproduction gate: CPU kernels here are deterministic anyway
    model, ckpt = load_run_model(args.run_dir)
    sitems, _ = npr.stratified_split_items()
    test_items = sitems[npr.STRAT_SPLIT]

    worst, n_identity = equivariance(model, test_items[:N_EQUIV])
    d1_pass = all(v <= EQUIV_TOL for v in worst.values())
    print(f"D1 equivariance, {N_EQUIV} expressions ({n_identity} identity permutations): "
          + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()) + f" -> {'PASS' if d1_pass else 'FAIL'}")

    feats = npr.layerwise_acts(model, sitems, features=("layer3",), with_ids=True)["layer3"]
    nope = npr.identity_probe(feats["train"], feats[npr.STRAT_SPLIT], "position")
    print(f"NoPE layer3: position accuracy {nope['accuracy']:.4f}, majority {nope['majority']:.4f}")
    d2 = {"nope_layer3": nope}
    for kind in ("B1", "B2", "B3"):
        b = {s: baseline_features(its, kind) for s, its in sitems.items()}
        for s in b:
            assert np.array_equal(b[s]["position"], feats[s]["position"]), f"{kind} {s}: rows misaligned"
        r = npr.identity_probe(b["train"], b[npr.STRAT_SPLIT], "position")
        r["dim"] = int(b["train"]["X"].shape[1])
        d2[kind] = r
        print(f"{kind}: position accuracy {r['accuracy']:.4f}, majority {r['majority']:.4f}, dim {r['dim']}, "
              f"convergence_warning {r['convergence_warning']}")

    gap = d2["B3"]["accuracy"] - nope["accuracy"]
    if not d1_pass:
        case = "(c) D1 fails: real order leak in the model"
    elif abs(gap) <= 0.02:
        case = "(a) D1 passes and B3 within 0.02 of NoPE layer 3: design floor mis-specified, model fine"
    elif gap < -0.02:
        case = "(b) D1 passes, B3 well below NoPE layer 3: unexplained signal"
    else:
        case = "D1 passes, B3 above NoPE layer 3 by more than 0.02 (outside the packet's three cases)"
    print(f"B3 - NoPE layer3 = {gap:+.4f}; case {case}")
    out = {"run_dir": str(args.run_dir), "positional_encoding": ckpt.get("positional_encoding"),
           "D1": {"n_expressions": N_EQUIV, "seed": EQUIV_SEED, "tolerance": EQUIV_TOL,
                  "n_identity_permutations": n_identity, "max_abs_diff": worst, "pass": d1_pass},
           "D2": d2, "B3_minus_nope": gap, "case": case}
    (args.run_dir / "nope_floor_diag.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

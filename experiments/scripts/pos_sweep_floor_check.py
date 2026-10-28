"""NoPE floor check for the positional-encoding sweep (design: docs/design/pos-encoding-sweep.md).

Layer-3 linear position accuracy of a trained run on the stratified split, computed as
test_probe_reproduction_gate does. A model with no positional signal should sit at the majority-class
rate; PASS if |accuracy - majority| <= 0.02, with the majority taken from identity_probe and
asserted to round to the recorded 0.053 (a different split fails loudly). Writes floor_check.json into the run dir.

  python experiments/scripts/pos_sweep_floor_check.py --run-dir results_v2/pos_sweep/nope/study1/seed0
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MAJORITY = 0.053
TOLERANCE = 0.02

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    npr.configure_device("cpu")
    torch.use_deterministic_algorithms(False)  # as the reproduction gate: CPU kernels here are deterministic anyway
    model, ckpt = load_run_model(args.run_dir)
    sitems, _ = npr.stratified_split_items()
    feats = npr.layerwise_acts(model, sitems, features=("layer3",), with_ids=True)["layer3"]
    probe = npr.identity_probe(feats["train"], feats[npr.STRAT_SPLIT], "position")
    acc, majority = probe["accuracy"], probe["majority"]
    assert round(majority, 3) == EXPECTED_MAJORITY, f"majority {majority:.4f} != {EXPECTED_MAJORITY} (split changed?)"
    delta = acc - majority
    passed = abs(delta) <= TOLERANCE
    result = {"run_dir": str(args.run_dir), "positional_encoding": ckpt.get("positional_encoding"),
              "layer3_position_accuracy": acc, "majority": majority, "expected_majority": EXPECTED_MAJORITY, "delta": delta,
              "tolerance": TOLERANCE, "pass": passed}
    (args.run_dir / "floor_check.json").write_text(json.dumps(result, indent=2))
    print(f"layer-3 linear position accuracy {acc:.4f}, majority {majority:.4f} (expected {EXPECTED_MAJORITY}), delta {delta:+.4f}: "
          f"{'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

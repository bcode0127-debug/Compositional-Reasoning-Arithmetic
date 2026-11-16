"""Append the positional-encoding sweep section to CHECKPOINTS.md, every value read from each run's own files.

  python experiments/scripts/pos_sweep_registry.py [--check]
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "results_v2" / "pos_sweep"
CHECKPOINTS = REPO_ROOT / "CHECKPOINTS.md"
HUB = "https://huggingface.co/Ajay0728/Compositional-reasoning-arithmetic/blob/main"
ARMS = ("learned", "rope", "alibi", "nope", "sinusoidal_retrain")
TITLE = "## Positional-encoding sweep checkpoints"
HEADER = ("| Path | Arm | Seed | Selected epoch | Val accuracy | Test accuracy | Training commit | GPU | "
          "Environment (torch, numpy, driver) | sha256 | Hub location |")


def run_dirs():
    found = sorted((p.parent for a in ARMS for p in RUNS_ROOT.glob(f"{a}/study1/seed*/best_model.pt")),
                   key=lambda d: (ARMS.index(d.parts[-3]), d.name))
    return found


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row(d):
    env = json.loads((d / "environment.json").read_text())
    test = json.loads((d / "test_results.json").read_text())
    arm, seed = d.parts[-3], test["seed"]
    assert d.name == f"seed{seed}" and test["n"] == 1000
    rel = d.relative_to(REPO_ROOT).as_posix()
    hub = d.relative_to(RUNS_ROOT).as_posix()
    return (f"| `{rel}/best_model.pt` | {arm} | {seed} | {test['epoch']} | {test['val_accuracy_fingerprint']}% | "
            f"{test['test_accuracy']}% | `{env['git_commit']}` | {env['gpu']} | torch {env['torch']}, numpy "
            f"{env['numpy']}, driver {env['driver_version']} | `{sha256_file(d / 'best_model.pt')}` | "
            f"[`pos_sweep/{hub}/best_model.pt`]({HUB}/pos_sweep/{hub}/best_model.pt) |")


NOTES = [
    "- The three `sinusoidal_retrain` rows are the hardware-control retrain of the paper's sinusoidal configuration on the RTX 4000 Ada. "
    "They are not used by any rule of the sweep, and the tuning control reports them only descriptively beside its reference R1 "
    "(`docs/design/tuning-control.md`). The paper's sinusoidal checkpoints are the `results_v2/transformer/study1/` rows in the table above.",
    "- Val accuracy is teacher-forced exact match (`utils/trainer.calculate_accuracy`) and is the measure checkpoints were selected on; "
    "test accuracy is free-running greedy decoding with exact match (`scripts/eval/eval_test_splits.py`). They are different measures and "
    "are never interchanged (`docs/design/pos-encoding-sweep.md`, Deviations, 2026-10-19, item 3).",
    "- The `nope` row is the floor check, not an arm: NoPE in the encoder only, one seed (`docs/design/pos-encoding-sweep.md`).",
    "- All thirteen runs train on the same `datasets_v2` Study 1 files as the rows above (generator commit "
    "`f5044574212d2a41e5091456096e0841cf5ae6f8`, the repository commit the generator ran at, recorded in each data file's "
    "`generator_commit` header field; the files themselves were committed later, in `402e0dd`, so they are not in the tree at `f5044574`); "
    "the train and validation files are byte-identical at both training commits and match `datasets_v2/CHECKSUMS.txt`.",
]


def section():
    dirs = run_dirs()
    assert len(dirs) == 13, len(dirs)
    return "\n".join([TITLE, "",
                      "Thirteen Study 1 Transformer checkpoints from the positional-encoding sweep, published on the "
                      "Hub under `pos_sweep/` with their run records and logs. The `.pt` files are not in git.", "",
                      HEADER, "|" + "---|" * 11] + [row(d) for d in dirs] + ["", "Notes:", ""] + NOTES) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if CHECKPOINTS.md lacks the current section")
    args = ap.parse_args()
    sec, text = section(), CHECKPOINTS.read_text()
    if args.check:
        assert sec in text, "CHECKPOINTS.md section out of date"
        return
    assert TITLE not in text, "section already present"
    CHECKPOINTS.write_text(text.rstrip("\n") + "\n\n" + sec)


if __name__ == "__main__":
    main()

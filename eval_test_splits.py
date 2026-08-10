#!/usr/bin/env python3
"""Evaluate all twelve saved checkpoints on the new independent test splits.

Protocol is identical to every reported number in the paper: genuine
free-running autoregressive greedy decoding, exact full-sequence match.
The decoder here is cold_verify.py's, not utils/trainer.py's teacher-forced
calculate_accuracy(), so the test numbers are produced the same way the
validation headline numbers were verified.

READ-ONLY with respect to checkpoints and existing datasets; writes only
results_v2/test_set_results.json.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from data.tokenizer import create_tokenizer
from models.transformer import create_transformer_model
from models.lstm import create_lstm_model

MAX_INPUT_LEN, MAX_OUTPUT_LEN = 64, 12
DATA_DIR = Path("datasets_v2")

tokenizer = create_tokenizer()
pad_idx = tokenizer.pad_idx


_parser = argparse.ArgumentParser()
_parser.add_argument("--device", default=None,
                     help="cpu/cuda/mps; default auto-detects. Pin to cpu for headless runs: "
                          "MPS init can stall in a detached background process.")
_args, _ = _parser.parse_known_args()


def device():
    if _args.device:
        return _args.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = device()


def encode_input(expr):
    ids = tokenizer.encode(expr)
    if len(ids) > MAX_INPUT_LEN:
        raise ValueError(f"too long: {expr!r} ({len(ids)} > {MAX_INPUT_LEN})")
    return torch.tensor(ids + [pad_idx] * (MAX_INPUT_LEN - len(ids)), dtype=torch.long)


def autoregressive_decode(model, src, max_len=MAX_OUTPUT_LEN):
    """Free-running greedy decode -- feeds the model's own prediction back in."""
    model.eval()
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(DEVICE)
        dec_ids = [tokenizer.sos_idx]
        for _ in range(max_len - 1):
            cur = (dec_ids + [pad_idx] * (max_len - len(dec_ids)))[:max_len]
            out = model(src_in, torch.tensor([cur], dtype=torch.long).to(DEVICE))
            nxt = out[0, len(dec_ids) - 1, :].argmax(dim=-1).item()
            if nxt in (tokenizer.eos_idx, pad_idx):
                break
            dec_ids.append(nxt)
    return dec_ids[1:]


def build(model_type):
    if model_type == "lstm":
        return create_lstm_model(vocab_size=tokenizer.vocab_size, embedding_dim=128, hidden_size=256)
    return create_transformer_model(
        vocab_size=tokenizer.vocab_size, d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3, pad_idx=pad_idx,
    )


def evaluate(model, items):
    """Returns (accuracy_pct, n, per_example) where per_example is a list of
    0/1 flags in dataset order. The per-example vector is what makes paired
    example-level tests (McNemar, paired bootstrap) possible; aggregate
    accuracy alone cannot support them."""
    correct = 0
    per_example = []
    for it in items:
        pred = tokenizer.decode(autoregressive_decode(model, encode_input(it["input"])))
        ok = int(pred == str(it["output"]))
        per_example.append(ok)
        correct += ok
    return 100.0 * correct / len(items), len(items), per_example


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    results = {}
    for study in ("study1", "study2"):
        items = json.loads((DATA_DIR / study / "test.json").read_text())["data"]
        for model_type in ("lstm", "transformer"):
            for seed in (0, 1, 2):
                ckpt_path = Path(f"results_v2/{model_type}/{study}/seed{seed}/best_model.pt")
                ckpt = torch.load(ckpt_path, map_location="cpu")
                model = build(model_type)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(DEVICE)
                print(f"  evaluating {model_type}/{study}/seed{seed} on {len(items)} test items...", flush=True)
                acc, n, per_example = evaluate(model, items)
                key = f"{model_type}_{study}_seed{seed}"
                results[key] = {
                    "test_accuracy": acc,
                    "n": n,
                    "val_accuracy_fingerprint": ckpt["val_accuracy"],
                    "epoch": ckpt["epoch"],
                    "dataset_version": ckpt.get("dataset_version"),
                    "per_example_correct": per_example,
                }
                print(f"{key}: test={acc:.2f}% (n={n}) | val(fingerprint)={ckpt['val_accuracy']:.2f}% epoch={ckpt['epoch']}", flush=True)

    out = Path("results_v2/test_set_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    print("\n=== summary: mean +/- SD across seeds ===")
    for study in ("study1", "study2"):
        for model_type in ("lstm", "transformer"):
            t = [results[f"{model_type}_{study}_seed{s}"]["test_accuracy"] for s in (0, 1, 2)]
            v = [results[f"{model_type}_{study}_seed{s}"]["val_accuracy_fingerprint"] for s in (0, 1, 2)]
            print(f"{model_type:12s} {study}: test {statistics.mean(t):.2f} +/- {statistics.stdev(t):.2f} | "
                  f"val {statistics.mean(v):.2f} +/- {statistics.stdev(v):.2f} | delta {statistics.mean(t)-statistics.mean(v):+.2f}")

#!/usr/bin/env python3
"""Evaluate all twelve checkpoints on every split, recording per-example correctness.

Needed because paired tests (McNemar, paired bootstrap) require example-level
outcomes, and those existed only for the test splits (all seeds) and Study 1
seed-0 val/OOD. This produces them uniformly for every (model, study, seed,
split), so model-vs-baseline pairing can run on every population the paper
reports.

Protocol matches every reported number: free-running autoregressive greedy
decode, exact full-sequence match. Read-only w.r.t. checkpoints and datasets;
writes results_v2/all_splits_per_example.json.
"""
import argparse
import json
from pathlib import Path

import torch

from data.tokenizer import create_tokenizer
from models.transformer import create_transformer_model
from models.lstm import create_lstm_model

MAX_INPUT_LEN, MAX_OUTPUT_LEN = 64, 12
DATA_DIR = Path("datasets_v2")
SPLITS = {
    "study1": ["val.json", "test.json", "ood_ops4.json", "ood_ops5.json",
               "ood_ops6.json", "ood_ops7.json"],
    "study2": ["val.json", "test.json", "ood.json"],
}

_p = argparse.ArgumentParser()
_p.add_argument("--device", default="cpu")
_args, _ = _p.parse_known_args()
DEVICE = _args.device

tokenizer = create_tokenizer()
pad_idx = tokenizer.pad_idx


def encode_input(expr):
    ids = tokenizer.encode(expr)
    if len(ids) > MAX_INPUT_LEN:
        raise ValueError(f"too long: {expr!r}")
    return torch.tensor(ids + [pad_idx] * (MAX_INPUT_LEN - len(ids)), dtype=torch.long)


def autoregressive_decode(model, src, max_len=MAX_OUTPUT_LEN):
    model.eval()
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(DEVICE)
        dec = [tokenizer.sos_idx]
        for _ in range(max_len - 1):
            cur = (dec + [pad_idx] * (max_len - len(dec)))[:max_len]
            out = model(src_in, torch.tensor([cur], dtype=torch.long).to(DEVICE))
            nxt = out[0, len(dec) - 1, :].argmax(dim=-1).item()
            if nxt in (tokenizer.eos_idx, pad_idx):
                break
            dec.append(nxt)
    return dec[1:]


def build(model_type):
    if model_type == "lstm":
        return create_lstm_model(vocab_size=tokenizer.vocab_size, embedding_dim=128, hidden_size=256)
    return create_transformer_model(vocab_size=tokenizer.vocab_size, d_model=256, nhead=8,
                                    num_encoder_layers=3, num_decoder_layers=3, pad_idx=pad_idx)


if __name__ == "__main__":
    print(f"Device: {DEVICE}", flush=True)
    results = {}
    for study, files in SPLITS.items():
        data = {f: json.loads((DATA_DIR / study / f).read_text())["data"] for f in files}
        for model_type in ("lstm", "transformer"):
            for seed in (0, 1, 2):
                ckpt = torch.load(f"results_v2/{model_type}/{study}/seed{seed}/best_model.pt",
                                  map_location="cpu")
                model = build(model_type)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(DEVICE)
                key = f"{model_type}_{study}_seed{seed}"
                results[key] = {"epoch": ckpt["epoch"], "val_accuracy_fingerprint": ckpt["val_accuracy"],
                                "splits": {}}
                for f in files:
                    items = data[f]
                    pe = []
                    for it in items:
                        pred = tokenizer.decode(autoregressive_decode(model, encode_input(it["input"])))
                        pe.append(int(pred == str(it["output"])))
                    acc = 100.0 * sum(pe) / len(pe)
                    results[key]["splits"][f] = {"n": len(pe), "accuracy": acc,
                                                 "per_example_correct": pe}
                    print(f"  {key:<28} {f:<16} acc={acc:6.2f}% (n={len(pe)})", flush=True)

    Path("results_v2/all_splits_per_example.json").write_text(json.dumps(results))
    print("\nWrote results_v2/all_splits_per_example.json")

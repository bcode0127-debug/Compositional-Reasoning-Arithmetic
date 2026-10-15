#!/usr/bin/env python3
"""Reviewer items 8, 9, 10.

Item 8  -- per-operation breakdown recomputed with OVERLAPPING categories:
           accuracy over all expressions containing each operator, so an
           expression with both + and * counts in both rows. Replaces the
           plurality-with-silent-tie-breaking rule.
Item 9  -- head-ablation denominator, read off the code rather than described.
Item 10 -- first-token confidence split by correctness, which the aggregated
           per-step means could not support.

Uses seed 0 throughout, matching the population the original Phase 1 and
Phase 3 analyses ran on. Writes results_v2/item_8_9_10_stats.json.
"""
import argparse
import json
from pathlib import Path

import torch

from data.tokenizer import create_tokenizer
from models.transformer import create_transformer_model
from models.lstm import create_lstm_model

DATA = Path("datasets_v2/study1")
SPLITS = ["val.json", "test.json", "ood_ops4.json", "ood_ops5.json", "ood_ops6.json", "ood_ops7.json"]
OPS = ["+", "-", "*", "/"]
MAX_INPUT_LEN, MAX_OUTPUT_LEN = 64, 12
N_CONF = 200          # matches the original confidence analysis
CONF_SPLITS = ["val.json", "ood_ops7.json"]

_p = argparse.ArgumentParser(); _p.add_argument("--device", default="cpu")
_a, _ = _p.parse_known_args(); DEVICE = _a.device

tok = create_tokenizer(); pad_idx = tok.pad_idx


def load_split(f):
    return json.loads((DATA / f).read_text())["data"]


def encode_input(expr):
    ids = tok.encode(expr)
    return torch.tensor(ids + [pad_idx] * (MAX_INPUT_LEN - len(ids)), dtype=torch.long)


def build(mt):
    if mt == "lstm":
        return create_lstm_model(vocab_size=tok.vocab_size, embedding_dim=128, hidden_size=256)
    return create_transformer_model(vocab_size=tok.vocab_size, d_model=256, nhead=8,
                                    num_encoder_layers=3, num_decoder_layers=3, pad_idx=pad_idx)


def load_model(mt):
    ck = torch.load(f"results_v2/{mt}/study1/seed0/best_model.pt", map_location="cpu")
    m = build(mt); m.load_state_dict(ck["model_state_dict"]); return m.to(DEVICE).eval()


def decode_with_conf(model, src, max_len=MAX_OUTPUT_LEN):
    """Free-running greedy decode, returning generated ids and the softmax
    probability of the argmax token at each step."""
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(DEVICE)
        dec, confs = [tok.sos_idx], []
        for _ in range(max_len - 1):
            cur = (dec + [pad_idx] * (max_len - len(dec)))[:max_len]
            out = model(src_in, torch.tensor([cur], dtype=torch.long).to(DEVICE))
            probs = torch.softmax(out[0, len(dec) - 1, :], dim=-1)
            c, nxt = probs.max(dim=-1)
            confs.append(c.item()); nxt = nxt.item()
            if nxt in (tok.eos_idx, pad_idx):
                break
            dec.append(nxt)
    return dec[1:], confs


def main():
    per_ex = json.loads(Path("results_v2/all_splits_per_example.json").read_text())
    report = {}

    # ---------------- Item 8 ----------------
    print("=" * 112)
    print("ITEM 8 -- per-operation accuracy, OVERLAPPING categories (all expressions containing the operator)")
    print("=" * 112)
    item8 = {}
    for mt in ("lstm", "transformer"):
        print(f"\n--- {mt} (seed 0) ---")
        print(f"{'split':<10}" + "".join(f"{o:>20}" for o in OPS))
        item8[mt] = {}
        for f in SPLITS:
            items = load_split(f)
            corr = per_ex[f"{mt}_study1_seed0"]["splits"][f]["per_example_correct"]
            row, cells = {}, ""
            for o in OPS:
                idx = [i for i, it in enumerate(items) if o in it["input"]]
                n = len(idx); c = sum(corr[i] for i in idx)
                acc = 100.0 * c / n if n else None
                row[o] = {"correct": c, "total": n, "accuracy": acc, "small_n": n < 10}
                cells += f"{(f'{c}/{n} ({acc:.1f}%)' + ('*' if n < 10 else '')) if n else 'n/a':>20}"
            item8[mt][f] = row
            print(f"{f.replace('.json',''):<10}{cells}")
    report["item8_overlapping"] = item8

    # overlap disclosure
    print("\n-- overlap: expressions falling into more than one operator category --")
    overlap = {}
    for f in SPLITS:
        items = load_split(f)
        counts = [len({ch for ch in it["input"] if ch in OPS}) for it in items]
        multi = sum(1 for c in counts if c > 1)
        overlap[f] = {"n": len(items), "in_multiple_categories": multi,
                      "pct": 100.0 * multi / len(items),
                      "mean_categories_per_expression": sum(counts) / len(counts)}
        print(f"  {f.replace('.json',''):<10} {multi}/{len(items)} = {100.0*multi/len(items):5.1f}%"
              f"   mean categories/expr = {sum(counts)/len(counts):.2f}")
    report["item8_overlap"] = overlap

    # Item 8 STEP 3: small-denominator cells
    print("\n-- cells with denominator < 10 --")
    small = [f"{mt}/{f.replace('.json','')}/{o} (n={item8[mt][f][o]['total']})"
             for mt in item8 for f in item8[mt] for o in OPS if item8[mt][f][o]["total"] < 10]
    print("  " + ("; ".join(small) if small else "none"))
    report["item8_small_denominator_cells"] = small

    # ---------------- Item 10 ----------------
    print("\n" + "=" * 112)
    print("ITEM 10 -- first-token confidence split by correctness (seed 0, first 200 per split)")
    print("=" * 112)
    item10 = {}
    for mt in ("lstm", "transformer"):
        model = load_model(mt); item10[mt] = {}
        for f in CONF_SPLITS:
            items = load_split(f)[:N_CONF]
            rows = []
            for it in items:
                gen, confs = decode_with_conf(model, encode_input(it["input"]))
                pred = tok.decode(gen); target = str(it["output"])
                tgt_ids = tok.encode(target)
                first_ok = bool(gen) and bool(tgt_ids) and gen[0] == tgt_ids[0]
                rows.append({"seq_correct": pred == target, "first_correct": first_ok,
                             "first_conf": confs[0] if confs else None})
            def agg(sel):
                v = [r["first_conf"] for r in rows if sel(r) and r["first_conf"] is not None]
                return {"n": len(v), "mean_first_token_confidence": (sum(v) / len(v)) if v else None}
            cells = {
                "sequence_correct":   agg(lambda r: r["seq_correct"]),
                "sequence_incorrect": agg(lambda r: not r["seq_correct"]),
                "first_token_correct":   agg(lambda r: r["first_correct"]),
                "first_token_incorrect": agg(lambda r: not r["first_correct"]),
                "all": agg(lambda r: True),
            }
            item10[mt][f] = cells
            print(f"\n  [{mt} / {f.replace('.json','')}]")
            for k, v in cells.items():
                m = f"{v['mean_first_token_confidence']:.3f}" if v["mean_first_token_confidence"] is not None else "n/a"
                print(f"    {k:<22} n={v['n']:<5} mean first-token confidence = {m}")
    report["item10_confidence_by_correctness"] = item10

    Path("results_v2/item_8_9_10_stats.json").write_text(json.dumps(report, indent=2))
    print("\nWrote results_v2/item_8_9_10_stats.json")


if __name__ == "__main__":
    main()

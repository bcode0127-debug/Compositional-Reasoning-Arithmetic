"""Cold, independent verification of both canary checkpoints (READ-ONLY).

Deliberately does NOT import anything from train_v2.py - only the core
library modules (data.tokenizer, models.transformer, models.lstm) and
fresh, from-scratch encoding/decoding/eval logic written in this file.
Nothing here is copy-pasted from utils/trainer.py's calculate_accuracy
either: this uses genuine autoregressive (free-running) decoding, matching
main.py's evaluate_model() greedy-decode loop, not teacher-forced
single-pass accuracy.
"""
import sys
import json
import subprocess
from pathlib import Path

sys.path.append(".")
import torch

from data.tokenizer import create_tokenizer
from models.transformer import create_transformer_model
from models.lstm import create_lstm_model

MAX_INPUT_LEN = 64
MAX_OUTPUT_LEN = 12
DATA_DIR = Path("datasets_v2")

tokenizer = create_tokenizer()
pad_idx = tokenizer.pad_idx


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


def encode_checked(text, max_len):
    ids = tokenizer.encode(text)
    if len(ids) > max_len:
        raise ValueError(f"too long: {text!r} ({len(ids)} > {max_len})")
    return ids


def load_split(study, filename):
    payload = json.loads((DATA_DIR / study / filename).read_text())
    return payload["data"]


def encode_input(expr):
    ids = encode_checked(expr, MAX_INPUT_LEN)
    ids = ids + [pad_idx] * (MAX_INPUT_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def autoregressive_decode(model, src, device, max_len=MAX_OUTPUT_LEN):
    """Genuine free-running greedy decode: feeds the model's own previous
    prediction back in at each step, matching main.py's evaluate_model()
    inference loop - NOT teacher-forced."""
    model.eval()
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(device)
        dec_ids = [tokenizer.sos_idx]
        for _ in range(max_len - 1):
            cur = dec_ids + [pad_idx] * (max_len - len(dec_ids))
            cur = cur[:max_len]
            dec_tensor = torch.tensor([cur], dtype=torch.long).to(device)
            out = model(src_in, dec_tensor)
            nxt = out[0, len(dec_ids) - 1, :].argmax(dim=-1).item()
            if nxt == tokenizer.eos_idx or nxt == pad_idx:
                break
            dec_ids.append(nxt)
    return dec_ids[1:]  # drop SOS


def is_correct(pred_ids, target_str):
    pred_str = tokenizer.decode(pred_ids)
    return pred_str == target_str, pred_str


def evaluate_file(model, device, study, filename):
    data = load_split(study, filename)
    correct = 0
    for item in data:
        src = encode_input(item["input"])
        pred_ids = autoregressive_decode(model, src, device)
        ok, _ = is_correct(pred_ids, str(item["output"]))
        if ok:
            correct += 1
    return 100.0 * correct / len(data), len(data)


def build_transformer(vocab_size, pad_idx):
    return create_transformer_model(
        vocab_size=vocab_size, d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3, pad_idx=pad_idx,
    )


def build_lstm(vocab_size):
    return create_lstm_model(vocab_size=vocab_size, embedding_dim=128, hidden_size=256)


def get_repo_commit_for(hash_str):
    try:
        out = subprocess.check_output(["git", "cat-file", "-e", hash_str], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False


def dump_examples(model, device, study, filename, n=10):
    data = load_split(study, filename)[:n]
    print(f"\n{'-'*100}")
    for i, item in enumerate(data):
        src = encode_input(item["input"])
        pred_ids = autoregressive_decode(model, src, device)
        pred_str = tokenizer.decode(pred_ids)
        target_str = str(item["output"])
        marker = "OK " if pred_str == target_str else "ERR"
        print(f"[{marker}] input={item['input']!r:60s} pred={pred_str!r:10s} target={target_str!r:10s}")


def main():
    print(f"Device: {DEVICE}")
    vocab_size = tokenizer.vocab_size

    reported = {
        "lstm": {"val": 35.1, "ops4": 14.4, "ops5": 9.5, "ops6": 2.8, "ops7": 2.1},
        "transformer": {"val": 7.5, "ops4": 1.2, "ops5": 0.5, "ops6": 0.1, "ops7": 0.6},
    }

    for model_name, ckpt_path, builder in [
        ("transformer", "results_v2/transformer/study1/seed0/best_model.pt", lambda: build_transformer(vocab_size, pad_idx)),
        ("lstm", "results_v2/lstm/study1/seed0/best_model.pt", lambda: build_lstm(vocab_size)),
    ]:
        print(f"\n{'='*100}\n{model_name.upper()}: cold verification\n{'='*100}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        fp = {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "optimizer_state_dict")}
        print("checkpoint path:", ckpt_path)
        print("fingerprint:", fp)

        dataset_meta = json.loads((DATA_DIR / "study1" / "train.json").read_text())
        print("\ndataset_version check:", fp.get("dataset_version"), "== 'datasets_v2':", fp.get("dataset_version") == "datasets_v2")
        print("generator_commit check:", fp.get("generator_commit"), "== dataset's own generator_commit:",
              fp.get("generator_commit") == dataset_meta.get("generator_commit"))
        print("commit exists in repo history:", get_repo_commit_for(fp.get("generator_commit", "")))

        model = builder()
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)
        model.eval()

        print("\nRe-evaluating (autoregressive, cold) ...")
        val_acc, val_n = evaluate_file(model, DEVICE, "study1", "val.json")
        print(f"  val.json: {val_acc:.2f}% (n={val_n}) -- reported: {reported[model_name]['val']}%")

        ood_accs = {}
        for n in (4, 5, 6, 7):
            acc, cnt = evaluate_file(model, DEVICE, "study1", f"ood_ops{n}.json")
            ood_accs[n] = acc
            print(f"  ood_ops{n}.json: {acc:.2f}% (n={cnt}) -- reported: {reported[model_name][f'ops{n}']}%")

        gate_pass = (
            abs(val_acc - reported[model_name]["val"]) < 1e-6
            and all(abs(ood_accs[n] - reported[model_name][f"ops{n}"]) < 1e-6 for n in (4, 5, 6, 7))
        )
        print(f"\nGATE (exact reproduction of reported numbers): {'PASS' if gate_pass else 'FAIL'}")

        dump_examples(model, DEVICE, "study1", "val.json", n=10)

    print(f"\n{'='*100}\nDONE\n{'='*100}")


if __name__ == "__main__":
    main()

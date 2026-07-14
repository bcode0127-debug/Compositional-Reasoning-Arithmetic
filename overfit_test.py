"""Overfit sanity test (scratch, not part of the pipeline): can a 5.5M-param
Transformer (and, as a control, the LSTM) memorize 100 training examples?

Train == eval set, deliberately, on exactly the first 100 rows of
datasets_v2/study1/train.json. No early stopping, no LR warmup - plain Adam,
so any failure to memorize points at a wiring fault (target/decoder-input
shift, tokenizer round-trip, EOS/PAD confusion, loss over wrong positions),
not an optimization-schedule issue.
"""
import sys
import json

sys.path.append(".")
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data.tokenizer import create_tokenizer
from models.transformer import create_transformer_model
from models.lstm import create_lstm_model

MAX_INPUT_LEN = 64
MAX_OUTPUT_LEN = 12
N_SAMPLES = 100
NUM_EPOCHS = 500
BATCH_SIZE = 32
PRINT_EVERY = 25
LR_TRANSFORMER = 3e-4
LR_LSTM = 0.001  # LSTM's established working lr, not forced to match the Transformer's

torch.manual_seed(0)

tokenizer = create_tokenizer()
pad_idx = tokenizer.pad_idx


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
print(f"Device: {DEVICE}", flush=True)


def encode_checked(text, max_len):
    ids = tokenizer.encode(text)
    if len(ids) > max_len:
        raise ValueError(f"too long: {text!r} ({len(ids)} > {max_len})")
    return ids


def prepare(raw):
    enc_inputs, dec_inputs, dec_targets, raw_inputs, raw_outputs = [], [], [], [], []
    for item in raw:
        input_expr = item["input"]
        output_expr = str(item["output"])

        enc = encode_checked(input_expr, MAX_INPUT_LEN)
        enc = enc + [pad_idx] * (MAX_INPUT_LEN - len(enc))

        answer_tokens = tokenizer.encode(output_expr)
        dec_in = [tokenizer.sos_idx] + answer_tokens
        assert len(dec_in) <= MAX_OUTPUT_LEN, f"decoder input too long: {output_expr!r}"
        dec_in = dec_in + [pad_idx] * (MAX_OUTPUT_LEN - len(dec_in))

        dec_tgt = answer_tokens + [tokenizer.eos_idx]
        assert len(dec_tgt) <= MAX_OUTPUT_LEN, f"decoder target too long: {output_expr!r}"
        dec_tgt = dec_tgt + [pad_idx] * (MAX_OUTPUT_LEN - len(dec_tgt))

        enc_inputs.append(enc)
        dec_inputs.append(dec_in)
        dec_targets.append(dec_tgt)
        raw_inputs.append(input_expr)
        raw_outputs.append(output_expr)

    return (
        torch.LongTensor(enc_inputs),
        torch.LongTensor(dec_inputs),
        torch.LongTensor(dec_targets),
        raw_inputs,
        raw_outputs,
    )


data = json.load(open("datasets_v2/study1/train.json"))["data"][:N_SAMPLES]
enc_inputs, dec_inputs, dec_targets, raw_inputs, raw_outputs = prepare(data)
print(f"Loaded {len(data)} samples (first {N_SAMPLES} of datasets_v2/study1/train.json)", flush=True)

dataset = TensorDataset(enc_inputs, dec_inputs, dec_targets)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def exact_match_accuracy(model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        e = enc_inputs.to(device)
        d = dec_inputs.to(device)
        t = dec_targets.to(device)
        out = model(e, d)
        preds = out.argmax(dim=-1)
        for b in range(e.size(0)):
            mask = t[b] != pad_idx
            if torch.equal(preds[b][mask], t[b][mask]):
                correct += 1
    return 100.0 * correct / e.size(0)


def greedy_decode(model, src, device, max_len=MAX_OUTPUT_LEN):
    model.eval()
    with torch.no_grad():
        src = src.unsqueeze(0).to(device)
        dec_ids = [tokenizer.sos_idx]
        for _ in range(max_len - 1):
            cur = dec_ids + [pad_idx] * (max_len - len(dec_ids))
            cur = cur[:max_len]
            dec_tensor = torch.tensor([cur], dtype=torch.long).to(device)
            out = model(src, dec_tensor)
            nxt = out[0, len(dec_ids) - 1, :].argmax(dim=-1).item()
            if nxt == tokenizer.eos_idx or nxt == pad_idx:
                break
            dec_ids.append(nxt)
        return dec_ids[1:]


def dump_diagnostics(model_name, model, device, n=5):
    print(f"\n{'='*80}\nDIAGNOSTICS for {model_name} ({n} samples)\n{'='*80}", flush=True)
    for i in range(n):
        print(f"\n--- sample {i} ---")
        print(f"  raw input      : {raw_inputs[i]!r}")
        print(f"  raw output     : {raw_outputs[i]!r}")
        print(f"  input ids      : {enc_inputs[i].tolist()}")
        print(f"  decoder in ids : {dec_inputs[i].tolist()}")
        print(f"  target ids     : {dec_targets[i].tolist()}")
        decoded_ids = greedy_decode(model, enc_inputs[i], device)
        print(f"  greedy decode ids: {decoded_ids}")
        print(f"  greedy decode str: {tokenizer.decode(decoded_ids)!r}")


def run_overfit_test(model_name, model, device, lr):
    print(f"\n{'='*80}\n{model_name}: overfit test on {N_SAMPLES} samples, lr={lr}, {NUM_EPOCHS} epochs\n{'='*80}", flush=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    crossed_50 = None
    crossed_95 = None
    final_acc = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for e, d, t in loader:
            e, d, t = e.to(device), d.to(device), t.to(device)
            optimizer.zero_grad()
            out = model(e, d)
            loss = criterion(out.view(-1, out.size(-1)), t.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)

        if (epoch + 1) % PRINT_EVERY == 0:
            acc = exact_match_accuracy(model, device)
            print(f"[{model_name}] epoch {epoch+1:4d}  train_loss={avg_loss:.4f}  exact_match_acc={acc:.2f}%", flush=True)
            if crossed_50 is None and acc >= 50.0:
                crossed_50 = epoch + 1
            if crossed_95 is None and acc >= 95.0:
                crossed_95 = epoch + 1
            final_acc = acc

    print(f"[{model_name}] DONE. crossed_50={crossed_50}  crossed_95={crossed_95}  final_acc={final_acc:.2f}%", flush=True)

    if final_acc is not None and final_acc < 50.0:
        dump_diagnostics(model_name, model, device, n=5)

    return {"model": model_name, "crossed_50": crossed_50, "crossed_95": crossed_95, "final_acc": final_acc}


def main():
    vocab_size = tokenizer.vocab_size

    transformer = create_transformer_model(
        vocab_size=vocab_size, d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3, pad_idx=pad_idx,
    )
    t_result = run_overfit_test("Transformer", transformer, DEVICE, LR_TRANSFORMER)

    lstm = create_lstm_model(vocab_size=vocab_size, embedding_dim=128, hidden_size=256)
    l_result = run_overfit_test("LSTM", lstm, DEVICE, LR_LSTM)

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(json.dumps({"transformer": t_result, "lstm": l_result}, indent=2))


if __name__ == "__main__":
    main()

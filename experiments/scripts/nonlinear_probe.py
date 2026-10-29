"""Nonlinear (two-layer MLP) probe on all twelve v2 checkpoints, matched to the
Phase 3 linear-probe protocol in experiments/notebooks/analysis_v2.ipynb.

Follow-up to the Phase 3 linear-probe run in analysis_v2.ipynb, whose nulls came
from weak instruments. For every checkpoint, encoder state, target (running
value, paren depth) and split (ID held-out + OOD), this runs:
  - the Phase 3 linear probes, unchanged (Ridge / LogisticRegression),
  - an MLP probe (3 probe seeds), epoch count by example-level early stopping
    on val[240:300], then refit on all of val[:300],
  - the same MLP on permuted training labels (control),
  - a majority-class baseline (depth),
  - a paired, example-level bootstrap of MLP minus control,
and writes one JSON per (model, study, seed, state, target), a summary table
with the pre-registered verdict rule, a capacity sweep, two figures, and an
environment record.

Stages (run in order by --stage all):
  bridge  Transformer study1 seed0, extraction exactly as the notebook (no
          key-padding mask), linear probes only. Reproduction gate.
  grid    All 12 checkpoints, all states/targets/splits, linear + MLP + control.
  sweep   Study 1 seed 0, both models, hidden in {64, 256, 1024}.
  addenda STEP 4b additions (depth rows): input baseline, label-space ceilings,
          masked/unmasked no-op check, epoch-cap check -> step4b_addenda.json.
  addenda4c STEP 4c additions (Study 1 depth): ordinal input baseline on the
          matched protocol, and a stratified 300/150 split -> step4c_addenda.json.
  addenda4d STEP 4d additions: stratified value rows; Transformer layer-wise
          depth sanity (seed 0) -> step4d_addenda.json.
  addenda4e STEP 4e additions: linear paired bootstraps (stratified depth),
          layer-wise depth for Transformer seeds 1-2 (+ decoder memory for all
          seeds), per-layer identity probes -> step4e_addenda.json.
  addenda4f STEP 4f additions: final-position final-answer probes (Transformer
          matched, masked; stratified LSTM h/c and Transformer) -> step4f_addenda.json.
  addenda4g STEP 4g: training R² beside the stored held-out R² for the matched
          LSTM seed-0 final-position rows -> step4f_addenda.json[matched_lstm_final_position].
  report  summary.md, figures (reads the JSONs written by the stages above).

Usage:
  python experiments/scripts/nonlinear_probe.py --stage all
  python experiments/scripts/nonlinear_probe.py --stage grid --only lstm_study1_seed0
"""
import os

# Must be set before CUDA initialises (torch.use_deterministic_algorithms on cuBLAS).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "notebooks"))

from data.tokenizer import create_tokenizer  # noqa: E402
from models.lstm import create_lstm_model  # noqa: E402
from models.transformer import create_transformer_model  # noqa: E402
from lstm_manual_unroll import manual_lstm_unroll  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_REPO = "Ajay0728/Compositional-reasoning-arithmetic"
DATA_REPO = "Ajay0728/Compositional-reasoning-arithmetic-datasets"
MODEL_REVISION = "516c5fc98ae8e75fc6363889231a12cb96c7418d"
DATA_REVISION = "df2dcff8828240910106c01851e5d57c8cdf6869"

MODELS = ("lstm", "transformer")
STUDIES = ("study1", "study2")
SEEDS = (0, 1, 2)
STATES = {"lstm": ("h", "c"), "transformer": ("pooled",)}
TARGETS = ("value", "depth")

MAX_INPUT_LEN = 64
MAX_OUTPUT_LEN = 12

# Transformer extraction. Default: forward hooks on the model's own modules, so every
# positional-encoding arm is probed on exactly what it computes (hooks also keep
# nn.TransformerEncoderLayer off its fused fast path, i.e. on the path training uses).
# --legacy-extraction restores the hand-written layer loop of the committed results
# (185b127 and earlier); it exists for reproduction tests only.
LEGACY_EXTRACTION = False
N_TRAIN, N_ID_TEST, N_OOD_TEST = 300, 150, 150
N_CARVE = 240  # early-stopping carve-out: train val[:240], stop on val[240:300]

OOD_SPLITS = {"study1": ("ops4", "ops5", "ops6", "ops7"), "study2": ("ood",)}
OOD_FILES = {"ops4": "ood_ops4.json", "ops5": "ood_ops5.json", "ops6": "ood_ops6.json",
             "ops7": "ood_ops7.json", "ood": "ood.json"}
DATA_FILES = {"study1": ("train.json", "val.json", "test.json", "ood_ops4.json", "ood_ops5.json",
                         "ood_ops6.json", "ood_ops7.json"),
              "study2": ("train.json", "val.json", "test.json", "ood.json")}

# Notebook cell 3 fingerprints (only defined for study1 seed0).
FINGERPRINTS = {("transformer", "study1", 0): 7.5, ("lstm", "study1", 0): 35.1}
UNFIT = {("transformer", "study2", 0): "unfit (train acc 5.85%)",
         ("transformer", "study2", 2): "unfit (train acc 14.51%)"}

# MLP protocol
PROBE_SEEDS = (0, 1, 2)
HIDDEN = 256
SWEEP_HIDDEN = (64, 256, 1024)
LR, WEIGHT_DECAY, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 256, 200, 10
CONTROL_PERM_SEED = 12345
N_BOOT = 1000
BOOT_SEED = 0

# Verdict rule thresholds
DEPTH_EFFECT = 0.05
VALUE_EFFECT = 0.10
VALUE_ABS = 0.10
SIGNAL_MARGIN = 0.05
CONTROL_R2_EXPECTED, CONTROL_R2_FLAG = 0.02, 0.05

# Recorded in CHECKPOINTS.md ("Evaluation environment (test-set numbers)").
REFERENCE_EVAL_ENV = {"source": "CHECKPOINTS.md, Evaluation environment (test-set numbers)",
                      "torch": "2.13.0", "numpy": "2.4.6", "device": "cpu",
                      "sklearn": "not recorded", "python": "not recorded"}

DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "analysis_v2" / "nonlinear_probe"
PHASE3_DIR = REPO_ROOT / "experiments" / "results" / "analysis_v2"

tokenizer = create_tokenizer()
pad_idx = tokenizer.pad_idx
DEVICE = "cpu"  # set in configure_device()


def configure_device(device_arg):
    global DEVICE
    if device_arg == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = device_arg
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return DEVICE


# ---------------------------------------------------------------------------
# Hugging Face downloads + integrity records
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_HF_META = {}


def _hf_lfs_sha256(repo_id, repo_type, revision, filename):
    from huggingface_hub import HfApi
    key = (repo_id, repo_type, revision)
    if key not in _HF_META:
        api = HfApi()
        info = (api.model_info(repo_id, revision=revision, files_metadata=True) if repo_type == "model"
                else api.dataset_info(repo_id, revision=revision, files_metadata=True))
        meta = {}
        for s in info.siblings:
            lfs = s.lfs
            sha = None
            if lfs is not None:
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            meta[s.rfilename] = sha
        _HF_META[key] = meta
    return _HF_META[key].get(filename)


FILE_RECORDS = {}


def hf_file(repo_id, filename, repo_type):
    """Download one file at the pinned commit; record commit, sha256, path."""
    from huggingface_hub import hf_hub_download
    revision = MODEL_REVISION if repo_type == "model" else DATA_REVISION
    path = hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision)
    key = f"{repo_type}:{filename}"
    if key not in FILE_RECORDS:
        local_sha = sha256_file(path)
        hf_sha = _hf_lfs_sha256(repo_id, repo_type, revision, filename)
        if hf_sha is not None:
            assert hf_sha == local_sha, f"sha256 mismatch vs Hub LFS record for {key}"
        FILE_RECORDS[key] = {"repo_id": repo_id, "repo_type": repo_type, "hf_commit": revision,
                             "filename": filename, "sha256": local_sha, "hub_lfs_sha256": hf_sha,
                             "local_path": str(path)}
    return Path(path)


def verify_data_checksums():
    """Every data file's sha256 vs the repo's datasets_v2/CHECKSUMS.txt."""
    expected = {}
    for line in (REPO_ROOT / "datasets_v2" / "CHECKSUMS.txt").read_text().splitlines():
        if line.strip():
            sha, rel = line.split()
            expected[rel.replace("datasets_v2/", "", 1)] = sha
    out = {}
    for study, files in DATA_FILES.items():
        for fn in files:
            rel = f"{study}/{fn}"
            got = sha256_file(hf_file(DATA_REPO, rel, "dataset"))
            out[rel] = {"expected": expected.get(rel), "actual": got,
                        "pass": expected.get(rel) == got}
    return out


_DATA_CACHE = {}


def load_split_items(study):
    """Phase 3 split protocol (notebook cells 13 and 23):
    train val[:300], ID held-out val[300:450], OOD buckets [:150]."""
    if study not in _DATA_CACHE:
        val = json.loads(hf_file(DATA_REPO, f"{study}/val.json", "dataset").read_text())["data"]
        splits = {"train": val[:N_TRAIN],
                  "ID_val_heldout": val[N_TRAIN:N_TRAIN + N_ID_TEST]}
        for name in OOD_SPLITS[study]:
            data = json.loads(hf_file(DATA_REPO, f"{study}/{OOD_FILES[name]}", "dataset").read_text())["data"]
            splits[name] = data[:N_OOD_TEST]
        _DATA_CACHE[study] = splits
    return _DATA_CACHE[study]


CKPT_META = {}


def load_model(model_name, study, seed, run_dir=None):
    """Default: the paper's checkpoint from the pinned Hub commit (FINGERPRINTS checked). With run_dir: the
    checkpoint <run_dir>/best_model.pt (a positional-encoding sweep run), which must carry
    dataset_version == "datasets_v2"; FINGERPRINTS describe the paper's checkpoints and are skipped."""
    if run_dir is None:
        path = hf_file(MODEL_REPO, f"{model_name}/{study}/seed{seed}/best_model.pt", "model")
    else:
        path = Path(run_dir) / "best_model.pt"
    # weights_only=False: the checkpoint dict carries plain metadata alongside
    # tensors, and these are this project's own files at a pinned Hub commit (or its own sweep runs).
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if run_dir is not None:
        assert ckpt.get("dataset_version") == "datasets_v2", (
            f"{run_dir}: dataset_version={ckpt.get('dataset_version')!r}")
    elif "dataset_version" in ckpt:
        assert ckpt["dataset_version"] == "datasets_v2", (
            f"{model_name}/{study}/seed{seed}: dataset_version={ckpt['dataset_version']!r}")
    fp = FINGERPRINTS.get((model_name, study, seed)) if run_dir is None else None
    if fp is not None:
        assert ckpt["val_accuracy"] == fp, (
            f"{model_name.upper()} FINGERPRINT MISMATCH: val_accuracy={ckpt['val_accuracy']} (expected {fp})")
    if model_name == "transformer":
        # Same construction as notebook cell 3 and train_v2.py build_model (pad_idx set).
        model = create_transformer_model(
            vocab_size=tokenizer.vocab_size, d_model=256, nhead=8,
            num_encoder_layers=3, num_decoder_layers=3, pad_idx=pad_idx,
            positional_encoding=ckpt.get("positional_encoding", "sinusoidal"),
        )
    else:
        model = create_lstm_model(vocab_size=tokenizer.vocab_size, embedding_dim=128, hidden_size=256)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    CKPT_META[f"{model_name}/{study}/seed{seed}" if run_dir is None else str(run_dir)] = {
        k: ckpt.get(k) for k in ("epoch", "val_accuracy", "val_loss", "seed", "dataset_version", "generator_commit",
                                 "positional_encoding")}
    return model


# ---------------------------------------------------------------------------
# Extraction + targets: copied verbatim from experiments/notebooks/analysis_v2.ipynb
# ---------------------------------------------------------------------------

# Source: analysis_v2.ipynb cell 4 (encode_checked, encode_input), verbatim.
def encode_checked(text, max_len):
    ids = tokenizer.encode(text)
    if len(ids) > max_len:
        raise ValueError(f"too long: {text!r} ({len(ids)} > {max_len})")
    return ids


def encode_input(expr):
    ids = encode_checked(expr, MAX_INPUT_LEN)
    ids = ids + [pad_idx] * (MAX_INPUT_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# Source: analysis_v2.ipynb cell 21 (TOKEN_RE, compute_position_targets), verbatim.
TOKEN_RE = re.compile(r'\(|\)|\+|\*|-?\d+|/|-')


def compute_position_targets(expr):
    """Exact position-level targets via a small recursive-descent parser over
    the fully-parenthesized grammar. depth_at_char: running paren-nesting depth.
    value_at_char: value of the most recently completed (closed) subexpression,
    forward-filled from each ')' - 0 before the first close."""
    n = len(expr)
    depth_at_char = [0] * n
    depth = 0
    for i, c in enumerate(expr):
        if c == '(':
            depth += 1
        depth_at_char[i] = depth
        if c == ')':
            depth -= 1

    tokens = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(expr)]
    value_events = []

    def parse(idx):
        tok, start, end = tokens[idx]
        if tok == '(':
            idx += 1
            left_val, idx = parse(idx)
            op_tok, _, _ = tokens[idx]
            idx += 1
            right_val, idx = parse(idx)
            close_tok, close_start, close_end = tokens[idx]
            assert close_tok == ')', f"expected ')' got {close_tok!r} in {expr!r}"
            if op_tok == '+':
                val = left_val + right_val
            elif op_tok == '-':
                val = left_val - right_val
            elif op_tok == '*':
                val = left_val * right_val
            elif op_tok == '/':
                val = left_val // right_val if right_val != 0 else 0
            else:
                raise ValueError(f"unknown op {op_tok!r}")
            value_events.append((close_end - 1, val))
            idx += 1
            return val, idx
        else:
            return int(tok), idx + 1

    parse(0)
    value_events.sort()
    value_at_char = [0] * n
    ei = 0
    current = 0
    for i in range(n):
        while ei < len(value_events) and value_events[ei][0] <= i:
            current = value_events[ei][1]
            ei += 1
        value_at_char[i] = current
    return depth_at_char, value_at_char


# Source: analysis_v2.ipynb cell 23 (lstm_encoder_states), verbatim.
def lstm_encoder_states(model, src):
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(DEVICE)
        outputs, _ = model.encoder(src_in)  # [1, seq, hidden*2]
    return outputs[0].cpu().numpy()


# Source: analysis_v2.ipynb cell 23 (transformer_encoder_states_mean_pooled), verbatim.
# Used only for the unmasked bridge (reproduction gate); no key_padding_mask.
def transformer_encoder_states_mean_pooled(model, src):
    """Manually replicate the encoder-layer loop (same as forward's
    return_attention path) to collect per-layer outputs, mean-pooled across
    the 3 layers per position."""
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(DEVICE)
        src_emb = model.encoder_embedding(src_in) * (model.d_model ** 0.5)
        src_emb = model.pos_encoder(src_emb)
        enc_out = src_emb
        per_layer = []
        for layer in model.transformer.encoder.layers:
            attn_out, _ = layer.self_attn(enc_out, enc_out, enc_out, need_weights=True, average_attn_weights=False)
            enc_out = layer.norm1(enc_out + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(enc_out))))
            enc_out = layer.norm2(enc_out + layer.dropout2(ff_out))
            per_layer.append(enc_out.clone())
        mean_pooled = torch.stack(per_layer, dim=0).mean(dim=0)
    return mean_pooled[0].cpu().numpy()


# Fixed version of the cell-23 function (primary Transformer extraction): passes
# the source key-padding mask into every encoder layer's self-attention, exactly
# as models/transformer.py forward() does when pad_idx is set (both its
# nn.Transformer path and its return_attention path). Only real positions are
# read downstream, so the layer mean is over real positions only.
def transformer_encoder_states_mean_pooled_masked(model, src):
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(DEVICE)
        src_key_padding_mask = (src_in == model.pad_idx)
        src_emb = model.encoder_embedding(src_in) * (model.d_model ** 0.5)
        src_emb = model.pos_encoder(src_emb)
        enc_out = src_emb
        per_layer = []
        for layer in model.transformer.encoder.layers:
            attn_out, _ = layer.self_attn(enc_out, enc_out, enc_out, key_padding_mask=src_key_padding_mask,
                                          need_weights=True, average_attn_weights=False)
            enc_out = layer.norm1(enc_out + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(enc_out))))
            enc_out = layer.norm2(enc_out + layer.dropout2(ff_out))
            per_layer.append(enc_out.clone())
        mean_pooled = torch.stack(per_layer, dim=0).mean(dim=0)
    return mean_pooled[0].cpu().numpy()


# Source: analysis_v2.ipynb cell 29 (lstm_cell_states), verbatim.
def lstm_cell_states(model, src):
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(DEVICE)
        embedded = model.encoder.dropout(model.encoder.embedding(src_in))
        _, c_fwd, _, c_bwd = manual_lstm_unroll(model.encoder.lstm, embedded)
        c_concat = torch.cat([c_fwd, c_bwd], dim=-1)
    return c_concat[0].cpu().numpy()


HOOKED_FEATURES = ("encoder_input", "layer1", "layer2", "layer3", "mean_pool", "memory")


def transformer_states_hooked(model, src):
    """Per-position encoder states read through forward hooks during model.forward.

    encoder_input = model.pos_encoder output (embedding * sqrt(d) plus the arm's input
    position signal; identity for rope / alibi / nope); layerN = output of
    transformer.encoder.layers[N-1]; memory = transformer.encoder output (layer 3 plus the
    final LayerNorm), the tensor cross-attention reads; mean_pool = mean of the three
    layer outputs. The decoder input is [SOS] + pads; encoder states do not depend on it.
    """
    captured = {}

    def keep(name):
        def hook(module, inputs, output):
            captured[name] = output.detach()
        return hook

    hooks = [model.pos_encoder.register_forward_hook(keep("encoder_input"))]
    for n, layer in enumerate(model.transformer.encoder.layers, 1):
        hooks.append(layer.register_forward_hook(keep(f"layer{n}")))
    hooks.append(model.transformer.encoder.register_forward_hook(keep("memory")))
    try:
        with torch.no_grad():
            src_in = src.unsqueeze(0).to(DEVICE)
            tgt = torch.full((1, MAX_OUTPUT_LEN), pad_idx, dtype=torch.long, device=src_in.device)
            tgt[0, 0] = tokenizer.sos_idx
            model(src_in, tgt)
    finally:
        for h in hooks:
            h.remove()
    layers = [captured[f"layer{n}"] for n in range(1, len(model.transformer.encoder.layers) + 1)]
    captured["mean_pool"] = torch.stack(layers, dim=0).mean(dim=0)
    return {k: captured[k][0].cpu().numpy() for k in HOOKED_FEATURES}


def transformer_pooled_states(model, src):
    """The 'pooled' Transformer state: hooked mean pool (default) or the legacy hand-written one."""
    if LEGACY_EXTRACTION:
        return transformer_encoder_states_mean_pooled_masked(model, src)
    return transformer_states_hooked(model, src)["mean_pool"]


def state_cache_tag(model_name, state):
    """Cache-file tag: hooked Transformer activations never share a file with legacy ones."""
    if model_name == "transformer" and state == "pooled" and not LEGACY_EXTRACTION:
        return "pooled-hooked"
    return state


STATE_FNS = {
    ("lstm", "h"): lstm_encoder_states,
    ("lstm", "c"): lstm_cell_states,
    ("transformer", "pooled"): transformer_pooled_states,
    ("transformer", "pooled_unmasked"): transformer_encoder_states_mean_pooled,
}


# Generalisation of analysis_v2.ipynb cell 29 build_probe_dataset_statefn: same
# per-position loop over range(len(expr)), additionally returning per-example
# offsets so whole expressions can be split/resampled.
def build_probe_dataset_statefn(state_fn, items):
    X, y_depth, y_value, lengths = [], [], [], []
    for item in items:
        expr = item["input"]
        depth_targets, value_targets = compute_position_targets(expr)
        src = encode_input(expr)
        states = state_fn(src)
        for pos in range(len(expr)):
            X.append(states[pos])
            y_depth.append(depth_targets[pos])
            y_value.append(value_targets[pos])
        lengths.append(len(expr))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return np.array(X), np.array(y_depth), np.array(y_value), offsets


def get_activations(model_name, study, seed, state, model, cache_dir):
    """{split: dict(X, depth, value, offsets)} for train + every eval split, cached as npz."""
    out = {}
    for split, items in load_split_items(study).items():
        f = Path(cache_dir) / f"{model_name}_{study}_seed{seed}_{state_cache_tag(model_name, state)}_{split}.npz"
        if f.exists():
            z = np.load(f)
            out[split] = {k: z[k] for k in ("X", "depth", "value", "offsets")}
            continue
        fn = STATE_FNS[(model_name, state)]
        X, yd, yv, off = build_probe_dataset_statefn(lambda s: fn(model, s), items)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, X=X, depth=yd, value=yv, offsets=off)
        out[split] = {"X": X, "depth": yd, "value": yv, "offsets": off}
    return out


# ---------------------------------------------------------------------------
# Baselines and linear probes (notebook cells 23, 24, 30)
# ---------------------------------------------------------------------------

def majority_baseline(y):
    # Source: analysis_v2.ipynb cell 24, applied to a split's own labels.
    counts = Counter(y.tolist())
    _, n = counts.most_common(1)[0]
    return n / len(y)


def majority_class(y):
    """The eval split's own majority class (same convention as majority_baseline)."""
    return Counter(y.tolist()).most_common(1)[0][0]


def linear_probe(target, train, evals):
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import r2_score, accuracy_score
    if target == "value":
        probe = Ridge(alpha=1.0)
        probe.fit(train["X"], train["value"])
        return {s: float(r2_score(d["value"], probe.predict(d["X"]))) for s, d in evals.items()}
    probe = LogisticRegression(max_iter=1000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        probe.fit(train["X"], train["depth"])
    conv = [str(x.message)[:200] for x in w if "converge" in str(x.message).lower()]
    res = {s: float(accuracy_score(d["depth"], probe.predict(d["X"]))) for s, d in evals.items()}
    if conv:
        res["_warnings"] = conv
    return res


# ---------------------------------------------------------------------------
# MLP probe
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    """y_hat = W2 relu(W1 x + b1) + b2"""

    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def subset_examples(d, lo, hi):
    """Positions of examples [lo, hi) of a split dict."""
    a, b = d["offsets"][lo], d["offsets"][hi]
    return {"X": d["X"][a:b], "depth": d["depth"][a:b], "value": d["value"][a:b],
            "offsets": d["offsets"][lo:hi + 1] - a}


class Prepared:
    """Input scaler (fit on the training positions only), target encoding."""

    def __init__(self, X_train, y_train, target, classes):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(X_train)
        self.target = target
        if target == "value":
            y = y_train.astype(np.float64)
            self.mu, self.sd = float(y.mean()), float(y.std()) or 1.0
            self.d_out = 1
        else:
            self.classes = classes
            self.cls_index = {c: i for i, c in enumerate(classes)}
            self.d_out = len(classes)

    def x(self, X):
        return torch.tensor(self.scaler.transform(X), dtype=torch.float32, device=DEVICE)

    def y(self, y):
        if self.target == "value":
            return torch.tensor((y.astype(np.float64) - self.mu) / self.sd, dtype=torch.float32, device=DEVICE)
        return torch.tensor([self.cls_index[v] for v in y.tolist()], dtype=torch.long, device=DEVICE)

    def loss(self, out, yt):
        if self.target == "value":
            return nn.functional.mse_loss(out.squeeze(-1), yt)
        return nn.functional.cross_entropy(out, yt)

    def predict(self, model, X):
        with torch.no_grad():
            out = model(self.x(X)).cpu().double()
        if self.target == "value":
            return out.squeeze(-1).numpy() * self.sd + self.mu
        return np.asarray(self.classes)[out.argmax(-1).numpy()]


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_mlp(prep, Xtr, ytr, hidden, seed, n_epochs=None, Xstop=None, ystop=None, history=None):
    """Fixed n_epochs, or early stopping (patience) on (Xstop, ystop).
    Returns (model, epochs) where epochs = chosen epoch count (1-indexed)."""
    set_seeds(seed)
    model = MLPProbe(Xtr.shape[1], hidden, prep.d_out).to(DEVICE)  # init drawn on CPU RNG
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    xt, yt = prep.x(Xtr), prep.y(ytr)
    if Xstop is not None:
        xs, ys = prep.x(Xstop), prep.y(ystop)
    g = torch.Generator().manual_seed(seed)
    n = xt.shape[0]
    best, best_epoch, best_state, since = float("inf"), 0, None, 0
    max_e = n_epochs if n_epochs is not None else MAX_EPOCHS
    for epoch in range(1, max_e + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            prep.loss(model(xt[idx]), yt[idx]).backward()
            opt.step()
        if n_epochs is None:
            model.eval()
            with torch.no_grad():
                vl = prep.loss(model(xs), ys).item()
            if history is not None:
                history.append(vl)
            if vl < best:
                best, best_epoch, since = vl, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
                if since >= PATIENCE:
                    break
    model.eval()
    if n_epochs is None:
        model.load_state_dict(best_state)
        return model, best_epoch
    return model, n_epochs


def metric(target, y, yhat):
    from sklearn.metrics import r2_score, accuracy_score
    return float(r2_score(y, yhat)) if target == "value" else float(accuracy_score(y, yhat))


def example_stats(target, y, yhat, offsets):
    """Per-example sufficient statistics for the example-level bootstrap."""
    idx = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    E = len(offsets) - 1
    n = np.bincount(idx, minlength=E).astype(np.float64)
    if target == "depth":
        return {"n": n, "correct": np.bincount(idx, weights=(y == yhat).astype(np.float64), minlength=E)}
    y = y.astype(np.float64)
    return {"n": n, "sy": np.bincount(idx, weights=y, minlength=E),
            "syy": np.bincount(idx, weights=y * y, minlength=E),
            "sres": np.bincount(idx, weights=(y - yhat) ** 2, minlength=E)}


def stat_metric(target, st, sel):
    """Metric over resampled examples; sel is [B, E] indices."""
    n = st["n"][sel].sum(-1)
    if target == "depth":
        return st["correct"][sel].sum(-1) / n
    sy, syy, sres = st["sy"][sel].sum(-1), st["syy"][sel].sum(-1), st["sres"][sel].sum(-1)
    sstot = syy - sy * sy / n
    return 1.0 - sres / sstot


def paired_bootstrap(target, st_a, st_b, n_examples, seed):
    rng = np.random.default_rng(seed)
    sel = rng.integers(0, n_examples, size=(N_BOOT, n_examples))
    diff = stat_metric(target, st_a, sel) - stat_metric(target, st_b, sel)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return [float(lo), float(hi)]


def run_mlp_protocol(target, acts, eval_splits, hidden, probe_seeds=PROBE_SEEDS, control=True):
    """Carve-out early stopping (train val[:240], stop val[240:300]), refit on
    val[:300] for the chosen epoch count; optional permuted-label control."""
    train = acts["train"]
    carve = subset_examples(train, 0, N_CARVE)
    stop = subset_examples(train, N_CARVE, N_TRAIN)
    classes = sorted(set(train["depth"].tolist())) if target == "depth" else None
    y_key = target

    perm = np.random.default_rng(CONTROL_PERM_SEED).permutation(len(train[y_key]))
    y_perm = train[y_key][perm]
    n_carve_pos = carve["offsets"][-1]

    out = {"mlp": {}, "control": {}}
    for kind in (("mlp", "control") if control else ("mlp",)):
        y_full = train[y_key] if kind == "mlp" else y_perm
        y_carve, y_stop = y_full[:n_carve_pos], y_full[n_carve_pos:]
        for seed in probe_seeds:
            prep_c = Prepared(carve["X"], y_carve, target, classes)
            m240, ep = train_mlp(prep_c, carve["X"], y_carve, hidden, seed, Xstop=stop["X"], ystop=y_stop)
            prep_f = Prepared(train["X"], y_full, target, classes)
            mfull, _ = train_mlp(prep_f, train["X"], y_full, hidden, seed, n_epochs=ep)
            rec = {"epochs": ep,
                   "train_metric": metric(target, y_full, prep_f.predict(mfull, train["X"])),
                   "heldout": {}, "heldout_240": {}, "stats": {}}
            for s in eval_splits:
                d = acts[s]
                yhat = prep_f.predict(mfull, d["X"])
                rec["heldout"][s] = metric(target, d[y_key], yhat)
                rec["heldout_240"][s] = metric(target, d[y_key], prep_c.predict(m240, d["X"]))
                rec["stats"][s] = example_stats(target, d[y_key], yhat, d["offsets"])
            out[kind][seed] = rec
    return out


# ---------------------------------------------------------------------------
# Verdict rule
# ---------------------------------------------------------------------------

def verdict(target, e):
    boot_pass = all(ci[0] > 0 for ci in e["bootstrap_ci"].values())
    if target == "depth":
        floor = e["mlp_mean"] - e["mlp_std"]
        effect = floor > e["control"] + DEPTH_EFFECT
        sanity = floor > e["majority"]
        bars = effect and sanity
    else:
        bars = (e["mlp_mean"] > e["control"] + VALUE_EFFECT) and (e["mlp_mean"] > VALUE_ABS)
    if boot_pass and bars:
        return "yes"
    if e["mlp_mean"] - e["linear"] > SIGNAL_MARGIN:
        return "signal, below threshold"
    return "no"


def assemble_split(target, s, acts, linear, prot):
    d = acts[s]
    mlp = prot["mlp"]
    ctrl = prot["control"]
    per_seed = [mlp[k]["heldout"][s] for k in PROBE_SEEDS]
    per_240 = [mlp[k]["heldout_240"][s] for k in PROBE_SEEDS]
    ctrl_seed = [ctrl[k]["heldout"][s] for k in PROBE_SEEDS]
    n_ex = len(d["offsets"]) - 1
    e = {
        "metric": "r2" if target == "value" else "accuracy",
        "linear": linear[s],
        "mlp_mean": float(np.mean(per_seed)),
        "mlp_std": float(np.std(per_seed)),
        "mlp_per_seed": per_seed,
        "mlp_240_trained": {"mean": float(np.mean(per_240)), "per_seed": per_240},
        "control": float(np.mean(ctrl_seed)),
        "control_per_seed": ctrl_seed,
        "control_train": float(np.mean([ctrl[k]["train_metric"] for k in PROBE_SEEDS])),
        "control_train_per_seed": [ctrl[k]["train_metric"] for k in PROBE_SEEDS],
        "mlp_train_per_seed": [mlp[k]["train_metric"] for k in PROBE_SEEDS],
        "epochs_chosen": {"mlp": [mlp[k]["epochs"] for k in PROBE_SEEDS],
                          "control": [ctrl[k]["epochs"] for k in PROBE_SEEDS]},
        "majority": majority_baseline(d["depth"]) if target == "depth" else None,
        "selectivity": float(np.mean(per_seed) - np.mean(ctrl_seed)),
        "bootstrap_ci": {f"seed{k}": paired_bootstrap(target, mlp[k]["stats"][s], ctrl[k]["stats"][s],
                                                      n_ex, BOOT_SEED + k) for k in PROBE_SEEDS},
        "n_examples": n_ex,
        "n_positions": int(len(d["depth"])),
    }
    if target == "depth":
        # Reported beside the verdict, not part of it: paired bootstrap of MLP minus
        # a constant predictor of this eval split's own majority class.
        maj_cls = majority_class(d["depth"])
        maj_st = example_stats(target, d["depth"], np.full_like(d["depth"], maj_cls), d["offsets"])
        e["majority_class"] = int(maj_cls)
        e["bootstrap_ci_vs_majority"] = {
            f"seed{k}": paired_bootstrap(target, mlp[k]["stats"][s], maj_st, n_ex, BOOT_SEED + k)
            for k in PROBE_SEEDS}
    e["verdict"] = verdict(target, e)
    return e


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def eval_split_names(study):
    return ("ID_val_heldout",) + OOD_SPLITS[study]


def out_name(model_name, study, seed, state, target):
    return f"{model_name}_{study}_seed{seed}_{state}_{target}.json"


def build_output_block(model_name, study, seed, target, acts):
    """{model_name: {split: fields}} - the per-file output schema."""
    splits = eval_split_names(study)
    evals = {s: acts[s] for s in splits}
    lin = linear_probe(target, acts["train"], evals)
    prot = run_mlp_protocol(target, acts, splits, HIDDEN)
    block = {}
    for s in splits:
        e = assemble_split(target, s, acts, lin, prot)
        e.update({
            "attention_mask": "masked" if model_name == "transformer" else "n/a (lstm)",
            "study_label": "exploratory" if study == "study2" else "primary",
            "fit_status": UNFIT.get((model_name, study, seed), "ok"),
            "hidden": HIDDEN,
        })
        if "_warnings" in lin:
            e["linear_warnings"] = lin["_warnings"]
        block[s] = e
    return {model_name: block}


def run_checkpoint(model_name, study, seed, out_dir, cache_dir, force=False):
    splits = eval_split_names(study)
    todo = [(st, t) for st in STATES[model_name] for t in TARGETS
            if force or not (out_dir / out_name(model_name, study, seed, st, t)).exists()]
    if not todo:
        print(f"[grid] {model_name}/{study}/seed{seed}: all outputs exist, skipping")
        return
    model = load_model(model_name, study, seed)
    for state in sorted({st for st, _ in todo}):
        acts = get_activations(model_name, study, seed, state, model, cache_dir)
        for target in [t for st, t in todo if st == state]:
            t0 = time.time()
            block = build_output_block(model_name, study, seed, target, acts)[model_name]
            with open(out_dir / out_name(model_name, study, seed, state, target), "w") as f:
                json.dump({model_name: block}, f, indent=2)
            print(f"[grid] {model_name}/{study}/seed{seed} {state} {target}: "
                  + " ".join(f"{s}={block[s]['linear']:.3f}/{block[s]['mlp_mean']:.3f}/{block[s]['control']:.3f}"
                             for s in splits) + f"  ({time.time() - t0:.0f}s)", flush=True)
    del model


def stage_bridge(out_dir, cache_dir):
    """Transformer study1 seed0, unmasked extraction exactly as the notebook,
    linear probes only; also the masked linear numbers for the bridge table."""
    model = load_model("transformer", "study1", 0)
    splits = eval_split_names("study1")
    res = {"transformer": {}}
    for state in ("pooled_unmasked", "pooled"):
        acts = get_activations("transformer", "study1", 0, state, model, cache_dir)
        evals = {s: acts[s] for s in splits}
        lin_v = linear_probe("value", acts["train"], evals)
        lin_d = linear_probe("depth", acts["train"], evals)
        for s in splits:
            res["transformer"].setdefault(s, {"n_positions": int(len(acts[s]["depth"])),
                                              "n_examples": len(acts[s]["offsets"]) - 1})
            tag = "unmasked" if state == "pooled_unmasked" else "masked"
            res["transformer"][s][f"value_r2_{tag}"] = lin_v[s]
            res["transformer"][s][f"depth_acc_{tag}"] = lin_d[s]
    stored = json.loads((PHASE3_DIR / "phase3_probe_results.json").read_text())["transformer"]
    for s in splits:
        res["transformer"][s]["value_r2_phase3_stored"] = stored[s]["value_r2"]
        res["transformer"][s]["depth_acc_phase3_stored"] = stored[s]["depth_acc"]
    with open(out_dir / "transformer_study1_seed0_pooled_unmasked_bridge.json", "w") as f:
        json.dump(res, f, indent=2)
    for s in splits:
        r = res["transformer"][s]
        print(f"[bridge] {s}: value stored/unmasked/masked = {r['value_r2_phase3_stored']:.3f}/"
              f"{r['value_r2_unmasked']:.3f}/{r['value_r2_masked']:.3f}  depth = {r['depth_acc_phase3_stored']:.3f}/"
              f"{r['depth_acc_unmasked']:.3f}/{r['depth_acc_masked']:.3f}")


def stage_grid(out_dir, cache_dir, only=None, force=False):
    for model_name in MODELS:
        for study in STUDIES:
            for seed in SEEDS:
                tag = f"{model_name}_{study}_seed{seed}"
                if only and tag not in only:
                    continue
                run_checkpoint(model_name, study, seed, out_dir, cache_dir, force=force)


def stage_sweep(out_dir, cache_dir, force=False):
    path = out_dir / "capacity_sweep.json"
    sweep = json.loads(path.read_text()) if path.exists() and not force else {}
    splits = eval_split_names("study1")
    for model_name in MODELS:
        model = None
        for state in STATES[model_name]:
            for target in TARGETS:
                lin = None
                for hidden in SWEEP_HIDDEN:
                    key = f"{model_name}_study1_seed0_{state}_{target}_h{hidden}"
                    if key in sweep:
                        continue
                    if model is None:
                        model = load_model(model_name, "study1", 0)
                    acts = get_activations(model_name, "study1", 0, state, model, cache_dir)
                    if lin is None:
                        lin = linear_probe(target, acts["train"], {s: acts[s] for s in splits})
                    prot = run_mlp_protocol(target, acts, splits, hidden)
                    entry = {"model": model_name, "study": "study1", "seed": 0, "state": state,
                             "target": target, "hidden": hidden,
                             "attention_mask": "masked" if model_name == "transformer" else "n/a (lstm)",
                             "splits": {}}
                    for s in splits:
                        e = assemble_split(target, s, acts, lin, prot)
                        entry["splits"][s] = {k: e[k] for k in (
                            "metric", "linear", "mlp_mean", "mlp_std", "mlp_per_seed", "mlp_240_trained",
                            "control", "control_train", "majority", "selectivity", "bootstrap_ci",
                            "epochs_chosen", "verdict", "n_examples", "n_positions")}
                    sweep[key] = entry
                    with open(path, "w") as f:
                        json.dump(sweep, f, indent=2)
                    print(f"[sweep] {key}: " + " ".join(
                        f"{s}={entry['splits'][s]['mlp_mean']:.3f}" for s in splits), flush=True)


# ---------------------------------------------------------------------------
# STEP 4b addenda (depth rows): input baseline, label-space ceilings, no-op and
# epoch-cap checks. Reads the grid outputs; writes step4b_addenda.json only.
# ---------------------------------------------------------------------------

DEPTH_BAR_MARGIN = 0.05  # 4b ceiling comparison: bar = majority + 0.05
CEILING_NEAR = 0.01
EPOCH_CAP_REPORT_FROM = MAX_EPOCHS - PATIENCE + 1  # best epoch >= 191: stopped by the cap, not by patience


def input_baseline_features(items):
    """one-hot(char) + one-hot(position index) + one-hot(expression length),
    concatenated. No prefix, no model activations."""
    V, L = tokenizer.vocab_size, MAX_INPUT_LEN
    X, y_depth, y_value, lengths = [], [], [], []
    for item in items:
        expr = item["input"]
        ids = encode_checked(expr, MAX_INPUT_LEN)
        depth_targets, value_targets = compute_position_targets(expr)
        n = len(expr)
        for pos in range(n):
            v = np.zeros(V + L + L + 1, dtype=np.float32)
            v[ids[pos]] = 1.0
            v[V + pos] = 1.0
            v[V + L + n] = 1.0
            X.append(v)
            y_depth.append(depth_targets[pos])
            y_value.append(value_targets[pos])
        lengths.append(n)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return {"X": np.array(X), "depth": np.array(y_depth), "value": np.array(y_value), "offsets": offsets}


def label_space_ceilings(study):
    """Per eval split: share of positions whose true depth is outside the
    training classes, and the resulting accuracy ceiling for any probe whose
    output space is the training classes (linear and MLP alike)."""
    items = load_split_items(study)
    train_depth = [d for it in items["train"] for d in compute_position_targets(it["input"])[0]]
    classes = sorted(set(train_depth))
    out = {"training_classes": classes, "training_majority": majority_baseline(np.array(train_depth))}
    for s in eval_split_names(study):
        y = np.array([d for it in items[s] for d in compute_position_targets(it["input"])[0]])
        outside = float(np.mean(~np.isin(y, classes)))
        maj = majority_baseline(y)
        out[s] = {"share_outside_training_classes": outside, "ceiling": 1.0 - outside,
                  "majority": maj, "bar": maj + DEPTH_BAR_MARGIN}
    return out


def qualified_verdict(base, ci_vs_input, ceil):
    if base == "yes":
        if all(c[0] > 0 for c in ci_vs_input.values()):
            return "yes, above input baseline"
        return "yes, not above input baseline"
    if base == "no" and ceil["ceiling"] < ceil["bar"]:
        return "structurally capped"
    if base == "no" and ceil["ceiling"] - ceil["bar"] < CEILING_NEAR:
        return f"no (ceiling within {ceil['ceiling'] - ceil['bar']:.3f} of bar)"
    return base


def cached_activations(model_name, study, seed, state, cache_dir):
    files = [Path(cache_dir) / f"{model_name}_{study}_seed{seed}_{state}_{s}.npz" for s in load_split_items(study)]
    model = None if all(f.exists() for f in files) else load_model(model_name, study, seed)
    return get_activations(model_name, study, seed, state, model, cache_dir), files


def epoch_cap_trace(acts, seed, hidden=HIDDEN):
    """Re-run the carve-out (early-stopping) fit of run_mlp_protocol with the
    per-epoch stop-set loss recorded."""
    train = acts["train"]
    carve, stop = subset_examples(train, 0, N_CARVE), subset_examples(train, N_CARVE, N_TRAIN)
    classes = sorted(set(train["depth"].tolist()))
    prep = Prepared(carve["X"], carve["depth"], "depth", classes)
    hist = []
    _, best = train_mlp(prep, carve["X"], carve["depth"], hidden, seed, Xstop=stop["X"], ystop=stop["depth"],
                        history=hist)
    return best, hist


def stage_addenda(out_dir, cache_dir):
    res = {"definitions": {
        "input_baseline_features": "one-hot(char, vocab 20) + one-hot(position 0..63) + one-hot(length 0..64); 149-d",
        "bar": "majority + 0.05 (4b ceiling comparison)",
        "ceiling": "1 - share of positions whose true depth is outside the training classes",
        "qualified_verdict": "yes -> 'yes, above input baseline' iff MLP(activations) - MLP(input baseline) "
                             "bootstrap lower bound > 0 for all three probe seeds, else 'yes, not above input "
                             "baseline'; no -> 'structurally capped' if ceiling < bar; no -> 'no (ceiling within "
                             "x of bar)' if 0 <= ceiling - bar < 0.01"},
        "input_baseline": {}, "ceilings": {}, "rows": {}, "noop_check": {}, "epoch_cap": {}}

    for study in STUDIES:
        splits = eval_split_names(study)
        items = load_split_items(study)
        acts_in = {k: input_baseline_features(v) for k, v in items.items()}
        lin_in = linear_probe("depth", acts_in["train"], {s: acts_in[s] for s in splits})
        prot_in = run_mlp_protocol("depth", acts_in, splits, HIDDEN)
        ib = {}
        for s in splits:
            e = assemble_split("depth", s, acts_in, lin_in, prot_in)
            ib[s] = {k: e[k] for k in ("linear", "mlp_mean", "mlp_std", "mlp_per_seed", "mlp_240_trained",
                                       "control", "control_per_seed", "control_train", "majority",
                                       "selectivity", "bootstrap_ci", "epochs_chosen", "n_examples",
                                       "n_positions")}
            ib[s]["n_features"] = int(acts_in[s]["X"].shape[1])
        res["input_baseline"][study] = ib
        ceil = label_space_ceilings(study)
        res["ceilings"][study] = ceil
        res.setdefault("distinct_depth_sequences", {})[study] = {
            k: len({tuple(compute_position_targets(it["input"])[0]) for it in v}) for k, v in items.items()}
        print(f"[addenda] {study} input baseline: " + " ".join(
            f"{s}={ib[s]['linear']:.3f}/{ib[s]['mlp_mean']:.3f}" for s in splits), flush=True)

        for model_name in MODELS:
            for seed in SEEDS:
                for state in STATES[model_name]:
                    p = out_dir / out_name(model_name, study, seed, state, "depth")
                    if not p.exists():
                        continue
                    stored = json.loads(p.read_text())[model_name]
                    acts, _ = cached_activations(model_name, study, seed, state, cache_dir)
                    prot = run_mlp_protocol("depth", acts, splits, HIDDEN, control=False)
                    rerun_diff = max(abs(prot["mlp"][k]["heldout"][s] - stored[s]["mlp_per_seed"][i])
                                     for i, k in enumerate(PROBE_SEEDS) for s in splits)
                    row = {"rerun_max_abs_diff_vs_stored_mlp": rerun_diff}
                    for s in splits:
                        n_ex = len(acts[s]["offsets"]) - 1
                        ci = {f"seed{k}": paired_bootstrap("depth", prot["mlp"][k]["stats"][s],
                                                           prot_in["mlp"][k]["stats"][s], n_ex, BOOT_SEED + k)
                              for k in PROBE_SEEDS}
                        row[s] = {"input_baseline_mlp": ib[s]["mlp_mean"],
                                  "input_baseline_linear": ib[s]["linear"],
                                  "bootstrap_ci_vs_input_baseline": ci,
                                  "ceiling": ceil[s]["ceiling"], "bar": ceil[s]["bar"],
                                  "verdict_rule": stored[s]["verdict"],
                                  "verdict_qualified": qualified_verdict(stored[s]["verdict"], ci, ceil[s])}
                    key = f"{model_name}_{study}_seed{seed}_{state}_depth"
                    res["rows"][key] = row
                    print(f"[addenda] {key}: rerun diff {rerun_diff:.2e}; " + " ".join(
                        f"{s}={row[s]['verdict_qualified']}" for s in splits), flush=True)

    # No-op check: Transformer study1 seed0 ID depth, masked vs unmasked linear predictions.
    from sklearn.linear_model import LogisticRegression
    acts_m, files_m = cached_activations("transformer", "study1", 0, "pooled", cache_dir)
    acts_u, files_u = cached_activations("transformer", "study1", 0, "pooled_unmasked", cache_dir)
    preds = {}
    for tag, a in (("masked", acts_m), ("unmasked", acts_u)):
        probe = LogisticRegression(max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe.fit(a["train"]["X"], a["train"]["depth"])
        preds[tag] = probe.predict(a["ID_val_heldout"]["X"])
    y = acts_m["ID_val_heldout"]["depth"]
    differ = preds["masked"] != preds["unmasked"]
    res["noop_check"] = {
        "split": "ID_val_heldout", "n_positions": int(len(y)),
        "positions_prediction_differs": int(differ.sum()),
        "differ_masked_correct": int((differ & (preds["masked"] == y)).sum()),
        "differ_unmasked_correct": int((differ & (preds["unmasked"] == y)).sum()),
        "masked_acc": float(np.mean(preds["masked"] == y)), "unmasked_acc": float(np.mean(preds["unmasked"] == y)),
        "max_abs_activation_diff_ID": float(np.abs(acts_m["ID_val_heldout"]["X"] - acts_u["ID_val_heldout"]["X"]).max()),
        "mean_abs_activation_diff_ID": float(np.abs(acts_m["ID_val_heldout"]["X"] - acts_u["ID_val_heldout"]["X"]).mean()),
        "masked_cache_files": [str(f) for f in files_m], "unmasked_cache_files": [str(f) for f in files_u],
        "masked_extraction_fn": STATE_FNS[("transformer", "pooled")].__name__,
        "unmasked_extraction_fn": STATE_FNS[("transformer", "pooled_unmasked")].__name__,
    }
    print(f"[addenda] no-op check: {res['noop_check']['positions_prediction_differs']} of {len(y)} ID positions differ", flush=True)

    # Epoch cap: LSTM depth fits whose chosen epoch is >= 191 were stopped by the
    # 200-epoch cap before patience could fire.
    for study in STUDIES:
        for seed in SEEDS:
            for state in STATES["lstm"]:
                p = out_dir / out_name("lstm", study, seed, state, "depth")
                if not p.exists():
                    continue
                stored = json.loads(p.read_text())["lstm"]["ID_val_heldout"]["epochs_chosen"]["mlp"]
                for i, k in enumerate(PROBE_SEEDS):
                    if stored[i] < EPOCH_CAP_REPORT_FROM:
                        continue
                    acts, _ = cached_activations("lstm", study, seed, state, cache_dir)
                    best, hist = epoch_cap_trace(acts, k)
                    res["epoch_cap"][f"lstm_{study}_seed{seed}_{state}_depth_probeseed{k}"] = {
                        "chosen_epoch_stored": stored[i], "chosen_epoch_rerun": best, "epochs_run": len(hist),
                        "stopped_by": "epoch cap" if len(hist) == MAX_EPOCHS and best > MAX_EPOCHS - PATIENCE else "patience",
                        "stop_loss_at_best": hist[best - 1], "stop_loss_final": hist[-1],
                        "stop_loss_last_20": hist[-20:],
                        "final_epoch_delta": hist[-1] - hist[-2],
                        "mean_delta_last_10": float(np.mean(np.diff(hist[-11:]))),
                        "still_falling_at_final_epoch": bool(hist[-1] < hist[-2]),
                    }
    for k, v in res["epoch_cap"].items():
        print(f"[addenda] epoch cap {k}: best {v['chosen_epoch_rerun']} final-epoch delta {v['final_epoch_delta']:+.5f} "
              f"still falling {v['still_falling_at_final_epoch']}", flush=True)

    with open(out_dir / "step4b_addenda.json", "w") as f:
        json.dump(res, f, indent=2)


# ---------------------------------------------------------------------------
# STEP 4c addenda (Study 1 depth): ordinal input baseline, stratified split.
# Reads grid + step4b outputs; writes step4c_addenda.json only.
# ---------------------------------------------------------------------------

STRAT_SEED = 0
STRAT_SPLIT = "stratified_test"


def ordinal_baseline_features(items):
    """one-hot(char) + position (scalar) + length (scalar) + position/length.
    No one-hot position or length, so it can extrapolate to unseen lengths."""
    V = tokenizer.vocab_size
    X, y_depth, y_value, lengths = [], [], [], []
    for item in items:
        expr = item["input"]
        ids = encode_checked(expr, MAX_INPUT_LEN)
        depth_targets, value_targets = compute_position_targets(expr)
        n = len(expr)
        for pos in range(n):
            v = np.zeros(V + 3, dtype=np.float32)
            v[ids[pos]] = 1.0
            v[V], v[V + 1], v[V + 2] = pos, n, pos / n
            X.append(v)
            y_depth.append(depth_targets[pos])
            y_value.append(value_targets[pos])
        lengths.append(n)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return {"X": np.array(X), "depth": np.array(y_depth), "value": np.array(y_value), "offsets": offsets}


def stratified_split_items(seed=STRAT_SEED):
    """Seeded random 300/150 split of study1 val.json, stratified by operation
    count. The training list is ordered so the 240/60 carve-out is stratified too."""
    val = json.loads(hf_file(DATA_REPO, "study1/val.json", "dataset").read_text())["data"]
    rng = np.random.default_rng(seed)
    groups = {}
    for i, it in enumerate(val):
        groups.setdefault(it["num_operations"], []).append(i)
    carve, stop, test = [], [], []
    for ops in sorted(groups):
        idx = rng.permutation(groups[ops])
        share = len(idx) / len(val)
        n_tr, n_te = round(N_TRAIN * share), round(N_ID_TEST * share)
        n_carve = round(N_CARVE * share)
        carve += list(idx[:n_carve])
        stop += list(idx[n_carve:n_tr])
        test += list(idx[n_tr:n_tr + n_te])
    assert len(carve) == N_CARVE and len(stop) == N_TRAIN - N_CARVE and len(test) == N_ID_TEST
    carve, stop, test = (list(rng.permutation(x)) for x in (carve, stop, test))
    return {"train": [val[i] for i in carve + stop], STRAT_SPLIT: [val[i] for i in test]}, \
           {"train_indices": [int(i) for i in carve + stop], "test_indices": [int(i) for i in test]}


def strat_activations(model_name, seed, state, model, items, cache_dir):
    out = {}
    for split, its in items.items():
        f = Path(cache_dir) / f"{model_name}_study1_seed{seed}_{state_cache_tag(model_name, state)}_strat{STRAT_SEED}_{split}.npz"
        if f.exists():
            z = np.load(f)
            out[split] = {k: z[k] for k in ("X", "depth", "value", "offsets")}
            continue
        fn = STATE_FNS[(model_name, state)]
        X, yd, yv, off = build_probe_dataset_statefn(lambda s_: fn(model, s_), its)
        np.savez(f, X=X, depth=yd, value=yv, offsets=off)
        out[split] = {"X": X, "depth": yd, "value": yv, "offsets": off}
    return out


def baseline_block(acts_in, splits, target="depth"):
    lin = linear_probe(target, acts_in["train"], {s: acts_in[s] for s in splits})
    prot = run_mlp_protocol(target, acts_in, splits, HIDDEN)
    out = {}
    for s in splits:
        e = assemble_split(target, s, acts_in, lin, prot)
        out[s] = {k: e[k] for k in ("linear", "mlp_mean", "mlp_std", "mlp_per_seed", "mlp_240_trained", "control",
                                    "control_per_seed", "control_train", "majority", "selectivity", "bootstrap_ci",
                                    "epochs_chosen", "n_examples", "n_positions")}
        out[s]["n_features"] = int(acts_in[s]["X"].shape[1])
    return out, prot


def paired_cis(prot_a, prot_b, s, n_ex, target="depth"):
    return {f"seed{k}": paired_bootstrap(target, prot_a["mlp"][k]["stats"][s], prot_b["mlp"][k]["stats"][s],
                                         n_ex, BOOT_SEED + k) for k in PROBE_SEEDS}


def verdict_4c(rule, ci_onehot, ci_ordinal, fallback):
    if rule == "yes":
        both = all(c[0] > 0 for ci in (ci_onehot, ci_ordinal) for c in ci.values())
        return "yes, above input baseline" if both else "yes, not above input baseline"
    return fallback


def stage_addenda4c(out_dir, cache_dir):
    add4b = json.loads((out_dir / "step4b_addenda.json").read_text())
    res = {"definitions": {
        "ordinal_baseline_features": "one-hot(char, vocab 20) + position (scalar) + length (scalar) + position/length; 23-d",
        "qualified_verdict_4c": "rule 'yes' -> 'yes, above input baseline' iff MLP(activations) - MLP(one-hot "
                                "baseline) and MLP(activations) - MLP(ordinal baseline) both have bootstrap lower "
                                "bound > 0 on all three probe seeds, else 'yes, not above input baseline'; other "
                                "verdicts as in step4b",
        "stratified_split": f"seeded (seed {STRAT_SEED}) random 300/150 split of study1 val.json stratified by "
                            "num_operations; carve-out 240/60 stratified the same way"},
        "ordinal_baseline_matched": {}, "rows_matched": {}, "stratified": {}}

    # 1. Ordinal baseline on the matched (Phase 3) protocol, Study 1 depth rows.
    study = "study1"
    splits = eval_split_names(study)
    items = load_split_items(study)
    acts_ord = {k: ordinal_baseline_features(v) for k, v in items.items()}
    res["ordinal_baseline_matched"], prot_ord = baseline_block(acts_ord, splits)
    print("[4c] ordinal baseline (matched): " + " ".join(
        f"{s}={res['ordinal_baseline_matched'][s]['linear']:.3f}/{res['ordinal_baseline_matched'][s]['mlp_mean']:.3f}"
        for s in splits), flush=True)
    for model_name in MODELS:
        for seed in SEEDS:
            for state in STATES[model_name]:
                key = f"{model_name}_{study}_seed{seed}_{state}_depth"
                stored = json.loads((out_dir / f"{key}.json").read_text())[model_name]
                acts, _ = cached_activations(model_name, study, seed, state, cache_dir)
                prot = run_mlp_protocol("depth", acts, splits, HIDDEN, control=False)
                rerun_diff = max(abs(prot["mlp"][k]["heldout"][s] - stored[s]["mlp_per_seed"][i])
                                 for i, k in enumerate(PROBE_SEEDS) for s in splits)
                row = {"rerun_max_abs_diff_vs_stored_mlp": rerun_diff}
                for s in splits:
                    q4b = add4b["rows"][key][s]
                    ci_ord = paired_cis(prot, prot_ord, s, len(acts[s]["offsets"]) - 1)
                    row[s] = {"ib_ordinal_mlp": res["ordinal_baseline_matched"][s]["mlp_mean"],
                              "ib_ordinal_linear": res["ordinal_baseline_matched"][s]["linear"],
                              "bootstrap_ci_vs_ordinal_baseline": ci_ord,
                              "verdict_rule": q4b["verdict_rule"], "verdict_4b": q4b["verdict_qualified"],
                              "verdict_4c": verdict_4c(q4b["verdict_rule"], q4b["bootstrap_ci_vs_input_baseline"],
                                                       ci_ord, q4b["verdict_qualified"])}
                res["rows_matched"][key] = row
                print(f"[4c] matched {key}: rerun diff {rerun_diff:.2e}; " + " ".join(
                    f"{s}={row[s]['verdict_4c']}" for s in splits), flush=True)

    # 2. Stratified split, Study 1 depth (LSTM h, c; Transformer pooled; seeds 0-2).
    sitems, sidx = stratified_split_items()
    ssplits = (STRAT_SPLIT,)
    st = {"split": {**sidx,
                    "train_ops": dict(Counter(it["num_operations"] for it in sitems["train"])),
                    "carve_ops": dict(Counter(it["num_operations"] for it in sitems["train"][:N_CARVE])),
                    "stop_ops": dict(Counter(it["num_operations"] for it in sitems["train"][N_CARVE:])),
                    "test_ops": dict(Counter(it["num_operations"] for it in sitems[STRAT_SPLIT])),
                    "train_lengths": sorted({len(it["input"]) for it in sitems["train"]}),
                    "test_lengths": sorted({len(it["input"]) for it in sitems[STRAT_SPLIT]})},
          "input_baseline_onehot": None, "input_baseline_ordinal": None, "ceiling": None, "rows": {}}
    st["split"] = {k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v) for k, v in st["split"].items()}
    acts_oh = {k: input_baseline_features(v) for k, v in sitems.items()}
    acts_or = {k: ordinal_baseline_features(v) for k, v in sitems.items()}
    st["input_baseline_onehot"], prot_oh_s = baseline_block(acts_oh, ssplits)
    st["input_baseline_ordinal"], prot_or_s = baseline_block(acts_or, ssplits)
    train_classes = sorted(set(acts_oh["train"]["depth"].tolist()))
    yt = acts_oh[STRAT_SPLIT]["depth"]
    maj = majority_baseline(yt)
    st["ceiling"] = {"training_classes": train_classes,
                     "ceiling": float(np.mean(np.isin(yt, train_classes))), "majority": maj,
                     "bar": maj + DEPTH_BAR_MARGIN}
    print(f"[4c] stratified split: train ops {st['split']['train_ops']} test ops {st['split']['test_ops']}; "
          f"one-hot {st['input_baseline_onehot'][STRAT_SPLIT]['mlp_mean']:.3f} "
          f"ordinal {st['input_baseline_ordinal'][STRAT_SPLIT]['mlp_mean']:.3f}", flush=True)
    for model_name in MODELS:
        for seed in SEEDS:
            model = None
            for state in STATES[model_name]:
                fs = [Path(cache_dir) / f"{model_name}_study1_seed{seed}_{state_cache_tag(model_name, state)}_strat{STRAT_SEED}_{k}.npz"
                      for k in sitems]
                if model is None and not all(f.exists() for f in fs):
                    model = load_model(model_name, "study1", seed)
                acts = strat_activations(model_name, seed, state, model, sitems, cache_dir)
                lin = linear_probe("depth", acts["train"], {STRAT_SPLIT: acts[STRAT_SPLIT]})
                prot = run_mlp_protocol("depth", acts, ssplits, HIDDEN)
                e = assemble_split("depth", STRAT_SPLIT, acts, lin, prot)
                n_ex = len(acts[STRAT_SPLIT]["offsets"]) - 1
                e = {k: v for k, v in e.items()}
                e["ib_onehot_mlp"] = st["input_baseline_onehot"][STRAT_SPLIT]["mlp_mean"]
                e["ib_ordinal_mlp"] = st["input_baseline_ordinal"][STRAT_SPLIT]["mlp_mean"]
                e["bootstrap_ci_vs_onehot_baseline"] = paired_cis(prot, prot_oh_s, STRAT_SPLIT, n_ex)
                e["bootstrap_ci_vs_ordinal_baseline"] = paired_cis(prot, prot_or_s, STRAT_SPLIT, n_ex)
                e["verdict_rule"] = e.pop("verdict")
                e["verdict_4c"] = verdict_4c(e["verdict_rule"], e["bootstrap_ci_vs_onehot_baseline"],
                                             e["bootstrap_ci_vs_ordinal_baseline"],
                                             qualified_verdict(e["verdict_rule"], {}, st["ceiling"]))
                e["attention_mask"] = "masked" if model_name == "transformer" else "n/a (lstm)"
                key = f"{model_name}_study1_seed{seed}_{state}_depth"
                st["rows"][key] = e
                print(f"[4c] stratified {key}: linear {e['linear']:.3f} mlp {e['mlp_mean']:.3f} control "
                      f"{e['control']:.3f} -> {e['verdict_4c']}", flush=True)
            del model
    res["stratified"] = st
    with open(out_dir / "step4c_addenda.json", "w") as f:
        json.dump(res, f, indent=2)


# ---------------------------------------------------------------------------
# STEP 4d addenda: stratified value rows; Transformer layer-wise depth sanity.
# Reads the strat0 split and caches; writes step4d_addenda.json only.
# ---------------------------------------------------------------------------

LAYERWISE_FEATURES = ("encoder_input", "layer1", "layer2", "layer3", "mean_pool")


def verdict_value_4d(e):
    """Bars as specified in STEP 4d: MLP mean minus std > control + 0.10 and > 0.10
    absolute, plus the bootstrap vs control (all three probe seeds)."""
    boot_pass = all(ci[0] > 0 for ci in e["bootstrap_ci"].values())
    floor = e["mlp_mean"] - e["mlp_std"]
    bars = floor > e["control"] + VALUE_EFFECT and floor > VALUE_ABS
    if boot_pass and bars:
        return "yes"
    if e["mlp_mean"] - e["linear"] > SIGNAL_MARGIN:
        return "signal, below threshold"
    return "no"


# Same layer loop as transformer_encoder_states_mean_pooled_masked, returning the
# encoder input and every layer output separately.
def transformer_layerwise_masked(model, src):
    with torch.no_grad():
        src_in = src.unsqueeze(0).to(DEVICE)
        src_key_padding_mask = (src_in == model.pad_idx)
        src_emb = model.encoder_embedding(src_in) * (model.d_model ** 0.5)
        src_emb = model.pos_encoder(src_emb)
        enc_out = src_emb
        per_layer = []
        for layer in model.transformer.encoder.layers:
            attn_out, _ = layer.self_attn(enc_out, enc_out, enc_out, key_padding_mask=src_key_padding_mask,
                                          need_weights=True, average_attn_weights=False)
            enc_out = layer.norm1(enc_out + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(enc_out))))
            enc_out = layer.norm2(enc_out + layer.dropout2(ff_out))
            per_layer.append(enc_out.clone())
        mean_pooled = torch.stack(per_layer, dim=0).mean(dim=0)
        # What the decoder cross-attends to: nn.Transformer's final encoder LayerNorm on layer 3.
        memory = model.transformer.encoder.norm(per_layer[-1])
    out = {"encoder_input": src_emb, "layer1": per_layer[0], "layer2": per_layer[1], "layer3": per_layer[2],
           "mean_pool": mean_pooled, "memory": memory}
    return {k: v[0].cpu().numpy() for k, v in out.items()}


def layerwise_acts(model, items, features=LAYERWISE_FEATURES, with_ids=False):
    feats = {k: {} for k in features}
    for split, its in items.items():
        rows = {k: [] for k in features}
        chars, poss = [], []
        yd, yv, lengths = [], [], []
        for item in its:
            expr = item["input"]
            d, v = compute_position_targets(expr)
            src = encode_input(expr)
            st = transformer_layerwise_masked(model, src) if LEGACY_EXTRACTION else transformer_states_hooked(model, src)
            for pos in range(len(expr)):
                for k in features:
                    rows[k].append(st[k][pos])
                yd.append(d[pos])
                yv.append(v[pos])
                chars.append(int(src[pos]))
                poss.append(pos)
            lengths.append(len(expr))
        off = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        for k in features:
            feats[k][split] = {"X": np.array(rows[k]), "depth": np.array(yd), "value": np.array(yv), "offsets": off}
            if with_ids:
                feats[k][split]["char"] = np.array(chars)
                feats[k][split]["position"] = np.array(poss)
    return feats


def char_position_features(items):
    """one-hot(char) + one-hot(position); no length (the encoder input carries none)."""
    V, L = tokenizer.vocab_size, MAX_INPUT_LEN
    X, y_depth, y_value, lengths = [], [], [], []
    for item in items:
        expr = item["input"]
        ids = encode_checked(expr, MAX_INPUT_LEN)
        d, v_ = compute_position_targets(expr)
        for pos in range(len(expr)):
            v = np.zeros(V + L, dtype=np.float32)
            v[ids[pos]] = 1.0
            v[V + pos] = 1.0
            X.append(v)
            y_depth.append(d[pos])
            y_value.append(v_[pos])
        lengths.append(len(expr))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return {"X": np.array(X), "depth": np.array(y_depth), "value": np.array(y_value), "offsets": offsets}


def stage_addenda4d(out_dir, cache_dir):
    sitems, _ = stratified_split_items()
    ss = (STRAT_SPLIT,)
    res = {"definitions": {
        "value_bars_4d": "MLP mean - std > control + 0.10 and > 0.10 absolute, plus bootstrap vs control "
                         "lower bound > 0 on all three probe seeds (as specified in STEP 4d)",
        "value_verdict_4d_qualified": "rule-4d 'yes' -> 'yes, above input baseline' iff both paired CIs "
                                      "(vs one-hot, vs ordinal) have lower bound > 0 on all three probe seeds",
        "layerwise": "Transformer study1 seed0, stratified split, depth; encoder input = embedding * sqrt(d) + "
                     "positional encoding; layer outputs computed with the source key-padding mask"},
        "stratified_value": {}, "layerwise_depth": {}}

    # 1. Stratified value rows.
    acts_oh = {k: input_baseline_features(v) for k, v in sitems.items()}
    acts_or = {k: ordinal_baseline_features(v) for k, v in sitems.items()}
    ib_oh, prot_oh = baseline_block(acts_oh, ss, target="value")
    ib_or, prot_or = baseline_block(acts_or, ss, target="value")
    sv = {"input_baseline_onehot": ib_oh, "input_baseline_ordinal": ib_or, "rows": {}}
    print(f"[4d] value baselines: one-hot {ib_oh[STRAT_SPLIT]['linear']:.3f}/{ib_oh[STRAT_SPLIT]['mlp_mean']:.3f} "
          f"ordinal {ib_or[STRAT_SPLIT]['linear']:.3f}/{ib_or[STRAT_SPLIT]['mlp_mean']:.3f}", flush=True)
    for model_name in MODELS:
        for seed in SEEDS:
            for state in STATES[model_name]:
                fs = [Path(cache_dir) / f"{model_name}_study1_seed{seed}_{state_cache_tag(model_name, state)}_strat{STRAT_SEED}_{k}.npz"
                      for k in sitems]
                assert all(f.exists() for f in fs), f"missing strat cache for {model_name} seed{seed} {state}"
                acts = strat_activations(model_name, seed, state, None, sitems, cache_dir)
                lin = linear_probe("value", acts["train"], {STRAT_SPLIT: acts[STRAT_SPLIT]})
                prot = run_mlp_protocol("value", acts, ss, HIDDEN)
                e = assemble_split("value", STRAT_SPLIT, acts, lin, prot)
                n_ex = len(acts[STRAT_SPLIT]["offsets"]) - 1
                e["verdict_rule_preregistered"] = e.pop("verdict")
                e["verdict_rule_4d"] = verdict_value_4d(e)
                e["ib_onehot_linear"] = ib_oh[STRAT_SPLIT]["linear"]
                e["ib_onehot_mlp"] = ib_oh[STRAT_SPLIT]["mlp_mean"]
                e["ib_ordinal_linear"] = ib_or[STRAT_SPLIT]["linear"]
                e["ib_ordinal_mlp"] = ib_or[STRAT_SPLIT]["mlp_mean"]
                e["bootstrap_ci_vs_onehot_baseline"] = paired_cis(prot, prot_oh, STRAT_SPLIT, n_ex, "value")
                e["bootstrap_ci_vs_ordinal_baseline"] = paired_cis(prot, prot_or, STRAT_SPLIT, n_ex, "value")
                e["verdict_4d"] = verdict_4c(e["verdict_rule_4d"], e["bootstrap_ci_vs_onehot_baseline"],
                                             e["bootstrap_ci_vs_ordinal_baseline"], e["verdict_rule_4d"])
                key = f"{model_name}_study1_seed{seed}_{state}_value"
                sv["rows"][key] = e
                print(f"[4d] stratified {key}: linear {e['linear']:.3f} mlp {e['mlp_mean']:.3f}±{e['mlp_std']:.3f} "
                      f"control {e['control']:.3f} -> {e['verdict_4d']} (pre-registered rule: "
                      f"{e['verdict_rule_preregistered']})", flush=True)
    res["stratified_value"] = sv

    # 2. Transformer layer-wise depth sanity, seed 0, stratified split.
    model = load_model("transformer", "study1", 0)
    feats = layerwise_acts(model, sitems)
    cached = strat_activations("transformer", 0, "pooled", None, sitems, cache_dir)
    pool_diff = max(float(np.abs(feats["mean_pool"][k]["X"] - cached[k]["X"]).max()) for k in sitems)
    acts_cp = {k: char_position_features(v) for k, v in sitems.items()}
    ib_cp, prot_cp = baseline_block(acts_cp, ss)
    lw = {"mean_pool_vs_cached_max_abs_diff": pool_diff, "majority": majority_baseline(acts_cp[STRAT_SPLIT]["depth"]),
          "baseline_char_position": ib_cp[STRAT_SPLIT], "features": {}}
    print(f"[4d] layer-wise: mean_pool vs cached strat activations max|diff| {pool_diff:.2e}; one-hot(char,pos) "
          f"{ib_cp[STRAT_SPLIT]['linear']:.3f}/{ib_cp[STRAT_SPLIT]['mlp_mean']:.3f}", flush=True)
    for k in LAYERWISE_FEATURES:
        a = feats[k]
        lin = linear_probe("depth", a["train"], {STRAT_SPLIT: a[STRAT_SPLIT]})
        prot = run_mlp_protocol("depth", a, ss, HIDDEN)
        e = assemble_split("depth", STRAT_SPLIT, a, lin, prot)
        e = {x: e[x] for x in ("linear", "mlp_mean", "mlp_std", "mlp_per_seed", "control", "control_train",
                               "majority", "bootstrap_ci", "n_examples", "n_positions")}
        e["dim"] = int(a[STRAT_SPLIT]["X"].shape[1])
        e["bootstrap_ci_vs_char_position_baseline"] = paired_cis(prot, prot_cp, STRAT_SPLIT,
                                                                 len(a[STRAT_SPLIT]["offsets"]) - 1)
        lw["features"][k] = e
        print(f"[4d] layer-wise {k}: linear {e['linear']:.3f} mlp {e['mlp_mean']:.3f} control {e['control']:.3f}",
              flush=True)
    res["layerwise_depth"] = lw
    with open(out_dir / "step4d_addenda.json", "w") as f:
        json.dump(res, f, indent=2)


# ---------------------------------------------------------------------------
# STEP 4e addenda: linear paired bootstraps on the stratified split; layer-wise
# depth for Transformer seeds 1-2 (+ decoder memory, all seeds); per-layer
# identity probes. Writes step4e_addenda.json only.
# ---------------------------------------------------------------------------

LAYERWISE_FEATURES_4E = LAYERWISE_FEATURES + ("memory",)


def linear_depth_predictions(train, test):
    from sklearn.linear_model import LogisticRegression
    probe = LogisticRegression(max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe.fit(train["X"], train["depth"])
    return probe.predict(test["X"])


def identity_probe(train, test, key):
    from sklearn.linear_model import LogisticRegression
    probe = LogisticRegression(max_iter=1000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        probe.fit(train["X"], train[key])
    acc = float(np.mean(probe.predict(test["X"]) == test[key]))
    conv = any("converge" in str(x.message).lower() for x in w)
    return {"accuracy": acc, "majority": majority_baseline(test[key]), "n_classes_train": int(len(set(train[key]))),
            "convergence_warning": conv}


def position_probe(train, test, key="position"):
    """identity_probe (one fit, same fields) plus per-expression correctness from example_stats, as the depth
    bootstraps use, for the positional-encoding sweep's paired bootstraps. Same LogisticRegression(max_iter=1000),
    unscaled; identity_probe and its committed outputs are untouched. `test` needs "offsets" (layerwise_acts
    provides it)."""
    from sklearn.linear_model import LogisticRegression
    probe = LogisticRegression(max_iter=1000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        probe.fit(train["X"], train[key])
    yhat = probe.predict(test["X"])
    return {"accuracy": float(np.mean(yhat == test[key])), "majority": majority_baseline(test[key]),
            "n_classes_train": int(len(set(train[key]))),
            "convergence_warning": any("converge" in str(x.message).lower() for x in w),
            "stats": example_stats("depth", test[key], yhat, test["offsets"])}


def layerwise_depth_row(a, prot_cp):
    lin = linear_probe("depth", a["train"], {STRAT_SPLIT: a[STRAT_SPLIT]})
    prot = run_mlp_protocol("depth", a, (STRAT_SPLIT,), HIDDEN)
    e = assemble_split("depth", STRAT_SPLIT, a, lin, prot)
    e = {x: e[x] for x in ("linear", "mlp_mean", "mlp_std", "mlp_per_seed", "control", "control_train",
                           "majority", "bootstrap_ci", "n_examples", "n_positions")}
    e["dim"] = int(a[STRAT_SPLIT]["X"].shape[1])
    e["bootstrap_ci_vs_char_position_baseline"] = paired_cis(prot, prot_cp, STRAT_SPLIT,
                                                             len(a[STRAT_SPLIT]["offsets"]) - 1)
    return e


def stage_addenda4e(out_dir, cache_dir):
    sitems, _ = stratified_split_items()
    res = {"definitions": {
        "linear_paired_bootstrap": "stratified split, LogisticRegression(max_iter=1000) on activations vs on the "
                                   "one-hot input features, and vs a constant predictor of the test split's majority "
                                   "class; 1,000 example-level resamples, 95% CI (one CI per row: the linear probe "
                                   "has no seed)",
        "memory": "nn.Transformer final encoder LayerNorm applied to layer 3: the tensor the decoder cross-attends to",
        "identity_probes": "LogisticRegression(max_iter=1000), unscaled, as Phase 3: current-character identity "
                           "(token id) and position index, accuracy on the stratified test split"},
        "linear_ci": {}, "layerwise_depth": {}, "identity": {}}

    # 1. Linear paired bootstraps.
    acts_oh = {k: input_baseline_features(v) for k, v in sitems.items()}
    test = acts_oh[STRAT_SPLIT]
    n_ex = len(test["offsets"]) - 1
    y = test["depth"]
    pred_oh = linear_depth_predictions(acts_oh["train"], test)
    st_oh = example_stats("depth", y, pred_oh, test["offsets"])
    maj_cls = majority_class(y)
    st_maj = example_stats("depth", y, np.full_like(y, maj_cls), test["offsets"])
    res["linear_ci"]["onehot_linear"] = float(np.mean(pred_oh == y))
    res["linear_ci"]["majority"] = majority_baseline(y)
    res["linear_ci"]["rows"] = {}
    for model_name in MODELS:
        for seed in SEEDS:
            for state in STATES[model_name]:
                acts = strat_activations(model_name, seed, state, None, sitems, cache_dir)
                assert np.array_equal(acts[STRAT_SPLIT]["depth"], y)
                pred = linear_depth_predictions(acts["train"], acts[STRAT_SPLIT])
                st_a = example_stats("depth", y, pred, test["offsets"])
                key = f"{model_name}_study1_seed{seed}_{state}_depth"
                res["linear_ci"]["rows"][key] = {
                    "linear": float(np.mean(pred == y)),
                    "ci_vs_onehot_linear": paired_bootstrap("depth", st_a, st_oh, n_ex, BOOT_SEED),
                    "ci_vs_majority": paired_bootstrap("depth", st_a, st_maj, n_ex, BOOT_SEED)}
                r = res["linear_ci"]["rows"][key]
                print(f"[4e] linear CI {key}: {r['linear']:.3f} vs one-hot {r['ci_vs_onehot_linear']} "
                      f"vs majority {r['ci_vs_majority']}", flush=True)

    # 2 + 3. Layer-wise depth (seeds 1-2 all features; memory for every seed) and identity probes.
    acts_cp = {k: char_position_features(v) for k, v in sitems.items()}
    _, prot_cp = baseline_block(acts_cp, (STRAT_SPLIT,))
    for seed in SEEDS:
        model = load_model("transformer", "study1", seed)
        feats = layerwise_acts(model, sitems, features=LAYERWISE_FEATURES_4E, with_ids=True)
        cached = strat_activations("transformer", seed, "pooled", None, sitems, cache_dir)
        pool_diff = max(float(np.abs(feats["mean_pool"][k]["X"] - cached[k]["X"]).max()) for k in sitems)
        lw = {"mean_pool_vs_cached_max_abs_diff": pool_diff, "features": {}}
        for k in LAYERWISE_FEATURES_4E:
            if seed == 0 and k != "memory":
                continue  # seed 0 non-memory rows are in step4d_addenda.json
            lw["features"][k] = layerwise_depth_row(feats[k], prot_cp)
            e = lw["features"][k]
            print(f"[4e] layer-wise seed{seed} {k}: linear {e['linear']:.3f} mlp {e['mlp_mean']:.3f} "
                  f"control {e['control']:.3f}", flush=True)
        res["layerwise_depth"][f"seed{seed}"] = lw
        idn = {}
        for k in LAYERWISE_FEATURES_4E:
            idn[k] = {t: identity_probe(feats[k]["train"], feats[k][STRAT_SPLIT], t) for t in ("char", "position")}
            print(f"[4e] identity seed{seed} {k}: char {idn[k]['char']['accuracy']:.3f} "
                  f"position {idn[k]['position']['accuracy']:.3f}", flush=True)
        res["identity"][f"seed{seed}"] = idn
        del model
    with open(out_dir / "step4e_addenda.json", "w") as f:
        json.dump(res, f, indent=2)


# ---------------------------------------------------------------------------
# STEP 4f addenda: final-position final-answer probes. The Phase 3 addendum
# (analysis_v2.ipynb cell 32) reads each encoder state at the final real
# position only; for the Transformer that state is the mean over the 3 encoder
# layers (no pooling over positions), extracted without the pad mask.
# Writes step4f_addenda.json only.
# ---------------------------------------------------------------------------

FINAL_INSTABILITY = -0.5  # held-out R² below this is flagged as small-n instability


# Source: analysis_v2.ipynb cell 32 (build_final_answer_dataset), verbatim.
def build_final_answer_dataset(state_fn, items):
    X, y = [], []
    for item in items:
        expr = item["input"]
        src = encode_input(expr)
        states = state_fn(src)
        final_pos = len(expr) - 1
        X.append(states[final_pos])
        y.append(int(item["output"]))
    return np.array(X), np.array(y)


def final_answer_acts(state_fn, items_by_split, cache_file=None):
    """{split: dict(X, value, depth, offsets)}: one vector per example, so offsets = 0..n."""
    if cache_file is not None and Path(cache_file).exists():
        z = np.load(cache_file, allow_pickle=False)
        return {s: {"X": z[f"{s}__X"], "value": z[f"{s}__y"], "depth": np.zeros(len(z[f"{s}__y"]), dtype=np.int64),
                    "offsets": np.arange(len(z[f"{s}__y"]) + 1)} for s in items_by_split}
    out, flat = {}, {}
    for s, its in items_by_split.items():
        X, y = build_final_answer_dataset(state_fn, its)
        out[s] = {"X": X, "value": y, "depth": np.zeros(len(y), dtype=np.int64), "offsets": np.arange(len(y) + 1)}
        flat[f"{s}__X"], flat[f"{s}__y"] = X, y
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **flat)
    return out


def final_length_features(items_by_split):
    """one-hot(expression length) only: the template control for a final-answer probe."""
    out = {}
    for s, its in items_by_split.items():
        X = np.zeros((len(its), MAX_INPUT_LEN + 1), dtype=np.float32)
        for i, it in enumerate(its):
            X[i, len(it["input"])] = 1.0
        y = np.array([int(it["output"]) for it in its])
        out[s] = {"X": X, "value": y, "depth": np.zeros(len(y), dtype=np.int64), "offsets": np.arange(len(y) + 1)}
    return out


def ridge_final(acts, splits):
    """Ridge(alpha=1.0) on raw features, as cell 32. Returns held-out R² per split and train R²."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    probe = Ridge(alpha=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe.fit(acts["train"]["X"], acts["train"]["value"])
    out = {s: float(r2_score(acts[s]["value"], probe.predict(acts[s]["X"]))) for s in splits}
    out["train"] = float(r2_score(acts["train"]["value"], probe.predict(acts["train"]["X"])))
    return out


def final_flags(e, ridge):
    flags = []
    if ridge[STRAT_SPLIT] < FINAL_INSTABILITY:
        flags.append(f"Ridge {ridge[STRAT_SPLIT]:.3f}")
    for k, v in zip(PROBE_SEEDS, e["mlp_per_seed"]):
        if v < FINAL_INSTABILITY:
            flags.append(f"MLP s{k} {v:.3f}")
    for k, v in zip(PROBE_SEEDS, e["control_per_seed"]):
        if v < FINAL_INSTABILITY:
            flags.append(f"control s{k} {v:.3f}")
    return flags


def stage_addenda4f(out_dir, cache_dir):
    import sklearn
    import platform
    t0 = time.time()
    res = {"definitions": {
        "addendum_pooling": "analysis_v2.ipynb cell 32: state at the final real position (len(expr) - 1) only; the "
                            "Transformer state is transformer_encoder_states_mean_pooled, i.e. the mean over the 3 "
                            "encoder layers at each position (pooling over layers, not over positions), unmasked; "
                            "Ridge(alpha=1.0) on raw features; target = int(output)",
        "restatement": "identical, except the Transformer layer outputs are computed with the source key-padding "
                       "mask (transformer_encoder_states_mean_pooled_masked)",
        "stratified": "strat0 split (300 train / 150 test single vectors); Ridge as cell 32; MLP protocol as the "
                      "grid (3 probe seeds, 240/60 carve-out, refit on 300) with the shuffled-label control and the "
                      "example-level paired bootstrap; template control = one-hot(expression length)",
        "instability_flag": f"held-out R² < {FINAL_INSTABILITY}: small-n instability (n = 300 training vectors), "
                            "as the paper reads the -2.99"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "device": DEVICE, "platform": platform.platform()},
        "matched_transformer": {}, "stratified": {}}

    # 1. Matched protocol: Transformer final-position, masked restatement (+ unmasked rerun as the gate).
    items = load_split_items("study1")
    splits = eval_split_names("study1")
    stored = json.loads((PHASE3_DIR / "phase3_addendum_final_answer_results.json").read_text())
    stored = stored["transformer_meanpooled_final"]
    model = load_model("transformer", "study1", 0)
    m = {}
    for tag, state in (("unmasked", "pooled_unmasked"), ("masked", "pooled")):
        fn = STATE_FNS[("transformer", state)]
        acts = final_answer_acts(lambda s_: fn(model, s_), items,
                                 Path(cache_dir) / f"final_transformer_study1_seed0_{state_cache_tag('transformer', state)}_matched.npz")
        m[tag] = ridge_final(acts, splits)
    res["matched_transformer"] = {
        s: {"stored_unmasked": stored[s]["value_r2"], "rerun_unmasked": m["unmasked"][s], "masked": m["masked"][s],
            "n_examples": stored[s]["n_examples"]} for s in splits}
    res["matched_transformer"]["train_r2"] = {"unmasked": m["unmasked"]["train"], "masked": m["masked"]["train"]}
    res["matched_transformer"]["rerun_max_abs_diff_vs_stored"] = max(
        abs(m["unmasked"][s] - stored[s]["value_r2"]) for s in splits)
    print("[4f] matched Transformer final-position R² (stored / unmasked rerun / masked): " + " ".join(
        f"{s}={stored[s]['value_r2']:.3f}/{m['unmasked'][s]:.3f}/{m['masked'][s]:.3f}" for s in splits), flush=True)
    del model

    # 2. Stratified split: LSTM h, c and Transformer (masked), all checkpoint seeds.
    sitems, _ = stratified_split_items()
    ss = (STRAT_SPLIT,)
    acts_len = final_length_features(sitems)
    ridge_len = ridge_final(acts_len, ss)
    prot_len = run_mlp_protocol("value", acts_len, ss, HIDDEN)
    e_len = assemble_split("value", STRAT_SPLIT, acts_len, {STRAT_SPLIT: ridge_len[STRAT_SPLIT]}, prot_len)
    res["stratified"]["length_baseline"] = {
        "ridge": ridge_len[STRAT_SPLIT], "ridge_train": ridge_len["train"], "mlp_mean": e_len["mlp_mean"],
        "mlp_std": e_len["mlp_std"], "mlp_per_seed": e_len["mlp_per_seed"], "mlp_train_per_seed": e_len["mlp_train_per_seed"],
        "control": e_len["control"], "control_per_seed": e_len["control_per_seed"],
        "control_train_per_seed": e_len["control_train_per_seed"], "bootstrap_ci": e_len["bootstrap_ci"],
        "flags": final_flags(e_len, ridge_len), "n_examples": e_len["n_examples"]}
    print(f"[4f] length baseline: Ridge {ridge_len[STRAT_SPLIT]:.3f} (train {ridge_len['train']:.3f}) MLP "
          f"{e_len['mlp_mean']:.3f}", flush=True)
    rows = {}
    for model_name in MODELS:
        for seed in SEEDS:
            model = None
            for state in STATES[model_name]:
                cf = Path(cache_dir) / f"final_{model_name}_study1_seed{seed}_{state_cache_tag(model_name, state)}_strat{STRAT_SEED}.npz"
                if model is None and not cf.exists():
                    model = load_model(model_name, "study1", seed)
                fn = STATE_FNS[(model_name, state)]
                acts = final_answer_acts(lambda s_: fn(model, s_), sitems, cf)
                rg = ridge_final(acts, ss)
                prot = run_mlp_protocol("value", acts, ss, HIDDEN)
                e = assemble_split("value", STRAT_SPLIT, acts, {STRAT_SPLIT: rg[STRAT_SPLIT]}, prot)
                row = {k: e[k] for k in ("linear", "mlp_mean", "mlp_std", "mlp_per_seed", "mlp_train_per_seed",
                                         "control", "control_per_seed", "control_train_per_seed", "bootstrap_ci",
                                         "epochs_chosen", "verdict", "n_examples")}
                row["ridge_train"] = rg["train"]
                row["bootstrap_ci_vs_length_baseline"] = paired_cis(prot, prot_len, STRAT_SPLIT, e["n_examples"], "value")
                row["flags"] = final_flags(e, rg)
                row["attention_mask"] = "masked" if model_name == "transformer" else "n/a (lstm)"
                key = f"{model_name}_study1_seed{seed}_{state}_final"
                rows[key] = row
                print(f"[4f] stratified {key}: Ridge {rg[STRAT_SPLIT]:.3f} (train {rg['train']:.3f}) MLP "
                      f"{e['mlp_mean']:.3f}±{e['mlp_std']:.3f} control {e['control']:.3f} flags {row['flags']}", flush=True)
            del model
    res["stratified"]["rows"] = rows
    res["wall_time_seconds"] = round(time.time() - t0, 1)
    with open(out_dir / "step4f_addenda.json", "w") as f:
        json.dump(res, f, indent=2)


LSTM_FINAL_TOL = 0.02


def matched_lstm_final_position(cache_dir):
    """Matched protocol, LSTM study1 seed 0, h and c: the Phase 3 addendum's final-position
    Ridge (cell 32), reproduced, with the training R² the notebook did not record."""
    stored = json.loads((PHASE3_DIR / "phase3_addendum_final_answer_results.json").read_text())
    items = load_split_items("study1")
    splits = eval_split_names("study1")
    model = load_model("lstm", "study1", 0)
    out = {}
    for state in ("h", "c"):
        fn = STATE_FNS[("lstm", state)]
        acts = final_answer_acts(lambda s_: fn(model, s_), items,
                                 Path(cache_dir) / f"final_lstm_study1_seed0_{state}_matched.npz")
        r = ridge_final(acts, splits)
        ref = stored[f"lstm_{state}_final"]
        out[state] = {"train_r2": r["train"], "n_train": int(len(acts["train"]["value"])),
                      **{s: {"stored": ref[s]["value_r2"], "rerun": r[s], "delta": r[s] - ref[s]["value_r2"],
                             "n_examples": ref[s]["n_examples"]} for s in splits}}
        out[state]["max_abs_delta"] = max(abs(out[state][s]["delta"]) for s in splits)
        out[state]["within_tolerance"] = out[state]["max_abs_delta"] <= LSTM_FINAL_TOL
    return out


def stage_addenda4g(out_dir, cache_dir):
    import sklearn
    import platform
    t0 = time.time()
    path = out_dir / "step4f_addenda.json"
    a = json.loads(path.read_text())
    block = matched_lstm_final_position(cache_dir)
    for state, e in block.items():
        assert e["within_tolerance"], f"LSTM {state} final-position reproduction outside ±{LSTM_FINAL_TOL}: {e}"
        print(f"[4g] LSTM {state} final-position (stored / rerun, train R² {e['train_r2']:.3f}): " + " ".join(
            f"{s}={e[s]['stored']:.3f}/{e[s]['rerun']:.3f}" for s in eval_split_names("study1")), flush=True)
    block["definition"] = ("analysis_v2.ipynb cell 32 reproduced for lstm_h_final and lstm_c_final (study1 seed 0): "
                           "state at the final real position, Ridge(alpha=1.0) on raw features, trained on val[:300]; "
                           f"held-out R² must match the stored value within ±{LSTM_FINAL_TOL}; train_r2 is the fit on "
                           "those 300 training vectors")
    block["environment"] = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                            "sklearn": sklearn.__version__, "device": DEVICE}
    block["wall_time_seconds"] = round(time.time() - t0, 1)
    a["matched_lstm_final_position"] = block
    with open(path, "w") as f:
        json.dump(a, f, indent=2)


def step4f_sections(a):
    j = lambda d: "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in d.values())
    mt = a["matched_transformer"]
    lines = ["## STEP 4f: final-position final-answer probes", "",
             f"**Pooling.** {a['definitions']['addendum_pooling']}. The restatement is {a['definitions']['restatement']}. "
             f"Run on {a['environment']['device']} (torch {a['environment']['torch']}, sklearn {a['environment']['sklearn']}).", "",
             "### Matched protocol, Transformer seed 0 (Ridge R²)", "",
             f"Unmasked rerun vs stored: max |diff| {mt['rerun_max_abs_diff_vs_stored']:.2e}. Train R²: unmasked "
             f"{mt['train_r2']['unmasked']:.3f}, masked {mt['train_r2']['masked']:.3f}.", "",
             "| split | n | stored (unmasked) | unmasked rerun | masked (restated) |", "|---|---|---|---|---|"]
    for s in eval_split_names("study1"):
        e = mt[s]
        lines.append(f"| {s} | {e['n_examples']} | {e['stored_unmasked']:.3f} | {e['rerun_unmasked']:.3f} | {e['masked']:.3f} |")
    st = a["stratified"]
    lb = st["length_baseline"]
    lines += ["", "### Stratified split (strat0), final-position final answer (R²)", "",
              f"n = 300 train / 150 test single vectors. Template control, one-hot(length): Ridge {lb['ridge']:.3f} "
              f"(train {lb['ridge_train']:.3f}), MLP {lb['mlp_mean']:.3f} ± {lb['mlp_std']:.3f} (train "
              f"{', '.join(f'{x:.3f}' for x in lb['mlp_train_per_seed'])}), control {lb['control']:.3f}. Held-out "
              f"R² below {FINAL_INSTABILITY} is flagged as small-n instability.", "",
              "| row | Ridge (train) | MLP mean ± std | MLP train (s0, s1, s2) | control (train) | boot CI MLP − control | "
              "boot CI MLP − length baseline | rule verdict | flags |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, e in st["rows"].items():
        lines.append(f"| {k} | {e['linear']:.3f} ({e['ridge_train']:.3f}) | {e['mlp_mean']:.3f} ± {e['mlp_std']:.3f} | "
                     f"{', '.join(f'{x:.3f}' for x in e['mlp_train_per_seed'])} | {e['control']:.3f} "
                     f"({', '.join(f'{x:.3f}' for x in e['control_train_per_seed'])}) | {j(e['bootstrap_ci'])} | "
                     f"{j(e['bootstrap_ci_vs_length_baseline'])} | {e['verdict']} | "
                     f"{'; '.join(e['flags']) if e['flags'] else '–'} |")
    lstm = {k: e for k, e in st["rows"].items() if k.startswith("lstm")}
    above = [(k, sd) for k, e in lstm.items() for sd, (lo, hi) in e["bootstrap_ci"].items() if lo > 0]
    by_row = {}
    for k, sd in above:
        by_row.setdefault(k, []).append(sd.replace("seed", ""))
    all3 = [k for k, e in lstm.items() if all(lo > 0 for lo, hi in e["bootstrap_ci"].values())]
    n_len = sum(len(e["bootstrap_ci_vs_length_baseline"]) for e in st["rows"].values())
    excl = sum(1 for e in st["rows"].values() for lo, hi in e["bootstrap_ci_vs_length_baseline"].values() if lo > 0 or hi < 0)
    desc = "; ".join(f"{k.replace('lstm_study1_', '').replace('_final', '').replace('_', ' ')} on probe seed{'s' if len(v) > 1 else ''} "
                     f"{' and '.join(v)}" for k, v in by_row.items())
    lines += ["", f"**Paired CIs.** {len(above)} of the {sum(len(e['bootstrap_ci']) for e in lstm.values())} LSTM "
              f"seed-level intervals against the control lie above zero ({desc}), "
              + (f"no row on all three probe seeds" if not all3 else f"rows on all three probe seeds: {', '.join(all3)}")
              + f", and {'none' if not excl else excl} of the {n_len} intervals against the length baseline "
              f"exclude{'s' if excl in (0, 1) else ''} zero."]
    ml = a.get("matched_lstm_final_position")
    if ml:
        sp = eval_split_names("study1")
        lines += ["", "**Matched LSTM seed 0, final position (STEP 4g).** Stored held-out R² reproduced (max |delta| h "
                  f"{ml['h']['max_abs_delta']:.1e}, c {ml['c']['max_abs_delta']:.1e}); training R² on the 300 vectors: "
                  f"h {ml['h']['train_r2']:.3f}, c {ml['c']['train_r2']:.3f}, against held-out h "
                  f"{min(ml['h'][s]['rerun'] for s in sp):.3f} to {max(ml['h'][s]['rerun'] for s in sp):.3f} and c "
                  f"{min(ml['c'][s]['rerun'] for s in sp):.3f} to {max(ml['c'][s]['rerun'] for s in sp):.3f}."]
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Report: summary.md + figures
# ---------------------------------------------------------------------------

VERDICT_RULE = """\
## Pre-registered verdict rule

- **Unit of inference is the expression, not the position.** Positions inside one expression are correlated, so n is 150 examples per held-out split (2,945 positions on the Study 1 ID split). Every test resamples whole expressions with all their positions.
- **Significance.** For each probe seed, a 1,000-resample example-level bootstrap of the paired held-out difference (MLP minus control). Pass iff the 95% CI lower bound exceeds 0 for all three seeds.
- **Effect size, depth.** MLP mean minus std exceeds control + 0.05. At n = 150 and p near 0.73 the example-level standard error is about 0.037, so 0.05 alone is roughly 1.4 standard errors and carries no significance; the bootstrap carries that. 0.05 is kept as the effect floor because it is triple the best linear margin over majority (+0.017, LSTM h, ID). Sanity bar: MLP mean minus std also exceeds majority.
- **Effect size, value.** MLP R² exceeds control + 0.10 and exceeds 0.10 absolute, with the same bootstrap on the paired R² difference. 0.10 is five times the best linear value (0.018).
- **Control check, not an assumption.** The shuffled-label R² is verified on every split. Expected at or below 0.02; any control R² above 0.05 is reported below before any verdict.
- **Verdict per target and split.** "yes" iff the bootstrap and both effect bars pass; else "no". Where the MLP beats linear by more than 0.05 but fails any bar: "signal, below threshold". The bootstrap CI (per probe seed) is reported beside every verdict.

## Caveats

Probes are correlational and sit beside the ablation and attention evidence. An MLP null at these widths narrows the paper's probe-scoping disjunction to "distributed, beyond an MLP of these widths, or not represented"; it does not prove absence.

Study 2 rows are **exploratory**. transformer/study2/seed0 and seed2 are **unfit** (train acc 5.85% / 14.51%): run, not interpreted.
Transformer activations are extracted **with** the source key-padding mask (as the model computes them); see the bridge table for the notebook's unmasked extraction.
"""


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def load_grid(out_dir):
    rows = []
    for model_name in MODELS:
        for study in STUDIES:
            for seed in SEEDS:
                for state in STATES[model_name]:
                    for target in TARGETS:
                        p = out_dir / out_name(model_name, study, seed, state, target)
                        if not p.exists():
                            continue
                        block = json.loads(p.read_text())[model_name]
                        for s, e in block.items():
                            rows.append(dict(model=model_name, study=study, seed=seed, state=state,
                                             target=target, split=s, **e))
    return rows


def step4b_header(rows, add, add4c=None):
    c1 = add["ceilings"]["study1"]
    ctrl = [r for r in rows if r["study"] == "study1" and r["target"] == "depth" and r["split"] == "ID_val_heldout"]
    ep = [x for r in ctrl for x in r["epochs_chosen"]["control"]]
    ct = [x for r in ctrl for x in r["control_train_per_seed"]]
    unfit = [r for r in rows if r["fit_status"] != "ok"]
    unfit_yes = sum(r["verdict"] == "yes" for r in unfit)
    ds = add["distinct_depth_sequences"]["study2"]
    return [
        "## STEP 4b header notes", "",
        f"**Label-space cap.** Every depth probe, linear and MLP, is trained on val[:300], whose depths are "
        f"{{{', '.join(map(str, c1['training_classes']))}}} in both studies, so its output space is those classes and "
        f"any position whose true depth lies outside them cannot be predicted. Per split, the ceiling is 1 minus "
        f"the share of such positions (table below), compared with the bar majority + 0.05. Where the ceiling is "
        f"below the bar (Study 1 ops6, ops7) a depth verdict of \"no\" reads \"structurally capped\": the test "
        f"could not have passed. Study 1 ops5 reads \"no (ceiling within x of bar)\". The same cap applies to the "
        f"Phase 3 linear depth rows (LogisticRegression trained on the same classes).", "",
        f"**Why an input baseline (item 1).** Study 2 is near-templated: {ds['train']} distinct depth sequences "
        f"across its 300 training expressions and {ds['ID_val_heldout']} across its 150 ID expressions. The two "
        f"unfit Study 2 Transformers (train acc 5.85% / 14.51%) still pass the rule on {unfit_yes} of "
        f"{len(unfit)} rows. Both suggest depth can be read off surface position and length without any learned "
        f"representation, so every depth row is compared with an MLP on one-hot(char) + one-hot(position) + "
        f"one-hot(length) features, with no prefix and no model activations.", "",
        "**Pre-registered reading for the input baseline.** If the activation MLP does not beat the input-baseline "
        "MLP with bootstrap lower bound above 0 on all three probe seeds, the Study 1 LSTM depth ID result is "
        "positional and is not a representation finding (verdict \"yes, not above input baseline\"). From STEP 4c, "
        "Study 1 \"above input baseline\" requires this for both input baselines, one-hot and ordinal.", "",
        "**What the matched \"ID\" split is.** In the matched protocol the probe trains on 300 two-operation "
        "expressions (lengths 13 to 16) and is tested on a slice that is 117 three-operation and 33 two-operation "
        "(lengths 13 to 23), so the \"ID\" test is a 2-to-3-operation transfer for the probe and the early-stop set "
        "never saw a 3-operation example; this applies to the Phase 3 linear rows, which used the same split. For "
        "the same reason the one-hot input baseline's 0.743 on that split is a floor: its length bits for lengths "
        "19 to 23 were never active in training. The stratified split (STEP 4c) answers the in-distribution "
        "question.", "",
        f"**Control.** Study 1 depth controls stop at epoch {min(ep)} to {max(ep)} with control_train "
        f"{min(ct):.2f} to {max(ct):.2f} against a {c1['training_majority']:.3f} training majority, so partial "
        f"memorization only.", ""]


def step4b_sections(add):
    lines = ["## Label-space ceilings (depth)", "",
             "| study | split | share outside training classes | ceiling | majority | bar (majority + 0.05) | ceiling − bar |",
             "|---|---|---|---|---|---|---|"]
    for study, c in add["ceilings"].items():
        for s in eval_split_names(study):
            e = c[s]
            lines.append(f"| {study} | {s} | {e['share_outside_training_classes']:.3f} | {e['ceiling']:.3f} | "
                         f"{e['majority']:.3f} | {e['bar']:.3f} | {e['ceiling'] - e['bar']:+.3f} |")
    lines += ["", "## Input baseline (depth): one-hot char + position + length, no model", "",
              "| study | split | linear | MLP mean ± std | control | boot CI MLP − control (s0; s1; s2) | majority |",
              "|---|---|---|---|---|---|---|"]
    for study, ib in add["input_baseline"].items():
        for s, e in ib.items():
            ci = "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in e["bootstrap_ci"].values())
            lines.append(f"| {study} | {s} | {fmt(e['linear'])} | {fmt(e['mlp_mean'])} ± {fmt(e['mlp_std'])} | "
                         f"{fmt(e['control'])} | {ci} | {fmt(e['majority'])} |")
    diffs = [r["rerun_max_abs_diff_vs_stored_mlp"] for r in add["rows"].values()]
    lines += ["", f"Activation-MLP depth fits were rerun to pair them with the input baseline; max |rerun − stored| "
                  f"over {len(diffs)} depth files: {max(diffs):.2e}.", ""]
    n = add["noop_check"]
    lines += ["## Masking no-op check (Transformer study1 seed0, ID depth, linear)", "",
              f"Masked and unmasked linear depth predictions differ at {n['positions_prediction_differs']} of "
              f"{n['n_positions']} positions (masked correct at {n['differ_masked_correct']}, unmasked correct at "
              f"{n['differ_unmasked_correct']}); accuracy {n['masked_acc']:.4f} vs {n['unmasked_acc']:.4f}. "
              f"Activations differ by max {n['max_abs_activation_diff_ID']:.3f}, mean "
              f"{n['mean_abs_activation_diff_ID']:.3f}. Masked run read {', '.join(Path(f).name for f in n['masked_cache_files'])} "
              f"({n['masked_extraction_fn']}).", ""]
    if add["epoch_cap"]:
        lines += ["## Epoch cap (LSTM depth fits whose chosen epoch is ≥ 191)", "",
                  "| fit | chosen epoch (stored / rerun) | stopped by | stop loss at best | final | final-epoch Δ | mean Δ last 10 | still falling |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, v in add["epoch_cap"].items():
            lines.append(f"| {k} | {v['chosen_epoch_stored']} / {v['chosen_epoch_rerun']} | {v['stopped_by']} | "
                         f"{v['stop_loss_at_best']:.4f} | {v['stop_loss_final']:.4f} | {v['final_epoch_delta']:+.5f} | "
                         f"{v['mean_delta_last_10']:+.5f} | {v['still_falling_at_final_epoch']} |")
        lines.append("")
    return lines


def step4c_sections(a, rows, add4b):
    lines = ["## STEP 4c: ordinal input baseline (matched protocol, Study 1 depth)", "",
             "| split | linear | MLP mean ± std | control | boot CI MLP − control (s0; s1; s2) | majority |",
             "|---|---|---|---|---|---|"]
    for s_, e in a["ordinal_baseline_matched"].items():
        ci = "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in e["bootstrap_ci"].values())
        lines.append(f"| {s_} | {fmt(e['linear'])} | {fmt(e['mlp_mean'])} ± {fmt(e['mlp_std'])} | {fmt(e['control'])} | "
                     f"{ci} | {fmt(e['majority'])} |")
    diffs = [r["rerun_max_abs_diff_vs_stored_mlp"] for r in a["rows_matched"].values()]
    lines += ["", f"Activation-MLP depth fits rerun for pairing; max |rerun − stored| over {len(diffs)} files: "
                  f"{max(diffs):.2e}.", ""]
    st = a["stratified"]
    sp, c = st["split"], st["ceiling"]
    oh, orl = st["input_baseline_onehot"][STRAT_SPLIT], st["input_baseline_ordinal"][STRAT_SPLIT]
    lines += ["## STEP 4c: stratified split (Study 1 depth, in-distribution)", "",
              f"Seeded (seed {STRAT_SEED}) random 300/150 split of study1 val.json, stratified by operation count: "
              f"train {sp['train_ops']} (carve-out {sp['carve_ops']}, early-stop {sp['stop_ops']}), test {sp['test_ops']}; "
              f"train lengths {sp['train_lengths'][0]}–{sp['train_lengths'][-1]}, test lengths "
              f"{sp['test_lengths'][0]}–{sp['test_lengths'][-1]}. Training depth classes {c['training_classes']}; "
              f"test ceiling {c['ceiling']:.3f}, majority {c['majority']:.3f}. Input baselines on this split: "
              f"one-hot linear {fmt(oh['linear'])}, MLP {fmt(oh['mlp_mean'])} ± {fmt(oh['mlp_std'])} "
              f"(control {fmt(oh['control'])}); ordinal linear {fmt(orl['linear'])}, MLP {fmt(orl['mlp_mean'])} ± "
              f"{fmt(orl['mlp_std'])} (control {fmt(orl['control'])}). The matched columns repeat the Phase 3 "
              f"comparison rows (ID_val_heldout) for the same checkpoint and state.", "",
              "| model | seed | state | protocol | linear | mlp | control | ctrl_train | majority | ib_onehot_mlp | ib_ordinal_mlp | boot CI vs control | boot CI vs one-hot | boot CI vs ordinal | n_ex | n_pos | rule verdict | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    j = lambda d: "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in d.values())
    for key, e in st["rows"].items():
        m, seed, state = key.split("_")[0], int(key.split("_seed")[1][0]), key.split("_")[3]
        mr = next((r for r in rows if r["model"] == m and r["study"] == "study1" and r["seed"] == seed
                   and r["state"] == state and r["target"] == "depth" and r["split"] == "ID_val_heldout"), None)
        lines.append(f"| {m} | {seed} | {state} | stratified | {fmt(e['linear'])} | {fmt(e['mlp_mean'])} ± "
                     f"{fmt(e['mlp_std'])} | {fmt(e['control'])} | {fmt(e['control_train'])} | {fmt(e['majority'])} | "
                     f"{fmt(e['ib_onehot_mlp'])} | {fmt(e['ib_ordinal_mlp'])} | {j(e['bootstrap_ci'])} | "
                     f"{j(e['bootstrap_ci_vs_onehot_baseline'])} | {j(e['bootstrap_ci_vs_ordinal_baseline'])} | "
                     f"{e['n_examples']} | {e['n_positions']} | {e['verdict_rule']} | {e['verdict_4c']} |")
        if mr:
            q4 = a["rows_matched"].get(f"{m}_study1_seed{seed}_{state}_depth", {}).get("ID_val_heldout", {})
            q4b = add4b.get("rows", {}).get(f"{m}_study1_seed{seed}_{state}_depth", {}).get("ID_val_heldout", {})
            lines.append(f"| {m} | {seed} | {state} | matched (ID) | {fmt(mr['linear'])} | {fmt(mr['mlp_mean'])} ± "
                         f"{fmt(mr['mlp_std'])} | {fmt(mr['control'])} | {fmt(mr['control_train'])} | "
                         f"{fmt(mr['majority'])} | {fmt(q4b.get('input_baseline_mlp'))} | "
                         f"{fmt(q4.get('ib_ordinal_mlp'))} | {j(mr['bootstrap_ci'])} | "
                         f"{j(q4b['bootstrap_ci_vs_input_baseline']) if q4b else '–'} | "
                         f"{j(q4['bootstrap_ci_vs_ordinal_baseline']) if q4 else '–'} | {mr['n_examples']} | "
                         f"{mr['n_positions']} | {mr['verdict']} | {q4.get('verdict_4c', '–')} |")
    lines.append("")
    return lines


def step4d_header(rows, a4c, a4d, a4e=None):
    st = a4c["stratified"]["rows"]
    lstm_c = [st[f"lstm_study1_seed{k}_c_depth"] for k in SEEDS]
    lin_c = [e["linear"] for e in lstm_c]
    gain = [e["mlp_mean"] - e["linear"] for e in lstm_c]
    c4c = a4c["stratified"]
    oh = c4c["input_baseline_onehot"][STRAT_SPLIT]
    orl = c4c["input_baseline_ordinal"][STRAT_SPLIT]
    om = a4c["ordinal_baseline_matched"]["ID_val_heldout"]
    mc = next(r for r in rows if r["model"] == "lstm" and r["study"] == "study1" and r["seed"] == 0
              and r["state"] == "c" and r["target"] == "depth" and r["split"] == "ID_val_heldout")
    count = lambda state: [k for k in SEEDS if st[f"lstm_study1_seed{k}_{state}_depth"]["verdict_4c"]
                           == "yes, above input baseline"]
    c_yes, h_yes = count("c"), count("h")
    lin_ci = ""
    id_pos = lambda k: float(np.mean([a4e["identity"][f"seed{sd}"][k]["position"]["accuracy"] for sd in SEEDS]))
    min_char = lambda: min(a4e["identity"][f"seed{sd}"][k]["char"]["accuracy"] for sd in SEEDS
                           for k in LAYERWISE_FEATURES_4E)
    if a4e:
        lc = [a4e["linear_ci"]["rows"][f"lstm_study1_seed{k}_c_depth"] for k in SEEDS]
        lin_ci = (" (paired 95% CIs, linear(c) − linear(one-hot): "
                  + "; ".join(f"[{r['ci_vs_onehot_linear'][0]:+.3f}, {r['ci_vs_onehot_linear'][1]:+.3f}]" for r in lc)
                  + "; linear(c) − majority: "
                  + "; ".join(f"[{r['ci_vs_majority'][0]:+.3f}, {r['ci_vs_majority'][1]:+.3f}]" for r in lc)
                  + ", seeds 0–2)")
    lines = [
        "## STEP 4d header notes", "",
        f"**Stratified split, linear.** On the stratified split the linear probe alone decodes depth from the LSTM "
        f"cell state at {min(lin_c):.3f}–{max(lin_c):.3f} against majority {c4c['ceiling']['majority']:.3f} and one-hot "
        f"linear {oh['linear']:.3f}{lin_ci}, so the in-distribution half of the paper's linear null does not hold for the "
        f"LSTM. The Phase 3 \"ID_val_heldout\" row was a 2-to-3-operation transfer test (train 300 two-operation, "
        f"test 117 three-operation and 33 two-operation), and its {mc['linear']:.3f} against {mc['majority']:.3f} "
        f"is a transfer null, not evidence of no representation.", "",
        f"**MLP over linear, LSTM c, stratified.** The MLP adds "
        f"{' / '.join(f'{g:+.3f}' for g in gain)} over linear across checkpoint seeds 0–2 (mean "
        f"{np.mean(gain):+.3f}), as measured.", "",
        f"**Input baselines.** The one-hot and ordinal baselines did not converge on the stratified split "
        f"({oh['mlp_mean']:.3f} vs {orl['mlp_mean']:.3f}), so the one-hot is the binding template baseline; on the "
        f"matched split the ordinal baseline scores below its own control ({om['mlp_mean']:.3f} vs "
        f"{om['control']:.3f}) and is not used as a bar there.", "",
        "**Limit.** Study 2 OOD \"above input baseline\" rows were not checked against an ordinal baseline.", "",
        f"**Stratified verdicts (depth).** LSTM c yes above input baseline {len(c_yes)} of 3 seeds; LSTM h "
        f"{len(h_yes)} of 3 (seed {', '.join(map(str, h_yes))}); " + (
            "Transformer no, pending the layer-wise check." if not a4d else
            f"Transformer no; on the stratified split depth decodability falls from "
            f"{a4d['layerwise_depth']['features']['encoder_input']['linear']:.3f} at the encoder input to "
            f"{a4d['layerwise_depth']['features']['layer3']['linear']:.3f} at layer 3 (seed 0 linear), so the encoder "
            + ("loses the information; seeds 1 and 2 in the layer-wise table." if not a4e else
               f"loses linearly decodable position ({id_pos('encoder_input'):.3f} at the input to "
               f"{id_pos('layer3'):.2f} at layer 3 and in the memory) and per-position depth (at majority by "
               f"layer 2 in all three seeds) while keeping token identity ({min_char():.3f} through the memory); "
               f"identity probes are linear only.")), ""]
    if a4d and a4e:
        lw = {sd: (a4d["layerwise_depth"]["features"] if sd == 0 else a4e["layerwise_depth"][f"seed{sd}"]["features"])
              for sd in SEEDS}
        order = ("encoder_input", "layer1", "layer2", "layer3")
        mono = all(all(lw[sd][order[i]]["linear"] > lw[sd][order[i + 1]]["linear"] for i in range(3)) for sd in SEEDS)
        peak = [sd for sd in SEEDS if max(order, key=lambda k: lw[sd][k]["mlp_mean"]) == "layer1"]
        cp_mlp = a4d["layerwise_depth"]["baseline_char_position"]["mlp_mean"]
        lines += [f"**Layer-wise, linear vs MLP.** Linear depth decodability "
                  + ("falls monotonically" if mono else "does NOT fall monotonically")
                  + f" in all three seeds. The MLP peaks at layer 1 in seed{'s' if len(peak) > 1 else ''} "
                  f"{' and '.join(map(str, peak))} ("
                  + ", ".join(f"{lw[sd]['layer1']['mlp_mean']:.3f}" for sd in peak)
                  + f", below the one-hot(char, position) MLP baseline of {cp_mlp:.3f}) before falling to majority "
                  f"at layer 2.", ""]
    if a4d:
        sv = a4d["stratified_value"]["rows"].values()
        r2 = [x for e in sv for x in (e["linear"], e["mlp_mean"])]
        top_key, top = max(((k, e) for k, e in a4d["stratified_value"]["rows"].items()), key=lambda kv: kv[1]["mlp_mean"])
        spans = all(lo < 0 < hi for e in sv for k in ("bootstrap_ci_vs_onehot_baseline", "bootstrap_ci_vs_ordinal_baseline")
                    for lo, hi in e[k].values())
        lines += [f"**Value, in-distribution.** Value is not decodable in-distribution by linear or MLP readers in "
                  f"either model (R² {min(r2):.2f} to {max(r2):.2f}; the {top['mlp_mean']:.3f} is "
                  f"{top_key.split('_')[0].capitalize()} seed {top_key.split('_seed')[1][0]} MLP, the one "
                  f"\"{top['verdict_4d']}\" row; "
                  + ("all CIs against input baselines span zero" if spans else
                     "NOT all CIs against input baselines span zero") + ").", ""]
    if a4d and a4e:
        f0 = a4d["layerwise_depth"]["features"]
        m0 = a4e["layerwise_depth"]["seed0"]["features"]["memory"]
        maj = a4d["layerwise_depth"]["majority"]
        lines += [f"**Decoder memory.** The decoder cross-attends only to the encoder output, which is layer 3 followed "
                  f"by the encoder's final LayerNorm; per-position depth there is at majority (linear "
                  f"{m0['linear']:.3f}, layer 3 before the norm {f0['layer3']['linear']:.3f}, vs majority {maj:.3f}, "
                  f"seed 0), so the first generated token is emitted from a memory with no per-position depth code. "
                  f"This is consistent with the Transformer's 70.7% first-error-at-step-0 rate from the behavioral "
                  f"analysis (Phase 1 failure histogram, seed 0: 3,463 of 4,901 failures); it is correlational, and "
                  f"whether the decoder could recover structure by aggregating over tokens is bounded by the identity "
                  f"probes in the layer-wise section.", ""]
    return lines


def step4d_sections(a):
    j = lambda d: "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in d.values())
    sv = a["stratified_value"]
    oh, orl = sv["input_baseline_onehot"][STRAT_SPLIT], sv["input_baseline_ordinal"][STRAT_SPLIT]
    lines = ["## STEP 4d: stratified split, value (Study 1, in-distribution)", "",
             f"Same strat0 split and cached activations as the stratified depth table above. Input baselines with "
             f"regression heads: one-hot Ridge R² {fmt(oh['linear'])}, MLP {fmt(oh['mlp_mean'])} ± {fmt(oh['mlp_std'])} "
             f"(control {fmt(oh['control'])}); ordinal Ridge {fmt(orl['linear'])}, MLP {fmt(orl['mlp_mean'])} ± "
             f"{fmt(orl['mlp_std'])} (control {fmt(orl['control'])}). Bars (STEP 4d): MLP mean − std > control + "
             f"0.10 and > 0.10, with the bootstrap vs control; the pre-registered rule (MLP mean, not mean − std) "
             f"is shown alongside.", "",
             "| model | seed | state | linear (Ridge) | mlp | control | ctrl_train | ib_onehot_mlp | ib_ordinal_mlp | boot CI vs control | boot CI vs one-hot | boot CI vs ordinal | n_ex | n_pos | pre-registered rule | 4d rule | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, e in sv["rows"].items():
        m, seed, state = key.split("_")[0], key.split("_seed")[1][0], key.split("_")[3]
        lines.append(f"| {m} | {seed} | {state} | {fmt(e['linear'])} | {fmt(e['mlp_mean'])} ± {fmt(e['mlp_std'])} | "
                     f"{fmt(e['control'])} | {fmt(e['control_train'])} | {fmt(e['ib_onehot_mlp'])} | "
                     f"{fmt(e['ib_ordinal_mlp'])} | {j(e['bootstrap_ci'])} | {j(e['bootstrap_ci_vs_onehot_baseline'])} | "
                     f"{j(e['bootstrap_ci_vs_ordinal_baseline'])} | {e['n_examples']} | {e['n_positions']} | "
                     f"{e['verdict_rule_preregistered']} | {e['verdict_rule_4d']} | {e['verdict_4d']} |")
    lw = a["layerwise_depth"]
    cp = lw["baseline_char_position"]
    lines += ["", "## STEP 4d: Transformer layer-wise depth sanity (study1 seed0, stratified split, report only)", "",
              "**Pre-registered reading.** If (a) the encoder input decodes near the one-hot(char, position) "
              "baseline and (b) decoding falls layer by layer, position information is lost inside the encoder and "
              "that is a finding; if (a) is at majority, the extraction or scaler is suspect.", "",
              f"Majority {lw['majority']:.3f}. One-hot(char, position) baseline, no length: linear "
              f"{fmt(cp['linear'])}, MLP {fmt(cp['mlp_mean'])} ± {fmt(cp['mlp_std'])} (control {fmt(cp['control'])}). "
              f"Mean pool recomputed here vs the cached stratified activations: max |diff| "
              f"{lw['mean_pool_vs_cached_max_abs_diff']:.2e}.", "",
              "| features | dim | linear | mlp | control | ctrl_train | boot CI vs control | boot CI vs one-hot(char, position) |",
              "|---|---|---|---|---|---|---|---|"]
    for k, e in lw["features"].items():
        lines.append(f"| {k} | {e['dim']} | {fmt(e['linear'])} | {fmt(e['mlp_mean'])} ± {fmt(e['mlp_std'])} | "
                     f"{fmt(e['control'])} | {fmt(e['control_train'])} | {j(e['bootstrap_ci'])} | "
                     f"{j(e['bootstrap_ci_vs_char_position_baseline'])} |")
    lines.append("")
    return lines


def step4e_sections(a4d, a4e):
    j1 = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    lc = a4e["linear_ci"]
    lines = ["## STEP 4e: linear paired bootstraps (stratified split, depth)", "",
             f"One-hot linear {lc['onehot_linear']:.3f}; majority {lc['majority']:.3f}. One CI per row (the linear "
             f"probe has no seed); 1,000 example-level resamples.", "",
             "| row | linear | CI linear − one-hot linear | CI linear − majority |", "|---|---|---|---|"]
    for k, r in lc["rows"].items():
        lines.append(f"| {k.replace('_study1', '').replace('_depth', '')} | {r['linear']:.3f} | "
                     f"{j1(r['ci_vs_onehot_linear'])} | {j1(r['ci_vs_majority'])} |")
    lines += ["", "## STEP 4e: Transformer layer-wise depth, all seeds (stratified split)", "",
              f"Majority {a4d['layerwise_depth']['majority']:.3f}; one-hot(char, position) baseline linear "
              f"{fmt(a4d['layerwise_depth']['baseline_char_position']['linear'])}, MLP "
              f"{fmt(a4d['layerwise_depth']['baseline_char_position']['mlp_mean'])}. Seed 0 rows other than memory "
              f"are from STEP 4d. memory = layer 3 after the encoder's final LayerNorm (what the decoder reads). "
              f"Cells: linear / MLP (control).", "",
              "| features | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|"]
    for k in LAYERWISE_FEATURES_4E:
        cells = []
        for seed in SEEDS:
            e = (a4d["layerwise_depth"]["features"].get(k) if seed == 0 and k != "memory"
                 else a4e["layerwise_depth"][f"seed{seed}"]["features"].get(k))
            cells.append("–" if e is None else f"{e['linear']:.3f} / {e['mlp_mean']:.3f} ({e['control']:.3f})")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "**Pre-registered reading (identity probes).** If character identity stays near 1.0 through layer 3 "
                  "while depth falls to majority, the encoder keeps the tokens but does not compute structure; if "
                  "character identity also decays, the decoder's memory loses the inputs themselves.", "",
              "## STEP 4e: per-layer identity probes, Transformer (stratified split, linear, report only)", "",
              "Cells: character-identity accuracy / position-index accuracy.", "",
              "| features | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|"]
    for k in LAYERWISE_FEATURES_4E:
        cells = [f"{a4e['identity'][f'seed{s_}'][k]['char']['accuracy']:.3f} / "
                 f"{a4e['identity'][f'seed{s_}'][k]['position']['accuracy']:.3f}" for s_ in SEEDS]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    i0 = a4e["identity"]["seed0"]["encoder_input"]
    lines += ["", f"Majority: character {i0['char']['majority']:.3f}, position {i0['position']['majority']:.3f}.", ""]
    return lines


def stage_report(out_dir, figures=True):
    rows = load_grid(out_dir)
    lines = ["# Nonlinear MLP probe: 12 checkpoints, matched to the Phase 3 linear protocol", "",
             VERDICT_RULE, ""]
    ap = out_dir / "step4b_addenda.json"
    add = json.loads(ap.read_text()) if ap.exists() else None
    ap4c = out_dir / "step4c_addenda.json"
    add4c = json.loads(ap4c.read_text()) if ap4c.exists() else None
    ap4d = out_dir / "step4d_addenda.json"
    add4d = json.loads(ap4d.read_text()) if ap4d.exists() else None
    if add:
        lines += step4b_header(rows, add, add4c)
    ap4e = out_dir / "step4e_addenda.json"
    add4e = json.loads(ap4e.read_text()) if ap4e.exists() else None
    if add4c:
        lines += step4d_header(rows, add4c, add4d, add4e)

    # Control R² check
    ctrl_rows = [r for r in rows if r["target"] == "value"]
    flagged = [r for r in ctrl_rows if max(r["control_per_seed"]) > CONTROL_R2_FLAG]
    above_exp = [r for r in ctrl_rows if max(r["control_per_seed"]) > CONTROL_R2_EXPECTED]
    lines += ["## Control R² check", "",
              f"{len(ctrl_rows)} value rows checked (3 control seeds each). "
              f"Max control R² over all rows and seeds: "
              f"{fmt(max((max(r['control_per_seed']) for r in ctrl_rows), default=None))}. "
              f"Rows with any seed > {CONTROL_R2_EXPECTED}: {len(above_exp)}. "
              f"Rows with any seed > {CONTROL_R2_FLAG}: {len(flagged)}.", ""]
    for r in flagged:
        lines.append(f"- FLAG: {r['model']}/{r['study']}/seed{r['seed']} {r['state']} {r['split']}: "
                     f"control R² per seed {[round(x, 3) for x in r['control_per_seed']]}")
    lines.append("")

    # Bridge table
    bp = out_dir / "transformer_study1_seed0_pooled_unmasked_bridge.json"
    if bp.exists():
        b = json.loads(bp.read_text())["transformer"]
        lines += ["## Transformer bridge (study1 seed0, linear probes)", "",
                  "| split | value R² Phase 3 stored | unmasked rerun | masked rerun | depth acc Phase 3 stored | unmasked rerun | masked rerun |",
                  "|---|---|---|---|---|---|---|"]
        for s, r in b.items():
            lines.append(f"| {s} | {fmt(r['value_r2_phase3_stored'])} | {fmt(r['value_r2_unmasked'])} | "
                         f"{fmt(r['value_r2_masked'])} | {fmt(r['depth_acc_phase3_stored'])} | "
                         f"{fmt(r['depth_acc_unmasked'])} | {fmt(r['depth_acc_masked'])} |")
        lines.append("")

    # 240 vs 300 comparison
    if rows:
        d = np.array([r["mlp_mean"] - r["mlp_240_trained"]["mean"] for r in rows])
        lines += ["## 240-trained (early-stopped) vs 300-refit", "",
                  f"Refit minus 240-trained, over {len(rows)} rows: mean {d.mean():+.4f}, "
                  f"median {np.median(d):+.4f}, min {d.min():+.4f}, max {d.max():+.4f}.", ""]

    if add:
        lines += step4b_sections(add)
    if add4c:
        lines += step4c_sections(add4c, rows, add or {})
    if add4d:
        lines += step4d_sections(add4d)
    if add4e and add4d:
        lines += step4e_sections(add4d, add4e)

    # Main table
    lines += ["## Results", "",
              "Metric: R² for value, accuracy for depth. mlp = 300-refit mean ± std over probe seeds 0–2; "
              "mlp240 = early-stopped 240-trained mean; control = permuted-label MLP held-out mean; "
              "ctrl_train = control on its own (permuted) training labels; boot CI vs control = 95% CI of paired "
              "MLP − control per probe seed (the verdict test); boot CI vs majority = 95% CI of paired MLP − "
              "constant predictor of the eval split's own majority class (depth only; reported, not part of "
              "the verdict). STEP 4b (depth only): ib_mlp / ib_linear = input baseline (one-hot char, "
              "position, length); boot CI vs input = 95% CI of paired MLP(activations) − MLP(input baseline) "
              "per probe seed; ib_ordinal_mlp / boot CI vs ordinal = STEP 4c ordinal input baseline (one-hot char, "
              "scalar position, length, position/length; Study 1 depth only); ceiling = label-space ceiling. "
              "verdict = 4c-qualified verdict for Study 1 depth rows, 4b-qualified elsewhere; rule verdict = the "
              "pre-registered rule's output before the qualifiers.", "",
              "| model | study | seed | state | target | split | linear | mlp | mlp240 | control | ctrl_train | majority | selectivity | boot CI vs control (s0; s1; s2) | boot CI vs majority (s0; s1; s2) | ib_mlp | ib_linear | boot CI vs input (s0; s1; s2) | ib_ordinal_mlp | boot CI vs ordinal (s0; s1; s2) | ceiling | n_ex | n_pos | rule verdict | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in r["bootstrap_ci"].values())
        ci_maj = ("–" if "bootstrap_ci_vs_majority" not in r else
                  "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in r["bootstrap_ci_vs_majority"].values()))
        v_rule = r["verdict"]
        ib_mlp = ib_lin = ci_in = ceil = "–"
        v = v_rule
        a = (add or {}).get("rows", {}).get(f"{r['model']}_{r['study']}_seed{r['seed']}_{r['state']}_depth", {})
        if r["target"] == "depth" and r["split"] in a:
            q = a[r["split"]]
            ib_mlp, ib_lin, ceil = fmt(q["input_baseline_mlp"]), fmt(q["input_baseline_linear"]), fmt(q["ceiling"])
            ci_in = "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in q["bootstrap_ci_vs_input_baseline"].values())
            v = q["verdict_qualified"]
        ib_ord = ci_ord = "–"
        m4c = (add4c or {}).get("rows_matched", {}).get(f"{r['model']}_{r['study']}_seed{r['seed']}_{r['state']}_depth", {})
        if r["target"] == "depth" and r["split"] in m4c:
            q4 = m4c[r["split"]]
            ib_ord = fmt(q4["ib_ordinal_mlp"])
            ci_ord = "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in q4["bootstrap_ci_vs_ordinal_baseline"].values())
            v = q4["verdict_4c"]
        if r["fit_status"] != "ok":
            v = f"{r['fit_status']}; rule: {v}"
        study = r["study"] + (" (exploratory)" if r["study_label"] == "exploratory" else "")
        lines.append(f"| {r['model']} | {study} | {r['seed']} | {r['state']} | {r['target']} | {r['split']} | "
                     f"{fmt(r['linear'])} | {fmt(r['mlp_mean'])} ± {fmt(r['mlp_std'])} | "
                     f"{fmt(r['mlp_240_trained']['mean'])} | {fmt(r['control'])} | {fmt(r['control_train'])} | "
                     f"{fmt(r['majority'])} | {fmt(r['selectivity'])} | {ci} | {ci_maj} | {ib_mlp} | {ib_lin} | "
                     f"{ci_in} | {ib_ord} | {ci_ord} | {ceil} | {r['n_examples']} | {r['n_positions']} | {v_rule} | {v} |")
    lines.append("")

    sp = out_dir / "capacity_sweep.json"
    if sp.exists():
        sw = json.loads(sp.read_text())
        lines += ["## Capacity sweep (Study 1 seed 0)", "",
                  "| model | state | target | split | " + " | ".join(f"h={h}" for h in SWEEP_HIDDEN) + " |",
                  "|---|---|---|---|" + "---|" * len(SWEEP_HIDDEN)]
        for model_name in MODELS:
            for state in STATES[model_name]:
                for target in TARGETS:
                    for s in eval_split_names("study1"):
                        cells = []
                        for h in SWEEP_HIDDEN:
                            e = sw.get(f"{model_name}_study1_seed0_{state}_{target}_h{h}")
                            cells.append("–" if e is None else
                                         f"{fmt(e['splits'][s]['mlp_mean'])} ({e['splits'][s]['verdict']})")
                        lines.append(f"| {model_name} | {state} | {target} | {s} | " + " | ".join(cells) + " |")
        lines.append("")

    ap4f = out_dir / "step4f_addenda.json"
    if ap4f.exists():
        lines += step4f_sections(json.loads(ap4f.read_text()))

    (out_dir / "summary.md").write_text("\n".join(lines))
    if figures:
        make_figures(rows, out_dir)
    print(f"[report] wrote summary.md ({len(rows)} rows)" + (" and figures" if figures else "; figures skipped"))


def make_figures(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = eval_split_names("study1")
    panels = [("lstm", "h"), ("lstm", "c"), ("transformer", "pooled")]
    for target, fname, series in (
            ("depth", "fig_depth_vs_length.png", ("linear", "mlp_mean", "control", "majority")),
            ("value", "fig_value_r2.png", ("linear", "mlp_mean", "control"))):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=(target == "depth"))
        for ax, (m, st) in zip(axes, panels):
            sub = [r for r in rows if r["model"] == m and r["state"] == st and r["target"] == target
                   and r["study"] == "study1"]
            if not sub:
                continue
            x = np.arange(len(splits))
            for key, label in zip(series, ("linear", "MLP (h=256)", "control (shuffled)", "majority")):
                vals = np.array([[next(r[key] for r in sub if r["seed"] == k and r["split"] == s)
                                  for s in splits] for k in SEEDS], dtype=float)
                ax.errorbar(x, vals.mean(0), yerr=vals.std(0), marker="o", capsize=3, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels(["ID"] + list(splits[1:]))
            title = {"h": "LSTM h", "c": "LSTM c", "pooled": "Transformer (masked)"}[st]
            ax.set_title(title)
            ax.set_xlabel("split (ops count)")
            ax.set_ylabel("depth accuracy" if target == "depth" else "value R²")
            ax.axhline(0, color="grey", lw=0.5) if target == "value" else None
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Study 1, {target}: mean ± std across checkpoint seeds 0–2 "
                     f"(MLP and control: probe-seed mean per checkpoint)")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------

def environment_record(out_dir, stage_times, checksums=None):
    import sklearn
    import huggingface_hub
    import subprocess
    path = out_dir / "environment.json"
    env = json.loads(path.read_text()) if path.exists() else {}
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    env.update({
        "probe_environment": {
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "sklearn": sklearn.__version__, "huggingface_hub": huggingface_hub.__version__,
            "device": DEVICE, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "reference_evaluation_environment": REFERENCE_EVAL_ENV,
        "determinism": {"torch.use_deterministic_algorithms": True,
                        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "tf32": False, "cudnn.benchmark": False},
        "repo_commit": commit,
        "hf": {"model_repo": MODEL_REPO, "model_commit": MODEL_REVISION,
               "data_repo": DATA_REPO, "data_commit": DATA_REVISION},
        "protocol": {"hidden": HIDDEN, "sweep_hidden": list(SWEEP_HIDDEN), "lr": LR, "weight_decay": WEIGHT_DECAY,
                     "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                     "probe_seeds": list(PROBE_SEEDS), "control_perm_seed": CONTROL_PERM_SEED,
                     "n_boot": N_BOOT, "boot_seed_base": BOOT_SEED, "n_train": N_TRAIN, "n_carve": N_CARVE,
                     "n_id_test": N_ID_TEST, "n_ood_test": N_OOD_TEST},
    })
    files = env.get("files", {})
    files.update(FILE_RECORDS)
    env["files"] = files
    meta = env.get("checkpoint_metadata", {})
    meta.update(CKPT_META)
    env["checkpoint_metadata"] = meta
    if checksums is not None:
        env["data_checksums_vs_CHECKSUMS_txt"] = checksums
    times = env.get("wall_time_seconds", {})
    for k, v in stage_times.items():  # never overwrite an earlier run's timing
        key, n = k, 2
        while key in times:
            key, n = f"{k} [run {n}]", n + 1
        times[key] = v
    env["wall_time_seconds"] = times
    with open(path, "w") as f:
        json.dump(env, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("bridge", "grid", "sweep", "addenda", "addenda4c", "addenda4d", "addenda4e", "addenda4f", "addenda4g", "report", "all"), default="all")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/nonlinear_probe_cache"))
    ap.add_argument("--only", nargs="*", help="checkpoint tags for --stage grid, e.g. lstm_study1_seed0")
    ap.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    ap.add_argument("--legacy-extraction", action="store_true",
                    help="Transformer states via the hand-written layer loop of the committed results "
                         "(reproduction tests only); default reads them through forward hooks")
    ap.add_argument("--no-env-record", action="store_true",
                    help="leave environment.json untouched (addenda4f records its environment in its own JSON)")
    ap.add_argument("--skip-figures", action="store_true", help="report stage: do not redraw the figures")
    args = ap.parse_args()
    if args.stage in ("addenda4f", "addenda4g"):
        args.no_env_record = True

    configure_device(args.device)
    global LEGACY_EXTRACTION
    LEGACY_EXTRACTION = args.legacy_extraction
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE} torch={torch.__version__} out={args.out_dir}")

    stages = ("bridge", "grid", "sweep", "addenda", "addenda4c", "addenda4d", "addenda4e", "addenda4f", "addenda4g", "report") if args.stage == "all" else (args.stage,)
    t = time.time()
    checksums = verify_data_checksums()
    bad = [k for k, v in checksums.items() if not v["pass"]]
    assert not bad, f"data checksum mismatch vs CHECKSUMS.txt: {bad}"
    times = {"download_and_checksums": round(time.time() - t, 1)}
    if not args.no_env_record:
        environment_record(args.out_dir, times, checksums)
    for st in stages:
        t = time.time()
        if st == "bridge":
            stage_bridge(args.out_dir, args.cache_dir)
        elif st == "grid":
            stage_grid(args.out_dir, args.cache_dir, only=args.only, force=args.force)
        elif st == "sweep":
            stage_sweep(args.out_dir, args.cache_dir, force=args.force)
        elif st == "addenda":
            stage_addenda(args.out_dir, args.cache_dir)
        elif st == "addenda4c":
            stage_addenda4c(args.out_dir, args.cache_dir)
        elif st == "addenda4d":
            stage_addenda4d(args.out_dir, args.cache_dir)
        elif st == "addenda4e":
            stage_addenda4e(args.out_dir, args.cache_dir)
        elif st == "addenda4f":
            stage_addenda4f(args.out_dir, args.cache_dir)
        elif st == "addenda4g":
            stage_addenda4g(args.out_dir, args.cache_dir)
        elif st == "report":
            stage_report(args.out_dir, figures=not args.skip_figures)
        key = st if not (st == "grid" and args.only) else f"grid[{','.join(args.only)}]"
        if not args.no_env_record:
            environment_record(args.out_dir, {key: round(time.time() - t, 1)})


if __name__ == "__main__":
    main()

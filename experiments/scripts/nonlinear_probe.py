"""Nonlinear (two-layer MLP) probe on all twelve v2 checkpoints, matched to the
Phase 3 linear-probe protocol in experiments/notebooks/analysis_v2.ipynb.

Revision plan Phase 0, referee item M5. For every checkpoint, encoder state,
target (running value, paren depth) and split (ID held-out + OOD), this runs:
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
  report  summary.md, figures (reads the JSONs written by grid/bridge/sweep).

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


def load_model(model_name, study, seed):
    path = hf_file(MODEL_REPO, f"{model_name}/{study}/seed{seed}/best_model.pt", "model")
    # weights_only=False: the checkpoint dict carries plain metadata alongside
    # tensors, and these are this project's own files at a pinned Hub commit.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "dataset_version" in ckpt:
        assert ckpt["dataset_version"] == "datasets_v2", (
            f"{model_name}/{study}/seed{seed}: dataset_version={ckpt['dataset_version']!r}")
    fp = FINGERPRINTS.get((model_name, study, seed))
    if fp is not None:
        assert ckpt["val_accuracy"] == fp, (
            f"{model_name.upper()} FINGERPRINT MISMATCH: val_accuracy={ckpt['val_accuracy']} (expected {fp})")
    if model_name == "transformer":
        # Same construction as notebook cell 3 and train_v2.py build_model (pad_idx set).
        model = create_transformer_model(
            vocab_size=tokenizer.vocab_size, d_model=256, nhead=8,
            num_encoder_layers=3, num_decoder_layers=3, pad_idx=pad_idx,
        )
    else:
        model = create_lstm_model(vocab_size=tokenizer.vocab_size, embedding_dim=128, hidden_size=256)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    CKPT_META[f"{model_name}/{study}/seed{seed}"] = {
        k: ckpt.get(k) for k in ("epoch", "val_accuracy", "val_loss", "seed", "dataset_version", "generator_commit")}
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


STATE_FNS = {
    ("lstm", "h"): lstm_encoder_states,
    ("lstm", "c"): lstm_cell_states,
    ("transformer", "pooled"): transformer_encoder_states_mean_pooled_masked,
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
        f = Path(cache_dir) / f"{model_name}_{study}_seed{seed}_{state}_{split}.npz"
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


def train_mlp(prep, Xtr, ytr, hidden, seed, n_epochs=None, Xstop=None, ystop=None):
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


def stage_report(out_dir):
    rows = load_grid(out_dir)
    lines = ["# Nonlinear MLP probe: 12 checkpoints, matched to the Phase 3 linear protocol", "",
             VERDICT_RULE, ""]

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

    # Main table
    lines += ["## Results", "",
              "Metric: R² for value, accuracy for depth. mlp = 300-refit mean ± std over probe seeds 0–2; "
              "mlp240 = early-stopped 240-trained mean; control = permuted-label MLP held-out mean; "
              "ctrl_train = control on its own (permuted) training labels; boot CI = 95% CI of paired "
              "MLP − control per probe seed.", "",
              "| model | study | seed | state | target | split | linear | mlp | mlp240 | control | ctrl_train | majority | selectivity | boot CI (s0; s1; s2) | n_ex | n_pos | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = "; ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in r["bootstrap_ci"].values())
        v = r["verdict"]
        if r["fit_status"] != "ok":
            v = f"{r['fit_status']}; rule: {v}"
        study = r["study"] + (" (exploratory)" if r["study_label"] == "exploratory" else "")
        lines.append(f"| {r['model']} | {study} | {r['seed']} | {r['state']} | {r['target']} | {r['split']} | "
                     f"{fmt(r['linear'])} | {fmt(r['mlp_mean'])} ± {fmt(r['mlp_std'])} | "
                     f"{fmt(r['mlp_240_trained']['mean'])} | {fmt(r['control'])} | {fmt(r['control_train'])} | "
                     f"{fmt(r['majority'])} | {fmt(r['selectivity'])} | {ci} | {r['n_examples']} | "
                     f"{r['n_positions']} | {v} |")
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

    (out_dir / "summary.md").write_text("\n".join(lines))
    make_figures(rows, out_dir)
    print(f"[report] wrote summary.md ({len(rows)} rows) and figures")


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
    times.update(stage_times)
    env["wall_time_seconds"] = times
    with open(path, "w") as f:
        json.dump(env, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("bridge", "grid", "sweep", "report", "all"), default="all")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/nonlinear_probe_cache"))
    ap.add_argument("--only", nargs="*", help="checkpoint tags for --stage grid, e.g. lstm_study1_seed0")
    ap.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    args = ap.parse_args()

    configure_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE} torch={torch.__version__} out={args.out_dir}")

    stages = ("bridge", "grid", "sweep", "report") if args.stage == "all" else (args.stage,)
    t = time.time()
    checksums = verify_data_checksums()
    bad = [k for k, v in checksums.items() if not v["pass"]]
    assert not bad, f"data checksum mismatch vs CHECKSUMS.txt: {bad}"
    times = {"download_and_checksums": round(time.time() - t, 1)}
    environment_record(args.out_dir, times, checksums)
    for st in stages:
        t = time.time()
        if st == "bridge":
            stage_bridge(args.out_dir, args.cache_dir)
        elif st == "grid":
            stage_grid(args.out_dir, args.cache_dir, only=args.only, force=args.force)
        elif st == "sweep":
            stage_sweep(args.out_dir, args.cache_dir, force=args.force)
        elif st == "report":
            stage_report(args.out_dir)
        key = st if not (st == "grid" and args.only) else f"grid[{','.join(args.only)}]"
        environment_record(args.out_dir, {key: round(time.time() - t, 1)})


if __name__ == "__main__":
    main()

"""Retraining entry point for the fixed pipeline (datasets_v2 + trainer fixes).

Does NOT touch the original main.py / datasets/ / results/ pipeline - this is
an additive parallel path, same philosophy as data/generate_controlled.py's
generate_study_datasets_v2(). Run one (model, study, seed) combination per
invocation; loop externally over the combinations you want (e.g. 2 models x
2 studies x 3 seeds = 12 runs).

Usage:
    python train_v2.py --model transformer --study 1 --seed 0
    python train_v2.py --model lstm --study 2 --seed 0 --num-epochs 1
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data.dataloader import MathDataPipeline
from data.tokenizer import create_tokenizer
from models.lstm import create_lstm_model
from models.transformer import create_transformer_model
from utils.trainer import train_model, calculate_accuracy

# Fixed v2 config - see the sequence-length audit that motivated this:
# max observed input char length across ALL datasets_v2 files is 51, max
# observed output char length is 5. These caps give real headroom instead of
# the old max_input_len=20 (which 35%+ of even in-distribution data already
# exceeded) / max_output_len=10.
TRAINING_CONFIG_V2 = {
    'max_input_len': 64,   # >= 60 required; observed max is 51
    'max_output_len': 12,  # theoretical max answer "-10000" (6 chars) + EOS(1) + margin(5)
    'learning_rate': {
        'lstm': 0.001,
        'transformer': 3e-4,  # raised from 1e-4 - safe with warmup
    },
    'warmup_steps': 400,
    'batch_size': 32,
    'num_epochs': 100,
    'early_stopping_patience': 25,
}

DATASET_VERSION = "datasets_v2"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_generator_commit(data_dir: Path, study: str) -> str:
    # Traceability: read the commit that actually generated this data file's
    # content, from its own v2 metadata header - not the training script's
    # current HEAD, which may differ if the repo has moved on since the data
    # was generated.
    train_file = data_dir / study / "train.json"
    payload = json.loads(train_file.read_text())
    return payload.get('generator_commit', 'unknown')


def build_model(model_type: str, vocab_size: int, pad_idx: int):
    if model_type == 'lstm':
        return create_lstm_model(vocab_size=vocab_size, embedding_dim=128, hidden_size=256)
    elif model_type == 'transformer':
        # pad_idx explicitly set (not None) - opts into the src/tgt padding
        # masks added to TransformerEncoderDecoder.forward for this training
        # path, without changing the default (pad_idx=None) behavior that
        # existing checkpoints/analysis code relies on.
        return create_transformer_model(
            vocab_size=vocab_size, d_model=256, nhead=8,
            num_encoder_layers=3, num_decoder_layers=3, pad_idx=pad_idx,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_ood(model, tokenizer, data_dir: Path, study: str, batch_size: int, device: str) -> dict:
    pipeline = MathDataPipeline(
        data_dir=str(data_dir), batch_size=batch_size,
        max_input_len=TRAINING_CONFIG_V2['max_input_len'],
        max_output_len=TRAINING_CONFIG_V2['max_output_len'],
    )
    results = {}
    if study == "study1":
        ood_files = [f"study1/ood_ops{n}.json" for n in (4, 5, 6, 7)]
    else:
        ood_files = ["study2/ood.json"]

    for f in ood_files:
        loader = pipeline.get_dataloaders_file(f, shuffle=False)
        acc = calculate_accuracy(model, loader, device, pad_idx=0)
        results[f] = acc
        print(f"  OOD accuracy [{f}]: {acc:.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrain on datasets_v2 with the fixed pipeline")
    parser.add_argument('--model', choices=['lstm', 'transformer'], required=True)
    parser.add_argument('--study', choices=['1', '2'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num-epochs', type=int, default=TRAINING_CONFIG_V2['num_epochs'])
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG_V2['batch_size'])
    parser.add_argument('--data-dir', type=str, default='datasets_v2')
    parser.add_argument('--device', type=str, default=None,
                         help="cpu/cuda/mps; default auto-detects")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    study_key = f"study{args.study}"
    data_dir = Path(args.data_dir)

    tokenizer = create_tokenizer()
    model = build_model(args.model, tokenizer.vocab_size, pad_idx=tokenizer.pad_idx)

    pipeline = MathDataPipeline(
        data_dir=str(data_dir), batch_size=args.batch_size,
        max_input_len=TRAINING_CONFIG_V2['max_input_len'],
        max_output_len=TRAINING_CONFIG_V2['max_output_len'],
    )
    train_loader = pipeline.get_dataloaders_file(f"{study_key}/train.json", shuffle=True)
    val_loader = pipeline.get_dataloaders_file(f"{study_key}/val.json", shuffle=False)

    generator_commit = get_generator_commit(data_dir, study_key)

    results_dir = Path("results_v2") / args.model / study_key / f"seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "best_model.pt"

    t_start = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=TRAINING_CONFIG_V2['learning_rate'][args.model],
        device=device,
        save_path=str(save_path),
        pad_idx=tokenizer.pad_idx,
        early_stopping_patience=TRAINING_CONFIG_V2['early_stopping_patience'],
        warmup_steps=TRAINING_CONFIG_V2['warmup_steps'],
        seed=args.seed,
        dataset_version=DATASET_VERSION,
        generator_commit=generator_commit,
    )
    elapsed = time.time() - t_start

    history_path = results_dir / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete in {elapsed:.1f}s. Checkpoint: {save_path}")

    # Reload best checkpoint for OOD eval, matching what's actually saved
    # (not just the in-memory end-of-training weights).
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"\nEvaluating best checkpoint (epoch={ckpt['epoch']}, "
          f"val_accuracy={ckpt['val_accuracy']:.2f}%, val_loss={ckpt['val_loss']:.4f}) on OOD:")
    ood_results = evaluate_ood(model, tokenizer, data_dir, study_key, args.batch_size, device)

    ood_path = results_dir / "ood_results.json"
    with open(ood_path, 'w') as f:
        json.dump(ood_results, f, indent=2)

    fingerprint = {k: v for k, v in ckpt.items() if k not in ('model_state_dict', 'optimizer_state_dict')}
    print(f"\nCheckpoint fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()

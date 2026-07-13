import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import time


def calculate_accuracy(model, dataloader, device, pad_idx=0):

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for enc_input, dec_input, dec_target in dataloader:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)
            
            # Get predictions
            output = model(enc_input, dec_input)  
            predictions = output.argmax(dim=-1)   
            
            # Check exact sequence match for each sample
            batch_size = predictions.size(0)
            for i in range(batch_size):
                target_seq = dec_target[i]
                pred_seq = predictions[i]
                
                # Find where padding starts
                mask = (target_seq != pad_idx)
                
                # Compare only non-padding tokens
                if torch.equal(pred_seq[mask], target_seq[mask]):
                    correct += 1
                total += 1
    
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate(model, dataloader, criterion, device, pad_idx=0):
    # Evaluate - return loss only
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for enc_input, dec_input, dec_target in dataloader:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)
            
            output = model(enc_input, dec_input)
            loss = criterion(output.view(-1, output.size(-1)), dec_target.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)  


def _make_scheduler_lambda(warmup_steps: int, total_steps: int, schedule: str, min_lr_ratio: float):
    # Linear warmup 0 -> 1 over warmup_steps optimizer steps, then either:
    #   "constant": hold at 1.0 (original no-decay behavior)
    #   "cosine": cosine-decay 1.0 -> min_lr_ratio over the remaining steps
    # Multiplies the optimizer's base lr, so this composes with whatever lr
    # was passed to the optimizer (the "target lr" is reached at step warmup_steps).
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and (step + 1) <= warmup_steps:
            return (step + 1) / warmup_steps
        if schedule != "cosine" or total_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor
    return lr_lambda


def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx=0, scheduler=None):
    # Train for one epoch - return loss only
    model.train()
    total_loss = 0.0

    for enc_input, dec_input, dec_target in dataloader:
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        dec_target = dec_target.to(device)

        optimizer.zero_grad()
        output = model(enc_input, dec_input)
        loss = criterion(output.view(-1, output.size(-1)), dec_target.view(-1))
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            # Stepped per-batch (not per-epoch): warmup is defined in
            # optimizer steps, not epochs.
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,  # callers override this; aligned to 100 to match real cap
    learning_rate: float = 0.001,
    device: str = "cpu",
    save_path: Optional[str] = None,
    pad_idx: int = 0,
    early_stopping_patience: int = 5,
    warmup_steps: int = 0,
    scheduler_type: str = "constant",
    min_lr: float = 1e-5,
    seed: Optional[int] = None,
    dataset_version: Optional[str] = None,
    generator_commit: Optional[str] = None,
) -> Dict[str, list]:
    """
    Checkpoint selection AND early stopping both now watch VAL_ACCURACY
    (this superseded an earlier decoupled design - val_loss vs val_acc
    tracking - now the same signal drives both: patience resets whenever
    val_accuracy improves, and that same improvement is what triggers
    saving the checkpoint). val_loss is still recorded in the returned
    history for inspection, just no longer used to gate anything.

    warmup_steps: linear LR warmup from 0 to `learning_rate` over this many
    optimizer steps (not epochs). Pass 0 to disable.
    scheduler_type: "constant" (hold at target lr after warmup - the
    original no-decay behavior) or "cosine" (cosine-decay from target lr
    down to `min_lr` over the remaining steps after warmup, computed as
    num_epochs * steps_per_epoch).
    """

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    min_lr_ratio = (min_lr / learning_rate) if learning_rate > 0 else 0.0
    scheduler = None
    if warmup_steps > 0 or scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_make_scheduler_lambda(warmup_steps, total_steps, scheduler_type, min_lr_ratio),
        )

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_val_accuracy = float('-inf')
    epochs_without_improvement = 0

    print("="*80)
    print("TRAINING MODEL")
    print("="*80)
    print(f"Model type: {type(model).__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate} (warmup_steps={warmup_steps}, scheduler={scheduler_type}, "
          f"min_lr={min_lr if scheduler_type == 'cosine' else 'n/a'}, total_steps={total_steps})")
    print(f"Epochs: {num_epochs}")
    print(f"Early stopping patience: {early_stopping_patience} (monitors val_accuracy)")
    print(f"Checkpoint selection: best val_accuracy")
    print("-"*80)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<12} {'Best':<8}")
    for epoch in range(num_epochs):
        start_time = time.time()

        # Train one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, pad_idx, scheduler=scheduler
        )
        current_lr = optimizer.param_groups[0]['lr']

        # Evaluate on validation set
        val_loss = evaluate(
            model, val_loader, criterion, device, pad_idx
        )

        # Calculate accuracies (expensive, so only once per epoch)
        train_acc = calculate_accuracy(model, train_loader, device, pad_idx)
        val_acc = calculate_accuracy(model, val_loader, device, pad_idx)

        # Record history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Both early-stopping patience and checkpoint save are keyed on
        # val_accuracy (see docstring)
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'seed': seed,
                    'dataset_version': dataset_version,
                    'generator_commit': generator_commit,
                }, save_path)
        else:
            epochs_without_improvement += 1

        elapsed_time = time.time() - start_time
        best_marker = "✓" if is_best else ""

        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.2f}% {val_loss:<12.4f} {val_acc:<12.2f}% {current_lr:<12.6f} {best_marker:<8}")

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print("="*80)
    print(f"TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy (checkpoint saved on this): {best_val_accuracy:.2f}%")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print("="*80)

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }


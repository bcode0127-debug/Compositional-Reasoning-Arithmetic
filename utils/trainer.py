import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import time


def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0
) -> float:

    # Get predicted tokens
    pred_tokens = predictions.argmax(dim=1)  # [batch*seq_len]
    
    # Create mask for non-padding tokens
    mask = targets != pad_idx
    
    # Calculate accuracy only on non-padding tokens
    correct = (pred_tokens == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item() * 100
    
    return accuracy


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
    pad_idx: int = 0
) -> Tuple[float, float]:
    
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (enc_input, dec_input, dec_target) in enumerate(dataloader):
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        dec_target = dec_target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(enc_input, dec_input)  # [batch, seq_len, vocab_size]
        
        # Reshape for loss calculation
        output_flat = output.view(-1, output.size(-1))  # [batch*seq_len, vocab_size]
        target_flat = dec_target.view(-1)               # [batch*seq_len]
        
        # Calculate loss
        loss = criterion(output_flat, target_flat)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(output_flat, target_flat, pad_idx)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_accuracy += accuracy
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
    pad_idx: int = 0
) -> Tuple[float, float]:
    
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for enc_input, dec_input, dec_target in dataloader:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)
            
            # Forward pass
            output = model(enc_input, dec_input)
            
            # Reshape for loss calculation
            output_flat = output.view(-1, output.size(-1))
            target_flat = dec_target.view(-1)
            
            # Calculate loss and accuracy
            loss = criterion(output_flat, target_flat)
            accuracy = calculate_accuracy(output_flat, target_flat, pad_idx)
            
            total_loss += loss.item()
            total_accuracy += accuracy
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    device: str = "cpu",
    save_path: Optional[str] = None,
    pad_idx: int = 0,
    early_stopping_patience: int = 5
) -> Dict[str, list]:
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print("="*80)
    print("TRAINING LSTM SEQ2SEQ MODEL")
    print("="*80)
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("-"*80)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Best':<8}")
    print("-"*80)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, pad_idx
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, pad_idx
        )
        
        # Record history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Check for improvement
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }, save_path)
        else:
            epochs_without_improvement += 1
        
        elapsed_time = time.time() - start_time
        best_marker = "✓" if is_best else ""
        
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.2f}% {val_loss:<12.4f} {val_acc:<12.2f}% {best_marker:<8}")
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("="*80)
    print(f"✅ TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print("="*80)
    
    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_loss": best_val_loss
    }


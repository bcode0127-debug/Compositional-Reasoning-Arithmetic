import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import time


def calculate_accuracy(model, dataloader, device, pad_idx=0):
    # Calculate sequence-level accuracy (exact match)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for enc_input, dec_input, dec_target in dataloader:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)
            
            # Get predictions
            output = model(enc_input, dec_input)  # [batch, seq_len, vocab]
            predictions = output.argmax(dim=-1)   # [batch, seq_len]
            
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


def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx=0):
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
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)  # Return loss only


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
    
    return total_loss / len(dataloader)  # Return loss only


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
        
        # Train one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, pad_idx
        )
        
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


import os
import time
import json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------
# Small helpers
# ---------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy from logits and integer class targets.
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


# ---------------------------
# One epoch: Train
# ---------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    Runs a single training epoch.
    Returns: {"loss": float, "acc": float}
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(outputs, targets) * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_acc = running_acc / max(1, total_samples)

    return {"loss": epoch_loss, "acc": epoch_acc}


# ---------------------------
# One epoch: Validate
# ---------------------------
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    Runs a single validation epoch (no grads).
    Returns: {"loss": float, "acc": float}
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(outputs, targets) * batch_size
            total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_acc = running_acc / max(1, total_samples)

    return {"loss": epoch_loss, "acc": epoch_acc}


# ---------------------------
# Full training loop
# ---------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 3,
    num_epochs: int = 20,
    device: Optional[torch.device] = None,

    # Defaults you can override in main_*.py
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,          # e.g., torch.optim.lr_scheduler.StepLR
    use_amp: bool = True,                      # mixed precision toggle
    early_stopping_patience: Optional[int] = 7,

    # bookkeeping
    fold_num: int = 0,
    model_name: str = "model",
    save_root: str = "outputs",
) -> Dict[str, Any]:
    """
    Trains a model with validation, saving the best checkpoint by val accuracy.
    Returns a history dict with per-epoch metrics and metadata.
    """
    # Setup
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Default criterion & optimizer if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Paths
    ckpt_dir = os.path.join(save_root, "checkpoints")
    metrics_dir = os.path.join(save_root, "metrics")
    logs_dir = os.path.join(save_root, "logs")
    for d in (ckpt_dir, metrics_dir, logs_dir):
        _ensure_dir(d)

    best_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_fold{fold_num}_best.pth")
    last_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_fold{fold_num}_last.pth")
    history_path = os.path.join(metrics_dir, f"{model_name}_fold{fold_num}_history.json")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": None,
        "best_val_acc": 0.0,
        "num_epochs": num_epochs,
        "fold": fold_num,
        "model_name": model_name,
        "num_classes": num_classes,
        "device": str(device),
    }

    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"\n=== Training {model_name} | Fold {fold_num} | Epochs: {num_epochs} | Device: {device} ===")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Train
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=use_amp
        )

        # Validate
        val_stats = validate_one_epoch(
            model, val_loader, criterion, device, use_amp=use_amp
        )

        # Scheduler step (if provided)
        if scheduler is not None:
            # Most schedulers step per-epoch; adjust if you use OneCycle, etc.
            scheduler.step()

        # Save history
        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])

        # Progress
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] "
            f"LR {lr:.2e} | "
            f"Train Loss {train_stats['loss']:.4f} Acc {train_stats['acc']*100:5.2f}% | "
            f"Val Loss {val_stats['loss']:.4f} Acc {val_stats['acc']*100:5.2f}% | "
            f"{elapsed:.1f}s"
        )

        # Checkpointing (best on val acc)
        current_val_acc = val_stats["acc"]
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "val_acc": current_val_acc,
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "train_loss": train_stats["loss"],
                "model_name": model_name,
                "fold": fold_num,
            },
            last_ckpt_path,
        )

        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            history["best_epoch"] = epoch
            history["best_val_acc"] = best_val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "val_acc": current_val_acc,
                    "train_acc": train_stats["acc"],
                    "val_loss": val_stats["loss"],
                    "train_loss": train_stats["loss"],
                    "model_name": model_name,
                    "fold": fold_num,
                },
                best_ckpt_path,
            )
            epochs_no_improve = 0
            improved = "✓"
        else:
            epochs_no_improve += 1
            improved = " "

        # Early stopping
        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            print(
                f"Early stopping triggered after {early_stopping_patience} epochs without improvement. "
                f"Best Val Acc: {best_val_acc*100:.2f}% at epoch {history['best_epoch']}."
            )
            break

        # Save history each epoch (safe in case of interruption)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Mark improvement in console line
        if improved.strip():
            print(f"  ↳ New best val acc: {best_val_acc*100:.2f}%  (ckpt saved: {os.path.basename(best_ckpt_path)})")

    # Final history save
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best Val Acc: {best_val_acc*100:.2f}% (epoch {history['best_epoch']}).")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"History saved to: {history_path}")

    return history

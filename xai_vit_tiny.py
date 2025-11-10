import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.dataset import get_dataloaders_for_fold
from models.vit_tiny import build_vit_tiny
from src.xai_utils import explain_indices_from_loader

# ================================
# CONFIG
# ================================
MODEL_NAME = "vit_tiny"
SAVE_ROOT = "outputs"
NUM_CLASSES = 3
NUM_FOLDS = 5
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 224  # must match dataset transforms

# !!! IMPORTANT !!!
# Use the SAME dropout you used during training (main_vit_tiny.py)
DROPOUT_USED_DURING_TRAINING = 0.1

# XAI settings
CLASS_NAMES = ["covid", "normal", "pneumonia"]
CAM_METHODS = ["gradcam"]  # add "gradcam++", "eigencam" if desired
SAMPLES_PER_CLASS_CORRECT = 5
SAMPLES_PER_CLASS_INCORRECT = 5
USE_AMP_PRED = True  # AMP for fast predictions; CAM runs with grads

# ================================
# HELPERS
# ================================
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def collect_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[List[int], List[int]]:
    """Fast prediction pass to get y_true / y_pred (NO gradient)."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
    return y_true, y_pred

def pick_indices_per_class(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
    k_correct: int,
    k_incorrect: int
) -> Dict[str, Dict[int, List[int]]]:
    """Select up to K correct and K incorrect indices per TRUE class."""
    correct_by_class = {c: [] for c in range(num_classes)}
    incorrect_by_class = {c: [] for c in range(num_classes)}
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if p == t:
            correct_by_class[t].append(idx)
        else:
            incorrect_by_class[t].append(idx)
    selected = {"correct": {}, "incorrect": {}}
    for c in range(num_classes):
        selected["correct"][c]   = correct_by_class[c][:k_correct]
        selected["incorrect"][c] = incorrect_by_class[c][:k_incorrect]
    return selected

def visualize_groups_for_fold(
    cam_model: nn.Module,
    val_loader,
    fold: int,
    class_names: List[str],
    selection: Dict[str, Dict[int, List[int]]],
    cam_methods: List[str],
    base_out_dir: str,
):
    """Create folders per group/class and run Grad-CAM via xai_utils."""
    for group_key in ["correct", "incorrect"]:
        for c, idx_list in selection[group_key].items():
            if not idx_list:
                continue
            cls_name = class_names[c] if 0 <= c < len(class_names) else f"class{c}"
            out_dir = os.path.join(base_out_dir, f"fold{fold}", group_key, cls_name)
            _ensure_dir(out_dir)
            for method in cam_methods:
                explain_indices_from_loader(
                    model=cam_model,
                    dataloader=val_loader,
                    indices=idx_list,
                    model_name=MODEL_NAME,      # ensures ViT target layer + reshape_transform
                    method=method,
                    out_dir=out_dir,
                    class_names=class_names,
                    pick_pred_for_target=True   # heatmap for predicted class
                )

# ================================
# MAIN SCRIPT
# ================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[XAI] Using device: {device}")

    for fold in range(NUM_FOLDS):
        print(f"\n===== XAI for {MODEL_NAME} | Fold {fold+1}/{NUM_FOLDS} =====")

        # 1) Val loader for this fold
        _, val_loader = get_dataloaders_for_fold(
            fold_index=fold,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )

        # 2) Prediction model (A) for y_true/y_pred (no grads)
        ckpt_path = os.path.join(SAVE_ROOT, "checkpoints", f"{MODEL_NAME}_fold{fold}_best.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        pred_model = build_vit_tiny(
            num_classes=NUM_CLASSES,
            pretrained=False,
            image_size=IMAGE_SIZE,
            dropout=DROPOUT_USED_DURING_TRAINING
        ).to(device)
        pred_model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        pred_model.eval()
        print(f"[XAI] Loaded checkpoint for predictions: {ckpt_path}")

        # 3) Collect predictions (no gradients)
        y_true, y_pred = collect_predictions(pred_model, val_loader, device, use_amp=USE_AMP_PRED)

        # 4) Fresh CAM model (B) with grads enabled
        cam_model = build_vit_tiny(
            num_classes=NUM_CLASSES,
            pretrained=False,
            image_size=IMAGE_SIZE,
            dropout=DROPOUT_USED_DURING_TRAINING
        ).to(device)
        cam_model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        for p in cam_model.parameters():
            p.requires_grad = True
        cam_model.eval()  # eval mode ok; grads still enabled

        # 5) Pick indices per class (correct/incorrect)
        selection = pick_indices_per_class(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=NUM_CLASSES,
            k_correct=SAMPLES_PER_CLASS_CORRECT,
            k_incorrect=SAMPLES_PER_CLASS_INCORRECT
        )

        # 6) Generate Grad-CAM overlays
        base_out_dir = os.path.join(SAVE_ROOT, "xai_visualizations", MODEL_NAME)
        visualize_groups_for_fold(
            cam_model=cam_model,
            val_loader=val_loader,
            fold=fold,
            class_names=CLASS_NAMES,
            selection=selection,
            cam_methods=CAM_METHODS,
            base_out_dir=base_out_dir,
        )

        print(f"[XAI] Saved CAM overlays for Fold {fold} under:")
        print(f"      {os.path.join(base_out_dir, f'fold{fold}')}")
    print("\n[XAI] Done generating Grad-CAM overlays for all folds.")

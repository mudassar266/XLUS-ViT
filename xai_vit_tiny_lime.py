import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import cv2

from lime import lime_image
from skimage.segmentation import mark_boundaries

from src.dataset import get_dataloaders_for_fold
from models.vit_tiny import build_vit_tiny

# ================================
# CONFIG
# ================================
MODEL_NAME = "vit_tiny"
SAVE_ROOT = "outputs"
NUM_CLASSES = 3
NUM_FOLDS = 5
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 224

CLASS_NAMES = ["covid", "normal", "pneumonia"]

# !!! IMPORTANT !!!
# Use SAME dropout as you used during training in main_vit_tiny.py
DROPOUT_USED_DURING_TRAINING = 0.1

# LIME settings
SAMPLES_PER_CLASS_CORRECT = 3
SAMPLES_PER_CLASS_INCORRECT = 3
LIME_NUM_SAMPLES = 1000      # number of perturbations
LIME_NUM_FEATURES = 10       # number of superpixels to highlight
USE_AMP_PRED = True          # AMP during predictions (not during LIME callback)


# ================================
# HELPERS
# ================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

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

def tensor_to_denorm_rgb(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: (C,H,W) tensor normalized with ImageNet stats.
    Return: (H,W,3) float in [0,1], RGB.
    """
    img = img_t.detach().cpu().float().clone()
    for c in range(3):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()  # HWC

def make_lime_predict_fn(model: nn.Module, device: torch.device):
    """
    Returns a function f(images: np.ndarray) -> probs (N, num_classes)
    images: (N, H, W, 3), dtype uint8 or float, values in [0,255] or [0,1].
    """
    model.eval()

    def predict(images: np.ndarray) -> np.ndarray:
        # images: (N, H, W, 3)
        imgs = images.astype(np.float32)
        if imgs.max() > 1.5:  # assume [0,255]
            imgs = imgs / 255.0

        # Normalize using ImageNet mean/std
        imgs_norm = (imgs - IMAGENET_MEAN) / IMAGENET_STD  # broadcasts over channels
        # to NCHW
        imgs_norm = torch.from_numpy(imgs_norm).permute(0, 3, 1, 2).float().to(device)

        with torch.no_grad():
            logits = model(imgs_norm)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    return predict

def explain_with_lime_for_index(
    model: nn.Module,
    dataloader,
    dataset_idx: int,
    device: torch.device,
    class_names: List[str],
    out_path: str,
    num_samples: int = 1000,
    num_features: int = 10,
):
    """
    Runs LIME for a single dataset index and saves the explanation image.
    """
    dataset = dataloader.dataset
    img_t, label = dataset[dataset_idx]  # (C,H,W), int
    if isinstance(img_t, np.ndarray):
        # If dataset ever returns numpy, convert to tensor
        img_t = torch.from_numpy(img_t).permute(2, 0, 1).float()

    # Denormalize to get RGB in [0,1]
    rgb_img = tensor_to_denorm_rgb(img_t)
    # Convert to uint8 for LIME
    lime_input = (rgb_img * 255).astype(np.uint8)

    # Build predict function for LIME
    predict_fn = make_lime_predict_fn(model, device)

    # Run one quick prediction to know predicted class
    with torch.no_grad():
        probs = predict_fn(lime_input[None, ...])  # (1, C)
    pred_class = int(np.argmax(probs, axis=1)[0])

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image=lime_input,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    # Get superpixel mask for the predicted class
    temp, mask = explanation.get_image_and_mask(
        label=pred_class,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )

    # temp is float [0,1], mask is 0/1
    # Overlay segmentation boundaries for nicer visualization
    lime_vis = mark_boundaries(temp, mask)

    # Build filename
    true_name = class_names[label] if 0 <= label < len(class_names) else f"true{label}"
    pred_name = class_names[pred_class] if 0 <= pred_class < len(class_names) else f"pred{pred_class}"
    fname = f"idx{dataset_idx:05d}_true-{true_name}_pred-{pred_name}_lime.png"
    save_path = os.path.join(out_path, fname)
    _ensure_dir(out_path)

    # Save as BGR PNG (OpenCV)
    cv2.imwrite(
        save_path,
        cv2.cvtColor((lime_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )
    print(f"[LIME] Saved: {save_path}")

# ================================
# MAIN SCRIPT
# ================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LIME-XAI] Using device: {device}")

    for fold in range(NUM_FOLDS):
        print(f"\n===== LIME XAI for {MODEL_NAME} | Fold {fold+1}/{NUM_FOLDS} =====")

        # 1) Val loader
        _, val_loader = get_dataloaders_for_fold(
            fold_index=fold,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )

        # 2) Build prediction model (for selection and LIME)
        ckpt_path = os.path.join(SAVE_ROOT, "checkpoints", f"{MODEL_NAME}_fold{fold}_best.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = build_vit_tiny(
            num_classes=NUM_CLASSES,
            pretrained=False,
            image_size=IMAGE_SIZE,
            dropout=DROPOUT_USED_DURING_TRAINING
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        model.eval()
        print(f"[LIME-XAI] Loaded checkpoint: {ckpt_path}")

        # 3) Collect predictions (to choose interesting indices)
        y_true, y_pred = collect_predictions(model, val_loader, device, use_amp=USE_AMP_PRED)

        # 4) Choose some correct & incorrect samples per class
        selection = pick_indices_per_class(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=NUM_CLASSES,
            k_correct=SAMPLES_PER_CLASS_CORRECT,
            k_incorrect=SAMPLES_PER_CLASS_INCORRECT
        )

        base_out_dir = os.path.join(SAVE_ROOT, "xai_lime", MODEL_NAME)

        # 5) Generate LIME explanations
        for group_key in ["correct", "incorrect"]:
            for c, idx_list in selection[group_key].items():
                if not idx_list:
                    continue
                cls_name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else f"class{c}"
                out_dir = os.path.join(base_out_dir, f"fold{fold}", group_key, cls_name)
                for dataset_idx in idx_list:
                    explain_with_lime_for_index(
                        model=model,
                        dataloader=val_loader,
                        dataset_idx=dataset_idx,
                        device=device,
                        class_names=CLASS_NAMES,
                        out_path=out_dir,
                        num_samples=LIME_NUM_SAMPLES,
                        num_features=LIME_NUM_FEATURES,
                    )

        print(f"[LIME-XAI] Finished Fold {fold}. Results under: {os.path.join(base_out_dir, f'fold{fold}')}")
    print("\n[LIME-XAI] Done generating LIME explanations for all folds.")

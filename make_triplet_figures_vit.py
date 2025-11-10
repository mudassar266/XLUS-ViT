import os
from typing import List, Dict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries

from src.dataset import get_dataloaders_for_fold
from src.xai_utils import explain_with_cam
from models.vit_tiny import build_vit_tiny


# ================================
# CONFIG
# ================================
MODEL_NAME = "vit_tiny"
SAVE_ROOT = "outputs"
NUM_CLASSES = 3
IMAGE_SIZE = 224
NUM_FOLDS = 5

# Choose which fold to visualize
FOLD_TO_VISUALIZE = 4

# Must match training
DROPOUT_USED_DURING_TRAINING = 0.1
CLASS_NAMES = ["covid", "normal", "pneumonia"]

# LIME settings
LIME_NUM_SAMPLES   = 1000
LIME_NUM_FEATURES  = 10

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ================================
# HELPERS
# ================================
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_to_denorm_rgb(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().float().clone()
    for c in range(3):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()  # HWC


def collect_predictions(model: nn.Module, dataset, device: torch.device):
    """Run model on entire val dataset (index by index) to get y_true, y_pred."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for idx in range(len(dataset)):
            img_t, label = dataset[idx]
            if isinstance(img_t, np.ndarray):
                img_t = torch.from_numpy(img_t).permute(2, 0, 1).float()
            inp = img_t.unsqueeze(0).to(device)
            logits = model(inp)
            pred = torch.argmax(logits, dim=1).item()
            y_true.append(int(label))
            y_pred.append(int(pred))
    return y_true, y_pred


def pick_one_correct_per_class(
    y_true: List[int], y_pred: List[int], num_classes: int
) -> Dict[int, int]:
    """
    Returns a dict: class_id -> dataset_index
    One correctly predicted example per class, if available.
    """
    result = {}
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p and t not in result:
            result[t] = idx
        if len(result) == num_classes:
            break
    return result


def make_lime_predict_fn(model: nn.Module, device: torch.device):
    model.eval()

    def predict(images: np.ndarray) -> np.ndarray:
        imgs = images.astype(np.float32)
        if imgs.max() > 1.5:  # assume [0,255]
            imgs = imgs / 255.0
        imgs_norm = (imgs - IMAGENET_MEAN) / IMAGENET_STD
        imgs_norm = torch.from_numpy(imgs_norm).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            logits = model(imgs_norm)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    return predict


def get_lime_explanation_image(
    model: nn.Module,
    device: torch.device,
    img_t: torch.Tensor,
    num_samples: int = 1000,
    num_features: int = 10,
) -> (np.ndarray, int):
    """
    Run LIME on a single tensor image and return:
      - explanation image (H,W,3) float in [0,1]
      - predicted class index
    """
    rgb_img = tensor_to_denorm_rgb(img_t)
    lime_input = (rgb_img * 255).astype(np.uint8)

    predict_fn = make_lime_predict_fn(model, device)

    # Predicted class
    with torch.no_grad():
        probs = predict_fn(lime_input[None, ...])
    pred_class = int(np.argmax(probs, axis=1)[0])

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=lime_input,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    temp, mask = explanation.get_image_and_mask(
        label=pred_class,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )

    lime_vis = mark_boundaries(temp, mask)
    return lime_vis, pred_class


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Triplets] Using device: {device}")

    # 1) Get val dataset for selected fold
    _, val_loader = get_dataloaders_for_fold(
        fold_index=FOLD_TO_VISUALIZE,
        batch_size=1,
        num_workers=4
    )
    dataset = val_loader.dataset

    # 2) Build ViT model and load checkpoint
    ckpt_path = os.path.join(
        SAVE_ROOT, "checkpoints",
        f"{MODEL_NAME}_fold{FOLD_TO_VISUALIZE}_best.pth"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_vit_tiny(
        num_classes=NUM_CLASSES,
        pretrained=False,
        image_size=IMAGE_SIZE,
        dropout=DROPOUT_USED_DURING_TRAINING
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Triplets] Loaded checkpoint: {ckpt_path}")

    # 3) Collect predictions on entire val set
    y_true, y_pred = collect_predictions(model, dataset, device)
    print("[Triplets] Class-wise correct counts:")
    for c in range(NUM_CLASSES):
        n_correct = sum((t == c and p == c) for t, p in zip(y_true, y_pred))
        print(f"  {CLASS_NAMES[c]}: {n_correct} correct")

    # 4) Pick one correctly predicted sample per class
    class_to_idx = pick_one_correct_per_class(y_true, y_pred, NUM_CLASSES)
    print(f"[Triplets] Selected indices per class: {class_to_idx}")

    out_root = os.path.join(SAVE_ROOT, "xai_triplets", MODEL_NAME, f"fold{FOLD_TO_VISUALIZE}")
    _ensure_dir(out_root)

    # 5) For each class that has at least one correct sample, build triplet
    for c, idx in class_to_idx.items():
        img_t, label = dataset[idx]
        if isinstance(img_t, np.ndarray):
            img_t = torch.from_numpy(img_t).permute(2, 0, 1).float()

        # Original
        orig_rgb = tensor_to_denorm_rgb(img_t)

        # Grad-CAM (using your xai_utils)
        _, cam_overlay = explain_with_cam(
            model=model,
            image_tensor=img_t,
            model_name=MODEL_NAME,
            method="gradcam",
            target_category=None,
            save_path=None
        )
        cam_rgb = cam_overlay.astype(np.uint8) / 255.0

        # LIME
        lime_vis, pred_class = get_lime_explanation_image(
            model=model,
            device=device,
            img_t=img_t,
            num_samples=LIME_NUM_SAMPLES,
            num_features=LIME_NUM_FEATURES,
        )

        true_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else f"true{label}"
        pred_name = CLASS_NAMES[pred_class] if 0 <= pred_class < len(CLASS_NAMES) else f"pred{pred_class}"

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=200)

        axes[0].imshow(orig_rgb)
        axes[0].set_title(f"Original\nTrue: {true_name}", fontsize=8)
        axes[0].axis("off")

        axes[1].imshow(cam_rgb)
        axes[1].set_title("Grad-CAM", fontsize=8)
        axes[1].axis("off")

        axes[2].imshow(lime_vis)
        axes[2].set_title("LIME", fontsize=8)
        axes[2].axis("off")

        plt.tight_layout()

        fname = f"triplet_class-{CLASS_NAMES[c]}_idx{idx:05d}_true-{true_name}_pred-{pred_name}.png"
        save_path = os.path.join(out_root, fname)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        print(f"[Triplets] Saved: {save_path}")

    print("\n[Triplets] Done generating side-by-side figures.")

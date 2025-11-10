"""
src/xai_utils.py

Grad-CAM utilities for CNNs (e.g., ResNet50) and ViT (timm).
Supports GradCAM, GradCAM++ and EigenCAM, with automatic target-layer selection.

Usage examples
--------------
Single image tensor (C,H,W), normalized with ImageNet stats:
    from src.xai_utils import explain_with_cam
    heatmap, overlay = explain_with_cam(
        model, image_tensor, model_name="resnet50",
        method="gradcam", target_category=None,
        save_path="outputs/xai_visualizations/resnet50_example.png"
    )

Batch/DataLoader (pick some indices to visualize):
    from src.xai_utils import explain_indices_from_loader
    explain_indices_from_loader(
        model, val_loader, indices=[0, 7, 19],
        model_name="vit_tiny", method="gradcam",
        out_dir="outputs/xai_visualizations/vit_tiny_fold0/"
    )
"""

import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ---------------------------
# Constants (ImageNet stats)
# ---------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------
# Small helpers
# ---------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_numpy_image_from_tensor(img_t: torch.Tensor,
                                mean=IMAGENET_MEAN,
                                std=IMAGENET_STD) -> np.ndarray:
    """
    Convert a normalized (C,H,W) tensor in [0,1] (after ToTensor) & Normalize(ImageNet)
    back to a float RGB numpy image in [0,1] for overlay.
    """
    assert img_t.ndim == 3, "Expected CHW tensor"
    img = img_t.detach().cpu().float().clone()
    # denormalize
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()  # HWC
    return img


def _enable_grads(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


# ---------------------------
# ViT reshape transform (timm)
# ---------------------------
def _vit_reshape_transform(tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    For timm ViT, convert token sequence (B, N, C) to spatial map (B, C, H, W)
    by removing class token and reshaping using model.patch_embed.grid_size.
    """
    # tensor: (B, N, C) from transformer blocks
    t = tensor[:, 1:, :]  # drop CLS token
    # grid size (H, W)
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        gh, gw = model.patch_embed.grid_size
    else:
        num_patches = t.shape[1]
        gh = gw = int(num_patches ** 0.5)

    B, N, C = t.shape
    assert N == gh * gw, f"Token count {N} != grid HxW {gh}x{gw}"
    t = t.reshape(B, gh, gw, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
    return t


# ---------------------------
# Pick target layer automatically
# ---------------------------
def _pick_target_layers(model: nn.Module, model_name: str):
    """
    Returns (target_layers, reshape_transform or None).
    - For ResNet50: last bottleneck block (layer4[-1]) works well.
    - For timm ViT-tiny: use the last block's norm with reshape_transform.
    """
    name = (model_name or "").lower()

    # torchvision resnet50
    if "resnet" in name:
        try:
            return [model.layer4[-1]], None
        except Exception:
            return [model.layer4], None

    # timm vit tiny
    if "vit" in name:
        try:
            target_layers = [model.blocks[-1].norm1]
        except Exception:
            target_layers = [model.blocks[-1]]
        reshape_transform = lambda x: _vit_reshape_transform(x, model)
        return target_layers, reshape_transform

    # Fallback: try common feature containers
    for attr in ["layer4", "features"]:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, (nn.Sequential, nn.ModuleList)) and len(layer) > 0:
                return [layer[-1]], None
            return [layer], None

    return [model], None  # last resort


# ---------------------------
# Select CAM method
# ---------------------------
def _make_cam(method: str, model: nn.Module, target_layers, reshape_transform=None):
    method = (method or "gradcam").lower()
    cam_class = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigencam": EigenCAM,
    }.get(method, GradCAM)

    cam = cam_class(model=model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform)
    return cam


# ---------------------------
# Core: single-image Grad-CAM
# ---------------------------
def explain_with_cam(
    model: nn.Module,
    image_tensor: torch.Tensor,                 # (C,H,W), normalized
    model_name: str = "resnet50",
    method: str = "gradcam",                    # "gradcam", "gradcam++", "eigencam"
    target_category: Optional[int] = None,      # class index; None -> model's top-1
    use_cuda: Optional[bool] = None,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      heatmap: 2D numpy array in [0,1]
      overlay: RGB uint8 image (H,W,3) heatmap over original image
    """
    model.eval()
    _enable_grads(model)  # ensure gradients are enabled

    device = next(model.parameters()).device
    if use_cuda is None:
        use_cuda = device.type == "cuda"

    # Ensure 4D (B,C,H,W)
    if image_tensor.ndim == 3:
        input_tensor = image_tensor.unsqueeze(0).to(device)
    else:
        input_tensor = image_tensor.to(device)

    # Prepare CAM
    target_layers, reshape_transform = _pick_target_layers(model, model_name)
    cam = _make_cam(method, model, target_layers, reshape_transform)

    # Targets for classifier
    targets = None
    if target_category is not None:
        targets = [ClassifierOutputTarget(int(target_category))]
    # If None, pytorch-grad-cam will default to the top-1 category internally.

    # Compute CAM (grayscale heatmap in [0,1]) — NO no_grad here
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # (H, W)

    # Build RGB image for overlay (denormalize)
    rgb_img = _to_numpy_image_from_tensor(image_tensor)
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save if requested
    if save_path is not None:
        _ensure_dir(os.path.dirname(save_path))
        import cv2
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return grayscale_cam, overlay


# ---------------------------
# Convenience: from DataLoader by indices
# ---------------------------
def explain_indices_from_loader(
    model: nn.Module,
    dataloader,
    indices: List[int],
    model_name: str = "resnet50",
    method: str = "gradcam",
    out_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    use_cuda: Optional[bool] = None,
    pick_pred_for_target: bool = True,
) -> None:
    """
    Pull chosen indices from a DataLoader and save CAM overlays.

    Args:
        indices: list of absolute indices in the dataset (not batch indices).
        pick_pred_for_target: if True, target_category = model's predicted class.

    Saves: out_dir/{idx}_true-<cls>_pred-<cls>_<method>.png
    """
    model.eval()
    _enable_grads(model)  # critical: allow gradients for CAM

    device = next(model.parameters()).device
    if use_cuda is None:
        use_cuda = device.type == "cuda"

    dataset = dataloader.dataset
    if class_names is None:
        class_names = ["covid", "normal", "pneumonia"]

    if out_dir is None:
        out_dir = "outputs/xai_visualizations"
    _ensure_dir(out_dir)

    target_layers, reshape_transform = _pick_target_layers(model, model_name)
    cam = _make_cam(method, model, target_layers, reshape_transform)

    import cv2

    for idx in indices:
        # 1) Fetch sample
        img, label = dataset[idx]  # (C,H,W), int
        if isinstance(img, np.ndarray):
            img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            img_t = img
        input_tensor = img_t.unsqueeze(0).to(device)

        # 2) Choose target category (optional quick prediction with no_grad)
        pred_idx = None
        if pick_pred_for_target:
            with torch.no_grad():
                logits = model(input_tensor)
                pred_idx = int(torch.argmax(logits, dim=1).item())

        targets = None
        if pick_pred_for_target and pred_idx is not None:
            targets = [ClassifierOutputTarget(pred_idx)]
        # else: None -> CAM will default to top-1

        # 3) Compute CAM (NO no_grad here)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # 4) Overlay and save
        rgb_img = _to_numpy_image_from_tensor(img_t)
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        pred_name = class_names[pred_idx] if (pred_idx is not None and 0 <= pred_idx < len(class_names)) else "auto"
        true_name = class_names[label] if 0 <= label < len(class_names) else f"true{label}"
        fname = f"idx{idx:05d}_true-{true_name}_pred-{pred_name}_{method}.png"
        save_path = os.path.join(out_dir, fname)

        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

"""
src/eval_utils.py

Evaluation utilities for 3-class image classifiers (covid, normal, pneumonia)
with 5-fold CV. Computes per-fold metrics, plots confusion matrices, and
aggregates results across folds.

Features:
- get_predictions(): y_true, y_pred, y_prob
- compute_classification_metrics(): Acc, per-class P/R/F1, macro/weighted P/R/F1,
  Cohen's Kappa, MCC, ROC-AUC (OvR: per-class + macro)
- plot_confusion_matrix(): save PNG (normalized or raw)
- evaluate_fold(): one-call per-fold evaluation + saving JSON/PNG
- aggregate_kfold_results(): mean ± std across folds
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    # x: (N, C)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------
# Predictions
# ---------------------------
@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        y_true: (N,)
        y_pred: (N,)
        y_prob: (N, C)  - softmax probabilities
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # NEW API: torch.amp.autocast
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            logits = model(images)  # (B, C)

        probs = torch.softmax(logits, dim=1)  # (B, C)
        preds = probs.argmax(dim=1)

        y_true.append(targets.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())
        y_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)

    return y_true, y_pred, y_prob


# ---------------------------
# Metrics
# ---------------------------
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes per-class & aggregate metrics. If y_prob provided, adds ROC-AUC (OvR).

    Returns a nested dict with:
      - accuracy
      - per_class: dict[class_name or index] -> precision/recall/f1/support
      - macro_avg: precision/recall/f1
      - weighted_avg: precision/recall/f1
      - kappa
      - mcc
      - auc: { per_class, macro_ovr }  (if y_prob is not None)
    """
    if class_names is None:
        class_names = ["covid", "normal", "pneumonia"]

    labels = list(range(len(class_names)))

    acc = float(accuracy_score(y_true, y_pred))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Macro/weighted
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # Agreement metrics
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Structure per-class dictionary
    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "per_class": per_class,
        "macro_avg": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "weighted_avg": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1),
        },
        "kappa": kappa,
        "mcc": mcc,
    }

    # ROC-AUC (OvR)
    if y_prob is not None:
        # Binarize ground truth for OvR AUC
        y_true_bin = label_binarize(y_true, classes=labels)  # (N, C)
        per_class_auc = {}
        auc_vals = []
        for c, cname in enumerate(class_names):
            try:
                auc_c = float(roc_auc_score(y_true_bin[:, c], y_prob[:, c]))
            except ValueError:
                # If class not present in GT, AUC undefined; set to nan
                auc_c = float("nan")
            per_class_auc[cname] = auc_c
            if not np.isnan(auc_c):
                auc_vals.append(auc_c)

        # Macro OvR: mean of per-class (excluding NaN)
        macro_auc = float(np.mean(auc_vals)) if len(auc_vals) > 0 else float("nan")

        metrics["auc"] = {
            "per_class": per_class_auc,
            "macro_ovr": macro_auc,
        }

    return metrics


# ---------------------------
# Confusion Matrix Plot
# ---------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    cmap: str = "Blues",
):
    if class_names is None:
        class_names = ["covid", "normal", "pneumonia"]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_display = cm.astype(np.float32)

    if normalize:
        with np.errstate(all="ignore"):
            row_sums = cm_display.sum(axis=1, keepdims=True)
            cm_display = np.divide(cm_display, row_sums, out=np.zeros_like(cm_display), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations (robust: floats for both raw & normalized)
    fmt = ".2f" if normalize else ".0f"
    thresh = (cm_display.max() + cm_display.min()) / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            ax.text(
                j, i, format(val, fmt),
                ha="center", va="center",
                color="white" if val > thresh else "black"
            )

    if title:
        ax.set_title(title)

    fig.tight_layout()

    if save_path is not None:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig  # In case you want to show in a notebook


# ---------------------------
# Per-fold evaluation wrapper
# ---------------------------
def evaluate_fold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str,
    fold_num: int,
    save_root: str = "outputs",
    class_names: Optional[List[str]] = None,
    normalize_cm: bool = True,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """
    One-call per-fold evaluation:
      - runs predictions
      - computes metrics (+AUC if probs available)
      - saves JSON and confusion matrix PNG
      - returns metrics dict
    """
    if class_names is None:
        class_names = ["covid", "normal", "pneumonia"]

    # 1) Predictions
    y_true, y_pred, y_prob = get_predictions(
        model=model, dataloader=dataloader, device=device, use_amp=use_amp
    )

    # 2) Metrics
    metrics = compute_classification_metrics(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob, class_names=class_names
    )

    # 3) Save JSON
    metrics_dir = os.path.join(save_root, "metrics")
    _ensure_dir(metrics_dir)
    metrics_path = os.path.join(metrics_dir, f"{model_name}_fold{fold_num}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Saved fold metrics to: {metrics_path}")

    # 4) Confusion matrix PNG (both raw and normalized for completeness)
    cm_dir = os.path.join(save_root, "confusion_matrices")
    _ensure_dir(cm_dir)

    cm_title = f"{model_name} | Fold {fold_num} Confusion Matrix"
    cm_path = os.path.join(cm_dir, f"{model_name}_fold{fold_num}_cm.png")
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        normalize=False,
        title=cm_title + " (raw)",
        save_path=cm_path,
    )
    print(f"[eval] Saved confusion matrix (raw) to: {cm_path}")

    if normalize_cm:
        cmn_path = os.path.join(cm_dir, f"{model_name}_fold{fold_num}_cm_normalized.png")
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            normalize=True,
            title=cm_title + " (normalized)",
            save_path=cmn_path,
        )
        print(f"[eval] Saved confusion matrix (normalized) to: {cmn_path}")

    # 5) Also return y_true, y_pred, y_prob if you want to do more downstream
    metrics["y_true"] = y_true.tolist()
    metrics["y_pred"] = y_pred.tolist()
    metrics["y_prob"] = y_prob.tolist()

    return metrics


# ---------------------------
# Aggregate across folds
# ---------------------------
def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }


def aggregate_kfold_results(
    fold_metrics_list: List[Dict[str, Any]],
    model_name: str,
    save_root: str = "outputs",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregates metrics across folds and saves a JSON summary with mean±std.
    Expects fold_metrics_list to be a list of dicts returned by evaluate_fold().
    """
    if class_names is None:
        class_names = ["covid", "normal", "pneumonia"]

    # Collect lists
    acc_list, kappa_list, mcc_list = [], [], []
    macro_p_list, macro_r_list, macro_f1_list = [], [], []
    weighted_p_list, weighted_r_list, weighted_f1_list = [], [], []

    # Per-class dict of lists
    per_class_prec: Dict[str, List[float]] = {c: [] for c in class_names}
    per_class_rec: Dict[str, List[float]] = {c: [] for c in class_names}
    per_class_f1:  Dict[str, List[float]] = {c: [] for c in class_names}
    # AUC
    auc_macro_list = []
    per_class_auc: Dict[str, List[float]] = {c: [] for c in class_names}

    for m in fold_metrics_list:
        acc_list.append(m["accuracy"])
        kappa_list.append(m["kappa"])
        mcc_list.append(m["mcc"])

        macro_p_list.append(m["macro_avg"]["precision"])
        macro_r_list.append(m["macro_avg"]["recall"])
        macro_f1_list.append(m["macro_avg"]["f1"])

        weighted_p_list.append(m["weighted_avg"]["precision"])
        weighted_r_list.append(m["weighted_avg"]["recall"])
        weighted_f1_list.append(m["weighted_avg"]["f1"])

        # Per-class
        for cname in class_names:
            per_class_prec[cname].append(m["per_class"][cname]["precision"])
            per_class_rec[cname].append(m["per_class"][cname]["recall"])
            per_class_f1[cname].append(m["per_class"][cname]["f1"])

        # AUC (if present)
        if "auc" in m and isinstance(m["auc"], dict):
            if "macro_ovr" in m["auc"]:
                auc_macro_list.append(m["auc"]["macro_ovr"])
            if "per_class" in m["auc"]:
                for cname in class_names:
                    val = m["auc"]["per_class"].get(cname, float("nan"))
                    per_class_auc[cname].append(val)

    summary: Dict[str, Any] = {
        "accuracy": _mean_std(acc_list),
        "kappa": _mean_std(kappa_list),
        "mcc": _mean_std(mcc_list),
        "macro_avg": {
            "precision": _mean_std(macro_p_list),
            "recall": _mean_std(macro_r_list),
            "f1": _mean_std(macro_f1_list),
        },
        "weighted_avg": {
            "precision": _mean_std(weighted_p_list),
            "recall": _mean_std(weighted_r_list),
            "f1": _mean_std(weighted_f1_list),
        },
        "per_class": {
            cname: {
                "precision": _mean_std(per_class_prec[cname]),
                "recall": _mean_std(per_class_rec[cname]),
                "f1": _mean_std(per_class_f1[cname]),
            }
            for cname in class_names
        },
    }

    # AUC summary (if collected)
    if len(auc_macro_list) > 0:
        summary["auc"] = {
            "macro_ovr": _mean_std(auc_macro_list),
            "per_class": {
                cname: _mean_std(per_class_auc[cname]) for cname in class_names
            },
        }

    # Save
    metrics_dir = os.path.join(save_root, "metrics")
    _ensure_dir(metrics_dir)
    summary_path = os.path.join(metrics_dir, f"{model_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Saved k-fold summary to: {summary_path}")

    return summary

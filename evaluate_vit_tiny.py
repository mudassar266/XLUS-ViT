import os
import torch
import torch.nn as nn

from src.dataset import get_dataloaders_for_fold
from src.eval_utils import evaluate_fold, aggregate_kfold_results
from models.vit_tiny import build_vit_tiny

if __name__ == "__main__":
    # ================================
    # CONFIGURATION
    # ================================
    MODEL_NAME = "vit_tiny"
    SAVE_ROOT = "outputs"
    NUM_CLASSES = 3
    NUM_FOLDS = 5
    BATCH_SIZE = 16
    IMAGE_SIZE = 224

    CLASS_NAMES = ["covid", "normal", "pneumonia"]

    # !!! IMPORTANT !!!
    # Set DROPOUT to the SAME value used during training in main_vit_tiny.py
    DROPOUT_USED_DURING_TRAINING = 0.1   # ⬅ Change if different

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    all_fold_metrics = []

    for fold in range(NUM_FOLDS):
        print(f"\n===== Evaluating Fold {fold+1}/{NUM_FOLDS} =====")

        # 1) Build model with SAME head structure as training
        model = build_vit_tiny(
            num_classes=NUM_CLASSES,
            pretrained=False,   # we will load checkpoint weights
            image_size=IMAGE_SIZE,
            dropout=DROPOUT_USED_DURING_TRAINING
        ).to(device)

        # 2) Load best checkpoint
        ckpt_path = os.path.join(SAVE_ROOT, "checkpoints", f"{MODEL_NAME}_fold{fold}_best.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded best checkpoint: {ckpt_path}")

        # 3) Get val loader (Option A: val = test)
        _, val_loader = get_dataloaders_for_fold(
            fold_index=fold,
            batch_size=BATCH_SIZE,
            num_workers=4
        )

        # 4) Evaluate this fold
        fold_metrics = evaluate_fold(
            model=model,
            dataloader=val_loader,
            device=device,
            model_name=MODEL_NAME,
            fold_num=fold,
            save_root=SAVE_ROOT,
            class_names=CLASS_NAMES,
            normalize_cm=True,
            use_amp=True
        )

        all_fold_metrics.append(fold_metrics)
        print(f"Fold {fold} Accuracy: {fold_metrics['accuracy']*100:.2f}%")

    # ================================
    # Aggregate results
    # ================================
    summary = aggregate_kfold_results(
        fold_metrics_list=all_fold_metrics,
        model_name=MODEL_NAME,
        save_root=SAVE_ROOT,
        class_names=CLASS_NAMES
    )

    print("\n========== FINAL 5-FOLD SUMMARY (ViT-Tiny) ==========")
    print("Accuracy (mean ± std): "
          f"{summary['accuracy']['mean']*100:.2f}% ± {summary['accuracy']['std']*100:.2f}%")
    print("Macro F1 (mean ± std): "
          f"{summary['macro_avg']['f1']['mean']*100:.2f}% ± {summary['macro_avg']['f1']['std']*100:.2f}%")
    print("Kappa (mean ± std): "
          f"{summary['kappa']['mean']:.4f} ± {summary['kappa']['std']:.4f}")
    print("MCC (mean ± std): "
          f"{summary['mcc']['mean']:.4f} ± {summary['mcc']['std']:.4f}")
    print("=====================================================\n")

    print("Evaluation complete! JSON summary saved in outputs/metrics/")

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.dataset import get_dataloaders_for_fold
from src.train_utils import train_model
from models.vit_tiny import build_vit_tiny


if __name__ == "__main__":
    # ========================
    # CONFIGURATIONS
    # ========================
    NUM_CLASSES = 3
    NUM_EPOCHS = 30
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    STEP_SIZE = 10         # for scheduler
    GAMMA = 0.1            # for scheduler
    EARLY_STOPPING_PATIENCE = 7
    MODEL_NAME = "vit_tiny"
    SAVE_ROOT = "outputs"  # checkpoints, metrics, logs will go here

    # ========================
    # DEVICE SETUP
    # ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========================
    # 5-FOLD TRAINING LOOP
    # ========================
    for fold in range(5):
        print(f"\n======== Fold {fold+1} / 5 ========")

        # 1. Get DataLoaders for this fold
        train_loader, val_loader = get_dataloaders_for_fold(
            fold_index=fold,
            batch_size=BATCH_SIZE,
            num_workers=4
        )

        # 2. Build the ViT-Tiny model
        model = build_vit_tiny(
            num_classes=NUM_CLASSES,
            pretrained=True,
            image_size=224,   # must match transforms in dataset.py
            dropout=0.1       # ViT benefits from dropout
        )

        # 3. Loss function
        criterion = nn.CrossEntropyLoss()

        # 4. Optimizer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        # 5. Learning rate scheduler (optional)
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

        # 6. Train the model for this fold
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=NUM_CLASSES,
            num_epochs=NUM_EPOCHS,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            use_amp=True,                        # ViT + AMP = faster 
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            fold_num=fold,
            model_name=MODEL_NAME,
            save_root=SAVE_ROOT
        )

        print(f"\nFold {fold+1} completed. Best Val Acc: {history['best_val_acc']*100:.2f}%")

    print("\n===============================")
    print("   All 5 folds training done!  ")
    print("===============================\n")

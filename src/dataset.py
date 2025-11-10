import os
import random
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import numpy as np

# =========================
# 1. FIX RANDOM SEEDS
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# =========================
# 2. IMAGE TRANSFORMS
# =========================

# ✅ Default size = 224 (GOOD FOR RESNET, DENSENET, EFFICIENTNET)
IMAGE_SIZE = 224

# ✅ If you want ViT or higher resolution later, just uncomment:
# IMAGE_SIZE = 384

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ✅ ImageNet mean
        std=[0.229, 0.224, 0.225]    # ✅ ImageNet std
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 3. CUSTOM DATASET CLASS
# =========================
class UltrasoundDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label


# =========================
# 4. READ ALL IMAGES + LABELS
# =========================
def load_image_paths_and_labels(root_dir="./data/Covid-19_Ultrasound/"):
    classes = ["covid", "normal", "pneumonia"]
    image_paths = []
    labels = []

    for idx, cls in enumerate(classes):
        class_dir = os.path.join(root_dir, cls, "*")
        paths = glob(class_dir)  # e.g., data/.../covid/*.jpg

        image_paths.extend(paths)
        labels.extend([idx] * len(paths))

    print(f"Total images found: {len(image_paths)}")
    for i, cls in enumerate(classes):
        print(f"{cls}: {labels.count(i)}")

    return image_paths, labels


# =========================
# 5. STRATIFIED 5-FOLD SPLIT
# =========================
def get_stratified_kfolds(image_paths, labels, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for train_idx, val_idx in skf.split(image_paths, labels):
        folds.append((train_idx, val_idx))

    return folds


# =========================
# 6. DATALOADER FOR A GIVEN FOLD
# =========================
def get_dataloaders_for_fold(fold_index, batch_size=16, num_workers=4):
    # 1. Load all paths + labels
    image_paths, labels = load_image_paths_and_labels()
    labels = np.array(labels)

    # 2. Create folds
    folds = get_stratified_kfolds(image_paths, labels, n_splits=5, seed=42)
    train_idx, val_idx = folds[fold_index]

    # 3. Create subsets
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = labels[train_idx]

    val_paths = [image_paths[i] for i in val_idx]
    val_labels = labels[val_idx]

    # 4. Create datasets
    train_dataset = UltrasoundDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = UltrasoundDataset(val_paths, val_labels, transform=val_transforms)

    # 5. Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    print(f"\n===== Fold {fold_index+1} / 5 =====")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    return train_loader, val_loader

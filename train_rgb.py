# train_rgb.py
# 2


# Task 2: RGB classification on EuroSAT
# - custom Dataset using split .txt files from Task 1
# - pretrained ResNet18 with full fine-tuning
# - at least two augmentation settings
# - log accuracy + per-class TPR on validation for each epoch
# - save best model per augmentation + overall best

import os
import csv
import random
from pathlib import Path
from config import RGB_DATASET_ROOT, SPLITS_ROOT, SEED


import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# *+*+*+*+*
# CONFIG
# *+*+*+*+*
DATASET_ROOT = RGB_DATASET_ROOT
PROJECT_ROOT = SPLITS_ROOT
# SEED = 1234567  # matrikelnr

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2 # 4 is good, but with 2 there are no warnings
# =========================


def set_global_seeds(seed: int):
    """reproducible for python, numpy, torch, cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EuroSATRGBDataset(Dataset):
    """
    Dataset reads split-files a la:
        some_relative/path/from/dataset_root class_name
    и returns (image_tensor, label_index).
    """

    def __init__(self, dataset_root: Path, project_root: Path, split: str, transform=None):
        """
        split: 'train', 'val', 'test'
        """
        self.dataset_root = dataset_root
        self.project_root = project_root
        self.split = split
        self.transform = transform

        split_path = project_root / "splits" / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        rel_paths = []
        class_names = []
        with split_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, cls = line.split()
                rel_paths.append(rel)
                class_names.append(cls)

        # define mapping class -> index (lexikographisch eingeordnet)
        unique_classes = sorted(set(class_names))
        self.class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        self.img_paths = [self.dataset_root / p for p in rel_paths]
        self.targets = [self.class_to_idx[c] for c in class_names]

        assert len(self.img_paths) == len(self.targets)

        print(
            f"[{split}] Loaded {len(self.img_paths)} samples, "
            f"{len(unique_classes)} classes."
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.targets[idx]

        # open converted as RGB
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_transforms(aug_name: str):
    """
    returns (train_transform, eval_transform) for this augmentation type.
    eval_transform is the same for val/test.
    """
    # Normalization for ImageNet-pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if aug_name == "light":
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_name == "strong":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError(f"Unknown augmentation name: {aug_name}")

    return train_transform, eval_transform


def create_model(num_classes: int):
    """
    create ResNet18 with pretrained weights and replace last layer.
    All layers are trainable -> full fine-tuning ^_^ :contentReference[oaicite:1]{index=1}
    """
    try:
        # new API torchvision >= 0.13
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except AttributeError:
        # previous API
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def compute_metrics_from_predictions(all_targets, all_preds, num_classes: int):
    """
    Returns:
      - accuracy
      - per_class_tpr (list lenght of num_classes)
    """
    all_targets = np.array(all_targets, dtype=np.int64)
    all_preds = np.array(all_preds, dtype=np.int64)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_targets, all_preds):
        confusion[t, p] += 1

    correct = confusion.diagonal().sum()
    total = confusion.sum()
    acc = correct / total

    per_class_tpr = []
    for c in range(num_classes):
        tp = confusion[c, c]
        support = confusion[c, :].sum()  # all objects of the class
        if support == 0:
            tpr = 0.0
        else:
            tpr = tp / support
        per_class_tpr.append(float(tpr))

    return float(acc), per_class_tpr


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, device, criterion, num_classes: int):
    model.eval()
    running_loss = 0.0
    total = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            total += targets.size(0)

            preds = logits.argmax(dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / total
    acc, per_class_tpr = compute_metrics_from_predictions(
        all_targets, all_preds, num_classes
    )

    return avg_loss, acc, per_class_tpr


def prepare_logging():
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "rgb_metrics.csv"

    # headers create
    if not metrics_path.is_file():
        with metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            # epoch, split, aug_name, loss, accuracy, tpr_class0, ...
            header = ["epoch", "split", "aug_name", "loss", "accuracy"]
            # assume 10 classes
            for c in range(10):
                header.append(f"tpr_class{c}")
            writer.writerow(header)

    return metrics_path


def append_metrics_row(metrics_path, epoch, split, aug_name, loss, acc, per_class_tpr):
    # garanteed tpr till 10 (more -> cut, less -> add)
    tpr = list(per_class_tpr)
    if len(tpr) < 10:
        tpr = tpr + [0.0] * (10 - len(tpr))
    elif len(tpr) > 10:
        tpr = tpr[:10]

    row = [epoch, split, aug_name, loss, acc] + tpr

    with metrics_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    set_global_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics_path = prepare_logging()

    # load val/test without augmentations again
    _, eval_transform = get_transforms("light")  # same eval_transform

    # here split='val', 'test' finds files from task 1
    dummy_train_dataset = EuroSATRGBDataset(
        DATASET_ROOT, PROJECT_ROOT, split="train", transform=eval_transform
    )
    num_classes = len(dummy_train_dataset.class_to_idx)
    print(f"Detected {num_classes} classes.")

    val_dataset = EuroSATRGBDataset(
        DATASET_ROOT, PROJECT_ROOT, split="val", transform=eval_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Optional: test will be needed for logits, top-5/bottom-5)
    # test_dataset = EuroSATRGBDataset(
    #     DATASET_ROOT, PROJECT_ROOT, split="test", transform=eval_transform
    # )

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    augmentation_setups = ["light", "strong"]

    best_overall_val_acc = -1.0
    best_overall_model_path = None
    best_overall_aug_name = None

    for aug_name in augmentation_setups:
        print("\n" + "=" * 80)
        print(f"Training with augmentation: {aug_name}")
        print("=" * 80)

        train_transform, eval_transform = get_transforms(aug_name)

        train_dataset = EuroSATRGBDataset(
            DATASET_ROOT, PROJECT_ROOT, split="train", transform=train_transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # model + optimizer
        model = create_model(num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_acc = -1.0
        best_model_path = models_dir / f"rgb_best_{aug_name}.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{NUM_EPOCHS} (aug={aug_name})")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, device, optimizer, criterion
            )
            print(f"Train   loss={train_loss:.4f}, acc={train_acc:.4f}")
            append_metrics_row(
                metrics_path,
                epoch,
                "train",
                aug_name,
                train_loss,
                train_acc,
                [0.0] * num_classes,  # TPR on classes is not really needed to be counted for train
            )

            val_loss, val_acc, val_tpr = evaluate(
                model, val_loader, device, criterion, num_classes
            )
            print(f"Val     loss={val_loss:.4f}, acc={val_acc:.4f}")
            print("Val TPR per class:", ["{:.3f}".format(x) for x in val_tpr])

            append_metrics_row(
                metrics_path,
                epoch,
                "val",
                aug_name,
                val_loss,
                val_acc,
                val_tpr,
            )

            # save the best model on its val accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New best val acc={best_val_acc:.4f}, model saved to {best_model_path}")

        print(
            f"\nFinished training with aug={aug_name}. "
            f"Best val acc={best_val_acc:.4f}"
        )

        # refresh global best model
        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_overall_model_path = best_model_path
            best_overall_aug_name = aug_name

    # give a name
    final_model_path = models_dir / "rgb_final.pt"
    if best_overall_model_path is not None:
        # save state_dict as a final file
        state_dict = torch.load(best_overall_model_path, map_location="cpu")
        torch.save(state_dict, final_model_path)
        print(
            f"\nOverall best model: aug={best_overall_aug_name}, "
            f"val acc={best_overall_val_acc:.4f}"
        )
        print(f"Final model saved to: {final_model_path}")
    else:
        print("No model was trained; something went wrong.")


if __name__ == "__main__":
    main()

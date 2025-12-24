# src/train_ms.py
# 5

# Task 3: Classification with multispectral EuroSAT_MS using late fusion.
#
# ! read .tif with skimage.io.imread
# !! normalize [0, 65535] -> [0, 1]
# !!! use 6 channels: B04, B03, B02, B05, B06, B12
# !!!! makes late fusion of two groups with 3 channels with help of ResNet18 feature extractor
# !!!!! fine-tuning of all layers
# !!!!!! at least 2 types of augmentation
# !!!!!!! choose the best model by its val accuracy, save as ms_final.pt

import os
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from skimage.io import imread # !python -m pip install --user scikit-image

from torchvision import transforms, models

from train_rgb import (
    set_global_seeds,
    compute_metrics_from_predictions,
    train_one_epoch,
    evaluate,
)

# =*=*=*=*=*=*=*=*=*=*=*=*=
# CONFIG
# =*=*=*=*=*=*=*=*=*=*=*==*
# to .tif
DATASET_ROOT = Path("/home/...your_path/deepl/datasets/EuroSAT_RGB/")  
PROJECT_ROOT = Path("/home/...your_path/deepl/")  

SEED = 1234567  # matrikelnr

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2  # 4 - got warning, thats why 2
# =========================

# indexes of channels in .tif (0-based):
# 0:B01, 1:B02(Blue), 2:B03(Green), 3:B04(Red),
# 4:B05, 5:B06, 6:B07, 7:B08, 8:B8A, 9:B09, 10:B10, 11:B11, 12:B12
MS_BAND_INDICES = [3, 2, 1, 4, 5, 12]  # [R, G, B, RedEdge1, RedEdge2, SWIR3]


class EuroSATMSDataset(Dataset):
    """
    Dataset of EuroSAT_MS.

    Uses same split-files like _RGB (splits/train.txt, ...),
    but converts .jpg/.jpeg/.png into .tif.
    """

    def __init__(self, dataset_root: Path, project_root: Path, split: str, transform=None):
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

        # mapping classes like in _RGB: sorted unique class names
        unique_classes = sorted(set(class_names))
        self.class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        # convert .tif
        img_paths = []
        for rel in rel_paths:
            p = Path(rel)
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                p = p.with_suffix(".tif")
            # if .tif -> take as a .tif
            img_paths.append(self.dataset_root / p)

        self.img_paths = img_paths
        self.targets = [self.class_to_idx[c] for c in class_names]

        assert len(self.img_paths) == len(self.targets)

        print(
            f"[{split}] MS dataset: {len(self.img_paths)} samples, "
            f"{len(unique_classes)} classes."
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.targets[idx]

        # read .tif with numpy
        img = imread(str(img_path))  # expecting: (H, W, 13)
        if img.ndim == 2:
            # in case if 2dim (H, W)
            img = img[:, :, None]

        # making float32 and if already [0,1], check it by dividing
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 65535.0

        # choose needed 6 channels
        if img.shape[2] < max(MS_BAND_INDICES) + 1:
            raise RuntimeError(
                f"Image {img_path} has only {img.shape[2]} channels, "
                f"expected >= {max(MS_BAND_INDICES)+1}"
            )

        img = img[:, :, MS_BAND_INDICES]  # (H, W, 6)

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))  # (6, H, W)

        img_tensor = torch.from_numpy(img)  # float32, [0,1]

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


# --- augmentation for multispectral tensors (C,H,W) ---

class ResizeMS(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, H, W)
        return transforms.Resize(self.size)(x)


class RandomHorizontalFlipMS(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            # horisontal flip: change W-axis
            return torch.flip(x, dims=[2])
        return x


class RandomVerticalFlipMS(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            # horisontal flip: change H-axis
            return torch.flip(x, dims=[1])
        return x


class RandomRotate90MS(nn.Module):
    """Random rotate to 0, 90, 180, 270 grade without interpolation."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            k = int(torch.randint(0, 4, (1,)).item())
            x = torch.rot90(x, k, dims=[1, 2])
        return x


def get_ms_transforms(aug_name: str):
    """
    Returns (train_transform, eval_transform) for multispectral data.
    Without ToTensor/Normalize, since we in [0,1].
    """
    resize = ResizeMS((224, 224))

    eval_transform = transforms.Compose([resize])

    if aug_name == "light":
        train_transform = transforms.Compose(
            [
                resize,
                RandomHorizontalFlipMS(p=0.5),
            ]
        )
    elif aug_name == "strong":
        train_transform = transforms.Compose(
            [
                resize,
                RandomHorizontalFlipMS(p=0.5),
                RandomVerticalFlipMS(p=0.5),
                RandomRotate90MS(p=0.5),
            ]
        )
    else:
        raise ValueError(f"Unknown augmentation name: {aug_name}")

    return train_transform, eval_transform


# --- model: ResNet18 features + late fusion 2х 3channel groups ---

class ResNet18Features(nn.Module):
    """
    wrapping ResNet18, so Features after adaptive avgpool
    (till last Linear) return
    """

    def __init__(self):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            base = models.resnet18(weights=weights)
        except AttributeError:
            base = models.resnet18(pretrained=True)

        # save feature part till avgpool
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.out_dim = base.fc.in_features

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)       # (B, C, 1, 1)
        x = torch.flatten(x, 1)   # (B, C)
        return x


class EuroSATMSLateFusionNet(nn.Module):
    """
    One-network late fusion:
    - input: (B, 6, H, W)
    - divide to 2 channels (3dim+batch): (B,3,H,W) и (B,3,H,W)
    - run throught ResNet18Features (shared weights)
    - fusion: concat of 2 features
    - final: one nn.Linear -> logits
    """

    def __init__(self, num_classes: int, fusion_mode: str = "concat"):
        super().__init__()
        self.feature_net = ResNet18Features()
        self.fusion_mode = fusion_mode

        feat_dim = self.feature_net.out_dim
        if fusion_mode == "concat":
            fused_dim = feat_dim * 2
        elif fusion_mode == "sum":
            fused_dim = feat_dim
        else:
            raise ValueError("fusion_mode must be 'concat' or 'sum'")

        self.fc = nn.Linear(fused_dim, num_classes)

    def forward(self, x):
        # x: (B, 6, H, W)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]

        f1 = self.feature_net(x1)  # (B, C)
        f2 = self.feature_net(x2)  # (B, C)

        if self.fusion_mode == "concat":
            fused = torch.cat([f1, f2], dim=1)  # (B, 2C)
        else:
            fused = 0.5 * (f1 + f2)             # (B, C)

        logits = self.fc(fused)
        return logits


def prepare_ms_logging():
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "ms_metrics.csv"

    if not metrics_path.is_file():
        with metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            header = ["epoch", "split", "aug_name", "loss", "accuracy"]
            for c in range(10):
                header.append(f"tpr_class{c}")
            writer.writerow(header)

    return metrics_path


def append_ms_metrics_row(metrics_path, epoch, split, aug_name, loss, acc, per_class_tpr):
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

    metrics_path = prepare_ms_logging()

    # eval transform is the same for val/test
    _, eval_transform = get_ms_transforms("light")

    # load train (once), to define num_classes
    dummy_train_dataset = EuroSATMSDataset(
        DATASET_ROOT, PROJECT_ROOT, split="train", transform=eval_transform
    )
    num_classes = len(dummy_train_dataset.class_to_idx)
    print(f"Detected {num_classes} classes in MS dataset.")

    val_dataset = EuroSATMSDataset(
        DATASET_ROOT, PROJECT_ROOT, split="val", transform=eval_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    augmentation_setups = ["light", "strong"]

    best_overall_val_acc = -1.0
    best_overall_model_path = None
    best_overall_aug_name = None

    for aug_name in augmentation_setups:
        print("\n" + "=" * 80)
        print(f"Training MS with augmentation: {aug_name}")
        print("=" * 80)

        train_transform, eval_transform = get_ms_transforms(aug_name)

        train_dataset = EuroSATMSDataset(
            DATASET_ROOT, PROJECT_ROOT, split="train", transform=train_transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        model = EuroSATMSLateFusionNet(num_classes=num_classes, fusion_mode="concat")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_acc = -1.0
        best_model_path = models_dir / f"ms_best_{aug_name}.pt"

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{NUM_EPOCHS} (aug={aug_name})")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, device, optimizer, criterion
            )
            print(f"Train   loss={train_loss:.4f}, acc={train_acc:.4f}")
            append_ms_metrics_row(
                metrics_path,
                epoch,
                "train",
                aug_name,
                train_loss,
                train_acc,
                [0.0] * num_classes,
            )

            val_loss, val_acc, val_tpr = evaluate(
                model, val_loader, device, criterion, num_classes
            )
            print(f"Val     loss={val_loss:.4f}, acc={val_acc:.4f}")
            print("Val TPR per class:", ["{:.3f}".format(x) for x in val_tpr])

            append_ms_metrics_row(
                metrics_path,
                epoch,
                "val",
                aug_name,
                val_loss,
                val_acc,
                val_tpr,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New best val acc={best_val_acc:.4f}, model saved to {best_model_path}")

        print(
            f"\nFinished MS training with aug={aug_name}. "
            f"Best val acc={best_val_acc:.4f}"
        )

        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_overall_model_path = best_model_path
            best_overall_aug_name = aug_name

    final_model_path = models_dir / "ms_final.pt"
    if best_overall_model_path is not None:
        state_dict = torch.load(best_overall_model_path, map_location="cpu")
        torch.save(state_dict, final_model_path)
        print(
            f"\nOverall best MS model: aug={best_overall_aug_name}, "
            f"val acc={best_overall_val_acc:.4f}"
        )
        print(f"Final MS model saved to: {final_model_path}")
    else:
        print("No MS model was trained; something went wrong.")


if __name__ == "__main__":
    main()

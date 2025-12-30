# ms_test_reproduce.py
# 6 and 7

# Task 3: testing final MS-model, saving logits and
# reproduction routine, like RGB.
#
# ! at first set SAVE_LOGITS = True and run, so we get:
#   * logits of test
#   * list of files
#   * accuracy + TPR by classes on test
# !! default set SAVE_LOGITS = False,
#   new run will compare logits with previously saved logits.
#
# classes from train_rgb.py and train_ms.py will be imported

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import MS_DATASET_ROOT, SPLITS_ROOT, SEED


from train_ms import (
    EuroSATMSDataset,
    EuroSATMSLateFusionNet,
    get_ms_transforms,
)
from train_rgb import (
    set_global_seeds,
    compute_metrics_from_predictions,
)

# =*=*=*=*=*=*=*=*=*=*=
# CONFIG
# ==+==+==+==+==*==*==*
DATASET_ROOT = MS_DATASET_ROOT
PROJECT_ROOT = SPLITS_ROOT

BATCH_SIZE = 64
NUM_WORKERS = 2

MODEL_PATH = PROJECT_ROOT / "models" / "ms_final.pt"

PRED_DIR = PROJECT_ROOT / "predictions"
LOGITS_PATH = PRED_DIR / "ms_test_logits.pt"
FILENAMES_PATH = PRED_DIR / "ms_test_filenames.txt"

# dont forget to switch from True to False
SAVE_LOGITS = False # False default


# classes to be used for top-5 / bottom-5
# index of classes (0,1,2, mapping class_name -> index on the first run)
CLASS_IDS_FOR_TOP_BOTTOM = [0, 1, 2] 
TOPK = 5


def do_top_bottom_analysis(logits_tensor, filenames, idx_to_class):
    if not CLASS_IDS_FOR_TOP_BOTTOM:
        return

    logits_np = logits_tensor.numpy()
    filenames = np.array(filenames)

    print("\nTop-5 and bottom-5 images per selected class (based on logits):")
    for class_id in CLASS_IDS_FOR_TOP_BOTTOM:
        if class_id < 0 or class_id >= logits_np.shape[1]:
            print(f"\n[WARN] class_id {class_id} is out of range, skipping.")
            continue

        class_name = idx_to_class.get(class_id, f"<id {class_id}>")
        scores = logits_np[:, class_id]

        sorted_idx = np.argsort(scores)

        bottom_idx = sorted_idx[:TOPK]
        top_idx = sorted_idx[-TOPK:]

        print(f"\nClass {class_id} ({class_name}):")

        print("  TOP-5 images (highest score):")
        for i in reversed(top_idx):
            print(f"    score={scores[i]:.4f}, file={filenames[i]}")

        print("  BOTTOM-5 images (lowest score):")
        for i in bottom_idx:
            print(f"    score={scores[i]:.4f}, file={filenames[i]}")


def run_and_save_logits(model, test_loader, test_dataset, device):
    model.eval()
    all_logits = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            all_logits.append(logits.cpu())

            preds = logits.argmax(dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    logits_tensor = torch.cat(all_logits, dim=0)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(logits_tensor, LOGITS_PATH)
    print(f"Saved MS logits to {LOGITS_PATH} with shape {tuple(logits_tensor.shape)}")

    filenames = [p.relative_to(DATASET_ROOT).as_posix() for p in test_dataset.img_paths]
    with FILENAMES_PATH.open("w") as f:
        for fn in filenames:
            f.write(fn + "\n")
    print(f"Saved MS filenames to {FILENAMES_PATH} (n={len(filenames)})")

    num_classes = len(test_dataset.class_to_idx)
    test_acc, test_tpr = compute_metrics_from_predictions(
        all_targets, all_preds, num_classes
    )
    print(f"\nMS TEST performance (using final model):")
    print(f"  accuracy = {test_acc:.4f}")
    print(f"  TPR per class = {[f'{x:.3f}' for x in test_tpr]}")

    print("\nClass index mapping (for interpretation):")
    idx_to_class = {i: cls for cls, i in test_dataset.class_to_idx.items()}
    for idx in sorted(idx_to_class.keys()):
        print(f"  {idx}: {idx_to_class[idx]}")

    do_top_bottom_analysis(logits_tensor, filenames, idx_to_class)


def run_and_compare_logits(model, test_loader, test_dataset, device):
    if not LOGITS_PATH.is_file() or not FILENAMES_PATH.is_file():
        raise RuntimeError(
            "Saved MS logits or filenames not found. "
            "Run once with SAVE_LOGITS = True to create them."
        )

    saved_logits = torch.load(LOGITS_PATH, map_location="cpu")
    with FILENAMES_PATH.open("r") as f:
        saved_filenames = [line.strip() for line in f if line.strip()]

    current_filenames = [
        p.relative_to(DATASET_ROOT).as_posix() for p in test_dataset.img_paths
    ]

    if saved_filenames != current_filenames:
        print("WARNING: filenames order / content differs from saved filenames!")
        print("  First 5 saved:   ", saved_filenames[:5])
        print("  First 5 current: ", current_filenames[:5])
    else:
        print("Filenames match between saved and current MS test split.")

    model.eval()
    all_logits = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())

    new_logits = torch.cat(all_logits, dim=0)

    if new_logits.shape != saved_logits.shape:
        raise RuntimeError(
            f"Shape mismatch: new_logits {tuple(new_logits.shape)} vs "
            f"saved_logits {tuple(saved_logits.shape)}"
        )

    diff = (new_logits - saved_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    allclose = torch.allclose(new_logits, saved_logits, rtol=1e-4, atol=1e-5)

    print("\nMS reproduction check (test logits):")
    print(f"  max |diff|  = {max_diff:.6e}")
    print(f"  mean |diff| = {mean_diff:.6e}")
    print(f"  torch.allclose(rtol=1e-4, atol=1e-5) = {allclose}")


def main():
    set_global_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, eval_transform = get_ms_transforms("light")

    test_dataset = EuroSATMSDataset(
        DATASET_ROOT, PROJECT_ROOT, split="test", transform=eval_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    num_classes = len(test_dataset.class_to_idx)
    print(f"MS test set: {len(test_dataset)} samples, {num_classes} classes")

    model = EuroSATMSLateFusionNet(num_classes=num_classes, fusion_mode="concat")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Final MS model not found: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    if SAVE_LOGITS:
        print("\n[MODE] SAVE_LOGITS = True → will compute and save MS logits on test.")
        run_and_save_logits(model, test_loader, test_dataset, device)
    else:
        print("\n[MODE] SAVE_LOGITS = False → will compare new logits with saved ones.")
        run_and_compare_logits(model, test_loader, test_dataset, device)


if __name__ == "__main__":
    main()

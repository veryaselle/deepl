# src/rgb_test_reproduce.py
# 3 and 4


# Task 2: test of final RGB model, logit saving and
# reproduction routine
#
# ! at first set SAVE_LOGITS = True and run, so we get:
#   * logits of test
#   * list of files
#   * accuracy + TPR by classes on test
# !! default set SAVE_LOGITS = False,
#   new run will compare logits with previously saved logits.
#
# classes from train_rgb.py will be imported

from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_rgb import (
    EuroSATRGBDataset,
    get_transforms,
    create_model,
    compute_metrics_from_predictions,
    set_global_seeds,
)

# ***let**it**snow***
# CONFIG
# +++happy++new++year++
DATASET_ROOT = Path("/home/...your_path/deepl/datasets/EuroSAT_RGB/")  
PROJECT_ROOT = Path("/home/...your_path/deepl/") 
SEED = 1234567  # matrileknr

BATCH_SIZE = 64
NUM_WORKERS = 4 # change to 2 if warnings coming :D

# final model path
MODEL_PATH = PROJECT_ROOT / "models" / "rgb_final.pt"

# saving, logits and filename pathes
PRED_DIR = PROJECT_ROOT / "predictions"
LOGITS_PATH = PRED_DIR / "rgb_test_logits.pt"
FILENAMES_PATH = PRED_DIR / "rgb_test_filenames.txt"

# "basic" logits run by SAVE_LOGITS = True
SAVE_LOGITS = False  # "switch is off by default"
# SAVE_LOGITS = True


# classes to be used for top-5 / bottom-5
# index of classes (0,1,2, mapping class_name -> index on the first run)
CLASS_IDS_FOR_TOP_BOTTOM = [0, 1, 2]
TOPK = 5


def run_and_save_logits(model, test_loader, test_dataset, device):
    """
    running test model, save:
      - logits (ndata, nclasses) in LOGITS_PATH
      - file lists in FILENAMES_PATH
    compute accuracy and TPR by classes on test.
    """
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

    logits_tensor = torch.cat(all_logits, dim=0)  # (N, C)

    # save logits
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(logits_tensor, LOGITS_PATH)
    print(f"Saved logits to {LOGITS_PATH} with shape {tuple(logits_tensor.shape)}")

    # save list of files im bezug auf DATASET_ROOT
    filenames = [p.relative_to(DATASET_ROOT).as_posix() for p in test_dataset.img_paths]
    with FILENAMES_PATH.open("w") as f:
        for fn in filenames:
            f.write(fn + "\n")
    print(f"Saved filenames to {FILENAMES_PATH} (n={len(filenames)})")

    # compute accuracy and TPR by classes on test
    num_classes = len(test_dataset.class_to_idx)
    test_acc, test_tpr = compute_metrics_from_predictions(
        all_targets, all_preds, num_classes
    )
    print(f"\nTEST performance (using final model):")
    print(f"  accuracy = {test_acc:.4f}")
    print(f"  TPR per class = {[f'{x:.3f}' for x in test_tpr]}")

    # print mapping index -> class
    print("\nClass index mapping (for interpretation):")
    idx_to_class = {i: cls for cls, i in test_dataset.class_to_idx.items()}
    for idx in sorted(idx_to_class.keys()):
        print(f"  {idx}: {idx_to_class[idx]}")

    # top-5 / bottom-5 for 3 classes
    do_top_bottom_analysis(logits_tensor, filenames, idx_to_class)


def do_top_bottom_analysis(logits_tensor, filenames, idx_to_class):
    """
    top-5 and bottom-5 images based on logit-score for selected classes.
    """
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

        # sort indexes by their score
        sorted_idx = np.argsort(scores)  # asc

        bottom_idx = sorted_idx[:TOPK]
        top_idx = sorted_idx[-TOPK:]

        print(f"\nClass {class_id} ({class_name}):")

        print("  TOP-5 images (highest score):")
        for i in reversed(top_idx):  # desc
            print(f"    score={scores[i]:.4f}, file={filenames[i]}")

        print("  BOTTOM-5 images (lowest score):")
        for i in bottom_idx:
            print(f"    score={scores[i]:.4f}, file={filenames[i]}")


def run_and_compare_logits(model, test_loader, test_dataset, device):
    """
    Reproduction routine:
    - the logit scores on test data using the saved model
    - which are computed newly when we run your code 
    - should be compared against the logit
    - scores which you saved when we ran your code
    - (compares max|diff| and torch.allclose)
    """
    if not LOGITS_PATH.is_file() or not FILENAMES_PATH.is_file():
        raise RuntimeError(
            f"Saved logits or filenames not found. "
            f"Run once with SAVE_LOGITS = True to create them."
        )

    # load saved dataset
    saved_logits = torch.load(LOGITS_PATH, map_location="cpu")
    with FILENAMES_PATH.open("r") as f:
        saved_filenames = [line.strip() for line in f if line.strip()]

    # ordered current filenames from dataset
    current_filenames = [
        p.relative_to(DATASET_ROOT).as_posix() for p in test_dataset.img_paths
    ]

    if saved_filenames != current_filenames:
        print("WARNING: filenames order / content differs from saved filenames!")
        print("  First 5 saved:   ", saved_filenames[:5])
        print("  First 5 current: ", current_filenames[:5])
    else:
        print("Filenames match between saved and current test split.")

    # compute new logits
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

    print("\nReproduction check (test logits):")
    print(f"  max |diff|  = {max_diff:.6e}")
    print(f"  mean |diff| = {mean_diff:.6e}")
    print(f"  torch.allclose(rtol=1e-4, atol=1e-5) = {allclose}")


def main():
    set_global_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # eval transform is the same like in train_rgb (for val/test)
    _, eval_transform = get_transforms("light")

    # load test dataset
    test_dataset = EuroSATRGBDataset(
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
    print(f"Test set: {len(test_dataset)} samples, {num_classes} classes")

    # create model and load final models weight
    model = create_model(num_classes)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Final model not found: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    if SAVE_LOGITS:
        print("\n[MODE] SAVE_LOGITS = True → will compute and save logits on test.")
        run_and_save_logits(model, test_loader, test_dataset, device)
    else:
        print("\n[MODE] SAVE_LOGITS = False → will compare new logits with saved ones.")
        run_and_compare_logits(model, test_loader, test_dataset, device)


if __name__ == "__main__":
    main()

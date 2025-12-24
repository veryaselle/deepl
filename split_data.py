# src/split_data.py
# task 1

import os
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split
import numpy as np
import random

# =========================
# CONFIG
# =========================
DATASET_ROOT = Path("/home/...your_path/deepl/datasets/EuroSAT_RGB/")  
PROJECT_ROOT = Path("/home/...your_path/deepl/")  
SEED = 1234567  # matrikelnr

# 60 / 20 / 20
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
# =========================


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def collect_files(dataset_root: Path):
    """
    gather files <DATASET_ROOT>/<class_name>/<filename>
    return list[(rel_path_str, class_name)]
    """
    all_items = []

    # each class is a separate subdirectory
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        # extract '*.jpg' or '*.tif')
        for f in sorted(class_dir.iterdir()):
            if not f.is_file():
                continue
            rel_path = f.relative_to(dataset_root)
            all_items.append((str(rel_path), class_name))

    return all_items


def stratified_split_per_class(items, train_ratio, val_ratio, test_ratio, seed):
    """
    items: list[(rel_path_str, class_name)]
    split class separately and compile global lists
    """
    by_class = defaultdict(list)
    for rel_path, cls in items:
        by_class[cls].append(rel_path)

    train_files = []
    val_files = []
    test_files = []

    for cls, file_list in by_class.items():
        file_list = np.array(file_list)

        # extract test
        train_val_files, test_cls = train_test_split(
            file_list,
            test_size=test_ratio,
            random_state=seed,
        )
        # then from train_val extract val
        remaining = 1.0 - test_ratio
        val_rel = val_ratio / remaining  # part of train_val -> to val

        train_cls, val_cls = train_test_split(
            train_val_files,
            test_size=val_rel,
            random_state=seed,
        )

        # classify with labels
        train_files.extend((p, cls) for p in train_cls)
        val_files.extend((p, cls) for p in val_cls)
        test_files.extend((p, cls) for p in test_cls)

    return train_files, val_files, test_files


def verify_disjointness(train_files, val_files, test_files, total_items):
    """
    We verify that the sets of paths do not intersect and that their union equals all elements
    """
    train_set = set(p for p, _ in train_files)
    val_set = set(p for p, _ in val_files)
    test_set = set(p for p, _ in test_files)

    assert train_set.isdisjoint(val_set), "Train and Val intersect!"
    assert train_set.isdisjoint(test_set), "Train and Test intersect!"
    assert val_set.isdisjoint(test_set), "Val and Test intersect!"

    union = train_set | val_set | test_set
    all_paths = set(p for p, _ in total_items)

    assert len(union) == len(all_paths), (
        f"The size of the union ({len(union)}) is not equal to the number of all files ({len(all_paths)})"
    )
    print("OK! Splits are disjoint and cover all files :)")


def save_split(split_files, split_name, project_root: Path):
    """
    split_files: list[(rel_path_str, class_name)]
    """
    splits_dir = project_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    out_path = splits_dir / f"{split_name}.txt"
    with out_path.open("w") as f:
        for rel_path, cls in split_files:
            f.write(f"{rel_path} {cls}\n")

    print(f"Saved {split_name} split to {out_path} (n={len(split_files)})")


def main():
    # 1. set your own seed or matrikelnr above
    set_seeds(SEED)

    # 2. gather files
    print(f"Scanning dataset in {DATASET_ROOT} ...")
    all_items = collect_files(DATASET_ROOT)
    print(f"Found {len(all_items)} files.")

    # check if files are enough
    if len(all_items) < 2500 + 1000 + 2000:
        raise RuntimeError("Required picture number is not satisfied, unzip your files correctly!")

    # 3. stratified splitting of each classes
    train_files, val_files, test_files = stratified_split_per_class(
        all_items, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
    )

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # 4. disjuct?
    verify_disjointness(train_files, val_files, test_files, all_items)

    # 5. then save
    save_split(train_files, "train", PROJECT_ROOT)
    save_split(val_files, "val", PROJECT_ROOT)
    save_split(test_files, "test", PROJECT_ROOT)

    print("Done.")


if __name__ == "__main__":
    main()

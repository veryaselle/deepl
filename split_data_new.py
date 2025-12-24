# split_data.py
import os
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from config import RGB_DATASET_ROOT, SPLITS_ROOT, SEED

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def collect_files(dataset_root: Path):
    all_items = []
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir(): continue
        class_name = class_dir.name
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in ['.jpg', '.jpeg', '.tif', '.tiff']:
                rel_path = f.relative_to(dataset_root)
                all_items.append((str(rel_path), class_name))
    return all_items

def stratified_split_per_class(items, train_ratio, val_ratio, test_ratio, seed):
    by_class = defaultdict(list)
    for rel_path, cls in items:
        by_class[cls].append(rel_path)

    train_files, val_files, test_files = [], [], []

    for cls, file_list in by_class.items():
        file_list = np.array(file_list)
        
        train_val_files, test_cls = train_test_split(
            file_list, test_size=test_ratio, random_state=seed
        )
        
        remaining = 1.0 - test_ratio
        val_rel = val_ratio / remaining
        train_cls, val_cls = train_test_split(
            train_val_files, test_size=val_rel, random_state=seed
        )
        
        train_files.extend((p, cls) for p in train_cls)
        val_files.extend((p, cls) for p in val_cls)
        test_files.extend((p, cls) for p in test_cls)
    return train_files, val_files, test_files

def save_split(split_files, split_name, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split_name}.txt"
    with out_path.open("w") as f:
        for rel_path, cls in split_files:
            f.write(f"{rel_path} {cls}\n")
    print(f"Saved {split_name} split to {out_path} (n={len(split_files)})")

def main():
    set_seeds(SEED)
    print(f"Scanning dataset in {RGB_DATASET_ROOT} ...")
    all_items = collect_files(RGB_DATASET_ROOT)
    
    train_f, val_f, test_f = stratified_split_per_class(
        all_items, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
    )
    
    save_split(train_f, "train", SPLITS_ROOT)
    save_split(val_f, "val", SPLITS_ROOT)
    save_split(test_f, "test", SPLITS_ROOT)
    print("Done.")

if __name__ == "__main__":
    main()
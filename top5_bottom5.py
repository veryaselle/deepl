# Visual reprasenation of pictures RGB and MS test set.
#
# top-5 / bottom-5 for each classes logits
# random

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from skimage.io import imread
import matplotlib.pyplot as plt
from config import RGB_DATASET_ROOT, MS_DATASET_ROOT, SEED



# ==*==*===*===*====*====*
# CONFIG
# ===*===*===*===*===*===*


RGB_DATASET_ROOT = RGB_DATASET_ROOT
MS_DATASET_ROOT = MS_DATASET_ROOT

RGB_FILENAMES_PATH = PROJECT_ROOT / "predictions" / "rgb_test_filenames.txt"
MS_FILENAMES_PATH = PROJECT_ROOT / "predictions" / "ms_test_filenames.txt"

RGB_LOGITS_PATH = PROJECT_ROOT / "predictions" / "rgb_test_logits.pt"
MS_LOGITS_PATH = PROJECT_ROOT / "predictions" / "ms_test_logits.pt"

# which channels from MS we use (0-based index)
# 0:B01, 1:B02(Blue), 2:B03(Green), 3:B04(Red), ...
MS_RGB_BANDS = (3, 2, 1)  # (Red, Green, Blue)

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def load_filenames(path):
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def check_alignment(rgb_files, ms_files):
    if len(rgb_files) != len(ms_files):
        print(f"[WARN] Different lengths: RGB={len(rgb_files)}, MS={len(ms_files)}")
        return False

    same_base = True
    for i, (r, m) in enumerate(zip(rgb_files, ms_files)):
        rb = Path(r).with_suffix("").as_posix()
        mb = Path(m).with_suffix("").as_posix()
        if rb != mb:
            print(f"[WARN] mismatch at index {i}:")
            print(f"  RGB: {r}")
            print(f"  MS : {m}")
            same_base = False
            break

    if same_base:
        print(f"[Hm...OK cat lovers...] RGB and MS filename lists align (same order, same basenames), we continue:")
    return same_base


def load_rgb_image(rel_path: str):
    """load RGB image"""
    path = RGB_DATASET_ROOT / rel_path
    img = Image.open(path).convert("RGB")
    return np.array(img)


# changes:
def load_ms_rgb_projection(rel_path: str):
    """load MS .tif and make rgb channel contrasted (B04,B03,B02)
    """
    path = MS_DATASET_ROOT / rel_path
    img = imread(str(path))  

    if img.ndim == 2:
        img = img[:, :, None]

    img = img.astype(np.float32)

    # if (C, H, W), -> (H, W, C)
    if img.shape[0] == 13 and img.shape[-1] != 13:
        # transposing CHW -> HWC
        img = np.transpose(img, (1, 2, 0))

    # normalize to [0,1]
    if img.max() > 1.0:
        img = img / 65535.0

    # 3 channels (Red, Green, Blue) taken
    c_r, c_g, c_b = MS_RGB_BANDS
    if img.shape[2] <= max(MS_RGB_BANDS):
        raise RuntimeError(
            f"MS image {path} has only {img.shape[2]} channels, expected > {max(MS_RGB_BANDS)}"
        )

    rgb_ms = img[:, :, [c_r, c_g, c_b]]  # (H, W, 3)

    # contrast into [0,1], otherwise ms picture shown as black
    mn = rgb_ms.min()
    mx = rgb_ms.max()
    if mx > mn:
        rgb_ms = (rgb_ms - mn) / (mx - mn)
    else:
        rgb_ms = np.zeros_like(rgb_ms)

    rgb_ms = np.clip(rgb_ms, 0.0, 1.0)
    return rgb_ms



def show_pair(rgb_rel, ms_rel, title=None):
    """show RGB + MS configured RGB"""
    rgb_img = load_rgb_image(rgb_rel)
    ms_img = load_ms_rgb_projection(ms_rel)

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.axis("off")
    plt.title("RGB")

    plt.subplot(1, 2, 2)
    plt.imshow(ms_img)
    plt.axis("off")
    plt.title("MS (B04,B03,B02)")

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def random_examples(rgb_files, ms_files, n=5):
    """show random n puctures"""
    idxs = list(range(len(rgb_files)))
    random.Random(RANDOM_SEED).shuffle(idxs)
    idxs = idxs[:n]

    for i in idxs:
        print(f"\nExample idx={i}: {rgb_files[i]}")
        show_pair(rgb_files[i], ms_files[i], title=f"idx={i}: {rgb_files[i]}")


def top_bottom_by_class(
    class_id: int, rgb_files, ms_files, rgb_logits, ms_logits, topk=5
):
    """show top-5 and bottom-5 for given classes with logits"""
    assert 0 <= class_id < rgb_logits.shape[1]

    class_name = CLASS_NAMES[class_id]

    rgb_scores = rgb_logits[:, class_id]
    ms_scores = ms_logits[:, class_id]

    # sort index by its score
    rgb_sorted = np.argsort(rgb_scores)
    ms_sorted = np.argsort(ms_scores)

    rgb_top = rgb_sorted[-topk:]
    rgb_bottom = rgb_sorted[:topk]

    ms_top = ms_sorted[-topk:]
    ms_bottom = ms_sorted[:topk]

    print(f"\n CLASS {class_id}: {class_name} ")

    # RGB TOP
    print("\nRGB TOP-5:")
    for idx in reversed(rgb_top):
        print(f"  score={rgb_scores[idx]:.4f}, file={rgb_files[idx]}")
        show_pair(rgb_files[idx], ms_files[idx], title=f"RGB TOP, class {class_name}, idx={idx}")

    # RGB BOTTOM
    print("\nRGB BOTTOM-5:")
    for idx in rgb_bottom:
        print(f"  score={rgb_scores[idx]:.4f}, file={rgb_files[idx]}")
        show_pair(rgb_files[idx], ms_files[idx], title=f"RGB BOTTOM, class {class_name}, idx={idx}")

    # MS TOP
    print("\nMS TOP-5:")
    for idx in reversed(ms_top):
        print(f"  score={ms_scores[idx]:.4f}, file={ms_files[idx]}")
        show_pair(rgb_files[idx], ms_files[idx], title=f"MS TOP, class {class_name}, idx={idx}")

    # MS BOTTOM
    print("\nMS BOTTOM-5:")
    for idx in ms_bottom:
        print(f"  score={ms_scores[idx]:.4f}, file={ms_files[idx]}")
        show_pair(rgb_files[idx], ms_files[idx], title=f"MS BOTTOM, class {class_name}, idx={idx}")


def main():
    print("Loading filename lists...")
    rgb_files = load_filenames(RGB_FILENAMES_PATH)
    ms_files = load_filenames(MS_FILENAMES_PATH)
    print(f"RGB files: {len(rgb_files)}, MS files: {len(ms_files)}")

    aligned = check_alignment(rgb_files, ms_files)
    if not aligned:
        print("WARNING: lists are not aligned; continue at your own risk :)")

    print("\nLoading logits...")
    rgb_logits = torch.load(RGB_LOGITS_PATH, map_location="cpu").numpy()  # (N, C)
    ms_logits = torch.load(MS_LOGITS_PATH, map_location="cpu").numpy()

    print(f"RGB logits shape: {rgb_logits.shape}")
    print(f"MS  logits shape: {ms_logits.shape}")

    # random samples
    print("\nShowing random RGB+MS pairs...")
    random_examples(rgb_files, ms_files, n=5)

    # top/bottom for selected class
    while True:
        print("\nEnter class id (0-9) to inspect top/bottom examples, or 'q' to quit.")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {i}: {name}")
        s = input("Class id (or q): ").strip()
        if s.lower() == "q":
            break
        try:
            cid = int(s)
        except ValueError:
            print("Not a valid integer, try again.")
            continue
        if 0 <= cid < len(CLASS_NAMES):
            top_bottom_by_class(cid, rgb_files, ms_files, rgb_logits, ms_logits)
        else:
            print("Class id out of range, try again.")


if __name__ == "__main__":
    main()

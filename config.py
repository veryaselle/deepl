# src/config.py


from pathlib import Path
import os

# make your own changes to place your project right
PROJECT_ROOT = Path("/home/...your_path")

DATASETS_ROOT = PROJECT_ROOT / "/...datasets"

#   datasets/EuroSAT_RGB
#   datasets/EuroSAT_MS
RGB_DATASET_ROOT = Path(
    os.environ.get("RGB_DATASET_ROOT", DATASETS_ROOT / "EuroSAT_RGB")
)

MS_DATASET_ROOT = Path(
    os.environ.get("MS_DATASET_ROOT", DATASETS_ROOT / "EuroSAT_MS")
)

SPLITS_ROOT = PROJECT_ROOT / "splits"

SEED = 123456  # matrikelnr

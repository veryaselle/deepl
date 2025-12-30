# config.py


from pathlib import Path
import os

# make sure to place your project right
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

SEED = 3719704  # matrikelnr


# on terminal
# export RGB_DATASET_ROOT=/home/to_your_path/deepl/datasets/EuroSAT_RGB
# export MS_DATASET_ROOT=/home/to_your_path/deepl/datasets/EuroSAT_MS


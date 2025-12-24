# config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS_ROOT = PROJECT_ROOT 

RGB_DATASET_ROOT = Path(
    os.environ.get("RGB_DATASET_ROOT", DATASETS_ROOT / "EuroSAT_RGB")
)

MS_DATASET_ROOT = Path(
    os.environ.get("MS_DATASET_ROOT", DATASETS_ROOT / "EuroSAT_MS")
)

SPLITS_ROOT = PROJECT_ROOT / "splits"

SEED = 3719704

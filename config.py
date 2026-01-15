# Start
# 0

# src/config.py 


# PROJECT_ROOT/
## src/...
## datasets/EuroSAT_RGB/
## datasets/EuroSAT_MS/
## splits/
## models/
## ..


from pathlib import Path

# As a tester your RGB and MS datasets should be saved one level up in "datasets" folder, so you dont have to copy/paste them into src folder 
# Automatic detection by default: you should not change anything manually, just pay attention to your dataset placement


PROJECT_ROOT = Path(__file__).resolve().parents[1] 


# update path to your local if necessary (DATASETS_ROOT = PROJECT_ROOT / "***your dataset folder****")
DATASETS_ROOT = PROJECT_ROOT / "datasets"
# alternatively: DATASETS_ROOT = PROJECT_ROOT # / "***your dataset folder****"

# splits will be saved as "splits"
SPLITS_ROOT = PROJECT_ROOT # train.txt / val.txt / test txt

# Dataset paths
RGB_DATASET_ROOT = DATASETS_ROOT / "EuroSAT_RGB"
MS_DATASET_ROOT = DATASETS_ROOT / "EuroSAT_MS"

# Fixed seed
SEED = 3719704

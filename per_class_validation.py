# Per-Class Validation Accuracy RGB


import pandas as pd
import matplotlib.pyplot as plt
from config import PROJECT_ROOT

df = pd.read_csv(PROJECT_ROOT / "logs" / "rgb_metrics.csv")

df_val = df[(df["split"] == "val") & (df["aug_name"] == "strong")]

class_names = [
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

def col_name(i: int) -> str:
    return f"tpr_class{i}"   

fig, ax = plt.subplots(figsize=(10, 5))

for i, name in enumerate(class_names):
    col = col_name(i)
    ax.plot(df_val["epoch"], df_val[col], marker="o", label=name)




ax.set_xlabel("Epochs")
ax.set_ylabel("Validation TPR")
ax.set_title("Per-Class Validation Accuracy (RGB, strong augmentation)")
ax.set_ylim(0.6, 1.0)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()



# Per-Class Validation Accuracy MS

df = pd.read_csv(PROJECT_ROOT / "logs" / "ms_metrics.csv")

df_val = df[(df["split"] == "val") & (df["aug_name"] == "strong")]

class_names = [
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

def col_name(i: int) -> str:
    return f"tpr_class{i}"   

fig, ax = plt.subplots(figsize=(10, 5))

for i, name in enumerate(class_names):
    col = col_name(i)
    ax.plot(df_val["epoch"], df_val[col], marker="o", label=name)




ax.set_xlabel("Epochs")
ax.set_ylabel("Validation TPR")
ax.set_title("Per-Class Validation Accuracy (MS, strong augmentation)")
ax.set_ylim(0.6, 1.0)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()

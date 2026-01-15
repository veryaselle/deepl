# !pip install --user pandas
# after 7-8 epochs MS model is getting overfitted: 
# Figures "MS accuracy and loss" illustrates the same curves for the multispectral model with strong augmentation. 
# Training accuracy quickly approaches almost 100% and training loss decreases towards zero whereas validation accuracy peaks at about 91% around epoch 7 and drops afterwards. 
# The validation loss reaches its minimum at epoch 7 and then increases sharply. 
# This behaviour is typical for overfitting, and therefore the checkpoint from epoch 7 is selected as the final MS model (..ms_final.pt) 
# which achieves approximately 90.8% accuracy on the held-out test set
# (for comparison RGB below)


import pandas as pd
import matplotlib.pyplot as plt

from config import PROJECT_ROOT

df = pd.read_csv(PROJECT_ROOT / "logs" / "ms_metrics.csv")

# strong augmentation
df_strong_train = df[(df["aug_name"] == "strong") & (df["split"] == "train")]
df_strong_val   = df[(df["aug_name"] == "strong") & (df["split"] == "val")]

# accuracy
plt.figure()
plt.plot(df_strong_train["epoch"], df_strong_train["accuracy"], label="train")
plt.plot(df_strong_val["epoch"],   df_strong_val["accuracy"],   label="val")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
plt.title("MS accuracy (strong aug)")
plt.show()

# loss
plt.figure()
plt.plot(df_strong_train["epoch"], df_strong_train["loss"], label="train")
plt.plot(df_strong_val["epoch"],   df_strong_val["loss"],   label="val")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.title("MS loss (strong aug)")
plt.show()






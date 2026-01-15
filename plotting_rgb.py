# !pip install --user pandas
# RGB model is stable:
# Figures "RGB accuracy and loss" show the training and validation curves for the RGB model with strong augmentation. 
# Both training and validation accuracy increase steadily and reach values around 97–98% after 10 epochs, while the corresponding losses decrease smoothly. 
# There is no clear divergence between train and validation curves, which indicates that the RGB model does not suffer from severe overfitting in this setting.


import pandas as pd
import matplotlib.pyplot as plt


from config import PROJECT_ROOT

df = pd.read_csv(PROJECT_ROOT / "logs" / "rgb_metrics.csv")


# strong augmentation
df_strong_train = df[(df["aug_name"] == "strong") & (df["split"] == "train")]
df_strong_val   = df[(df["aug_name"] == "strong") & (df["split"] == "val")]

# accuracy
plt.figure()
plt.plot(df_strong_train["epoch"], df_strong_train["accuracy"], label="train")
plt.plot(df_strong_val["epoch"],   df_strong_val["accuracy"],   label="val")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
plt.title("RGB accuracy (strong aug)")
plt.show()

# loss
plt.figure()
plt.plot(df_strong_train["epoch"], df_strong_train["loss"], label="train")
plt.plot(df_strong_val["epoch"],   df_strong_val["loss"],   label="val")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.title("RGB loss (strong aug)")
plt.show()
Data source: https://huggingface.co/datasets/torchgeo/eurosat



Start with downloading pytorch :D

Code is good commented, but anyways:

0. Change pathes to your own properly in config.py

1. split_data.py
task 1



2. train_rgb.py
Task 2a

RGB classification on EuroSAT
- custom Dataset using split .txt files from Task 1
- pretrained ResNet18 with full fine-tuning
- at least two augmentation settings
- log accuracy + per-class TPR on validation for each epoch
- save best model per augmentation + overall best


3. rgb_test_reproduce.py

Task 2b: test of final RGB model, logit saving and reproduction routine

At first set SAVE_LOGITS = True and run, so we get:
* logits of test
* list of files
* accuracy + TPR by classes on test

Then default set SAVE_LOGITS = False,
new run will compare logits with previously saved logits.



4. train_ms.py

Task 3a: Classification with multispectral EuroSAT_MS using late fusion

* read .tif with skimage.io.imread
* normalize [0, 65535] -> [0, 1]
* use 6 channels: B04, B03, B02, B05, B06, B12
* makes late fusion of two groups with 3 channels with help of ResNet18 feature extractor
* fine-tuning of all layers
* at least 2 types of augmentation
* choose the best model by its val accuracy, save as ms_final.pt


5. ms_test_reproduce.py

Task 3b: testing final MS-model, saving logits and reproduction routine, like RGB.

At first set SAVE_LOGITS = True and run, so we get:
* logits of test
* list of files
* accuracy + TPR by classes on test

Then default set SAVE_LOGITS = False,
new run will compare logits with previously saved logits.


If everything ok, look through monitoring_plotting.ipynb, run and compare.
Have fun!





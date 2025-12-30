Data source: https://huggingface.co/datasets/torchgeo/eurosat



Start with downloading pytorch :D



## Paths and project structure

This project strictly follows the path requirements from the assignment:

- **No absolute paths are hard-coded in the code.** 
  All paths are defined **relative to the project root** via `config.py`.

- **Dataset roots** are defined as:
  - `RGB_DATASET_ROOT`
  - `MS_DATASET_ROOT` 
  By default, the following relative directories are used:
  - `datasets/EuroSAT_RGB`
  - `datasets/EuroSAT_MS` 
  This means that if the datasets are placed in these folders, **no code changes are required**.

- Optionally, the dataset locations can be overridden **without modifying the code** by using environment variables:
  ```bash
  export RGB_DATASET_ROOT=/path/to/EuroSAT_RGB
  export MS_DATASET_ROOT=/path/to/EuroSAT_MS





Code is good commented, but anyways:

0. Export datasets and take a look at **config.py** properly

1. **split_data.py**
Task 1



2. **train_rgb.py**
Task 2a

RGB classification on EuroSAT
- custom Dataset using split .txt files from Task 1
- pretrained ResNet18 with full fine-tuning
- at least two augmentation settings
- log accuracy + per-class TPR on validation for each epoch
- save best model per augmentation + overall best


3. **rgb_test_reproduce.py**

Task 2b: test of final RGB model, logit saving and reproduction routine

At first set SAVE_LOGITS = True and run, so we get:
* logits of test
* list of files
* accuracy + TPR by classes on test

Then default set SAVE_LOGITS = False,
new run will compare logits with previously saved logits.



4. **train_ms.py**

Task 3a: Classification with multispectral EuroSAT_MS using late fusion

* read .tif with skimage.io.imread
* normalize [0, 65535] -> [0, 1]
* use 6 channels: B04, B03, B02, B05, B06, B12
* makes late fusion of two groups with 3 channels with help of ResNet18 feature extractor
* fine-tuning of all layers
* at least 2 types of augmentation
* choose the best model by its val accuracy, save as ms_final.pt


5. **ms_test_reproduce.py**

Task 3b: testing final MS-model, saving logits and reproduction routine, like RGB.

At first set SAVE_LOGITS = True and run, so we get:
* logits of test
* list of files
* accuracy + TPR by classes on test

Then default set SAVE_LOGITS = False,
new run will compare logits with previously saved logits.


If everything ok, look through **monitoring_plotting.ipynb**, run and compare.
Have fun!





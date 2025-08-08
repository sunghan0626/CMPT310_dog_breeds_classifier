# SFU CMPT 310 Project – Dog Breed Classifier

**DogBreedVision** is a deep learning–based application for classifying dog breeds from images using a fine-tuned ResNet-18 model trained on the Stanford Dogs dataset.

## Team Members
- Yein Hwang  
- Sung Han  
- Cody Huang  

---

## Table of Contents
1. [Demo](#demo)  
2. [Installation](#installation)  
3. [Reproduction](#reproduction)  
4. [Running the Web Application](#webapp)  
5. [Guidance](#guidance)  

---

<a name="demo"></a>
## 1. Example Demo

Minimal example of using the trained model in Python:

```python
from PIL import Image
from prediction import predict_with_confidence

img = Image.open("example_dog.jpg").convert("RGB")
results = predict_with_confidence(img)

for breed, conf in results:
    print(f"{breed}: {conf:.2%}")
```

---

### Repository Structure
```bash
repository
├── augmentations.py       # Data augmentation definitions
├── config.py              # Global configuration values (paths, hyperparameters)
├── data_utils.py          # Dataset loading, preprocessing, and splitting
├── evaluate.py            # Model evaluation & metrics visualization
├── main.py                # Main training entry point
├── model_utils.py         # Model architecture and related utilities
├── prediction.py          # Prediction helper functions for UI/backend
├── train.py               # Training loop logic
├── training_plot.py       # Training convergence plotting
├── visualization.py       # Per-class accuracy & other visualization
├── web_ui.py              # Streamlit web interface for breed classification
├── requirements.yml       # Conda environment specification
├── best_model.pth         # Trained model weights (saved)
└── data/                  # Dataset (Stanford Dogs raw + processed)
```

---

<a name="installation"></a>
## 2. Installation

> **Note:** Project was developed in a Windows environment.  
> Ensure Conda is installed and functional.

```bash
git clone <this_repo_url>
cd <this_repo_name>
conda env create -f requirements.yml
conda activate dogbreed_project
```

---

<a name="reproduction"></a>
## 3. Reproduction

Steps to preprocess the dataset, train the model, and save the checkpoint:

```bash
# Step 1: Download Stanford Dogs dataset and place it in data/stanford_dogs/Images

# Step 2: Split and preprocess
python -c "from data_utils import load_dataset, split_and_save_dataset; paths, labels = load_dataset('data/stanford_dogs/Images'); split_and_save_dataset(paths, labels)"

# Step 3: Train the model
python main.py
```

**Output:**  
- Model checkpoints saved as `best_model.pth` and `baseline_cnn.pth` in the project root.  
- Processed dataset stored under `data/Processed/`.

---

<a name="webapp"></a>
## 4. Running the Web Application

After training or placing `best_model.pth` in the root folder:

```bash
conda activate dogbreed_project
streamlit run web_ui.py
```

Then, open the local Streamlit URL in your browser, upload a dog image, and view predictions.

---

<a name="guidance"></a>
## 5. Guidance

- **Git Usage**
  - No rebasing history for shared branches.
  - Commit messages must be informative.
  - Exclude large datasets from version control — add `data/Processed/` and raw dataset paths to `.gitignore` if not sharing processed data.
- **Code**
  - Modularized — keep preprocessing, augmentation, training, and prediction separate.
  - Follow consistent style for imports and constants.
- **Environment**
  - Tested in Conda with PyTorch and Streamlit.
  - Use `requirements.yml` for reproducibility.

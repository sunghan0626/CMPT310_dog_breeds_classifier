# SFU CMPT 310 Project – Dog Breed Classifier

**Dog Breed Classifier** is a deep learning–based application for classifying dog breeds from images using a fine-tuned ResNet-18 model trained on the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

## Team Members
- Yein Hwang  
- Sung Han  
- Cody Huang  

---

## Table of Contents
1. [Overview](#overview)  
2. [Repository Structure](#structure)  
3. [Installation](#installation)  
4. [Usage Options](#usage)  
   - Option 1: Run the web demo with the existing model  
   - Option 2: Reproduce training and evaluation  
5. [Example CLI Prediction](#cli)  

---

<a name="overview"></a>
## 1. Overview

This project trains and evaluates a ResNet-18 convolutional neural network on the Stanford Dogs dataset to predict the breed of a dog from an image.  
The workflow includes:
- Preprocessing & splitting the dataset  
- Applying data augmentation strategies  
- Fine-tuning the model  
- Saving checkpoints (`best_model.pth` and `model.pth`)  
- Deploying a **Streamlit** web app for interactive predictions  

You can open **main.ipynb** in **Jupyter Notebook** or **VS Code** and run cells in sequence (or simply click “Run All”).

---

<a name="structure"></a>
## 2. Repository Structure
```
repository/
├── data/
│   ├── stanford_dogs/Images/     # Raw dataset (class folders with JPGs)
│   └── Processed/                 # Created by main.ipynb (train/val/test splits)
├── main.ipynb                     # Full training + evaluation pipeline
├── prediction.py                  # CLI single-image prediction
├── web_ui.py                      # Streamlit web demo (top-5 with confidences)
├── best_model.pth                 # Best validation accuracy weights
├── model.pth                      # Final model after training
├── requirements.yml               # Conda environment specification
└── README.md
```

---

<a name="installation"></a>
## 3. Installation

> **Note:** Developed in a Windows environment. Works on Linux/Mac with Conda as well.

```bash
git clone <this-repo>
cd <this-repo>
conda env create -f requirements.yml
conda activate dogbreed
```

---

<a name="usage"></a>
## 4. Usage Options

### **Option 1 – Use the pre-trained model with the web demo**
If you just want to try the classifier:
```bash
streamlit run web_ui.py
```
- Opens a browser interface where you can upload an image and see predictions.
- Uses the included `best_model.pth` (already in the repository).

---

### **Option 2 – Reproduce full training & evaluation**
1. **Download** the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) and place it at:
   ```
   data/stanford_dogs/Images/
   ```
2. **Open** `main.ipynb` in Jupyter Notebook or VS Code.
3. **Run cells** in order (or click "Run All").
4. The notebook will:
   - Preprocess & split the dataset
   - Train the ResNet-18 model
   - Save:
     - `best_model.pth` → highest validation accuracy  
     - `model.pth` → final epoch’s model  
5. After training, you can run the web app (Option 1) with your new `.pth` files.

---

<a name="cli"></a>
## 5. Example CLI Prediction

You can also predict from the command line:
```bash
python prediction.py path/to/image.jpg
```
**Example output:**
```
Predicted dog breed: golden_retriever
```

---

## Notes
- `best_model.pth` is used by default for web and CLI predictions; `model.pth` is just the final model after training completion.
- Environment is fully specified in `requirements.yml` for reproducibility.

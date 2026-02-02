# DCFICSH

**DCFICSH: A Dual-Channel Fusion Model Combining Multi-Modal Data for Identifying Cell-Specific Silencers and Their Strength in the Human Genome**

This repository contains the datasets and source code for the paper "DCFICSH", a deep learning model designed to identify cell-specific silencers and predict their strength (Strong/Weak) in the human genome, specifically focusing on HepG2 and K562 cell lines.

## 📌 Introduction

DCFICSH is a dual-channel fusion model that integrates multi-modal data to effectively predict silencers. The project is organized by cell line, with separate directories for identifying silencers and classifying their strength.

## 📂 Directory Structure

The repository is organized as follows:

```text
DCFICSH/
├── HepG2/                  # Data and Code for HepG2 cell line
│   ├── dataset/            # All datasets used for HepG2
│   └── Code/               # Source code for HepG2 models
│       ├── base_seq.py         # Data preprocessing script
│       ├── model_DCFICSH.py    # Model architecture definition
│       └── predict.py          # Training and prediction script
│
├── K562/                   # Data and Code for K562 cell line
│   ├── dataset/            # All datasets used for K562
│   └── Code/               # Source code for K562 models
│       ├── (Same structure as HepG2)
│
└── K562__SS__WS/           # Special module for identifying Silencer Strength (Strong/Weak)
    ├── dataset/
    └── Code/               # Code path and usage are identical to main modules
```


## 🛠️ Dependencies

The code is implemented in Python. Based on the environment used, the key dependencies are listed below. You can install them using `pip`.

### Core Requirements
* `tensorflow==2.4.1`
* `Keras==2.4.1`
* `numpy==1.19.5`
* `pandas==1.2.3`
* `scikit-learn==1.3.2`
* `scipy==1.10.1`
* `matplotlib==3.3.4`
* `absl-py==0.15.0`

### Other Utilities
* `h5py==2.10.0`
* `tqdm==4.67.0`
* `joblib==1.4.2`

## 🚀 Usage

The workflow is consistent across `HepG2`, `K562`, and `K562__SS__WS` directories. Navigate to the `Code` directory of the specific cell line or task you wish to run.

### 1. Data Preprocessing
Use `base_seq.py` to preprocess the raw data and prepare it for the model.

```bash
python base_seq.py
```

Input: Raw sequence data from the ../dataset/ folder.

Output: Preprocessed sequences/features ready for model input.

### 2. Model Architecture
The file model_DCFICSH.py contains the definition of the Dual-Channel Fusion Model. You usually do not need to run this file directly, as it is imported by the training script.

### 3. Training and Prediction
Use predict.py to train the model and perform predictions.

```Bash
python predict.py
```
This script loads the preprocessed data, builds the model defined in model_DCFICSH.py, trains it, and outputs the prediction results.


## 📖 Citation

If you use this code or dataset in your research, please cite the following paper:

**BibTeX:**

```bibtex
@inproceedings{yuan2025dcficsh,
  title={DCFICSH: A Dual-Channel Fusion Model Combining Multi-Modal Data for Identifying Cell-Specific Silencers and Their Strength in the Human Genome},
  author={Yuan, Jingdong and others},
  booktitle={International Conference on Intelligent Computing},
  year={2025},
  publisher={Springer Nature Singapore},
  address={Singapore}
}

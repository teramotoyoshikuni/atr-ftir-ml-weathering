# atr-ftir-ml-weathering
ATR-FTIR and machine learning models for diagnosing latent deterioration in exterior wood coatings

# FTIR Spectroscopy × Machine Learning

This repository provides Python scripts for regression and classification of FTIR spectral data
using classical chemometric and machine-learning methods.

All scripts are based on publicly available codes developed by **Kaneko Lab** (https://github.com/hkaneko1985), with minimal
modifications (file paths, variable names, reproducibility settings).

The codes are intended for research and educational purposes.

---

## Repository Structure

```text
your-repo/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ sample_functions.py
├─ scripts/
│  ├─ 01_pls_train_test.py
│  ├─ 02_pls_loocv.py
│  ├─ 03_gawnspls.py
│  ├─ 04_rf_regression.py
│  ├─ 05_gbr_regression.py
│  ├─ 06_knn_classification.py
│  ├─ 07_dt_classification.py
│  ├─ 08_rf_classification.py
│  └─ 09_svm_classification.py
├─ data/
│  ├─ for_regression.csv
│  └─ for_classification.csv
└─ outputs/
   └─ (generated automatically; not tracked by git)
```

# Input Data Format
## Regression (`data/for_regression.csv`)
- **1st column**: target variable (y)
- **2nd and later columns**: explanatory variables (X; e.g., FTIR absorbance values)

## Classification (`data/for_classification.csv`)
- **1st column**: class label
- **2nd and later columns**: explanatory variables (X)
Column names must be consistent across scripts.

# How to Run
## 1. Install dependencies
`pip install -r requirements.txt`

## 2. Run scripts
```
python scripts/04_rf_regression.py
python scripts/09_svm_classification.py
```

# Outputs
Each script generates output files such as:
- Prediction results (CSV)
- Error metrics (R², RMSE, MAE, accuracy)
- Feature importances (RF)
- Confusion matrices (classification)
All outputs are saved in the `outputs/` directory.
Existing files may be overwritten.

# Reproducibility
- Random seeds are fixed where applicable.
- Explicit train/test split is used for:
   - PLS (train/test)
   - RF regression
   - GBR
   - kNN / DT / RF / SVM classification
- LOOCV is used only in the dedicated PLS-LOOCV script.

# About `sample_functions.py`
This repository includes `sample_functions.py` (MIT License), originally developed and distributed by **Kaneko Lab**.

Source repository:
https://github.com/hkaneko1985/python_data_analysis_ohmsha

The file is used **without modification**.

# License
This repository is distributed under the **MIT License**.

Third-party code:
- `sample_functions.py` © Kaneko Lab (MIT License)

# Notes
These scripts prioritize clarity and reproducibility over heavy optimization.
Minor differences from the original Kaneko Lab scripts include:
- Unified input file names
- Explicit directory structure
- Fixed random seeds







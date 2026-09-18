# WoundVision-ML
Automated Animal Model Wound Localization and Multi-Output Dimension Regressor

Predicts wound dimensions (length/width) over healing days from wound images using Support Vector Regression (SVR) and Random Forest (RF) models.

## Project Structure

```
├── data/
│   ├── Training/          # Training images + myData.csv (ground-truth dimensions)
│   └── Test/              # Test images + myData.csv
├── models/
│   ├── svr_model.sav      # Trained SVR model (pickle)
│   └── rf_model.sav       # Trained Random Forest model (pickle)
├── results/               # Prediction outputs for train and test sets
├── src/
│   ├── trainer.ipynb      # Trains SVR + RF models
│   ├── tester.ipynb       # Loads saved models and evaluates on test set
│   ├── svr_tuning.ipynb   # Hyperparameter tuning for SVR
│   └── rf_tuning.ipynb    # Hyperparameter tuning for Random Forest
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. **Train**: run `src/trainer.ipynb` — trains SVR and RF regressors on `data/Training`, saves models to `models/`.
2. **Test**: run `src/tester.ipynb` — loads the saved models and produces predictions in `results/`.
3. **Tune**: optional — `src/svr_tuning.ipynb` and `src/rf_tuning.ipynb` for hyperparameter search.

## Approach

- Images are resized to 32×32, converted to grayscale, and contrast-enhanced with CLAHE (skimage `equalize_adapthist`).
- The flattened 32×32 pixel values (1024 features) feed the regressors directly.
- Both models predict 4 outputs: wound center (x, y) and size (width, height).
- SVR: `MultiOutputRegressor` with a linear-kernel SVR (C = 2.1544); RF: `RandomForestRegressor` (100 estimators).

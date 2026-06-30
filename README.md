# AI-ML_Emrul

## Credit Card Fraud Detection

Notebook-only ML project for the ULB credit-card fraud dataset (`creditcard.csv`): 284,807 transactions, 492 fraud cases, and severe class imbalance. The notebook treats **AUPRC**, not accuracy, as the primary metric.

## What It Does

`fraud_detection.ipynb` runs the full workflow:

1. Loads `creditcard.csv` and drops exact duplicates before splitting.
2. Performs imbalance-first EDA.
3. Keeps PCA features `V1..V28`, adds cyclic time features (`hour_sin`, `hour_cos`), and keeps raw `Amount`.
4. Trains XGBoost and LightGBM with GPU auto-detection and CPU fallback.
5. Evaluates with 5-fold stratified CV, PR-curve F2 thresholding, and a temporal holdout.

## Dataset

Provided by BR23 email, so not included in the repo.

## Requirements

- Python 3.12+
- Jupyter or Google Colab
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `lightgbm`

The notebook installs `xgboost` and `lightgbm` if needed.

## Run

### Google Colab

1. Open `fraud_detection.ipynb`.
2. Optional: set Runtime > Change runtime type > GPU.
3. Upload `creditcard.csv` when prompted.
4. Run all cells.

### Local Jupyter

```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm jupyter
jupyter notebook fraud_detection.ipynb
```

## Results From The Executed Notebook

After deduplication: 283,726 rows, 473 fraud cases, ~600:1 imbalance.

| Model | 5-fold CV AUPRC | ROC-AUC | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.8467 +/- 0.016 | 0.9833 | 0.899 | 0.825 |
| LightGBM | 0.8447 +/- 0.015 | 0.9811 | 0.891 | 0.827 |

Temporal holdout AUPRC:

- XGBoost: `0.7853`
- LightGBM: `0.7809`

## Repository Structure

```text
.
|-- README.md
`-- fraud_detection.ipynb
```

## Notes

This is not a packaged library, API, or deployment service. There are no saved model artifacts, tests, lockfiles, or environment files; rerun the notebook to reproduce the analysis.

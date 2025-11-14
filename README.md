# Auto Boost

Auto Boost is a small command-line helper that mirrors the original `Auto_boost.ipynb`
Kaggle workflow. It handles missing values, categorical encoding, cross-validated
training, and submission generation for gradient-boosting models (LightGBM,
XGBoost, or CatBoost) without needing to run a notebook.

## Installation

Install directly from the repo root (or future wheel/PyPI artifact):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .  # add '.[lightgbm]' or other extras if you need specific boosters
```

Base dependencies are `pandas`, `numpy`, and `scikit-learn`. Install at least one
booster extra (`lightgbm`, `xgboost`, `catboost`) depending on what you plan to run.

## Quickstart

```bash
auto-boost \
  --train train.csv \
  --test test.csv \
  --target Transported \
  --model-type classification \
  --metric accuracy \
  --booster lgbm \
  --folds 10 \
  --random-state 10 \
  --id-col PassengerId \
  --prediction-col Transported \
  --output submission.csv
```

Key flags:

- `--train` / `--test`: paths to CSV files.
- `--target`: column in train.csv you want to predict.
- `--model-type`: `classification` or `regression`.
- `--booster`: choose between `lgbm`, `xgb`, `catboost`.
- `--metric`: auto-detected if omitted (`accuracy` for classification, `rmse` for regression).
- `--output`: optional CSV to save predictions (includes ID column when `--id-col` is supplied).

Run `auto-boost --help` for the full reference. The legacy `python auto_boost.py`
shim is still provided for local scripts and backwards compatibility.

## About the Notebook

The original `Auto_boost.ipynb` is retained for reference, but the script fixes
several issues (missing class instantiation, incorrect feature-importance labels,
buggy categorical handling) and is the recommended entry point for automation.

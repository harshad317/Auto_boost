# Auto Boost

Auto Boost is a small command-line helper that mirrors the original `Auto_boost.ipynb`
Kaggle workflow. It handles missing values, categorical encoding, cross-validated
training, and submission generation for gradient-boosting models (LightGBM,
XGBoost, or CatBoost) without needing to run a notebook.

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- At least one of: `lightgbm`, `xgboost`, `catboost` (install only what you need)

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn lightgbm xgboost catboost
```

## Quickstart

```bash
python auto_boost.py \
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

Run `python auto_boost.py --help` for the full reference.

## About the Notebook

The original `Auto_boost.ipynb` is retained for reference, but the script fixes
several issues (missing class instantiation, incorrect feature-importance labels,
buggy categorical handling) and is the recommended entry point for automation.

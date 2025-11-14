# Auto Boost

Auto Boost is a small command-line helper that mirrors the original `Auto_boost.ipynb`
Kaggle workflow. It handles missing values, categorical encoding, cross-validated
training, and submission generation for gradient-boosting models (LightGBM,
XGBoost, or CatBoost) without needing to run a notebook.

## Installation

### From PyPI (recommended)

```bash
python -m pip install --upgrade auto_boost
# or include extras:
python -m pip install "auto_boost[lightgbm]"
```

### From source

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[lightgbm,xgboost]"  # select any boosters you need
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

Run `auto-boost --help` (or `auto_boost --help`) for the full reference. The legacy
`python auto_boost.py` shim has been removed in favor of the installable entrypoints.

## Development & Packaging

For local development install in editable mode:

```bash
python -m pip install --upgrade pip build twine
python -m pip install -e ".[lightgbm]"
```

To produce distributable artifacts (wheel + sdist):

```bash
python -m pip install build
python -m build
ls dist/
```

The files under `dist/` can be uploaded with `twine upload dist/*` when
publishing to PyPI. Generated folders such as `dist/`, `*.egg-info`, and
`__pycache__` are ignored via `.gitignore`.

### Releasing to TestPyPI / PyPI

Following the official [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-your-project-to-pypi) guide:

```bash
# Build fresh artifacts
rm -rf dist/
python -m build

# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Verify install from TestPyPI (optional)
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple auto_boost

# When satisfied, push to PyPI for public install via:
# python -m pip install auto_boost
python -m twine upload dist/*
```

Bump `auto_boost.__version__` before every upload to avoid version conflicts.

## About the Notebook

The original `Auto_boost.ipynb` is retained for reference, but the script fixes
several issues (missing class instantiation, incorrect feature-importance labels,
buggy categorical handling) and is the recommended entry point for automation.

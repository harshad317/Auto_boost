#!/usr/bin/env python3
"""Command-line interface for the Auto Boost notebook workflow."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None


CLASSIFICATION_METRICS = ["accuracy", "f1", "precision", "recall", "log_loss", "roc_auc"]
REGRESSION_METRICS = ["mse", "rmse", "r2", "mae"]
PROBA_METRICS = {"log_loss", "roc_auc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto Boost pipeline runner.")
    parser.add_argument("--train", required=True, help="Path to the training CSV file.")
    parser.add_argument("--test", required=True, help="Path to the test CSV file.")
    parser.add_argument("--target", required=True, help="Target column present in the training data.")
    parser.add_argument(
        "--model-type",
        choices=["classification", "regression"],
        default="classification",
        help="Type of problem to solve.",
    )
    parser.add_argument(
        "--metric",
        help="Metric to optimize. Defaults to accuracy for classification and rmse for regression.",
    )
    parser.add_argument(
        "--booster",
        choices=["lgbm", "xgb", "catboost"],
        default="lgbm",
        help="Gradient boosting implementation to use.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n-estimators", type=int, default=1500, help="Number of boosting rounds.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for the booster.")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds for libraries that support it (set to 0 to disable).",
    )
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity passed to the underlying estimator.")
    parser.add_argument(
        "--id-col",
        help="Optional identifier column that should not be used for training but included in the output CSV.",
    )
    parser.add_argument(
        "--prediction-col",
        default="prediction",
        help="Name of the prediction column in the saved submission file.",
    )
    parser.add_argument("--output", help="Optional path to save predictions as CSV.")
    return parser.parse_args()


def require_dependency(obj, package_name: str) -> None:
    if obj is None:
        raise ImportError(f"{package_name} is required for the selected booster but is not installed.")


def build_model(args: argparse.Namespace):
    booster = args.booster.lower()
    model_type = args.model_type
    n_estimators = args.n_estimators
    learning_rate = args.learning_rate
    random_state = args.random_state

    if model_type == "classification":
        if booster == "lgbm":
            require_dependency(LGBMClassifier, "lightgbm")
            return LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        if booster == "xgb":
            require_dependency(XGBClassifier, "xgboost")
            return XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                use_label_encoder=False,
            )
        if booster == "catboost":
            require_dependency(CatBoostClassifier, "catboost")
            return CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=6,
                random_seed=random_state,
                verbose=False,
            )
    else:
        if booster == "lgbm":
            require_dependency(LGBMRegressor, "lightgbm")
            return LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        if booster == "xgb":
            require_dependency(XGBRegressor, "xgboost")
            return XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=random_state,
            )
        if booster == "catboost":
            require_dependency(CatBoostRegressor, "catboost")
            return CatBoostRegressor(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=6,
                random_seed=random_state,
                verbose=False,
            )
    raise ValueError(f"Unsupported combination: booster={booster}, model_type={model_type}")


class AutoBoost:
    def __init__(
        self,
        estimator,
        model_type: str,
        metric: str,
        folds: int,
        random_state: int,
        early_stopping_rounds: int,
        verbose: int,
    ):
        self.estimator = estimator
        self.model_type = model_type
        self.metric = metric
        self.folds = folds
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.uses_proba = metric in PROBA_METRICS

    def preprocess(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        combined = self.fill_missing_values(combined)
        combined = self.categorical_columns(combined)
        train_processed = combined.iloc[: len(train_df)].reset_index(drop=True)
        test_processed = combined.iloc[len(train_df) :].reset_index(drop=True)
        return train_processed, test_processed

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing == 0:
                continue
            unique_values = df[col].nunique(dropna=True)
            if df[col].dtype == "O":
                if unique_values > 25:
                    df = df.drop(columns=[col])
                    continue
                mode_series = df[col].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else ""
                df[col] = df[col].fillna(fill_value)
            else:
                if unique_values < 25:
                    mode_series = df[col].mode(dropna=True)
                    fill_value = mode_series.iloc[0] if not mode_series.empty else 0
                    df[col] = df[col].fillna(fill_value)
                else:
                    median_value = df[col].median()
                    if pd.isna(median_value):
                        median_value = 0.0
                    df[col] = df[col].fillna(median_value)
        return df

    def categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        categorical_cols = [col for col in df.columns if df[col].dtype == "O"]
        dummy_frames: List[pd.DataFrame] = []
        drop_cols: List[str] = []
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if unique_values <= 1:
                drop_cols.append(col)
                continue
            if unique_values <= 15:
                dummy_frames.append(pd.get_dummies(df[col], prefix=col, drop_first=True))
                drop_cols.append(col)
            else:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
        if drop_cols:
            df = df.drop(columns=drop_cols)
        if dummy_frames:
            df = pd.concat([df] + dummy_frames, axis=1)
        return df

    def train(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> Dict[str, object]:
        splitter = (
            StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
            if self.model_type == "classification"
            else KFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
        )

        fold_scores: List[float] = []
        test_predictions: List[np.ndarray] = []
        feature_importances: List[np.ndarray] = []
        X = X.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y = y.reset_index(drop=True)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = clone(self.estimator)
            self._fit(model, X_train, y_train, X_val, y_val)

            val_pred = self._predict_for_metric(model, X_val)
            test_pred = self._predict_for_metric(model, X_test)
            test_predictions.append(test_pred)
            score = self._compute_score(y_val, val_pred)
            fold_scores.append(score)

            if hasattr(model, "feature_importances_"):
                feature_importances.append(model.feature_importances_)

            print(f"Fold {fold + 1}/{self.folds} - {self.metric}: {score:.5f}")

        aggregated_predictions = self._aggregate_predictions(test_predictions)
        feature_importance_df = None
        if feature_importances:
            mean_importance = np.mean(feature_importances, axis=0)
            feature_importance_df = (
                pd.DataFrame({"feature": X.columns, "importance": mean_importance})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        return {
            "fold_scores": fold_scores,
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "predictions": aggregated_predictions,
            "feature_importance": feature_importance_df,
        }

    def _fit(self, model, X_train, y_train, X_val, y_val) -> None:
        fit_kwargs = {}
        fit_method = model.fit
        if self._fit_supports_arg(fit_method, "eval_set"):
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if self.early_stopping_rounds and self._fit_supports_arg(fit_method, "early_stopping_rounds"):
            fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
        if self._fit_supports_arg(fit_method, "verbose"):
            fit_kwargs["verbose"] = self.verbose
        fit_method(X_train, y_train, **fit_kwargs)

    @staticmethod
    def _fit_supports_arg(method, argument: str) -> bool:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):  # C-extensions
            return True
        params = signature.parameters
        if argument in params:
            return True
        return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())

    def _predict_for_metric(self, model, X: pd.DataFrame) -> np.ndarray:
        if self.uses_proba:
            if not hasattr(model, "predict_proba"):
                raise AttributeError(f"{model.__class__.__name__} does not implement predict_proba, required for {self.metric}.")
            return model.predict_proba(X)
        return model.predict(X)

    def _compute_score(self, y_true: pd.Series, predictions: np.ndarray) -> float:
        if self.model_type == "classification":
            if self.metric == "accuracy":
                return accuracy_score(y_true, self._probabilities_to_labels(predictions) if self.uses_proba else predictions)
            if self.metric == "f1":
                preds = self._probabilities_to_labels(predictions) if self.uses_proba else predictions
                return f1_score(y_true, preds)
            if self.metric == "precision":
                preds = self._probabilities_to_labels(predictions) if self.uses_proba else predictions
                return precision_score(y_true, preds, zero_division=0)
            if self.metric == "recall":
                preds = self._probabilities_to_labels(predictions) if self.uses_proba else predictions
                return recall_score(y_true, preds, zero_division=0)
            if self.metric == "log_loss":
                return log_loss(y_true, predictions)
            if self.metric == "roc_auc":
                probs = predictions
                if probs.ndim == 2 and probs.shape[1] > 1:
                    probs = probs[:, 1]
                return roc_auc_score(y_true, probs)
        else:
            if self.metric == "mse":
                return mean_squared_error(y_true, predictions)
            if self.metric == "rmse":
                return mean_squared_error(y_true, predictions, squared=False)
            if self.metric == "r2":
                return r2_score(y_true, predictions)
            if self.metric == "mae":
                return mean_absolute_error(y_true, predictions)
        raise ValueError(f"Unsupported metric {self.metric}")

    def _aggregate_predictions(self, predictions: List[np.ndarray]) -> np.ndarray:
        if not predictions:
            raise ValueError("No predictions to aggregate.")
        if self.model_type == "regression":
            stacked = np.stack(predictions, axis=0)
            return stacked.mean(axis=0)
        if self.uses_proba:
            stacked = np.stack(predictions, axis=0)
            averaged = stacked.mean(axis=0)
            return self._probabilities_to_labels(averaged)
        stacked = np.stack(predictions, axis=0)
        return self._majority_vote(stacked)

    @staticmethod
    def _majority_vote(pred_matrix: np.ndarray) -> np.ndarray:
        votes = []
        for column in pred_matrix.swapaxes(0, 1):
            values, counts = np.unique(column, return_counts=True)
            votes.append(values[np.argmax(counts)])
        return np.array(votes)

    @staticmethod
    def _probabilities_to_labels(probs: np.ndarray) -> np.ndarray:
        if probs.ndim == 1 or probs.shape[1] == 1:
            return (probs.ravel() >= 0.5).astype(int)
        return np.argmax(probs, axis=1)


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file {train_path} does not exist.")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file {test_path} does not exist.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if args.target not in train_df.columns:
        raise ValueError(f"Target column '{args.target}' not found in training data.")

    metric = args.metric
    if metric is None:
        metric = "accuracy" if args.model_type == "classification" else "rmse"
    if args.model_type == "classification" and metric not in CLASSIFICATION_METRICS:
        raise ValueError(f"Metric '{metric}' is not supported for classification.")
    if args.model_type == "regression" and metric not in REGRESSION_METRICS:
        raise ValueError(f"Metric '{metric}' is not supported for regression.")

    estimator = build_model(args)
    model = AutoBoost(
        estimator=estimator,
        model_type=args.model_type,
        metric=metric,
        folds=args.folds,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
    )

    feature_columns = train_df.drop(columns=[args.target])
    test_features = test_df.copy()

    if args.id_col and args.id_col in feature_columns.columns:
        feature_columns = feature_columns.drop(columns=[args.id_col])
    if args.id_col and args.id_col in test_features.columns:
        test_features_no_id = test_features.drop(columns=[args.id_col])
    else:
        test_features_no_id = test_features

    target_series = train_df[args.target]
    target_is_bool = is_bool_dtype(target_series)
    y = target_series.astype(int) if target_is_bool else target_series

    processed_train, processed_test = model.preprocess(feature_columns, test_features_no_id)
    result = model.train(processed_train, y, processed_test)

    mean_score = result["mean_score"]
    std_score = result["std_score"]
    print(f"{args.model_type.capitalize()} {metric}: {mean_score:.5f} ± {std_score:.5f}")

    predictions = result["predictions"]
    if args.model_type == "classification" and target_is_bool:
        predictions = predictions.astype(bool)

    if args.output:
        output_df = pd.DataFrame({args.prediction_col: predictions})
        if args.id_col:
            if args.id_col not in test_df.columns:
                raise ValueError(f"id column '{args.id_col}' missing from test data.")
            output_df.insert(0, args.id_col, test_df[args.id_col].values)
        output_path = Path(args.output)
        output_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    feature_importance = result.get("feature_importance")
    if feature_importance is not None:
        top_features = feature_importance.head(15)
        print("Top features:")
        print(top_features.to_string(index=False))


if __name__ == "__main__":
    main()

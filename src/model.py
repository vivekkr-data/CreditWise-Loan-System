"""Training and evaluation utilities for the CreditWise loan model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "loan_approval_data.csv"
TARGET = "Loan_Approved"

# These columns remain available for data-quality and fairness analysis but are
# deliberately excluded from model decisions. Age was also removed because it
# is a sensitive lending attribute and did not improve holdout performance.
EXCLUDED_FEATURES = ["Applicant_ID", "Gender", "Marital_Status", "Age"]

NUMERIC_FEATURES = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]

CATEGORICAL_FEATURES = [
    "Employment_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Employer_Category",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_development_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the single holdout split shared by evaluation and benchmarking."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset, validate its schema, and return model features/target."""
    data = pd.read_csv(path)
    required = set(MODEL_FEATURES + EXCLUDED_FEATURES + [TARGET])
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data.dropna(subset=[TARGET]).copy()
    unknown_targets = set(data[TARGET].dropna().unique()).difference({"Yes", "No"})
    if unknown_targets:
        raise ValueError(f"Unexpected target labels: {sorted(unknown_targets)}")

    features = data[MODEL_FEATURES].copy()
    target = data[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    return features, target


def build_pipeline() -> Pipeline:
    """Build a leakage-safe preprocessing and classification pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    classifier = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def numeric_feature_ranges(
    features: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """Return observed numeric ranges for inference guardrails."""
    return {
        column: (
            float(features[column].min(skipna=True)),
            float(features[column].max(skipna=True)),
        )
        for column in NUMERIC_FEATURES
    }


def validate_numeric_ranges(
    application: pd.DataFrame,
    ranges: dict[str, tuple[float, float]],
) -> list[str]:
    """Describe numeric values that fall outside the model's observed data."""
    errors = []
    for column, (minimum, maximum) in ranges.items():
        if column not in application.columns:
            errors.append(f"Missing required feature: {column}")
            continue

        values = pd.to_numeric(application[column], errors="coerce")
        if values.isna().any():
            errors.append(f"{column} must contain a numeric value")
            continue

        if ((values < minimum) | (values > maximum)).any():
            errors.append(f"{column} must be between {minimum:g} and {maximum:g}")
    return errors


def evaluate_model(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Evaluate on one untouched holdout and CV within development data only."""
    train_x, test_x, train_y, test_y = split_development_holdout(
        features, target, test_size=test_size, random_state=random_state
    )
    model = build_pipeline()
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    probability = model.predict_proba(test_x)[:, 1]

    holdout = {
        "test_rows": len(test_y),
        "accuracy": float(accuracy_score(test_y, prediction)),
        "precision": float(precision_score(test_y, prediction, zero_division=0)),
        "recall": float(recall_score(test_y, prediction, zero_division=0)),
        "f1": float(f1_score(test_y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_y, probability)),
        "confusion_matrix": confusion_matrix(test_y, prediction).tolist(),
    }

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(
        build_pipeline(), train_x, train_y, cv=folds, scoring=scoring, n_jobs=1
    )
    cross_validation = {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"])),
        }
        for metric in scoring
    }
    protocol = {
        "labelled_rows": len(target),
        "development_rows": len(train_y),
        "holdout_rows": len(test_y),
        "cross_validation_folds": folds.n_splits,
        "random_state": random_state,
    }
    return {
        "evaluation_protocol": protocol,
        "holdout": holdout,
        "development_cross_validation": cross_validation,
    }


def train_production_model(features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """Fit the final pipeline on every labelled row for application inference."""
    model = build_pipeline()
    model.fit(features, target)
    return model


def feature_importance(model: Pipeline, limit: int = 10) -> pd.DataFrame:
    """Return the most influential transformed features in the fitted model."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = model.named_steps["classifier"].feature_importances_
    importance = pd.DataFrame({"feature": names, "importance": values})
    return importance.sort_values("importance", ascending=False).head(limit)

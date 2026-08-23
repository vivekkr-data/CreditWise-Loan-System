"""Compare candidate classifiers using the same preprocessing and CV folds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.model import build_pipeline, load_data


def main() -> None:
    features, target = load_data()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2_000),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    rows = []
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for name, classifier in candidates.items():
        pipeline = build_pipeline().set_params(classifier=classifier)
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=folds,
            scoring=scoring,
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                **{
                    metric: scores[f"test_{metric}"].mean()
                    for metric in scoring
                },
            }
        )

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

"""Train CreditWise, print evaluation metrics, and save a reusable model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.model import evaluate_model, load_data, train_production_model


def main() -> None:
    features, target = load_data()
    metrics = evaluate_model(features, target)
    model = train_production_model(features, target)

    artifact_dir = Path("artifacts")
    report_dir = Path("reports")
    artifact_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    joblib.dump(model, artifact_dir / "creditwise_pipeline.joblib")
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    holdout = metrics["holdout"]
    print(f"Holdout accuracy : {holdout['accuracy']:.2%}")
    print(f"Holdout precision: {holdout['precision']:.2%}")
    print(f"Holdout recall   : {holdout['recall']:.2%}")
    print(f"Holdout F1 score : {holdout['f1']:.2%}")
    print(f"Holdout ROC-AUC  : {holdout['roc_auc']:.2%}")
    print("Saved model to artifacts/creditwise_pipeline.joblib")


if __name__ == "__main__":
    main()

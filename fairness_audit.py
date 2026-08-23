"""Generate a small holdout subgroup audit for responsible-model review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from src.model import (
    DEFAULT_DATA_PATH,
    TARGET,
    build_pipeline,
    load_data,
    split_development_holdout,
)


def safe_rate(numerator: int, denominator: int) -> float:
    """Return a rate without failing when a subgroup has no eligible rows."""
    return float(numerator / denominator) if denominator else float("nan")


def main() -> None:
    features, target = load_data()
    train_x, test_x, train_y, test_y = split_development_holdout(features, target)

    model = build_pipeline().fit(train_x, train_y)
    predictions = pd.Series(
        model.predict(test_x), index=test_x.index, name="prediction"
    )

    labelled = pd.read_csv(DEFAULT_DATA_PATH).dropna(subset=[TARGET])
    audit = pd.DataFrame(
        {
            "Gender": labelled.loc[test_x.index, "Gender"],
            "actual": test_y,
            "prediction": predictions,
        }
    ).dropna(subset=["Gender"])

    rows = []
    for group, subset in audit.groupby("Gender", sort=True):
        rejected = subset["actual"] == 0
        approved = subset["actual"] == 1
        false_approvals = int((rejected & (subset["prediction"] == 1)).sum())
        false_rejections = int((approved & (subset["prediction"] == 0)).sum())
        rows.append(
            {
                "group": group,
                "rows": len(subset),
                "actual_approval_rate": subset["actual"].mean(),
                "predicted_approval_rate": subset["prediction"].mean(),
                "accuracy": accuracy_score(subset["actual"], subset["prediction"]),
                "false_positive_rate": safe_rate(false_approvals, int(rejected.sum())),
                "false_negative_rate": safe_rate(false_rejections, int(approved.sum())),
            }
        )

    report = pd.DataFrame(rows)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report.to_csv(report_dir / "fairness_audit.csv", index=False)
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "\nAudit attributes are not model inputs. Small subgroup samples are "
        "diagnostic only and do not establish fairness."
    )


if __name__ == "__main__":
    main()

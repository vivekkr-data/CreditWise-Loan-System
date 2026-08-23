import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import (
    EXCLUDED_FEATURES,
    MODEL_FEATURES,
    build_pipeline,
    evaluate_model,
    load_data,
    numeric_feature_ranges,
    validate_numeric_ranges,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "loan_approval_data.csv"


class ModelTests(unittest.TestCase):
    def test_dataset_loads_with_expected_schema(self):
        features, target = load_data(DATA_PATH)

        self.assertEqual(list(features.columns), MODEL_FEATURES)
        self.assertEqual(len(features), 950)
        self.assertFalse(target.isna().any())
        self.assertEqual(set(target.unique()), {0, 1})
        self.assertTrue(set(EXCLUDED_FEATURES).isdisjoint(MODEL_FEATURES))
        self.assertNotIn("Age", MODEL_FEATURES)

    def test_pipeline_handles_missing_values_and_predicts(self):
        features, target = load_data(DATA_PATH)
        train_x = features.iloc[:800]
        train_y = target.iloc[:800]
        sample = features.iloc[[800]].copy()
        sample.loc[:, "Credit_Score"] = np.nan
        sample.loc[:, "Employment_Status"] = None

        model = build_pipeline().fit(train_x, train_y)
        prediction = model.predict(sample)
        probability = model.predict_proba(sample)

        self.assertEqual(prediction.shape, (1,))
        self.assertEqual(probability.shape, (1, 2))
        self.assertGreaterEqual(probability[0, 1], 0.0)
        self.assertLessEqual(probability[0, 1], 1.0)

    def test_numeric_guardrails_reject_out_of_distribution_values(self):
        features, _ = load_data(DATA_PATH)
        ranges = numeric_feature_ranges(features)
        valid = features.iloc[[0]].copy()
        invalid = valid.copy()
        invalid.loc[:, "Applicant_Income"] = ranges["Applicant_Income"][1] + 1

        self.assertEqual(validate_numeric_ranges(valid, ranges), [])
        self.assertIn(
            "Applicant_Income must be between 2009 and 19988",
            validate_numeric_ranges(invalid, ranges),
        )

        missing = pd.DataFrame({"Applicant_Income": [10_000]})
        self.assertTrue(validate_numeric_ranges(missing, ranges))

    def test_evaluation_uses_development_only_cv_and_verified_holdout(self):
        features, target = load_data(DATA_PATH)
        metrics = evaluate_model(features, target)

        self.assertEqual(metrics["evaluation_protocol"]["development_rows"], 760)
        self.assertEqual(metrics["evaluation_protocol"]["holdout_rows"], 190)
        self.assertEqual(metrics["holdout"]["confusion_matrix"], [[124, 6], [1, 59]])
        self.assertAlmostEqual(metrics["holdout"]["accuracy"], 183 / 190)
        self.assertGreater(metrics["development_cross_validation"]["f1"]["mean"], 0.90)


if __name__ == "__main__":
    unittest.main()

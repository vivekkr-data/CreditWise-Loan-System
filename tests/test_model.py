from pathlib import Path
import unittest

import numpy as np

from src.model import MODEL_FEATURES, build_pipeline, load_data


DATA_PATH = Path(__file__).resolve().parents[1] / "loan_approval_data.csv"


class ModelTests(unittest.TestCase):
    def test_dataset_loads_with_expected_schema(self):
        features, target = load_data(DATA_PATH)

        self.assertEqual(list(features.columns), MODEL_FEATURES)
        self.assertEqual(len(features), 950)
        self.assertFalse(target.isna().any())
        self.assertEqual(set(target.unique()), {0, 1})

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


if __name__ == "__main__":
    unittest.main()

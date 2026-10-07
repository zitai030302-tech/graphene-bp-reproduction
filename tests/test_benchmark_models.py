import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_models import models, run, splits, synthetic


class BenchmarkTests(unittest.TestCase):
    def test_group_protocols_are_disjoint(self):
        frame = synthetic(subjects=4, samples=12)
        covered = []
        for protocol in ("group_holdout", "loso"):
            for _, train, test in splits(frame, protocol):
                self.assertFalse(set(frame.iloc[train].subject_id) & set(frame.iloc[test].subject_id))
                self.assertFalse(set(train) & set(test))
                if protocol == "loso":
                    covered.extend(test)
        self.assertEqual(sorted(covered), list(range(len(frame))))

    def test_imputation_uses_only_training_rows(self):
        x_train = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 4.0]])
        for model in models().values():
            model.fit(x_train, [100, 110, 120])
            before = model[0].statistics_.copy()
            model.predict(np.array([[9999, np.nan], [np.nan, 9999]]))
            np.testing.assert_allclose(model[0].statistics_, [3, 3])
            np.testing.assert_array_equal(before, model[0].statistics_)

    def test_label_and_identifier_features_rejected(self):
        with tempfile.TemporaryDirectory() as output:
            for features in (["SBP"], ["subject_id"], ["f0", "f0"]):
                with self.assertRaises(ValueError):
                    run(synthetic(), features, "SBP", output)

    def test_report_recomputes_from_predictions(self):
        with tempfile.TemporaryDirectory() as output:
            summary = run(synthetic(subjects=3, samples=15), ["f0", "f1"], "SBP", output)
            predictions = pd.read_csv(Path(output) / "predictions.csv")
            for row in summary.itertuples():
                subset = predictions[(predictions.protocol == row.protocol) & (predictions.model == row.model)]
                error = subset.predicted - subset.true
                self.assertAlmostEqual(row.mae, error.abs().mean())
                self.assertAlmostEqual(row.rmse, np.sqrt((error ** 2).mean()))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np

from benchmark import BUDGET, corrupt_yields, flag_indices, mean_ci, oof_predictions, split_indices


class BenchmarkTests(unittest.TestCase):
    def test_corruption_touches_only_sampled_training_rows(self):
        y = np.arange(100, dtype=float)
        for rate in (0.05, 0.10, 0.20):
            observed, selected = corrupt_yields(y, rate, np.random.default_rng(12))
            self.assertEqual(selected.sum(), round(rate * len(y)))
            np.testing.assert_array_equal(observed[~selected], y[~selected])
            self.assertTrue(np.all(observed[selected] == 0))
            np.testing.assert_array_equal(y, np.arange(100))

    def test_existing_zero_can_be_selected(self):
        observed, selected = corrupt_yields(np.zeros(20), 0.1, np.random.default_rng(12))
        self.assertEqual(selected.sum(), 2)
        self.assertFalse(observed.any())

    def test_held_out_additive_never_appears_in_training(self):
        x = np.array([["other", a] for a in ("c", "b", "a", "c", "a", "b")])
        for seed, additive in enumerate(("a", "b", "c")):
            train, test = split_indices(x, "held-out additive", seed)
            self.assertEqual(set(x[test, -1]), {additive})
            self.assertNotIn(additive, x[train, -1])
            np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(x)))

    def test_oof_predicts_each_row_without_fitting_it(self):
        ids = np.arange(50)
        x = ids[:, None]

        def predict(x_fit, y_fit, x_val, seed):
            self.assertFalse(set(x_fit[:, 0]) & set(x_val[:, 0]))
            np.testing.assert_array_equal(y_fit, x_fit[:, 0])
            return x_val[:, 0]

        with patch("benchmark.forest_predict", side_effect=predict) as model:
            np.testing.assert_array_equal(oof_predictions(x, ids, 4), ids)
            self.assertEqual(model.call_count, 5)

    def test_rankings_have_same_fixed_budget(self):
        observed = np.arange(30, dtype=float)
        predicted = observed.copy()
        predicted[-3:] += 50
        flags = flag_indices(observed, predicted, np.random.default_rng(9))
        for flagged in flags.values():
            self.assertEqual(len(set(flagged)), round(BUDGET * len(observed)))
        self.assertEqual(set(flags["residual"]), {27, 28, 29})
        self.assertEqual(set(flags["lowest"]), {0, 1, 2})

    def test_paired_interval_keeps_constant_effect(self):
        self.assertEqual(mean_ci([2, 2, 2]), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()

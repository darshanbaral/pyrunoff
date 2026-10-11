import unittest
import warnings

import numpy
import pandas

from pyrunoff.metrics import kge, nse

INDEX = pandas.date_range("2024-01-01", periods=6, freq="h")
OBSERVED = pandas.Series([1.0, 3.0, 8.0, 5.0, 2.0, 1.0], index=INDEX)


class TestNse(unittest.TestCase):
    def test_perfect(self):
        self.assertAlmostEqual(nse(OBSERVED, OBSERVED), 1.0)

    def test_mean_prediction_scores_zero(self):
        mean = pandas.Series(OBSERVED.mean(), index=INDEX)
        self.assertAlmostEqual(nse(OBSERVED, mean), 0.0)

    def test_ignores_missing_pairs(self):
        simulated = OBSERVED.copy()
        simulated.iloc[2] = numpy.nan
        self.assertAlmostEqual(nse(OBSERVED, simulated), 1.0)

    def test_constant_observed(self):
        constant = pandas.Series(2.0, index=INDEX)
        self.assertTrue(numpy.isnan(nse(constant, OBSERVED)))

    def test_low_flow_transform(self):
        self.assertAlmostEqual(
            nse(OBSERVED, OBSERVED, use_low_flow_transform=True), 1.0
        )
        with self.assertRaisesRegex(ValueError, "low_flow_percentile"):
            nse(OBSERVED, OBSERVED, use_low_flow_transform=True, low_flow_percentile=0)


class TestKge(unittest.TestCase):
    def test_perfect(self):
        result = kge(OBSERVED, OBSERVED)
        self.assertAlmostEqual(result.kge, 1.0)
        self.assertAlmostEqual(result.corr_coef, 1.0)
        self.assertAlmostEqual(result.gamma, 1.0)
        self.assertAlmostEqual(result.beta, 1.0)

    def test_scaled_simulation(self):
        # doubling keeps correlation and variability ratio, doubles the mean
        result = kge(OBSERVED, OBSERVED * 2)
        self.assertAlmostEqual(result.corr_coef, 1.0)
        self.assertAlmostEqual(result.gamma, 1.0)
        self.assertAlmostEqual(result.beta, 2.0)
        self.assertAlmostEqual(result.kge, 0.0)

    def test_undefined_without_numpy_warnings(self):
        constant = pandas.Series(2.0, index=INDEX)
        zeros = pandas.Series(0.0, index=INDEX)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for observed, simulated in [
                (constant, OBSERVED),  # constant observed
                (OBSERVED, constant),  # constant simulated
                (OBSERVED, zeros),  # zero simulated mean
            ]:
                self.assertTrue(numpy.isnan(kge(observed, simulated).kge))


if __name__ == "__main__":
    unittest.main()

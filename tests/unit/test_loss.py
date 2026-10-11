import unittest
import unittest.mock
from datetime import timedelta

import pandas

from pyrunoff.loss import DeficitConstant, InitialConstant
from pyrunoff.utils import RainfallSeries

INDEX = pandas.date_range("2024-01-01", periods=8, freq="15min")


def rainfall(depth=0.1):
    return RainfallSeries(pandas.Series(depth, index=INDEX))


@unittest.mock.patch("warnings.warn")
class TestInitialConstant(unittest.TestCase):
    def test_losses(self, _):
        # 0.25 in initial loss, then 0.2 in/hr = 0.05 in per 15-minute step
        result = InitialConstant(
            rainfall(), initial_loss=0.25, constant_loss=0.2
        ).route()
        expected_loss = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]
        pandas.testing.assert_series_equal(
            result.total_loss.data,
            pandas.Series(expected_loss, index=INDEX),
            check_names=False,
        )

    def test_conserves_rainfall(self, _):
        result = InitialConstant(
            rainfall(), initial_loss=0.3, constant_loss=0.1
        ).route()
        pandas.testing.assert_series_equal(
            result.excess.data + result.total_loss.data,
            rainfall().data,
            check_names=False,
        )

    def test_rejects_negative_loss(self, _):
        with self.assertRaisesRegex(ValueError, "initial_loss"):
            InitialConstant(rainfall(), initial_loss=-0.1, constant_loss=0.1)


@unittest.mock.patch("warnings.warn")
class TestDeficitConstant(unittest.TestCase):
    def test_default_recovery_requires_recovery_time(self, _):
        with self.assertRaisesRegex(ValueError, "recovery_time must be specified"):
            DeficitConstant(rainfall(), max_deficit=0.5, constant_loss=0.1)

    def test_without_recovery(self, _):
        result = DeficitConstant(
            rainfall(),
            max_deficit=0.5,
            constant_loss=0.2,
            recover_deficit=False,
            initial_deficit=0.25,
        ).route()
        # the deficit absorbs the first 0.25 in, then 0.05 in per step is lost
        expected_loss = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]
        pandas.testing.assert_series_equal(
            result.total_loss.data,
            pandas.Series(expected_loss, index=INDEX),
            check_names=False,
        )

    def test_conserves_rainfall(self, _):
        result = DeficitConstant(
            rainfall(),
            max_deficit=0.5,
            constant_loss=0.1,
            recovery_time=timedelta(hours=24),
            initial_deficit=0.2,
        ).route()
        pandas.testing.assert_series_equal(
            result.excess.data + result.total_loss.data,
            rainfall().data,
            check_names=False,
        )

    def test_failed_update_leaves_params(self, _):
        loss = DeficitConstant(
            rainfall(), max_deficit=0.5, constant_loss=0.1, recover_deficit=False
        )
        with self.assertRaisesRegex(ValueError, "recovery_time must be specified"):
            loss.update_params(recover_deficit=True)
        self.assertFalse(loss.recover_deficit)


if __name__ == "__main__":
    unittest.main()

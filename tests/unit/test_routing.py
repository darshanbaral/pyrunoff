import unittest
import unittest.mock
from datetime import timedelta

import pandas

from pyrunoff.baseflow import LinearReservoir
from pyrunoff.reach import Muskingum
from pyrunoff.transform import ClarkUh
from pyrunoff.utils import DischargeSeries, RainfallSeries
from pyrunoff.utils.constants import ACRE_TO_SQFEET

# long enough for every hydrograph below to recede
INDEX = pandas.date_range("2024-01-01", periods=600, freq="15min")
STEP_SECONDS = 15 * 60


def storm(start=0, steps=4, depth=0.25):
    """Rainfall of ``depth`` per step for ``steps`` steps from ``start``."""
    rain = pandas.Series(0.0, index=INDEX)
    rain.iloc[start : start + steps] = depth
    return RainfallSeries(rain)


def depth_to_cubic_feet(depth, area):
    return depth / 12 * area * ACRE_TO_SQFEET


@unittest.mock.patch("warnings.warn")
class TestClarkUh(unittest.TestCase):
    def runoff_volume(self, rain, area):
        uh = ClarkUh(
            rain, basin_area=area, tc=timedelta(hours=1), storage_r=timedelta(hours=1)
        )
        flow = uh.route(return_at_computed_timestep=True)
        return flow.data.sum() * flow.time_step.total_seconds()

    def test_conserves_volume_when_rain_starts_at_first_step(self, _):
        rain = storm(start=0)
        expected = depth_to_cubic_feet(rain.data.sum(), 640)
        self.assertAlmostEqual(self.runoff_volume(rain, 640) / expected, 1, places=2)

    def test_conserves_volume_after_dry_start(self, _):
        rain = storm(start=10)
        expected = depth_to_cubic_feet(rain.data.sum(), 640)
        self.assertAlmostEqual(self.runoff_volume(rain, 640) / expected, 1, places=2)

    def test_returns_original_interval(self, _):
        flow = ClarkUh(
            storm(), basin_area=640, tc=timedelta(hours=1), storage_r=timedelta(hours=1)
        ).route()
        self.assertTrue(flow.data.index.equals(INDEX))
        self.assertEqual(flow.time_step, pandas.Timedelta(minutes=15))

    def test_rejects_non_positive_compute_time_step(self, _):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            ClarkUh(
                storm(),
                basin_area=640,
                tc=timedelta(hours=1),
                storage_r=timedelta(hours=1),
                compute_time_step=timedelta(0),
            )


@unittest.mock.patch("warnings.warn")
class TestLinearReservoir(unittest.TestCase):
    def test_conserves_routed_volume(self, _):
        loss = storm(start=4)
        flow = LinearReservoir(
            loss, gw_fraction=0.3, k=timedelta(hours=2), basin_area=500
        ).route()
        expected = depth_to_cubic_feet(loss.data.sum() * 0.3, 500)
        self.assertAlmostEqual(flow.data.sum() * STEP_SECONDS / expected, 1, places=3)

    def test_no_groundwater(self, _):
        flow = LinearReservoir(
            storm(), gw_fraction=0.0, k=timedelta(hours=2), basin_area=500
        ).route()
        self.assertTrue((flow.data == 0).all())

    def test_rejects_negative_initial_q(self, _):
        with self.assertRaisesRegex(ValueError, "should not be negative"):
            LinearReservoir(
                storm(),
                gw_fraction=0.3,
                k=timedelta(hours=2),
                basin_area=500,
                initial_q=-1,
            )


@unittest.mock.patch("warnings.warn")
class TestMuskingum(unittest.TestCase):
    def test_steady_inflow_passes_through(self, _):
        inflow = DischargeSeries(pandas.Series(100.0, index=INDEX))
        outflow = Muskingum(inflow, k=timedelta(minutes=30), x=0.2).route()
        self.assertTrue((outflow.total_outflow.data - 100.0).abs().max() < 1e-9)

    def test_conserves_volume_and_attenuates(self, _):
        flow = pandas.Series(0.0, index=INDEX)
        flow.iloc[4:12] = 500.0
        inflow = DischargeSeries(flow)
        outflow = Muskingum(inflow, k=timedelta(hours=1), x=0.2).route().total_outflow
        self.assertAlmostEqual(outflow.data.sum() / flow.sum(), 1, places=6)
        self.assertLess(outflow.data.max(), flow.max())
        self.assertGreater(outflow.data.idxmax(), flow.idxmax())

    def test_rejects_unstable_time_step(self, _):
        inflow = DischargeSeries(pandas.Series(100.0, index=INDEX))
        with self.assertRaisesRegex(ValueError, "stability"):
            Muskingum(inflow, k=timedelta(minutes=5), x=0.2)


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import timedelta

import pandas

from pyrunoff.baseflow import LinearReservoir
from pyrunoff.loss import DeficitConstant, InitialConstant
from pyrunoff.reach import Muskingum
from pyrunoff.reservoir import ElevationStorage, LevelPool, StorageDischarge
from pyrunoff.transform import ClarkUh
from pyrunoff.utils import DischargeSeries, RainfallSeries
from pyrunoff.utils.time_series import TimeSeries

INDEX = pandas.date_range("2024-01-01", periods=8, freq="15min")


def rainfall():
    return RainfallSeries(pandas.Series(0.1, index=INDEX))


def discharge():
    return DischargeSeries(pandas.Series(10.0, index=INDEX))


class TestCopy(unittest.TestCase):
    def test_copy_keeps_class(self):
        self.assertIs(type(discharge().copy()), DischargeSeries)
        self.assertIs(type(rainfall().copy()), RainfallSeries)


class TestResample(unittest.TestCase):
    def test_keeps_total_depth(self):
        resampled = rainfall().resample_at(timedelta(minutes=5))
        self.assertAlmostEqual(resampled.sum(), rainfall().data.sum())

    def test_first_value_covers_interval_before_first_timestamp(self):
        resampled = rainfall().resample_at(timedelta(minutes=5))
        # 0.1 in over the 15 minutes ending at INDEX[0], spread over 5-minute steps
        self.assertEqual(resampled.index[0], INDEX[0] - timedelta(minutes=10))
        self.assertEqual(resampled.index[-1], INDEX[-1])
        self.assertAlmostEqual(resampled.iloc[0], 0.1 / 3)
        self.assertEqual(len(resampled), 3 * len(INDEX))

    def test_not_supported_on_other_series(self):
        with self.assertRaises(NotImplementedError):
            discharge().resample_at(timedelta(minutes=5))


class TestEmptySeriesEnd(unittest.TestCase):
    def test_fix_end_on_empty_series(self):
        empty = TimeSeries(
            pandas.Series([], index=pandas.DatetimeIndex([]), dtype=float),
            _time_step=pandas.Timedelta(minutes=15),
        )
        with self.assertRaisesRegex(ValueError, "empty series"):
            empty.fix_end(INDEX[0])


class TestSum(unittest.TestCase):
    def test_sum_same_class(self):
        total = DischargeSeries.sum([discharge(), discharge()])
        self.assertIsInstance(total, DischargeSeries)
        self.assertTrue((total.data == 20.0).all())

    def test_sum_rejects_other_class(self):
        with self.assertRaises(TypeError) as cm:
            DischargeSeries.sum([discharge(), rainfall()])
        self.assertIn("DischargeSeries", str(cm.exception))

    def test_base_class_sum_accepts_any_series(self):
        total = TimeSeries.sum([discharge(), rainfall()])
        self.assertIs(type(total), TimeSeries)


class TestDictFromDataframe(unittest.TestCase):
    def table(self):
        return pandas.DataFrame({"B1": 0.1, "B2": 0.2, "B3": 0.3}, index=INDEX)

    def test_series_per_column(self):
        series = RainfallSeries.dict_from_dataframe(self.table())
        self.assertEqual(list(series), ["B1", "B2", "B3"])
        for ts in series.values():
            self.assertIs(type(ts), RainfallSeries)
            self.assertEqual(ts.time_step, pandas.Timedelta(minutes=15))
        self.assertTrue((series["B2"].data == 0.2).all())

    def test_class_of_caller(self):
        series = DischargeSeries.dict_from_dataframe(self.table())
        self.assertIs(type(series["B1"]), DischargeSeries)

    def test_selected_columns(self):
        series = RainfallSeries.dict_from_dataframe(self.table(), columns=["B3", "B1"])
        self.assertEqual(list(series), ["B3", "B1"])

    def test_sorts_index(self):
        series = RainfallSeries.dict_from_dataframe(self.table().iloc[::-1])
        self.assertTrue(series["B1"].data.index.is_monotonic_increasing)

    def test_missing_columns(self):
        with self.assertRaisesRegex(ValueError, r"Columns \['B9'\] are not in"):
            RainfallSeries.dict_from_dataframe(self.table(), columns=["B1", "B9"])

    def test_duplicated_columns(self):
        with self.assertRaisesRegex(ValueError, r"\['B1'\] appear more than once"):
            RainfallSeries.dict_from_dataframe(self.table(), columns=["B1", "B1"])
        table = pandas.concat([self.table(), self.table()["B1"]], axis=1)
        with self.assertRaisesRegex(ValueError, r"\['B1'\] appear more than once"):
            RainfallSeries.dict_from_dataframe(table)

    def test_error_names_column(self):
        table = self.table()
        table.loc[INDEX[3], "B2"] = -1.0
        with self.assertRaisesRegex(ValueError, "Column 'B2': Rainfall cannot be"):
            RainfallSeries.dict_from_dataframe(table)

    def test_end_extends(self):
        end = INDEX[-1] + timedelta(hours=1)
        series = RainfallSeries.dict_from_dataframe(self.table(), end=end)
        for ts in series.values():
            self.assertEqual(ts.data.index[-1], end)
            self.assertEqual(ts.data.iloc[-1], 0.0)
            self.assertEqual(ts.time_step, pandas.Timedelta(minutes=15))
        self.assertEqual(len(series["B1"].data), len(INDEX) + 4)

    def test_end_truncates(self):
        series = RainfallSeries.dict_from_dataframe(self.table(), end=INDEX[3])
        self.assertEqual(series["B1"].data.index[-1], INDEX[3])
        self.assertEqual(len(series["B1"].data), 4)


class TestFixEnd(unittest.TestCase):
    def setUp(self):
        self.original = DischargeSeries(
            pandas.Series(
                [1.0 for _ in range(20)] + [5.0 for _ in range(5)],
                index=pandas.date_range(
                    "2017-01-01 00:00:00", "2017-01-02 00:00:00", freq="1h"
                ),
            )
        )
        self.last = self.original.data.index[-1]

    def test_truncates(self):
        clipped = self.original.fix_end(self.last - pandas.Timedelta(hours=5))
        self.assertEqual(clipped.data.index[-1], self.last - pandas.Timedelta(hours=5))
        self.assertEqual(clipped.data.iloc[-1], 1.0)

    def test_extends_with_zeros(self):
        extended = self.original.fix_end(self.last + pandas.Timedelta(hours=5))
        self.assertEqual(extended.data.index[-1], self.last + pandas.Timedelta(hours=5))
        self.assertEqual(extended.data.iloc[-1], 0.0)
        self.assertTrue(
            extended.data.loc[self.original.data.index].equals(self.original.data)
        )
        self.assertEqual(extended.time_step, self.original.time_step)

    def test_keeps_class_and_leaves_original_unchanged(self):
        before = self.original.data.copy()
        self.assertIs(type(self.original.fix_end(self.last)), DischargeSeries)
        self.assertIs(type(rainfall().fix_end(INDEX[-1])), RainfallSeries)
        self.original.fix_end(self.last + pandas.Timedelta(hours=5))
        self.assertTrue(self.original.data.equals(before))

    def test_method_does_not_change_caller_series(self):
        inflow = discharge()
        before = inflow.data.copy()
        Muskingum(
            inflow, k=timedelta(minutes=30), x=0.2, end=INDEX[-1] + timedelta(hours=1)
        )
        self.assertTrue(inflow.data.equals(before))


class TestMethodInputTypes(unittest.TestCase):
    def assert_rejects(self, build):
        with self.assertRaises(TypeError):
            build()

    def test_loss(self):
        self.assert_rejects(
            lambda: InitialConstant(
                rainfall_series=discharge(), initial_loss=0.1, constant_loss=0.1
            )
        )
        self.assert_rejects(
            lambda: DeficitConstant(
                rainfall_series=discharge(),
                max_deficit=0.3,
                constant_loss=0.1,
                recovery_time=timedelta(hours=24),
            )
        )

    def test_transform(self):
        self.assert_rejects(
            lambda: ClarkUh(
                rainfall_series=discharge(),
                basin_area=100,
                tc=timedelta(hours=1),
                storage_r=timedelta(hours=1),
            )
        )

    def test_baseflow(self):
        self.assert_rejects(
            lambda: LinearReservoir(
                rainfall_series=discharge(),
                gw_fraction=0.2,
                k=timedelta(hours=10),
                basin_area=100,
            )
        )

    def test_reach(self):
        self.assert_rejects(
            lambda: Muskingum(inflow_series=rainfall(), k=timedelta(hours=1), x=0.2)
        )

    def test_reservoir(self):
        curve = pandas.Series([0.0, 1.0, 2.0])
        self.assert_rejects(
            lambda: LevelPool(
                inflow_series=rainfall(),
                elevation_storage=ElevationStorage(curve, curve),
                storage_discharges={"spillway": StorageDischarge(curve, curve)},
                initial_storage=0.0,
            )
        )

    def test_accepts_correct_types(self):
        Muskingum(inflow_series=discharge(), k=timedelta(hours=1), x=0.2)
        InitialConstant(rainfall_series=rainfall(), initial_loss=0.1, constant_loss=0.1)


if __name__ == "__main__":
    unittest.main()

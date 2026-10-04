import pandas
import pytest

from pyrunoff.reservoir import (
    ElevationStorage,
    LevelPool,
    LevelPoolOutput,
    StorageDischarge,
)
from pyrunoff.utils import DischargeSeries, ElevationSeries, StorageSeries


def make_level_pool(inflow: list[float]) -> LevelPool:
    index = pandas.date_range("2025-01-01", periods=len(inflow), freq="min")
    inflow_series = DischargeSeries(
        pandas.Series(inflow, index=index),
        sorted=True,
        _time_step=pandas.Timedelta(minutes=1),
    )
    elevation_storage = ElevationStorage(
        elevation=pandas.Series([100.0, 200.0]),
        storage=pandas.Series([0.0, 100.0]),
    )
    storage_discharge = StorageDischarge(
        storage=pandas.Series([0.0, 100.0]),
        discharge=pandas.Series([0.0, 10.0]),
    )
    return LevelPool(
        inflow_series=inflow_series,
        elevation_storage=elevation_storage,
        storage_discharges={"outlet": storage_discharge},
        initial_storage=10.0,
    )


def test_route_returns_typed_level_pool_output():
    result = make_level_pool([0.0, 0.0]).route()

    assert isinstance(result, LevelPoolOutput)
    assert set(result.outflows) == {"outlet"}
    assert isinstance(result.outflows["outlet"], DischargeSeries)
    assert isinstance(result.total_outflow, DischargeSeries)
    assert isinstance(result.storage, StorageSeries)
    assert isinstance(result.elevation, ElevationSeries)
    pandas.testing.assert_series_equal(
        result.total_outflow.data,
        result.outflows["outlet"].data.rename("total_outflow"),
    )
    assert result.storage.data.iloc[0] == pytest.approx(10.0)
    assert result.elevation.data.iloc[0] == pytest.approx(110.0)


def test_route_returns_typed_empty_series():
    result = make_level_pool([]).route()

    assert isinstance(result, LevelPoolOutput)
    assert result.outflows["outlet"].data.empty
    assert result.total_outflow.data.empty
    assert result.storage.data.empty
    assert result.elevation.data.empty
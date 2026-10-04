from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

import pandas

from pyrunoff.utils.constants import ABSOLUTE_EPSILON, NANOSECONDS_PER_SECOND
from pyrunoff.utils.tso import check_time_step, get_time_step


@dataclass(slots=True)
class TimeSeries:
    """
    Base class for numeric time series indexed by timestamps.

    :param data: Numeric values indexed by timestamps.
    :param sorted: Whether to assume the index is already sorted.
    :param _time_step: Internal override for the series time step. For
        user-supplied data, the time step should normally be inferred from the
        DatetimeIndex. This override is not a trusted public API path and should
        only be used when the caller understands the implications of using it.
    """

    data: pandas.Series
    sorted: bool = False
    _time_step: pandas.Timedelta | None = None

    def __post_init__(self):
        """Convert values to floats, sort the index, and validate the time step."""
        if not pandas.api.types.is_float_dtype(self.data):
            try:
                self.data = self.data.astype(float)
            except ValueError as err:
                raise TypeError("TimeSeries data must be float-convertible.") from err

        if not self.sorted:
            self.data = self.data.sort_index()
            self.sorted = True

        if self._time_step is None:
            time_step = get_time_step(self.data.index)
            check_time_step(time_step)
            self._time_step = time_step

        self.validate_time_step(self._time_step)
        self.validate_data()

    @property
    def time_step(self):
        """Return the interval between consecutive time-series values."""
        return self._time_step

    @time_step.setter
    def time_step(self, value: pandas.Timedelta | timedelta):
        """Set the series interval after validating its units."""
        value = pandas.to_timedelta(value)
        self.validate_time_step(value)
        self._time_step = value

    @staticmethod
    def validate_time_step(time_step: pandas.Timedelta):
        """Require a pandas duration expressed in whole minutes."""
        if not isinstance(time_step, pandas.Timedelta):
            raise TypeError("_time_step must be pandas.Timedelta")

        if time_step.value % (60 * NANOSECONDS_PER_SECOND) != 0:
            raise ValueError(f"time_step must be whole minutes, not {time_step}")

    @classmethod
    def sum(cls, series: Iterable["TimeSeries"]) -> "TimeSeries":
        """Return the elementwise sum of aligned series with matching intervals."""
        series: list[TimeSeries] = list(series)

        if not series:
            raise ValueError("At least one TimeSeries is required.")

        if not all(isinstance(ts, TimeSeries) for ts in series):
            raise TypeError("All elements must be TimeSeries instances.")

        first = series[0]

        # validate time_step
        for ts in series:
            if not ts.sorted:
                raise ValueError("Time series must be sorted by index.")

            if ts.time_step != first.time_step:
                raise ValueError("time step of all time series must be the same.")

        dfs = [ts.data for ts in series]
        df = pandas.concat(dfs, axis=1)

        if df.isna().any().any():
            raise ValueError("Time series indices do not align exactly.")

        result = df.sum(axis=1)

        return cls(data=result, sorted=True, _time_step=first.time_step)

    def copy(self) -> "TimeSeries":
        """Return a copy with independent underlying data."""
        return TimeSeries(
            data=self.data.copy(deep=True),
            sorted=self.sorted,
            _time_step=self._time_step,
        )

    def validate_data(self):
        """Validate values specific to a time-series subclass."""
        pass

    def resample_at(self, time_step: pandas.Timedelta | timedelta) -> pandas.Series:
        """Return series values resampled to the requested interval."""
        pass


class DischargeSeries(TimeSeries):
    """Time series of discharge values."""

    pass


class StageSeries(TimeSeries):
    """Time series of water-level values."""

    pass


class ElevationSeries(TimeSeries):
    """Time series of elevation values."""

    pass


class StorageSeries(TimeSeries):
    """Time series of storage values."""

    pass


class RainfallSeries(TimeSeries):
    """Time series of incremental rainfall values."""

    def validate_data(self):
        """Reject rainfall values below the accepted negative tolerance."""
        self.validate_minimum_rain(self.data)

    @staticmethod
    def validate_minimum_rain(data: pandas.Series):
        """Raise an error if any rainfall value is negative beyond tolerance."""
        if (data < -ABSOLUTE_EPSILON).any():
            raise ValueError("Rainfall cannot be negative")

    def resample_at(self, time_step: pandas.Timedelta | timedelta) -> pandas.Series:
        """Resample rainfall while preserving accumulated depth."""
        time_step = pandas.to_timedelta(time_step)
        data_series = self.data
        new_index = pandas.date_range(
            start=data_series.index[0],
            end=data_series.index[-1],
            freq=time_step,
        )

        new_data_series = (
            data_series.cumsum()
            .reindex(new_index)
            .interpolate(method="time")
            .diff()
            .fillna(0)
            .clip(lower=0)
        )

        return new_data_series

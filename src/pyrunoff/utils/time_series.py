from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Self, TypeVar

import pandas

from pyrunoff.utils.constants import ABSOLUTE_EPSILON, NANOSECONDS_PER_SECOND
from pyrunoff.utils.tso import check_time_step, get_time_step

SeriesOrFrame = TypeVar("SeriesOrFrame", pandas.Series, pandas.DataFrame)


def _fix_end(
    data: SeriesOrFrame, end: pandas.Timestamp | datetime, time_step: pandas.Timedelta
) -> SeriesOrFrame:
    """Truncate ``data`` at ``end``, or extend it to ``end`` with zero rows.

    :param data: Values with a sorted ``DatetimeIndex``.
    :param end: Timestamp at which the result ends.
    :param time_step: Interval of the added timestamps.
    """
    if not isinstance(data.index, pandas.DatetimeIndex):
        raise TypeError("Series must have a DatetimeIndex.")
    if len(data.index) == 0:
        raise ValueError("Cannot change the end of an empty series.")

    last_time = data.index[-1]
    end_ts = pandas.Timestamp(end)

    if end_ts <= last_time:
        return data.loc[data.index <= end_ts]

    extend_index = pandas.date_range(
        start=last_time + time_step, end=end_ts, freq=time_step
    )
    return data.reindex(data.index.append(extend_index), fill_value=0.0)


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
        """Return the elementwise sum of aligned series with matching intervals.

        Every series must be an instance of the class ``sum`` is called on, so
        ``DischargeSeries.sum`` only accepts discharge series.
        """
        series: list[TimeSeries] = list(series)

        if not series:
            raise ValueError(f"At least one {cls.__name__} is required.")

        if not all(isinstance(ts, cls) for ts in series):
            raise TypeError(f"All elements must be {cls.__name__} instances.")

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

    @classmethod
    def dict_from_dataframe(
        cls,
        data: pandas.DataFrame,
        columns: Iterable[str] | None = None,
        end: pandas.Timestamp | datetime | None = None,
    ) -> dict[str, Self]:
        """Return one series per column of a table indexed by timestamps.

        The index is sorted and its time step inferred once for all columns.

        :param data: Values indexed by timestamps, one column per series.
        :param columns: Columns to convert; defaults to all of them.
        :param end: Optional timestamp at which every series ends; see
            :meth:`fix_end`.
        :return: Series keyed by column name.
        :raises ValueError: If columns are missing or duplicated, or a column's
            values are invalid for this class (the error names the column).
        :raises TypeError: If a column's values are not float-convertible.
        """
        names = list(data.columns if columns is None else columns)
        missing = [name for name in names if name not in data.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} are not in the data; it has {list(data.columns)}"
            )
        duplicated = sorted(
            {
                name
                for name in names
                if names.count(name) > 1 or (data.columns == name).sum() > 1
            }
        )
        if duplicated:
            raise ValueError(f"Columns {duplicated} appear more than once")

        data = data[names].sort_index()
        time_step = get_time_step(data.index)
        check_time_step(time_step)
        if end is not None:
            data = _fix_end(data, end, time_step)

        result = {}
        for name in names:
            try:
                result[name] = cls(data[name], sorted=True, _time_step=time_step)
            except (TypeError, ValueError) as err:
                raise type(err)(f"Column '{name}': {err}") from err
        return result

    def copy(self) -> "TimeSeries":
        """Return a copy of the same class with independent underlying data."""
        return type(self)(
            data=self.data.copy(deep=False),
            sorted=self.sorted,
            _time_step=self._time_step,
        )

    def fix_end(self, end: pandas.Timestamp | datetime) -> Self:
        """Return a series of the same class ending at ``end``.

        Values after ``end`` are dropped; timestamps up to ``end`` are added at
        :attr:`time_step` with zero values.

        :param end: Timestamp at which the returned series ends.
        :return: New series; this one is unchanged.
        """
        data = _fix_end(self.data, end, self.time_step)
        return type(self)(data=data, sorted=True, _time_step=self.time_step)

    def validate_data(self):
        """Validate values specific to a time-series subclass."""
        pass

    def resample_at(self, time_step: pandas.Timedelta | timedelta) -> pandas.Series:
        """Return series values resampled to the requested interval."""
        raise NotImplementedError(f"{type(self).__name__} does not support resampling")


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
        """Resample rainfall while preserving accumulated depth.

        Each value is the depth over the interval ending at its timestamp, so
        the first value covers the interval before the first timestamp. The
        result therefore starts one new ``time_step`` after
        ``index[0] - self.time_step`` and keeps the total depth.

        :param time_step: Interval of the returned values.
        :return: Incremental depth at ``time_step`` intervals.
        """
        time_step = pandas.to_timedelta(time_step)
        data = self.data
        start = data.index[0] - self.time_step

        # cumulative depth, anchored at zero where the first interval begins
        cumulative = pandas.concat([pandas.Series([0.0], index=[start]), data.cumsum()])
        new_index = pandas.date_range(start=start, end=data.index[-1], freq=time_step)

        return (
            cumulative.reindex(cumulative.index.union(new_index))
            .interpolate(method="time")
            .reindex(new_index)
            .diff()
            .iloc[1:]
            .clip(lower=0)
        )

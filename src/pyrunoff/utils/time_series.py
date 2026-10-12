"""Time series of rainfall, discharge, elevation and storage values."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Self

import pandas

from pyrunoff.utils.constants import ABSOLUTE_EPSILON, NANOSECONDS_PER_SECOND


def _get_time_step(
    dt_idx: pandas.DatetimeIndex,
) -> pandas.Timedelta:
    """
    Infer the constant interval between values in a datetime index.

    :param dt_idx: Datetime index with at least two uniformly spaced values.
    :return: Time interval between adjacent values.
    """
    if not isinstance(dt_idx, pandas.DatetimeIndex):
        raise TypeError("input must be a DatetimeIndex")

    if len(dt_idx) <= 1:
        raise ValueError("Index must have at least 2 values to determine a timestep.")

    idx_arr = dt_idx.to_numpy()
    diffs_us = (idx_arr[1:] - idx_arr[:-1]).astype("timedelta64[us]")  # micro-seconds
    diffs_s = (diffs_us.astype(float) / 1e6).round().astype("timedelta64[s]")
    first_diff_s = diffs_s[0]

    if not (diffs_s == first_diff_s).all():
        raise ValueError("Data is not uniformly spaced (Internal step mismatch).")

    return pandas.Timedelta(first_diff_s)


def _check_time_step(
    time_step: timedelta,
    tc: timedelta | None = None,
    name: str = "time_step",
):
    """Validate a positive time step and its relationship to an optional limit.

    Steps shorter than an hour must evenly divide an hour; longer steps must
    be whole hours. When ``tc`` is provided, the step cannot exceed it.

    :param time_step: Duration to validate.
    :param tc: Optional upper bound for the duration.
    :param name: Label used in validation errors.
    :raises TypeError: If either duration has an unsupported type.
    :raises ValueError: If a duration is non-positive or violates the rules.
    """
    if not isinstance(time_step, pandas.Timedelta | timedelta):
        raise TypeError("time_step needs to be a timedelta")

    ts_s = int(time_step.total_seconds())

    if ts_s <= 0:
        raise ValueError(f"{name} must be a positive duration")

    hr_s = 3600
    if ts_s < hr_s:
        if hr_s % ts_s != 0:
            raise ValueError(f"{name} must evenly divide 60 minutes.")
    else:
        if ts_s % hr_s != 0:
            raise ValueError(f"{name} must be a multiple of 60 minutes.")

    if tc is not None:
        if not isinstance(tc, pandas.Timedelta | timedelta):
            raise TypeError("tc needs to be a timedelta")

        tc_s = int(tc.total_seconds())
        if ts_s > tc_s:
            raise ValueError(f"{name} time_step cannot be bigger than tc.")


def _fix_end[SeriesOrFrame: (pandas.Series, pandas.DataFrame)](
    data: SeriesOrFrame, end: pandas.Timestamp | datetime, time_step: pandas.Timedelta
) -> SeriesOrFrame:
    """Truncate ``data`` at ``end``, or extend it to ``end`` with zero rows.

    :param data: Values with a sorted ``DatetimeIndex``.
    :param end: Timestamp at which the result ends.
    :param time_step: Interval of the added timestamps.
    :return: Values ending at ``end``.
    :raises TypeError: If ``data`` has no ``DatetimeIndex``.
    :raises ValueError: If ``data`` is empty.
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
            time_step = _get_time_step(self.data.index)
            _check_time_step(time_step)
            self._time_step = time_step

        self._validate_time_step(self._time_step)
        self._validate_data()

    @property
    def time_step(self):
        """Return the interval between consecutive time-series values."""
        return self._time_step

    @time_step.setter
    def time_step(self, value: pandas.Timedelta | timedelta):
        """Set the series interval after validating its units.

        :param value: New interval, in whole minutes.
        """
        value = pandas.to_timedelta(value)
        self._validate_time_step(value)
        self._time_step = value

    @staticmethod
    def _validate_time_step(time_step: pandas.Timedelta):
        """Require a pandas duration expressed in whole minutes.

        :param time_step: Interval to check.
        :raises TypeError: If ``time_step`` is not a ``pandas.Timedelta``.
        :raises ValueError: If ``time_step`` is not a whole number of minutes.
        """
        if not isinstance(time_step, pandas.Timedelta):
            raise TypeError("_time_step must be pandas.Timedelta")

        if time_step.value % (60 * NANOSECONDS_PER_SECOND) != 0:
            raise ValueError(f"time_step must be whole minutes, not {time_step}")

    @classmethod
    def sum(cls, series: Iterable[TimeSeries]) -> TimeSeries:
        """Return the elementwise sum of aligned series with matching intervals.

        Every series must be an instance of the class ``sum`` is called on, so
        ``DischargeSeries.sum`` only accepts discharge series.

        :param series: Series to add, with identical indexes and time steps.
        :return: Sum, of the class ``sum`` is called on.
        :raises TypeError: If a series is not of that class.
        :raises ValueError: If no series are given, or they are unsorted or
            their time steps or indexes differ.
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
        time_step = _get_time_step(data.index)
        _check_time_step(time_step)
        if end is not None:
            data = _fix_end(data, end, time_step)

        result = {}
        for name in names:
            try:
                result[name] = cls(data[name], sorted=True, _time_step=time_step)
            except (TypeError, ValueError) as err:
                raise type(err)(f"Column '{name}': {err}") from err
        return result

    def copy(self) -> TimeSeries:
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

    def _validate_data(self):
        """Validate values specific to a time-series subclass."""
        pass

    def resample_at(self, time_step: pandas.Timedelta | timedelta) -> pandas.Series:
        """Return series values resampled to the requested interval.

        :param time_step: Interval of the returned values.
        :return: Resampled values.
        :raises NotImplementedError: Unless a subclass supports resampling.
        """
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

    def _validate_data(self):
        """Reject rainfall values below the accepted negative tolerance."""
        self._validate_minimum_rain(self.data)

    @staticmethod
    def _validate_minimum_rain(data: pandas.Series):
        """Raise an error if any rainfall value is negative beyond tolerance.

        :param data: Rainfall values to check.
        :raises ValueError: If a value is below ``-ABSOLUTE_EPSILON``.
        """
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

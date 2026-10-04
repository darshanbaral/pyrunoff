from datetime import datetime, timedelta

import pandas


def get_time_step(
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


def check_time_step(
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


def fix_ts_end(
    ts: pandas.Series, end: pandas.Timestamp | datetime, time_step: pandas.Timedelta
) -> pandas.Series:
    """
    Extend a time series with zeros or truncate it to a specified end time.

    :param ts: Time series with a datetime index.
    :param end: Timestamp at which the returned series ends.
    :param time_step: Interval used to create added timestamps.

    :return: Time series ending at ``end``.
    """
    if not isinstance(ts.index, pandas.DatetimeIndex):
        raise TypeError("Series must have a DatetimeIndex.")

    ts_index: pandas.DatetimeIndex = ts.index
    last_time = ts_index[-1]
    end_ts = pandas.Timestamp(end)

    if end_ts <= last_time:
        return ts.loc[ts_index <= end_ts]

    extend_index = pandas.date_range(
        start=last_time + time_step, end=end_ts, freq=time_step
    )
    extend_data = pandas.Series(0.0, index=extend_index)
    return pandas.concat([ts, extend_data])

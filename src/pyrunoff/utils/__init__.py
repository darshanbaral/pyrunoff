"""Time series, validation helpers and the network model."""

from .time_series import DischargeSeries, ElevationSeries, RainfallSeries, StorageSeries


def _validate_non_negative(value, name: str | None = None):
    """Raise ``ValueError`` when ``value`` is negative.

    :param value: Number to check.
    :param name: Name used in the error message; defaults to ``"value"``.
    :raises ValueError: If ``value`` is negative.
    """
    name = name or "value"
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero.")


def _validate_positive(value, name: str | None = None):
    """Raise ``ValueError`` when ``value`` is not greater than zero.

    :param value: Number to check.
    :param name: Name used in the error message; defaults to ``"value"``.
    :raises ValueError: If ``value`` is zero or negative.
    """
    name = name or "value"
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _validate_series_type(series, expected: type, name: str | None = None):
    """Raise ``TypeError`` when ``series`` is not an instance of ``expected``.

    :param series: Object to check.
    :param expected: Required type, such as :class:`.RainfallSeries`.
    :param name: Name used in the error message; defaults to ``"series"``.
    :raises TypeError: If ``series`` is not an ``expected`` instance.
    """
    name = name or "series"
    if not isinstance(series, expected):
        raise TypeError(
            f"{name} must be a {expected.__name__}, not {type(series).__name__}."
        )


__all__ = [
    "DischargeSeries",
    "ElevationSeries",
    "RainfallSeries",
    "StorageSeries",
    "_validate_non_negative",
    "_validate_positive",
    "_validate_series_type",
]

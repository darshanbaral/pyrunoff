from .time_series import DischargeSeries, ElevationSeries, RainfallSeries, StorageSeries


def validate_non_negative(value, name: str | None = None):
    """Raise ``ValueError`` when ``value`` is negative."""
    name = name or "value"
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero.")


def validate_positive(value, name: str | None = None):
    """Raise ``ValueError`` when ``value`` is not greater than zero."""
    name = name or "value"
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def validate_series_type(series, expected: type, name: str | None = None):
    """Raise ``TypeError`` when ``series`` is not an instance of ``expected``."""
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
    "validate_non_negative",
    "validate_positive",
    "validate_series_type",
]

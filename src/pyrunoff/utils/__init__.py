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


__all__ = [
    "DischargeSeries",
    "ElevationSeries",
    "RainfallSeries",
    "StorageSeries",
    "validate_non_negative",
    "validate_positive",
]

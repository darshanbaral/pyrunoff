from dataclasses import dataclass

import pandas


def check(
    series1: pandas.Series,
    series2: pandas.Series,
    name1: str,
    name2: str,
):
    """Validate input series types, ordering, uniqueness, and matching lengths.

    The first series must have unique, increasing values. Both inputs must be
    pandas Series with the same number of values.

    :param series1: Series whose values are checked for order and uniqueness.
    :param series2: Series checked for type and matching length.
    :param name1: Label used for errors related to the first series.
    :param name2: Label used for errors related to the second series.
    """
    if not isinstance(series1, pandas.Series):
        raise TypeError(f"{name1} must be a pandas Series")

    if not isinstance(series2, pandas.Series):
        raise TypeError(f"{name2} must be a pandas Series")

    if not series1.is_unique:
        raise ValueError(f"{name1} series must be unique")

    if not series1.is_monotonic_increasing:
        raise ValueError(f"{name1} must be in increasing order")

    if len(series1) != len(series2):
        raise ValueError(
            f"The two series, {name1} and {name2}, must have the same length"
        )


@dataclass(slots=True)
class ElevationStorage:
    """Pair elevation and storage values for a reservoir rating curve.

    :param elevation: Elevation values in increasing order.
    :param storage: Corresponding storage values.
    """

    elevation: pandas.Series
    storage: pandas.Series

    def __post_init__(self):
        """Validate the elevation and storage series."""
        check(self.elevation, self.storage, "elevation", "storage")


@dataclass(slots=True)
class StorageDischarge:
    """Pair storage and discharge values for a reservoir outlet curve.

    :param storage: Storage values in increasing order.
    :param discharge: Corresponding discharge values.
    """

    storage: pandas.Series
    discharge: pandas.Series

    def __post_init__(self):
        """Validate the storage and discharge series."""
        check(self.storage, self.discharge, "storage", "discharge")

"""Shared unit-conversion factors and numerical tolerances."""

#: Absolute tolerance, e.g. for rainfall treated as zero.
ABSOLUTE_EPSILON = 1e-9
#: Relative tolerance for stability-limit comparisons.
RELATIVE_EPSILON = 1e-12
#: Square feet per acre.
ACRE_TO_SQFEET = 43560.0
#: Seconds per hour.
SECONDS_PER_HOUR = 3600.0
#: Nanoseconds per second, the unit of ``pandas.Timedelta.value``.
NANOSECONDS_PER_SECOND = 1_000_000_000

from dataclasses import dataclass
from datetime import timedelta

import numpy
import pandas

from pyrunoff.utils import DischargeSeries, RainfallSeries, validate_positive
from pyrunoff.utils.compat import jit
from pyrunoff.utils.constants import ACRE_TO_SQFEET, RELATIVE_EPSILON, SECONDS_PER_HOUR
from pyrunoff.utils.tso import fix_ts_end


@dataclass(slots=True)
class LinearReservoir:
    """Route groundwater contributions through a linear reservoir.

    :param rainfall_series: Incremental rainfall loss series in inches.
    :param gw_fraction: Fraction of rainfall loss that contributes to
        groundwater flow, from 0 to 1.
    :param k: Reservoir recession time.
    :param basin_area: Drainage area in acres.
    :param initial_q: Initial discharge in cubic feet per second.
    :param end: Optional final timestamp for the rainfall series.
    """

    rainfall_series: RainfallSeries
    gw_fraction: float
    k: pandas.Timedelta | timedelta
    basin_area: float
    initial_q: float = 0.0
    end: pandas.Timestamp | None = None

    def __post_init__(self):
        """Validate parameters and apply the optional end time."""
        validate_positive(self.basin_area, "basin_area")
        self.fix_end()
        self.update_params(
            initial_q=self.initial_q, gw_fraction=self.gw_fraction, k=self.k
        )

    def fix_end(self):
        """Extend or truncate rainfall data to the configured end time."""
        if self.end is not None:
            new_rainfall_data = fix_ts_end(
                self.rainfall_series.data,
                end=self.end,
                time_step=self.rainfall_series.time_step,
            )
            self.rainfall_series = RainfallSeries(
                new_rainfall_data,
                sorted=True,
                _time_step=self.rainfall_series.time_step,
            )

    @staticmethod
    def validate_gw_fraction(gw_fraction):
        """Require the groundwater fraction to lie between zero and one."""
        if gw_fraction > 1 or gw_fraction < 0:
            raise ValueError("'gw_fraction' should be between 0 and 1")

    @staticmethod
    def validate_initial_q(initial_q):
        """Require the initial discharge to be non-negative."""
        if initial_q < 0:
            raise ValueError("'initial_q' should be greater than 0")

    @staticmethod
    def validate_k(k: pandas.Timedelta):
        """Require a positive reservoir recession time."""
        if k <= pandas.Timedelta(minutes=0):
            raise ValueError("'k' should be greater than 0")

    @staticmethod
    def validate_stability(k: pandas.Timedelta, time_step: pandas.Timedelta):
        """Require a time step below the linear-reservoir stability limit."""
        time_step_hr = time_step.total_seconds() / SECONDS_PER_HOUR
        k_hr = k.total_seconds() / SECONDS_PER_HOUR
        if time_step_hr >= 2 * k_hr * (1 - RELATIVE_EPSILON):
            raise ValueError(
                f"time step ({time_step_hr} hr) too large for k={k_hr} hr. "
                "Must satisfy dt < 2k for numerical stability."
            )

    def update_params(
        self,
        initial_q: float | None = None,
        gw_fraction: float | None = None,
        k: pandas.Timedelta | timedelta | None = None,
    ):
        """Validate and update any supplied routing parameters."""
        if initial_q is not None:
            self.validate_initial_q(initial_q)
            self.initial_q = initial_q

        if gw_fraction is not None:
            self.validate_gw_fraction(gw_fraction)
            self.gw_fraction = gw_fraction

        if k is not None:
            k = pandas.Timedelta(k)
            self.validate_k(k)
            self.validate_stability(k, self.rainfall_series.time_step)
            self.k = k

    def route(
        self,
    ) -> DischargeSeries:
        """Route the groundwater contribution and return discharge values."""

        time_step = self.rainfall_series.time_step
        time_step_seconds = time_step.total_seconds()
        rain = self.rainfall_series.data

        if self.gw_fraction == 0 and self.initial_q == 0:
            return DischargeSeries(pandas.Series(0.0, index=rain.index))

        acre_inch_to_cubic_feet = ACRE_TO_SQFEET / 12.0

        inflow_vol = rain * self.gw_fraction * self.basin_area * acre_inch_to_cubic_feet

        inflow = inflow_vol.to_numpy() / time_step_seconds

        time_step_hr = time_step_seconds / SECONDS_PER_HOUR

        k_hr = self.k.total_seconds() / SECONDS_PER_HOUR

        denominator = 2 * k_hr + time_step_hr
        c1 = time_step_hr / denominator
        c2 = (2 * k_hr - time_step_hr) / denominator

        base_flow = _linear_reservoir_kernel(
            inflow=inflow, initial_q=float(self.initial_q), c1=float(c1), c2=float(c2)
        )

        return DischargeSeries(
            pandas.Series(base_flow, index=rain.index),
            sorted=True,
            _time_step=time_step,
        )


@jit(nopython=True, cache=True)  # Added
def _linear_reservoir_kernel(inflow, initial_q, c1, c2):
    """Calculate discharge using the linear-reservoir recurrence."""
    n = len(inflow)
    base_flow = numpy.empty(n, dtype=numpy.float64)
    base_flow[0] = initial_q

    for i in range(1, n):
        inflow_avg = (inflow[i] + inflow[i - 1]) * 0.5  # trapezoidal inflow
        # The routing equation: Q_t = 2 * c1 * I_avg + c2 * Q_{t-1}
        base_flow[i] = 2.0 * c1 * inflow_avg + c2 * base_flow[i - 1]

    return base_flow

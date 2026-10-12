"""Muskingum reach routing method."""

from dataclasses import dataclass
from datetime import timedelta

import numpy
import pandas

from pyrunoff.utils import DischargeSeries, _validate_series_type
from pyrunoff.utils.compat import jit
from pyrunoff.utils.constants import RELATIVE_EPSILON, SECONDS_PER_HOUR


@dataclass(slots=True)
class MuskingumOutput:
    """Results returned by :meth:`Muskingum.route`.

    :param total_outflow: Routed discharge leaving the reach.
    """

    total_outflow: DischargeSeries


@dataclass(slots=True)
class Muskingum:
    """Route discharge through a reach using the Muskingum method.

    :param inflow_series: Discharge entering the reach.
    :param k: Muskingum storage time constant. It must satisfy the stability
        condition for the inflow time step.
    :param x: Weighting factor between zero and 0.5.
    :param end: Optional final timestamp for the inflow series.
    """

    inflow_series: DischargeSeries
    k: pandas.Timedelta | timedelta
    x: float
    end: pandas.Timestamp | None = None

    def __post_init__(self):
        """Apply the optional end time and validate routing parameters."""
        _validate_series_type(self.inflow_series, DischargeSeries, "inflow_series")
        self._fix_end()
        self.update_params(self.k, self.x)

    def _fix_end(self):
        """Extend or truncate inflow data to the configured end time."""
        if self.end is not None:
            self.inflow_series = self.inflow_series.fix_end(self.end)

    @staticmethod
    def _validate_params(k: pandas.Timedelta, x: float, time_step: pandas.Timedelta):
        """Validate the Muskingum parameters and numerical stability condition.

        :param k: Muskingum storage time constant.
        :param x: Weighting factor, from zero to 0.5.
        :param time_step: Interval between inflow values.
        :raises ValueError: If a parameter is outside its valid range or the
            time step violates the stability condition.
        """
        if not (0 <= x <= 0.5):
            raise ValueError("Muskingum x parameter must be between 0 and 0.5.")

        if k <= pandas.Timedelta(minutes=0):
            raise ValueError("Muskingum k must be greater than 0")

        time_step_hr = time_step.total_seconds() / SECONDS_PER_HOUR
        k_hr = k.total_seconds() / SECONDS_PER_HOUR
        if time_step_hr > 2 * k_hr * (1 - x) + RELATIVE_EPSILON:  # f3 becomes negative
            raise ValueError("Time step violates Muskingum stability condition.")

    def update_params(self, k: pandas.Timedelta, x: float):
        """Validate and update the storage time constant and weighting factor.

        :param k: Muskingum storage time constant.
        :param x: Weighting factor, from zero to 0.5.
        :raises ValueError: If a value is invalid or unstable for the time step.
        """
        k = pandas.to_timedelta(k)
        self._validate_params(k=k, x=x, time_step=self.inflow_series.time_step)
        self.k = k
        self.x = x

    def route(
        self,
    ) -> MuskingumOutput:
        """
        Route inflow through the reach and return the resulting discharge.

        :return: Typed output holding the routed discharge series.
        """

        inflow = self.inflow_series.data
        k = self.k
        k_hr = k.total_seconds() / SECONDS_PER_HOUR
        x = self.x

        if inflow.empty:
            return MuskingumOutput(
                DischargeSeries(
                    inflow.copy(), sorted=True, _time_step=self.inflow_series.time_step
                )
            )

        time_step = self.inflow_series.time_step

        time_step_hr = time_step.total_seconds() / SECONDS_PER_HOUR

        denominator = 2 * k_hr * (1 - x) + time_step_hr

        f1 = (time_step_hr - 2 * k_hr * x) / denominator
        f2 = (time_step_hr + 2 * k_hr * x) / denominator
        f3 = (2 * k_hr * (1 - x) - time_step_hr) / denominator

        outflow_arr = _muskingum_kernel(
            inflow_arr=inflow.to_numpy(), f1=float(f1), f2=float(f2), f3=float(f3)
        )

        outflow = pandas.Series(outflow_arr, index=inflow.index)

        return MuskingumOutput(
            DischargeSeries(
                outflow, sorted=True, _time_step=self.inflow_series.time_step
            )
        )


@jit(nopython=True, cache=True)
def _muskingum_kernel(inflow_arr, f1, f2, f3):
    """Calculate routed discharge using Muskingum recurrence coefficients.

    :param inflow_arr: Inflow per step in cubic feet per second.
    :param f1: Current-inflow coefficient.
    :param f2: Previous-inflow coefficient.
    :param f3: Previous-outflow coefficient.
    :return: Routed discharge per step.
    """
    n = len(inflow_arr)
    outflow_arr = numpy.zeros(n, dtype=numpy.float64)

    if n > 0:
        outflow_arr[0] = inflow_arr[0]

    for ind in range(1, n):
        outflow_arr[ind] = (
            f1 * inflow_arr[ind]  # curr inflow component
            + f2 * inflow_arr[ind - 1]  # previous inflow component
            + f3 * outflow_arr[ind - 1]  # previous outflow component
        )
    return outflow_arr

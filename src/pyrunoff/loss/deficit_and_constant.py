from dataclasses import dataclass
from datetime import timedelta

import numpy
import pandas

from pyrunoff.utils import validate_non_negative
from pyrunoff.utils.compat import jit
from pyrunoff.utils.constants import SECONDS_PER_HOUR
from pyrunoff.utils.time_series import RainfallSeries
from pyrunoff.utils.tso import check_time_step, get_time_step


@dataclass(slots=True)
class DeficitConstantOutput:
    """
    Output series produced by the deficit-constant loss model.

    :param excess: Rainfall remaining after all losses.
    :param deficit: Soil-moisture deficit at each time step.
    :param constant_loss: Constant loss applied at each time step.
    :param total_loss: Total rainfall loss at each time step.
    """

    excess: RainfallSeries
    deficit: RainfallSeries
    constant_loss: RainfallSeries
    total_loss: RainfallSeries


@dataclass(slots=True)
class DeficitConstant:
    """
    Apply soil-moisture deficit and constant-rate losses to rainfall.

    :param rainfall_series: Incremental rainfall series in inches.
    :param max_deficit: Maximum soil-moisture deficit in inches.
    :param constant_loss: Constant loss rate applied to excess rainfall, in
        inches per hour.
    :param recovery_time: Duration for the deficit to recover to 99% of its
        maximum during dry periods.
    :param recover_deficit: Whether to recover the deficit during dry time steps.
    :param initial_deficit: Starting soil-moisture deficit in inches, clipped to
        the range from zero to ``max_deficit``.
    """

    rainfall_series: RainfallSeries  # incremental inch
    max_deficit: float  # inch
    constant_loss: float  # inch per hour
    recovery_time: timedelta | pandas.Timedelta | None = None
    recover_deficit: bool = True
    initial_deficit: float = 0  # inch

    def __post_init__(self):
        """Validate the initial model parameters."""
        self.update_params(
            self.max_deficit,
            self.constant_loss,
            self.recovery_time,
            self.recover_deficit,
            self.initial_deficit,
        )

    @staticmethod
    def validate_recovery_time(recovery_time: timedelta, recover_deficit: bool):
        """Require a positive recovery duration when recovery is enabled."""
        if recover_deficit:
            if recovery_time is None:
                raise ValueError(
                    "recovery_time must be specified when recover_deficit=True"
                )

            if recovery_time <= timedelta(minutes=0):
                raise ValueError("recovery_time must be > 0 when recover_deficit=True")

    def update_params(
        self,
        max_deficit: float | None = None,
        constant_loss: float | None = None,
        recovery_time: timedelta | None = None,
        recover_deficit: bool | None = None,
        initial_deficit: float | None = None,
    ):
        """Validate and update any supplied loss-model parameters."""
        if max_deficit is not None:
            validate_non_negative(max_deficit, "max_deficit")
            self.max_deficit = max_deficit

        if constant_loss is not None:
            validate_non_negative(constant_loss, "constant_loss")
            self.constant_loss = constant_loss

        if recover_deficit is not None:
            self.recover_deficit = recover_deficit

        if recovery_time is not None:
            self.validate_recovery_time(recovery_time, self.recover_deficit)
            self.recovery_time = recovery_time

        if initial_deficit is not None:
            validate_non_negative(initial_deficit, "initial_deficit")
            self.initial_deficit = initial_deficit

    def route(self) -> DeficitConstantOutput:
        """Return excess rainfall, deficit, and applied loss series."""
        rain = self.rainfall_series.data
        constant_loss = self.constant_loss

        time_step = get_time_step(rain.index)
        check_time_step(time_step)
        time_step_hours = time_step.total_seconds() / SECONDS_PER_HOUR
        constant_loss_per_time_step = constant_loss * time_step_hours

        full_recovery_fraction = 0.99

        if self.recover_deficit:
            recovery_time_hrs = self.recovery_time.total_seconds() / SECONDS_PER_HOUR

            e_folding_time = -recovery_time_hrs / numpy.log(
                1.0 - full_recovery_fraction
            )
            alpha = 1.0 - numpy.exp(-time_step_hours / e_folding_time)
        else:
            alpha = 0.0

        excess_precip_arr, deficit_arr, constant_loss_arr = _deficit_constant_kernel(
            rain_arr=self.rainfall_series.data.to_numpy(),
            initial_deficit=self.initial_deficit,
            max_deficit=self.max_deficit,
            alpha=alpha,
            constant_loss_per_ts=constant_loss_per_time_step,
            recover_deficit=self.recover_deficit,
        )

        excess = RainfallSeries(
            pandas.Series(excess_precip_arr, index=rain.index),
            sorted=True,
            _time_step=self.rainfall_series.time_step,
        )
        deficit = RainfallSeries(
            pandas.Series(deficit_arr, index=rain.index),
            sorted=True,
            _time_step=self.rainfall_series.time_step,
        )
        constant_loss = RainfallSeries(
            pandas.Series(constant_loss_arr, index=rain.index),
            sorted=True,
            _time_step=self.rainfall_series.time_step,
        )
        loss = RainfallSeries(
            rain - excess.data, sorted=True, _time_step=self.rainfall_series.time_step
        )

        return DeficitConstantOutput(
            excess=excess, deficit=deficit, constant_loss=constant_loss, total_loss=loss
        )


@jit(nopython=True, cache=True)
def _deficit_constant_kernel(
    rain_arr, initial_deficit, max_deficit, alpha, constant_loss_per_ts, recover_deficit
):
    """Calculate rainfall excess, soil deficit, and constant loss per step."""
    n = len(rain_arr)
    excess_precip_arr = numpy.zeros(n, dtype=numpy.float64)
    deficit_arr = numpy.zeros(n, dtype=numpy.float64)
    constant_loss_arr = numpy.zeros(n, dtype=numpy.float64)

    running_deficit = max(0.0, min(initial_deficit, max_deficit))

    for idx in range(n):
        curr_precip = max(rain_arr[idx], 0.0)

        # 1) Exponential recovery (Dry period)
        if recover_deficit and curr_precip < 1e-6:
            running_deficit += (max_deficit - running_deficit) * alpha

        # 2) Rain fills deficit
        filled_deficit = min(curr_precip, running_deficit)
        running_deficit -= filled_deficit
        excess_precip = curr_precip - filled_deficit

        # 3) Constant loss applied to excess
        applied_constant_loss = min(excess_precip, constant_loss_per_ts)
        excess_precip -= applied_constant_loss

        # Stability check & Storage
        running_deficit = max(0.0, min(running_deficit, max_deficit))

        constant_loss_arr[idx] = applied_constant_loss
        excess_precip_arr[idx] = max(excess_precip, 0.0)
        deficit_arr[idx] = running_deficit

    return excess_precip_arr, deficit_arr, constant_loss_arr

from dataclasses import dataclass
from datetime import timedelta

import numpy
import pandas

from pyrunoff.utils import validate_non_negative
from pyrunoff.utils.compat import jit
from pyrunoff.utils.constants import SECONDS_PER_HOUR
from pyrunoff.utils.time_series import RainfallSeries


@dataclass(slots=True)
class InitialConstantOutput:
    """Excess rainfall and loss series produced by the initial-constant model.

    :param excess: Rainfall remaining after losses are applied.
    :param total_loss: Rainfall removed by the model.
    """

    excess: RainfallSeries
    total_loss: RainfallSeries


@dataclass(slots=True)
class InitialConstant:
    """
    Apply an initial loss followed by a constant rainfall loss rate.

    :param rainfall_series: Incremental rainfall series in inches.
    :param initial_loss: Initial loss depth in inches.
    :param constant_loss: Constant loss rate in inches per hour.
    :param reset_initial_duration: Optional dry duration after which the initial
        loss is restored.
    """

    rainfall_series: RainfallSeries
    initial_loss: float  # inch
    constant_loss: float  # inch per hour
    reset_initial_duration: timedelta | pandas.Timedelta | None = None

    def __post_init__(self):
        """Validate loss parameters and the optional reset duration."""
        self.update_params(self.initial_loss, self.constant_loss)
        if (
            self.reset_initial_duration is not None
            and self.reset_initial_duration < self.rainfall_series.time_step
        ):
            raise ValueError(
                "Reset duration must be greater than or equal to time step "
                "of rainfall series."
            )

    def update_params(
        self, initial_loss: float | None = None, constant_loss: float | None = None
    ):
        """Validate and update any supplied loss parameters."""
        if initial_loss is not None:
            validate_non_negative(initial_loss, "initial_loss")
            self.initial_loss = initial_loss

        if constant_loss is not None:
            validate_non_negative(constant_loss, "constant_loss")
            self.constant_loss = constant_loss

    def route(self) -> InitialConstantOutput:
        """Separate rainfall into excess and total-loss series."""
        time_step = self.rainfall_series.time_step
        time_step_secs = time_step.total_seconds()
        time_step_hrs = time_step_secs / SECONDS_PER_HOUR

        constant_loss_per_step = self.constant_loss * time_step_hrs

        reset_duration_secs = (
            0.0
            if self.reset_initial_duration is None
            else self.reset_initial_duration.total_seconds()
        )
        loss_arr = _initial_constant_kernel(
            rain=self.rainfall_series.data.to_numpy(),
            time_step_secs=time_step_secs,
            reset_duration_secs=reset_duration_secs,
            initial_loss=self.initial_loss,
            constant_loss_per_step=constant_loss_per_step,
        )

        total_loss = pandas.Series(loss_arr, index=self.rainfall_series.data.index)
        excess = self.rainfall_series.data - total_loss

        return InitialConstantOutput(
            RainfallSeries(excess, sorted=True, _time_step=time_step),
            RainfallSeries(total_loss, sorted=True, _time_step=time_step),
        )


@jit(nopython=True, cache=True)
def _initial_constant_kernel(
    rain,
    time_step_secs,
    reset_duration_secs,
    initial_loss,
    constant_loss_per_step,
):
    """Calculate per-step rainfall losses for the initial-constant model."""
    curr_dry_duration = 0.0
    remaining_initial_loss = initial_loss

    loss_arr = numpy.zeros_like(rain)

    for i in range(rain.size):
        curr_rain = rain[i]
        if reset_duration_secs > 0:
            if curr_rain <= 1e-9:
                curr_dry_duration += time_step_secs
                if curr_dry_duration >= reset_duration_secs:
                    remaining_initial_loss = initial_loss
            else:
                curr_dry_duration = 0.0

        curr_initial_loss = min(remaining_initial_loss, curr_rain)
        leftover_rain = max(0.0, curr_rain - curr_initial_loss)

        remaining_initial_loss -= curr_initial_loss
        curr_constant_loss = min(leftover_rain, constant_loss_per_step)

        actual_loss = curr_constant_loss + curr_initial_loss
        loss_arr[i] = actual_loss

    return loss_arr

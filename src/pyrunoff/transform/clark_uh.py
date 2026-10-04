from dataclasses import dataclass
from datetime import timedelta

import numpy
import pandas

from pyrunoff.utils import DischargeSeries, RainfallSeries, validate_positive
from pyrunoff.utils.compat import jit
from pyrunoff.utils.constants import SECONDS_PER_HOUR
from pyrunoff.utils.tso import check_time_step, fix_ts_end


@dataclass(slots=True)
class ClarkUh:
    """
    Transform excess rainfall into discharge using a Clark unit hydrograph.

    :param rainfall_series: Incremental excess rainfall in inches.
    :param basin_area: Drainage area in acres.
    :param tc: Basin time of concentration.
    :param storage_r: Linear-reservoir storage coefficient.
    :param compute_time_step: Optional interval for resampling rainfall before
        routing. Defaults to one minute.
    :param resampled_data: Internal cache of rainfall resampled for routing.
    :param end: Optional final timestamp for the rainfall series.
    """

    rainfall_series: RainfallSeries
    basin_area: float
    tc: timedelta | pandas.Timedelta
    storage_r: timedelta | pandas.Timedelta
    compute_time_step: timedelta | pandas.Timedelta | None = timedelta(minutes=1)
    resampled_data: pandas.Series | None = None
    end: pandas.Timestamp = None

    def __post_init__(self):
        """Validate the basin parameters and prepare rainfall for routing."""
        validate_positive(self.basin_area, "basin_area")
        self._fix_end()
        self.update_params(self.tc, self.storage_r)
        self.validate_compute_time_step()
        self._resample_data()

    @staticmethod
    def validate_tc(tc: pandas.Timedelta, time_step: pandas.Timedelta) -> None:
        """Require a positive concentration time no shorter than the time step."""
        if tc.total_seconds() <= 0:
            raise ValueError("Time of Concentration (tc) must be greater than zero.")

        if time_step > tc:
            raise ValueError(
                f"Time step ({time_step}) cannot be larger than tc ({tc}). "
                "Decrease compute_time_step."
            )

    @staticmethod
    def validate_storage_r(
        storage_r: pandas.Timedelta, time_step: pandas.Timedelta
    ) -> None:
        """Require a positive storage coefficient within the stability limit."""
        if storage_r.total_seconds() <= 0:
            raise ValueError(
                "Storage Coefficient (storage_r) must be greater than zero."
            )

        ts_hr = time_step.total_seconds() / 3600.0
        r_hr = storage_r.total_seconds() / 3600.0
        if ts_hr >= 2 * r_hr:
            raise ValueError(
                f"Time step ({ts_hr} hr) is too large for R={r_hr} hr. "
                "Must satisfy dt < 2R for numerical stability."
            )

    def update_params(
        self,
        tc: pandas.Timedelta | None = None,
        storage_r: pandas.Timedelta | None = None,
    ):
        """Validate and update any supplied routing parameters."""
        time_step = self.compute_time_step or self.rainfall_series.time_step
        if tc is not None:
            self.validate_tc(tc, time_step)
            self.tc = pandas.to_timedelta(tc)

        if storage_r is not None:
            self.validate_storage_r(storage_r, time_step)
            self.storage_r = pandas.to_timedelta(storage_r)

    def validate_compute_time_step(
        self,
    ) -> None:
        """Validate the optional rainfall resampling interval."""
        if self.compute_time_step is not None:
            self.compute_time_step = pandas.Timedelta(self.compute_time_step)
            if (
                self.rainfall_series.time_step > self.compute_time_step
                and self.compute_time_step <= pandas.Timedelta(minutes=0)
            ):
                raise ValueError("'compute_time_step' must be greater than 1 minute.")

            if self.rainfall_series.time_step.value % self.compute_time_step.value != 0:
                raise ValueError(
                    f"original time step ({self.rainfall_series.time_step}) must be a "
                    f"multiple of compute_time_step ({self.compute_time_step})."
                )
            check_time_step(self.compute_time_step, name="compute_time_step")

    def _resample_data(self):
        """Cache rainfall resampled to the configured computation interval."""
        if self.compute_time_step is not None:
            self.resampled_data = self.rainfall_series.resample_at(
                self.compute_time_step
            )

    def _fix_end(self):
        """Extend or truncate rainfall data to the configured end time."""
        if self.end is not None:
            original_data = self.rainfall_series.data
            fixed_data = fix_ts_end(
                original_data, self.end, self.rainfall_series.time_step
            )
            new_rainfall_series = RainfallSeries(
                fixed_data, sorted=True, _time_step=self.rainfall_series.time_step
            )
            self.rainfall_series = new_rainfall_series

    @staticmethod
    def get_cumulative_area_fraction(time_step: pandas.Timedelta, tc: pandas.Timedelta):
        """Calculate the cumulative basin area contributing by each time step."""
        time_step_mins = int(time_step.total_seconds() // 60)

        tc_mins = int(tc.total_seconds() // 60.0)
        half_tc_mins = tc_mins // 2

        num_steps = tc_mins // time_step_mins
        all_times = numpy.arange(num_steps + 1) * time_step_mins
        fractions = all_times / tc_mins
        mask = all_times <= half_tc_mins

        sqrt_2 = numpy.sqrt(2)

        cumulative_area_fraction = numpy.zeros_like(all_times, dtype=float)
        cumulative_area_fraction[mask] = sqrt_2 * numpy.power(fractions[mask], 1.5)
        cumulative_area_fraction[~mask] = 1.0 - sqrt_2 * numpy.power(
            numpy.clip(1.0 - fractions[~mask], 0, 1), 1.5
        )
        cumulative_area_fraction[-1] = 1.0
        return cumulative_area_fraction

    @staticmethod
    def get_uh(
        basin_area: float,
        tc: pandas.Timedelta,
        storage_r: pandas.Timedelta,
        time_step: pandas.Timedelta,
    ):
        """Calculate Clark unit-hydrograph ordinates for the basin parameters."""
        time_step_secs = time_step.total_seconds()

        cumulative_area_fraction = ClarkUh.get_cumulative_area_fraction(time_step, tc)
        num_steps = cumulative_area_fraction.size

        total_volume = basin_area * 43560 / 12  # cubic feet
        cumulative_volume = cumulative_area_fraction * total_volume
        translation_inflow = (
            numpy.diff(cumulative_volume, prepend=cumulative_volume[0]) / time_step_secs
        )

        storage_r_hr = storage_r.total_seconds() / SECONDS_PER_HOUR
        time_step_hr = time_step_secs / SECONDS_PER_HOUR
        max_steps = int(num_steps + (30 * storage_r_hr / time_step_hr) + 1000)

        inflow_weight = time_step_hr / (storage_r_hr + 0.5 * time_step_hr)  # c1
        outflow_weight = 1 - inflow_weight  # c2

        uh_ordinates, running_volume = _clark_reservoir_kernel(
            translation_inflow=translation_inflow,
            inflow_weight=float(inflow_weight),
            outflow_weight=float(outflow_weight),
            time_step_secs=float(time_step_secs),
            total_volume=total_volume,
            max_steps=max_steps,
        )

        captured_volume_fraction = running_volume / total_volume

        return uh_ordinates / captured_volume_fraction

    def route(
        self,
        return_at_computed_timestep: bool = False,
    ) -> DischargeSeries:
        """
        Route excess rainfall through the Clark unit hydrograph.

        :param return_at_computed_timestep: If true, return values at the
            computation interval; otherwise, use the original rainfall interval.
        :return: Routed discharge series.
        """

        rain_data = (
            self.resampled_data
            if self.resampled_data is not None
            else self.rainfall_series.data
        )

        time_step = (
            self.compute_time_step
            if self.compute_time_step is not None
            else self.rainfall_series.time_step
        )
        tc = self.tc
        check_time_step(time_step, tc)

        uh_ordinates = self.get_uh(self.basin_area, tc, self.storage_r, time_step)

        basin_outflow = numpy.convolve(rain_data.to_numpy(), uh_ordinates, mode="full")

        basin_outflow = basin_outflow[: rain_data.size]
        basin_outflow = pandas.Series(basin_outflow, index=rain_data.index)

        if self.compute_time_step and not return_at_computed_timestep:
            original_index = self.rainfall_series.data.index
            basin_outflow = basin_outflow.reindex(original_index)

        result_time_step = (
            self.compute_time_step
            if (self.compute_time_step and return_at_computed_timestep)
            else self.rainfall_series.time_step
        )

        discharge_series = DischargeSeries(
            basin_outflow, sorted=True, _time_step=result_time_step
        )

        return discharge_series


@jit(nopython=True, cache=True)
def _clark_reservoir_kernel(
    translation_inflow,
    inflow_weight,
    outflow_weight,
    time_step_secs,
    total_volume,
    max_steps,
):
    """Route translation inflow through the linear-reservoir component."""
    idx = 0
    running_volume = 0.0
    previous_uh_outflow = 0.0
    uh_ordinates = numpy.zeros(max_steps)

    num_inflow = len(translation_inflow)

    while running_volume / total_volume < 0.999 or idx < num_inflow:
        if idx >= max_steps:
            break

        inflow = translation_inflow[idx] if idx < num_inflow else 0.0
        uh_outflow = (inflow_weight * inflow) + (outflow_weight * previous_uh_outflow)
        uh_ordinates[idx] = uh_outflow

        running_volume += 0.5 * (uh_outflow + previous_uh_outflow) * time_step_secs
        previous_uh_outflow = uh_outflow
        idx += 1

    return uh_ordinates[:idx], running_volume


if __name__ == "__main__":
    hms_data = pandas.read_csv(
        "../../../tests/test_data/clarks_uh/hms_scs2_5in.txt",
        sep="\t",
        header=None,
        na_values=["", " "],
    )
    hms_data.columns = [
        "date",
        "time",
        "precip_in",
        "precip_loss_in",
        "precip_excess_in",
        "direct_flow_cfs",
        "baseflow_cfs",
        "total_flow_cfs",
    ]
    hms_data["timestamp"] = pandas.to_datetime(
        hms_data["date"] + "T" + hms_data["time"], format="%d%b%YT%H:%M"
    )
    hms_data = hms_data.set_index("timestamp")

    clark_uh = ClarkUh(
        rainfall_series=RainfallSeries(hms_data["precip_excess_in"].bfill()),
        basin_area=1280,
        tc=timedelta(hours=2),
        storage_r=timedelta(hours=3),
    )

    results = clark_uh.route()
    py_flow = results.data.round(2)

    print({"max": py_flow.max(), "idx_max": py_flow.idxmax()})

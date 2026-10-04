from dataclasses import dataclass

import numpy
import pandas

from pyrunoff.utils import DischargeSeries, ElevationSeries, StorageSeries
from pyrunoff.utils.compat import jit
from pyrunoff.utils.tso import fix_ts_end

from .reservoir_utils import ElevationStorage, StorageDischarge


@dataclass(slots=True)
class LevelPoolOutput:
    """Typed output series produced by level-pool routing.

    :param outflows: Discharge series for each named outlet.
    :param total_outflow: Sum of discharge across all outlets.
    :param storage: Reservoir storage at each time step.
    :param elevation: Reservoir elevation corresponding to each storage value.
    """

    outflows: dict[str, DischargeSeries]
    total_outflow: DischargeSeries
    storage: StorageSeries
    elevation: ElevationSeries


@dataclass(slots=True)
class LevelPool:
    """Route inflow through a reservoir with one or more outlet curves.

    :param inflow_series: Discharge entering the reservoir.
    :param elevation_storage: Elevation-storage rating curve.
    :param storage_discharges: Named storage-discharge curves, one per outlet/spillway.
    :param initial_storage: Starting reservoir storage in the curve's units.
    :param end: Optional final timestamp for the inflow series.
    :param storage_adjustment_factor: Factor converting storage to the units
        used by the routing calculations.
    """

    inflow_series: DischargeSeries
    elevation_storage: ElevationStorage
    storage_discharges: dict[str, StorageDischarge]
    initial_storage: float
    end: pandas.Timestamp | None = None
    storage_adjustment_factor: float = 43_560

    def __post_init__(self):
        """Validate the storage conversion factor and apply the optional end."""
        if self.storage_adjustment_factor <= 0:
            raise ValueError("storage_adjustment_factor must be greater than zero.")
        self._fix_end()

    def _fix_end(self):
        """Extend or truncate inflow data to the configured end time."""
        if self.end is not None:
            self.inflow_series.data = fix_ts_end(
                ts=self.inflow_series.data,
                end=self.end,
                time_step=self.inflow_series.time_step,
            )

    def route(self) -> LevelPoolOutput:
        """Route inflow and return outlet flows, total flow, storage, and elevation.

        :return: Typed output series aligned to the original inflow index.
        """
        inflow = self.inflow_series.data
        time_step = self.inflow_series.time_step
        original_index = inflow.index
        names = list(self.storage_discharges)
        if not names:
            raise ValueError("At least one storage-discharge curve is required.")
        if inflow.empty:
            outflow = pandas.DataFrame(index=inflow.index, columns=names, dtype=float)
            storage = pandas.Series(index=inflow.index, name="storage", dtype=float)
        else:
            is_original_index = time_step <= pandas.Timedelta(minutes=1)
            if not is_original_index:
                minute_index = pandas.date_range(
                    start=inflow.index[0], end=inflow.index[-1], freq="min"
                )
                inflow = inflow.reindex(minute_index).interpolate(method="time")

            curves, curve_lengths = _pack_storage_discharge_curves(
                [self.storage_discharges[name] for name in names],
                self.storage_adjustment_factor,
            )
            outflow_arr, storage_arr = _level_pool_kernel(
                inflow.to_numpy(dtype=numpy.float64),
                float(self.initial_storage) * self.storage_adjustment_factor,
                curves,
                curve_lengths,
                float(min(time_step, pandas.Timedelta(minutes=1)).total_seconds()),
            )
            outflow = pandas.DataFrame(outflow_arr, index=inflow.index, columns=names)
            storage = pandas.Series(
                storage_arr, index=inflow.index, name="storage"
            ) / self.storage_adjustment_factor
            if not is_original_index:
                outflow = outflow.loc[original_index]
                storage = storage.loc[original_index]

        outflows = {
            name: DischargeSeries(outflow[name], sorted=True, _time_step=time_step)
            for name in names
        }
        total_outflow = DischargeSeries(
            outflow.sum(axis=1).rename("total_outflow"),
            sorted=True,
            _time_step=time_step,
        )
        storage_series = StorageSeries(storage, sorted=True, _time_step=time_step)
        elevation_values = numpy.interp(
            storage.to_numpy(dtype=numpy.float64),
            self.elevation_storage.storage.to_numpy(dtype=numpy.float64),
            self.elevation_storage.elevation.to_numpy(dtype=numpy.float64),
        )
        elevation = ElevationSeries(
            pandas.Series(elevation_values, index=storage.index, name="elevation"),
            sorted=True,
            _time_step=time_step,
        )

        return LevelPoolOutput(
            outflows=outflows,
            total_outflow=total_outflow,
            storage=storage_series,
            elevation=elevation,
        )


def _pack_storage_discharge_curves(
    storage_discharges: list[StorageDischarge], storage_adjustment_factor: float
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Pack rating curves of different lengths into arrays for Numba routing."""
    lengths = numpy.asarray(
        [len(curve.storage) for curve in storage_discharges], dtype=numpy.int64
    )
    if numpy.any(lengths < 2):
        raise ValueError(
            "Each storage-discharge curve must contain at least two points."
        )

    curves = numpy.empty(
        (len(storage_discharges), 2, int(lengths.max())), dtype=numpy.float64
    )
    for index, curve in enumerate(storage_discharges):
        length = lengths[index]
        curves[index, 0, :length] = (
            curve.storage.to_numpy(dtype=numpy.float64) * storage_adjustment_factor
        )
        curves[index, 1, :length] = curve.discharge.to_numpy(dtype=numpy.float64)
    return curves, lengths


@jit(nopython=True, cache=True)
def _level_pool_kernel(
    inflow_arr, initial_storage, storage_discharge_arr, curve_lengths, time_step_seconds
):
    """Calculate per-outlet outflows and reservoir storage at each time step."""
    n_times = len(inflow_arr)
    n_outlets = storage_discharge_arr.shape[0]
    outflow_arr = numpy.zeros((n_times, n_outlets), dtype=numpy.float64)
    storage_arr = numpy.zeros(n_times, dtype=numpy.float64)
    if n_times == 0:
        return outflow_arr, storage_arr

    storage_arr[0] = initial_storage
    previous_total_outflow = 0.0
    for outlet in range(n_outlets):
        discharge = _interpolate_discharge(
            initial_storage, storage_discharge_arr[outlet], curve_lengths[outlet]
        )
        outflow_arr[0, outlet] = discharge
        previous_total_outflow += discharge

    for time_index in range(1, n_times):
        current_storage = (
            storage_arr[time_index - 1]
            + (inflow_arr[time_index] - previous_total_outflow) * time_step_seconds
        )
        current_storage = max(current_storage, 0.0)

        storage_arr[time_index] = current_storage
        previous_total_outflow = 0.0
        for outlet in range(n_outlets):
            discharge = _interpolate_discharge(
                current_storage, storage_discharge_arr[outlet], curve_lengths[outlet]
            )
            outflow_arr[time_index, outlet] = discharge
            previous_total_outflow += discharge
    return outflow_arr, storage_arr


@jit(nopython=True, cache=True)
def _interpolate_discharge(storage, curve, length):
    """Interpolate discharge and hold endpoint values outside the curve range."""
    if storage <= curve[0, 0]:
        return curve[1, 0]
    if storage >= curve[0, length - 1]:
        return curve[1, length - 1]
    for index in range(1, length):
        upper_storage = curve[0, index]
        if storage <= upper_storage:
            lower_storage = curve[0, index - 1]
            fraction = (storage - lower_storage) / (upper_storage - lower_storage)
            return curve[1, index - 1] + fraction * (
                curve[1, index] - curve[1, index - 1]
            )
    return curve[1, length - 1]

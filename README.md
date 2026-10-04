# pyrunoff

`pyrunoff` is a Python library for hydrologic runoff and routing calculations. It exposes reusable routing components for rainfall losses, unit hydrograph transformation, baseflow, reach routing, and reservoir routing.

## Installation

We recommend installing with `uv` for a fast, reproducible Python environment.

### Install from GitHub

```bash
uv add "git+https://github.com/darshanbaral/pyrunoff.git"
```

This records the dependency in the project configuration and installs it into the active `uv` environment.

The package requires Python 3.14 or later.

To install the optional Numba extra from a local checkout, run:

```bash
uv sync --extra with-numba
```

The numerical kernels can use Numba to compile hot loops for faster execution on large time series. Without Numba, the library still works but runs those kernels in Python mode and may be slower for long simulations.

## Core API

The library provides:

- `RainfallSeries`, `DischargeSeries`, `ElevationSeries`, and `StorageSeries` for time-indexed data
- `InitialConstant` and `DeficitConstant` loss models, with typed output dataclasses
- `ClarkUh` for Clark unit-hydrograph transformation
- `LinearReservoir` for baseflow routing
- `Muskingum` for channel routing
- `LevelPool` for reservoir routing, returning `LevelPoolOutput`
- `nse` and `kge` model-performance metrics in `pyrunoff.metrics`
- `Network` for topological ordering of flow paths in `pyrunoff.utils.network`

The library is designed as a modular toolkit for composing hydrologic routing operations in Python.

## Quick start

```python
import pandas as pd

from pyrunoff.baseflow import LinearReservoir
from pyrunoff.loss import InitialConstant
from pyrunoff.reach import Muskingum
from pyrunoff.transform import ClarkUh
from pyrunoff.utils import RainfallSeries

# Example rainfall data
rain = pd.Series(
    [0.0, 0.8, 1.4, 0.4, 0.0, 0.0],
    index=pd.date_range("2024-01-01", periods=6, freq="1h"),
)

rain_series = RainfallSeries(rain, sorted=True)

# 1) Compute losses
loss_model = InitialConstant(
    rainfall_series=rain_series,
    initial_loss=0.25,
    constant_loss=0.15,
)
loss_output = loss_model.route()

# 2) Transform excess precipitation into direct runoff
clark = ClarkUh(
    rainfall_series=loss_output.excess,
    basin_area=1000.0,
    tc=pd.Timedelta(hours=2),
    storage_r=pd.Timedelta(hours=3),
)
direct_runoff = clark.route()

# 3) Generate baseflow
baseflow = LinearReservoir(
    rainfall_series=loss_output.total_loss,
    gw_fraction=0.7,
    k=pd.Timedelta(hours=6),
    basin_area=1000.0,
).route()

# 4) Route channel flow with Muskingum
routed_flow = Muskingum(
    inflow_series=direct_runoff,
    k=pd.Timedelta(hours=2),
    x=0.25,
).route()

print(direct_runoff.data)
print(baseflow.data)
print(routed_flow.data)
```

## Time series utilities

The library relies on pandas time series with a `DatetimeIndex` and a consistent time step.

```python
import pandas as pd

from pyrunoff.utils import RainfallSeries

series = RainfallSeries(
    pd.Series([0.0, 0.5, 1.0], index=pd.date_range("2024-01-01", periods=3, freq="1h")),
    sorted=True,
)

print(series.time_step)
```

Time series normally infer their interval from at least two uniformly spaced timestamps. Intervals must be whole minutes. Routing methods may impose additional time-step constraints. `RainfallSeries` rejects negative rainfall, while `DischargeSeries`, `ElevationSeries`, and `StorageSeries` identify flow, elevation, and storage data, respectively.

### Network ordering

`Network` derives a topological execution order and direct upstream-node mapping from ordered flow paths. By default, it rejects paths that split from a node to multiple downstream nodes.

```python
from pyrunoff.utils.network import Network

network = Network(paths=[["A", "B"], ["C", "B"]])
print(network.sorted_nodes)
print(network.upstream_nodes["B"])
```

## Loss methods

### `InitialConstant`

```python
from pyrunoff.loss import InitialConstant

loss = InitialConstant(
    rainfall_series=rain_series,
    initial_loss=0.2,
    constant_loss=0.25,
)
loss_output = loss.route()
```

This method calculates both total losses and excess rainfall.
The returned `InitialConstantOutput` contains `excess` and `total_loss` rainfall series.

### `DeficitConstant`

```python
from pyrunoff.loss import DeficitConstant

loss = DeficitConstant(
    rainfall_series=rain_series,
    max_deficit=0.2,
    constant_loss=0.1,
    recovery_time=pd.Timedelta(hours=12),
    initial_deficit=0.2,
)
loss_output = loss.route()
```

`DeficitConstantOutput` contains `excess`, `deficit`, `constant_loss`, and `total_loss` series. `max_deficit` is the maximum soil-moisture deficit; `initial_deficit` defaults to zero.

## Transformation methods

### `ClarkUh`

```python
from pyrunoff.transform import ClarkUh

uh = ClarkUh(
    rainfall_series=loss_output.excess,
    basin_area=6236.6,
    tc=pd.Timedelta(hours=2.5),
    storage_r=pd.Timedelta(hours=7.5),
)
runoff = uh.route()
```

`tc` is the time of concentration and `storage_r` is the Clark storage coefficient; both are expressed as timedeltas. Rainfall is resampled to a one-minute computation interval by default. `route(return_at_computed_timestep=True)` returns at that interval instead of the original rainfall interval.

## Baseflow methods

### `LinearReservoir`

```python
from pyrunoff.baseflow import LinearReservoir

baseflow = LinearReservoir(
    rainfall_series=loss_output.total_loss,
    gw_fraction=0.7,
    k=pd.Timedelta(hours=12),
    basin_area=6236.6,
).route()
```

`gw_fraction` must be between 0 and 1, and `k` is the recession coefficient as a timedelta.

## Reach routing

### `Muskingum`

```python
import pandas as pd

from pyrunoff.reach import Muskingum
from pyrunoff.utils import DischargeSeries

inflow = DischargeSeries(
    pd.Series(
        [0.0, 5.0, 15.0, 8.0], index=pd.date_range("2024-01-01", periods=4, freq="1h")
    ),
    sorted=True,
)
routed = Muskingum(
    inflow_series=inflow,
    k=pd.Timedelta(hours=2),
    x=0.25,
).route()
```

The `x` parameter must be between 0 and 0.5. `k` is the storage time constant and must satisfy the model's time-step stability condition.

## Performance metrics

`nse` returns a Nash–Sutcliffe Efficiency score. `kge` returns a `KgeResult` containing the 2012 Kling–Gupta Efficiency score (`kge`) and its components (`corr_coef`, `gamma`, and `beta`). Both functions align the input series by index and omit pairs with missing values.

```python
import pandas as pd

from pyrunoff.metrics import kge, nse

observed = pd.Series([2.0, 4.0, 6.0], index=pd.date_range("2024-01-01", periods=3, freq="1h"))
simulated = pd.Series([2.5, 3.5, 6.5], index=observed.index)

nse_score = nse(observed, simulated)
kge_result = kge(observed, simulated)
print(nse_score, kge_result.kge, kge_result.corr_coef)
```

Set `use_low_flow_transform=True` to emphasize low-flow performance. When enabled, `low_flow_percentile` sets the transform scale and must be between 0 and 100.

## Reservoir routing

### `LevelPool`

```python
import pandas as pd

from pyrunoff.reservoir import ElevationStorage, LevelPool, StorageDischarge
from pyrunoff.utils import DischargeSeries

inflow = DischargeSeries(
    pd.Series(
        [0.0, 5.0, 15.0, 8.0], index=pd.date_range("2024-01-01", periods=4, freq="1h")
    ),
    sorted=True,
)

storage_curve = ElevationStorage(
    # Elevation is in feet and storage is in acre-feet.
    elevation=pd.Series([650.0, 651.0, 652.0]),
    storage=pd.Series([0.0, 80.0, 200.0]),
)

rating_curve = StorageDischarge(
    # Storage is in acre-feet and discharge is in cubic feet per second.
    storage=pd.Series([0.0, 80.0, 200.0]),
    discharge=pd.Series([0.0, 50.0, 150.0]),
)

reservoir = LevelPool(
    inflow_series=inflow,
    elevation_storage=storage_curve,
    storage_discharges={"outlet": rating_curve},
    initial_storage=0.0,
)
result = reservoir.route()
outlet_flow = result.outflows["outlet"]
total_outflow = result.total_outflow
storage = result.storage
elevation = result.elevation
```

`LevelPool.route()` returns a `LevelPoolOutput`. Its `outflows` field maps each
outlet name to a `DischargeSeries`; `total_outflow`, `storage`, and `elevation`
are `DischargeSeries`, `StorageSeries`, and `ElevationSeries` values,
respectively. Access the underlying pandas series through each value's `.data`
attribute. Inflow and discharge are in cubic feet per second, storage is in
acre-feet, and elevation is in feet.

## Notes

- The package expects regular `DatetimeIndex` data.
- Time steps must be whole minutes and are validated automatically.
- If Numba is not installed, the numeric kernels run in Python mode with a warning. Install `numba` for the accelerated implementation.

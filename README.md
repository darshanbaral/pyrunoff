# pyrunoff

`pyrunoff` is a Python library for hydrologic runoff and routing calculations. It exposes reusable routing components for rainfall losses, unit hydrograph transformation, baseflow, reach routing, and reservoir routing.

## Installation

We recommend installing with `uv` for a fast, reproducible Python environment.

The current PyPI release is `0.1.0a1`, an alpha release, and requires Python 3.14 or later. `pip` commands include `--pre` to select this prerelease.

### Recommended: Install with Numba acceleration

```bash
uv add "pyrunoff[with-numba]"
# or
python -m pip install --pre "pyrunoff[with-numba]"
```

Numba compiles numerical kernels for faster execution on long time series.

### Install without Numba

```bash
uv add pyrunoff
# or
python -m pip install --pre pyrunoff
```

Without Numba, the library still works but runs its numerical kernels in Python mode, which may be slower for long simulations.

## Core API

The library provides:

- `RainfallSeries`, `DischargeSeries`, `ElevationSeries`, and `StorageSeries` for time-indexed data
- `InitialConstant` and `DeficitConstant` loss models, with typed output dataclasses
- `ClarkUh` for Clark unit-hydrograph transformation
- `LinearReservoir` for baseflow routing
- `Muskingum` for channel routing, returning `MuskingumOutput`
- `LevelPool` for reservoir routing, returning `LevelPoolOutput`
- `nse` and `kge` model-performance metrics in `pyrunoff.metrics`
- `Network` for topological ordering of flow paths in `pyrunoff.utils.network`
- `Model` for routing rainfall through a network of basins, reaches, junctions and reservoirs, with `load_model`, `load_params` and `load_rainfall` for reading its inputs from files

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
routed_flow = (
    Muskingum(
        inflow_series=direct_runoff,
        k=pd.Timedelta(hours=2),
        x=0.25,
    )
    .route()
    .total_outflow
)

print(direct_runoff.data)
print(baseflow.data)
print(routed_flow.data)
```

## Network models

A `Model` links basins, reaches, reservoirs and junctions into a network and routes rainfall through it in upstream-to-downstream order. Three files describe a run: the model, its parameters and the rainfall.

### Element types, processes and methods

The model and parameter files use three terms:

- **Element type**: what kind of node an element is in the network. A `basin` turns rainfall into flow, a `reach` routes flow along a channel, a `reservoir` routes flow through storage, a `junction` adds the flows that enter it, and the `sink` is the outlet where the network ends.
- **Process**: one hydrologic step an element computes. A basin has three processes, applied in order: `loss` (rainfall that does not run off), `transform` (excess rainfall to direct runoff) and `baseflow` (lost rainfall that returns as slow flow). Reaches and reservoirs have one, `routing`. Junctions and sinks have none.
- **Method**: the calculation used for a process, such as `initial_constant` or `deficit_constant` for loss. Each method is a class in the library (`InitialConstant`, `DeficitConstant`), and its parameters are that class's fields.

So each element has a type, its type fixes which processes it has, and the model file picks one method per process. The available methods are:

| Element type | Process     | Methods                                  |
| ------------ | ----------- | ---------------------------------------- |
| `basin`      | `loss`      | `initial_constant`, `deficit_constant`   |
| `basin`      | `transform` | `clark_uh`                               |
| `basin`      | `baseflow`  | `linear_reservoir`                       |
| `reach`      | `routing`   | `muskingum`                              |
| `reservoir`  | `routing`   | `level_pool`                             |
| `junction`   | —           | —                                        |
| `sink`       | —           | —                                        |

### Model file

The model file lists each element, the element it drains to, and the method it uses for each process. Methods set in `defaults` apply to every element of that type unless the element sets its own. Exactly one element, the sink, has no `downstream`.

```json
{
  "defaults": {
    "basin": {"loss": "initial_constant", "transform": "clark_uh", "baseflow": "linear_reservoir"},
    "reach": {"routing": "muskingum"}
  },
  "elements": {
    "B1": {"type": "basin", "downstream": "R1"},
    "B2": {"type": "basin", "downstream": "J1", "loss": "deficit_constant"},
    "R1": {"type": "reach", "downstream": "J1"},
    "J1": {"type": "junction", "downstream": "S"},
    "S": {"type": "sink"}
  }
}
```

### Parameter file

Parameters are grouped by element type, element, process and method. Each value is an object with a `value` and, for durations, `units` (`seconds`, `minutes` or `hours`). A process may hold parameters for several methods; only the method chosen in the model file is used, so you can switch methods without editing this file. `basin_area` (acres) is given once per basin and passed to every method that needs it.

```json
{
  "basins": {
    "B1": {
      "basin_area": {"value": 640},
      "loss": {"initial_constant": {"initial_loss": {"value": 0.5}, "constant_loss": {"value": 0.1}}},
      "transform": {"clark_uh": {"tc": {"value": 1, "units": "hours"}, "storage_r": {"value": 2, "units": "hours"}}},
      "baseflow": {"linear_reservoir": {"gw_fraction": {"value": 0.2}, "k": {"value": 24, "units": "hours"}}}
    },
    "B2": {
      "basin_area": {"value": 320},
      "loss": {
        "deficit_constant": {
          "max_deficit": {"value": 0.4},
          "constant_loss": {"value": 0.1},
          "recovery_time": {"value": 48, "units": "hours"}
        }
      },
      "transform": {"clark_uh": {"tc": {"value": 45, "units": "minutes"}, "storage_r": {"value": 90, "units": "minutes"}}},
      "baseflow": {"linear_reservoir": {"gw_fraction": {"value": 0.2}, "k": {"value": 24, "units": "hours"}}}
    }
  },
  "reaches": {
    "R1": {"routing": {"muskingum": {"k": {"value": 30, "units": "minutes"}, "x": {"value": 0.2}}}}
  }
}
```

Reservoir rating curves are read from CSV files, with paths relative to the parameter file:

```json
"reservoirs": {
  "RES": {
    "routing": {
      "level_pool": {
        "elevation_storage": {"path": "elevation_storage.csv", "elevation_column_name": "Stage_ft", "storage_column_name": "Storage_ac-ft"},
        "storage_discharges": {
          "spillway": {"path": "spillway.csv", "storage_column_name": "Storage", "discharge_column_name": "Discharge"}
        },
        "initial_storage": {"value": 0.0}
      }
    }
  }
}
```

Unknown methods or parameter names, and missing parameters for a chosen method, raise a `ValueError` that names the element.

### Rainfall file

Rainfall is a CSV with a timestamp column and one column of incremental depth (inches) per basin, named after the basin. Other columns are ignored.

```text
timestamp,B1,B2
2024-01-01 00:15,0.10,0.05
2024-01-01 00:30,0.25,0.12
```

### Running the model

```python
import pandas as pd

from pyrunoff.utils.params import load_model, load_params, load_rainfall

model = load_model("model.json")
params = load_params("model_params.json", model)
rainfall = load_rainfall(
    "rainfall.csv",
    model,
    time_column="timestamp",
    date_format="%Y-%m-%d %H:%M",
    end_offset=pd.Timedelta(hours=24),  # 24 h of zero rainfall so hydrographs recede
)

results = model.route(rainfall, **params)
print(results["S"].input.data)  # flow reaching the sink
```

`load_rainfall` reads only the model's basins. Use `end` for an absolute end time, or `end_offset` for an end relative to the last timestamp in the file (negative values drop the last values). `date_format` is inferred when omitted.

`model.route` returns results keyed by element name: a `BasinOutput` (`excess`, `total_loss`, `direct_runoff`, `baseflow`, `total_outflow`) for each basin, a `ReachOutput` or `ReservoirOutput` (`input`, `routing_output`, `total_outflow`) for each reach or reservoir, a `JunctionOutput` for each junction, and a `SinkOutput` (`input`) for the sink.

Rainfall already in a DataFrame (one column per basin) can be converted with `RainfallSeries.dict_from_dataframe`:

```python
rainfall = RainfallSeries.dict_from_dataframe(df, columns=["B1", "B2"])
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

Rainfall values are incremental depths in inches: the value at time `t` is the depth that fell during the interval ending at `t`. The first value therefore covers the interval before the first timestamp.

`fix_end(end)` returns a copy of a series ending at `end`, dropping later values or adding zeros up to `end` at the series' time step. The routing methods accept the same `end` argument.

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

`DeficitConstantOutput` contains `excess`, `deficit`, `constant_loss`, and `total_loss` series. `max_deficit` is the maximum soil-moisture deficit; `initial_deficit` defaults to zero. The deficit recovers during dry periods by default, which requires `recovery_time`; pass `recover_deficit=False` to turn recovery off.

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

`tc` is the time of concentration and `storage_r` is the Clark storage coefficient; both are expressed as timedeltas. Rainfall is resampled to a one-minute computation interval by default, keeping its total depth. `route(return_at_computed_timestep=True)` returns at that interval instead of the original rainfall interval; because the first rainfall value covers the interval before the first timestamp, this output starts inside that interval (14 minutes before the first timestamp for 15-minute rainfall).

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
routed = (
    Muskingum(
        inflow_series=inflow,
        k=pd.Timedelta(hours=2),
        x=0.25,
    )
    .route()
    .total_outflow
)
```

The `x` parameter must be between 0 and 0.5. `k` is the storage time constant and must satisfy the model's time-step stability condition.

## Performance metrics

`nse` returns a Nash–Sutcliffe Efficiency score. `kge` returns a `KgeResult` containing the 2012 Kling–Gupta Efficiency score (`kge`) and its components (`corr_coef`, `gamma`, and `beta`). Both functions align the input series by index and omit pairs with missing values.

```python
import pandas as pd

from pyrunoff.metrics import kge, nse

observed = pd.Series(
    [2.0, 4.0, 6.0], index=pd.date_range("2024-01-01", periods=3, freq="1h")
)
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

# pyrunoff

`pyrunoff` is a Python library for hydrologic runoff and routing calculations. It exposes reusable routing components for rainfall losses, unit hydrograph transformation, baseflow, reach routing, and reservoir routing.

Documentation: <https://darshanbaral.github.io/pyrunoff/>

<!-- docs-include-start -->

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

<!-- docs-include-end -->

## Documentation

The [documentation](https://darshanbaral.github.io/pyrunoff/) covers:

- [Network models](https://darshanbaral.github.io/pyrunoff/network_models.html): the model, parameter and rainfall files, and running a model
- [Methods](https://darshanbaral.github.io/pyrunoff/methods.html): using each loss, transform, baseflow and routing method on its own, and the performance metrics
- [Time series](https://darshanbaral.github.io/pyrunoff/time_series.html): time series types, time steps and network ordering
- [API reference](https://darshanbaral.github.io/pyrunoff/api/index.html): every class and function, with parameter units and constraints

# Methods

Each method is a dataclass whose fields are its parameters; `route()` runs it.
The examples below use the methods on their own. In a network model they are
chosen in the model file and supplied with their inputs at run time (see
{doc}`network_models`). Units, defaults and constraints for each parameter are
in the {doc}`api/index`.

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

Returns an {class}`~pyrunoff.loss.initial_and_constant.InitialConstantOutput`
with `excess` and `total_loss` rainfall series.

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

Returns a {class}`~pyrunoff.loss.deficit_and_constant.DeficitConstantOutput`
with `excess`, `deficit`, `constant_loss` and `total_loss` series. The deficit
recovers during dry periods by default, which requires `recovery_time`; pass
`recover_deficit=False` to turn recovery off.

## Transform methods

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

`tc` is the time of concentration and `storage_r` the Clark storage
coefficient. See {meth}`~pyrunoff.transform.clark_uh.ClarkUh.route` for
returning results at the one-minute computation interval.

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

Returns a {class}`~pyrunoff.reservoir.level_pool.LevelPoolOutput`. Its
`outflows` field maps each outlet name to a `DischargeSeries`; `total_outflow`,
`storage` and `elevation` are `DischargeSeries`, `StorageSeries` and
`ElevationSeries` values. Access the underlying pandas series through each
value's `.data` attribute.

## Performance metrics

{func}`~pyrunoff.metrics.nse` returns a Nash–Sutcliffe Efficiency score.
{func}`~pyrunoff.metrics.kge` returns a {class}`~pyrunoff.metrics.KgeResult`
holding the 2012 Kling–Gupta Efficiency score (`kge`) and its components
(`corr_coef`, `gamma` and `beta`).

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

Set `use_low_flow_transform=True` to emphasize low-flow performance;
`low_flow_percentile` sets the transform scale.

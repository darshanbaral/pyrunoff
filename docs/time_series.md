# Time series utilities

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

Time series normally infer their interval from at least two uniformly spaced timestamps. Intervals must be whole minutes and are validated automatically. Routing methods may impose additional time-step constraints. `RainfallSeries` rejects negative rainfall, while `DischargeSeries`, `ElevationSeries`, and `StorageSeries` identify flow, elevation, and storage data, respectively.

Rainfall values are incremental depths in inches: the value at time `t` is the depth that fell during the interval ending at `t`. The first value therefore covers the interval before the first timestamp.

`fix_end(end)` returns a copy of a series ending at `end`, dropping later values or adding zeros up to `end` at the series' time step. The routing methods accept the same `end` argument.

## Network ordering

`Network` derives a topological execution order and direct upstream-node mapping from ordered flow paths. By default, it rejects paths that split from a node to multiple downstream nodes.

```python
from pyrunoff.utils.network import Network

network = Network(paths=[["A", "B"], ["C", "B"]])
print(network.sorted_nodes)
print(network.upstream_nodes["B"])
```

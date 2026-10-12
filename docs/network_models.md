# Network models

A `Model` links basins, reaches, reservoirs and junctions into a network and routes rainfall through it in upstream-to-downstream order. Three files describe a run: the model, its parameters and the rainfall.

## Element types, processes and methods

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

## Model file

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

## Parameter file

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

## Rainfall file

Rainfall is a CSV with a timestamp column and one column of incremental depth (inches) per basin, named after the basin. Other columns are ignored.

```text
timestamp,B1,B2
2024-01-01 00:15,0.10,0.05
2024-01-01 00:30,0.25,0.12
```

## Running the model

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

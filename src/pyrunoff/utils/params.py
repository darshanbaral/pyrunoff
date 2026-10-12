"""Read model, parameter and rainfall files for a :class:`.Model`."""

import dataclasses
import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas

from pyrunoff.reservoir import ElevationStorage, StorageDischarge
from pyrunoff.utils import RainfallSeries
from pyrunoff.utils.model import (
    ELEMENT_PROCESSES,
    METHOD_CLASSES,
    Element,
    ElementParams,
    ElementType,
    Method,
    Model,
    Process,
)

#: Section of the parameter file holding each element type.
_SECTIONS: dict[str, ElementType] = {
    "basins": ElementType.BASIN,
    "reaches": ElementType.REACH,
    "reservoirs": ElementType.RESERVOIR,
}

#: :meth:`.Model.route` argument that receives each process's parameters.
_ROUTE_ARGUMENTS: dict[tuple[ElementType, Process], str] = {
    (ElementType.BASIN, Process.LOSS): "loss_params",
    (ElementType.BASIN, Process.TRANSFORM): "transform_params",
    (ElementType.BASIN, Process.BASEFLOW): "baseflow_params",
    (ElementType.REACH, Process.ROUTING): "reach_params",
    (ElementType.RESERVOIR, Process.ROUTING): "reservoir_params",
}

#: Element-level parameters, passed to any of the element's methods that take them.
_SHARED_PARAMS = {"basin_area"}

#: Method arguments :meth:`.Model.route` supplies, so not read from the file.
_ROUTE_INPUTS = {"rainfall_series", "inflow_series"}

#: Units accepted for durations, converted to ``pandas.Timedelta``.
TIME_UNITS = {"seconds", "minutes", "hours"}


def load_model(path: str | Path) -> Model:
    """Read a model file; see :meth:`.Model.from_dict` for its form.

    :param path: JSON model file.
    :return: The model.
    :raises ValueError: If the model is invalid.
    """
    with open(path) as f:
        return Model.from_dict(json.load(f))


def load_rainfall(
    path: str | Path,
    model: Model,
    time_column: str = "timestamp",
    date_format: str | None = None,
    end: pandas.Timestamp | datetime | None = None,
    end_offset: pandas.Timedelta | timedelta | None = None,
) -> dict[str, RainfallSeries]:
    """Read incremental rainfall for each of the model's basins from a CSV file.

    The file has a timestamp column and one column of rainfall depths (inches)
    per basin, named after the basin. Other columns are ignored.

    :param path: CSV file to read.
    :param model: Model whose basins need rainfall.
    :param time_column: Name of the timestamp column.
    :param date_format: ``strftime`` format of the timestamps, e.g.
        ``"%m/%d/%Y %H:%M"``; inferred when ``None``.
    :param end: Optional timestamp at which every series ends; see
        :meth:`.TimeSeries.fix_end`.
    :param end_offset: Optional end relative to the last timestamp in the file:
        positive adds zero rainfall (e.g. to let hydrographs recede), negative
        drops the last values; not combined with ``end``.
    :return: Rainfall keyed by basin name, ready for :meth:`.Model.route`.
    :raises ValueError: If both ``end`` and ``end_offset`` are given, the end
        is before the first timestamp, the timestamp column is missing or does
        not match ``date_format``, a basin has no column, or a column is invalid.
    """
    if end is not None and end_offset is not None:
        raise ValueError("Pass either 'end' or 'end_offset', not both")

    table = pandas.read_csv(path)
    if time_column not in table:
        raise ValueError(
            f"No timestamp column '{time_column}' in {path}; "
            f"it has {list(table.columns)}"
        )
    try:
        times = pandas.to_datetime(table.pop(time_column), format=date_format)
    except ValueError as err:
        raise ValueError(
            f"Cannot parse '{time_column}' in {path} with format {date_format!r}: {err}"
        ) from err
    table.index = pandas.DatetimeIndex(times)
    if end_offset is not None:
        end = table.index.max() + pandas.Timedelta(end_offset)
    if end is not None and pandas.Timestamp(end) < table.index.min():
        raise ValueError(
            f"End {pandas.Timestamp(end)} is before the first timestamp "
            f"{table.index.min()} in {path}"
        )

    basins = [el.name for el in model.elements if el.type == ElementType.BASIN]
    return RainfallSeries.dict_from_dataframe(table, columns=basins, end=end)


def load_params(path: str | Path, model: Model) -> dict[str, ElementParams]:
    """Read a parameter file for ``model``; see :func:`parse_params`.

    Curve file paths in the parameter file are relative to its directory.

    :param path: JSON parameter file.
    :param model: Model the parameters are for.
    :return: Parameters keyed by :meth:`.Model.route` argument name.
    :raises ValueError: If the parameters are invalid; see :func:`parse_params`.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return parse_params(data, model, base_dir=path.parent)


def parse_params(
    data: Mapping,
    model: Model,
    base_dir: str | Path = ".",
) -> dict[str, ElementParams]:
    """Convert a parameter dictionary into the arguments of :meth:`.Model.route`.

    The dictionary groups elements by type, and each element's parameters by
    process and method::

        {
            "basins": {
                "<basin>": {
                    "basin_area": {"value": 465.4},
                    "<process>": {
                        "<method>": {"<param>": {"value": 3384, "units": "seconds"}}
                    }
                }
            },
            "reaches": {"<reach>": {"routing": {"<method>": {...}}}},
            "reservoirs": {"<reservoir>": {"routing": {"<method>": {...}}}}
        }

    A parameter is one of:

    - ``{"value": v}``: the number ``v``.
    - ``{"value": v, "units": u}``: a ``pandas.Timedelta`` when ``u`` is in
      :data:`TIME_UNITS`.
    - ``{"path": p, "<x>_column_name": ..., "<y>_column_name": ...}``: a curve
      read from the CSV file ``p``; an :class:`.ElevationStorage` for elevation
      and storage columns, a :class:`.StorageDischarge` for storage and
      discharge columns.
    - Any other dictionary: a dictionary of parameters, e.g. one curve per
      outlet.

    The model chooses each element's methods; a process may hold parameters
    for other methods too, so switching methods in the model needs no change
    here. Only the chosen method's parameters are read, but every method's
    parameter names are checked against its class.

    Element-level parameters (``basin_area``) are passed to every method of the
    element whose class takes them, unless the method's own parameters set
    them.

    :param data: Parameters in the form above.
    :param model: Model the parameters are for; it gives each element's methods.
    :param base_dir: Directory curve file paths are relative to.
    :return: Parameters keyed by :meth:`.Model.route` argument name, so that
        ``model.route(rainfall, **parse_params(data, model))`` runs the model.
    :raises ValueError: If an element is not in the model or has another type;
        a section, element key, method or parameter name is unknown; a used
        parameter is malformed; a process has no parameters for the model's
        method; or a parameter a method requires is missing.
    """
    base_dir = Path(base_dir)
    elements = {el.name: el for el in model.elements}
    result: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in _ROUTE_ARGUMENTS.values()
    }

    for section, entries in data.items():
        if section not in _SECTIONS:
            raise ValueError(
                f"Unknown parameter section '{section}'; "
                f"expected one of {list(_SECTIONS)}"
            )
        type_ = _SECTIONS[section]
        for name, entry in entries.items():
            element = elements.get(name)
            if element is None:
                raise ValueError(
                    f"'{name}' in '{section}' is not an element of the model"
                )
            if element.type != type_:
                raise ValueError(
                    f"'{name}' is listed in '{section}' but is a {element.type} "
                    "element in the model"
                )
            for process, params in _element_params(element, entry, base_dir).items():
                result[_ROUTE_ARGUMENTS[(type_, process)]][name] = params

    return result


def _element_params(
    element: Element, entry: Mapping, base_dir: Path
) -> dict[Process, dict[str, Any]]:
    """Return the constructor arguments of each of the element's methods.

    :param element: Element the parameters are for.
    :param entry: The element's entry in the parameter file.
    :param base_dir: Directory curve file paths are relative to.
    :return: Arguments of the chosen method, keyed by process.
    :raises ValueError: If a key is unknown, the chosen method has no
        parameters, or a required parameter is missing.
    """
    processes = ELEMENT_PROCESSES[element.type]
    shared = {}
    groups = {}
    for key, value in entry.items():
        if key in _SHARED_PARAMS:
            shared[key] = _parse_value(value, base_dir, key)
        elif key in processes:
            groups[Process(key)] = value
        else:
            raise ValueError(
                f"Unknown parameter '{key}' for {element.type} '{element.name}'; "
                f"must be one of {[str(p) for p in processes] + sorted(_SHARED_PARAMS)}"
            )

    params = {}
    for process, group in groups.items():
        method_params = _method_params(element, process, group)
        method = element.methods[process]
        if method not in method_params:
            raise ValueError(
                f"No {process} parameters for '{element.name}' method '{method}'; "
                f"found methods {[str(m) for m in method_params]}"
            )
        method_class = METHOD_CLASSES[method]
        class_fields = [f for f in dataclasses.fields(method_class) if f.init]
        fields = {f.name for f in class_fields}
        params[process] = {
            **{k: v for k, v in shared.items() if k in fields},
            **{
                k: _parse_value(v, base_dir, k)
                for k, v in method_params[method].items()
            },
        }
        required = {
            f.name
            for f in class_fields
            if f.name not in _ROUTE_INPUTS
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        }
        missing = required - set(params[process])
        if missing:
            raise ValueError(
                f"Missing {process} parameters {sorted(missing)} for "
                f"'{element.name}'; {method_class.__name__} requires {sorted(required)}"
            )
    return params


def _method_params(
    element: Element, process: Process, group: Any
) -> dict[Method, Mapping]:
    """Key a process's parameters by method, checking methods and names.

    :param element: Element the parameters are for.
    :param process: Process the parameters are for.
    :param group: The process's entry, keyed by method name.
    :return: Unparsed parameters keyed by method.
    :raises ValueError: If ``group`` is malformed, or a method or parameter
        name is unknown.
    """
    if not isinstance(group, Mapping):
        raise ValueError(
            f"{process} parameters for '{element.name}' must be an object keyed "
            f"by method, not {group!r}"
        )
    allowed = ELEMENT_PROCESSES[element.type][process]
    by_method = {}
    for method_name, method_params in group.items():
        if method_name not in allowed:
            raise ValueError(
                f"Unknown {process} method '{method_name}' for '{element.name}'; "
                f"expected one of {[str(m) for m in allowed]}"
            )
        if not isinstance(method_params, Mapping):
            raise ValueError(
                f"'{method_name}' parameters for '{element.name}' must be an "
                f"object, not {method_params!r}"
            )
        method = Method(method_name)
        fields = {
            f.name
            for f in dataclasses.fields(METHOD_CLASSES[method])
            if f.init and f.name not in _ROUTE_INPUTS
        }
        unknown = sorted(set(method_params) - fields)
        if unknown:
            raise ValueError(
                f"Unknown '{method}' parameters {unknown} for '{element.name}'; "
                f"expected some of {sorted(fields)}"
            )
        by_method[method] = method_params
    return by_method


def _parse_value(value: Any, base_dir: Path, name: str) -> Any:
    """Convert one parameter from its file form (see :func:`parse_params`).

    :param value: Parameter as read from the file.
    :param base_dir: Directory curve file paths are relative to.
    :param name: Parameter name, used in error messages.
    :return: The number, duration, curve or dictionary the parameter holds.
    :raises ValueError: If ``value`` is not an object or has unknown units.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"Parameter '{name}' must be an object, not {value!r}")

    if "value" in value:
        units = value.get("units")
        if units is None:
            return value["value"]
        if units in TIME_UNITS:
            return pandas.Timedelta(value["value"], unit=units)
        raise ValueError(
            f"Unknown units '{units}' for parameter '{name}'; "
            f"expected one of {sorted(TIME_UNITS)}"
        )

    if "path" in value:
        return _read_curve(value, base_dir, name)

    return {k: _parse_value(v, base_dir, k) for k, v in value.items()}


def _read_curve(
    value: Mapping, base_dir: Path, name: str
) -> ElevationStorage | StorageDischarge:
    """Read a rating curve from the CSV file and columns named in ``value``.

    :param value: Curve parameter with ``path`` and ``<x>_column_name`` keys.
    :param base_dir: Directory ``path`` is relative to.
    :param name: Parameter name, used in error messages.
    :return: The curve; its class depends on which columns are named.
    :raises ValueError: If the column names do not form a known curve or are
        not in the file.
    """
    columns = {
        key.removesuffix("_column_name"): column
        for key, column in value.items()
        if key.endswith("_column_name")
    }
    match sorted(columns):
        case ["elevation", "storage"]:
            curve_class = ElevationStorage
        case ["discharge", "storage"]:
            curve_class = StorageDischarge
        case _:
            raise ValueError(
                f"Curve '{name}' needs elevation and storage or storage and "
                f"discharge column names, not {sorted(columns)}"
            )

    path = base_dir / value["path"]
    table = pandas.read_csv(path)
    missing = [column for column in columns.values() if column not in table]
    if missing:
        raise ValueError(
            f"Columns {missing} for curve '{name}' are not in {path}; "
            f"it has {list(table.columns)}"
        )
    return curve_class(**{field: table[col] for field, col in columns.items()})

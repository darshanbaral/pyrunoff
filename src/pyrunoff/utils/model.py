from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from pyrunoff.baseflow import LinearReservoir
from pyrunoff.loss import DeficitConstant, InitialConstant
from pyrunoff.reach import Muskingum
from pyrunoff.reservoir import LevelPool
from pyrunoff.transform import ClarkUh
from pyrunoff.utils import DischargeSeries, RainfallSeries, validate_series_type
from pyrunoff.utils.network import Network


class ElementType(StrEnum):
    BASIN = "basin"
    REACH = "reach"
    JUNCTION = "junction"
    RESERVOIR = "reservoir"
    SINK = "sink"


class Process(StrEnum):
    """Hydrologic process an element computes with a selectable method."""

    LOSS = "loss"
    TRANSFORM = "transform"
    BASEFLOW = "baseflow"
    ROUTING = "routing"


class Method(StrEnum):
    """Calculation method used for a process."""

    INITIAL_CONSTANT = "initial_constant"
    DEFICIT_CONSTANT = "deficit_constant"
    CLARK_UH = "clark_uh"
    LINEAR_RESERVOIR = "linear_reservoir"
    MUSKINGUM = "muskingum"
    LEVEL_POOL = "level_pool"


@dataclass
class BasinOutput:
    """Results of routing a basin.

    :param input: Rainfall on the basin.
    :param excess: Rainfall remaining after losses.
    :param total_loss: Rainfall removed by the loss method.
    :param baseflow: Baseflow from the rainfall loss.
    :param direct_runoff: Runoff from the excess rainfall.
    :param total_outflow: Direct runoff plus baseflow.
    """

    input: RainfallSeries
    excess: RainfallSeries
    total_loss: RainfallSeries
    baseflow: DischargeSeries
    direct_runoff: DischargeSeries
    total_outflow: DischargeSeries


class LossResult(Protocol):
    """Output every loss method's ``route()`` must return."""

    excess: RainfallSeries
    total_loss: RainfallSeries


class RoutingResult(Protocol):
    """Output every reach or reservoir routing method's ``route()`` must return.

    Methods may add their own fields (e.g. storage for reservoirs).
    """

    total_outflow: DischargeSeries


@dataclass
class ReachOutput:
    """Results of routing a reach.

    :param input: Total inflow from upstream elements.
    :param routing_output: Full output of the routing method.
    :param total_outflow: Routed outflow.
    """

    input: DischargeSeries
    routing_output: RoutingResult
    total_outflow: DischargeSeries


@dataclass
class ReservoirOutput:
    """Results of routing a reservoir.

    :param input: Total inflow from upstream elements.
    :param routing_output: Full output of the routing method (outflow per outlet,
        storage and elevation).
    :param total_outflow: Sum of outflow across all outlets.
    """

    input: DischargeSeries
    routing_output: RoutingResult
    total_outflow: DischargeSeries


@dataclass
class JunctionOutput:
    """Results of routing a junction.

    :param input: Total inflow from upstream elements.
    :param total_outflow: Outflow, equal to the inflow.
    """

    input: DischargeSeries
    total_outflow: DischargeSeries


@dataclass
class SinkOutput:
    """Results of routing the sink, which has no outflow.

    :param input: Total inflow from upstream elements.
    """

    input: DischargeSeries


ElementOutput = (
    BasinOutput | ReachOutput | ReservoirOutput | JunctionOutput | SinkOutput
)


# Processes each element type computes, and the methods allowed for each.
ELEMENT_PROCESSES: dict[ElementType, dict[Process, tuple[Method, ...]]] = {
    ElementType.BASIN: {
        Process.LOSS: (Method.INITIAL_CONSTANT, Method.DEFICIT_CONSTANT),
        Process.TRANSFORM: (Method.CLARK_UH,),
        Process.BASEFLOW: (Method.LINEAR_RESERVOIR,),
    },
    ElementType.REACH: {Process.ROUTING: (Method.MUSKINGUM,)},
    ElementType.RESERVOIR: {Process.ROUTING: (Method.LEVEL_POOL,)},
    ElementType.JUNCTION: {},
    ElementType.SINK: {},
}

# Class implementing each method.
METHOD_CLASSES: dict[Method, type] = {
    Method.INITIAL_CONSTANT: InitialConstant,
    Method.DEFICIT_CONSTANT: DeficitConstant,
    Method.CLARK_UH: ClarkUh,
    Method.LINEAR_RESERVOIR: LinearReservoir,
    Method.MUSKINGUM: Muskingum,
    Method.LEVEL_POOL: LevelPool,
}

# Constructor arguments for a method, keyed by element name,
# e.g. ``loss_params["B1"] = {"initial_loss": 0.5, "constant_loss": 0.1}``.
ElementParams = Mapping[str, Mapping[str, Any]]


def parse_methods(
    element_type: ElementType, methods: Mapping[str, str]
) -> dict[Process, Method]:
    """Convert method names keyed by process name, checking each is allowed.

    :param element_type: Type of the element the methods are for.
    :param methods: Method names keyed by process name; may be partial.
    :return: The methods keyed by process.
    :raises ValueError: If a process or method is not valid for the type.
    """
    allowed = ELEMENT_PROCESSES[element_type]
    parsed = {}
    for process_name, method_name in methods.items():
        process, method = Process(process_name), Method(method_name)
        if process not in allowed:
            raise ValueError(
                f"{element_type} elements do not compute '{process}' process"
            )
        if method not in allowed[process]:
            raise ValueError(
                f"'{method}' is not a valid {process} method for {element_type} "
                f"elements; expected one of {[str(m) for m in allowed[process]]}"
            )
        parsed[process] = method
    return parsed


@dataclass(frozen=True)
class Element:
    """Model element; identity (equality and hash) is its name and type.

    :param name: Unique element name.
    :param type: Element type.
    :param methods: Method for every process the element type computes.
    :raises ValueError: If a method is invalid or a process has no method.
    """

    name: str
    type: ElementType
    methods: Mapping[Process, Method] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self):
        """Validate the methods and make them read-only."""
        methods = parse_methods(
            self.type, {str(k): str(v) for k, v in self.methods.items()}
        )
        missing = [str(p) for p in ELEMENT_PROCESSES[self.type] if p not in methods]
        if missing:
            raise ValueError(
                f"Element '{self.name}' ({self.type}) has no method for {missing}; "
                "specify it on the element or in the model's 'defaults'"
            )
        object.__setattr__(self, "methods", MappingProxyType(methods))


@dataclass
class Model:
    """Network of elements in execution order.

    The dictionary form read by :meth:`from_dict` and written by
    :meth:`to_dict` lists each element once::

        {
            "defaults": {"<element type>": {"<process>": "<method>"}},
            "elements": {
                "<element>": {
                    "type": "<element type>",
                    "downstream": "<element>",
                    "<process>": "<method>"
                }
            }
        }

    Every element needs a ``type``. Exactly one element has no ``downstream``;
    it is the model's sink and its ``type`` must be ``"sink"``. No other
    element may be a sink.

    An element's method for a process is taken from its own entry, otherwise
    from ``defaults`` for its type. Every process of every element must end
    up with a method; :data:`ELEMENT_PROCESSES` lists the allowed ones.
    """

    elements: list[Element]
    upstream_elements: dict[Element, set[Element]]
    defaults: dict[ElementType, dict[Process, Method]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        """Build a model from its dictionary form (see class docstring)."""
        defaults = {}
        for type_name, methods in data.get("defaults", {}).items():
            type_ = ElementType(type_name)
            defaults[type_] = parse_methods(type_, methods)

        entries = data["elements"]

        sinks = [name for name, entry in entries.items() if "downstream" not in entry]
        if len(sinks) != 1:
            raise ValueError(
                "Exactly one element (the sink) must have no 'downstream'; "
                f"found {len(sinks)}: {sinks}"
            )

        all_elements = {}
        paths = []
        for name, entry in entries.items():
            entry = dict(entry)
            if "type" not in entry:
                raise ValueError(f"Element '{name}' has no 'type'")
            type_ = ElementType(entry.pop("type"))
            downstream = entry.pop("downstream", None)

            if downstream is None:
                if type_ != ElementType.SINK:
                    raise ValueError(
                        f"'{name}' has no 'downstream', so it is the sink; "
                        f"its type must be 'sink', not '{type_}'"
                    )
                paths.append([name])
            else:
                if type_ == ElementType.SINK:
                    raise ValueError(
                        f"'{name}' has a 'downstream', so it cannot be a sink"
                    )
                if downstream not in entries:
                    raise ValueError(
                        f"Downstream element '{downstream}' of '{name}' is not defined"
                    )
                paths.append([name, downstream])

            element_methods = {
                **defaults.get(type_, {}),
                **parse_methods(type_, entry),
            }
            all_elements[name] = Element(name, type_, element_methods)

        sorted_path, upstream_elements = Network.toposort(paths)
        new_sorted_path = [all_elements[name] for name in sorted_path]

        new_upstream_elements = {}
        for el, uel in upstream_elements.items():
            new_upstream_elements[all_elements[el]] = {
                all_elements[name] for name in uel
            }

        return cls(new_sorted_path, new_upstream_elements, defaults)

    def to_dict(self) -> dict:
        """Return the dictionary form; elements list only methods that differ
        from :attr:`defaults`."""
        downstream = {
            upstream: element
            for element, upstreams in self.upstream_elements.items()
            for upstream in upstreams
        }

        elements = {}
        for el in self.elements:
            entry = {"type": str(el.type)}
            if el in downstream:
                entry["downstream"] = downstream[el].name
            type_defaults = self.defaults.get(el.type, {})
            for process, method in el.methods.items():
                if type_defaults.get(process) != method:
                    entry[str(process)] = str(method)
            elements[el.name] = entry

        return {
            "defaults": {
                str(type_): {str(p): str(m) for p, m in methods.items()}
                for type_, methods in self.defaults.items()
            },
            "elements": elements,
        }

    def route(
        self,
        rainfall: Mapping[str, RainfallSeries],
        loss_params: ElementParams | None = None,
        transform_params: ElementParams | None = None,
        baseflow_params: ElementParams | None = None,
        reach_params: ElementParams | None = None,
        reservoir_params: ElementParams | None = None,
    ) -> dict[str, ElementOutput]:
        """Route rainfall through the model in execution order.

        Each element uses the method given in :attr:`Element.methods` for each
        of its processes. Parameters are keyed by element name and passed as
        given to the class of the method chosen for that element, so each
        element's parameters must be consistent with its chosen method and
        match that class's constructor (e.g. ``initial_loss`` and
        ``constant_loss`` for ``initial_constant``, ``max_deficit`` and
        ``constant_loss`` for ``deficit_constant``; timedeltas for durations).

        :param rainfall: Incremental rainfall in inches for each basin, keyed
            by basin name.
        :param loss_params: Loss method arguments for each basin.
        :param transform_params: Transform method arguments for each basin.
        :param baseflow_params: Baseflow method arguments for each basin.
        :param reach_params: Routing method arguments for each reach.
        :param reservoir_params: Routing method arguments for each reservoir.
        :return: Results keyed by element name: a :class:`BasinOutput`,
            :class:`ReachOutput`, :class:`ReservoirOutput`,
            :class:`JunctionOutput` or :class:`SinkOutput` depending on the
            element type.
        :raises TypeError: If a basin's rainfall is not a :class:`RainfallSeries`.
        :raises ValueError: If rainfall or parameters are missing for an
            element, a basin has upstream elements, or another element has none.
        """
        results: dict[str, ElementOutput] = {}

        for element in self.elements:
            name = element.name
            upstream = sorted(self.upstream_elements[element], key=lambda el: el.name)

            if element.type == ElementType.BASIN:
                if upstream:
                    raise ValueError(
                        f"Basin '{name}' cannot have upstream elements: "
                        f"{[el.name for el in upstream]}"
                    )
                if name not in rainfall:
                    raise ValueError(f"No rainfall for basin '{name}'")
                basin_rainfall = rainfall[name]
                validate_series_type(
                    basin_rainfall, RainfallSeries, f"rainfall for basin '{name}'"
                )

                loss: LossResult = self._build(
                    element, Process.LOSS, basin_rainfall, loss_params, "loss_params"
                ).route()
                direct_runoff = self._build(
                    element,
                    Process.TRANSFORM,
                    loss.excess,
                    transform_params,
                    "transform_params",
                ).route()
                baseflow = self._build(
                    element,
                    Process.BASEFLOW,
                    loss.total_loss,
                    baseflow_params,
                    "baseflow_params",
                ).route()

                results[name] = BasinOutput(
                    input=basin_rainfall,
                    excess=loss.excess,
                    total_loss=loss.total_loss,
                    baseflow=baseflow,
                    direct_runoff=direct_runoff,
                    total_outflow=DischargeSeries.sum([direct_runoff, baseflow]),
                )
                continue

            if not upstream:
                raise ValueError(f"{element.type} '{name}' has no upstream elements")
            inflow = DischargeSeries.sum(
                results[el.name].total_outflow for el in upstream
            )

            match element.type:
                case ElementType.REACH:
                    routing: RoutingResult = self._build(
                        element, Process.ROUTING, inflow, reach_params, "reach_params"
                    ).route()
                    results[name] = ReachOutput(
                        input=inflow,
                        routing_output=routing,
                        total_outflow=routing.total_outflow,
                    )
                case ElementType.RESERVOIR:
                    routing = self._build(
                        element,
                        Process.ROUTING,
                        inflow,
                        reservoir_params,
                        "reservoir_params",
                    ).route()
                    results[name] = ReservoirOutput(
                        input=inflow,
                        routing_output=routing,
                        total_outflow=routing.total_outflow,
                    )
                case ElementType.JUNCTION:
                    results[name] = JunctionOutput(
                        input=inflow, total_outflow=inflow.copy()
                    )
                case ElementType.SINK:
                    results[name] = SinkOutput(input=inflow)

        return results

    @staticmethod
    def _build(
        element: Element,
        process: Process,
        series,
        params: ElementParams | None,
        params_name: str,
    ):
        """Create the element's method for ``process`` with ``series`` as input.

        Loss, transform and baseflow methods take ``rainfall_series``; routing
        methods take ``inflow_series``.
        """
        if params is None or element.name not in params:
            raise ValueError(
                f"No {process} parameters for {element.type} '{element.name}' "
                f"in {params_name}"
            )

        input_name = (
            "inflow_series" if process == Process.ROUTING else "rainfall_series"
        )
        method_class = METHOD_CLASSES[element.methods[process]]
        return method_class(**{input_name: series}, **params[element.name])

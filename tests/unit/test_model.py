import json
import unittest
import unittest.mock
from datetime import timedelta
from pathlib import Path

import numpy
import pandas

from pyrunoff.reach import MuskingumOutput
from pyrunoff.utils import DischargeSeries, RainfallSeries
from pyrunoff.utils.model import (
    ELEMENT_PROCESSES,
    METHOD_CLASSES,
    BasinOutput,
    Element,
    ElementType,
    JunctionOutput,
    Method,
    Model,
    Process,
    ReachOutput,
    SinkOutput,
)

DATA_DIR = Path(__file__).parents[1] / "data"


BASIN_METHODS = {
    "loss": "initial_constant",
    "transform": "clark_uh",
    "baseflow": "linear_reservoir",
}


def simple_model_data(**extra):
    data = {
        "defaults": {
            "basin": dict(BASIN_METHODS),
            "reach": {"routing": "muskingum"},
        },
        "elements": {
            "B1": {"type": "basin", "downstream": "R1"},
            "B2": {"type": "basin", "downstream": "J1"},
            "R1": {"type": "reach", "downstream": "J1"},
            "J1": {"type": "junction", "downstream": "S"},
            "S": {"type": "sink"},
        },
    }
    data.update(extra)
    return data


class TestElement(unittest.TestCase):
    def test_methods(self):
        basin = Element("B1", ElementType.BASIN, BASIN_METHODS)
        self.assertEqual(
            dict(basin.methods),
            {
                Process.LOSS: Method.INITIAL_CONSTANT,
                Process.TRANSFORM: Method.CLARK_UH,
                Process.BASEFLOW: Method.LINEAR_RESERVOIR,
            },
        )
        self.assertEqual(dict(Element("J", ElementType.JUNCTION).methods), {})

    def test_missing_method(self):
        with self.assertRaises(ValueError) as cm:
            Element("B1", ElementType.BASIN, {"loss": "initial_constant"})
        self.assertIn("transform", str(cm.exception))

    def test_invalid_method_for_process(self):
        with self.assertRaises(ValueError):
            Element("B1", ElementType.BASIN, {**BASIN_METHODS, "loss": "clark_uh"})

    def test_invalid_process_for_type(self):
        with self.assertRaises(ValueError):
            Element(
                "R1",
                ElementType.REACH,
                {"routing": "muskingum", "loss": "initial_constant"},
            )

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            Element("R1", ElementType.REACH, {"routing": "kinematic_wave"})

    def test_identity_ignores_methods(self):
        a = Element("B1", ElementType.BASIN, BASIN_METHODS)
        b = Element(
            "B1", ElementType.BASIN, {**BASIN_METHODS, "loss": "deficit_constant"}
        )
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_methods_read_only(self):
        basin = Element("B1", ElementType.BASIN, BASIN_METHODS)
        with self.assertRaises(TypeError):
            basin.methods[Process.LOSS] = Method.DEFICIT_CONSTANT


class TestModel(unittest.TestCase):
    def test_from_dict(self):
        model = Model.from_dict(simple_model_data())
        names = [el.name for el in model.elements]
        self.assertLess(names.index("R1"), names.index("J1"))
        self.assertEqual(names[-1], "S")
        by_name = {el.name: el for el in model.elements}
        self.assertEqual(
            model.upstream_elements[by_name["J1"]], {by_name["R1"], by_name["B2"]}
        )
        self.assertEqual(by_name["B1"].methods[Process.LOSS], Method.INITIAL_CONSTANT)

    def test_element_overrides_default(self):
        data = simple_model_data()
        data["defaults"]["basin"]["loss"] = "deficit_constant"
        data["elements"]["B2"]["loss"] = "initial_constant"
        by_name = {el.name: el for el in Model.from_dict(data).elements}
        self.assertEqual(by_name["B1"].methods[Process.LOSS], Method.DEFICIT_CONSTANT)
        self.assertEqual(by_name["B2"].methods[Process.LOSS], Method.INITIAL_CONSTANT)

    def test_missing_default(self):
        data = simple_model_data()
        del data["defaults"]["reach"]
        with self.assertRaises(ValueError) as cm:
            Model.from_dict(data)
        self.assertIn("R1", str(cm.exception))

    def test_defaults_optional_when_set_on_elements(self):
        data = simple_model_data()
        del data["defaults"]["reach"]
        data["elements"]["R1"]["routing"] = "muskingum"
        Model.from_dict(data)

    def test_invalid_default(self):
        data = simple_model_data()
        data["defaults"]["reach"]["routing"] = "level_pool"
        with self.assertRaises(ValueError):
            Model.from_dict(data)

    def test_unknown_key_in_element(self):
        data = simple_model_data()
        data["elements"]["B1"]["downsteam"] = "R1"
        with self.assertRaises(ValueError):
            Model.from_dict(data)

    def test_undefined_downstream(self):
        data = simple_model_data()
        data["elements"]["B1"]["downstream"] = "X"
        with self.assertRaises(ValueError):
            Model.from_dict(data)

    def test_to_dict_lists_only_non_default_methods(self):
        data = simple_model_data()
        data["elements"]["B2"]["loss"] = "deficit_constant"
        out = Model.from_dict(data).to_dict()
        self.assertEqual(out["elements"]["B1"], {"type": "basin", "downstream": "R1"})
        self.assertEqual(out["elements"]["B2"]["loss"], "deficit_constant")
        self.assertEqual(out["defaults"], simple_model_data()["defaults"])

    def test_to_dict_keeps_declared_defaults(self):
        data = simple_model_data()
        data["elements"]["B1"]["loss"] = "deficit_constant"
        data["elements"]["B2"]["loss"] = "deficit_constant"
        out = Model.from_dict(data).to_dict()
        self.assertEqual(out["defaults"], data["defaults"])
        self.assertEqual(out["elements"]["B1"]["loss"], "deficit_constant")

    def test_defaults_field(self):
        model = Model.from_dict(simple_model_data())
        self.assertEqual(
            model.defaults,
            {
                ElementType.BASIN: {
                    Process.LOSS: Method.INITIAL_CONSTANT,
                    Process.TRANSFORM: Method.CLARK_UH,
                    Process.BASEFLOW: Method.LINEAR_RESERVOIR,
                },
                ElementType.REACH: {Process.ROUTING: Method.MUSKINGUM},
            },
        )

    def test_round_trip(self):
        data = simple_model_data()
        data["elements"]["B2"]["loss"] = "deficit_constant"
        model = Model.from_dict(data)
        restored = Model.from_dict(json.loads(json.dumps(model.to_dict())))

        self.assertEqual(restored.elements, model.elements)
        self.assertEqual(restored.upstream_elements, model.upstream_elements)
        for a, b in zip(restored.elements, model.elements, strict=True):
            self.assertEqual(dict(a.methods), dict(b.methods))

    def test_type_required(self):
        data = simple_model_data()
        del data["elements"]["S"]["type"]
        with self.assertRaises(ValueError) as cm:
            Model.from_dict(data)
        self.assertIn("S", str(cm.exception))

    def test_sink_with_other_type(self):
        data = simple_model_data()
        data["elements"]["S"]["type"] = "junction"
        with self.assertRaises(ValueError):
            Model.from_dict(data)

    def test_sink_type_with_downstream(self):
        data = simple_model_data()
        data["elements"]["J1"]["type"] = "sink"
        with self.assertRaises(ValueError):
            Model.from_dict(data)

    def test_multiple_elements_without_downstream(self):
        data = simple_model_data()
        del data["elements"]["B2"]["downstream"]
        with self.assertRaises(ValueError) as cm:
            Model.from_dict(data)
        self.assertIn("B2", str(cm.exception))
        self.assertIn("S", str(cm.exception))

    def test_no_element_without_downstream(self):
        data = simple_model_data()
        data["elements"]["S"] = {"type": "junction", "downstream": "J1"}
        with self.assertRaises(ValueError):
            Model.from_dict(data)

    def test_to_dict_sink(self):
        out = Model.from_dict(simple_model_data()).to_dict()
        self.assertEqual(out["elements"]["S"], {"type": "sink"})

    def test_single_sink_round_trip(self):
        data = {"elements": {"A": {"type": "sink"}}}
        model = Model.from_dict(Model.from_dict(data).to_dict())
        self.assertEqual([el.name for el in model.elements], ["A"])

    def test_model_json(self):
        with open(DATA_DIR / "model.json") as f:
            data = json.load(f)
        model = Model.from_dict(data)
        self.assertEqual(len(model.elements), len(data["elements"]))
        out = model.to_dict()
        self.assertEqual(out["defaults"], data["defaults"])
        restored = Model.from_dict(out)
        self.assertEqual(restored.upstream_elements, model.upstream_elements)


if __name__ == "__main__":
    unittest.main()


def route_inputs():
    index = pandas.date_range("2024-01-01", periods=96, freq="15min")
    rain = numpy.zeros(len(index))
    rain[4:20] = 0.1
    rainfall = RainfallSeries.dict_from_dataframe(
        pandas.DataFrame({"B1": rain, "B2": rain * 0.5}, index=index)
    )

    def basin(area):
        return {
            "basin_area": area,
            "tc": timedelta(hours=1),
            "storage_r": timedelta(hours=2),
        }

    def baseflow(area):
        return {"gw_fraction": 0.3, "k": timedelta(hours=20), "basin_area": area}

    params = {
        "loss_params": {
            "B1": {"initial_loss": 0.2, "constant_loss": 0.05},
            "B2": {
                "max_deficit": 0.3,
                "constant_loss": 0.05,
                "recovery_time": timedelta(hours=24),
            },
        },
        "transform_params": {"B1": basin(500), "B2": basin(300)},
        "baseflow_params": {"B1": baseflow(500), "B2": baseflow(300)},
        "reach_params": {"R1": {"k": timedelta(minutes=30), "x": 0.2}},
    }
    data = simple_model_data()
    data["elements"]["B2"]["loss"] = "deficit_constant"
    return Model.from_dict(data), rainfall, params


@unittest.mock.patch("warnings.warn")
class TestRoute(unittest.TestCase):
    def test_route(self, _):
        model, rainfall, params = route_inputs()
        results = model.route(rainfall, **params)

        self.assertEqual(set(results), {"B1", "B2", "R1", "J1", "S"})
        self.assertIsInstance(results["B1"], BasinOutput)
        self.assertIsInstance(results["R1"], ReachOutput)
        self.assertIsInstance(results["J1"], JunctionOutput)
        self.assertIsInstance(results["S"], SinkOutput)
        self.assertIsInstance(results["B1"].total_outflow, DischargeSeries)
        self.assertIsInstance(results["J1"].total_outflow, DischargeSeries)
        self.assertIsInstance(results["R1"].routing_output, MuskingumOutput)
        self.assertIs(
            results["R1"].total_outflow, results["R1"].routing_output.total_outflow
        )
        self.assertGreater(results["S"].input.data.max(), 0)

        # junction inflow is reach outflow plus B2 outflow
        expected = results["R1"].total_outflow.data + results["B2"].total_outflow.data
        pandas.testing.assert_series_equal(
            results["J1"].input.data, expected, check_names=False
        )

    def test_uses_element_method(self, _):
        model, rainfall, params = route_inputs()
        results = model.route(rainfall, **params)
        # deficit and constant loss never recovers the initial loss, so B2 loses
        # at most max_deficit plus the constant rate; initial-constant B1 differs
        self.assertNotEqual(
            results["B1"].total_loss.data.sum(), results["B2"].total_loss.data.sum()
        )

    def test_missing_params(self, _):
        model, rainfall, params = route_inputs()
        del params["reach_params"]["R1"]
        with self.assertRaises(ValueError) as cm:
            model.route(rainfall, **params)
        self.assertIn("R1", str(cm.exception))
        self.assertIn("reach_params", str(cm.exception))

    def test_missing_param_group(self, _):
        model, rainfall, params = route_inputs()
        del params["baseflow_params"]
        with self.assertRaises(ValueError) as cm:
            model.route(rainfall, **params)
        self.assertIn("baseflow_params", str(cm.exception))

    def test_missing_rainfall(self, _):
        model, rainfall, params = route_inputs()
        with self.assertRaises(ValueError) as cm:
            model.route({"B1": rainfall["B1"]}, **params)
        self.assertIn("B2", str(cm.exception))

    def test_rainfall_not_series_type(self, _):
        model, rainfall, params = route_inputs()
        rainfall["B2"] = rainfall["B2"].data
        with self.assertRaisesRegex(TypeError, "basin 'B2'.*RainfallSeries"):
            model.route(rainfall, **params)

    def test_element_without_upstream(self, _):
        model, rainfall, params = route_inputs()
        data = model.to_dict()
        data["elements"]["R2"] = {"type": "reach", "downstream": "J1"}
        model = Model.from_dict(data)
        with self.assertRaises(ValueError) as cm:
            model.route(rainfall, **params)
        self.assertIn("R2", str(cm.exception))


class TestMethodRegistry(unittest.TestCase):
    def test_every_method_has_a_class(self):
        self.assertEqual(set(METHOD_CLASSES), set(Method))

    def test_every_allowed_method_has_a_class(self):
        for processes in ELEMENT_PROCESSES.values():
            for methods in processes.values():
                for method in methods:
                    self.assertIn(method, METHOD_CLASSES)

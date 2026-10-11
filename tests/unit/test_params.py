import json
import tempfile
import unittest
from pathlib import Path

import pandas

from pyrunoff.reservoir import ElevationStorage, StorageDischarge
from pyrunoff.utils import RainfallSeries
from pyrunoff.utils.model import Model
from pyrunoff.utils.params import load_model, load_params, load_rainfall, parse_params

DATA_DIR = Path(__file__).parents[1] / "data"


def simple_model():
    return Model.from_dict(
        {
            "defaults": {
                "basin": {
                    "loss": "initial_constant",
                    "transform": "clark_uh",
                    "baseflow": "linear_reservoir",
                },
                "reach": {"routing": "muskingum"},
            },
            "elements": {
                "B1": {"type": "basin", "downstream": "R1"},
                "R1": {"type": "reach", "downstream": "S"},
                "S": {"type": "sink"},
            },
        }
    )


def basin_entry():
    return {
        "basin_area": {"value": 100.0},
        "loss": {
            "initial_constant": {
                "initial_loss": {"value": 0.5},
                "constant_loss": {"value": 0.1},
            }
        },
        "transform": {
            "clark_uh": {
                "tc": {"value": 1, "units": "hours"},
                "storage_r": {"value": 7200, "units": "seconds"},
            }
        },
        "baseflow": {
            "linear_reservoir": {
                "gw_fraction": {"value": 0.2},
                "k": {"value": 10, "units": "hours"},
            }
        },
    }


class TestParseParams(unittest.TestCase):
    def test_basin(self):
        params = parse_params({"basins": {"B1": basin_entry()}}, simple_model())
        self.assertEqual(
            params["loss_params"]["B1"], {"initial_loss": 0.5, "constant_loss": 0.1}
        )
        self.assertEqual(
            params["transform_params"]["B1"],
            {
                "basin_area": 100.0,
                "tc": pandas.Timedelta(hours=1),
                "storage_r": pandas.Timedelta(hours=2),
            },
        )
        self.assertEqual(
            params["baseflow_params"]["B1"],
            {
                "basin_area": 100.0,
                "gw_fraction": 0.2,
                "k": pandas.Timedelta(hours=10),
            },
        )

    def test_method_params_override_shared(self):
        entry = basin_entry()
        entry["baseflow"]["linear_reservoir"]["basin_area"] = {"value": 50.0}
        params = parse_params({"basins": {"B1": entry}}, simple_model())
        self.assertEqual(params["baseflow_params"]["B1"]["basin_area"], 50.0)
        self.assertEqual(params["transform_params"]["B1"]["basin_area"], 100.0)

    def test_reach(self):
        data = {
            "reaches": {
                "R1": {
                    "routing": {
                        "muskingum": {
                            "k": {"value": 15, "units": "minutes"},
                            "x": {"value": 0.2},
                        }
                    }
                }
            }
        }
        params = parse_params(data, simple_model())
        self.assertEqual(
            params["reach_params"]["R1"],
            {"k": pandas.Timedelta(minutes=15), "x": 0.2},
        )

    def test_unknown_section(self):
        with self.assertRaisesRegex(ValueError, "Unknown parameter section"):
            parse_params({"junctions": {}}, simple_model())

    def test_element_not_in_model(self):
        with self.assertRaisesRegex(ValueError, "not an element of the model"):
            parse_params({"basins": {"B2": basin_entry()}}, simple_model())

    def test_wrong_section_for_type(self):
        with self.assertRaisesRegex(ValueError, "is a reach element"):
            parse_params({"basins": {"R1": basin_entry()}}, simple_model())

    def test_unknown_element_key(self):
        entry = basin_entry()
        entry["routing"] = {}
        with self.assertRaisesRegex(ValueError, "Unknown parameter 'routing'"):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_other_method_params_not_used(self):
        entry = basin_entry()
        entry["loss"]["deficit_constant"] = {
            "initial_deficit": {"path": "missing.csv"},
            "constant_loss": {"value": 0.9},
        }
        params = parse_params({"basins": {"B1": entry}}, simple_model())
        self.assertEqual(
            params["loss_params"]["B1"], {"initial_loss": 0.5, "constant_loss": 0.1}
        )

    def test_unknown_method(self):
        entry = basin_entry()
        entry["loss"]["scs_curve_number"] = {}
        with self.assertRaisesRegex(
            ValueError, "Unknown loss method 'scs_curve_number'"
        ):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_method_for_other_process(self):
        entry = basin_entry()
        entry["loss"]["clark_uh"] = {}
        with self.assertRaisesRegex(ValueError, "Unknown loss method 'clark_uh'"):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_unknown_param(self):
        entry = basin_entry()
        entry["loss"]["initial_constant"]["max_deficit"] = {"value": 1.0}
        with self.assertRaisesRegex(
            ValueError, r"Unknown 'initial_constant' parameters \['max_deficit'\]"
        ):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_unknown_param_in_other_method(self):
        entry = basin_entry()
        entry["loss"]["deficit_constant"] = {"typo": {"value": 1.0}}
        with self.assertRaisesRegex(
            ValueError, r"Unknown 'deficit_constant' parameters \['typo'\]"
        ):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_no_params_for_model_method(self):
        entry = basin_entry()
        entry["loss"] = {
            "deficit_constant": {
                "max_deficit": {"value": 1.0},
                "constant_loss": {"value": 0.1},
            }
        }
        with self.assertRaisesRegex(
            ValueError, "No loss parameters for 'B1' method 'initial_constant'"
        ):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_params_not_keyed_by_method(self):
        entry = basin_entry()
        entry["loss"] = entry["loss"]["initial_constant"]
        with self.assertRaisesRegex(ValueError, "Unknown loss method 'initial_loss'"):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_missing_method_param(self):
        entry = basin_entry()
        del entry["loss"]["initial_constant"]["constant_loss"]
        with self.assertRaisesRegex(
            ValueError, r"Missing loss parameters \['constant_loss'\]"
        ):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_missing_shared_param(self):
        entry = basin_entry()
        del entry["basin_area"]
        with self.assertRaisesRegex(
            ValueError, r"Missing transform parameters \['basin_area'\]"
        ):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_unknown_units(self):
        entry = basin_entry()
        entry["transform"]["clark_uh"]["tc"]["units"] = "fortnights"
        with self.assertRaisesRegex(ValueError, "Unknown units 'fortnights'"):
            parse_params({"basins": {"B1": entry}}, simple_model())

    def test_bare_value(self):
        entry = basin_entry()
        entry["loss"]["initial_constant"]["initial_loss"] = 0.5
        with self.assertRaisesRegex(ValueError, "must be an object"):
            parse_params({"basins": {"B1": entry}}, simple_model())


class TestCurves(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        pandas.DataFrame({"stage": [1.0, 2.0], "vol": [0.0, 10.0]}).to_csv(
            Path(self.dir.name) / "es.csv", index=False
        )
        self.model = Model.from_dict(
            {
                "elements": {
                    "B1": {
                        "type": "basin",
                        "downstream": "RES",
                        "loss": "initial_constant",
                        "transform": "clark_uh",
                        "baseflow": "linear_reservoir",
                    },
                    "RES": {
                        "type": "reservoir",
                        "downstream": "S",
                        "routing": "level_pool",
                    },
                    "S": {"type": "sink"},
                }
            }
        )

    def reservoir_data(self, **curve):
        return {
            "reservoirs": {
                "RES": {
                    "routing": {
                        "level_pool": {
                            "elevation_storage": {"path": "es.csv", **curve},
                            "storage_discharges": {
                                "spillway": {
                                    "path": "es.csv",
                                    "storage_column_name": "vol",
                                    "discharge_column_name": "stage",
                                }
                            },
                            "initial_storage": {"value": 0.0},
                        }
                    }
                }
            }
        }

    def test_reads_curves_relative_to_base_dir(self):
        data = self.reservoir_data(
            elevation_column_name="stage", storage_column_name="vol"
        )
        routing = parse_params(data, self.model, self.dir.name)["reservoir_params"][
            "RES"
        ]
        es = routing["elevation_storage"]
        self.assertIsInstance(es, ElevationStorage)
        self.assertEqual(es.elevation.tolist(), [1.0, 2.0])
        self.assertEqual(es.storage.tolist(), [0.0, 10.0])
        self.assertIsInstance(
            routing["storage_discharges"]["spillway"], StorageDischarge
        )
        self.assertEqual(routing["initial_storage"], 0.0)

    def test_missing_column(self):
        data = self.reservoir_data(
            elevation_column_name="Stage_ft", storage_column_name="vol"
        )
        with self.assertRaisesRegex(ValueError, r"\['Stage_ft'\]"):
            parse_params(data, self.model, self.dir.name)

    def test_unrecognised_columns(self):
        data = self.reservoir_data(stage_column_name="stage")
        with self.assertRaisesRegex(ValueError, "column names"):
            parse_params(data, self.model, self.dir.name)


class TestLoadModel(unittest.TestCase):
    def test_model_json(self):
        model = load_model(DATA_DIR / "model.json")
        with open(DATA_DIR / "model.json") as f:
            expected = Model.from_dict(json.load(f))
        self.assertEqual(model.to_dict(), expected.to_dict())

    def test_accepts_str_path(self):
        model = load_model(str(DATA_DIR / "model.json"))
        self.assertIsInstance(model, Model)


class TestLoadRainfall(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "rain.csv"
        self.write(
            "Time,B1,gauge\n"
            "01/02/2024 00:00,0.1,9\n"
            "01/02/2024 00:15,0.2,9\n"
            "01/02/2024 00:30,0.3,9\n"
        )

    def write(self, text):
        self.path.write_text(text)

    def test_reads_basins_with_format(self):
        rainfall = load_rainfall(
            self.path, simple_model(), "Time", date_format="%m/%d/%Y %H:%M"
        )
        self.assertEqual(list(rainfall), ["B1"])
        series = rainfall["B1"]
        self.assertIsInstance(series, RainfallSeries)
        self.assertEqual(series.data.index[0], pandas.Timestamp("2024-01-02 00:00"))
        self.assertEqual(series.time_step, pandas.Timedelta(minutes=15))
        self.assertEqual(series.data.tolist(), [0.1, 0.2, 0.3])

    def test_infers_format(self):
        self.write("t,B1\n2024-01-02 00:00,0.1\n2024-01-02 00:15,0.2\n")
        rainfall = load_rainfall(self.path, simple_model(), "t")
        self.assertEqual(
            rainfall["B1"].data.index[-1], pandas.Timestamp("2024-01-02 00:15")
        )

    def test_end(self):
        rainfall = load_rainfall(
            self.path,
            simple_model(),
            "Time",
            date_format="%m/%d/%Y %H:%M",
            end=pandas.Timestamp("2024-01-02 01:00"),
        )
        self.assertEqual(rainfall["B1"].data.tolist(), [0.1, 0.2, 0.3, 0.0, 0.0])

    def test_end_offset_extends(self):
        rainfall = load_rainfall(
            self.path,
            simple_model(),
            "Time",
            date_format="%m/%d/%Y %H:%M",
            end_offset=pandas.Timedelta(minutes=30),
        )
        series = rainfall["B1"]
        self.assertEqual(series.data.tolist(), [0.1, 0.2, 0.3, 0.0, 0.0])
        self.assertEqual(series.data.index[-1], pandas.Timestamp("2024-01-02 01:00"))

    def test_end_offset_truncates(self):
        rainfall = load_rainfall(
            self.path,
            simple_model(),
            "Time",
            date_format="%m/%d/%Y %H:%M",
            end_offset=-pandas.Timedelta(minutes=15),
        )
        self.assertEqual(rainfall["B1"].data.tolist(), [0.1, 0.2])

    def test_end_offset_to_start(self):
        rainfall = load_rainfall(
            self.path,
            simple_model(),
            "Time",
            date_format="%m/%d/%Y %H:%M",
            end_offset=-pandas.Timedelta(minutes=30),
        )
        self.assertEqual(rainfall["B1"].data.tolist(), [0.1])

    def test_end_offset_before_start(self):
        with self.assertRaisesRegex(ValueError, "before the first timestamp"):
            load_rainfall(
                self.path,
                simple_model(),
                "Time",
                date_format="%m/%d/%Y %H:%M",
                end_offset=-pandas.Timedelta(minutes=45),
            )

    def test_end_before_start(self):
        with self.assertRaisesRegex(ValueError, "before the first timestamp"):
            load_rainfall(
                self.path,
                simple_model(),
                "Time",
                date_format="%m/%d/%Y %H:%M",
                end=pandas.Timestamp("2024-01-01"),
            )

    def test_end_and_end_offset(self):
        with self.assertRaisesRegex(ValueError, "either 'end' or 'end_offset'"):
            load_rainfall(
                self.path,
                simple_model(),
                "Time",
                end=pandas.Timestamp("2024-01-02 01:00"),
                end_offset=pandas.Timedelta(minutes=30),
            )

    def test_missing_time_column(self):
        with self.assertRaisesRegex(ValueError, "No timestamp column 'timestamp'"):
            load_rainfall(self.path, simple_model(), "timestamp")

    def test_wrong_date_format(self):
        with self.assertRaisesRegex(ValueError, "Cannot parse 'Time'.*%Y-%m-%d"):
            load_rainfall(
                self.path, simple_model(), "Time", date_format="%Y-%m-%d %H:%M"
            )

    def test_missing_basin_column(self):
        self.write("Time,B2\n01/02/2024 00:00,0.1\n01/02/2024 00:15,0.2\n")
        with self.assertRaisesRegex(ValueError, r"\['B1'\] are not in"):
            load_rainfall(
                self.path, simple_model(), "Time", date_format="%m/%d/%Y %H:%M"
            )


class TestLoadParams(unittest.TestCase):
    def test_model_params_json(self):
        model = load_model(DATA_DIR / "model.json")
        params = load_params(DATA_DIR / "model_params.json", model)

        basins = [el.name for el in model.elements if el.type == "basin"]
        index = pandas.date_range("2020-01-01", periods=8, freq="15min")
        rainfall = RainfallSeries.dict_from_dataframe(
            pandas.DataFrame(0.05, index=index, columns=basins)
        )
        results = model.route(rainfall, **params)
        self.assertEqual(len(results), len(model.elements))

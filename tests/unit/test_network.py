import unittest

from pyrunoff.utils.network import Network

toposort = Network.toposort


class TestToposort(unittest.TestCase):
    def test_single_node(self):
        """Tests that a single-node path is handled correctly."""
        paths = [["a"]]
        order, upstream = toposort(paths)
        self.assertEqual(order, ["a"])
        self.assertEqual(upstream["a"], set())

    def test_simple_linear_routing(self):
        """Tests a simple upstream to downstream reach."""
        paths = [["Reach1", "Reach2", "Outlet"]]
        order, upstream = toposort(paths)
        self.assertEqual(order, ["Reach1", "Reach2", "Outlet"])
        self.assertEqual(upstream["Reach2"], {"Reach1"})
        self.assertEqual(upstream["Outlet"], {"Reach2"})

    def test_convergence(self):
        """Tests multiple tributaries joining a single junction."""
        paths = [["Trib_A", "Junc_1", "Outlet"], ["Trib_B", "Junc_1"]]
        order, upstream = toposort(paths)

        # Verify Junc_1 comes after both tributaries
        self.assertTrue(order.index("Junc_1") > order.index("Trib_A"))
        self.assertTrue(order.index("Junc_1") > order.index("Trib_B"))
        self.assertTrue(order.index("Outlet") > order.index("Junc_1"))

        # Verify upstream mapping
        self.assertEqual(upstream["Junc_1"], {"Trib_A", "Trib_B"})

    def test_redundant_paths(self):
        """Tests that overlapping path definitions don't break the counter."""
        paths = [
            ["A", "B", "C"],
            ["B", "C"],  # Redundant definition of B->C
        ]
        order, upstream = toposort(paths)
        self.assertEqual(order, ["A", "B", "C"])
        self.assertEqual(upstream["C"], {"B"})

    def test_deterministic_sorting(self):
        """Tests that multiple headwaters are sorted alphabetically for consistency."""
        paths = [["Z", "Outlet"], ["A", "Outlet"]]
        order, _ = toposort(paths)
        # Because of the sorted() call on headwaters, the A should come before Z
        self.assertEqual(order, ["A", "Z", "Outlet"])

    def test_divergence_error(self):
        """Tests that a flow split (one to many) raises a ValueError."""
        # A flows to both B and C
        paths = [["A", "B"], ["A", "C"]]
        with self.assertRaises(ValueError) as cm:
            toposort(paths)
        self.assertIn("divergence", str(cm.exception))

    def test_cycle_detection(self):
        """Tests that a circular flow path raises a ValueError."""
        paths = [["A", "B", "C", "A"]]
        with self.assertRaises(ValueError) as cm:
            toposort(paths)
        self.assertIn("Cycle detected", str(cm.exception))

    def test_disconnected_basins(self):
        """Tests that independent watersheds are both included in the sort."""
        paths = [["Basin1_Head", "Basin1_Outlet"], ["Basin2_Head", "Basin2_Outlet"]]
        order, _ = toposort(paths)
        self.assertEqual(len(order), 4)
        self.assertIn("Basin1_Head", order)
        self.assertIn("Basin2_Head", order)

    def test_simple_case(self):
        paths = [["a", "b", "c"]]
        expected_order = ["a", "b", "c"]
        sorted_nodes, _ = toposort(paths)
        self.assertEqual(sorted_nodes, expected_order)

    def test_multiple_paths_no_divergence(self):
        paths = [["a", "b"], ["b", "c"], ["c", "d"]]
        expected_order = ["a", "b", "c", "d"]
        sorted_nodes, _ = toposort(paths)
        self.assertEqual(sorted_nodes, expected_order)

    def test_repeating_paths_no_divergence(self):
        paths = [["a", "b", "c"], ["a", "b"], ["c", "d"]]
        expected_order = ["a", "b", "c", "d"]
        sorted_nodes, _ = toposort(paths)
        self.assertEqual(sorted_nodes, expected_order)

    def test_disconnected_graph(self):
        paths = [["a", "b"], ["x", "y"]]
        sorted_nodes, upstream = toposort(paths)
        for destination, sources in upstream.items():
            dest_idx = sorted_nodes.index(destination)
            for src in sources:
                self.assertLess(sorted_nodes.index(src), dest_idx)

    def test_complex_graph(self):
        paths = [
            ["a", "b", "c", "d", "e", "f", "g", "h"],
            ["i", "j", "k", "e"],
            ["l", "e"],
            ["m", "n", "o", "d", "e"],
            ["p", "d", "e"],
            ["q", "h"],
            ["r", "g"],
            ["s", "t", "u", "g"],
        ]
        sorted_nodes, upstream = toposort(paths)
        for destination, sources in upstream.items():
            dest_idx = sorted_nodes.index(destination)
            for src in sources:
                self.assertLess(sorted_nodes.index(src), dest_idx)

    def test_cycle_detection_2(self):
        paths = [["a", "b"], ["b", "c"], ["c", "a"]]
        with self.assertRaises(ValueError) as context:
            toposort(paths)
        self.assertIn("Cycle detected", str(context.exception))

    def test_upstream_map(self):
        paths = [["a", "b", "c"], ["d", "c"], ["e", "c"]]
        expected_upstream = {
            "a": set(),
            "b": {"a"},
            "c": {"b", "d", "e"},
            "d": set(),
            "e": set(),
        }
        _, upstream = toposort(paths)
        self.assertEqual(upstream, expected_upstream)

    def test_empty_graph(self):
        paths = []
        sorted_nodes, upstream = toposort(paths)
        self.assertEqual(sorted_nodes, [])
        self.assertEqual(upstream, {})

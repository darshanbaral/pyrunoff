import unittest
import unittest.mock
import warnings

from pyrunoff.utils import compat


class TestPythonJit(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch.object(compat, "_warned", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_runs_function_uncompiled(self):
        @compat.python_jit(nopython=True, cache=True)
        def add(a, b):
            return a + b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(add(2, 3), 5)
        self.assertEqual(add.__name__, "add")

    def test_warns_once_across_functions_and_calls(self):
        @compat.python_jit(nopython=True)
        def first():
            return 1

        @compat.python_jit(nopython=True)
        def second():
            return 2

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                first()
                second()

        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RuntimeWarning)
        self.assertIn("Numba not found", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()

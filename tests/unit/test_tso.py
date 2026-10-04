import unittest

import pandas

from pyrunoff.utils.tso import fix_ts_end


class TestTso(unittest.TestCase):
    def test_fix_ts_end(self):
        original = pandas.Series(
            [1.0 for _ in range(20)] + [5.0 for _ in range(5)],
            index=pandas.date_range(
                "2017-01-01 00:00:00", "2017-01-02 00:00:00", freq="1h"
            ),
        )
        hour_delta = 5
        end1 = original.index[-1] - pandas.Timedelta(hours=hour_delta)
        end2 = original.index[-1] + pandas.Timedelta(hours=hour_delta)
        clipped = fix_ts_end(original, end1, pandas.Timedelta(hours=1))
        extended = fix_ts_end(original, end2, pandas.Timedelta(hours=1))

        assert clipped.index[-1] == original.index[-1] - pandas.Timedelta(
            hours=hour_delta
        )
        assert clipped.iloc[-1] == 1
        assert extended.index[-1] == original.index[-1] + pandas.Timedelta(
            hours=hour_delta
        )
        assert extended.iloc[-1] == 0
        assert extended.loc[original.index].equals(original)

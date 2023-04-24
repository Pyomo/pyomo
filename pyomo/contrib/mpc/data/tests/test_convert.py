#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import pytest
import random

import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentMap
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import (
    _process_to_dynamic_data,
    interval_to_series,
    series_to_interval,
)


def _make_model():
    m = pyo.ConcreteModel()
    m.time = pyo.Set(initialize=[0.1 * i for i in range(11)])
    m.comp = pyo.Set(initialize=["A", "B"])
    m.var = pyo.Var(m.time, m.comp, initialize=1.0)
    return m


class TestIntervalToSeries(unittest.TestCase):
    def test_no_time_points(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.7, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)

        series_data = interval_to_series(interval_data)
        # Default uses right endpoint of each interval
        pred_time_points = [0.2, 0.5, 1.0]
        pred_data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        self.assertEqual(series_data, TimeSeriesData(pred_data, pred_time_points))

    def test_no_time_points_left_endpoints(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.7, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)

        series_data = interval_to_series(interval_data, use_left_endpoints=True)
        pred_time_points = [0.0, 0.2, 0.7]
        pred_data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        self.assertEqual(series_data, TimeSeriesData(pred_data, pred_time_points))

    def test_time_points_provided_no_boundary(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)

        # Choose some time points that don't lie on interval boundaries
        time_points = [0.05 + i * 0.1 for i in range(10)]
        series_data = interval_to_series(interval_data, time_points=time_points)
        pred_data = {
            m.var[:, "A"]: [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            m.var[:, "B"]: [4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0],
        }
        self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))

    def test_time_points_provided_some_on_boundary(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)

        time_points = [0.1 * i for i in range(11)]
        series_data = interval_to_series(interval_data, time_points=time_points)
        # Some of the time points are on interval boundaries. By default we
        # use the values from the intervals on the left.
        pred_data = {
            m.var[:, "A"]: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            m.var[:, "B"]: [4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0],
        }
        self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))

    def test_time_points_provided_some_on_boundary_use_right(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)

        time_points = [0.1 * i for i in range(11)]
        series_data = interval_to_series(
            interval_data, time_points=time_points, prefer_left=False
        )
        # Some of the time points are on interval boundaries. By default we
        # use the values from the intervals on the left.
        pred_data = {
            m.var[:, "A"]: [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            m.var[:, "B"]: [4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
        }
        self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))

    def test_with_roundoff_error(self):
        m = _make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)

        # Simulate roundoff error in these time points.
        random.seed(12710)
        time_points = [i * 0.1 + random.uniform(-1e-8, 1e-8) for i in range(11)]
        series_data = interval_to_series(
            interval_data, time_points=time_points, tolerance=1e-7
        )
        pred_data = {
            m.var[:, "A"]: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            m.var[:, "B"]: [4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0],
        }
        self.assertEqual(series_data, TimeSeriesData(pred_data, time_points))


class TestSeriesToInterval(unittest.TestCase):
    def test_singleton(self):
        m = _make_model()
        time_points = [0.1]
        data = {m.var[:, "A"]: [0.5], m.var[:, "B"]: [2.0]}
        series_data = TimeSeriesData(data, time_points)
        interval_data = series_to_interval(series_data)
        pred_data = IntervalData(
            {m.var[:, "A"]: [0.5], m.var[:, "B"]: [2.0]}, [(0.1, 0.1)]
        )
        self.assertEqual(interval_data, pred_data)

    def test_convert(self):
        m = _make_model()
        time_points = [0.1, 0.2, 0.3, 0.4, 0.5]
        data = {
            m.var[:, "A"]: [1.0, 2.0, 3.0, 4.0, 5.0],
            m.var[:, "B"]: [6.0, 7.0, 8.0, 9.0, 10.0],
        }
        series_data = TimeSeriesData(data, time_points)
        interval_data = series_to_interval(series_data)

        pred_data = IntervalData(
            {m.var[:, "A"]: [2.0, 3.0, 4.0, 5.0], m.var[:, "B"]: [7.0, 8.0, 9.0, 10.0]},
            [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)],
        )
        self.assertEqual(pred_data, interval_data)

    def test_convert_use_right(self):
        m = _make_model()
        time_points = [0.1, 0.2, 0.3, 0.4, 0.5]
        data = {
            m.var[:, "A"]: [1.0, 2.0, 3.0, 4.0, 5.0],
            m.var[:, "B"]: [6.0, 7.0, 8.0, 9.0, 10.0],
        }
        series_data = TimeSeriesData(data, time_points)
        interval_data = series_to_interval(series_data, use_left_endpoints=True)

        pred_data = IntervalData(
            {m.var[:, "A"]: [1.0, 2.0, 3.0, 4.0], m.var[:, "B"]: [6.0, 7.0, 8.0, 9.0]},
            [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)],
        )
        self.assertEqual(pred_data, interval_data)


class TestProcessToDynamic(unittest.TestCase):
    def test_non_time_indexed_data(self):
        m = _make_model()
        m.scalar_var = pyo.Var(m.comp, initialize=3.0)
        data = ComponentMap([(m.scalar_var["A"], 3.1), (m.scalar_var["B"], 3.2)])
        # Passing non-time-indexed data to ScalarData just returns
        # a ScalarData object with the non-time-indexed CUIDs as keys.
        dyn_data = _process_to_dynamic_data(data)
        self.assertTrue(isinstance(dyn_data, ScalarData))
        self.assertIn(pyo.ComponentUID(m.scalar_var["A"]), dyn_data.get_data())
        self.assertIn(pyo.ComponentUID(m.scalar_var["B"]), dyn_data.get_data())


if __name__ == "__main__":
    unittest.main()

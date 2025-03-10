#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

import pyomo.common.unittest as unittest
import pytest

import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData


class TestIntervalData(unittest.TestCase):
    def _make_model(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[0.1 * i for i in range(11)])
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(m.time, m.comp, initialize=1.0)
        return m

    def test_construct(self):
        m = self._make_model()
        intervals = [(0.0, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0], m.var[:, "B"]: [3.0, 4.0]}
        interval_data = IntervalData(data, intervals)

        self.assertEqual(
            interval_data.get_data(),
            {pyo.ComponentUID(key): val for key, val in data.items()},
        )
        self.assertEqual(intervals, interval_data.get_intervals())

    def test_eq(self):
        m = self._make_model()
        intervals = [(0.0, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0], m.var[:, "B"]: [3.0, 4.0]}
        interval_data_1 = IntervalData(data, intervals)

        data = {m.var[:, "A"]: [1.0, 2.0], m.var[:, "B"]: [3.0, 4.0]}
        interval_data_2 = IntervalData(data, intervals)

        self.assertEqual(interval_data_1, interval_data_2)

        data = {m.var[:, "A"]: [1.0, 3.0], m.var[:, "B"]: [3.0, 4.0]}
        interval_data_3 = IntervalData(data, intervals)

        self.assertNotEqual(interval_data_1, interval_data_3)

    def test_get_data_at_indices_multiple(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        data = interval_data.get_data_at_interval_indices([0, 2])

        pred_data = IntervalData(
            {m.var[:, "A"]: [1.0, 3.0], m.var[:, "B"]: [4.0, 6.0]},
            [(0.0, 0.2), (0.5, 1.0)],
        )
        self.assertEqual(pred_data, data)

    def test_get_data_at_indices_singleton(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        data = interval_data.get_data_at_interval_indices(1)
        pred_data = ScalarData({m.var[:, "A"]: 2.0, m.var[:, "B"]: 5.0})
        self.assertEqual(data, pred_data)

    def test_get_data_at_time_scalar(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)

        data = interval_data.get_data_at_time(0.1)
        pred_data = ScalarData({m.var[:, "A"]: 1.0, m.var[:, "B"]: 4.0})
        self.assertEqual(data, pred_data)

        # Default is to allow time points outside of intervals
        # (finds the nearest interval)
        data = interval_data.get_data_at_time(1.1)
        pred_data = ScalarData({m.var[:, "A"]: 3.0, m.var[:, "B"]: 6.0})
        self.assertEqual(data, pred_data)

        msg = "Time point.*not found"
        with self.assertRaisesRegex(RuntimeError, msg):
            data = interval_data.get_data_at_time(1.1, tolerance=1e-3)

        # If a point on an interval boundary is supplied, default is to
        # use value on left.
        data = interval_data.get_data_at_time(0.5)
        pred_data = ScalarData({m.var[:, "A"]: 2.0, m.var[:, "B"]: 5.0})
        self.assertEqual(data, pred_data)

        data = interval_data.get_data_at_time(0.5, prefer_left=False)
        pred_data = ScalarData({m.var[:, "A"]: 3.0, m.var[:, "B"]: 6.0})
        self.assertEqual(data, pred_data)

    def test_to_serializable(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        json_data = interval_data.to_serializable()
        self.assertEqual(
            json_data,
            (
                {"var[*,A]": [1.0, 2.0, 3.0], "var[*,B]": [4.0, 5.0, 6.0]},
                [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)],
            ),
        )

    def test_concatenate(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data_1 = IntervalData(data, intervals)

        intervals = [(1.0, 1.5), (2.0, 3.0)]
        data = {m.var[:, "A"]: [7.0, 8.0], m.var[:, "B"]: [9.0, 10.0]}
        interval_data_2 = IntervalData(data, intervals)

        interval_data_1.concatenate(interval_data_2)

        new_intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (2.0, 3.0)]
        new_values = {
            m.var[:, "A"]: [1.0, 2.0, 3.0, 7.0, 8.0],
            m.var[:, "B"]: [4.0, 5.0, 6.0, 9.0, 10.0],
        }
        new_data = IntervalData(new_values, new_intervals)
        self.assertEqual(interval_data_1, new_data)

    def test_shift_time_points(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals)
        interval_data.shift_time_points(1.0)

        intervals = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        new_interval_data = IntervalData(data, intervals)
        self.assertEqual(interval_data, new_interval_data)

    def test_extract_variables(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals, time_set=m.time)
        new_data = interval_data.extract_variables([m.var[:, "B"]])
        value_dict = {m.var[:, "B"]: [4.0, 5.0, 6.0]}
        pred_data = IntervalData(value_dict, intervals)
        self.assertEqual(new_data, pred_data)

    def test_extract_variables_exception(self):
        m = self._make_model()
        intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
        data = {m.var[:, "A"]: [1.0, 2.0, 3.0], m.var[:, "B"]: [4.0, 5.0, 6.0]}
        interval_data = IntervalData(data, intervals, time_set=m.time)
        msg = "only accepts a list or tuple"
        with self.assertRaisesRegex(TypeError, msg):
            new_data = interval_data.extract_variables(m.var[:, "B"])


class TestAssertDisjoint(unittest.TestCase):
    def test_disjoint(self):
        intervals = [(0, 1), (1, 2)]
        assert_disjoint_intervals(intervals)

        intervals = [(2, 3), (0, 1)]
        assert_disjoint_intervals(intervals)

        intervals = [(0, 1), (1, 1)]
        assert_disjoint_intervals(intervals)

    def test_backwards_endpoints(self):
        intervals = [(0, 1), (3, 2)]
        msg = "Lower endpoint of interval is higher"
        with self.assertRaisesRegex(RuntimeError, msg):
            assert_disjoint_intervals(intervals)

    def test_not_disjoint(self):
        intervals = [(0, 2), (1, 3)]
        msg = "are not disjoint"
        with self.assertRaisesRegex(RuntimeError, msg):
            assert_disjoint_intervals(intervals)


class TestLoadInputs(unittest.TestCase):
    def make_model(self):
        m = pyo.ConcreteModel()
        m.time = dae.ContinuousSet(initialize=[0, 1, 2, 3, 4, 5, 6])
        m.v = pyo.Var(m.time, initialize=0)
        return m

    def test_load_inputs_some_time(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({"v": [1.0]}, [(2, 4)])
        interface.load_data(inputs)

        for t in m.time:
            # Note that by default, the left endpoint is not loaded.
            if t == 3 or t == 4:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 0.0)

    def test_load_inputs_some_time_include_endpoints(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({"v": [1.0]}, [(2, 4)])

        # Default is to exclude right and include left
        interface.load_data(inputs, exclude_left_endpoint=False)

        for t in m.time:
            if t == 2 or t == 3 or t == 4:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 0.0)

    def test_load_inputs_some_time_exclude_endpoints(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({"v": [1.0]}, [(2, 4)])

        # Default is to exclude right and include left
        interface.load_data(inputs, exclude_right_endpoint=True)

        for t in m.time:
            if t == 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 0.0)

    def test_load_inputs_all_time_default(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({"v": [1.0, 2.0]}, [(0, 3), (3, 6)])
        interface.load_data(inputs)
        for t in m.time:
            if t == 0:
                self.assertEqual(m.v[t].value, 0.0)
            elif t <= 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def test_load_inputs_all_time_prefer_right(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({"v": [1.0, 2.0]}, [(0, 3), (3, 6)])
        interface.load_data(inputs, prefer_left=False)
        for t in m.time:
            if t < 3:
                self.assertEqual(m.v[t].value, 1.0)
            elif t == 6:
                # By default, preferring intervals to the right of time
                # points will exclude the right endpoints of intervals.
                self.assertEqual(m.v[t].value, 0.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def test_load_inputs_all_time_prefer_right_dont_exclude(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({"v": [1.0, 2.0]}, [(0, 3), (3, 6)])
        interface.load_data(inputs, prefer_left=False, exclude_right_endpoint=False)
        # Note that all time points have been set.
        for t in m.time:
            if t < 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def load_inputs_invalid_time(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = mpc.IntervalData({"v": [1.0, 2.0, 3.0]}, [(0, 3), (3, 6), (6, 9)])
        interface.load_data(inputs)
        for t in m.time:
            if t == 0:
                self.assertEqual(m.v[t].value, 0.0)
            elif t <= 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def load_inputs_exception(self):
        m = self.make_model()
        interface = mpc.DynamicModelInterface(m, m.time)
        inputs = {"_v": {(0, 3): 1.0, (3, 6): 2.0, (6, 9): 3.0}}
        inputs = mpc.IntervalData({"_v": [1.0, 2.0, 3.0]}, [(0, 3), (3, 6), (6, 9)])
        with self.assertRaisesRegex(RuntimeError, "Cannot find"):
            interface.load_data(inputs)


class TestIntervalFromTimeSeries(unittest.TestCase):
    def test_singleton(self):
        name = "name"
        series = mpc.TimeSeriesData({name: [2.0]}, [1.0])
        interval = mpc.data.convert.series_to_interval(series)
        self.assertEqual(interval, IntervalData({name: [2.0]}, [(1.0, 1.0)]))

    def test_empty(self):
        name = "name"
        series = mpc.TimeSeriesData({name: []}, [])
        interval = mpc.data.convert.series_to_interval(series)
        self.assertEqual(interval, mpc.IntervalData({name: []}, []))

    def test_interval_from_series(self):
        name = "name"
        series = mpc.TimeSeriesData({name: [4.0, 5.0, 6.0]}, [1, 2, 3])
        interval = mpc.data.convert.series_to_interval(series)
        self.assertEqual(
            interval, mpc.IntervalData({name: [5.0, 6.0]}, [(1, 2), (2, 3)])
        )

    def test_use_left_endpoint(self):
        name = "name"
        series = mpc.TimeSeriesData({name: [4.0, 5.0, 6.0]}, [1, 2, 3])
        interval = mpc.data.convert.series_to_interval(series, use_left_endpoints=True)
        self.assertEqual(
            interval, mpc.IntervalData({name: [4.0, 5.0]}, [(1, 2), (2, 3)])
        )


if __name__ == "__main__":
    unittest.main()

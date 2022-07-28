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
from idaes.apps.nmpc.dynamic_data import (
    assert_disjoint_intervals,
    load_inputs_into_model,
    interval_data_from_time_series,
)

@pytest.mark.unit
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


@pytest.mark.unit
class TestLoadInputs(unittest.TestCase):

    def make_model(self):
        m = pyo.ConcreteModel()
        m.time = dae.ContinuousSet(initialize=[0, 1, 2, 3, 4, 5, 6])
        m.v = pyo.Var(m.time, initialize=0)
        return m

    def test_load_inputs_some_time(self):
        m = self.make_model()
        inputs = {
            "v": {(2, 4): 1.0}
        }
        load_inputs_into_model(m, m.time, inputs)

        for t in m.time:
            if t == 3 or t == 4:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 0.0)

    def test_load_inputs_all_time(self):
        m = self.make_model()
        inputs = {
            "v": {(0, 3): 1.0, (3, 6): 2.0},
        }
        load_inputs_into_model(m, m.time, inputs)
        for t in m.time:
            if t == 0:
                self.assertEqual(m.v[t].value, 0.0)
            elif t <= 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def load_inputs_invalid_time(self):
        m = self.make_model()
        inputs = {
            "v": {(0, 3): 1.0, (3, 6): 2.0, (6, 9): 3.0},
        }
        load_inputs_into_model(m, m.time, inputs)
        for t in m.time:
            if t == 0:
                self.assertEqual(m.v[t].value, 0.0)
            elif t <= 3:
                self.assertEqual(m.v[t].value, 1.0)
            else:
                self.assertEqual(m.v[t].value, 2.0)

    def load_inputs_exception(self):
        m = self.make_model()
        inputs = {
            "_v": {(0, 3): 1.0, (3, 6): 2.0, (6, 9): 3.0},
        }
        with self.assertRaisesRegex(RuntimeError, "Could not find"):
            load_inputs_into_model(m, m.time, inputs)


@pytest.mark.unit
class TestIntervalFromTimeSeries(unittest.TestCase):

    def test_singleton(self):
        name = "name"
        series = (
            [1.0],
            {
                name: [2.0],
            },
        )
        interval = interval_data_from_time_series(series)
        self.assertEqual(
            interval,
            {name: {(1.0, 1.0): 2.0}},
        )

    def test_empty(self):
        name = "name"
        series = ([], {name: []})
        interval = interval_data_from_time_series(series)
        self.assertEqual(interval, {name: {}})

    def test_interval_from_series(self):
        name = "name"
        series = (
            [1, 2, 3],
            {
                name: [4.0, 5.0, 6.0],
            },
        )
        interval = interval_data_from_time_series(series)
        self.assertEqual(
            interval,
            {
                name: {(1, 2): 5.0, (2, 3): 6.0},
            },
        )

    def test_use_left_endpoint(self):
        name = "name"
        series = (
            [1, 2, 3],
            {
                name: [4.0, 5.0, 6.0],
            },
        )
        interval = interval_data_from_time_series(
            series,
            use_left_endpoint=True,
        )
        self.assertEqual(
            interval,
            {
                name: {(1, 2): 4.0, (2, 3): 5.0},
            },
        )

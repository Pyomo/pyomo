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

import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
    get_penalty_from_constant_target,
    get_penalty_from_piecewise_constant_target,
    get_penalty_from_time_varying_target,
    get_penalty_from_target,
)
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData


class TestTrackingCostConstantSetpoint(unittest.TestCase):
    def test_penalty_no_weights(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[1, 2, 3])
        m.v1 = pyo.Var(m.time, initialize={i: 1 * i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2 * i for i in m.time})

        setpoint_data = ScalarData({m.v1[:]: 3.0, m.v2[:]: 4.0})
        variables = [m.v1, m.v2]
        m.var_set, m.tracking_expr = get_penalty_from_constant_target(
            variables, m.time, setpoint_data
        )
        self.assertEqual(len(m.var_set), 2)
        self.assertIn(0, m.var_set)
        self.assertIn(1, m.var_set)

        var_sets = {
            (i, t): ComponentSet(identify_variables(m.tracking_expr[i, t]))
            for i in m.var_set
            for t in m.time
        }
        for i in m.time:
            for j in m.var_set:
                self.assertIn(variables[j][i], var_sets[j, i])
                pred_value = (1 * i - 3) ** 2 if j == 0 else (2 * i - 4) ** 2
                self.assertEqual(pred_value, pyo.value(m.tracking_expr[j, i]))
                pred_expr = (m.v1[i] - 3) ** 2 if j == 0 else (m.v2[i] - 4) ** 2
                self.assertTrue(
                    compare_expressions(pred_expr, m.tracking_expr[j, i].expr)
                )

    def test_penalty_with_weights(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[1, 2, 3])
        m.v1 = pyo.Var(m.time, initialize={i: 1 * i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2 * i for i in m.time})

        setpoint_data = ScalarData({m.v1[:]: 3.0, m.v2[:]: 4.0})
        weight_data = ScalarData({m.v1[:]: 0.1, m.v2[:]: 0.5})
        m.var_set = pyo.Set(initialize=[0, 1])
        variables = [m.v1, m.v2]
        new_set, m.tracking_expr = get_penalty_from_constant_target(
            variables,
            m.time,
            setpoint_data,
            weight_data=weight_data,
            variable_set=m.var_set,
        )
        self.assertIs(new_set, m.var_set)

        var_sets = {
            (i, t): ComponentSet(identify_variables(m.tracking_expr[i, t]))
            for i in m.var_set
            for t in m.time
        }
        for i in m.time:
            for j in m.var_set:
                self.assertIn(variables[j][i], var_sets[j, i])
                pred_value = (
                    0.1 * (1 * i - 3) ** 2 if j == 0 else 0.5 * (2 * i - 4) ** 2
                )
                self.assertAlmostEqual(pred_value, pyo.value(m.tracking_expr[j, i]))
                pred_expr = (
                    0.1 * (m.v1[i] - 3) ** 2 if j == 0 else 0.5 * (m.v2[i] - 4) ** 2
                )
                self.assertTrue(
                    compare_expressions(pred_expr, m.tracking_expr[j, i].expr)
                )

    def test_exceptions(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[1, 2, 3])
        m.v1 = pyo.Var(m.time, initialize={i: 1 * i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2 * i for i in m.time})

        setpoint_data = ScalarData({m.v1[:]: 3.0})
        weight_data = ScalarData({m.v2[:]: 0.1})
        with self.assertRaisesRegex(KeyError, "Setpoint data"):
            _, m.tracking_expr = get_penalty_from_constant_target(
                [m.v1, m.v2], m.time, setpoint_data
            )

        setpoint_data = ScalarData({m.v1[:]: 3.0, m.v2[:]: 4.0})
        with self.assertRaisesRegex(KeyError, "Tracking weight"):
            _, m.tracking_expr = get_penalty_from_constant_target(
                [m.v1, m.v2], m.time, setpoint_data, weight_data=weight_data
            )

    def test_add_set_after_expr(self):
        # A small gotcha that may come up. This is known behavior
        # due to Pyomo's "implicit set" addition.
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[1, 2, 3])
        m.v1 = pyo.Var(m.time, initialize={i: 1 * i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2 * i for i in m.time})

        setpoint_data = ScalarData({m.v1[:]: 3.0, m.v2[:]: 4.0})
        weight_data = ScalarData({m.v1[:]: 0.1, m.v2[:]: 0.5})
        m.var_set = pyo.Set(initialize=[0, 1])
        variables = [m.v1, m.v2]
        new_set, tr_expr = get_penalty_from_constant_target(
            variables,
            m.time,
            setpoint_data,
            weight_data=weight_data,
            variable_set=m.var_set,
        )
        m.tracking_expr = tr_expr  # new_set gets added and assigned a name
        msg = "Attempting to re-assign"
        with self.assertRaisesRegex(RuntimeError, msg):
            # attempting to add the same component twice
            m.variable_set = new_set


class TestTrackingCostPiecewiseSetpoint(unittest.TestCase):
    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=list(range(n_time_points)))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time, m.comp, initialize={(i, j): 1.1 * i for i, j in m.time * m.comp}
        )
        return m

    def test_piecewise_penalty_no_weights(self):
        m = self._make_model(n_time_points=5)

        variables = [pyo.Reference(m.var[:, "A"]), pyo.Reference(m.var[:, "B"])]
        setpoint_data = IntervalData(
            {m.var[:, "A"]: [2.0, 2.5], m.var[:, "B"]: [3.0, 3.5]}, [(0, 2), (2, 4)]
        )
        m.var_set, m.tracking_cost = get_penalty_from_piecewise_constant_target(
            variables, m.time, setpoint_data
        )
        for i in m.time:
            for j in m.var_set:
                if i <= 2:
                    pred_expr = (
                        (m.var[i, "A"] - 2.0) ** 2
                        if j == 0
                        else (m.var[i, "B"] - 3.0) ** 2
                    )
                else:
                    pred_expr = (
                        (m.var[i, "A"] - 2.5) ** 2
                        if j == 0
                        else (m.var[i, "B"] - 3.5) ** 2
                    )
                pred_value = pyo.value(pred_expr)
                self.assertEqual(pred_value, pyo.value(m.tracking_cost[j, i]))
                self.assertTrue(
                    compare_expressions(pred_expr, m.tracking_cost[j, i].expr)
                )

    def test_piecewise_penalty_with_weights(self):
        m = self._make_model(n_time_points=5)

        variables = [pyo.Reference(m.var[:, "A"]), pyo.Reference(m.var[:, "B"])]
        setpoint_data = IntervalData(
            {m.var[:, "A"]: [2.0, 2.5], m.var[:, "B"]: [3.0, 3.5]}, [(0, 2), (2, 4)]
        )
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.var[:, "B"]): 0.1,
        }
        m.var_set, m.tracking_cost = get_penalty_from_piecewise_constant_target(
            variables, m.time, setpoint_data, weight_data=weight_data
        )
        for i in m.time:
            for j in m.var_set:
                if i <= 2:
                    pred_expr = (
                        10.0 * (m.var[i, "A"] - 2.0) ** 2
                        if j == 0
                        else 0.1 * (m.var[i, "B"] - 3.0) ** 2
                    )
                else:
                    pred_expr = (
                        10.0 * (m.var[i, "A"] - 2.5) ** 2
                        if j == 0
                        else 0.1 * (m.var[i, "B"] - 3.5) ** 2
                    )
                pred_value = pyo.value(pred_expr)
                self.assertEqual(pred_value, pyo.value(m.tracking_cost[j, i]))
                self.assertTrue(
                    compare_expressions(pred_expr, m.tracking_cost[j, i].expr)
                )

    def test_piecewise_penalty_exceptions(self):
        m = self._make_model(n_time_points=5)

        variables = [pyo.Reference(m.var[:, "A"]), pyo.Reference(m.var[:, "B"])]
        setpoint_data = IntervalData({m.var[:, "A"]: [2.0, 2.5]}, [(0, 2), (2, 4)])
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.var[:, "B"]): 0.1,
        }
        msg = "Setpoint data does not contain"
        with self.assertRaisesRegex(KeyError, msg):
            tr_cost = get_penalty_from_piecewise_constant_target(
                variables, m.time, setpoint_data, weight_data=weight_data
            )

        setpoint_data = IntervalData(
            {m.var[:, "A"]: [2.0, 2.5], m.var[:, "B"]: [3.0, 3.5]}, [(0, 2), (2, 4)]
        )
        weight_data = {pyo.ComponentUID(m.var[:, "A"]): 10.0}
        msg = "Tracking weight does not contain"
        with self.assertRaisesRegex(KeyError, msg):
            tr_cost = get_penalty_from_piecewise_constant_target(
                variables, m.time, setpoint_data, weight_data=weight_data
            )


class TestTrackingCostVaryingSetpoint(unittest.TestCase):
    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=list(range(n_time_points)))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time, m.comp, initialize={(i, j): 1.1 * i for i, j in m.time * m.comp}
        )
        return m

    def test_varying_setpoint_no_weights(self):
        m = self._make_model(n_time_points=5)
        variables = [pyo.Reference(m.var[:, "A"]), pyo.Reference(m.var[:, "B"])]
        A_setpoint = [1.0 - 0.1 * i for i in range(len(m.time))]
        B_setpoint = [5.0 + 0.1 * i for i in range(len(m.time))]
        setpoint_data = TimeSeriesData(
            {m.var[:, "A"]: A_setpoint, m.var[:, "B"]: B_setpoint}, m.time
        )
        m.var_set, m.tracking_cost = get_penalty_from_time_varying_target(
            variables, m.time, setpoint_data
        )
        for i, t in enumerate(m.time):
            for j in m.var_set:
                pred_expr = (
                    (m.var[t, "A"] - A_setpoint[i]) ** 2
                    if j == 0
                    else (m.var[t, "B"] - B_setpoint[i]) ** 2
                )
                pred_value = pyo.value(pred_expr)
                self.assertEqual(pred_value, pyo.value(m.tracking_cost[j, t]))
                self.assertTrue(
                    compare_expressions(pred_expr, m.tracking_cost[j, t].expr)
                )

    def test_varying_setpoint_with_weights(self):
        m = self._make_model(n_time_points=5)
        variables = [pyo.Reference(m.var[:, "A"]), pyo.Reference(m.var[:, "B"])]
        A_setpoint = [1.0 - 0.1 * i for i in range(len(m.time))]
        B_setpoint = [5.0 + 0.1 * i for i in range(len(m.time))]
        setpoint_data = TimeSeriesData(
            {m.var[:, "A"]: A_setpoint, m.var[:, "B"]: B_setpoint}, m.time
        )
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.var[:, "B"]): 0.1,
        }
        m.var_set, m.tracking_cost = get_penalty_from_time_varying_target(
            variables, m.time, setpoint_data, weight_data=weight_data
        )
        for i, t in enumerate(m.time):
            for j in m.var_set:
                pred_expr = (
                    10.0 * (m.var[t, "A"] - A_setpoint[i]) ** 2
                    if j == 0
                    else 0.1 * (m.var[t, "B"] - B_setpoint[i]) ** 2
                )
                pred_value = pyo.value(pred_expr)
                self.assertEqual(pred_value, pyo.value(m.tracking_cost[j, t]))
                self.assertTrue(
                    compare_expressions(pred_expr, m.tracking_cost[j, t].expr)
                )

    def test_varying_setpoint_exceptions(self):
        m = self._make_model(n_time_points=5)
        variables = [pyo.Reference(m.var[:, "A"]), pyo.Reference(m.var[:, "B"])]
        A_setpoint = [1.0 - 0.1 * i for i in range(len(m.time))]
        B_setpoint = [5.0 + 0.1 * i for i in range(len(m.time))]
        setpoint_data = TimeSeriesData(
            {m.var[:, "A"]: A_setpoint, m.var[:, "B"]: B_setpoint},
            [i + 10 for i in m.time],
        )
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.var[:, "B"]): 0.1,
        }
        msg = "Mismatch in time points"
        with self.assertRaisesRegex(RuntimeError, msg):
            # Time-varying setpoint specifies different time points
            # from our time set.
            var_set, tr_cost = get_penalty_from_time_varying_target(
                variables, m.time, setpoint_data, weight_data=weight_data
            )

        setpoint_data = TimeSeriesData({m.var[:, "A"]: A_setpoint}, m.time)
        msg = "Setpoint data does not contain"
        with self.assertRaisesRegex(KeyError, msg):
            var_set, tr_cost = get_penalty_from_time_varying_target(
                variables, m.time, setpoint_data, weight_data=weight_data
            )

        setpoint_data = TimeSeriesData(
            {m.var[:, "A"]: A_setpoint, m.var[:, "B"]: B_setpoint}, m.time
        )
        weight_data = {pyo.ComponentUID(m.var[:, "A"]): 10.0}
        msg = "Tracking weight does not contain"
        with self.assertRaisesRegex(KeyError, msg):
            tr_cost = get_penalty_from_time_varying_target(
                variables, m.time, setpoint_data, weight_data=weight_data
            )


class TestGetPenaltyFromTarget(unittest.TestCase):
    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=list(range(n_time_points)))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time, m.comp, initialize={(i, j): 1.1 * i for i, j in m.time * m.comp}
        )
        return m

    def test_constant_setpoint(self):
        m = self._make_model()
        setpoint = {m.var[:, "A"]: 0.3, m.var[:, "B"]: 0.4}
        variables = [pyo.Reference(m.var[:, "B"]), pyo.Reference(m.var[:, "A"])]
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
        pred_expr = {
            (i, t): (
                (m.var[t, "B"] - 0.4) ** 2 if i == 0 else (m.var[t, "A"] - 0.3) ** 2
            )
            for i, t in m.var_set * m.time
        }
        for t in m.time:
            for i in m.var_set:
                self.assertTrue(
                    compare_expressions(pred_expr[i, t], m.penalty[i, t].expr)
                )
                self.assertEqual(pyo.value(pred_expr[i, t]), pyo.value(m.penalty[i, t]))

    def test_constant_setpoint_with_ScalarData(self):
        m = self._make_model()
        setpoint = ScalarData({m.var[:, "A"]: 0.3, m.var[:, "B"]: 0.4})
        variables = [pyo.Reference(m.var[:, "B"]), pyo.Reference(m.var[:, "A"])]
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
        pred_expr = {
            (i, t): (
                (m.var[t, "B"] - 0.4) ** 2 if i == 0 else (m.var[t, "A"] - 0.3) ** 2
            )
            for i, t in m.var_set * m.time
        }
        for t in m.time:
            for i in m.var_set:
                self.assertTrue(
                    compare_expressions(pred_expr[i, t], m.penalty[i, t].expr)
                )
                self.assertEqual(pyo.value(pred_expr[i, t]), pyo.value(m.penalty[i, t]))

    def test_varying_setpoint(self):
        m = self._make_model(n_time_points=5)
        A_target = [0.4, 0.6, 0.1, 0.0, 1.1]
        B_target = [0.8, 0.9, 1.3, 1.5, 1.4]
        setpoint = ({m.var[:, "A"]: A_target, m.var[:, "B"]: B_target}, m.time)
        variables = [pyo.Reference(m.var[:, "B"]), pyo.Reference(m.var[:, "A"])]
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)

        target = {
            (i, t): A_target[j] if i == 1 else B_target[t]
            for i in m.var_set
            for (j, t) in enumerate(m.time)
        }
        for i, t in m.var_set * m.time:
            pred_expr = (variables[i][t] - target[i, t]) ** 2
            self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))

    def test_piecewise_constant_setpoint(self):
        m = self._make_model(n_time_points=5)
        A_target = [0.3, 0.9, 0.7]
        B_target = [1.1, 0.1, 0.5]
        setpoint = (
            {m.var[:, "A"]: A_target, m.var[:, "B"]: B_target},
            [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)],
        )
        variables = [pyo.Reference(m.var[:, "B"]), pyo.Reference(m.var[:, "A"])]
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
        target = {
            (i, j): A_target[j] if i == 1 else B_target[j]
            for i in m.var_set
            for j in range(len(A_target))
        }
        for i, t in m.var_set * m.time:
            if t == 0:
                idx = 0
            elif t <= 2.0:
                idx = 1
            elif t <= 4.0:
                idx = 2
            pred_expr = (variables[i][t] - target[i, idx]) ** 2
            self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))

    def test_bad_argument(self):
        m = self._make_model(n_time_points=3)
        A_target = [0.4, 0.6, 0.1]
        B_target = [0.8, 0.9, 1.3]
        setpoint = ({m.var[:, "A"]: A_target, m.var[:, "B"]: B_target}, m.time)
        variables = [pyo.Reference(m.var[:, "B"]), pyo.Reference(m.var[:, "A"])]
        msg = "tolerance.*can only be used"
        with self.assertRaisesRegex(RuntimeError, msg):
            m.var_set, m.penalty = get_penalty_from_target(
                variables, m.time, setpoint, tolerance=1e-8
            )

    def test_bad_data_tuple(self):
        m = self._make_model(n_time_points=3)
        A_target = [0.4, 0.6, 0.1]
        B_target = [0.8, 0.9, 1.3]
        setpoint = (
            {m.var[:, "A"]: A_target, m.var[:, "B"]: B_target},
            m.time,
            "something else",
        )
        variables = [pyo.Reference(m.var[:, "B"]), pyo.Reference(m.var[:, "A"])]
        msg = "tuple of length two"
        with self.assertRaisesRegex(TypeError, msg):
            m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)

    def test_bad_data_tuple_entry_0(self):
        m = self._make_model(n_time_points=3)
        A_target = [0.4, 0.6, 0.1]
        B_target = [0.8, 0.9, 1.3]
        setpoint = ([(m.var[:, "A"], A_target), (m.var[:, "B"], B_target)], m.time)
        variables = [pyo.Reference(m.var[:, "B"]), pyo.Reference(m.var[:, "A"])]
        msg = "must be instance of MutableMapping"
        with self.assertRaisesRegex(TypeError, msg):
            m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)

    def test_empty_time_list(self):
        m = self._make_model(n_time_points=3)
        A_target = []
        B_target = []
        setpoint = ({m.var[:, "A"]: A_target, m.var[:, "B"]: B_target}, [])
        variables = [pyo.Reference(m.var[:, "B"]), pyo.Reference(m.var[:, "A"])]
        msg = "Time sequence.*is empty"
        with self.assertRaisesRegex(ValueError, msg):
            m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)

    def test_bad_time_list(self):
        m = self._make_model(n_time_points=3)
        A_target = [0.4, 0.6, 0.1]
        B_target = [0.8, 0.9, 1.3]
        setpoint = (
            dict([(m.var[:, "A"], A_target), (m.var[:, "B"], B_target)]),
            [0.0, (0.1, 0.2), 0.3],
        )
        variables = [pyo.Reference(m.var[:, "B"]), pyo.Reference(m.var[:, "A"])]
        msg = "Second entry of data tuple must be"
        with self.assertRaisesRegex(TypeError, msg):
            m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)


if __name__ == "__main__":
    unittest.main()

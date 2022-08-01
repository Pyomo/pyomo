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
    get_tracking_cost_from_constant_setpoint,
    get_tracking_cost_from_piecewise_constant_setpoint,
    get_tracking_cost_from_time_varying_setpoint,
)
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData


class TestTrackingCostConstantSetpoint(unittest.TestCase):

    def test_tracking_cost_no_weights(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[1, 2, 3])
        m.v1 = pyo.Var(m.time, initialize={i: 1*i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2*i for i in m.time})

        setpoint_data = ScalarData({m.v1[:]: 3.0, m.v2[:]: 4.0})
        m.tracking_expr = get_tracking_cost_from_constant_setpoint(
            [m.v1, m.v2],
            m.time,
            setpoint_data,
        )

        var_sets = {
            i: ComponentSet(identify_variables(m.tracking_expr[i]))
            for i in m.time
        }
        for i in m.time:
            self.assertIn(m.v1[i], var_sets[i])
            self.assertIn(m.v2[i], var_sets[i])
            pred_value = (1*i - 3)**2 + (2*i - 4)**2
            self.assertEqual(pred_value, pyo.value(m.tracking_expr[i]))
            pred_expr = (m.v1[i] - 3)**2 + (m.v2[i] - 4)**2
            self.assertTrue(compare_expressions(
                pred_expr, m.tracking_expr[i].expr
            ))

    def test_tracking_cost_with_weights(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[1, 2, 3])
        m.v1 = pyo.Var(m.time, initialize={i: 1*i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2*i for i in m.time})

        setpoint_data = ScalarData({m.v1[:]: 3.0, m.v2[:]: 4.0})
        weight_data = ScalarData({m.v1[:]: 0.1, m.v2[:]: 0.5})
        m.tracking_expr = get_tracking_cost_from_constant_setpoint(
            [m.v1, m.v2],
            m.time,
            setpoint_data,
            weight_data=weight_data,
        )

        var_sets = {
            i: ComponentSet(identify_variables(m.tracking_expr[i]))
            for i in m.time
        }
        for i in m.time:
            self.assertIn(m.v1[i], var_sets[i])
            self.assertIn(m.v2[i], var_sets[i])
            pred_value = 0.1*(1*i - 3)**2 + 0.5*(2*i - 4)**2
            self.assertAlmostEqual(pred_value, pyo.value(m.tracking_expr[i]))
            pred_expr = 0.1*(m.v1[i] - 3)**2 + 0.5*(m.v2[i] - 4)**2
            self.assertTrue(compare_expressions(
                pred_expr, m.tracking_expr[i].expr
            ))

    def test_exceptions(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[1, 2, 3])
        m.v1 = pyo.Var(m.time, initialize={i: 1*i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2*i for i in m.time})

        setpoint_data = ScalarData({m.v1[:]: 3.0})
        weight_data = ScalarData({m.v2[:]: 0.1})
        with self.assertRaisesRegex(KeyError, "Setpoint data"):
            m.tracking_expr = get_tracking_cost_from_constant_setpoint(
                [m.v1, m.v2],
                m.time,
                setpoint_data,
            )

        setpoint_data = ScalarData({m.v1[:]: 3.0, m.v2[:]: 4.0})
        with self.assertRaisesRegex(KeyError, "Tracking weight"):
            m.tracking_expr = get_tracking_cost_from_constant_setpoint(
                [m.v1, m.v2],
                m.time,
                setpoint_data,
                weight_data=weight_data,
            )


class TestTrackingCostPiecewiseSetpoint(unittest.TestCase):

    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=list(range(n_time_points)))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time,
            m.comp,
            initialize={(i, j): 1.1*i for i, j in m.time*m.comp},
        )
        return m

    def test_piecewise_tracking_cost_no_weights(self):
        m = self._make_model(n_time_points=5)

        variables = [
            pyo.Reference(m.var[:, "A"]),
            pyo.Reference(m.var[:, "B"]),
        ]
        setpoint_data = {
            pyo.ComponentUID(m.var[:, "A"]): {(0, 2): 2.0, (2, 4): 2.5},
            pyo.ComponentUID(m.var[:, "B"]): {(0, 2): 3.0, (2, 4): 3.5},
        }
        m.tracking_cost = get_tracking_cost_from_piecewise_constant_setpoint(
            variables,
            m.time,
            setpoint_data,
        )
        for i in m.time:
            if i <= 2:
                pred_expr = (m.var[i, "A"] - 2.0)**2 + (m.var[i, "B"] - 3.0)**2
            else:
                pred_expr = (m.var[i, "A"] - 2.5)**2 + (m.var[i, "B"] - 3.5)**2
            pred_value = pyo.value(pred_expr)
            self.assertEqual(pred_value, pyo.value(m.tracking_cost[i]))
            self.assertTrue(compare_expressions(
                pred_expr, m.tracking_cost[i].expr
            ))

    def test_piecewise_tracking_cost_with_weights(self):
        m = self._make_model(n_time_points=5)

        variables = [
            pyo.Reference(m.var[:, "A"]),
            pyo.Reference(m.var[:, "B"]),
        ]
        setpoint_data = {
            pyo.ComponentUID(m.var[:, "A"]): {(0, 2): 2.0, (2, 4): 2.5},
            pyo.ComponentUID(m.var[:, "B"]): {(0, 2): 3.0, (2, 4): 3.5},
        }
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.var[:, "B"]): 0.1,
        }
        m.tracking_cost = get_tracking_cost_from_piecewise_constant_setpoint(
            variables,
            m.time,
            setpoint_data,
            weight_data=weight_data,
        )
        for i in m.time:
            if i <= 2:
                pred_expr = (
                    10.0*(m.var[i, "A"] - 2.0)**2
                    + 0.1*(m.var[i, "B"] - 3.0)**2
                )
            else:
                pred_expr = (
                    10.0*(m.var[i, "A"] - 2.5)**2
                    + 0.1*(m.var[i, "B"] - 3.5)**2
                )
            pred_value = pyo.value(pred_expr)
            self.assertEqual(pred_value, pyo.value(m.tracking_cost[i]))
            self.assertTrue(compare_expressions(
                pred_expr, m.tracking_cost[i].expr
            ))

    def test_piecewise_tracking_cost_exceptions(self):
        m = self._make_model(n_time_points=5)

        variables = [
            pyo.Reference(m.var[:, "A"]),
            pyo.Reference(m.var[:, "B"]),
        ]
        setpoint_data = {
            pyo.ComponentUID(m.var[:, "A"]): {(0, 2): 2.0, (2, 4): 2.5},
        }
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.var[:, "B"]): 0.1,
        }
        msg = "Setpoint data dictionary does not contain"
        with self.assertRaisesRegex(KeyError, msg):
            tr_cost = get_tracking_cost_from_piecewise_constant_setpoint(
                variables,
                m.time,
                setpoint_data,
                weight_data=weight_data,
            )

        setpoint_data = {
            pyo.ComponentUID(m.var[:, "A"]): {(0, 2): 2.0, (2, 4): 2.5},
            pyo.ComponentUID(m.var[:, "B"]): {(0, 2): 3.0, (2, 4): 3.5},
        }
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
        }
        msg = "Tracking weight dictionary does not contain"
        with self.assertRaisesRegex(KeyError, msg):
            tr_cost = get_tracking_cost_from_piecewise_constant_setpoint(
                variables,
                m.time,
                setpoint_data,
                weight_data=weight_data,
            )


class TestTrackingCostVaryingSetpoint(unittest.TestCase):

    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=list(range(n_time_points)))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time,
            m.comp,
            initialize={(i, j): 1.1*i for i, j in m.time*m.comp},
        )
        return m

    def test_varying_setpoint_no_weights(self):
        m = self._make_model(n_time_points=5)
        variables = [
            pyo.Reference(m.var[:, "A"]),
            pyo.Reference(m.var[:, "B"]),
        ]
        A_setpoint = [1.0 - 0.1*i for i in range(len(m.time))]
        B_setpoint = [5.0 + 0.1*i for i in range(len(m.time))]
        setpoint_data = TimeSeriesData(
            {m.var[:, "A"]: A_setpoint, m.var[:, "B"]: B_setpoint},
            m.time,
        )
        m.tracking_cost = get_tracking_cost_from_time_varying_setpoint(
            variables,
            m.time,
            setpoint_data,
        )
        for i, t in enumerate(m.time):
            pred_expr = (
                (m.var[t, "A"] - A_setpoint[i])**2
                + (m.var[t, "B"] - B_setpoint[i])**2
            )
            pred_value = pyo.value(pred_expr)
            self.assertEqual(pred_value, pyo.value(m.tracking_cost[t]))
            self.assertTrue(compare_expressions(
                pred_expr, m.tracking_cost[t].expr
            ))

    def test_varying_setpoint_with_weights(self):
        m = self._make_model(n_time_points=5)
        variables = [
            pyo.Reference(m.var[:, "A"]),
            pyo.Reference(m.var[:, "B"]),
        ]
        A_setpoint = [1.0 - 0.1*i for i in range(len(m.time))]
        B_setpoint = [5.0 + 0.1*i for i in range(len(m.time))]
        setpoint_data = TimeSeriesData(
            {m.var[:, "A"]: A_setpoint, m.var[:, "B"]: B_setpoint},
            m.time,
        )
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.var[:, "B"]): 0.1,
        }
        m.tracking_cost = get_tracking_cost_from_time_varying_setpoint(
            variables,
            m.time,
            setpoint_data,
            weight_data=weight_data,
        )
        for i, t in enumerate(m.time):
            pred_expr = (
                10.0*(m.var[t, "A"] - A_setpoint[i])**2
                + 0.1*(m.var[t, "B"] - B_setpoint[i])**2
            )
            pred_value = pyo.value(pred_expr)
            self.assertEqual(pred_value, pyo.value(m.tracking_cost[t]))
            self.assertTrue(compare_expressions(
                pred_expr, m.tracking_cost[t].expr
            ))

    def test_varying_setpoint_exceptions(self):
        m = self._make_model(n_time_points=5)
        variables = [
            pyo.Reference(m.var[:, "A"]),
            pyo.Reference(m.var[:, "B"]),
        ]
        A_setpoint = [1.0 - 0.1*i for i in range(len(m.time))]
        B_setpoint = [5.0 + 0.1*i for i in range(len(m.time))]
        setpoint_data = TimeSeriesData(
            {m.var[:, "A"]: A_setpoint, m.var[:, "B"]: B_setpoint},
            [i + 10 for i in m.time],
        )
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.var[:, "B"]): 0.1,
        }
        msg = "Mismatch in time points"
        with self.assertRaisesRegex(ValueError, msg):
            # Time-varying setpoint specifies different time points
            # from our time set.
            tr_cost = get_tracking_cost_from_time_varying_setpoint(
                variables,
                m.time,
                setpoint_data,
                weight_data=weight_data,
            )

        setpoint_data = TimeSeriesData({m.var[:, "A"]: A_setpoint}, m.time)
        msg = "Setpoint data dictionary does not contain"
        with self.assertRaisesRegex(KeyError, msg):
            tr_cost = get_tracking_cost_from_time_varying_setpoint(
                variables,
                m.time,
                setpoint_data,
                weight_data=weight_data,
            )

        setpoint_data = TimeSeriesData(
            {m.var[:, "A"]: A_setpoint, m.var[:, "B"]: B_setpoint},
            m.time,
        )
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
        }
        msg = "Tracking weight dictionary does not contain"
        with self.assertRaisesRegex(KeyError, msg):
            tr_cost = get_tracking_cost_from_time_varying_setpoint(
                variables,
                m.time,
                setpoint_data,
                weight_data=weight_data,
            )


if __name__ == "__main__":
    unittest.main()

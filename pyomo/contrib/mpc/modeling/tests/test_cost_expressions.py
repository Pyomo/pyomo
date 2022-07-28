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
)


class TestTrackingCost(unittest.TestCase):

    def test_tracking_cost_no_weights(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[1, 2, 3])
        m.v1 = pyo.Var(m.time, initialize={i: 1*i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2*i for i in m.time})

        setpoint_data = {
            str(pyo.ComponentUID(m.v1)): 3.0,
            str(pyo.ComponentUID(m.v2)): 4.0,
        }

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

        setpoint_data = {
            str(pyo.ComponentUID(m.v1)): 3.0,
            str(pyo.ComponentUID(m.v2)): 4.0,
        }
        weight_data = {
            str(pyo.ComponentUID(m.v1)): 0.1,
            str(pyo.ComponentUID(m.v2)): 0.5,
        }

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

        setpoint_data = {
            str(pyo.ComponentUID(m.v1)): 3.0,
        }
        weight_data = {
            str(pyo.ComponentUID(m.v1)): 0.1,
        }

        with self.assertRaisesRegex(KeyError, "Setpoint data"):
            m.tracking_expr = get_tracking_cost_from_constant_setpoint(
                [m.v1, m.v2],
                m.time,
                setpoint_data,
            )

        setpoint_data = {
            str(pyo.ComponentUID(m.v1)): 3.0,
            str(pyo.ComponentUID(m.v2)): 4.0,
        }

        with self.assertRaisesRegex(KeyError, "Tracking weight"):
            m.tracking_expr = get_tracking_cost_from_constant_setpoint(
                [m.v1, m.v2],
                m.time,
                setpoint_data,
                weight_data=weight_data,
            )

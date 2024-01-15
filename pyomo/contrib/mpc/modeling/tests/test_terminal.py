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
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.terminal import (
    get_penalty_at_time,
    get_terminal_penalty,
)
from pyomo.contrib.mpc.data.scalar_data import ScalarData


class TestTerminalPenalty(unittest.TestCase):
    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=list(range(n_time_points)))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time, m.comp, initialize={(i, j): 1.1 * i for i, j in m.time * m.comp}
        )
        m.input = pyo.Var(m.time, initialize={i: 3.3 * i for i in m.time})
        return m

    def test_get_penalty(self):
        m = self._make_model()

        variables = [pyo.Reference(m.var[:, "A"]), m.input]
        target = ScalarData({m.var[:, "A"]: 4.4, m.input[:]: 5.5})
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.input[:]): 0.2,
        }
        tp = 1
        m.var_set, m.penalty = get_penalty_at_time(
            variables, tp, target, weight_data=weight_data
        )
        for i in m.var_set:
            pred_expr = (
                10.0 * (m.var[tp, "A"] - 4.4) ** 2
                if i == 0
                else 0.2 * (m.input[tp] - 5.5) ** 2
            )
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i].expr))
            self.assertTrue(compare_expressions(pred_expr, m.penalty[i].expr))

    def test_get_terminal_penalty(self):
        m = self._make_model()

        variables = [pyo.Reference(m.var[:, "A"]), m.input]
        target = ScalarData({m.var[:, "A"]: 4.4, m.input[:]: 5.5})
        weight_data = {
            pyo.ComponentUID(m.var[:, "A"]): 10.0,
            pyo.ComponentUID(m.input[:]): 0.2,
        }
        m.var_set, m.penalty = get_terminal_penalty(
            variables, m.time, target, weight_data=weight_data
        )

        for i in m.var_set:
            tf = m.time.last()
            pred_expr = (
                10.0 * (m.var[tf, "A"] - 4.4) ** 2
                if i == 0
                else 0.2 * (m.input[tf] - 5.5) ** 2
            )
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i].expr))
            self.assertTrue(compare_expressions(pred_expr, m.penalty[i].expr))


if __name__ == "__main__":
    unittest.main()

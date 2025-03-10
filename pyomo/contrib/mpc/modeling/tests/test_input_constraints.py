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

import pyomo.environ as pyo
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.constraints import get_piecewise_constant_constraints


class TestPiecewiseConstantConstraints(unittest.TestCase):
    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=list(range(n_time_points)))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time, m.comp, initialize={(i, j): 1.1 * i for i, j in m.time * m.comp}
        )
        m.input = pyo.Var(m.time, initialize={i: 3.3 * i for i in m.time})
        return m

    def test_pwc_constraint_backward(self):
        # Tests the form of piecewise constant constraints that
        # should be used for a backward discretization, i.e.
        # that each input at non-sampling points is set equal to
        # the next value in the time set, rather than the previous.
        # This is the default.
        n_time_points = 5
        sample_points = [0, 2, 4]
        sample_points_set = set(sample_points)
        m = self._make_model(n_time_points=n_time_points)
        inputs = [pyo.Reference(m.var[:, "B"]), m.input]
        m.input_set, m.pwc_con = get_piecewise_constant_constraints(
            inputs, m.time, sample_points
        )
        pred_expr = {
            # Here we rely on knowledge that delta t == 1
            (i, t): inputs[i][t] - inputs[i][t + 1] == 0
            for t in m.time
            if t not in sample_points_set
            for i in range(len(inputs))
        }
        self.assertEqual(list(m.input_set), list(range(len(inputs))))
        for i in range(len(inputs)):
            for t in m.time:
                if t in sample_points_set:
                    self.assertNotIn((i, t), m.pwc_con)
                else:
                    self.assertIn((i, t), m.pwc_con)
                    self.assertEqual(
                        pyo.value(pred_expr[i, t]), pyo.value(m.pwc_con[i, t].expr)
                    )
                    self.assertTrue(
                        compare_expressions(pred_expr[i, t], m.pwc_con[i, t].expr)
                    )

    def test_pwc_constraint_forward(self):
        # The form of piecewise constant constraints that
        # should be used for a forward discretization.
        n_time_points = 5
        sample_points = [0, 2, 4]
        sample_points_set = set(sample_points)
        m = self._make_model(n_time_points=n_time_points)
        inputs = [pyo.Reference(m.var[:, "B"]), m.input]
        m.input_set, m.pwc_con = get_piecewise_constant_constraints(
            inputs, m.time, sample_points, use_next=False
        )
        pred_expr = {
            # Here we rely on knowledge that delta t == 1
            (i, t): inputs[i][t - 1] - inputs[i][t] == 0
            for t in m.time
            if t not in sample_points_set
            for i in range(len(inputs))
        }
        self.assertEqual(list(m.input_set), list(range(len(inputs))))
        for i in range(len(inputs)):
            for t in m.time:
                if t in sample_points_set:
                    self.assertNotIn((i, t), m.pwc_con)
                else:
                    self.assertIn((i, t), m.pwc_con)
                    self.assertEqual(
                        pyo.value(pred_expr[i, t]), pyo.value(m.pwc_con[i, t].expr)
                    )
                    self.assertTrue(
                        compare_expressions(pred_expr[i, t], m.pwc_con[i, t].expr)
                    )


if __name__ == "__main__":
    unittest.main()

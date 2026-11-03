# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.environ as pyo
from pyomo.common import unittest
from pyomo.util.vars_from_expressions import get_vars, get_vars_from_components


class TestVarsFromExpressions(unittest.TestCase):
    def test_get_vars(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(list(range(5)))
        m.c1 = pyo.Constraint(expr=m.x[0] + m.x[1] == 0)
        m.c2 = pyo.Constraint(expr=m.x[1] + m.x[2] == 0)
        m.obj = pyo.Objective(expr=m.x[3] + m.x[4])

        self.assertEqual(list(get_vars(m)), [m.x[0], m.x[1], m.x[2], m.x[3], m.x[4]])
        # verify the default values for active and include_fixed
        m.x[0].fix(0)
        m.c2.deactivate()
        self.assertEqual(list(get_vars(m)), [m.x[1], m.x[3], m.x[4]])

    def test_get_vars_from_components(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(list(range(5)))
        m.c1 = pyo.Constraint(expr=m.x[0] + m.x[1] == 0)
        m.c2 = pyo.Constraint(expr=m.x[1] + m.x[2] == 0)
        m.obj = pyo.Objective(expr=m.x[3] + m.x[4])

        self.assertEqual(
            list(get_vars_from_components(m, pyo.Constraint)), [m.x[0], m.x[1], m.x[2]]
        )
        self.assertEqual(
            list(get_vars_from_components(m, pyo.Objective)), [m.x[3], m.x[4]]
        )
        self.assertEqual(
            list(get_vars_from_components(m, (pyo.Constraint, pyo.Objective))),
            [m.x[0], m.x[1], m.x[2], m.x[3], m.x[4]],
        )

        # verify the default values for active and include_fixed
        m.x[0].fix(0)
        m.c2.deactivate()
        self.assertEqual(
            list(get_vars_from_components(m, pyo.Constraint)), [m.x[0], m.x[1], m.x[2]]
        )
        self.assertEqual(
            list(get_vars_from_components(m, pyo.Objective)), [m.x[3], m.x[4]]
        )
        self.assertEqual(
            list(get_vars_from_components(m, (pyo.Constraint, pyo.Objective))),
            [m.x[0], m.x[1], m.x[2], m.x[3], m.x[4]],
        )

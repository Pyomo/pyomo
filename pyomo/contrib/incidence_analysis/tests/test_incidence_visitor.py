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

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.visitor import (
    get_incident_variables,
    _get_ampl_expr,
)


class TestUninitialized(unittest.TestCase):
    def test_assumed_behavior(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2])
        self.assertIs(m.x[1].value, None)

    def test_product_one_fixed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2])
        m.x[1].fix()

        variables = get_incident_variables(m.x[1] * m.x[2])
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

        variables = get_incident_variables(m.x[2] * m.x[1])
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[2])

    def test_uninit_named_expr_times_linear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.expr = pyo.Expression(expr=m.x[1] + 2 * m.x[2])
        m.x[1].fix()
        m.x[2].fix()
        variables = get_incident_variables(m.x[3] * m.expr)
        self.assertEqual(len(variables), 1)
        self.assertIs(variables[0], m.x[3])

    def test_nonlinear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] * m.x[2] * m.x[3]
        m.x[2].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(var_set), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)

        expr = m.x[3] * pyo.exp(m.x[1] ** m.x[2])
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(var_set), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)

        expr = m.x[1] * m.x[2] * m.x[3] - m.x[3] * pyo.exp(m.x[1] ** m.x[2])
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(var_set), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)


class TestInitialized(unittest.TestCase):
    def test_nonlinear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1)
        expr = m.x[1] * m.x[2] * m.x[3]
        m.x[2].fix()
        variables = get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(len(var_set), 2)
        self.assertIn(m.x[1], var_set)
        self.assertIn(m.x[3], var_set)


if __name__ == "__main__":
    unittest.main()

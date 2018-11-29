#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.environ import ConcreteModel, Var, Constraint, value
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.core.base.symbolic import _sympy_available

class Test_calc_var(unittest.TestCase):
    def test_initialize_value(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x == 5)

        m.x.set_value(None)
        calculate_variable_from_constraint(m.x, m.c)
        self.assertEqual(value(m.x), 5)

        m.x.set_value(None)
        m.x.setlb(3)
        calculate_variable_from_constraint(m.x, m.c)
        self.assertEqual(value(m.x), 5)

        m.x.set_value(None)
        m.x.setlb(-10)
        calculate_variable_from_constraint(m.x, m.c)
        self.assertEqual(value(m.x), 5)

        m.x.set_value(None)
        m.x.setub(10)
        calculate_variable_from_constraint(m.x, m.c)
        self.assertEqual(value(m.x), 5)

        m.x.set_value(None)
        m.x.setlb(3)
        calculate_variable_from_constraint(m.x, m.c)
        self.assertEqual(value(m.x), 5)

        m.x.set_value(None)
        m.x.setlb(None)
        calculate_variable_from_constraint(m.x, m.c)
        self.assertEqual(value(m.x), 5)

        m.x.set_value(None)
        m.x.setub(-10)
        calculate_variable_from_constraint(m.x, m.c)
        self.assertEqual(value(m.x), 5)

    def test_linear(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=5*m.x == 10)

        calculate_variable_from_constraint(m.x, m.c)
        self.assertEqual(value(m.x), 2)

    @unittest.skipIf(not _sympy_available, "this test requires sympy")
    def test_nonlinear(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x**2 == 16)

        calculate_variable_from_constraint(m.x, m.c)
        self.assertAlmostEqual(value(m.x), 4)

        m.d = Constraint(expr=m.x**2 == -1)
        with self.assertRaisesRegexp(
                RuntimeError, 'Iteration limit \(10\) reached'):
            calculate_variable_from_constraint(m.x, m.d, iterlim=10)

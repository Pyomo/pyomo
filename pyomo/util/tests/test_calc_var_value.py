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

from pyomo.environ import ConcreteModel, Var, Constraint, value, exp
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.core.base.symbolic import _sympy_available

class Test_calc_var(unittest.TestCase):
    def test_initialize_value(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var(initialize=0)
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

        m.lt = Constraint(expr=m.x <= m.y)
        with self.assertRaisesRegexp(
                ValueError, "Constraint must be an equality constraint"):
            calculate_variable_from_constraint(m.x, m.lt)


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
        m.y = Var(initialize=0)

        m.c = Constraint(expr=m.x**2 == 16)
        m.x.set_value(1.0) # set an initial value
        calculate_variable_from_constraint(m.x, m.c, linesearch=False)
        self.assertAlmostEqual(value(m.x), 4)

        # test that infeasible constraint throws error
        m.d = Constraint(expr=m.x**2 == -1)
        m.x.set_value(1.25) # set the initial value
        with self.assertRaisesRegexp(
               RuntimeError, 'Iteration limit \(10\) reached'):
           calculate_variable_from_constraint(
               m.x, m.d, iterlim=10, linesearch=False)

        # same problem should throw a linesearch error if linesearch is on
        m.x.set_value(1.25) # set the initial value
        with self.assertRaisesRegexp(
                RuntimeError, "Linesearch iteration limit reached"):
           calculate_variable_from_constraint(
               m.x, m.d, iterlim=10, linesearch=True)

        # same problem should raise an error if initialized at 0
        m.x = 0
        with self.assertRaisesRegexp(
                RuntimeError, "Initial value for variable results in a "
                "derivative value that is very close to zero."):
            calculate_variable_from_constraint(m.x, m.c)

        # same problem should raise a value error if we are asked to
        # solve for a variable that is not present
        with self.assertRaisesRegexp(
                ValueError, "Variable derivative == 0"):
            calculate_variable_from_constraint(m.y, m.c)


        # should succeed with or without a linesearch
        m.e = Constraint(expr=(m.x - 2.0)**2 - 1 == 0)
        m.x.set_value(3.1)
        calculate_variable_from_constraint(m.x, m.e, linesearch=False)

        m.x.set_value(3.1)
        calculate_variable_from_constraint(m.x, m.e, linesearch=True)


        # we expect this to succeed with the linesearch
        m.e = Constraint(expr=1.0/(1.0+exp(-m.x))-0.5 == 0)
        m.x.set_value(3.0)
        calculate_variable_from_constraint(m.x, m.e, linesearch=True)

        # we expect this to fail without a linesearch
        m.x.set_value(3.0)
        with self.assertRaisesRegexp(
                RuntimeError, "Newton's method encountered a derivative "
                "that was too close to zero"):
            calculate_variable_from_constraint(m.x, m.e, linesearch=False)


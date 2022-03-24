#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
    ConcreteModel, Var, Constraint, Param, value, exp, NonNegativeReals,
    Binary,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available

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
        with self.assertRaisesRegex(
                ValueError, "Constraint must be an equality constraint"):
            calculate_variable_from_constraint(m.x, m.lt)

    def test_linear(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=5*m.x == 10)

        calculate_variable_from_constraint(m.x, m.c)
        self.assertEqual(value(m.x), 2)

    def test_constraint_as_tuple(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(initialize=15, mutable=True)

        calculate_variable_from_constraint(m.x, 5*m.x == 5)
        self.assertEqual(value(m.x), 1)
        calculate_variable_from_constraint(m.x, (5*m.x, 10))
        self.assertEqual(value(m.x), 2)
        calculate_variable_from_constraint(m.x, (15, 5*m.x, m.p))
        self.assertEqual(value(m.x), 3)
        with self.assertRaisesRegex(
                ValueError, "Constraint 'tuple' is a Ranged Inequality "
                "with a variable upper bound."):
            calculate_variable_from_constraint(m.x, (15, 5*m.x, m.x))


    @unittest.skipIf(not differentiate_available, "this test requires sympy")
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
        with self.assertRaisesRegex(
               RuntimeError, r'Iteration limit \(10\) reached'):
           calculate_variable_from_constraint(
               m.x, m.d, iterlim=10, linesearch=False)

        # same problem should throw a linesearch error if linesearch is on
        m.x.set_value(1.25) # set the initial value
        with self.assertRaisesRegex(
                RuntimeError, "Linesearch iteration limit reached"):
           calculate_variable_from_constraint(
               m.x, m.d, iterlim=10, linesearch=True)

        # same problem should raise an error if initialized at 0
        m.x = 0
        with self.assertRaisesRegex(
                RuntimeError, "Initial value for variable results in a "
                "derivative value that is very close to zero."):
            calculate_variable_from_constraint(m.x, m.c)

        # same problem should raise a value error if we are asked to
        # solve for a variable that is not present
        with self.assertRaisesRegex(
                ValueError, "Variable derivative == 0"):
            calculate_variable_from_constraint(m.y, m.c)


        # should succeed with or without a linesearch
        m.e = Constraint(expr=(m.x - 2.0)**2 - 1 == 0)
        m.x.set_value(3.1)
        calculate_variable_from_constraint(m.x, m.e, linesearch=False)
        self.assertAlmostEqual(value(m.x), 3)

        m.x.set_value(3.1)
        calculate_variable_from_constraint(m.x, m.e, linesearch=True)
        self.assertAlmostEqual(value(m.x), 3)


        # we expect this to succeed with the linesearch
        m.f = Constraint(expr=1.0/(1.0+exp(-m.x))-0.5 == 0)
        m.x.set_value(3.0)
        calculate_variable_from_constraint(m.x, m.f, linesearch=True)
        self.assertAlmostEqual(value(m.x), 0)

        # we expect this to fail without a linesearch
        m.x.set_value(3.0)
        with self.assertRaisesRegex(
                RuntimeError, "Newton's method encountered a derivative "
                "that was too close to zero"):
            calculate_variable_from_constraint(m.x, m.f, linesearch=False)

        # Calculate the bubble point of Benzene.  THe first step
        # computed by calculate_variable_from_constraint will make the
        # second term become complex, and the evaluation will fail.
        # This tests that the algorithm cleanly continues
        m = ConcreteModel()
        m.x = Var()
        m.pc = 48.9e5
        m.tc = 562.2
        m.psc = {'A': -6.98273,
                 'B': 1.33213,
                 'C': -2.62863,
                 'D': -3.33399,
        }
        m.p = 101325
        @m.Constraint()
        def f(m):
            return m.pc * \
                exp((m.psc['A'] * (1 - m.x / m.tc) +
                     m.psc['B'] * (1 - m.x / m.tc)**1.5 +
                     m.psc['C'] * (1 - m.x / m.tc)**3 +
                     m.psc['D'] * (1 - m.x / m.tc)**6
                 ) / (1 - (1 - m.x / m.tc))) - m.p == 0
        m.x.set_value(298.15)
        calculate_variable_from_constraint(m.x, m.f, linesearch=False)
        self.assertAlmostEqual(value(m.x), 353.31855602)
        m.x.set_value(298.15)
        calculate_variable_from_constraint(m.x, m.f, linesearch=True)
        self.assertAlmostEqual(value(m.x), 353.31855602)

        # Starting with an invalid guess (above TC) should raise an
        # exception
        m.x.set_value(600)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo', logging.WARNING):
            with self.assertRaises(TypeError):
                calculate_variable_from_constraint(m.x, m.f, linesearch=False)
        self.assertIn('Encountered an error evaluating the expression '
                      'at the initial guess', output.getvalue())

        # This example triggers an expression evaluation error if the
        # linesearch is turned off because the first step in Newton's
        # method will cause the LHS to become complex
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=(1/m.x**3)**0.5 == 100)
        m.x = .1
        calculate_variable_from_constraint(m.x, m.c, linesearch=True)
        self.assertAlmostEqual(value(m.x), 0.046415888)
        m.x = .1
        output = StringIO()
        with LoggingIntercept(output, 'pyomo', logging.WARNING):
            with self.assertRaises(ValueError):
                # Note that the ValueError is different between Python 2
                # and Python 3: in Python 2 it is a specific error
                # "negative number cannot be raised to a fractional
                # power", and We mock up that error in Python 3 by
                # raising a generic ValueError in
                # calculate_variable_from_constraint
                calculate_variable_from_constraint(m.x, m.c, linesearch=False)
        self.assertIn("Newton's method encountered an error evaluating "
                      "the expression.", output.getvalue())

        # This is a completely contrived example where the linesearch
        # hits the iteration limit before Newton's method ever finds a
        # feasible step
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x**0.5 == -1e-8)
        m.x = 1e-8#197.932807183
        with self.assertRaisesRegex(
                RuntimeError, "Linesearch iteration limit reached; "
                "remaining residual = {function evaluation error}"):
            calculate_variable_from_constraint(m.x, m.c, linesearch=True,
                                               alpha_min=.5)

    def test_bound_violation(self):
        # Test Issue #2176: solving a constraint where the intermediate
        # value can step outside the bounds
        m = ConcreteModel()
        m.v1 = Var(initialize=1, domain=NonNegativeReals)
        m.c1 = Constraint(expr=m.v1 == 0)

        # Calculate value of v1 using constraint c1
        calculate_variable_from_constraint(m.v1, m.c1)
        self.assertEqual(value(m.v1), 0)

        # Calculate value of v1 using a scaled constraint c2
        m.c2 = Constraint(expr=m.v1*10 == 0)
        m.v1.set_value(1)
        calculate_variable_from_constraint(m.v1, m.c2)
        self.assertEqual(value(m.v1), 0)

        # Test linear solution falling outside bounds
        m.c3 = Constraint(expr=m.v1*10 == -1)
        m.v1.set_value(1)
        calculate_variable_from_constraint(m.v1, m.c3)
        self.assertEqual(value(m.v1), -0.1)

    @unittest.skipUnless(differentiate_available, "this test requires sympy")
    def test_nonlinear_bound_violation(self):
        m = ConcreteModel()
        m.v1 = Var(initialize=1, domain=NonNegativeReals)
        m.c1 = Constraint(expr=m.v1 == 0)

        # Test nonlinear solution falling outside bounds
        m.c4 = Constraint(expr=m.v1**3 == -8)
        m.v1.set_value(1)
        calculate_variable_from_constraint(m.v1, m.c4)
        self.assertEqual(value(m.v1), -2)

    def test_warn_final_value_linear(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0,1))
        m.c1 = Constraint(expr=m.x == 10)
        m.c2 = Constraint(expr=5*m.x == 10)

        with LoggingIntercept() as LOG:
            calculate_variable_from_constraint(m.x, m.c1)
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'x' to a numeric value `10` outside the "
            "bounds (0, 1).")
        self.assertEqual(value(m.x), 10)

        with LoggingIntercept() as LOG:
            calculate_variable_from_constraint(m.x, m.c2)
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'x' to a numeric value `2.0` outside the "
            "bounds (0, 1).")
        self.assertEqual(value(m.x), 2)

    @unittest.skipUnless(differentiate_available, "this test requires sympy")
    def test_warn_final_value_nonlinear(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0,1))
        m.c3 = Constraint(expr=(m.x - 3.5)**2 == 0)

        with LoggingIntercept() as LOG:
            calculate_variable_from_constraint(m.x, m.c3)
        self.assertRegex(
            LOG.getvalue().strip(),
            r"Setting Var 'x' to a numeric value `[0-9\.]+` outside the "
            r"bounds \(0, 1\).")
        self.assertAlmostEqual(value(m.x), 3.5, 3)

        m.x.domain = Binary
        with LoggingIntercept() as LOG:
            calculate_variable_from_constraint(m.x, m.c3)
        self.assertRegex(
            LOG.getvalue().strip(),
            r"Setting Var 'x' to a value `[0-9\.]+` \(float\) not in "
            "domain Binary.")
        self.assertAlmostEqual(value(m.x), 3.5, 3)

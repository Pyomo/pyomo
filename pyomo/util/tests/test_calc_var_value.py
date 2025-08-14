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

import logging
from io import StringIO

import pyomo.common.unittest as unittest

from pyomo.common.errors import IterationLimitError
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Param,
    ExternalFunction,
    value,
    exp,
    NonNegativeReals,
    Binary,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.sympy_tools import sympy_available


all_diff_modes = [
    differentiate.Modes.sympy,
    differentiate.Modes.reverse_symbolic,
    differentiate.Modes.reverse_numeric,
]


def sum_sq(args, fixed, fgh):
    f = sum(arg**2 for arg in args)
    g = [2 * arg for arg in args]
    h = None
    return f, g, h


class Test_calc_var(unittest.TestCase):
    def test_initialize_value(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var(initialize=0)
        m.c = Constraint(expr=m.x == 5)

        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
            self.assertEqual(value(m.x), 5)

        m.x.setlb(3)
        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
            self.assertEqual(value(m.x), 5)

        m.x.setlb(-10)
        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
            self.assertEqual(value(m.x), 5)

        m.x.setub(10)
        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
            self.assertEqual(value(m.x), 5)

        m.x.setlb(3)
        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
            self.assertEqual(value(m.x), 5)

        m.x.setlb(None)
        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
            self.assertEqual(value(m.x), 5)

        m.x.setub(-10)
        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
            self.assertEqual(value(m.x), 5)

        m.lt = Constraint(expr=m.x <= m.y)
        with self.assertRaisesRegex(
            ValueError, "Constraint 'lt' must be an equality constraint"
        ):
            calculate_variable_from_constraint(m.x, m.lt)

        m.indexed = Constraint([1, 2], rule=lambda m, i: m.x <= i)
        with self.assertRaisesRegex(
            ValueError,
            r"calculate_variable_from_constraint\(\): constraint must be a scalar "
            r"constraint or a single ConstraintData.  Received IndexedConstraint "
            r'\("indexed"\)',
        ):
            calculate_variable_from_constraint(m.x, m.indexed)

    def test_linear(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=5 * m.x == 10)

        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)
            self.assertEqual(value(m.x), 2)

    def test_constraint_as_tuple(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(initialize=15, mutable=True)

        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, 5 * m.x == 5, diff_mode=mode)
            self.assertEqual(value(m.x), 1)
        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, (5 * m.x, 10), diff_mode=mode)
            self.assertEqual(value(m.x), 2)
        for mode in all_diff_modes:
            m.x.set_value(None)
            calculate_variable_from_constraint(m.x, (15, 5 * m.x, m.p), diff_mode=mode)
            self.assertEqual(value(m.x), 3)
        with self.assertRaisesRegex(
            ValueError,
            "Constraint 'tuple' is a Ranged Inequality with a variable upper bound.",
        ):
            calculate_variable_from_constraint(m.x, (15, 5 * m.x, m.x))

    @unittest.skipIf(not differentiate_available, "this test requires sympy")
    def test_nonlinear(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var(initialize=0)

        m.c = Constraint(expr=m.x**2 == 16)
        for mode in all_diff_modes:
            m.x.set_value(1.0)  # set an initial value
            calculate_variable_from_constraint(
                m.x, m.c, linesearch=False, diff_mode=mode
            )
            self.assertEqual(value(m.x), 4)

        # test that infeasible constraint throws error
        m.d = Constraint(expr=m.x**2 == -1)
        for mode in all_diff_modes:
            m.x.set_value(1.25)  # set the initial value
            with self.assertRaisesRegex(
                IterationLimitError, r'Iteration limit \(10\) reached'
            ):
                calculate_variable_from_constraint(
                    m.x, m.d, iterlim=10, linesearch=False, diff_mode=mode
                )

        # same problem should throw a linesearch error if linesearch is on
        for mode in all_diff_modes:
            m.x.set_value(1.25)  # set the initial value
            with self.assertRaisesRegex(
                IterationLimitError, "Linesearch iteration limit reached"
            ):
                calculate_variable_from_constraint(
                    m.x, m.d, iterlim=10, linesearch=True, diff_mode=mode
                )

        # same problem should raise an error if initialized at 0
        for mode in all_diff_modes:
            m.x = 0
            with self.assertRaisesRegex(
                ValueError,
                "Initial value for variable 'x' results in a "
                "derivative value for constraint 'c' that is very close to zero.",
            ):
                calculate_variable_from_constraint(m.x, m.c, diff_mode=mode)

        # same problem should raise a value error if we are asked to
        # solve for a variable that is not present
        for mode in all_diff_modes:
            if mode == differentiate.Modes.reverse_numeric:
                # numeric differentiation should not be used to check if a
                # derivative is always zero
                with self.assertRaisesRegex(
                    ValueError,
                    "Initial value for variable 'y' results in a "
                    "derivative value for constraint 'c' that is very close to zero.",
                ):
                    calculate_variable_from_constraint(m.y, m.c, diff_mode=mode)
            else:
                with self.assertRaisesRegex(
                    ValueError, "Variable 'y' derivative == 0 in constraint 'c'"
                ):
                    calculate_variable_from_constraint(m.y, m.c, diff_mode=mode)

        # should succeed with or without a linesearch
        m.e = Constraint(expr=(m.x - 2.0) ** 2 - 1 == 0)
        for mode in all_diff_modes:
            m.x.set_value(3.1)
            calculate_variable_from_constraint(
                m.x, m.e, linesearch=False, diff_mode=mode
            )
            self.assertAlmostEqual(value(m.x), 3)

        for mode in all_diff_modes:
            m.x.set_value(3.1)
            calculate_variable_from_constraint(
                m.x, m.e, linesearch=True, diff_mode=mode
            )
            self.assertAlmostEqual(value(m.x), 3)

        # we expect this to succeed with the linesearch
        m.f = Constraint(expr=1.0 / (1.0 + exp(-m.x)) - 0.5 == 0)
        for mode in all_diff_modes:
            m.x.set_value(3.0)
            calculate_variable_from_constraint(
                m.x, m.f, linesearch=True, diff_mode=mode
            )
            self.assertAlmostEqual(value(m.x), 0)

        # we expect this to fail without a linesearch
        for mode in all_diff_modes:
            m.x.set_value(3.0)
            with self.assertRaisesRegex(
                RuntimeError,
                "Newton's method encountered a derivative of constraint 'f' "
                "with respect to variable 'x' that was too close to zero",
            ):
                calculate_variable_from_constraint(
                    m.x, m.f, linesearch=False, diff_mode=mode
                )

        # Calculate the bubble point of Benzene.  THe first step
        # computed by calculate_variable_from_constraint will make the
        # second term become complex, and the evaluation will fail.
        # This tests that the algorithm cleanly continues
        m = ConcreteModel()
        m.x = Var()
        m.pc = 48.9e5
        m.tc = 562.2
        m.psc = {'A': -6.98273, 'B': 1.33213, 'C': -2.62863, 'D': -3.33399}
        m.p = 101325

        @m.Constraint()
        def f(m):
            return (
                m.pc
                * exp(
                    (
                        m.psc['A'] * (1 - m.x / m.tc)
                        + m.psc['B'] * (1 - m.x / m.tc) ** 1.5
                        + m.psc['C'] * (1 - m.x / m.tc) ** 3
                        + m.psc['D'] * (1 - m.x / m.tc) ** 6
                    )
                    / (1 - (1 - m.x / m.tc))
                )
                - m.p
                == 0
            )

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
        self.assertIn(
            'Encountered an error evaluating the expression at the initial guess',
            output.getvalue(),
        )

        # This example triggers an expression evaluation error if the
        # linesearch is turned off because the first step in Newton's
        # method will cause the LHS to become complex
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=(1 / m.x**3) ** 0.5 == 100)
        m.x = 0.1
        calculate_variable_from_constraint(m.x, m.c, linesearch=True)
        self.assertAlmostEqual(value(m.x), 0.046415888)
        m.x = 0.1
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
        self.assertIn(
            "Newton's method encountered an error evaluating the expression for "
            "constraint 'c'.",
            output.getvalue(),
        )

        # This is a completely contrived example where the linesearch
        # hits the iteration limit before Newton's method ever finds a
        # feasible step
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x**0.5 == -1e-8)
        m.x = 1e-8  # 197.932807183
        with self.assertRaisesRegex(
            IterationLimitError,
            "Linesearch iteration limit reached solving for variable 'x' using "
            "constraint 'c'; remaining residual = {function evaluation error}",
        ):
            calculate_variable_from_constraint(m.x, m.c, linesearch=True, alpha_min=0.5)

    def test_bound_violation(self):
        # Test Issue #2176: solving a constraint where the intermediate
        # value can step outside the bounds
        m = ConcreteModel()
        m.v1 = Var(initialize=1, domain=NonNegativeReals)
        m.c1 = Constraint(expr=m.v1 == 0)

        # Calculate value of v1 using constraint c1
        for mode in all_diff_modes:
            m.v1.set_value(None)
            calculate_variable_from_constraint(m.v1, m.c1, diff_mode=mode)
            self.assertEqual(value(m.v1), 0)

        # Calculate value of v1 using a scaled constraint c2
        m.c2 = Constraint(expr=m.v1 * 10 == 0)
        for mode in all_diff_modes:
            m.v1.set_value(1)
            calculate_variable_from_constraint(m.v1, m.c2, diff_mode=mode)
            self.assertEqual(value(m.v1), 0)

        # Test linear solution falling outside bounds
        m.c3 = Constraint(expr=m.v1 * 10 == -1)
        for mode in all_diff_modes:
            m.v1.set_value(1)
            calculate_variable_from_constraint(m.v1, m.c3, diff_mode=mode)
            self.assertEqual(value(m.v1), -0.1)

    @unittest.skipUnless(differentiate_available, "this test requires sympy")
    def test_nonlinear_bound_violation(self):
        m = ConcreteModel()
        m.v1 = Var(initialize=1, domain=NonNegativeReals)
        m.c1 = Constraint(expr=m.v1 == 0)

        # Test nonlinear solution falling outside bounds
        m.c4 = Constraint(expr=m.v1**3 == -8)
        for mode in all_diff_modes:
            m.v1.set_value(1)
            calculate_variable_from_constraint(m.v1, m.c4, diff_mode=mode)
            self.assertEqual(value(m.v1), -2)

    def test_warn_final_value_linear(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.c1 = Constraint(expr=m.x == 10)
        m.c2 = Constraint(expr=5 * m.x == 10)

        with LoggingIntercept() as LOG:
            calculate_variable_from_constraint(m.x, m.c1)
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'x' to a numeric value `10` outside the bounds (0, 1).",
        )
        self.assertEqual(value(m.x), 10)

        with LoggingIntercept() as LOG:
            calculate_variable_from_constraint(m.x, m.c2)
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'x' to a numeric value `2.0` outside the bounds (0, 1).",
        )
        self.assertEqual(value(m.x), 2)

    @unittest.skipUnless(differentiate_available, "this test requires sympy")
    def test_warn_final_value_nonlinear(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.c3 = Constraint(expr=(m.x - 3.5) ** 2 == 0)

        with LoggingIntercept() as LOG:
            calculate_variable_from_constraint(m.x, m.c3)
        self.assertRegex(
            LOG.getvalue().strip(),
            r"Setting Var 'x' to a numeric value `[0-9\.]+` outside the "
            r"bounds \(0, 1\).",
        )
        self.assertAlmostEqual(value(m.x), 3.5, 3)

        m.x.domain = Binary
        with LoggingIntercept() as LOG:
            calculate_variable_from_constraint(m.x, m.c3)
        self.assertRegex(
            LOG.getvalue().strip(),
            r"Setting Var 'x' to a value `[0-9\.]+` \(float\) not in domain Binary.",
        )
        self.assertAlmostEqual(value(m.x), 3.5, 3)

    @unittest.skipUnless(differentiate_available, "this test requires sympy")
    def test_nonlinear_overflow(self):
        # Regression check to make sure calculate_variable_from_constraint
        # can handle extreme non-linear cases where assuming linear behaviour
        # results in OverflowErrors
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.c = Constraint(expr=exp(1e2 * m.x**2) == 100)

        calculate_variable_from_constraint(m.x, m.c)

        self.assertAlmostEqual(value(m.x), 0.214597, 5)

    def test_external_function(self):
        m = ConcreteModel()
        m.x = Var()
        m.sq = ExternalFunction(fgh=sum_sq)
        m.c = Constraint(expr=m.sq(m.x - 3) == 0)

        with LoggingIntercept(level=logging.DEBUG) as LOG:
            calculate_variable_from_constraint(m.x, m.c)
        self.assertAlmostEqual(value(m.x), 3, 3)
        self.assertEqual(
            LOG.getvalue(),
            "Calculating symbolic derivative of expression failed. "
            "Reverting to numeric differentiation\n",
        )

    @unittest.skipUnless(sympy_available, 'test expects that sympy is available')
    def test_external_function_explicit_sympy(self):
        m = ConcreteModel()
        m.x = Var()
        m.sq = ExternalFunction(fgh=sum_sq)
        m.c = Constraint(expr=m.sq(m.x - 3) == 0)

        with self.assertRaisesRegex(
            TypeError,
            r"Expressions containing external functions are not convertible "
            r"to sympy expressions \(found 'f\(x0 - 3",
        ):
            calculate_variable_from_constraint(
                m.x, m.c, diff_mode=differentiate.Modes.sympy
            )

    def test_linea_search_overflow(self):
        # This tests the example from #3540
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.y = Var(initialize=1)
        m.con = Constraint(expr=m.y == 10 ** (-m.x))

        calculate_variable_from_constraint(m.x, m.con)
        self.assertAlmostEqual(value(m.x), 0)

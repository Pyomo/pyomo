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

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.gsl import find_GSL
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.calculus.diff_with_pyomo import (
    reverse_ad,
    reverse_sd,
    DifferentiationException,
)
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.core.expr.sympy_tools import sympy_available

tol = 6


def approx_deriv(expr, wrt, delta=0.001):
    numerator = 0
    wrt.value += 2 * delta
    numerator -= pyo.value(expr)
    wrt.value -= delta
    numerator += 8 * pyo.value(expr)
    wrt.value -= 2 * delta
    numerator -= 8 * pyo.value(expr)
    wrt.value -= delta
    numerator += pyo.value(expr)
    wrt.value += 2 * delta
    return numerator / (12 * delta)


class TestDerivs(unittest.TestCase):
    def test_prod(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=3.0)
        e = m.x * m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.y], pyo.value(symbolic[m.y]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_sum(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=3.0)
        e = 2.0 * m.x + 3.0 * m.y - m.x * m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.y], pyo.value(symbolic[m.y]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_div(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=3.0)
        e = m.x / m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.y], pyo.value(symbolic[m.y]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_pow(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=3.0)
        e = m.x**m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.y], pyo.value(symbolic[m.y]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_sqrt(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=3.0)
        e = pyo.sqrt(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_exp(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        e = pyo.exp(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_log(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        e = pyo.log(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_log10(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        e = pyo.log10(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_sin(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        e = pyo.sin(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_cos(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        e = pyo.cos(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_tan(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        e = pyo.tan(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_asin(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.5)
        e = pyo.asin(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_acos(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.5)
        e = pyo.acos(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_atan(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        e = pyo.atan(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_abs(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        e = 2 * abs(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        m.x.value = -2
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        m.x.value = 0
        with self.assertRaisesRegex(
            DifferentiationException, r'Cannot differentiate abs\(x\) at x=0'
        ):
            reverse_ad(e)

    def test_nested(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2)
        m.y = pyo.Var(initialize=3)
        m.p = pyo.Param(initialize=0.5, mutable=True)
        e = pyo.exp(m.x**m.p + 3.2 * m.y - 12)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.y], pyo.value(symbolic[m.y]), tol + 3)
        self.assertAlmostEqual(derivs[m.p], pyo.value(symbolic[m.p]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)
        self.assertAlmostEqual(derivs[m.p], approx_deriv(e, m.p), tol)

    def test_expressiondata(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=3)
        m.e = pyo.Expression(expr=m.x * 2)

        @m.Expression([1, 2])
        def e2(m, i):
            if i == 1:
                return m.x + 4
            else:
                return m.x**2

        m.o = pyo.Objective(expr=m.e + 1 + m.e2[1] + m.e2[2])
        derivs = reverse_ad(m.o.expr)
        symbolic = reverse_sd(m.o.expr)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol)

    def test_constant_named_expressions(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=3)
        m.e = pyo.Expression(expr=2)

        e = m.x * m.e
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_multiple_named_expressions(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.x.value = 1
        m.y.value = 1
        m.E = pyo.Expression(expr=m.x * m.y)
        e = m.E - m.E
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], 0)
        self.assertAlmostEqual(derivs[m.y], 0)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(pyo.value(symbolic[m.x]), 0)
        self.assertAlmostEqual(pyo.value(symbolic[m.y]), 0)

    def test_external(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find the amplgsl.dll library')

        m = pyo.ConcreteModel()
        m.hypot = pyo.ExternalFunction(library=DLL, function='gsl_hypot')
        m.x = pyo.Var(initialize=0.5)
        m.y = pyo.Var(initialize=1.5)
        e = 2 * m.hypot(m.x, m.x * m.y)
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_linear_expression(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=3.0)
        m.p = pyo.Param(initialize=2.5, mutable=True)
        e = LinearExpression(
            constant=m.p, linear_vars=[m.x, m.y], linear_coefs=[1.8, m.p]
        )
        e = pyo.log(e)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        for v in [m.x, m.y, m.p]:
            self.assertAlmostEqual(derivs[v], pyo.value(symbolic[v]), tol)
            self.assertAlmostEqual(derivs[v], approx_deriv(e, v), tol)

    def test_NPV(self):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(initialize=2.0, mutable=True)
        e = pyo.log(m.p)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.p], pyo.value(symbolic[m.p]), tol)
        self.assertAlmostEqual(derivs[m.p], approx_deriv(e, m.p), tol)

    def test_duplicate_expressions(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.23)
        m.y = pyo.Var(initialize=0.88)
        a = (m.x + 1) ** 2
        b = 3 * (a + m.y)
        e = 2 * a + 2 * b + 2 * b + 2 * a
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], pyo.value(symbolic[m.y]), tol + 3)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_nested_named_expressions(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.23)
        m.y = pyo.Var(initialize=0.88)
        m.a = pyo.Expression(expr=(m.x + 1) ** 2)
        m.b = pyo.Expression(expr=3 * (m.a + m.y))
        e = 2 * m.a + 2 * m.b + 2 * m.b + 2 * m.a
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pyo.value(symbolic[m.x]), tol + 3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], pyo.value(symbolic[m.y]), tol + 3)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_linear_exprs_issue_3096(self):
        m = pyo.ConcreteModel()
        m.y1 = pyo.Var(initialize=10)
        m.y2 = pyo.Var(initialize=100)
        e = (m.y1 - 0.5) * (m.y1 - 0.5) + (m.y2 - 0.5) * (m.y2 - 0.5)
        derivs = reverse_ad(e)
        self.assertEqual(derivs[m.y1], 19)
        self.assertEqual(derivs[m.y2], 199)
        symbolic = reverse_sd(e)
        self.assertExpressionsEqual(symbolic[m.y1], m.y1 - 0.5 + m.y1 - 0.5)
        self.assertExpressionsEqual(symbolic[m.y2], m.y2 - 0.5 + m.y2 - 0.5)


class TestDifferentiate(unittest.TestCase):
    @unittest.skipUnless(sympy_available, "test requires sympy")
    def test_sympy(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.23)
        m.y = pyo.Var(initialize=0.88)
        ddx = differentiate(m.x**2, wrt=m.x, mode='sympy')
        self.assertTrue(compare_expressions(ddx, 2 * m.x))
        self.assertAlmostEqual(ddx(), 0.46)
        ddy = differentiate(m.x**2, wrt=m.y, mode='sympy')
        self.assertEqual(ddy, 0)

        ddx = differentiate(m.x**2, wrt_list=[m.x, m.y], mode='sympy')
        self.assertIsInstance(ddx, list)
        self.assertEqual(len(ddx), 2)
        self.assertTrue(compare_expressions(ddx[0], 2 * m.x))
        self.assertAlmostEqual(ddx[0](), 0.46)
        self.assertEqual(ddx[1], 0)

    def test_reverse_symbolic(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.23)
        m.y = pyo.Var(initialize=0.88)
        ddx = differentiate(m.x**2, wrt=m.x, mode='reverse_symbolic')
        assertExpressionsEqual(self, ddx, 2 * m.x)
        self.assertAlmostEqual(ddx(), 0.46)
        ddy = differentiate(m.x**2, wrt=m.y, mode='reverse_symbolic')
        self.assertEqual(ddy, 0)

        ddx = differentiate(m.x**2, wrt_list=[m.x, m.y], mode='reverse_symbolic')
        self.assertIsInstance(ddx, list)
        self.assertEqual(len(ddx), 2)
        assertExpressionsEqual(self, ddx[0], 2 * m.x)
        self.assertAlmostEqual(ddx[0](), 0.46)
        self.assertEqual(ddx[1], 0)

    def test_reverse_numeric(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.23)
        m.y = pyo.Var(initialize=0.88)
        ddx = differentiate(m.x**2, wrt=m.x, mode='reverse_numeric')
        self.assertIsInstance(ddx, float)
        self.assertAlmostEqual(ddx, 0.46)
        ddy = differentiate(m.x**2, wrt=m.y, mode='reverse_numeric')
        self.assertEqual(ddy, 0)

        ddx = differentiate(m.x**2, wrt_list=[m.x, m.y], mode='reverse_numeric')
        self.assertIsInstance(ddx, list)
        self.assertEqual(len(ddx), 2)
        self.assertIsInstance(ddx[0], float)
        self.assertAlmostEqual(ddx[0], 0.46)
        self.assertEqual(ddx[1], 0)

    def test_bad_mode(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.23)
        with self.assertRaisesRegex(
            ValueError,
            r'Unrecognized differentiation mode: foo\n'
            r"Expected one of \['sympy', 'reverse_symbolic', "
            r"'reverse_numeric'\]",
        ):
            ddx = differentiate(m.x**2, m.x, mode='foo')

    def test_bad_wrt(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.23)
        with self.assertRaisesRegex(
            ValueError, r'Cannot specify both wrt and wrt_list'
        ):
            ddx = differentiate(m.x**2, wrt=m.x, wrt_list=[m.x])

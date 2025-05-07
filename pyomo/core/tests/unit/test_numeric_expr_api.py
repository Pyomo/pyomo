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
#
# Unit Tests for expression generation
#
import logging
import math

import pyomo.common.unittest as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.compare import assertExpressionsStructurallyEqual
from pyomo.core.expr import (
    value,
    sin,
    Expr_if,
    SumExpression,
    LinearExpression,
    NPV_SumExpression,
    AbsExpression,
    DivisionExpression,
    NPV_DivisionExpression,
    ExternalFunctionExpression,
    NPV_ExternalFunctionExpression,
    MonomialTermExpression,
    PowExpression,
    NPV_PowExpression,
    ProductExpression,
    NPV_ProductExpression,
    NegationExpression,
    NPV_NegationExpression,
    UnaryFunctionExpression,
    NPV_UnaryFunctionExpression,
)
from pyomo.core.expr.numeric_expr import (
    mutable_expression,
    nonlinear_expression,
    linear_expression,
    _MutableSumExpression,
    _MutableLinearExpression,
    _MutableNPVSumExpression,
    Expr_ifExpression,
    NPV_Expr_ifExpression,
    MaxExpression,
    NPV_MaxExpression,
    MinExpression,
    NPV_MinExpression,
)
from pyomo.environ import ConcreteModel, Param, Var, ExternalFunction


class MockExternalFunction(object):
    def evaluate(self, args):
        (x,) = args
        return (math.log(x) / math.log(2)) ** 2

    def getname(self):
        return 'mock_fcn'


class TestExpressionAPI(unittest.TestCase):
    def test_deprecated_functions(self):
        m = ConcreteModel()
        m.x = Var()
        e = m.x**10
        self.assertIs(type(e), PowExpression)
        with LoggingIntercept() as LOG:
            f = e.create_potentially_variable_object()
        self.assertIs(e, f)
        self.assertIs(type(e), PowExpression)
        self.assertIn(
            'DEPRECATED: The implicit recasting of a "not potentially variable" '
            'expression node to a potentially variable one is no longer supported',
            LOG.getvalue().replace('\n', ' '),
        )
        self.assertNotIn(
            'recasting a non-potentially variable expression to a potentially variable '
            'one violates the immutability promise for Pyomo expression trees.',
            LOG.getvalue().replace('\n', ' '),
        )

        m.p = Param(mutable=True)
        e = m.p**10
        self.assertIs(type(e), NPV_PowExpression)
        with LoggingIntercept() as LOG:
            f = e.create_potentially_variable_object()
        self.assertIs(e, f)
        self.assertIs(type(e), PowExpression)
        self.assertIn(
            'DEPRECATED: The implicit recasting of a "not potentially variable" '
            'expression node to a potentially variable one is no longer supported',
            LOG.getvalue().replace('\n', ' '),
        )
        self.assertIn(
            'recasting a non-potentially variable expression to a potentially variable '
            'one violates the immutability promise for Pyomo expression trees.',
            LOG.getvalue().replace('\n', ' '),
        )

        e = m.x + m.x
        with LoggingIntercept() as LOG:
            f = e.add(5)
        self.assertIn(
            'DEPRECATED: SumExpression.add() is deprecated.  Please use regular '
            'Python operators',
            LOG.getvalue().replace('\n', ' '),
        )
        self.assertEqual(str(e), 'x + x')
        self.assertEqual(str(f), 'x + x + 5')

    def test_mutable_expression(self):
        m = ConcreteModel()
        m.x = Var(range(3))
        with mutable_expression() as e:
            f = e
            self.assertIs(type(e), _MutableNPVSumExpression)
            e += 1
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableNPVSumExpression)
            e += m.x[0]
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableLinearExpression)
            e += 100 * m.x[1]
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableLinearExpression)
            e += m.x[0] ** 2
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableSumExpression)
        self.assertIs(e, f)
        self.assertIs(type(e), SumExpression)

    def test_linear_expression(self):
        m = ConcreteModel()
        m.x = Var(range(3))
        with linear_expression() as e:
            f = e
            self.assertIs(type(e), _MutableNPVSumExpression)
            e += 1
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableNPVSumExpression)
            e += m.x[0]
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableLinearExpression)
            e += 100 * m.x[1]
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableLinearExpression)
            e += m.x[0] ** 2
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableSumExpression)
        self.assertIs(e, f)
        self.assertIs(type(e), SumExpression)

    def test_nonlinear_expression(self):
        m = ConcreteModel()
        m.x = Var(range(3))
        with nonlinear_expression() as e:
            f = e
            self.assertIs(type(e), _MutableSumExpression)
            e += 1
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableSumExpression)
            e += m.x[0]
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableSumExpression)
            e += 100 * m.x[1]
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableSumExpression)
            e += m.x[0] ** 2
            self.assertIs(e, f)
            self.assertIs(type(e), _MutableSumExpression)
        self.assertIs(e, f)
        self.assertIs(type(e), SumExpression)

    def test_negation(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)

        e = NegationExpression((5,))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), -5)
        self.assertEqual(str(e), "- 5")
        self.assertEqual(e.to_string(verbose=True), "neg(5)")

        e = NegationExpression((-5,))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 5)
        self.assertEqual(str(e), "5")
        self.assertEqual(e.to_string(verbose=True), "neg(-5)")

        e = NegationExpression((m.x,))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), -5)
        self.assertEqual(str(e), "- x")
        self.assertEqual(e.to_string(verbose=True), "neg(x)")

        m.p = Param(initialize=10, mutable=True)
        e = NPV_NegationExpression((m.p,))
        self.assertFalse(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), -10)
        self.assertEqual(str(e), "- p")
        self.assertEqual(e.to_string(verbose=True), "neg(p)")

        e = -(m.x + 2 * m.x)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), -15)
        self.assertEqual(str(e), "- (x + 2*x)")
        self.assertEqual(e.to_string(verbose=True), "neg(sum(x, mon(2, x)))")

        # This can't occur through operator overloading, but could
        # through expression substitution
        e = NegationExpression((NegationExpression((m.x,)),))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 5)
        self.assertEqual(str(e), "x")
        self.assertEqual(e.to_string(verbose=True), "neg(neg(x))")

    def test_pow(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        e = PowExpression((m.x, 2))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 2)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 25)
        self.assertEqual(str(e), "x**2")
        self.assertEqual(e.to_string(verbose=True), "pow(x, 2)")

        e = PowExpression((2, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 32)
        self.assertEqual(str(e), "2**x")
        self.assertEqual(e.to_string(verbose=True), "pow(2, x)")

        m.p = Param(initialize=3, mutable=True)
        e = NPV_PowExpression((m.p, 2))
        self.assertFalse(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 9)
        self.assertEqual(str(e), "p**2")
        self.assertEqual(e.to_string(verbose=True), "pow(p, 2)")

        e = NPV_PowExpression((2, m.p))
        self.assertFalse(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 8)
        self.assertEqual(str(e), "2**p")
        self.assertEqual(e.to_string(verbose=True), "pow(2, p)")

        e = PowExpression((m.x, m.p))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 3)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 125)
        self.assertEqual(str(e), "x**p")
        self.assertEqual(e.to_string(verbose=True), "pow(x, p)")

        m.p = 0
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 1)
        self.assertEqual(str(e), "x**p")
        self.assertEqual(e.to_string(verbose=True), "pow(x, p)")

        m.x.fix(2)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 1)
        self.assertEqual(str(e), "x**p")
        self.assertEqual(e.to_string(verbose=True), "pow(x, p)")

        m.p = 3
        e = PowExpression((m.p, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 9)
        self.assertEqual(str(e), "p**x")
        self.assertEqual(e.to_string(verbose=True), "pow(p, x)")

        m.y = Var()
        m.x.fix(None)
        e = PowExpression((m.y, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        with self.assertRaisesRegex(
            ValueError, 'No value for uninitialized ScalarVar object y'
        ):
            self.assertEqual(value(e), None)
        self.assertEqual(str(e), "y**x")
        self.assertEqual(e.to_string(verbose=True), "pow(y, x)")

    def test_min(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        e = MinExpression((m.x, 2))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 2)
        self.assertEqual(str(e), "min(x, 2)")
        self.assertEqual(e.to_string(verbose=True), "min(x, 2)")

        m.x.fix(1)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 1)
        self.assertEqual(str(e), "min(x, 2)")
        self.assertEqual(e.to_string(verbose=True), "min(x, 2)")

    def test_max(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        e = MaxExpression((m.x, 2))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 5)
        self.assertEqual(str(e), "max(x, 2)")
        self.assertEqual(e.to_string(verbose=True), "max(x, 2)")

        m.x.fix(10)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 10)
        self.assertEqual(str(e), "max(x, 2)")
        self.assertEqual(e.to_string(verbose=True), "max(x, 2)")

    def test_prod(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        e = ProductExpression((m.x, 2))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 10)
        self.assertEqual(str(e), "x*2")
        self.assertEqual(e.to_string(verbose=True), "prod(x, 2)")

        e = ProductExpression((2, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 10)
        self.assertEqual(str(e), "2*x")
        self.assertEqual(e.to_string(verbose=True), "prod(2, x)")

        m.p = Param(initialize=3, mutable=True)
        e = NPV_ProductExpression((m.p, 2))
        self.assertFalse(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 6)
        self.assertEqual(str(e), "p*2")
        self.assertEqual(e.to_string(verbose=True), "prod(p, 2)")

        e = NPV_ProductExpression((2, m.p))
        self.assertFalse(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 6)
        self.assertEqual(str(e), "2*p")
        self.assertEqual(e.to_string(verbose=True), "prod(2, p)")

        e = ProductExpression((m.x, m.p))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 15)
        self.assertEqual(str(e), "x*p")
        self.assertEqual(e.to_string(verbose=True), "prod(x, p)")

        m.p = 0
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 0)
        self.assertEqual(str(e), "x*p")
        self.assertEqual(e.to_string(verbose=True), "prod(x, p)")

        e = ProductExpression((m.p, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 0)
        self.assertEqual(str(e), "p*x")
        self.assertEqual(e.to_string(verbose=True), "prod(p, x)")

        e = ProductExpression((-1, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), -5)
        self.assertEqual(str(e), "- x")
        self.assertEqual(e.to_string(verbose=True), "prod(-1, x)")

        m.y = Var()
        e = ProductExpression((m.y, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 2)
        self.assertEqual(is_fixed(e), False)
        with self.assertRaisesRegex(
            ValueError, 'No value for uninitialized ScalarVar object y'
        ):
            self.assertEqual(value(e), None)
        self.assertEqual(str(e), "y*x")
        self.assertEqual(e.to_string(verbose=True), "prod(y, x)")

        m.x.fix(0)
        e = ProductExpression((m.y, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        with self.assertRaisesRegex(
            ValueError, 'No value for uninitialized ScalarVar object y'
        ):
            self.assertEqual(value(e), None)
        self.assertEqual(str(e), "y*x")
        self.assertEqual(e.to_string(verbose=True), "prod(y, x)")

        m.x.fix(None)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        with self.assertRaisesRegex(
            ValueError, 'No value for uninitialized ScalarVar object y'
        ):
            self.assertEqual(value(e), None)
        self.assertEqual(str(e), "y*x")
        self.assertEqual(e.to_string(verbose=True), "prod(y, x)")

        m.y = 5
        e = ProductExpression((1 / m.y, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        with self.assertRaisesRegex(
            ValueError, 'No value for uninitialized ScalarVar object x'
        ):
            self.assertEqual(value(e), None)
        self.assertEqual(str(e), "1/y*x")
        self.assertEqual(e.to_string(verbose=True), "prod(div(1, y), x)")

    def test_monomial(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        e = MonomialTermExpression((2, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 10)
        self.assertEqual(str(e), "2*x")
        self.assertEqual(e.to_string(verbose=True), "mon(2, x)")

        e = MonomialTermExpression((-2, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), -10)
        self.assertEqual(str(e), "-2*x")
        self.assertEqual(e.to_string(verbose=True), "mon(-2, x)")

        m.x.fix(2)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), -4)
        self.assertEqual(str(e), "-2*x")
        self.assertEqual(e.to_string(verbose=True), "mon(-2, x)")

    def test_division(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)

        e = DivisionExpression((m.x, 2))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 2.5)
        self.assertEqual(str(e), "x/2")
        self.assertEqual(e.to_string(verbose=True), "div(x, 2)")

        e = DivisionExpression((2, m.x))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 0.4)
        self.assertEqual(str(e), "2/x")
        self.assertEqual(e.to_string(verbose=True), "div(2, x)")

        m.x.fix(2)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 1)
        self.assertEqual(str(e), "2/x")
        self.assertEqual(e.to_string(verbose=True), "div(2, x)")

    def test_sum(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)

        e = SumExpression(())
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 0)
        self.assertEqual(str(e), "0")
        self.assertEqual(e.to_string(verbose=True), "sum(0)")

        e = SumExpression((m.x, 2))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 7)
        self.assertEqual(str(e), "x + 2")
        self.assertEqual(e.to_string(verbose=True), "sum(x, 2)")

        e = SumExpression((m.x, -2))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 3)
        self.assertEqual(str(e), "x - 2")
        self.assertEqual(e.to_string(verbose=True), "sum(x, -2)")

        e = SumExpression((-2, m.x, -2))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 1)
        self.assertEqual(str(e), "-2 + x - 2")
        self.assertEqual(e.to_string(verbose=True), "sum(-2, x, -2)")

        e = SumExpression([-2, m.x, AbsExpression((m.x,)), m.x])
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 13)
        self.assertEqual(str(e), "-2 + x + abs(x) + x")
        self.assertEqual(e.to_string(verbose=True), "sum(-2, x, abs(x), x)")

        e = SumExpression([-2, m.x, SumExpression([-2])])
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 1)
        self.assertEqual(str(e), "-2 + x + (-2)")
        self.assertEqual(e.to_string(verbose=True), "sum(-2, x, sum(-2))")

        e = SumExpression([-2, m.x, SumExpression([-2, 3])])
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 4)
        self.assertEqual(str(e), "-2 + x + (-2 + 3)")
        self.assertEqual(e.to_string(verbose=True), "sum(-2, x, sum(-2, 3))")

        e = SumExpression([-2, m.x, SumExpression([2, 3])])
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 8)
        self.assertEqual(str(e), "-2 + x + (2 + 3)")
        self.assertEqual(e.to_string(verbose=True), "sum(-2, x, sum(2, 3))")

        e = SumExpression([-2, m.x, NegationExpression((SumExpression([-2]),))])
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 5)
        self.assertEqual(str(e), "-2 + x - (-2)")
        self.assertEqual(e.to_string(verbose=True), "sum(-2, x, neg(sum(-2)))")

        e = SumExpression((NegationExpression((m.x,)), -2))
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), -7)
        self.assertEqual(str(e), "- x - 2")
        self.assertEqual(e.to_string(verbose=True), "sum(neg(x), -2)")

        m.x.fix(2)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), -4)
        self.assertEqual(str(e), "- x - 2")
        self.assertEqual(e.to_string(verbose=True), "sum(neg(x), -2)")

    def test_linear(self):
        m = ConcreteModel()
        m.x = Var(range(3), initialize=range(3))
        m.y = Var(initialize=5)
        with mutable_expression() as e:
            for i in range(3):
                e += i * m.x[i]
            e += 5
            e += m.y
            e -= 3

        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 1 + 4 + 5 + 2)
        self.assertEqual(str(e), "0*x[0] + x[1] + 2*x[2] + 5 + y - 3")
        self.assertEqual(
            e.to_string(verbose=True), "sum(mon(0, x[0]), x[1], mon(2, x[2]), 5, y, -3)"
        )

        self.assertIs(type(e), LinearExpression)
        self.assertEqual(e.constant, 2)
        cache = e._cache
        self.assertEqual(e.linear_coefs, [0, 1, 2, 1])
        self.assertIs(cache, e._cache)
        self.assertEqual(e.linear_vars, [m.x[0], m.x[1], m.x[2], m.y])
        self.assertIs(cache, e._cache)

        e = LinearExpression()
        self.assertEqual(e.linear_coefs, [])
        self.assertIsNot(cache, e._cache)
        cache = e._cache
        e = LinearExpression()
        self.assertEqual(e.constant, 0)
        self.assertIsNot(cache, e._cache)
        cache = e._cache
        e = LinearExpression()
        self.assertEqual(e.linear_vars, [])
        self.assertIsNot(cache, e._cache)

        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 0)
        self.assertEqual(str(e), "0")
        self.assertEqual(e.to_string(verbose=True), "sum(0)")

        e = LinearExpression(constant=5, linear_vars=[m.y, m.x[1]], linear_coefs=[3, 5])
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 25)
        self.assertEqual(str(e), "5 + 3*y + 5*x[1]")
        self.assertEqual(e.to_string(verbose=True), "sum(5, mon(3, y), mon(5, x[1]))")

        with self.assertRaisesRegex(
            ValueError,
            "Cannot specify both args and any of "
            "{constant, linear_coefs, or linear_vars}",
        ):
            LinearExpression(5, constant=5)

        with self.assertRaisesRegex(
            ValueError,
            r"linear_vars \(\[y\]\) is not compatible with linear_coefs \(\[3, 5\]\)",
        ):
            LinearExpression(constant=5, linear_vars=[m.y], linear_coefs=[3, 5])

    def test_expr_if(self):
        m = ConcreteModel()
        m.x = Var(range(3), initialize=range(3))
        m.y = Var(initialize=5)
        e = Expr_if(IF=m.y >= 5, THEN=m.x[0] + 5, ELSE=m.x[1] ** 2)

        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 5)
        self.assertEqual(
            str(e), "Expr_if( ( 5  <=  y ), then=( x[0] + 5 ), else=( x[1]**2 ) )"
        )
        self.assertEqual(
            e.to_string(verbose=True),
            "Expr_if( ( 5  <=  y ), then=( sum(x[0], 5) ), else=( pow(x[1], 2) ) )",
        )

        m.y.fix()
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 5)

        m.y.fix(4)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 2)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 1)

        e = Expr_if(IF=m.y >= 5, THEN=m.x[0] + 5, ELSE=m.x[1] + 10)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 11)

        m.y.fix(5)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 5)

    def test_unary(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        e = sin(2 * m.x)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), math.sin(10))
        self.assertEqual(str(e), "sin(2*x)")
        self.assertEqual(e.to_string(verbose=True), "sin(mon(2, x))")

        m.x.fix(1)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), math.sin(2))
        self.assertEqual(str(e), "sin(2*x)")
        self.assertEqual(e.to_string(verbose=True), "sin(mon(2, x))")

    def test_external(self):
        m = ConcreteModel()
        m.x = Var(initialize=16)
        fcn = MockExternalFunction()
        e = ExternalFunctionExpression((2 * m.x,), fcn)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), None)
        self.assertEqual(is_fixed(e), False)
        self.assertEqual(value(e), 25)
        self.assertEqual(str(e), "mock_fcn(2*x)")
        self.assertEqual(e.to_string(verbose=True), "mock_fcn(mon(2, x))")

        m.x.fix(1)
        self.assertTrue(e.is_potentially_variable())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(is_fixed(e), True)
        self.assertEqual(value(e), 1)
        self.assertEqual(str(e), "mock_fcn(2*x)")
        self.assertEqual(e.to_string(verbose=True), "mock_fcn(mon(2, x))")


class TestExpressionDuplicateAPI(unittest.TestCase):
    def test_negation(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = NegationExpression((m.x,))

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)

        f = e.create_node_with_local_data((2,))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (2,))

        e = NPV_NegationExpression((m.p,))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))

        f = e.create_node_with_local_data((m.x,))
        self.assertIsNot(f, e)
        self.assertIs(type(f), NegationExpression)

    def test_pow(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = PowExpression((m.x, 2))

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)

        f = e.create_node_with_local_data((2, 3))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (2, 3))

        e = NPV_PowExpression((m.p, 2))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))

        f = e.create_node_with_local_data((m.p, m.x))
        self.assertIsNot(f, e)
        self.assertIs(type(f), PowExpression)

    def test_min(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = MinExpression((m.x, 2))

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)

        f = e.create_node_with_local_data((2, 3, 4))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (2, 3, 4))

        e = NPV_MinExpression((m.p, 2))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))

        f = e.create_node_with_local_data((m.p, m.x))
        self.assertIsNot(f, e)
        self.assertIs(type(f), MinExpression)

    def test_max(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = MaxExpression((m.x, 2))

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)

        f = e.create_node_with_local_data((2, 3, 4))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (2, 3, 4))

        e = NPV_MaxExpression((m.p, 2))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))

        f = e.create_node_with_local_data((m.p, m.x))
        self.assertIsNot(f, e)
        self.assertIs(type(f), MaxExpression)

    def test_prod(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = ProductExpression((m.x, 2))

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)

        f = e.create_node_with_local_data((m.p, 3))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (m.p, 3))

        e = NPV_ProductExpression((m.p, 2))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))

        f = e.create_node_with_local_data((m.p, m.x))
        self.assertIsNot(f, e)
        self.assertIs(type(f), ProductExpression)

    def test_monomial(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = MonomialTermExpression((2, m.x))

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        # Note that MonomialTermExpression recreates the args tuple
        self.assertEqual(f.args, e.args)

        f = e.create_node_with_local_data((m.p, 3))
        self.assertIsNot(f, e)
        self.assertIs(type(f), NPV_ProductExpression)
        self.assertEqual(f.args, (m.p, 3))

        f = e.create_node_with_local_data((m.x, m.x))
        self.assertIsNot(f, e)
        self.assertIs(type(f), ProductExpression)
        self.assertEqual(f.args, (m.x, m.x))

    def test_division(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = DivisionExpression((m.x, 2))

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)

        f = e.create_node_with_local_data((2, 3))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (2, 3))

        e = NPV_DivisionExpression((m.p, 2))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))

        f = e.create_node_with_local_data((m.p, m.x))
        self.assertIsNot(f, e)
        self.assertIs(type(f), DivisionExpression)

    def test_sum(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = SumExpression((m.x, 2))

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIsNot(f._args_, e._args_)
        self.assertIsNot(f.args, e.args)

        f = e.create_node_with_local_data(e._args_)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f._args_, e._args_)
        self.assertIsNot(f.args, e.args)

        f = e.create_node_with_local_data((m.x, 2, 3))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, [m.x, 2, 3])

        e = NPV_SumExpression((m.p, 2))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))

        f = e.create_node_with_local_data((m.p, m.x))
        self.assertIsNot(f, e)
        self.assertIs(type(f), LinearExpression)
        assertExpressionsStructurallyEqual(self, f.args, [m.p, m.x])

        f = e.create_node_with_local_data((m.p, m.x**2))
        self.assertIsNot(f, e)
        self.assertIs(type(f), SumExpression)
        assertExpressionsStructurallyEqual(self, f.args, [m.p, PowExpression((m.x, 2))])

    def test_linear(self):
        m = ConcreteModel()
        m.x = Var(range(3), initialize=range(3))
        m.y = Var(initialize=5)
        with mutable_expression() as e:
            for i in range(3):
                e += i * m.x[i]
            e += 5
            e += m.y
            e -= 3

    def test_expr_if(self):
        m = ConcreteModel()
        m.x = Var(range(3), initialize=range(3))
        m.y = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = Expr_if(IF=m.y >= 5, THEN=m.x[0] + 5, ELSE=m.x[1] ** 2)

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)

        f = e.create_node_with_local_data((2, 3, 4))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (2, 3, 4))

        e = NPV_Expr_ifExpression((m.p <= 5, 2, m.p))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))

        f = e.create_node_with_local_data((m.p <= 5, m.x, m.p))
        self.assertIsNot(f, e)
        self.assertIs(type(f), Expr_ifExpression)

    def test_unary_fcn(self):
        m = ConcreteModel()
        m.x = Var(range(3), initialize=range(3))
        m.y = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = sin(2 * m.y)

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)
        self.assertIs(e._fcn, f._fcn)
        self.assertIs(e._name, f._name)

        f = e.create_node_with_local_data((m.x[1],))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (m.x[1],))
        self.assertIs(e._fcn, f._fcn)
        self.assertIs(e._name, f._name)

        e = sin((2 * m.p))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(e._fcn, f._fcn)
        self.assertIs(e._name, f._name)

        f = e.create_node_with_local_data((m.x[1],))
        self.assertIsNot(f, e)
        self.assertIs(type(f), UnaryFunctionExpression)
        self.assertEqual(f.args, (m.x[1],))
        self.assertIs(e._fcn, f._fcn)
        self.assertIs(e._name, f._name)

    def test_abs(self):
        m = ConcreteModel()
        m.x = Var(range(3), initialize=range(3))
        m.y = Var(initialize=5)
        m.p = Param(initialize=3, mutable=True)
        e = abs(2 * m.y)

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)
        self.assertIs(e._fcn, f._fcn)
        self.assertIs(e._name, f._name)

        f = e.create_node_with_local_data((m.x[1],))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (m.x[1],))
        self.assertIs(e._fcn, f._fcn)
        self.assertIs(e._name, f._name)

        e = abs((2 * m.p))
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(e._fcn, f._fcn)
        self.assertIs(e._name, f._name)

        f = e.create_node_with_local_data((m.x[1],))
        self.assertIsNot(f, e)
        self.assertIs(type(f), AbsExpression)
        self.assertEqual(f.args, (m.x[1],))
        self.assertIs(e._fcn, f._fcn)
        self.assertIs(e._name, f._name)

    def test_external(self):
        m = ConcreteModel()
        m.x = Var(initialize=16)
        m.p = Param(initialize=32, mutable=True)
        fcn = MockExternalFunction()
        e = ExternalFunctionExpression((2 * m.x,), fcn)

        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(f.args, e.args)
        self.assertIs(e._fcn, f._fcn)

        f = e.create_node_with_local_data((m.x,))
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertEqual(f.args, (m.x,))
        self.assertIs(e._fcn, f._fcn)

        e = NPV_ExternalFunctionExpression((2 * m.p,), fcn)
        f = e.create_node_with_local_data(e.args)
        self.assertIsNot(f, e)
        self.assertIs(type(f), type(e))
        self.assertIs(e._fcn, f._fcn)

        f = e.create_node_with_local_data((m.x,))
        self.assertIsNot(f, e)
        self.assertIs(type(f), ExternalFunctionExpression)
        self.assertEqual(f.args, (m.x,))
        self.assertIs(e._fcn, f._fcn)

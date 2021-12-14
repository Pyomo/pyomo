#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.common import unittest
import pyomo.environ as pe
from pyomo.core.expr.numeric_expr import (
    LinearExpression, MonomialTermExpression, SumExpression,
    ProductExpression, DivisionExpression, PowExpression,
    NegationExpression, UnaryFunctionExpression, ExternalFunctionExpression,
    Expr_ifExpression, AbsExpression
)
from pyomo.core.expr.logical_expr import (
    InequalityExpression, EqualityExpression, RangedExpression
)
from pyomo.core.expr.compare import (
    convert_expression_to_prefix_notation, compare_expressions
)
from pyomo.common.getGSL import find_GSL


class TestConvertToPrefixNotation(unittest.TestCase):
    def test_linear_expression(self):
        m = pe.ConcreteModel()
        m.x = pe.Var([1, 2, 3, 4])
        e = LinearExpression(constant=3, linear_coefs=list(m.x.keys()), linear_vars=list(m.x.values()))
        expected = [(LinearExpression, 9), 3, 1, 2, 3, 4, m.x[1], m.x[2], m.x[3], m.x[4]]
        pn = convert_expression_to_prefix_notation(e)
        self.assertEqual(pn, expected)

    def test_multiple(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()

        e = m.x**2 + m.x*m.y/3 + 4
        expected = [(SumExpression, 3),
                    (PowExpression, 2),
                    m.x,
                    2,
                    (DivisionExpression, 2),
                    (ProductExpression, 2),
                    m.x,
                    m.y,
                    3,
                    4]
        pn = convert_expression_to_prefix_notation(e)
        self.assertEqual(pn, expected)
        e2 = m.x**2 + m.x*m.y/3 + 4
        e3 = m.y**2 + m.x*m.y/3 + 4
        self.assertTrue(compare_expressions(e, e2))
        self.assertFalse(compare_expressions(e, e3))

    def test_unary(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        e = pe.log(m.x)
        expected = [(UnaryFunctionExpression, 1, 'log'), m.x]
        pn = convert_expression_to_prefix_notation(e)
        self.assertEqual(expected, pn)

    def test_external_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find the amplgsl.dll library')

        m = pe.ConcreteModel()
        m.hypot = pe.ExternalFunction(library=DLL, function='gsl_hypot')
        m.x = pe.Var(initialize=0.5)
        m.y = pe.Var(initialize=1.5)
        e = 2 * m.hypot(m.x, m.x*m.y)
        expected = [(ProductExpression, 2),
                    2,
                    (ExternalFunctionExpression, 2, m.hypot),
                    m.x,
                    (ProductExpression, 2),
                    m.x,
                    m.y]
        pn = convert_expression_to_prefix_notation(e)
        self.assertEqual(expected, pn)

    def test_var(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        pn = convert_expression_to_prefix_notation(m.x)
        self.assertEqual(pn, [m.x])

    def test_float(self):
        pn = convert_expression_to_prefix_notation(4.3)
        self.assertEqual(pn, [4.3])

    def test_monomial(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        e = 2*m.x
        pn = convert_expression_to_prefix_notation(e)
        expected = [(MonomialTermExpression, 2),
                    2,
                    m.x]
        self.assertEqual(pn, expected)

    def test_negation(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        e = -m.x**2
        pn = convert_expression_to_prefix_notation(e)
        expected = [(NegationExpression, 1),
                    (PowExpression, 2),
                    m.x,
                    2]
        self.assertEqual(pn, expected)

    def test_abs(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        e = abs(m.x)
        pn = convert_expression_to_prefix_notation(e)
        expected = [(AbsExpression, 1, 'abs'),
                    m.x]
        self.assertEqual(pn, expected)

    def test_expr_if(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        e = pe.Expr_if(m.x <= 0, m.y + m.x == 0, m.y - m.x == 0)
        pn = convert_expression_to_prefix_notation(e)
        expected = [(Expr_ifExpression, 3),
                    (InequalityExpression, 2),
                    m.x,
                    0,
                    (EqualityExpression, 2),
                    (SumExpression, 2),
                    m.y,
                    m.x,
                    0,
                    (EqualityExpression, 2),
                    (SumExpression, 2),
                    m.y,
                    (MonomialTermExpression, 2),
                    -1,
                    m.x,
                    0]
        self.assertEqual(pn, expected)

    def test_ranged_expression(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        e = pe.inequality(-1, m.x, 1)
        pn = convert_expression_to_prefix_notation(e)
        expected = [(RangedExpression, 3),
                    -1,
                    m.x,
                    1]
        self.assertEqual(pn, expected)

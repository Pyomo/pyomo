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

from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest

from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.expr.numeric_expr import (
    LinearExpression,
    MonomialTermExpression,
    SumExpression,
)
from pyomo.repn.quadratic import QuadraticRepnVisitor

from pyomo.environ import ConcreteModel, Var


class VisitorConfig(object):
    def __init__(self):
        self.subexpr = {}
        self.var_map = {}
        self.var_order = {}

    def __iter__(self):
        return iter((self.subexpr, self.var_map, self.var_order))


class TestQuadratic(unittest.TestCase):
    def test_product(self):
        m = ConcreteModel()
        m.x = Var()

        e = 2 + 3 * m.x + 4 * m.x**2

        cfg = VisitorConfig()
        visitor = QuadraticRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertStructuredAlmostEqual(repn.constant, 2)
        self.assertStructuredAlmostEqual(repn.linear, {id(m.x): 3})
        self.assertStructuredAlmostEqual(repn.quadratic, {(id(m.x), id(m.x)): 4})
        self.assertEqual(repn.nonlinear, None)

        e = (2 + 3 * m.x + 4 * m.x**2) * (5 + 6 * m.x + 7 * m.x**2)

        cfg = VisitorConfig()
        visitor = QuadraticRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        QE4 = SumExpression([4 * m.x**2])
        QE7 = SumExpression([7 * m.x**2])
        LE3 = MonomialTermExpression((3, m.x))
        LE6 = MonomialTermExpression((6, m.x))
        NL = +QE4 * (QE7 + LE6) + (LE3) * (QE7)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertStructuredAlmostEqual(repn.constant, 10)
        self.assertStructuredAlmostEqual(repn.linear, {id(m.x): 27})
        self.assertStructuredAlmostEqual(repn.quadratic, {(id(m.x), id(m.x)): 52})
        assertExpressionsEqual(self, repn.nonlinear, NL)

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

from pyomo.core import ConcreteModel, Var, Expression
import pyomo.core.base.expr as EXPR
from pyomo.core.base.expression import _ExpressionData
from pyomo.gdp.util import clone_without_expression_components

class TestGDPUtils(unittest.TestCase):
    def test_clone_expr_no_expressions(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.y = Var(initialize=3)
        m.e = Expression(expr=m.x**2 + m.x - 1)

        base = m.x**2 + 1
        test = clone_without_expression_components(base, {})
        self.assertIsNot(base, test)
        self.assertEqual(base(), test())
        test = clone_without_expression_components(base, {id(m.x): m.y})
        self.assertEqual(3**2+1, test())

        base = m.e
        test = clone_without_expression_components(base, {})
        self.assertIsNot(base, test)
        self.assertEqual(base(), test())
        self.assertIsInstance(base, _ExpressionData)
        self.assertIsInstance(test, EXPR._SumExpression)
        test = clone_without_expression_components(base, {id(m.x): m.y})
        self.assertEqual(3**2+3-1, test())

        base = m.e + m.x
        test = clone_without_expression_components(base, {})
        self.assertIsNot(base, test)
        self.assertEqual(base(), test())
        self.assertIsInstance(base, EXPR._SumExpression)
        self.assertIsInstance(test, EXPR._SumExpression)
        self.assertIsInstance(base._args[0], _ExpressionData)
        self.assertIsInstance(test._args[0], EXPR._SumExpression)
        test = clone_without_expression_components(base, {id(m.x): m.y})
        self.assertEqual(3**2+3-1 + 3, test())

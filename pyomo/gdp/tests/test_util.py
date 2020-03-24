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

from pyomo.core import ConcreteModel, Var, Expression, Block
import pyomo.core.expr.current as EXPR
from pyomo.core.base.expression import _ExpressionData
from pyomo.gdp.util import clone_without_expression_components, is_child_of


class TestGDPUtils(unittest.TestCase):
    def test_clone_without_expression_components(self):
        m = ConcreteModel()
        m.x = Var(initialize=5)
        m.y = Var(initialize=3)
        m.e = Expression(expr=m.x**2 + m.x - 1)

        base = m.x**2 + 1
        test = clone_without_expression_components(base, {})
        self.assertIs(base, test)
        self.assertEqual(base(), test())
        test = clone_without_expression_components(base, {id(m.x): m.y})
        self.assertEqual(3**2+1, test())

        base = m.e
        test = clone_without_expression_components(base, {})
        self.assertIsNot(base, test)
        self.assertEqual(base(), test())
        self.assertIsInstance(base, _ExpressionData)
        self.assertIsInstance(test, EXPR.SumExpression)
        test = clone_without_expression_components(base, {id(m.x): m.y})
        self.assertEqual(3**2+3-1, test())

        base = m.e + m.x
        test = clone_without_expression_components(base, {})
        self.assertIsNot(base, test)
        self.assertEqual(base(), test())
        self.assertIsInstance(base, EXPR.SumExpression)
        self.assertIsInstance(test, EXPR.SumExpression)
        self.assertIsInstance(base.arg(0), _ExpressionData)
        self.assertIsInstance(test.arg(0), EXPR.SumExpression)
        test = clone_without_expression_components(base, {id(m.x): m.y})
        self.assertEqual(3**2+3-1 + 3, test())

    def test_is_child_of(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.b_indexed = Block([1,2])
        m.b_parallel = Block()
        
        knownBlocks = {}
        self.assertFalse(is_child_of(parent=m.b, child=m.b_parallel,
                                     knownBlocks=knownBlocks))
        self.assertEqual(len(knownBlocks), 2)
        self.assertFalse(knownBlocks.get(m))
        self.assertFalse(knownBlocks.get(m.b_parallel))
        self.assertTrue(is_child_of(parent=m.b, child=m.b.b_indexed[1],
                                    knownBlocks=knownBlocks))
        self.assertEqual(len(knownBlocks), 4)
        self.assertFalse(knownBlocks.get(m))
        self.assertFalse(knownBlocks.get(m.b_parallel))
        self.assertTrue(knownBlocks.get(m.b.b_indexed[1]))
        self.assertTrue(knownBlocks.get(m.b.b_indexed))

if __name__ == '__main__':
    unittest.main()

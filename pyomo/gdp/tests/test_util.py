#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

from pyomo.core import ConcreteModel, Var, Expression, Block
import pyomo.core.expr.current as EXPR
from pyomo.core.base.expression import _ExpressionData
from pyomo.gdp.util import (clone_without_expression_components, is_child_of,
                            get_gdp_tree)
from pyomo.gdp import Disjunct, Disjunction

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

    def test_gdp_tree(self):
        m = ConcreteModel()
        m.x = Var()
        m.block = Block()
        m.block.d1 = Disjunct()
        m.block.d1.dd1 = Disjunct()
        m.disj1 = Disjunct()
        m.block.disjunction = Disjunction(expr=[m.block.d1, m.disj1])
        m.block.d1.b = Block()
        m.block.d1.b.dd2 = Disjunct()
        m.block.d1.b.dd3 = Disjunct()
        m.block.d1.disjunction = Disjunction(expr=[m.block.d1.dd1,
                                                   m.block.d1.b.dd2,
                                                   m.block.d1.b.dd3])
        m.block.d1.b.dd2.disjunction = Disjunction(expr=[[m.x >= 1], [m.x <=
                                                                      -1]])
        targets = (m,)
        knownBlocks = {}
        tree = get_gdp_tree(targets, m, knownBlocks)

        # check tree structure first
        vertices = tree.vertices
        self.assertEqual(len(vertices), 10)
        in_degrees = {m.block.d1 : 1,
                      m.block.disjunction : 0,
                      m.disj1 : 1,
                      m.block.d1.disjunction : 1,
                      m.block.d1.dd1 : 1,
                      m.block.d1.b.dd2 : 1,
                      m.block.d1.b.dd3 : 1,
                      m.block.d1.b.dd2.disjunction : 1,
                      m.block.d1.b.dd2.disjunction.disjuncts[0] : 1,
                      m.block.d1.b.dd2.disjunction.disjuncts[1] : 1
                      }
        for key, val in in_degrees.items():
            self.assertEqual(tree.in_degree(key), val)

        # This should be deterministic, so we can just check the order
        topo_sort = [m.block.disjunction, m.disj1, m.block.d1,
                     m.block.d1.disjunction, m.block.d1.b.dd3, m.block.d1.b.dd2,
                     m.block.d1.b.dd2.disjunction,
                     m.block.d1.b.dd2.disjunction.disjuncts[1],
                     m.block.d1.b.dd2.disjunction.disjuncts[0], m.block.d1.dd1]
        sort = tree.topological_sort()
        for i, node in enumerate(sort):
            self.assertIs(node, topo_sort[i])

if __name__ == '__main__':
    unittest.main()

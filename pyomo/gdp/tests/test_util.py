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

from pyomo.core import ConcreteModel, Var, Expression, Block, RangeSet, Any
import pyomo.core.expr as EXPR
from pyomo.core.base.expression import NamedExpressionData
from pyomo.gdp.util import (
    clone_without_expression_components,
    is_child_of,
    get_gdp_tree,
)
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
        self.assertEqual(3**2 + 1, test())

        base = m.e
        test = clone_without_expression_components(base, {})
        self.assertIsNot(base, test)
        self.assertEqual(base(), test())
        self.assertIsInstance(base, NamedExpressionData)
        self.assertIsInstance(test, EXPR.SumExpression)
        test = clone_without_expression_components(base, {id(m.x): m.y})
        self.assertEqual(3**2 + 3 - 1, test())

        base = m.e + m.x
        test = clone_without_expression_components(base, {})
        self.assertIsNot(base, test)
        self.assertEqual(base(), test())
        self.assertIsInstance(base, EXPR.SumExpression)
        self.assertIsInstance(test, EXPR.SumExpression)
        self.assertIsInstance(base.arg(0), NamedExpressionData)
        self.assertIsInstance(test.arg(0), EXPR.SumExpression)
        test = clone_without_expression_components(base, {id(m.x): m.y})
        self.assertEqual(3**2 + 3 - 1 + 3, test())

    def test_is_child_of(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.b_indexed = Block([1, 2])
        m.b_parallel = Block()

        knownBlocks = {}
        self.assertFalse(
            is_child_of(parent=m.b, child=m.b_parallel, knownBlocks=knownBlocks)
        )
        self.assertEqual(len(knownBlocks), 2)
        self.assertFalse(knownBlocks.get(m))
        self.assertFalse(knownBlocks.get(m.b_parallel))
        self.assertTrue(
            is_child_of(parent=m.b, child=m.b.b_indexed[1], knownBlocks=knownBlocks)
        )
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
        m.block.d1.disjunction = Disjunction(
            expr=[m.block.d1.dd1, m.block.d1.b.dd2, m.block.d1.b.dd3]
        )
        m.block.d1.b.dd2.disjunction = Disjunction(expr=[[m.x >= 1], [m.x <= -1]])
        targets = (m,)
        knownBlocks = {}
        tree = get_gdp_tree(targets, m, knownBlocks)

        # check tree structure first
        vertices = tree.vertices
        self.assertEqual(len(vertices), 10)
        in_degrees = {
            m.block.d1: 1,
            m.block.disjunction: 0,
            m.disj1: 1,
            m.block.d1.disjunction: 1,
            m.block.d1.dd1: 1,
            m.block.d1.b.dd2: 1,
            m.block.d1.b.dd3: 1,
            m.block.d1.b.dd2.disjunction: 1,
            m.block.d1.b.dd2.disjunction.disjuncts[0]: 1,
            m.block.d1.b.dd2.disjunction.disjuncts[1]: 1,
        }
        for key, val in in_degrees.items():
            self.assertEqual(tree.in_degree(key), val)

        # This should be deterministic, so we can just check the order
        topo_sort = [
            m.block.disjunction,
            m.disj1,
            m.block.d1,
            m.block.d1.disjunction,
            m.block.d1.b.dd3,
            m.block.d1.b.dd2,
            m.block.d1.b.dd2.disjunction,
            m.block.d1.b.dd2.disjunction.disjuncts[1],
            m.block.d1.b.dd2.disjunction.disjuncts[0],
            m.block.d1.dd1,
        ]
        sort = tree.topological_sort()
        for i, node in enumerate(sort):
            self.assertIs(node, topo_sort[i])

    def add_indexed_disjunction(self, parent, m):
        parent.indexed = Disjunction(Any)
        parent.indexed[1] = [
            [sum(m.x[i] ** 2 for i in m.I) <= 1],
            [sum((3 - m.x[i]) ** 2 for i in m.I) <= 1],
        ]
        parent.indexed[0] = [
            [(m.x[1] - 1) ** 2 + m.x[2] ** 2 <= 1],
            [-((m.x[1] - 2) ** 2) - (m.x[2] - 3) ** 2 >= -1],
        ]

    def test_gdp_tree_indexed_disjunction(self):
        # This is to check that indexed components never actually appear as
        # nodes in the tree. We should only have DisjunctionDatas and
        # DisjunctDatas.
        m = ConcreteModel()
        m.I = RangeSet(1, 4)
        m.x = Var(m.I, bounds=(-2, 6))
        self.add_indexed_disjunction(m, m)

        targets = (m.indexed,)
        knownBlocks = {}
        tree = get_gdp_tree(targets, m, knownBlocks)

        vertices = tree.vertices
        self.assertEqual(len(vertices), 6)
        in_degrees = {
            m.indexed[0]: 0,
            m.indexed[1]: 0,
            m.indexed[0].disjuncts[0]: 1,
            m.indexed[0].disjuncts[1]: 1,
            m.indexed[1].disjuncts[0]: 1,
            m.indexed[1].disjuncts[1]: 1,
        }
        for key, val in in_degrees.items():
            self.assertEqual(tree.in_degree(key), val)

        topo_sort = [
            m.indexed[0],
            m.indexed[0].disjuncts[1],
            m.indexed[0].disjuncts[0],
            m.indexed[1],
            m.indexed[1].disjuncts[1],
            m.indexed[1].disjuncts[0],
        ]
        sort = tree.topological_sort()
        for i, node in enumerate(sort):
            self.assertIs(node, topo_sort[i])

    def test_gdp_tree_nested_indexed_disjunction(self):
        m = ConcreteModel()
        m.I = RangeSet(1, 4)
        m.x = Var(m.I, bounds=(-2, 6))
        m.disj1 = Disjunct()
        self.add_indexed_disjunction(m.disj1, m)
        m.disj2 = Disjunct()
        m.another_disjunction = Disjunction(expr=[m.disj1, m.disj2])

        # First, we still just give the indexed disjunction as a target, and
        # make sure that we don't pick up the parent Disjunct in the tree, since
        # it is not in targets.
        targets = (m.disj1.indexed,)
        knownBlocks = {}
        tree = get_gdp_tree(targets, m, knownBlocks)

        vertices = tree.vertices
        self.assertEqual(len(vertices), 6)
        in_degrees = {
            m.disj1.indexed[0]: 0,
            m.disj1.indexed[1]: 0,
            m.disj1.indexed[0].disjuncts[0]: 1,
            m.disj1.indexed[0].disjuncts[1]: 1,
            m.disj1.indexed[1].disjuncts[0]: 1,
            m.disj1.indexed[1].disjuncts[1]: 1,
        }
        for key, val in in_degrees.items():
            self.assertEqual(tree.in_degree(key), val)

        topo_sort = [
            m.disj1.indexed[0],
            m.disj1.indexed[0].disjuncts[1],
            m.disj1.indexed[0].disjuncts[0],
            m.disj1.indexed[1],
            m.disj1.indexed[1].disjuncts[1],
            m.disj1.indexed[1].disjuncts[0],
        ]
        sort = tree.topological_sort()
        for i, node in enumerate(sort):
            self.assertIs(node, topo_sort[i])

        # Now, let targets be everything and make sure that we get the correct
        # tree.
        targets = (m,)
        tree = get_gdp_tree(targets, m, knownBlocks)
        vertices = tree.vertices
        self.assertEqual(len(vertices), 9)
        # update that now the disjunctions have a parent
        in_degrees[m.disj1.indexed[0]] = 1
        in_degrees[m.disj1.indexed[1]] = 1
        # and add new nodes
        in_degrees[m.disj1] = 1
        in_degrees[m.disj2] = 1
        in_degrees[m.another_disjunction] = 0
        for key, val in in_degrees.items():
            self.assertEqual(tree.in_degree(key), val)

        topo_sort = [
            m.another_disjunction,
            m.disj2,
            m.disj1,
            m.disj1.indexed[1],
            m.disj1.indexed[1].disjuncts[1],
            m.disj1.indexed[1].disjuncts[0],
            m.disj1.indexed[0],
            m.disj1.indexed[0].disjuncts[1],
            m.disj1.indexed[0].disjuncts[0],
        ]
        sort = tree.topological_sort()
        for i, node in enumerate(sort):
            self.assertIs(node, topo_sort[i])


if __name__ == '__main__':
    unittest.main()

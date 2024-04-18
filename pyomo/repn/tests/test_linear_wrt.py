#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import Binary, ConcreteModel, Var, log
from pyomo.repn.linear_wrt import MultilevelLinearRepnVisitor
from pyomo.repn.tests.test_linear import VisitorConfig


class TestMultilevelLinearRepnVisitor(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 45))
        m.y = Var(domain=Binary)

        return m

    def test_walk_sum(self):
        m = self.make_model()
        e = m.x + m.y
        cfg = VisitorConfig()
        visitor = MultilevelLinearRepnVisitor(*cfg, wrt=[m.x])

        repn = visitor.walk_expression(e)

        self.assertIsNone(repn.nonlinear)
        self.assertEqual(len(repn.linear), 1)
        self.assertIn(id(m.x), repn.linear)
        self.assertEqual(repn.linear[id(m.x)], 1)
        self.assertIs(repn.constant, m.y)
        self.assertEqual(repn.multiplier, 1)

    def test_bilinear_term(self):
        m = self.make_model()
        e = m.x * m.y
        cfg = VisitorConfig()
        visitor = MultilevelLinearRepnVisitor(*cfg, wrt=[m.x])

        repn = visitor.walk_expression(e)

        self.assertIsNone(repn.nonlinear)
        self.assertEqual(len(repn.linear), 1)
        self.assertIn(id(m.x), repn.linear)
        self.assertIs(repn.linear[id(m.x)], m.y)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.multiplier, 1)

    def test_distributed_bilinear_term(self):
        m = self.make_model()
        e = m.y * (m.x + 7)
        cfg = VisitorConfig()
        visitor = MultilevelLinearRepnVisitor(*cfg, wrt=[m.x])

        repn = visitor.walk_expression(e)

        self.assertIsNone(repn.nonlinear)
        self.assertEqual(len(repn.linear), 1)
        self.assertIn(id(m.x), repn.linear)
        self.assertIs(repn.linear[id(m.x)], m.y)
        assertExpressionsEqual(
            self,
            repn.constant,
            m.y * 7
        )
        self.assertEqual(repn.multiplier, 1)

    def test_monomial(self):
        m = self.make_model()
        e = 45 * m.y
        cfg = VisitorConfig()
        visitor = MultilevelLinearRepnVisitor(*cfg, wrt=[m.y])

        repn = visitor.walk_expression(e)

        self.assertIsNone(repn.nonlinear)
        self.assertEqual(len(repn.linear), 1)
        self.assertIn(id(m.y), repn.linear)
        self.assertEqual(repn.linear[id(m.y)], 45)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.multiplier, 1)

    def test_constant(self):
        m = self.make_model()
        e = 45 * m.y
        cfg = VisitorConfig()
        visitor = MultilevelLinearRepnVisitor(*cfg, wrt=[m.x])

        repn = visitor.walk_expression(e)
        
        self.assertIsNone(repn.nonlinear)
        self.assertEqual(len(repn.linear), 0)
        assertExpressionsEqual(
            self,
            repn.constant,
            45 * m.y
        )
        self.assertEqual(repn.multiplier, 1)

    def test_fixed_var(self):
        m = self.make_model()
        m.x.fix(42)
        e = (m.y ** 2) * (m.x + m.x ** 2)

        cfg = VisitorConfig()
        visitor = MultilevelLinearRepnVisitor(*cfg, wrt=[m.x])

        repn = visitor.walk_expression(e)

        self.assertIsNone(repn.nonlinear)
        self.assertEqual(len(repn.linear), 0)
        assertExpressionsEqual(
            self,
            repn.constant,
            (m.y ** 2) * 1806
        )
        self.assertEqual(repn.multiplier, 1)

    def test_nonlinear(self):
        m = self.make_model()
        e = (m.y * log(m.x)) * (m.y + 2) / m.x
        
        cfg = VisitorConfig()
        visitor = MultilevelLinearRepnVisitor(*cfg, wrt=[m.x])

        repn = visitor.walk_expression(e)

        self.assertEqual(len(repn.linear), 0)
        self.assertEqual(repn.multiplier, 1)
        print(repn.nonlinear)
        assertExpressionsEqual(
            self,
            repn.nonlinear,
            log(m.x) * (m.y *(m.y + 2))/m.x
        )

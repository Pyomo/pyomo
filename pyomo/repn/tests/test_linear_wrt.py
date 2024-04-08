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
from pyomo.environ import Binary, ConcreteModel, Var
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
        print("constructing")
        visitor = MultilevelLinearRepnVisitor(*cfg, wrt=[m.x])

        repn = visitor.walk_expression(e)

        self.assertIsNone(repn.nonlinear)
        self.assertEqual(len(repn.linear), 1)
        self.assertIn(id(m.x), repn.linear)
        self.assertEqual(repn.linear[id(m.x)], 1)
        self.assertIs(repn.constant, m.y)
        self.assertEqual(repn.multiplier, 1)

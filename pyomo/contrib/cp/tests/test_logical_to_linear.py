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

import pyomo.common.unittest as unittest
from pyomo.contrib.cp.logical_to_linear import LogicalToLinearVisitor
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (ConcreteModel, BooleanVar)

class TestLogicalToLinear(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()

        m.a = BooleanVar()
        m.b = BooleanVar()
        m.c = BooleanVar()
        
        return m

    def test_implication(self):
        m = self.make_model()
        e = m.a.implies(m.b.land(m.c))

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        m.pprint()
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        self.assertIs(m.c.get_associated_binary(), m.z[3])

        self.assertEqual(len(m.cons), 5)
        # z4 = b ^ c
        assertExpressionsEqual(self, m.cons[1].expr, m.z[2] <= m.z[4])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[3] <= m.z[4])
        # z5 = a -> z4
        # which means z5 = !a v z4
        assertExpressionsEqual(self, m.cons[3].expr, 
                               (1 - m.z[5]) + (1 - m.z[1]) + m.z[4] >= 1)
        assertExpressionsEqual(self, m.cons[4].expr, m.z5 >= 1 - m.z[1])
        assertExpressionsEqual(self, m.cons[5].expr, m.z5 >= z4)
        

        # z5 is fixed 'True'
        self.assertTrue(m.z[5].fixed)
        self.assertTrue(value(m.z[5]))

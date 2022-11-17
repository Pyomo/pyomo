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
from pyomo.environ import (
    atmost, atleast, BooleanVar, ConcreteModel, Expression, Integers, land,
    lnot, lor, value, Var)

class TestLogicalToLinearVisitor(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()

        m.a = BooleanVar()
        m.b = BooleanVar()
        m.c = BooleanVar()

        return m

    def test_logical_or(self):
        m = self.make_model()
        e = lor(m.a, m.b, m.c)

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)

        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        self.assertIs(m.c.get_associated_binary(), m.z[3])

        self.assertEqual(len(m.cons), 4)
        self.assertEqual(len(m.z), 4)
        # !z4 v a v b v c
        assertExpressionsEqual(
            self, m.cons[1].expr, 1 - m.z[4] + m.z[1] + m.z[2] + m.z[3] >= 1)
        # z4 v !a
        assertExpressionsEqual(
            self, m.cons[2].expr, m.z[4] + (1 - m.z[1]) >= 1)
        # z4 v !b
        assertExpressionsEqual(
            self, m.cons[3].expr, m.z[4] + (1 - m.z[2]) >= 1)
        # z4 v !c
        assertExpressionsEqual(
            self, m.cons[4].expr, m.z[4] + (1 - m.z[3]) >= 1)

        # z4 is fixed 'True'
        self.assertTrue(m.z[4].fixed)
        self.assertEqual(value(m.z[4]), 1)

    def test_logical_and(self):
        m = self.make_model()
        e = land(m.a, m.b, m.c)

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)

        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        self.assertIs(m.c.get_associated_binary(), m.z[3])

        self.assertEqual(len(m.cons), 3)
        self.assertEqual(len(m.z), 4)
        assertExpressionsEqual(
            self, m.cons[1].expr, m.z[4] <= m.z[1])
        assertExpressionsEqual(
            self, m.cons[2].expr, m.z[4] <= m.z[2])
        assertExpressionsEqual(
            self, m.cons[3].expr, m.z[4] <= m.z[3])

        self.assertTrue(m.z[4].fixed)
        self.assertEqual(value(m.z[4]), 1)

    def test_logical_not(self):
        m = self.make_model()
        e = lnot(m.a)

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.cons), 1)
        self.assertEqual(len(m.z), 2)
        assertExpressionsEqual(
            self, m.cons[1].expr, m.z[2] == 1 - m.z[1])
        self.assertTrue(m.z[2].fixed)
        self.assertEqual(value(m.z[2]), 1)

    def test_implication(self):
        m = self.make_model()
        e = m.a.implies(m.b.land(m.c))

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        self.assertIs(m.c.get_associated_binary(), m.z[3])

        self.assertEqual(len(m.cons), 5)
        # z4 = b ^ c
        assertExpressionsEqual(self, m.cons[1].expr, m.z[4] <= m.z[2])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[4] <= m.z[3])
        # z5 = a -> z4
        # which means z5 = !a v z4
        assertExpressionsEqual(self, m.cons[3].expr,
                               (1 - m.z[5]) + (1 - m.z[1]) + m.z[4] >= 1)
        # z5 >= 1 - z1
        assertExpressionsEqual(self, m.cons[4].expr,
                               m.z[5] + (1 - (1 - m.z[1])) >= 1)
        # z5 >= z4
        assertExpressionsEqual(self, m.cons[5].expr, m.z[5] + (1 - m.z[4]) >= 1)

        # z5 is fixed 'True'
        self.assertTrue(m.z[5].fixed)
        self.assertTrue(value(m.z[5]))

    def test_equivalence(self):
        m = self.make_model()
        e = m.a.equivalent_to(m.c)

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.c.get_associated_binary(), m.z[2])
        m.pprint()
        self.assertEqual(len(m.z), 5)
        self.assertEqual(len(m.cons), 8)

        assertExpressionsEqual(
            self, m.cons[1].expr, (1 - m.z[3]) + (1 - m.z[1]) + m.z[2] >= 1)
        assertExpressionsEqual(
            self, m.cons[2].expr, 1 - (1 - m.z[1]) + m.z[3] >= 1)
        assertExpressionsEqual(
            self, m.cons[3].expr, m.z[3] + (1 - m.z[2]) >= 1)

        assertExpressionsEqual(
            self, m.cons[4].expr, (1 - m.z[4]) + (1 - m.z[2]) + m.z[1] >= 1)
        assertExpressionsEqual(
            self, m.cons[5].expr, m.z[4] + (1 - m.z[1]) >= 1)
        assertExpressionsEqual(
            self, m.cons[6].expr, 1 - (1 - m.z[2]) + m.z[4] >= 1)

        assertExpressionsEqual(
            self, m.cons[7].expr, m.z[5] <= m.z[3])
        assertExpressionsEqual(
            self, m.cons[8].expr, m.z[5] <= m.z[4])

        self.assertTrue(m.z[5].fixed)
        self.assertEqual(value(m.z[5]), 1)

    def test_xor(self):
        m = self.make_model()
        e = m.a.xor(m.b)

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])

        self.assertEqual(len(m.z), 5)
        self.assertEqual(len(m.cons), 5)

        assertExpressionsEqual(
            self, m.cons[1].expr, m.z[1] + m.z[2] <= 1 + (1 - m.z[3]))
        assertExpressionsEqual(
            self, m.cons[2].expr, m.z[1] + m.z[2] >= 1 - (1 - m.z[3]))

        assertExpressionsEqual(
            self, m.cons[3].expr, 1 - m.z[3] == m.z[4] + m.z[5])
        assertExpressionsEqual(
            self, m.cons[4].expr, m.z[1] + m.z[2] >= 2 - 2*(1 - m.z[4]))
        assertExpressionsEqual(
            self, m.cons[5].expr, m.z[1] + m.z[2] <= 2*(1 - m.z[5]))

        self.assertTrue(m.z[3].fixed)
        self.assertEqual(value(m.z[3]), 1)

    def test_at_most(self):
        m = self.make_model()
        e = atmost(2, m.a, (m.a.land(m.b)), m.c)

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        a = m.z[1]
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        b = m.z[2]
        self.assertIs(m.c.get_associated_binary(), m.z[4])
        c = m.z[4]

        self.assertEqual(len(m.z), 5)
        self.assertEqual(len(m.cons), 4)

        # z3 = a ^ b
        assertExpressionsEqual(
            self, m.cons[1].expr, m.z[3] <= a)
        assertExpressionsEqual(
            self, m.cons[2].expr, m.z[3] <= b)

        # bigm of atmost disjunction
        assertExpressionsEqual(
            self, m.cons[3].expr, a + m.z[3] + c <= 2 + (1 - m.z[5]))
        assertExpressionsEqual(
            self, m.cons[4].expr, a + m.z[3] + c >= 3 - 3*m.z[5])

        self.assertTrue(m.z[5].fixed)
        self.assertEqual(value(m.z[5]), 1)

    def test_at_least(self):
        m = self.make_model()
        e = atleast(2, m.a, m.b, m.c)
        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        a = m.z[1]
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        b = m.z[2]
        self.assertIs(m.c.get_associated_binary(), m.z[3])
        c = m.z[3]

        self.assertEqual(len(m.z), 4)
        self.assertEqual(len(m.cons), 2)

        assertExpressionsEqual(
            self, m.cons[1].expr, a + b + c >= 2 - 2*(1 - m.z[4]))
        assertExpressionsEqual(
            self, m.cons[2].expr, a + b + c <= 1 + 2*m.z[4])

        self.assertTrue(m.z[4].fixed)
        self.assertEqual(value(m.z[4]), 1)

    def test_no_need_to_walk(self):
        m = self.make_model()
        e = m.a

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertEqual(len(m.z), 1)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.cons), 0)
        self.assertTrue(m.z[1].fixed)
        self.assertEqual(value(m.z[1]), 1)

    def test_integer_var_in_at_least(self):
        m = self.make_model()
        m.x = Var(bounds=(0, 10), domain=Integers)
        e = atleast(m.x, m.a, m.b, m.c)

        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        a = m.z[1]
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        b = m.z[2]
        self.assertIs(m.c.get_associated_binary(), m.z[3])
        c = m.z[3]

        self.assertEqual(len(m.z), 4)
        self.assertEqual(len(m.cons), 2)

        # TODO: This is not linear. This is not what should happen. What
        # *should* happen?
        assertExpressionsEqual(
            self, m.cons[1].expr, a + b + c >= m.x - m.x*(1 - m.z[4]))
        assertExpressionsEqual(
            self, m.cons[2].expr, a + b + c <= m.x - 1 + (4 - m.x)*m.z[4])

        self.assertTrue(m.z[4].fixed)
        self.assertEqual(value(m.z[4]), 1)

    def test_named_expression_in_at_most(self):
        m = self.make_model()
        m.x = Var([1, 2], domain=Integers, bounds=(0, 5))
        m.e = Expression(expr=m.x[1]**2)

        # TODO: If we allow this, we need to let for all sort of algebriac
        # expressions in beforechild
        e = atmost(m.e + m.x[2], m.a, m.b)
        visitor = LogicalToLinearVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        # self.assertIs(m.a.get_associated_binary(), m.z[1])
        # a = m.z[1]
        # self.assertIs(m.b.get_associated_binary(), m.z[2])
        # b = m.z[2]

        # self.assertEqual(len(m.z), 3)
        # self.assertEqual(len(m.cons), 2)

        # assertExpressionsEqual(
        #     self, m.cons[1].expr, a + b <= m.e + m.x[2] - (2 + m.e + m.x[2])*

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

from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
    LogicalToDisjunctiveVisitor
)
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
    atmost, atleast, BooleanVar, Binary, ConcreteModel, Expression, Integers,
    land, lnot, lor, value, Var)

class TestLogicalToDisjunctiveVisitor(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()

        m.a = BooleanVar()
        m.b = BooleanVar()
        m.c = BooleanVar()

        return m

    def test_logical_or(self):
        m = self.make_model()
        e = lor(m.a, m.b, m.c)

        visitor = LogicalToDisjunctiveVisitor()
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

        visitor = LogicalToDisjunctiveVisitor()
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

        visitor = LogicalToDisjunctiveVisitor()
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

        visitor = LogicalToDisjunctiveVisitor()
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

        visitor = LogicalToDisjunctiveVisitor()
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

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])

        self.assertEqual(len(m.z), 2)
        self.assertEqual(len(m.cons), 0)
        self.assertEqual(len(m.disjuncts), 2)
        self.assertEqual(len(m.disjunctions), 1)

        assertExpressionsEqual(
            self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[2] == 1)
        assertExpressionsEqual(
            self,
            m.disjuncts[1].disjunction.disjuncts[0].constraint[1].expr,
            m.z[1] + m.z[2] <= 0)
        assertExpressionsEqual(
            self,
            m.disjuncts[1].disjunction.disjuncts[1].constraint[1].expr,
            m.z[1] + m.z[2] >= 2)

        self.assertTrue(m.disjuncts[0].binary_indicator_var.fixed)
        self.assertEqual(value(m.disjuncts[0].binary_indicator_var), 1)

    def test_at_most(self):
        m = self.make_model()
        e = atmost(2, m.a, (m.a.land(m.b)), m.c)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        a = m.z[1]
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        b = m.z[2]
        self.assertIs(m.c.get_associated_binary(), m.z[4])
        c = m.z[4]

        self.assertEqual(len(m.z), 4)
        self.assertEqual(len(m.cons), 2)
        self.assertEqual(len(m.disjuncts), 2)
        self.assertEqual(len(m.disjunctions), 1)

        # z3 = a ^ b
        assertExpressionsEqual(
            self, m.cons[1].expr, m.z[3] <= a)
        assertExpressionsEqual(
            self, m.cons[2].expr, m.z[3] <= b)

        # atmost in disjunctive form
        assertExpressionsEqual(
            self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[3] + m.z[4] <= 2)
        assertExpressionsEqual(
            self,
            m.disjuncts[1].constraint.expr,
            m.z[1] + m.z[3] + m.z[4] >= 3)

        self.assertTrue(m.disjuncts[0].binary_indicator_var.fixed)
        self.assertEqual(value(m.disjuncts[0].binary_indicator_var), 1)

    def test_at_least(self):
        m = self.make_model()
        e = atleast(2, m.a, m.b, m.c)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        a = m.z[1]
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        b = m.z[2]
        self.assertIs(m.c.get_associated_binary(), m.z[3])
        c = m.z[3]

        self.assertEqual(len(m.z), 3)
        self.assertEqual(len(m.cons), 0)

        # atleast in disjunctive form
        assertExpressionsEqual(
            self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[2] + m.z[3] >= 2)
        assertExpressionsEqual(
            self,
            m.disjuncts[1].constraint.expr,
            m.z[1] + m.z[2] + m.z[3] <= 1)

        self.assertTrue(m.disjuncts[0].binary_indicator_var.fixed)
        self.assertEqual(value(m.disjuncts[0].binary_indicator_var), 1)

    def test_no_need_to_walk(self):
        m = self.make_model()
        e = m.a

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertEqual(len(m.z), 1)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.cons), 0)
        self.assertTrue(m.z[1].fixed)
        self.assertEqual(value(m.z[1]), 1)

    def test_binary_already_associated(self):
        m = self.make_model()
        m.mine = Var(domain=Binary)
        m.a.associate_binary_var(m.mine)

        e = m.a.land(m.b)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)

        self.assertEqual(len(m.z), 2)
        self.assertIs(m.b.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.cons), 2)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[2] <= m.mine)
        assertExpressionsEqual(self, m.cons[2].expr, m.z[2] <= m.z[1])
        self.assertTrue(m.z[2].fixed)
        self.assertEqual(value(m.z[2]), 1)

    # [ESJ 11/22]: We'll probably eventually support all of these examples, but
    # for now test that we handle them gracefully:
    def test_integer_var_in_at_least(self):
        m = self.make_model()
        m.x = Var(bounds=(0, 10), domain=Integers)
        e = atleast(m.x, m.a, m.b, m.c)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        with self.assertRaisesRegex(
                MouseTrap,
                r"The first argument 'x' to "
                r"'atleast\(x: \[a, b, c\]\)' is potentially variable. "
                r"This may be a mathematically coherent expression; However "
                r"it is not yet supported to convert it to a disjunctive "
                r"program."):
            visitor.walk_expression(e)

    def test_numeric_expression_in_at_most(self):
        m = self.make_model()
        m.x = Var([1, 2], bounds=(0, 10), domain=Integers)
        m.y = Var(domain=Integers)
        m.e = Expression(expr=m.x[1]*m.x[2])
        e = atmost(m.e + m.y, m.a, m.b, m.c)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        with self.assertRaisesRegex(
                MouseTrap,
                r"The first argument '\(x\[1\]\*x\[2\]\) \+ y' to "
                r"'atmost\(\(x\[1\]\*x\[2\]\) \+ y: \[a, b, c\]\)' is "
                r"potentially variable. "
                r"This may be a mathematically coherent expression; However "
                r"it is not yet supported to convert it to a disjunctive "
                r"program"):
            visitor.walk_expression(e)

    def test_named_expression_in_at_most(self):
        m = self.make_model()
        m.x = Var([1, 2], bounds=(0, 10), domain=Integers)
        m.y = Var(domain=Integers)
        m.e = Expression(expr=m.x[1]*m.x[2])
        e = atmost(m.e, m.a, m.b, m.c)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        with self.assertRaisesRegex(
                MouseTrap,
                r"The first argument 'x\[1\]\*x\[2\]' to "
                r"'atmost\(\(x\[1\]\*x\[2\]\): \[a, b, c\]\)' is "
                r"potentially variable. "
                r"This may be a mathematically coherent expression; However "
                r"it is not yet supported to convert it to a disjunctive "
                r"program"):
            visitor.walk_expression(e)

    def test_relational_expr_as_boolean_atom(self):
        m = self.make_model()
        m.x = Var()
        e = m.a.land(m.x >= 3)
        visitor = LogicalToDisjunctiveVisitor()

        with self.assertRaisesRegex(
                MouseTrap,
                "The RelationalExpression '3  <=  x' was used as a Boolean "
                "term in a logical proposition. This is not yet supported "
                "when transforming to disjunctive form."):
            visitor.walk_expression(e)


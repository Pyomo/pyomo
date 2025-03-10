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

from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
    LogicalToDisjunctive,
)
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
    LogicalToDisjunctiveVisitor,
)
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
    update_boolean_vars_from_binary,
)
from pyomo.gdp import Disjunct
from pyomo.environ import (
    atmost,
    atleast,
    exactly,
    Block,
    BooleanVar,
    Binary,
    ConcreteModel,
    Constraint,
    Expression,
    Integers,
    land,
    lnot,
    lor,
    LogicalConstraint,
    Objective,
    Param,
    SolverFactory,
    value,
    Var,
    TransformationFactory,
)

gurobi_available = (
    SolverFactory('gurobi').available(exception_flag=False)
    and SolverFactory('gurobi').license_is_valid()
)


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

        self.assertEqual(len(m.cons), 5)
        self.assertEqual(len(m.z), 4)
        # !z4 v a v b v c
        assertExpressionsEqual(
            self, m.cons[1].expr, 1 - m.z[4] + m.z[1] + m.z[2] + m.z[3] >= 1
        )
        # z4 v !a
        assertExpressionsEqual(self, m.cons[2].expr, m.z[4] + (1 - m.z[1]) >= 1)
        # z4 v !b
        assertExpressionsEqual(self, m.cons[3].expr, m.z[4] + (1 - m.z[2]) >= 1)
        # z4 v !c
        assertExpressionsEqual(self, m.cons[4].expr, m.z[4] + (1 - m.z[3]) >= 1)

        # z4 is constrained to be 'True'
        assertExpressionsEqual(self, m.cons[5].expr, m.z[4] >= 1)

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

        self.assertEqual(len(m.cons), 5)
        self.assertEqual(len(m.z), 4)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[4] <= m.z[1])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[4] <= m.z[2])
        assertExpressionsEqual(self, m.cons[3].expr, m.z[4] <= m.z[3])
        assertExpressionsEqual(
            self, m.cons[4].expr, 1 - m.z[4] <= 3 - (m.z[1] + m.z[2] + m.z[3])
        )
        assertExpressionsEqual(self, m.cons[5].expr, m.z[4] >= 1)

    def test_logical_not(self):
        m = self.make_model()
        e = lnot(m.a)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.cons), 2)
        self.assertEqual(len(m.z), 2)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[2] == 1 - m.z[1])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[2] >= 1)

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

        self.assertEqual(len(m.cons), 7)
        # z4 = b ^ c
        assertExpressionsEqual(self, m.cons[1].expr, m.z[4] <= m.z[2])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[4] <= m.z[3])
        assertExpressionsEqual(
            self, m.cons[3].expr, 1 - m.z[4] <= 2 - (m.z[2] + m.z[3])
        )
        # z5 = a -> z4
        # which means z5 = !a v z4
        assertExpressionsEqual(
            self, m.cons[4].expr, (1 - m.z[5]) + (1 - m.z[1]) + m.z[4] >= 1
        )
        # z5 >= 1 - z1
        assertExpressionsEqual(self, m.cons[5].expr, m.z[5] + (1 - (1 - m.z[1])) >= 1)
        # z5 >= z4
        assertExpressionsEqual(self, m.cons[6].expr, m.z[5] + (1 - m.z[4]) >= 1)

        # z5 is constrained to be 'True'
        assertExpressionsEqual(self, m.cons[7].expr, m.z[5] >= 1)

    def test_equivalence(self):
        m = self.make_model()
        e = m.a.equivalent_to(m.c)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.c.get_associated_binary(), m.z[2])
        self.assertEqual(len(m.z), 5)
        self.assertEqual(len(m.cons), 10)

        # z[3] == !a v c
        assertExpressionsEqual(
            self, m.cons[1].expr, (1 - m.z[3]) + (1 - m.z[1]) + m.z[2] >= 1
        )
        assertExpressionsEqual(self, m.cons[2].expr, 1 - (1 - m.z[1]) + m.z[3] >= 1)
        assertExpressionsEqual(self, m.cons[3].expr, m.z[3] + (1 - m.z[2]) >= 1)

        # z[4] == a v ! c
        assertExpressionsEqual(
            self, m.cons[4].expr, (1 - m.z[4]) + (1 - m.z[2]) + m.z[1] >= 1
        )
        assertExpressionsEqual(self, m.cons[5].expr, m.z[4] + (1 - m.z[1]) >= 1)
        assertExpressionsEqual(self, m.cons[6].expr, 1 - (1 - m.z[2]) + m.z[4] >= 1)

        # z[5] == z[3] ^ z[4]
        assertExpressionsEqual(self, m.cons[7].expr, m.z[5] <= m.z[3])
        assertExpressionsEqual(self, m.cons[8].expr, m.z[5] <= m.z[4])
        assertExpressionsEqual(
            self, m.cons[9].expr, 1 - m.z[5] <= 2 - (m.z[3] + m.z[4])
        )

        assertExpressionsEqual(self, m.cons[10].expr, m.z[5] >= 1)

    def test_equivalent_to_True(self):
        m = self.make_model()
        e = m.a.equivalent_to(True)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)

        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.z), 4)
        self.assertEqual(len(m.cons), 10)

        # z[2] == !a v True
        assertExpressionsEqual(
            self, m.cons[1].expr, (1 - m.z[2]) + (1 - m.z[1]) + 1 >= 1
        )
        assertExpressionsEqual(self, m.cons[2].expr, 1 - (1 - m.z[1]) + m.z[2] >= 1)
        assertExpressionsEqual(self, m.cons[3].expr, m.z[2] + (1 - 1) >= 1)

        # z[3] == a v ! c
        assertExpressionsEqual(self, m.cons[4].expr, (1 - m.z[3]) + m.z[1] >= 1)
        assertExpressionsEqual(self, m.cons[5].expr, m.z[3] + (1 - m.z[1]) >= 1)
        assertExpressionsEqual(self, m.cons[6].expr, m.z[3] + 1 >= 1)

        # z[4] == z[2] ^ z[3]
        assertExpressionsEqual(self, m.cons[7].expr, m.z[4] <= m.z[2])
        assertExpressionsEqual(self, m.cons[8].expr, m.z[4] <= m.z[3])
        assertExpressionsEqual(
            self, m.cons[9].expr, 1 - m.z[4] <= 2 - (m.z[2] + m.z[3])
        )

        assertExpressionsEqual(self, m.cons[10].expr, m.z[4] >= 1)

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
        self.assertEqual(len(m.cons), 1)
        self.assertEqual(len(m.disjuncts), 2)
        self.assertEqual(len(m.disjunctions), 1)

        assertExpressionsEqual(
            self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[2] == 1
        )
        assertExpressionsEqual(
            self,
            m.disjuncts[1].disjunction.disjuncts[0].constraint[1].expr,
            m.z[1] + m.z[2] <= 0,
        )
        assertExpressionsEqual(
            self,
            m.disjuncts[1].disjunction.disjuncts[1].constraint[1].expr,
            m.z[1] + m.z[2] >= 2,
        )

        assertExpressionsEqual(
            self, m.cons[1].expr, m.disjuncts[0].binary_indicator_var >= 1
        )

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
        self.assertEqual(len(m.cons), 4)
        self.assertEqual(len(m.disjuncts), 2)
        self.assertEqual(len(m.disjunctions), 1)

        # z3 = a ^ b
        assertExpressionsEqual(self, m.cons[1].expr, m.z[3] <= a)
        assertExpressionsEqual(self, m.cons[2].expr, m.z[3] <= b)
        assertExpressionsEqual(self, m.cons[3].expr, 1 - m.z[3] <= 2 - sum([a, b]))

        # atmost in disjunctive form
        assertExpressionsEqual(
            self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[3] + m.z[4] <= 2
        )
        assertExpressionsEqual(
            self, m.disjuncts[1].constraint.expr, m.z[1] + m.z[3] + m.z[4] >= 3
        )
        assertExpressionsEqual(
            self, m.cons[4].expr, m.disjuncts[0].binary_indicator_var >= 1
        )

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
        self.assertEqual(len(m.cons), 1)

        # atleast in disjunctive form
        assertExpressionsEqual(
            self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[2] + m.z[3] >= 2
        )
        assertExpressionsEqual(
            self, m.disjuncts[1].constraint.expr, m.z[1] + m.z[2] + m.z[3] <= 1
        )

        assertExpressionsEqual(
            self, m.cons[1].expr, m.disjuncts[0].binary_indicator_var >= 1
        )

    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_logical_integration(self):
        """
        This is kind of a ridiculous test, but I bothered type it and it has
        a lot of logical things together, so adding it.
        """
        m = self.make_model()
        m.d = BooleanVar()
        m.t = BooleanVar()
        e = m.t.equivalent_to(lnot(lor(m.a, m.b)).land(exactly(1, [m.c, m.d])))

        # We're forcing t to be True.
        m.c.fix(True)
        m.d.fix(False)
        m.a.fix(False)
        m.b.fix(False)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions

        visitor.walk_expression(e)

        self.assertEqual(len(m.z), 11)
        self.assertIs(m.a.get_associated_binary(), m.z[2])
        a = m.z[2]
        self.assertIs(m.b.get_associated_binary(), m.z[3])
        b = m.z[3]
        # apologies to the universe for this one, but because my own notation
        # is awful:
        z3 = m.z[4]
        z4 = m.z[5]
        z5 = m.z[8]
        self.assertIs(m.t.get_associated_binary(), m.z[1])
        t = m.z[1]
        z6 = m.z[9]
        z7 = m.z[10]
        self.assertIs(m.c.get_associated_binary(), m.z[6])
        c = m.z[6]
        self.assertIs(m.d.get_associated_binary(), m.z[7])
        d = m.z[7]
        z8 = m.z[11]

        self.assertEqual(len(m.disjuncts), 2)
        self.assertEqual(
            len(
                list(
                    m.disjuncts[0].component_data_objects(
                        Constraint, descend_into=False
                    )
                )
            ),
            1,
        )
        assertExpressionsEqual(self, m.disjuncts[0].constraint.expr, c + d == 1)
        # not nested
        self.assertEqual(
            len(
                list(
                    m.disjuncts[0].component_data_objects(Disjunct, descend_into=False)
                )
            ),
            0,
        )

        # nested
        self.assertEqual(
            len(
                list(
                    m.disjuncts[1].component_data_objects(
                        Constraint, descend_into=False
                    )
                )
            ),
            0,
        )
        self.assertEqual(
            len(
                list(
                    m.disjuncts[1].component_data_objects(Disjunct, descend_into=False)
                )
            ),
            2,
        )
        self.assertEqual(len(m.disjuncts[1].disjunction.disjuncts), 2)
        assertExpressionsEqual(
            self, m.disjuncts[1].disjunction.disjuncts[0].constraint[1].expr, c + d <= 0
        )
        assertExpressionsEqual(
            self, m.disjuncts[1].disjunction.disjuncts[1].constraint[1].expr, c + d >= 2
        )

        self.assertEqual(len(m.disjunctions), 1)
        self.assertIs(m.disjunctions[0].disjuncts[0], m.disjuncts[0])
        self.assertIs(m.disjunctions[0].disjuncts[1], m.disjuncts[1])

        self.assertEqual(len(m.cons), 17)
        assertExpressionsEqual(self, m.cons[1].expr, (1 - z3) + a + b >= 1)
        assertExpressionsEqual(self, m.cons[2].expr, z3 + (1 - a) >= 1)
        assertExpressionsEqual(self, m.cons[3].expr, z3 + (1 - b) >= 1)
        assertExpressionsEqual(self, m.cons[4].expr, z4 == 1 - z3)
        assertExpressionsEqual(self, m.cons[5].expr, z5 <= z4)
        assertExpressionsEqual(
            self, m.cons[6].expr, z5 <= m.disjuncts[0].binary_indicator_var
        )
        assertExpressionsEqual(
            self,
            m.cons[7].expr,
            1 - z5 <= 2 - (z4 + m.disjuncts[0].binary_indicator_var),
        )
        assertExpressionsEqual(self, m.cons[8].expr, (1 - z6) + (1 - t) + z5 >= 1)
        assertExpressionsEqual(self, m.cons[9].expr, 1 - (1 - t) + z6 >= 1)
        assertExpressionsEqual(self, m.cons[10].expr, z6 + (1 - z5) >= 1)
        assertExpressionsEqual(self, m.cons[11].expr, (1 - z7) + (1 - z5) + t >= 1)
        assertExpressionsEqual(self, m.cons[12].expr, z7 + (1 - t) >= 1)
        assertExpressionsEqual(self, m.cons[13].expr, 1 - (1 - z5) + z7 >= 1)
        assertExpressionsEqual(self, m.cons[14].expr, z8 <= z6)
        assertExpressionsEqual(self, m.cons[15].expr, z8 <= z7)
        assertExpressionsEqual(self, m.cons[16].expr, 1 - z8 <= 2 - (z6 + z7))
        assertExpressionsEqual(self, m.cons[17].expr, z8 >= 1)

        TransformationFactory('gdp.bigm').apply_to(m)
        m.obj = Objective(expr=m.t.get_associated_binary())
        SolverFactory('gurobi').solve(m, tee=True)
        update_boolean_vars_from_binary(m)

        self.assertTrue(value(e))
        self.assertEqual(value(m.obj), 1)
        self.assertTrue(value(m.t))

    def test_boolean_fixed_true(self):
        m = self.make_model()
        e = m.a.implies(m.b)
        m.a.fix(True)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions

        visitor.walk_expression(e)
        # we'll get !a v b
        self.assertEqual(len(m.z), 3)
        self.assertEqual(len(m.cons), 4)

        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertTrue(m.z[1].fixed)
        self.assertEqual(value(m.z[1]), 1)
        self.assertIs(m.b.get_associated_binary(), m.z[2])

        assertExpressionsEqual(
            self, m.cons[1].expr, (1 - m.z[3]) + (1 - m.z[1]) + m.z[2] >= 1
        )
        assertExpressionsEqual(self, m.cons[2].expr, 1 - (1 - m.z[1]) + m.z[3] >= 1)
        assertExpressionsEqual(self, m.cons[3].expr, m.z[3] + (1 - m.z[2]) >= 1)
        assertExpressionsEqual(self, m.cons[4].expr, m.z[3] >= 1)

    def test_boolean_fixed_false(self):
        m = self.make_model()
        e = m.a & m.b
        m.a.fix(False)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions

        visitor.walk_expression(e)
        # we'll get !a v b
        self.assertEqual(len(m.z), 3)
        self.assertEqual(len(m.cons), 4)

        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertTrue(m.z[1].fixed)
        self.assertEqual(value(m.z[1]), 0)
        self.assertIs(m.b.get_associated_binary(), m.z[2])

        assertExpressionsEqual(self, m.cons[1].expr, m.z[1] >= m.z[3])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[2] >= m.z[3])
        assertExpressionsEqual(
            self, m.cons[3].expr, 1 - m.z[3] <= 2 - (m.z[1] + m.z[2])
        )
        assertExpressionsEqual(self, m.cons[4].expr, m.z[3] >= 1)

    def test_boolean_fixed_none(self):
        m = self.make_model()
        e = m.a & m.b
        # I don't get what this means, but you can do it, so... I guess we need
        # to handle it.
        m.a.fix(None)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions

        visitor.walk_expression(e)
        # we'll get !a v b
        self.assertEqual(len(m.z), 3)
        self.assertEqual(len(m.cons), 4)

        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertTrue(m.z[1].fixed)
        self.assertIsNone(m.z[1].value)
        self.assertIs(m.b.get_associated_binary(), m.z[2])

        assertExpressionsEqual(self, m.cons[1].expr, m.z[1] >= m.z[3])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[2] >= m.z[3])
        assertExpressionsEqual(
            self, m.cons[3].expr, 1 - m.z[3] <= 2 - (m.z[1] + m.z[2])
        )
        assertExpressionsEqual(self, m.cons[4].expr, m.z[3] >= 1)

    def test_no_need_to_walk(self):
        m = self.make_model()
        e = m.a

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        visitor.walk_expression(e)
        self.assertEqual(len(m.z), 1)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.cons), 1)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[1] >= 1)

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
        self.assertEqual(len(m.cons), 4)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[2] <= m.mine)
        assertExpressionsEqual(self, m.cons[2].expr, m.z[2] <= m.z[1])
        assertExpressionsEqual(
            self, m.cons[3].expr, 1 - m.z[2] <= 2 - (m.mine + m.z[1])
        )
        assertExpressionsEqual(self, m.cons[4].expr, m.z[2] >= 1)

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
            r"program.",
            normalize_whitespace=True,
        ):
            visitor.walk_expression(e)

    def test_numeric_expression_in_at_most(self):
        m = self.make_model()
        m.x = Var([1, 2], bounds=(0, 10), domain=Integers)
        m.y = Var(domain=Integers)
        m.e = Expression(expr=m.x[1] * m.x[2])
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
            r"program",
            normalize_whitespace=True,
        ):
            visitor.walk_expression(e)

    def test_named_expression_in_at_most(self):
        m = self.make_model()
        m.x = Var([1, 2], bounds=(0, 10), domain=Integers)
        m.y = Var(domain=Integers)
        m.e = Expression(expr=m.x[1] * m.x[2])
        e = atmost(m.e, m.a, m.b, m.c)

        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars

        with self.assertRaisesRegex(
            MouseTrap,
            r"The first argument 'e' to "
            r"'atmost\(\(x\[1\]\*x\[2\]\): \[a, b, c\]\)' is "
            r"potentially variable. "
            r"This may be a mathematically coherent expression; However "
            r"it is not yet supported to convert it to a disjunctive "
            r"program",
            normalize_whitespace=True,
        ):
            visitor.walk_expression(e)

    def test_relational_expr_as_boolean_atom(self):
        m = self.make_model()
        m.x = Var()
        e = m.a.land(m.x >= 3)
        visitor = LogicalToDisjunctiveVisitor()

        with self.assertRaisesRegex(
            MouseTrap,
            "The RelationalExpression '3 <= x' was used as a Boolean "
            "term in a logical proposition. This is not yet supported "
            "when transforming to disjunctive form.",
            normalize_whitespace=True,
        ):
            visitor.walk_expression(e)


class TestLogicalToDisjunctiveTransformation(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()

        m.a = BooleanVar()
        m.b = BooleanVar([1, 2])
        m.p = Param(initialize=1)
        m.p2 = Param([1, 2], mutable=True)
        m.p2[1] = 1
        m.p2[2] = 2

        m.block = Block()
        m.block.c1 = LogicalConstraint(expr=m.a.land(m.b[1]))
        m.block.c2 = LogicalConstraint(
            expr=exactly(m.p2[2], m.a, m.b[1], m.b[2].lor(m.b[1]))
        )

        m.c1 = LogicalConstraint(expr=atmost(m.p + m.p2[1], m.a, m.b[1], m.b[2]))

        return m

    def check_and_constraints(self, a, b1, z, transBlock):
        assertExpressionsEqual(self, transBlock.transformed_constraints[1].expr, z <= a)
        assertExpressionsEqual(
            self, transBlock.transformed_constraints[2].expr, z <= b1
        )
        assertExpressionsEqual(
            self, transBlock.transformed_constraints[3].expr, 1 - z <= 2 - (a + b1)
        )
        assertExpressionsEqual(self, transBlock.transformed_constraints[4].expr, z >= 1)

    def check_block_c1_transformed(self, m, transBlock):
        self.assertFalse(m.block.c1.active)
        self.assertIs(m.a.get_associated_binary(), transBlock.auxiliary_vars[1])
        self.assertIs(m.b[1].get_associated_binary(), transBlock.auxiliary_vars[2])
        self.check_and_constraints(
            transBlock.auxiliary_vars[1],
            transBlock.auxiliary_vars[2],
            transBlock.auxiliary_vars[3],
            transBlock,
        )

    def check_block_exactly(self, a, b1, b2, z4, transBlock):
        m = transBlock.model()

        # z[4] = b[2] v b[1]
        assertExpressionsEqual(
            self, transBlock.transformed_constraints[5].expr, (1 - z4) + b2 + b1 >= 1
        )
        assertExpressionsEqual(
            self, transBlock.transformed_constraints[6].expr, z4 + (1 - b2) >= 1
        )
        assertExpressionsEqual(
            self, transBlock.transformed_constraints[7].expr, z4 + (1 - b1) >= 1
        )

        # exactly in disjunctive form
        assertExpressionsEqual(
            self,
            transBlock.auxiliary_disjuncts[0].constraint.expr,
            a + b1 + z4 == m.p2[2],
        )
        assertExpressionsEqual(
            self,
            transBlock.auxiliary_disjuncts[1]
            .disjunction.disjuncts[0]
            .constraint[1]
            .expr,
            a + b1 + z4 <= m.p2[2] - 1,
        )
        assertExpressionsEqual(
            self,
            transBlock.auxiliary_disjuncts[1]
            .disjunction.disjuncts[1]
            .constraint[1]
            .expr,
            a + b1 + z4 >= m.p2[2] + 1,
        )

        assertExpressionsEqual(
            self,
            transBlock.transformed_constraints[8].expr,
            transBlock.auxiliary_disjuncts[0].binary_indicator_var >= 1,
        )

    def check_block_transformed(self, m):
        self.assertFalse(m.block.c2.active)
        transBlock = m.block._logical_to_disjunctive
        self.assertEqual(len(transBlock.auxiliary_vars), 5)
        self.assertEqual(len(transBlock.transformed_constraints), 8)
        self.assertEqual(len(transBlock.auxiliary_disjuncts), 2)
        self.assertEqual(len(transBlock.auxiliary_disjunctions), 1)

        self.check_block_c1_transformed(m, transBlock)

        self.assertIs(m.b[2].get_associated_binary(), transBlock.auxiliary_vars[4])

        z4 = transBlock.auxiliary_vars[5]
        a = transBlock.auxiliary_vars[1]
        b1 = transBlock.auxiliary_vars[2]
        b2 = transBlock.auxiliary_vars[4]
        self.check_block_exactly(a, b1, b2, z4, transBlock)

    def test_constraint_target(self):
        m = self.make_model()
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(
            m, targets=[m.block.c1]
        )

        transBlock = m.block._logical_to_disjunctive
        self.assertEqual(len(transBlock.auxiliary_vars), 3)
        self.assertEqual(len(transBlock.transformed_constraints), 4)
        self.assertEqual(len(transBlock.auxiliary_disjuncts), 0)
        self.assertEqual(len(transBlock.auxiliary_disjunctions), 0)
        self.check_block_c1_transformed(m, transBlock)
        self.assertTrue(m.block.c2.active)
        self.assertTrue(m.c1.active)

    def test_block_target(self):
        m = self.make_model()
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(
            m, targets=[m.block]
        )

        self.check_block_transformed(m)
        self.assertTrue(m.c1.active)

    def test_transform_block(self):
        m = self.make_model()
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(m.block)

        self.check_block_transformed(m)
        self.assertTrue(m.c1.active)

    def test_transform_model(self):
        m = self.make_model()
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(m)

        # c1 got transformed first
        self.assertFalse(m.c1.active)
        transBlock = m._logical_to_disjunctive
        self.assertEqual(len(transBlock.auxiliary_vars), 3)
        self.assertEqual(len(transBlock.transformed_constraints), 1)
        self.assertEqual(len(transBlock.auxiliary_disjuncts), 2)
        self.assertEqual(len(transBlock.auxiliary_disjunctions), 1)

        a = m._logical_to_disjunctive.auxiliary_vars[1]
        b1 = m._logical_to_disjunctive.auxiliary_vars[2]
        b2 = m._logical_to_disjunctive.auxiliary_vars[3]

        # atmost in disjunctive form
        assertExpressionsEqual(
            self,
            transBlock.auxiliary_disjuncts[0].constraint.expr,
            a + b1 + b2 <= 1 + m.p2[1],
        )
        assertExpressionsEqual(
            self,
            transBlock.auxiliary_disjuncts[1].constraint.expr,
            a + b1 + b2 >= 1 + m.p2[1] + 1,
        )

        assertExpressionsEqual(
            self,
            transBlock.transformed_constraints[1].expr,
            transBlock.auxiliary_disjuncts[0].binary_indicator_var >= 1,
        )

        # and everything on the block is transformed too
        transBlock = m.block._logical_to_disjunctive
        self.assertEqual(len(transBlock.auxiliary_vars), 2)
        self.assertEqual(len(transBlock.transformed_constraints), 8)
        self.assertEqual(len(transBlock.auxiliary_disjuncts), 2)
        self.assertEqual(len(transBlock.auxiliary_disjunctions), 1)
        self.check_and_constraints(a, b1, transBlock.auxiliary_vars[1], transBlock)
        self.check_block_exactly(a, b1, b2, transBlock.auxiliary_vars[2], transBlock)

    @unittest.skipUnless(gurobi_available, "Gurobi is not available")
    def test_reverse_implication_for_land(self):
        m = ConcreteModel()

        m.t = BooleanVar()
        m.a = BooleanVar()
        m.d = BooleanVar()

        m.c = LogicalConstraint(expr=m.t.equivalent_to(m.a.land(m.d)))

        m.a.fix(True)
        m.d.fix(True)

        m.binary = Var(domain=Binary)
        m.t.associate_binary_var(m.binary)

        m.obj = Objective(expr=m.binary)

        TransformationFactory('contrib.logical_to_disjunctive').apply_to(m)
        TransformationFactory('gdp.bigm').apply_to(m)

        SolverFactory('gurobi').solve(m)

        update_boolean_vars_from_binary(m)
        # Should be 1 (we forced it)
        self.assertEqual(value(m.obj), 1)
        self.assertTrue(value(m.t))

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

from io import StringIO

import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR

from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, AutoLinkedBinaryVar


class TestDisjunction(unittest.TestCase):
    def test_empty_disjunction(self):
        m = ConcreteModel()
        m.d = Disjunct()
        m.e = Disjunct()

        m.x1 = Disjunction()
        self.assertEqual(len(m.x1), 0)

        m.x1 = [m.d, m.e]
        self.assertEqual(len(m.x1), 1)
        self.assertEqual(m.x1.disjuncts, [m.d, m.e])

        m.x2 = Disjunction([1, 2, 3, 4])
        self.assertEqual(len(m.x2), 0)

        m.x2[2] = [m.d, m.e]
        self.assertEqual(len(m.x2), 1)
        self.assertEqual(m.x2[2].disjuncts, [m.d, m.e])

    def test_construct_implicit_disjuncts(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.d = Disjunction(expr=[m.x <= 0, m.y >= 1])
        self.assertEqual(len(m.component_map(Disjunction)), 1)
        self.assertEqual(len(m.component_map(Disjunct)), 1)

        implicit_disjuncts = list(m.component_map(Disjunct).keys())
        self.assertEqual(implicit_disjuncts[0][:2], "d_")
        disjuncts = m.d.disjuncts
        self.assertEqual(len(disjuncts), 2)
        self.assertIs(disjuncts[0].parent_block(), m)
        self.assertIs(disjuncts[0].constraint[1].body, m.x)
        self.assertIs(disjuncts[1].parent_block(), m)
        self.assertIs(disjuncts[1].constraint[1].body, m.y)

        # Test that the implicit disjuncts get a unique name
        m.add_component('e_disjuncts', Var())
        m.e = Disjunction(expr=[m.y <= 0, m.x >= 1])
        self.assertEqual(len(m.component_map(Disjunction)), 2)
        self.assertEqual(len(m.component_map(Disjunct)), 2)
        implicit_disjuncts = list(m.component_map(Disjunct).keys())
        self.assertEqual(implicit_disjuncts[1][:12], "e_disjuncts_")
        disjuncts = m.e.disjuncts
        self.assertEqual(len(disjuncts), 2)
        self.assertIs(disjuncts[0].parent_block(), m)
        self.assertIs(disjuncts[0].constraint[1].body, m.y)
        self.assertIs(disjuncts[1].parent_block(), m)
        self.assertIs(disjuncts[1].constraint[1].body, m.x)
        self.assertEqual(len(disjuncts[0].parent_component().name), 13)
        self.assertEqual(disjuncts[0].name[:12], "e_disjuncts_")

        # Test that the implicit disjuncts can be lists/tuples/generators
        def _gen():
            yield m.y <= 4
            yield m.x >= 5

        m.f = Disjunction(expr=[[m.y <= 0, m.x >= 1], (m.y <= 2, m.x >= 3), _gen()])
        self.assertEqual(len(m.component_map(Disjunction)), 3)
        self.assertEqual(len(m.component_map(Disjunct)), 3)
        implicit_disjuncts = list(m.component_map(Disjunct).keys())
        self.assertEqual(implicit_disjuncts[2][:12], "f_disjuncts")
        disjuncts = m.f.disjuncts
        self.assertEqual(len(disjuncts), 3)
        self.assertIs(disjuncts[0].parent_block(), m)
        self.assertIs(disjuncts[0].constraint[1].body, m.y)
        self.assertEqual(disjuncts[0].constraint[1].upper, 0)
        self.assertIs(disjuncts[0].constraint[2].body, m.x)
        self.assertEqual(disjuncts[0].constraint[2].lower, 1)

        self.assertIs(disjuncts[1].parent_block(), m)
        self.assertIs(disjuncts[1].constraint[1].body, m.y)
        self.assertEqual(disjuncts[1].constraint[1].upper, 2)
        self.assertIs(disjuncts[1].constraint[2].body, m.x)
        self.assertEqual(disjuncts[1].constraint[2].lower, 3)

        self.assertIs(disjuncts[2].parent_block(), m)
        self.assertIs(disjuncts[2].constraint[1].body, m.y)
        self.assertEqual(disjuncts[2].constraint[1].upper, 4)
        self.assertIs(disjuncts[2].constraint[2].body, m.x)
        self.assertEqual(disjuncts[2].constraint[2].lower, 5)

        self.assertEqual(len(disjuncts[0].parent_component().name), 11)
        self.assertEqual(disjuncts[0].name, "f_disjuncts[0]")

    def test_construct_invalid_component(self):
        m = ConcreteModel()
        m.d = Disjunct([1, 2])
        with self.assertRaisesRegex(
            ValueError,
            "Unexpected term for Disjunction 'dd'.\n    "
            "Expected a Disjunct object, relational or logical "
            "expression, or\n    iterable of relational/logical "
            "expressions but got 'IndexedDisjunct'",
        ):
            m.dd = Disjunction(expr=[m.d])
        with self.assertRaisesRegex(
            ValueError,
            "Unexpected term for Disjunction 'ee'.\n    "
            "Expected a Disjunct object, relational or logical "
            "expression, or\n    iterable of relational/logical "
            "expressions but got 'str' in 'list'",
        ):
            m.ee = Disjunction(expr=[['a']])
        with self.assertRaisesRegex(
            ValueError,
            "Unexpected term for Disjunction 'ff'.\n    "
            "Expected a Disjunct object, relational or logical "
            "expression, or\n    iterable of relational/logical "
            "expressions but got 'str'",
        ):
            m.ff = Disjunction(expr=['a'])


class TestDisjunct(unittest.TestCase):
    def test_deactivate(self):
        m = ConcreteModel()
        m.x = Var()
        m.d1 = Disjunct()
        m.d1.constraint = Constraint(expr=m.x <= 0)
        m.d = Disjunction(expr=[m.d1, m.x >= 1, m.x >= 5])
        d2 = m.d.disjuncts[1].parent_component()
        self.assertEqual(len(m.component_map(Disjunction)), 1)
        self.assertEqual(len(m.component_map(Disjunct)), 2)
        self.assertIsNot(m.d1, d2)

        self.assertTrue(m.d1.active)
        self.assertTrue(d2.active)
        self.assertTrue(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[0].deactivate()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[1].deactivate()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertFalse(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertTrue(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        d2.deactivate()
        self.assertFalse(m.d1.active)
        self.assertFalse(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertFalse(m.d.disjuncts[1].active)
        self.assertFalse(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertTrue(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertTrue(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[2].activate()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertFalse(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertTrue(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        d2.activate()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertTrue(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d1.activate()
        self.assertTrue(m.d1.active)
        self.assertTrue(d2.active)
        self.assertTrue(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

    def test_deactivate_without_fixing_indicator(self):
        m = ConcreteModel()
        m.x = Var()
        m.d1 = Disjunct()
        m.d1.constraint = Constraint(expr=m.x <= 0)
        m.d = Disjunction(expr=[m.d1, m.x >= 1, m.x >= 5])
        d2 = m.d.disjuncts[1].parent_component()
        self.assertEqual(len(m.component_map(Disjunction)), 1)
        self.assertEqual(len(m.component_map(Disjunct)), 2)
        self.assertIsNot(m.d1, d2)

        self.assertTrue(m.d1.active)
        self.assertTrue(d2.active)
        self.assertTrue(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[0]._deactivate_without_fixing_indicator()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertTrue(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

        m.d.disjuncts[1]._deactivate_without_fixing_indicator()
        self.assertFalse(m.d1.active)
        self.assertTrue(d2.active)
        self.assertFalse(m.d.disjuncts[0].active)
        self.assertFalse(m.d.disjuncts[1].active)
        self.assertTrue(m.d.disjuncts[2].active)
        self.assertFalse(m.d.disjuncts[0].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[1].indicator_var.is_fixed())
        self.assertFalse(m.d.disjuncts[2].indicator_var.is_fixed())

    def test_indexed_disjunct_active_property(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 12))

        @m.Disjunct([0, 1, 2])
        def disjunct(d, i):
            m = d.model()
            if i == 0:
                d.cons = Constraint(expr=m.x >= 3)
            elif i == 1:
                d.cons = Constraint(expr=m.x >= 8)
            else:
                d.cons = Constraint(expr=m.x == 12)

        self.assertTrue(m.disjunct.active)
        m.disjunct[1].deactivate()
        self.assertTrue(m.disjunct.active)
        m.disjunct[0].deactivate()
        m.disjunct[2].deactivate()
        self.assertFalse(m.disjunct.active)
        m.disjunct.activate()
        self.assertTrue(m.disjunct.active)
        m.disjunct.deactivate()
        self.assertFalse(m.disjunct.active)
        for i in range(3):
            self.assertFalse(m.disjunct[i].active)

    def test_indexed_disjunction_active_property(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 12))

        @m.Disjunction([0, 1, 2])
        def disjunction(m, i):
            return [m.x == i * 5, m.x == i * 5 + 1]

        self.assertTrue(m.disjunction.active)
        m.disjunction[2].deactivate()
        self.assertTrue(m.disjunction.active)
        m.disjunction[0].deactivate()
        m.disjunction[1].deactivate()
        self.assertFalse(m.disjunction.active)
        m.disjunction.activate()
        self.assertTrue(m.disjunction.active)
        m.disjunction.deactivate()
        self.assertFalse(m.disjunction.active)
        for i in range(3):
            self.assertFalse(m.disjunction[i].active)


class TestAutoVars(unittest.TestCase):
    def test_synchronize_value(self):
        m = ConcreteModel()
        m.iv = AutoLinkedBooleanVar()
        m.biv = AutoLinkedBinaryVar(m.iv)
        m.iv.associate_binary_var(m.biv)

        self.assertIsNone(m.iv.value)
        self.assertIsNone(m.biv.value)

        # Note: test the following twice to exercise the "no update"
        # situation, and to ensure no infinite loops

        m.iv = True
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)
        m.iv = True
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)

        m.iv = False
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)
        m.iv = False
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)

        m.iv = None
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)
        m.iv = None
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)

        m.biv = 1
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)
        m.biv = 1
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)

        eps = AutoLinkedBinaryVar.INTEGER_TOLERANCE / 10

        m.biv = None
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)
        m.biv = None
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)

        m.biv.value = 1 - eps
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1 - eps)
        m.biv.value = 1 - eps
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1 - eps)

        m.biv.value = eps
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, eps)
        m.biv.value = eps
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, eps)

        m.biv.value = 0.5
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, 0.5)
        m.biv.value = 0.5
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, 0.5)

    def test_fix_value(self):
        m = ConcreteModel()
        m.iv = AutoLinkedBooleanVar()
        m.biv = AutoLinkedBinaryVar(m.iv)
        m.iv.associate_binary_var(m.biv)

        m.iv.fix()
        self.assertTrue(m.iv.is_fixed())
        self.assertTrue(m.biv.is_fixed())
        self.assertIsNone(m.iv.value)
        self.assertIsNone(m.biv.value)

        m.iv.fix(True)
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)

        m.iv.fix(False)
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)

        m.iv.fix(None)
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, None)

        m.biv.fix(1)
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)

        with LoggingIntercept() as LOG:
            m.biv.fix(0.5)
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'biv' to a value `0.5` (float) not in domain Binary.",
        )
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, 0.5)

        with LoggingIntercept() as LOG:
            m.biv.fix(0.55, True)
        self.assertEqual(LOG.getvalue().strip(), "")
        self.assertEqual(m.iv.value, None)
        self.assertEqual(m.biv.value, 0.55)

        m.biv.fix(0)
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)

        eps = AutoLinkedBinaryVar.INTEGER_TOLERANCE / 10

        # Note that fixing to a near-True value will toggle the iv
        with LoggingIntercept() as LOG:
            m.biv.fix(1 - eps)
        self.assertEqual(
            LOG.getvalue().strip(),
            "Setting Var 'biv' to a "
            "value `%s` (float) not in domain Binary." % (1 - eps),
        )
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1 - eps)

        with LoggingIntercept() as LOG:
            m.biv.fix(eps, True)
        self.assertEqual(LOG.getvalue().strip(), "")
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, eps)

        m.iv.fix(True)
        self.assertEqual(m.iv.value, True)
        self.assertEqual(m.biv.value, 1)

        m.iv.fix(False)
        self.assertEqual(m.iv.value, False)
        self.assertEqual(m.biv.value, 0)

    def test_fix_unfix(self):
        m = ConcreteModel()
        m.iv = AutoLinkedBooleanVar()
        m.biv = AutoLinkedBinaryVar(m.iv)
        m.iv.associate_binary_var(m.biv)

        m.iv.fix()
        self.assertTrue(m.iv.is_fixed())
        self.assertTrue(m.biv.is_fixed())

        m.iv.unfix()
        self.assertFalse(m.iv.is_fixed())
        self.assertFalse(m.biv.is_fixed())

        m.iv.unfix()
        self.assertFalse(m.iv.is_fixed())
        self.assertFalse(m.biv.is_fixed())

        m.biv.fix()
        self.assertTrue(m.iv.is_fixed())
        self.assertTrue(m.biv.is_fixed())

        m.biv.unfix()
        self.assertFalse(m.iv.is_fixed())
        self.assertFalse(m.biv.is_fixed())

        m.biv.unfix()
        self.assertFalse(m.iv.is_fixed())
        self.assertFalse(m.biv.is_fixed())

    def test_cast_to_binary(self):
        m = ConcreteModel()
        m.iv = AutoLinkedBooleanVar()
        m.biv = AutoLinkedBinaryVar(m.iv)
        m.iv.associate_binary_var(m.biv)

        m.biv = 1

        deprecation_msg = "Implicit conversion of the Boolean indicator_var 'iv'"

        out = StringIO()
        with LoggingIntercept(out):
            self.assertEqual(m.iv.lb, 0)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertEqual(m.iv.ub, 1)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertEqual(m.iv.bounds, (0, 1))
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            m.iv.lb = 1
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            m.iv.ub = 1
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            m.iv.bounds = (1, 1)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            m.iv.setlb(1)
            self.assertEqual(m.biv.lb, 1)
            m.biv.setlb(0)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            m.iv.setub(0)
            self.assertEqual(m.biv.ub, 0)
            m.biv.setub(1)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs(abs(m.iv).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            with self.assertRaisesRegex(
                PyomoException,
                r"Cannot convert non-constant Pyomo numeric value \(biv\) to bool",
            ):
                bool(m.iv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            with self.assertRaisesRegex(
                TypeError,
                r"Implicit conversion of Pyomo numeric value \(biv\) to float",
            ):
                float(m.iv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            with self.assertRaisesRegex(
                TypeError, r"Implicit conversion of Pyomo numeric value \(biv\) to int"
            ):
                int(m.iv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((-m.iv).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs(+m.iv, m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(m.iv.has_lb())
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(m.iv.has_ub())
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(m.iv.is_binary())
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertFalse(m.iv.is_continuous())
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertTrue(m.iv.is_integer())
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertEqual(m.iv.polynomial_degree(), 1)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv == 0).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv <= 0).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv >= 0).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv < 0).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv > 0).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            e = m.iv + 1
        assertExpressionsEqual(self, e, EXPR.LinearExpression([m.biv, 1]))
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            e = m.iv - 1
        assertExpressionsEqual(self, e, EXPR.LinearExpression([m.biv, -1]))
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv * 2).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv / 2).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((m.iv**2).args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            e = 1 + m.iv
        assertExpressionsEqual(self, e, EXPR.LinearExpression([1, m.biv]))
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            e = 1 - m.iv
        assertExpressionsEqual(
            self,
            e,
            EXPR.LinearExpression([1, EXPR.MonomialTermExpression((-1, m.biv))]),
        )
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((2 * m.iv).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((2 / m.iv).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            self.assertIs((2**m.iv).args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a += 1
        assertExpressionsEqual(self, a, EXPR.LinearExpression([m.biv, 1]))
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a -= 1
        assertExpressionsEqual(self, a, EXPR.LinearExpression([m.biv, -1]))
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a *= 2
            self.assertIs(a.args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a /= 2
            self.assertIs(a.args[1], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())

        out = StringIO()
        with LoggingIntercept(out):
            a = m.iv
            a **= 2
            self.assertIs(a.args[0], m.biv)
        self.assertIn(deprecation_msg, out.getvalue())


if __name__ == '__main__':
    unittest.main()

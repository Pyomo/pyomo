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

from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction


class TestTransformCurrentDisjunctiveLogic(unittest.TestCase):
    def make_two_term_disjunction(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d2.c = Constraint()
        m.disjunction1 = Disjunction(expr=[m.d1, m.d2])

        return m

    def check_fixed_mip(self, m):
        self.assertTrue(m.d1.indicator_var.fixed)
        self.assertTrue(m.d1.active)
        self.assertIs(m.d1.ctype, Block)

        self.assertTrue(m.d2.indicator_var.fixed)
        self.assertFalse(m.d2.active)
        self.assertFalse(m.disjunction1.active)

    def test_fix_disjuncts_and_reverse(self):
        m = self.make_two_term_disjunction()

        m.d1.indicator_var.set_value(True)
        m.d2.indicator_var.set_value(False)

        reverse = TransformationFactory(
            'gdp.transform_current_disjunctive_logic'
        ).apply_to(m)
        self.check_fixed_mip(m)

        TransformationFactory('gdp.transform_current_disjunctive_logic').apply_to(
            m, reverse=reverse
        )
        self.assertFalse(m.d1.indicator_var.fixed)
        self.assertTrue(m.d1.active)
        self.assertTrue(m.d1.indicator_var.value)
        self.assertFalse(m.d2.indicator_var.fixed)
        self.assertTrue(m.d2.active)
        self.assertFalse(m.d2.indicator_var.value)
        self.assertIs(m.d1.ctype, Disjunct)
        self.assertIs(m.d2.ctype, Disjunct)
        self.assertTrue(m.disjunction1.active)

    def test_fix_disjuncts_implied_by_true_disjunct(self):
        m = self.make_two_term_disjunction()

        m.d1.indicator_var.set_value(True)

        reverse = TransformationFactory(
            'gdp.transform_current_disjunctive_logic'
        ).apply_to(m)
        self.check_fixed_mip(m)

        TransformationFactory('gdp.transform_current_disjunctive_logic').apply_to(
            m, reverse=reverse
        )
        self.assertFalse(m.d1.indicator_var.fixed)
        self.assertTrue(m.d1.active)
        self.assertTrue(m.d1.indicator_var.value)
        self.assertFalse(m.d2.indicator_var.fixed)
        self.assertTrue(m.d2.active)
        self.assertIsNone(m.d2.indicator_var.value)
        self.assertIs(m.d1.ctype, Disjunct)
        self.assertIs(m.d2.ctype, Disjunct)
        self.assertTrue(m.disjunction1.active)

    def test_fix_disjuncts_implied_by_false_disjunct(self):
        m = self.make_two_term_disjunction()

        m.d2.indicator_var.set_value(False)

        reverse = TransformationFactory(
            'gdp.transform_current_disjunctive_logic'
        ).apply_to(m)
        self.check_fixed_mip(m)

        TransformationFactory('gdp.transform_current_disjunctive_logic').apply_to(
            m, reverse=reverse
        )
        self.assertFalse(m.d1.indicator_var.fixed)
        self.assertTrue(m.d1.active)
        self.assertIsNone(m.d1.indicator_var.value)
        self.assertFalse(m.d2.indicator_var.fixed)
        self.assertTrue(m.d2.active)
        self.assertFalse(m.d2.indicator_var.value)
        self.assertIs(m.d1.ctype, Disjunct)
        self.assertIs(m.d2.ctype, Disjunct)
        self.assertTrue(m.disjunction1.active)

    def test_xor_sums_to_0(self):
        m = self.make_two_term_disjunction()
        m.d1.indicator_var.set_value(False)
        m.d2.indicator_var.set_value(False)
        with self.assertRaisesRegex(
            InfeasibleConstraintException,
            "Exactly-one constraint for Disjunction "
            "'disjunction1' is violated. The following Disjuncts "
            "are selected: ",
        ):
            TransformationFactory('gdp.transform_current_disjunctive_logic').apply_to(m)

    def test_xor_sums_to_more_than_1(self):
        m = self.make_two_term_disjunction()
        m.d1.indicator_var.set_value(True)
        m.d2.indicator_var.set_value(True)
        with self.assertRaisesRegex(
            InfeasibleConstraintException,
            "Exactly-one constraint for Disjunction "
            "'disjunction1' is violated. The following Disjuncts "
            "are selected: d., d.",
        ):
            TransformationFactory('gdp.transform_current_disjunctive_logic').apply_to(m)

    def add_three_term_disjunction(self, m):
        m.d = Disjunct([1, 2, 3])
        m.disjunction2 = Disjunction(expr=[m.d[1], m.d[2], m.d[3]])

    def test_ignore_deactivated_disjuncts(self):
        m = self.make_two_term_disjunction()
        self.add_three_term_disjunction(m)

        m.d[1].deactivate()
        m.d[2].indicator_var = True
        m.d[3].indicator_var = False

        reverse = TransformationFactory(
            'gdp.transform_current_disjunctive_logic'
        ).apply_to(m, targets=m.disjunction2)

        self.assertTrue(m.d[1].indicator_var.fixed)
        self.assertFalse(m.d[1].indicator_var.value)
        self.assertFalse(m.d[1].active)

        self.assertTrue(m.d[2].indicator_var.fixed)
        self.assertTrue(m.d[2].indicator_var.value)
        self.assertTrue(m.d[2].active)
        self.assertIs(m.d[2].ctype, Block)

        self.assertTrue(m.d[3].indicator_var.fixed)
        self.assertFalse(m.d[3].indicator_var.value)
        self.assertFalse(m.d[3].active)

        self.assertFalse(m.disjunction2.active)

        TransformationFactory('gdp.transform_current_disjunctive_logic').apply_to(
            m, reverse=reverse
        )

        self.assertTrue(m.d[1].indicator_var.fixed)
        self.assertFalse(m.d[1].indicator_var.value)
        self.assertFalse(m.d[1].active)
        self.assertIs(m.d[1].ctype, Disjunct)

        self.assertFalse(m.d[2].indicator_var.fixed)
        self.assertTrue(m.d[2].indicator_var.value)
        self.assertTrue(m.d[2].active)
        self.assertIs(m.d[2].ctype, Disjunct)

        self.assertFalse(m.d[3].indicator_var.fixed)
        self.assertFalse(m.d[3].indicator_var.value)
        self.assertTrue(m.d[3].active)

        self.assertTrue(m.disjunction2.active)

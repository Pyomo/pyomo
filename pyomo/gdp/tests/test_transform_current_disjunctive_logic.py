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
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction


class TestTransformCurrentDisjunctiveLogic(unittest.TestCase):
    def make_two_term_disjunction(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d2.c = Constraint()
        m.d = Disjunction(expr=[m.d1, m.d2])

        return m

    def check_fixed_mip(self, m):
        self.assertTrue(m.d1.indicator_var.fixed)
        self.assertTrue(m.d1.active)
        self.assertIs(m.d1.ctype, Block)

        self.assertTrue(m.d2.indicator_var.fixed)
        self.assertFalse(m.d2.active)
        self.assertFalse(m.d.active)

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
        self.assertTrue(m.d.active)

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
        self.assertTrue(m.d.active)

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
        self.assertTrue(m.d.active)

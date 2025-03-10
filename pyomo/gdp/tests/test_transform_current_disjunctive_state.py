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

from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error


class TestTransformCurrentDisjunctiveState(unittest.TestCase):
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
            'gdp.transform_current_disjunctive_state'
        ).apply_to(m)
        self.check_fixed_mip(m)

        TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(
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
            'gdp.transform_current_disjunctive_state'
        ).apply_to(m)
        self.check_fixed_mip(m)

        TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(
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
            'gdp.transform_current_disjunctive_state'
        ).apply_to(m)
        self.check_fixed_mip(m)

        TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(
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
            "Logical constraint for Disjunction "
            "'disjunction1' is violated: All the "
            "Disjunct indicator_vars are 'False.'",
        ):
            TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)

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
            TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)

    def add_three_term_disjunction(self, m, exactly_one=True):
        m.d = Disjunct([1, 2, 3])
        m.disjunction2 = Disjunction(expr=[m.d[1], m.d[2], m.d[3]], xor=exactly_one)

    def test_ignore_deactivated_disjuncts(self):
        m = self.make_two_term_disjunction()
        self.add_three_term_disjunction(m)

        m.d[1].deactivate()
        m.d[2].indicator_var = True
        m.d[3].indicator_var = False

        reverse = TransformationFactory(
            'gdp.transform_current_disjunctive_state'
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
        self.assertIs(m.d[3].ctype, Block)

        self.assertFalse(m.disjunction2.active)

        # disjunction1 should be untransformed
        self.assertTrue(m.disjunction1.active)
        self.assertTrue(m.d1.active)
        self.assertTrue(m.d2.active)
        self.assertIs(m.d1.ctype, Disjunct)
        self.assertIs(m.d2.ctype, Disjunct)
        self.assertIsNone(m.d1.indicator_var.value)
        self.assertIsNone(m.d2.indicator_var.value)

        TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(
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
        self.assertIs(m.d[3].ctype, Disjunct)

        self.assertTrue(m.disjunction2.active)

    def test_no_partial_reverse(self):
        m = self.make_two_term_disjunction()
        self.add_three_term_disjunction(m)

        m.d1.indicator_var.set_value(True)
        m.d2.indicator_var.set_value(False)
        m.d[2].indicator_var = True
        m.d[3].indicator_var = False

        reverse = TransformationFactory(
            'gdp.transform_current_disjunctive_state'
        ).apply_to(m)

        self.check_fixed_mip(m)

        with self.assertRaisesRegex(
            ValueError,
            "The 'gdp.transform_current_disjunctive_state' transformation "
            "cannot be called with both targets and a reverse token "
            "specified. If reversing the transformation, do not include "
            "targets: The reverse transformation will restore all the "
            "components the original transformation call transformed.",
        ):
            TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(
                m, reverse=reverse, targets=m.disjunction2
            )

    def test_at_least_one_disjunction(self):
        m = ConcreteModel()
        self.add_three_term_disjunction(m, exactly_one=False)

        m.d[1].indicator_var.fix(True)
        m.d[2].indicator_var = True
        m.d[3].indicator_var = False

        reverse = TransformationFactory(
            'gdp.transform_current_disjunctive_state'
        ).apply_to(m)

        self.assertTrue(m.d[1].indicator_var.fixed)
        self.assertTrue(m.d[1].indicator_var.value)
        self.assertTrue(m.d[1].active)
        self.assertIs(m.d[1].ctype, Block)

        self.assertTrue(m.d[2].indicator_var.fixed)
        self.assertTrue(m.d[2].indicator_var.value)
        self.assertTrue(m.d[2].active)
        self.assertIs(m.d[2].ctype, Block)

        self.assertTrue(m.d[3].indicator_var.fixed)
        self.assertFalse(m.d[3].indicator_var.value)
        self.assertFalse(m.d[3].active)
        self.assertIs(m.d[3].ctype, Block)

        self.assertFalse(m.disjunction2.active)

        reverse = TransformationFactory(
            'gdp.transform_current_disjunctive_state'
        ).apply_to(m, reverse=reverse)

        self.assertTrue(m.d[1].indicator_var.fixed)
        self.assertTrue(m.d[1].indicator_var.value)
        self.assertTrue(m.d[1].active)
        self.assertIs(m.d[1].ctype, Disjunct)

        self.assertFalse(m.d[2].indicator_var.fixed)
        self.assertTrue(m.d[2].indicator_var.value)
        self.assertTrue(m.d[2].active)
        self.assertIs(m.d[2].ctype, Disjunct)

        self.assertFalse(m.d[3].indicator_var.fixed)
        self.assertFalse(m.d[3].indicator_var.value)
        self.assertTrue(m.d[3].active)
        self.assertIs(m.d[3].ctype, Disjunct)

        self.assertTrue(m.disjunction2.active)

    def test_at_least_one_disjunction_infeasible(self):
        m = ConcreteModel()
        self.add_three_term_disjunction(m, exactly_one=False)

        m.d[1].indicator_var.fix(False)
        m.d[2].indicator_var = False
        m.d[3].indicator_var = False

        with self.assertRaisesRegex(
            InfeasibleConstraintException,
            "Logical constraint for Disjunction "
            "'disjunction2' is violated: All "
            "the Disjunct indicator_vars are 'False.'",
        ):
            TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)

    def test_not_enough_info_to_fully_transform(self):
        m = ConcreteModel()
        m.d = Disjunct([1, 2, 3, 4])
        m.disj1 = Disjunction(expr=[m.d[1], m.d[2]])
        m.disj2 = Disjunction(expr=[m.d[3], m.d[4]])

        m.d[1].indicator_var = True

        with self.assertRaisesRegex(
            GDP_Error,
            "Disjunction 'disj2' does not contain enough Disjuncts with "
            "values in their indicator_vars to specify which Disjuncts "
            "are True. Cannot fully transform model.",
        ):
            reverse = TransformationFactory(
                'gdp.transform_current_disjunctive_state'
            ).apply_to(m)

    def test_not_enough_info_in_single_disjunction_to_fully_transform_xor(self):
        m = ConcreteModel()
        m.d = Disjunct([1, 2, 3, 4])
        m.disj1 = Disjunction(expr=[m.d[1], m.d[2], m.d[3], m.d[4]])
        m.d[1].indicator_var = False
        m.d[2].indicator_var = False

        with self.assertRaisesRegex(
            GDP_Error,
            "Disjunction 'disj1' does not contain enough Disjuncts with "
            "values in their indicator_vars to specify which Disjuncts "
            "are True. Cannot fully transform model.",
        ):
            reverse = TransformationFactory(
                'gdp.transform_current_disjunctive_state'
            ).apply_to(m)

    def test_not_enough_info_in_single_disjunction_to_fully_transform_or(self):
        m = ConcreteModel()
        m.d = Disjunct([1, 2, 3, 4])
        m.disj1 = Disjunction(expr=[m.d[1], m.d[2], m.d[3], m.d[4]], xor=False)
        m.d[1].indicator_var = True
        m.d[2].indicator_var = False

        with self.assertRaisesRegex(
            GDP_Error,
            "Disjunction 'disj1' does not contain enough Disjuncts with "
            "values in their indicator_vars to specify which Disjuncts "
            "are True. Cannot fully transform model.",
        ):
            reverse = TransformationFactory(
                'gdp.transform_current_disjunctive_state'
            ).apply_to(m)

    def test_complain_about_dangling_disjuncts(self):
        m = ConcreteModel()
        m.d = Disjunct([1, 2, 3, 4])
        m.disj1 = Disjunction(expr=[m.d[1], m.d[2], m.d[3]])
        m.d[1].indicator_var = True

        with self.assertRaisesRegex(
            GDP_Error,
            r"Found active Disjuncts on the model that "
            r"were not included in any Disjunctions:\nd\[4\]\nPlease "
            r"deactivate them or include them in a Disjunction.",
        ):
            reverse = TransformationFactory(
                'gdp.transform_current_disjunctive_state'
            ).apply_to(m)

# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging

import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import identify_variables
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory, Var
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins import simple_disjunction_transform


class CommonModels:
    def make_two_term_model(self):
        """A flat two-term Disjunction whose Disjuncts each have two Constraints."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(bounds=(0, 10))
        m.d1 = Disjunct()
        m.d1.c1 = Constraint(expr=m.x >= 1)
        m.d1.c2 = Constraint(expr=m.y <= 5)
        m.d2 = Disjunct()
        m.d2.c1 = Constraint(expr=m.x <= 2)
        m.d2.c2 = Constraint(expr=m.y >= 3)
        m.disjunction = Disjunction(expr=[m.d1, m.d2])
        return m

    def make_nested_model(self):
        """A two-term Disjunction with a nested Disjunction on the first Disjunct."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(bounds=(0, 10))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d1.inner1 = Disjunct()
        m.d1.inner1.c = Constraint(expr=m.y >= 2)
        m.d1.inner2 = Disjunct()
        m.d1.inner2.c = Constraint(expr=m.y <= 1)
        m.d1.inner = Disjunction(expr=[m.d1.inner1, m.d1.inner2])
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 2)
        m.disjunction = Disjunction(expr=[m.d1, m.d2])
        return m


class TestSimpleDisjunctionTransformation(unittest.TestCase, CommonModels):
    def get_transformation_block(self, parent):
        return parent.component('_pyomo_gdp_simple_disjunction_reformulation')

    def constraints_of(self, disjunct):
        return list(
            disjunct.component_data_objects(Constraint, active=True, descend_into=Block)
        )

    # ------------------------------------------------------------------ #
    # 'first' constraint selection (the default)                         #
    # ------------------------------------------------------------------ #
    def test_first_method_builds_simple_disjunction(self):
        m = self.make_two_term_model()
        TransformationFactory('gdp.simple_disjunction').apply_to(m)

        trans = self.get_transformation_block(m)
        self.assertIsNotNone(trans)
        simple = trans.simple_disjunction
        self.assertIsInstance(simple, Disjunction.__mro__[0])
        self.assertEqual(len(simple.disjuncts), 2)

        first_disjunct, second_disjunct = simple.disjuncts
        self.assertEqual(len(self.constraints_of(first_disjunct)), 1)
        self.assertEqual(len(self.constraints_of(second_disjunct)), 1)
        # 'first' keeps c1 of each original Disjunct
        assertExpressionsEqual(
            self, self.constraints_of(first_disjunct)[0].expr, m.d1.c1.expr
        )
        assertExpressionsEqual(
            self, self.constraints_of(second_disjunct)[0].expr, m.d2.c1.expr
        )

    def test_block_added_to_parent_of_disjunction(self):
        m = ConcreteModel()
        m.sub = Block()
        m.sub.x = Var(bounds=(0, 10))
        m.sub.d1 = Disjunct()
        m.sub.d1.c = Constraint(expr=m.sub.x >= 1)
        m.sub.d2 = Disjunct()
        m.sub.d2.c = Constraint(expr=m.sub.x <= 2)
        m.sub.disjunction = Disjunction(expr=[m.sub.d1, m.sub.d2])

        TransformationFactory('gdp.simple_disjunction').apply_to(
            m, targets=m.sub.disjunction
        )
        # The new Block lives on the parent Block of the Disjunction, not the model
        self.assertIsNotNone(self.get_transformation_block(m.sub))
        self.assertIsNone(self.get_transformation_block(m))

    def test_original_disjunction_untouched(self):
        m = self.make_two_term_model()
        TransformationFactory('gdp.simple_disjunction').apply_to(m)

        self.assertTrue(m.disjunction.active)
        self.assertIsNone(m.disjunction.algebraic_constraint)
        for d in (m.d1, m.d2):
            self.assertTrue(d.active)
            self.assertEqual(len(self.constraints_of(d)), 2)

    def test_new_constraints_reference_original_variables(self):
        m = self.make_two_term_model()
        TransformationFactory('gdp.simple_disjunction').apply_to(m)
        simple = self.get_transformation_block(m).simple_disjunction
        new_con = self.constraints_of(simple.disjuncts[0])[0]
        # The reused expression must point at the original model Var
        self.assertIn(m.x, ComponentSet(identify_variables(new_con.expr)))

    def test_multiple_disjunctions(self):
        m = self.make_two_term_model()
        m.z = Var(bounds=(0, 10))
        m.e1 = Disjunct()
        m.e1.c = Constraint(expr=m.z >= 1)
        m.e2 = Disjunct()
        m.e2.c = Constraint(expr=m.z <= 2)
        m.disjunction2 = Disjunction(expr=[m.e1, m.e2])

        TransformationFactory('gdp.simple_disjunction').apply_to(m)
        # Each Disjunction gets its own reformulation Block on the parent
        blocks = [
            b
            for b in m.component_objects(Block)
            if b.local_name.startswith('_pyomo_gdp_simple_disjunction_reformulation')
        ]
        self.assertEqual(len(blocks), 2)

    # ------------------------------------------------------------------ #
    # 'user_specified' constraint selection                             #
    # ------------------------------------------------------------------ #
    def test_user_specified_selection(self):
        m = self.make_two_term_model()
        TransformationFactory('gdp.simple_disjunction').apply_to(
            m,
            constraint_selection_method='user_specified',
            selected_constraints={m.d1: m.d1.c2, m.d2: m.d2.c2},
        )
        simple = self.get_transformation_block(m).simple_disjunction
        assertExpressionsEqual(
            self, self.constraints_of(simple.disjuncts[0])[0].expr, m.d1.c2.expr
        )
        assertExpressionsEqual(
            self, self.constraints_of(simple.disjuncts[1])[0].expr, m.d2.c2.expr
        )

    def test_user_specified_accepts_component_map(self):
        m = self.make_two_term_model()
        selection = ComponentMap()
        selection[m.d1] = m.d1.c1
        selection[m.d2] = m.d2.c2
        TransformationFactory('gdp.simple_disjunction').apply_to(
            m,
            constraint_selection_method='user_specified',
            selected_constraints=selection,
        )
        simple = self.get_transformation_block(m).simple_disjunction
        self.assertEqual(len(simple.disjuncts), 2)

    def test_user_specified_accepts_single_constraint_in_list(self):
        m = self.make_two_term_model()
        TransformationFactory('gdp.simple_disjunction').apply_to(
            m,
            constraint_selection_method='user_specified',
            selected_constraints={m.d1: [m.d1.c2], m.d2: [m.d2.c1]},
        )
        simple = self.get_transformation_block(m).simple_disjunction
        assertExpressionsEqual(
            self, self.constraints_of(simple.disjuncts[0])[0].expr, m.d1.c2.expr
        )

    def test_user_specified_multiple_constraints_not_yet_supported(self):
        # The data structure accepts several constraints per Disjunct (so that
        # future aggregating methods can use it), but the current methods keep
        # exactly one, so this is a clear, signposted error rather than a crash.
        m = self.make_two_term_model()
        with self.assertRaisesRegex(GDP_Error, "not\\s+yet\\s+implemented"):
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m,
                constraint_selection_method='user_specified',
                selected_constraints={m.d1: [m.d1.c1, m.d1.c2], m.d2: m.d2.c1},
            )

    def test_user_specified_requires_mapping(self):
        m = self.make_two_term_model()
        with self.assertRaisesRegex(
            GDP_Error, "no 'selected_constraints' mapping was provided"
        ):
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m, constraint_selection_method='user_specified'
            )

    def test_user_specified_inactive_constraint_error(self):
        m = self.make_two_term_model()
        m.d1.c2.deactivate()
        with self.assertRaisesRegex(GDP_Error, "is not active"):
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m,
                constraint_selection_method='user_specified',
                selected_constraints={m.d1: m.d1.c2, m.d2: m.d2.c1},
            )

    def test_user_specified_foreign_constraint_error(self):
        m = self.make_two_term_model()
        with self.assertRaisesRegex(GDP_Error, "is not one of that Disjunct's own"):
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m,
                constraint_selection_method='user_specified',
                selected_constraints={m.d1: m.d2.c1, m.d2: m.d2.c1},
            )

    def test_user_specified_missing_disjunct_error(self):
        m = self.make_two_term_model()
        with self.assertRaisesRegex(
            GDP_Error, "was not assigned one in 'selected_constraints'"
        ):
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m,
                constraint_selection_method='user_specified',
                selected_constraints={m.d1: m.d1.c1},
            )

    def test_user_specified_skips_disjunct_without_constraints(self):
        m = self.make_two_term_model()
        m.d3 = Disjunct()  # no constraints
        m.disjunction.deactivate()
        m.disjunction2 = Disjunction(expr=[m.d1, m.d2, m.d3])
        TransformationFactory('gdp.simple_disjunction').apply_to(
            m,
            targets=m.disjunction2,
            constraint_selection_method='user_specified',
            selected_constraints={m.d1: m.d1.c1, m.d2: m.d2.c1},
        )
        simple = self.get_transformation_block(m).simple_disjunction
        self.assertEqual(len(simple.disjuncts), 2)

    def test_selected_constraints_ignored_warning(self):
        m = self.make_two_term_model()
        with LoggingIntercept(level=logging.WARNING) as log:
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m, selected_constraints={m.d1: m.d1.c1}
            )
        self.assertIn("the mapping will be ignored", log.getvalue())

    # ------------------------------------------------------------------ #
    # Skipping and error behavior                                        #
    # ------------------------------------------------------------------ #
    def test_disjunct_without_constraints_skipped(self):
        m = self.make_two_term_model()
        m.d3 = Disjunct()  # no constraints
        m.disjunction.deactivate()
        m.disjunction2 = Disjunction(expr=[m.d1, m.d2, m.d3])
        TransformationFactory('gdp.simple_disjunction').apply_to(
            m, targets=m.disjunction2
        )
        simple = self.get_transformation_block(m).simple_disjunction
        self.assertEqual(len(simple.disjuncts), 2)

    def test_empty_simple_disjunction_raises(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.disjunction = Disjunction(expr=[m.d1, m.d2])
        with self.assertRaisesRegex(
            GDP_Error, "none of its active Disjuncts produced a constraint"
        ):
            TransformationFactory('gdp.simple_disjunction').apply_to(m)
        # No reformulation Block should have been created
        self.assertIsNone(self.get_transformation_block(m))

    def test_inactive_disjunction_target_raises(self):
        m = self.make_two_term_model()
        m.disjunction.deactivate()
        with self.assertRaisesRegex(GDP_Error, "is deactivated"):
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m, targets=m.disjunction
            )

    def test_inactive_disjunction_skipped_when_no_targets(self):
        m = self.make_two_term_model()
        m.disjunction.deactivate()
        TransformationFactory('gdp.simple_disjunction').apply_to(m)
        self.assertIsNone(self.get_transformation_block(m))

    def test_only_active_constraints_selected_by_first(self):
        m = self.make_two_term_model()
        m.d1.c1.deactivate()  # 'first' should now skip c1 and pick c2
        TransformationFactory('gdp.simple_disjunction').apply_to(m)
        simple = self.get_transformation_block(m).simple_disjunction
        assertExpressionsEqual(
            self, self.constraints_of(simple.disjuncts[0])[0].expr, m.d1.c2.expr
        )

    # ------------------------------------------------------------------ #
    # Nested disjunctions                                                #
    # ------------------------------------------------------------------ #
    def test_nested_disjunction_skipped_when_no_targets(self):
        m = self.make_nested_model()
        TransformationFactory('gdp.simple_disjunction').apply_to(m)
        # Neither the outer (contains nesting) nor the inner (is nested) is built
        self.assertIsNone(self.get_transformation_block(m))
        self.assertIsNone(self.get_transformation_block(m.d1))

    def test_nested_outer_disjunction_target_raises(self):
        m = self.make_nested_model()
        with self.assertRaisesRegex(GDP_Error, "contains a nested Disjunction"):
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m, targets=m.disjunction
            )

    def test_nested_inner_disjunction_target_raises(self):
        m = self.make_nested_model()
        with self.assertRaisesRegex(GDP_Error, "is nested in another Disjunct"):
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m, targets=m.d1.inner
            )

    # ------------------------------------------------------------------ #
    # Targets and retrieval                                              #
    # ------------------------------------------------------------------ #
    def test_block_target_collects_disjunctions(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.x = Var(bounds=(0, 10))
        m.b.d1 = Disjunct()
        m.b.d1.c = Constraint(expr=m.b.x >= 1)
        m.b.d2 = Disjunct()
        m.b.d2.c = Constraint(expr=m.b.x <= 2)
        m.b.disjunction = Disjunction(expr=[m.b.d1, m.b.d2])

        TransformationFactory('gdp.simple_disjunction').apply_to(m, targets=m.b)
        self.assertIsNotNone(self.get_transformation_block(m.b))

    def test_target_not_on_instance_raises(self):
        m = self.make_two_term_model()
        other = self.make_two_term_model()
        with self.assertRaisesRegex(GDP_Error, "is not a component on instance"):
            TransformationFactory('gdp.simple_disjunction').apply_to(
                m, targets=other.disjunction
            )

    def test_get_simple_disjunction(self):
        m = self.make_two_term_model()
        xform = TransformationFactory('gdp.simple_disjunction')
        xform.apply_to(m)
        simple = xform.get_simple_disjunction(m.disjunction)
        self.assertIs(simple, self.get_transformation_block(m).simple_disjunction)

    def test_get_simple_disjunction_untransformed_raises(self):
        m = self.make_two_term_model()
        xform = TransformationFactory('gdp.simple_disjunction')
        with self.assertRaisesRegex(GDP_Error, "has not been transformed"):
            xform.get_simple_disjunction(m.disjunction)

    def test_get_src_disjunction(self):
        m = self.make_two_term_model()
        xform = TransformationFactory('gdp.simple_disjunction')
        xform.apply_to(m)
        simple = xform.get_simple_disjunction(m.disjunction)
        self.assertIs(xform.get_src_disjunction(simple), m.disjunction)

    def test_src_and_simple_are_inverse(self):
        m = self.make_two_term_model()
        m.z = Var(bounds=(0, 10))
        m.e1 = Disjunct()
        m.e1.c = Constraint(expr=m.z >= 1)
        m.e2 = Disjunct()
        m.e2.c = Constraint(expr=m.z <= 2)
        m.disjunction2 = Disjunction(expr=[m.e1, m.e2])

        xform = TransformationFactory('gdp.simple_disjunction')
        xform.apply_to(m)
        for src in (m.disjunction, m.disjunction2):
            simple = xform.get_simple_disjunction(src)
            self.assertIs(xform.get_src_disjunction(simple), src)

    def test_get_src_disjunction_bad_input_raises(self):
        m = self.make_two_term_model()
        xform = TransformationFactory('gdp.simple_disjunction')
        xform.apply_to(m)
        # m.disjunction is an original Disjunction, not a generated simple one
        with self.assertRaisesRegex(GDP_Error, "No source disjunction found"):
            xform.get_src_disjunction(m.disjunction)

    def test_transformation_out_of_place(self):
        m = self.make_two_term_model()
        new = TransformationFactory('gdp.simple_disjunction').create_using(m)
        self.assertIsNot(new, m)
        self.assertIsNotNone(self.get_transformation_block(new))
        self.assertIsNone(self.get_transformation_block(m))

    def test_result_is_transformable_by_bigm(self):
        m = self.make_two_term_model()
        xform = TransformationFactory('gdp.simple_disjunction')
        xform.apply_to(m)
        simple = xform.get_simple_disjunction(m.disjunction)
        # The generated Disjunction is a normal GDP component
        TransformationFactory('gdp.bigm').apply_to(m, targets=[simple])
        self.assertIsNotNone(simple.algebraic_constraint)

    def test_bad_instance_type_raises(self):
        m = self.make_two_term_model()
        with self.assertRaisesRegex(GDP_Error, "must be a ConcreteModel or Block"):
            TransformationFactory('gdp.simple_disjunction').apply_to(m.x)

    def test_disjunct_instance_raises(self):
        m = self.make_two_term_model()
        with self.assertRaisesRegex(GDP_Error, "must be a ConcreteModel or Block"):
            TransformationFactory('gdp.simple_disjunction').apply_to(m.d1)


if __name__ == '__main__':
    unittest.main()

import pyutilib.th as unittest

from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import update_boolean_vars_from_binary
from pyomo.environ import (
    ConcreteModel, BooleanVar, LogicalConstraint, lor, TransformationFactory, RangeSet,
    Var, Constraint, ComponentMap, value, BooleanSet, atleast, atmost, exactly)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn


def _generate_boolean_model(nvars):
    m = ConcreteModel()
    m.s = RangeSet(nvars)
    m.Y = BooleanVar(m.s)
    return m


def _constrs_contained_within(test_case, test_constr_tuples, constraint_list):
    """Checks to see if constraints defined by test_constr_tuples are in the constraint list.

    Parameters
    ----------
    constraint_list : Constraint
    test_constr_tuples : list of tuple
    test_case : unittest.TestCase
    """
    # Move const term from body
    def _move_const_from_body(lower, repn, upper):
        if repn.constant is not None and not repn.constant == 0:
            if lower is not None:
                lower -= repn.constant
            if upper is not None:
                upper -= repn.constant
        return value(lower), repn, value(upper)

    def _repns_match(repn, test_repn):
        if not len(repn.linear_vars) == len(test_repn.linear_vars):
            return False
        coef_map = ComponentMap((var, coef) for var, coef in zip(repn.linear_vars, repn.linear_coefs))
        for var, coef in zip(test_repn.linear_vars, test_repn.linear_coefs):
            if not coef_map.get(var, 0) == coef:
                return False
        return True

    constr_list_tuples = [
        _move_const_from_body(constr.lower, generate_standard_repn(constr.body), constr.upper)
        for constr in constraint_list.values()]
    for test_lower, test_body, test_upper in test_constr_tuples:
        test_repn = generate_standard_repn(test_body)
        test_lower, test_repn, test_upper = _move_const_from_body(test_lower, test_repn, test_upper)
        found_match = False
        # Make sure one of the list tuples matches
        for lower, repn, upper in constr_list_tuples:
            if lower == test_lower and upper == test_upper and _repns_match(repn, test_repn):
                found_match = True
                break
        test_case.assertTrue(found_match, "{} <= {} <= {} was not found in constraint list.".format(
            test_lower, test_body, test_upper))


@unittest.skipUnless(sympy_available, "Sympy not available")
class TestAtomicTransformations(unittest.TestCase):

    def test_implies(self):
        m = ConcreteModel()
        m.x = BooleanVar()
        m.y = BooleanVar()
        m.p = LogicalConstraint(expr=m.x.implies(m.y))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(
            self, [(1, (1 - m.x.get_associated_binary()) + m.y.get_associated_binary(), None)],
            m.logic_to_linear.transformed_constraints)

    def test_literal(self):
        m = ConcreteModel()
        m.Y = BooleanVar()
        m.p = LogicalConstraint(expr=m.Y)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(
            self, [(1, m.Y.get_associated_binary(), 1)], m.logic_to_linear.transformed_constraints)

    def test_constant_True(self):
        m = ConcreteModel()
        with self.assertRaisesRegex(ValueError, "LogicalConstraint 'p' is always True."):
            m.p = LogicalConstraint(expr=True)
            TransformationFactory('core.logical_to_linear').apply_to(m)
        self.assertIsNone(m.component('logic_to_linear'))

    def test_nothing_to_do(self):
        m = ConcreteModel()
        m.p = LogicalConstraint()
        TransformationFactory('core.logical_to_linear').apply_to(m)
        self.assertIsNone(m.component('logic_to_linear'))


@unittest.skipUnless(sympy_available, "Sympy not available")
class TestLogicalToLinearTransformation(unittest.TestCase):
    def test_longer_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=m.Y[1].implies(lor(m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(
            self, [
                (1,
                 m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary()
                 + (1 - m.Y[1].get_associated_binary()),
                 None)
            ], m.logic_to_linear.transformed_constraints)

    def test_xfrm_atleast_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=atleast(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(
            self, [
                (2,
                 m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary(),
                 None)
            ], m.logic_to_linear.transformed_constraints)

    def test_xfrm_atmost_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=atmost(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(
            self, [
                (None,
                 m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary(),
                 2)
            ], m.logic_to_linear.transformed_constraints)

    def test_xfrm_exactly_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=exactly(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(
            self, [
                (2, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary(), 2)
            ], m.logic_to_linear.transformed_constraints)

    def test_xfrm_special_atoms_nonroot(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=m.Y[1].implies(atleast(2, m.Y[1], m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 1)
        self.assertEqual(Y_aug[1].domain, BooleanSet)
        _constrs_contained_within(
            self, [
                (None, sum(m.Y[:].get_associated_binary()) - (1 + 2 * Y_aug[1].get_associated_binary()), 0),
                (1, (1 - m.Y[1].get_associated_binary()) + Y_aug[1].get_associated_binary(), None),
                (None, 2 - 2 * (1 - Y_aug[1].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0)
            ], m.logic_to_linear.transformed_constraints)

        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=m.Y[1].implies(atmost(2, m.Y[1], m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 1)
        self.assertEqual(Y_aug[1].domain, BooleanSet)
        _constrs_contained_within(
            self, [
                (None, sum(m.Y[:].get_associated_binary()) - (1 - Y_aug[1].get_associated_binary() + 2), 0),
                (1, (1 - m.Y[1].get_associated_binary()) + Y_aug[1].get_associated_binary(), None),
                (None, 3 - 3 * Y_aug[1].get_associated_binary() - sum(m.Y[:].get_associated_binary()), 0)
            ], m.logic_to_linear.transformed_constraints)

        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=m.Y[1].implies(exactly(2, m.Y[1], m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 3)
        self.assertEqual(Y_aug[1].domain, BooleanSet)
        _constrs_contained_within(
            self, [
                (1, (1 - m.Y[1].get_associated_binary()) + Y_aug[1].get_associated_binary(), None),
                (None, sum(m.Y[:].get_associated_binary()) - (1 - Y_aug[1].get_associated_binary() + 2), 0),
                (None, 2 - 2 * (1 - Y_aug[1].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0),
                (1, sum(Y_aug[:].get_associated_binary()), None),
                (None, sum(m.Y[:].get_associated_binary()) - (1 + 2 * (1 - Y_aug[2].get_associated_binary())), 0),
                (None, 3 - 3 * (1 - Y_aug[3].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0),
            ], m.logic_to_linear.transformed_constraints)

        # Note: x is now a variable
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.x = Var(bounds=(1, 3))
        m.p = LogicalConstraint(expr=m.Y[1].implies(exactly(m.x, m.Y[1], m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 3)
        self.assertEqual(Y_aug[1].domain, BooleanSet)
        _constrs_contained_within(
            self, [
                (1, (1 - m.Y[1].get_associated_binary()) + Y_aug[1].get_associated_binary(), None),
                (None, sum(m.Y[:].get_associated_binary()) - (m.x + 2 * (1 - Y_aug[1].get_associated_binary())), 0),
                (None, m.x - 3 * (1 - Y_aug[1].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0),
                (1, sum(Y_aug[:].get_associated_binary()), None),
                (None, sum(m.Y[:].get_associated_binary()) - (m.x - 1 + 3 * (1 - Y_aug[2].get_associated_binary())), 0),
                (None, m.x + 1 - 4 * (1 - Y_aug[3].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0),
            ], m.logic_to_linear.transformed_constraints)

    def test_xfrm_atleast_nested(self):
        m = _generate_boolean_model(4)
        m.p = LogicalConstraint(expr=atleast(1, atleast(2, m.Y[1], m.Y[1].lor(m.Y[2]), m.Y[2]).lor(m.Y[3]), m.Y[4]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 3)
        _constrs_contained_within(
            self, [
                (1, Y_aug[1].get_associated_binary() + m.Y[4].get_associated_binary(), None),
                (1, 1 - Y_aug[2].get_associated_binary() + Y_aug[1].get_associated_binary(), None),
                (1, 1 - m.Y[3].get_associated_binary() + Y_aug[1].get_associated_binary(), None),
                (1,
                 Y_aug[2].get_associated_binary() + m.Y[3].get_associated_binary()
                 + 1 - Y_aug[1].get_associated_binary(),
                 None),
                (1, 1 - m.Y[1].get_associated_binary() + Y_aug[3].get_associated_binary(), None),
                (1, 1 - m.Y[2].get_associated_binary() + Y_aug[3].get_associated_binary(), None),
                (1,
                 m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + 1 - Y_aug[3].get_associated_binary(),
                 None),
                (None,
                 2 - 2 * (1 - Y_aug[2].get_associated_binary())
                 - (m.Y[1].get_associated_binary() + Y_aug[3].get_associated_binary() + m.Y[2].get_associated_binary()),
                 0),
                (None,
                 m.Y[1].get_associated_binary() + Y_aug[3].get_associated_binary() + m.Y[2].get_associated_binary()
                 - (1 + 2 * Y_aug[2].get_associated_binary()),
                 0)
            ], m.logic_to_linear.transformed_constraints)

    def test_link_with_gdp_indicators(self):
        m = _generate_boolean_model(4)
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.x = Var()
        m.dd = Disjunct([1, 2])
        m.d1.c = Constraint(expr=m.x >= 2)
        m.d2.c = Constraint(expr=m.x <= 10)
        m.dd[1].c = Constraint(expr=m.x >= 5)
        m.dd[2].c = Constraint(expr=m.x <= 6)
        m.Y[1].associate_binary_var(m.d1.indicator_var)
        m.Y[2].associate_binary_var(m.d2.indicator_var)
        m.Y[3].associate_binary_var(m.dd[1].indicator_var)
        m.Y[4].associate_binary_var(m.dd[2].indicator_var)
        m.p = LogicalConstraint(expr=m.Y[1].implies(lor(m.Y[3], m.Y[4])))
        m.p2 = LogicalConstraint(expr=atmost(2, *m.Y[:]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(
            self, [
                (1, m.dd[1].indicator_var + m.dd[2].indicator_var + 1 - m.d1.indicator_var, None),
                (None, m.d1.indicator_var + m.d2.indicator_var + m.dd[1].indicator_var + m.dd[2].indicator_var, 2)
            ], m.logic_to_linear.transformed_constraints)

    def test_gdp_nesting(self):
        m = _generate_boolean_model(2)
        m.disj = Disjunction(expr=[
            [m.Y[1].implies(m.Y[2])],
            [m.Y[2].equivalent_to(False)]
        ])
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(
            self, [
                (1, 1 - m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary(), None),
            ], m.disj_disjuncts[0].logic_to_linear.transformed_constraints)
        _constrs_contained_within(
            self, [
                (1, 1 - m.Y[2].get_associated_binary(), 1),
            ], m.disj_disjuncts[1].logic_to_linear.transformed_constraints)


@unittest.skipUnless(sympy_available, "Sympy not available")
class TestLogicalToLinearBackmap(unittest.TestCase):
    def test_backmap(self):
        m = _generate_boolean_model(3)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.Y_asbinary[1].value = 1
        m.Y_asbinary[2].value = 0
        update_boolean_vars_from_binary(m)
        self.assertTrue(m.Y[1].value)
        self.assertFalse(m.Y[2].value)
        self.assertIsNone(m.Y[3].value)

    def test_backmap_noninteger(self):
        m = _generate_boolean_model(2)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.Y_asbinary[1].value = 0.9
        update_boolean_vars_from_binary(m, integer_tolerance=0.1)
        self.assertTrue(m.Y[1].value)
        # Now try it without the tolerance set
        with self.assertRaisesRegexp(ValueError, r"Binary variable has non-\{0,1\} value"):
            update_boolean_vars_from_binary(m)


if __name__ == "__main__":
    unittest.main()

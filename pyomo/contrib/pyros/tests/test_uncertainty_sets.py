"""
Tests for PyROS uncertainty set.
"""

import itertools as it
import unittest

from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.environ import SolverFactory
from pyomo.core.base import (
    Block,
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    Var,
    value,
)
from pyomo.core.expr import RangedExpression
from pyomo.core.expr.compare import assertExpressionsEqual, assertExpressionsStructurallyEqual

from pyomo.contrib.pyros.uncertainty_sets import (
    AxisAlignedEllipsoidalSet,
    BoxSet,
    BudgetSet,
    CardinalitySet,
    DiscreteScenarioSet,
    EllipsoidalSet,
    FactorModelSet,
    IntersectionSet,
    PolyhedralSet,
    UncertaintySet,
)

import logging

logger = logging.getLogger(__name__)


if not (numpy_available and scipy_available):
    raise unittest.SkipTest('PyROS unit tests require parameterized, numpy, and scipy')

# === Config args for testing
nlp_solver = 'ipopt'
global_solver = 'baron'
global_solver_args = dict()
nlp_solver_args = dict()

_baron = SolverFactory('baron')
baron_available = _baron.available(exception_flag=False)
if baron_available:
    baron_license_is_valid = _baron.license_is_valid()
    baron_version = _baron.version()
else:
    baron_license_is_valid = False
    baron_version = (0, 0, 0)

_scip = SolverFactory('scip')
scip_available = _scip.available(exception_flag=False)
if scip_available:
    scip_license_is_valid = _scip.license_is_valid()
    scip_version = _scip.version()
else:
    scip_license_is_valid = False
    scip_version = (0, 0, 0)

_ipopt = SolverFactory("ipopt")
ipopt_available = _ipopt.available(exception_flag=False)


class TestBoxSet(unittest.TestCase):
    """
    Tests for the BoxSet.
    """

    def test_normal_construction_and_update(self):
        """
        Test BoxSet constructor and setter work normally
        when bounds are appropriate.
        """
        bounds = [[1, 2], [3, 4]]
        bset = BoxSet(bounds=bounds)
        np.testing.assert_allclose(
            bounds, bset.bounds, err_msg="BoxSet bounds not as expected"
        )

        # check bounds update
        new_bounds = [[3, 4], [5, 6]]
        bset.bounds = new_bounds
        np.testing.assert_allclose(
            new_bounds, bset.bounds, err_msg="BoxSet bounds not as expected"
        )

    def test_error_on_box_set_dim_change(self):
        """
        BoxSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        box set dimension (i.e. number of rows of `bounds`).
        """
        bounds = [[1, 2], [3, 4]]
        bset = BoxSet(bounds=bounds)  # 2-dimensional set

        exc_str = r"Attempting to set.*dimension 2 to a value of dimension 3"
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = [[1, 2], [3, 4], [5, 6]]

    def test_error_on_lb_exceeds_ub(self):
        """
        Test exception raised when an LB exceeds a UB.
        """
        bad_bounds = [[1, 2], [4, 3]]

        exc_str = r"Lower bound 4 exceeds upper bound 3"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(bad_bounds)

        # construct a valid box set
        bset = BoxSet([[1, 2], [3, 4]])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = bad_bounds

    def test_error_on_ragged_bounds_array(self):
        """
        Test ValueError raised on attempting to set BoxSet bounds
        to a ragged array.

        This test also validates `uncertainty_sets.is_ragged` for all
        pre-defined array-like attributes of all set-types, as the
        `is_ragged` method is used throughout.
        """
        # example ragged arrays
        ragged_arrays = (
            [[1, 2], 3],  # list and int in same sequence
            [[1, 2], [3, [4, 5]]],  # 2nd row ragged (list and int)
            [[1, 2], [3]],  # variable row lengths
        )

        # construct valid box set
        bset = BoxSet(bounds=[[1, 2], [3, 4]])

        # exception message should match this regex
        exc_str = r"Argument `bounds` should not be a ragged array-like.*"
        for ragged_arr in ragged_arrays:
            # assert error on construction
            with self.assertRaisesRegex(ValueError, exc_str):
                BoxSet(bounds=ragged_arr)

            # assert error on update
            with self.assertRaisesRegex(ValueError, exc_str):
                bset.bounds = ragged_arr

    def test_error_on_invalid_bounds_shape(self):
        """
        Test ValueError raised when attempting to set
        Box set bounds to array of incorrect shape
        (should be a 2-D array with 2 columns).
        """
        # 3d array
        three_d_arr = [[[1, 2], [3, 4], [5, 6]]]
        exc_str = (
            r"Argument `bounds` must be a 2-dimensional.*"
            r"\(detected 3 dimensions.*\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(three_d_arr)

        # construct valid box set
        bset = BoxSet([[1, 2], [3, 4], [5, 6]])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = three_d_arr

    def test_error_on_wrong_number_columns(self):
        """
        BoxSet bounds should be a 2D array-like with 2 columns.
        ValueError raised if number columns wrong
        """
        three_col_arr = [[1, 2, 3], [4, 5, 6]]
        exc_str = (
            r"Attribute 'bounds' should be of shape \(\.{3},2\), "
            r"but detected shape \(\.{3},3\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(three_col_arr)

        # construct a valid box set
        bset = BoxSet([[1, 2], [3, 4]])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = three_col_arr

    def test_error_on_empty_last_dimension(self):
        """
        Check ValueError raised when last dimension of BoxSet bounds is
        empty.
        """
        empty_2d_arr = [[], [], []]
        exc_str = (
            r"Last dimension of argument `bounds` must be non-empty "
            r"\(detected shape \(3, 0\)\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BoxSet(bounds=empty_2d_arr)

        # create a valid box set
        bset = BoxSet([[1, 2], [3, 4]])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            bset.bounds = empty_2d_arr

    def test_error_on_non_numeric_bounds(self):
        """
        Test that ValueError is raised if box set bounds
        are set to array-like with entries of a non-numeric
        type (such as int, float).
        """
        # invalid bounds (contains an entry type str)
        new_bounds = [[1, "test"], [3, 2]]

        exc_str = (
            r"Entry 'test' of the argument `bounds` "
            r"is not a valid numeric type \(provided type 'str'\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(TypeError, exc_str):
            BoxSet(new_bounds)

        # construct a valid box set
        bset = BoxSet(bounds=[[1, 2], [3, 4]])

        # assert error on update
        with self.assertRaisesRegex(TypeError, exc_str):
            bset.bounds = new_bounds

    def test_error_on_bounds_with_nan_or_inf(self):
        """
        Box set bounds set to array-like with inf or nan.
        """
        # construct a valid box set
        bset = BoxSet(bounds=[[1, 2], [3, 4]])

        for val_str in ["inf", "nan"]:
            bad_bounds = [[1, float(val_str)], [2, 3]]
            exc_str = (
                fr"Entry '{val_str}' of the argument `bounds` "
                fr"is not a finite numeric value"
            )
            # assert error on construction
            with self.assertRaisesRegex(ValueError, exc_str):
                BoxSet(bad_bounds)

            # assert error on update
            with self.assertRaisesRegex(ValueError, exc_str):
                bset.bounds = bad_bounds

    def test_set_as_constraint(self):
        """
        Test box set constraints added correctly.
        """
        m = ConcreteModel()
        box_set = BoxSet(bounds=[[1, 2], [3, 4]])
        uq = box_set.set_as_constraint(uncertain_params=None, block=m)

        con1, con2 = uq.uncertainty_cons
        var1, var2 = uq.uncertain_param_vars

        assertExpressionsEqual(
            self,
            con1.expr,
            RangedExpression((np.int64(1), var1, np.int64(2)), False),
        )
        assertExpressionsEqual(
            self,
            con2.expr,
            RangedExpression((np.int64(3), var2, np.int64(4)), False),
        )

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain parameters
        does not match the dimension.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        box_set = BoxSet(bounds=[[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, ".*dimension"):
            box_set.set_as_constraint(uncertain_params=[m.v1], block=m)

    def test_set_as_constraint_type_mismatch(self):
        """
        Check exception raised if uncertain parameter variables
        are of invalid type.
        """
        m = ConcreteModel()
        m.p1 = Param([0, 1], initialize=0, mutable=True)
        box_set = BoxSet(bounds=[[1, 2], [3, 4]])
        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            box_set.set_as_constraint(uncertain_params=[m.p1[0], m.p1[1]], block=m)

        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            box_set.set_as_constraint(uncertain_params=m.p1, block=m)

    @unittest.skipUnless(baron_available, "BARON is not available.")
    def test_compute_parameter_bounds(self):
        """
        Test parameter bounds computation with global solver
        is as expected.
        """
        box_set = BoxSet([[1, 2], [3, 4]])
        computed_bounds = box_set._compute_parameter_bounds(SolverFactory("baron"))
        np.testing.assert_allclose(computed_bounds, [[1, 2], [3, 4]])
        np.testing.assert_allclose(computed_bounds, box_set.parameter_bounds)

    def test_point_in_set(self):
        """
        Test point in set check works as expected.
        """
        box_set = BoxSet(bounds=[[1, 2], [3, 4]])

        in_set_points = [(1, 3), (1, 4), (2, 3), (2, 4), (1.5, 3.5)]
        out_of_set_points = [(0, 0), (0, 3), (0, 4), (1, 2), (3, 4)]
        for point in in_set_points:
            self.assertTrue(
                box_set.point_in_set(point),
                msg=f"Point {point} should not be in uncertainty set {box_set}."
            )
        for point in out_of_set_points:
            self.assertFalse(
                box_set.point_in_set(point),
                msg=f"Point {point} should not be in uncertainty set {box_set}."
            )

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var([0, 1], initialize=0)

        box_set = BoxSet(bounds=[(1, 2), (3, 4)])
        config = Block()
        config.uncertainty_set = box_set

        box_set.add_bounds_on_uncertain_parameters(
            config=config,
            uncertain_param_vars=m.uncertain_param_vars,
        )

        self.assertEqual(m.uncertain_param_vars[0].bounds, (1, 2))
        self.assertEqual(m.uncertain_param_vars[1].bounds, (3, 4))


class TestBudgetSet(unittest.TestCase):
    """
    Tests for the BudgetSet.
    """

    def test_normal_budget_construction_and_update(self):
        """
        Test BudgetSet constructor and attribute setters work
        appropriately.
        """
        budget_mat = [[1, 0, 1], [0, 1, 0]]
        budget_rhs_vec = [1, 3]

        # check attributes are as expected
        buset = BudgetSet(budget_mat, budget_rhs_vec)

        np.testing.assert_allclose(budget_mat, buset.budget_membership_mat)
        np.testing.assert_allclose(budget_rhs_vec, buset.budget_rhs_vec)
        np.testing.assert_allclose(
            [[1, 0, 1], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            buset.coefficients_mat,
        )
        np.testing.assert_allclose([1, 3, 0, 0, 0], buset.rhs_vec)
        np.testing.assert_allclose(np.zeros(3), buset.origin)

        # update the set
        buset.budget_membership_mat = [[1, 1, 0], [0, 0, 1]]
        buset.budget_rhs_vec = [3, 4]

        # check updates work
        np.testing.assert_allclose([[1, 1, 0], [0, 0, 1]], buset.budget_membership_mat)
        np.testing.assert_allclose([3, 4], buset.budget_rhs_vec)
        np.testing.assert_allclose(
            [[1, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            buset.coefficients_mat,
        )
        np.testing.assert_allclose([3, 4, 0, 0, 0], buset.rhs_vec)

        # update origin
        buset.origin = [1, 0, -1.5]
        np.testing.assert_allclose([1, 0, -1.5], buset.origin)

    def test_error_on_budget_set_dim_change(self):
        """
        BudgetSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        budget set dimension.
        """
        budget_mat = [[1, 0, 1], [0, 1, 0]]
        budget_rhs_vec = [1, 3]
        bu_set = BudgetSet(budget_mat, budget_rhs_vec)

        # error on budget incidence matrix update
        exc_str = (
            r".*must have 3 columns to match set dimension \(provided.*1 columns\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.budget_membership_mat = [[1], [1]]

        # error on origin update
        exc_str = (
            r".*must have 3 entries to match set dimension \(provided.*4 entries\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.origin = [1, 2, 1, 0]

    def test_error_on_budget_member_mat_row_change(self):
        """
        Number of rows of budget membership mat is immutable.
        Hence, size of budget_rhs_vec is also immutable.
        """
        budget_mat = [[1, 0, 1], [0, 1, 0]]
        budget_rhs_vec = [1, 3]
        bu_set = BudgetSet(budget_mat, budget_rhs_vec)

        exc_str = (
            r".*must have 2 rows to match shape of attribute 'budget_rhs_vec' "
            r"\(provided.*1 rows\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.budget_membership_mat = [[1, 0, 1]]

        exc_str = (
            r".*must have 2 entries to match shape of attribute "
            r"'budget_membership_mat' \(provided.*1 entries\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.budget_rhs_vec = [1]

    def test_error_on_neg_budget_rhs_vec_entry(self):
        """
        Test ValueError raised if budget RHS vec has entry
        with negative value entry.
        """
        budget_mat = [[1, 0, 1], [1, 1, 0]]
        neg_val_rhs_vec = [1, -1]

        exc_str = r"Entry -1 of.*'budget_rhs_vec' is negative*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(budget_mat, neg_val_rhs_vec)

        # construct a valid budget set
        buset = BudgetSet(budget_mat, [1, 1])

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_rhs_vec = neg_val_rhs_vec

    def test_error_on_non_bool_budget_mat_entry(self):
        """
        Test ValueError raised if budget membership mat has
        entry which is not a 0-1 value.
        """
        invalid_budget_mat = [[1, 0, 1], [1, 1, 0.1]]
        budget_rhs_vec = [1, 1]

        exc_str = r"Attempting.*entries.*not 0-1 values \(example: 0.1\).*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(invalid_budget_mat, budget_rhs_vec)

        # construct a valid budget set
        buset = BudgetSet([[1, 0, 1], [1, 1, 0]], budget_rhs_vec)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_membership_mat = invalid_budget_mat

    def test_error_on_budget_mat_all_zero_rows(self):
        """
        Test ValueError raised if budget membership mat
        has a row with all zeros.
        """
        invalid_row_mat = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        budget_rhs_vec = [1, 1, 2]

        exc_str = r".*all entries zero in rows at indexes: 0, 2.*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(invalid_row_mat, budget_rhs_vec)

        # construct a valid budget set
        buset = BudgetSet([[1, 0, 1], [1, 1, 0], [1, 1, 1]], budget_rhs_vec)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_membership_mat = invalid_row_mat

    def test_error_on_budget_mat_all_zero_columns(self):
        """
        Test ValueError raised if budget membership mat
        has a column with all zeros.
        """
        invalid_col_mat = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        budget_rhs_vec = [1, 1, 2]

        exc_str = r".*all entries zero in columns at indexes: 0, 1.*"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(invalid_col_mat, budget_rhs_vec)

        # construct a valid budget set
        buset = BudgetSet([[1, 0, 1], [1, 1, 0], [1, 1, 1]], budget_rhs_vec)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_membership_mat = invalid_col_mat

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_compute_parameter_bounds(self):
        """
        Test parameter bounds computations are correct.
        """
        solver = SolverFactory("baron")

        buset1 = BudgetSet([[1, 1], [0, 1]], rhs_vec=[2, 3], origin=None)
        np.testing.assert_allclose(
            buset1.parameter_bounds,
            buset1._compute_parameter_bounds(solver),
        )

        # this also checks that the list entries are tuples
        self.assertEqual(buset1.parameter_bounds, [(0, 2), (0, 2)])

        buset2 = BudgetSet([[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 2])
        self.assertEqual(
            buset2.parameter_bounds,
            buset2._compute_parameter_bounds(solver),
        )
        np.testing.assert_allclose(
            buset2.parameter_bounds,
            buset2._compute_parameter_bounds(solver),
        )
        self.assertEqual(buset2.parameter_bounds, [(1, 3), (2, 4)])

    def test_set_as_constraint(self):
        """
        Test `set_as_constraint` raises Exception if Params
        passed instead of Vars.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        m.v2 = Var(initialize=0)
        buset = BudgetSet([[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 3])

        _, uncertainty_cons, _, _ = buset.set_as_constraint(
            uncertain_params=[m.v1, m.v2],
        )
        assertExpressionsEqual(
            self,
            uncertainty_cons[0].expr,
            m.v1 + np.float64(0) * m.v2 <= np.int64(4),
        )
        assertExpressionsEqual(
            self,
            uncertainty_cons[1].expr,
            m.v1 + m.v2 <= np.int64(6),
        )
        assertExpressionsEqual(
            self,
            uncertainty_cons[2].expr,
            -np.float64(1.0) * m.v1 - np.float64(0) * m.v2 <= np.int64(-1),
        )
        assertExpressionsEqual(
            self,
            uncertainty_cons[3].expr,
            -np.float64(0) * m.v1 + np.float64(-1.0) * m.v2 <= np.int64(-3),
        )

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain parameters
        does not match the dimension.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        buset = BudgetSet([[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 3])
        with self.assertRaisesRegex(ValueError, ".*dimension"):
            buset.set_as_constraint(uncertain_params=[m.v1], block=m)

    def test_set_as_constraint_type_mismatch(self):
        """
        Check exception raised if uncertain parameter variables
        are of invalid type.
        """
        m = ConcreteModel()
        m.p1 = Param([0, 1], initialize=0, mutable=True)
        buset = BudgetSet([[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 3])
        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            buset.set_as_constraint(uncertain_params=[m.p1[0], m.p1[1]], block=m)

        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            buset.set_as_constraint(uncertain_params=m.p1, block=m)

    def test_point_in_set(self):
        """
        Test point in set checks work as expected.
        """
        buset = BudgetSet([[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 3])
        self.assertTrue(buset.point_in_set([1, 3]))
        self.assertTrue(buset.point_in_set([3, 3]))
        self.assertTrue(buset.point_in_set([2, 4]))
        self.assertFalse(buset.point_in_set([0, 0]))
        self.assertFalse(buset.point_in_set([0, 3]))
        self.assertFalse(buset.point_in_set([4, 2]))

    def test_add_bounds_on_uncertain_parameters(self):
        """
        Test method for adding bounds on uncertain params
        works as expected.
        """
        m = ConcreteModel()
        m.v = Var([0, 1], initialize=0.5)

        config = Bunch()
        config.uncertainty_set = BudgetSet(
            [[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 3]
        )
        UncertaintySet.add_bounds_on_uncertain_parameters(
            config=config,
            uncertain_param_vars=m.v,
        )

        self.assertEqual(m.v[0].bounds, (1, 3))
        self.assertEqual(m.v[1].bounds, (3, 5))


class TestFactorModelSet(unittest.TestCase):
    """
    Tests for the FactorModelSet.
    """

    def test_normal_factor_model_construction_and_update(self):
        """
        Test FactorModelSet constructor and setter work normally
        when attribute values are appropriate.
        """
        # valid inputs
        fset = FactorModelSet(
            origin=[0, 0, 1],
            number_of_factors=2,
            psi_mat=[[1, 2], [0, 1], [1, 0]],
            beta=0.1,
        )

        # check attributes are as expected
        np.testing.assert_allclose(fset.origin, [0, 0, 1])
        np.testing.assert_allclose(fset.psi_mat, [[1, 2], [0, 1], [1, 0]])
        np.testing.assert_allclose(fset.number_of_factors, 2)
        np.testing.assert_allclose(fset.beta, 0.1)
        self.assertEqual(fset.dim, 3)

        # update the set
        fset.origin = [1, 1, 0]
        fset.psi_mat = [[1, 0], [0, 1], [1, 1]]
        fset.beta = 0.5

        # check updates work
        np.testing.assert_allclose(fset.origin, [1, 1, 0])
        np.testing.assert_allclose(fset.psi_mat, [[1, 0], [0, 1], [1, 1]])
        np.testing.assert_allclose(fset.beta, 0.5)

    def test_error_on_factor_model_set_dim_change(self):
        """
        Test ValueError raised when attempting to change FactorModelSet
        dimension (by changing number of entries in origin
        or number of rows of psi_mat).
        """
        origin = [0, 0, 0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        beta = 0.5

        # construct factor model set
        fset = FactorModelSet(origin, number_of_factors, psi_mat, beta)

        # assert error on psi mat update
        exc_str = (
            r"should be of shape \(3, 2\) to match.*dimensions "
            r"\(provided shape \(2, 2\)\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            fset.psi_mat = [[1, 0], [1, 2]]

        # assert error on origin update
        exc_str = r"Attempting.*factor model set of dimension 3 to value of dimension 2"
        with self.assertRaisesRegex(ValueError, exc_str):
            fset.origin = [1, 3]

    def test_error_on_invalid_number_of_factors(self):
        """
        Test ValueError raised if number of factors
        is negative int, or AttributeError
        if attempting to update (should be immutable).
        """
        exc_str = r".*'number_of_factors' must be a positive int \(provided value -1\)"
        with self.assertRaisesRegex(ValueError, exc_str):
            FactorModelSet(origin=[0], number_of_factors=-1, psi_mat=[[1, 1]], beta=0.1)

        fset = FactorModelSet(
            origin=[0], number_of_factors=2, psi_mat=[[1, 1]], beta=0.1
        )

        exc_str = r".*'number_of_factors' is immutable"
        with self.assertRaisesRegex(AttributeError, exc_str):
            fset.number_of_factors = 3

    def test_error_on_invalid_beta(self):
        """
        Test ValueError raised if beta is invalid (exceeds 1 or
        is negative)
        """
        origin = [0, 0, 0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        neg_beta = -0.5
        big_beta = 1.5

        # assert error on construction
        neg_exc_str = (
            r".*must be a real number between 0 and 1.*\(provided value -0.5\)"
        )
        big_exc_str = r".*must be a real number between 0 and 1.*\(provided value 1.5\)"
        with self.assertRaisesRegex(ValueError, neg_exc_str):
            FactorModelSet(origin, number_of_factors, psi_mat, neg_beta)
        with self.assertRaisesRegex(ValueError, big_exc_str):
            FactorModelSet(origin, number_of_factors, psi_mat, big_beta)

        # create a valid factor model set
        fset = FactorModelSet(origin, number_of_factors, psi_mat, 1)

        # assert error on update
        with self.assertRaisesRegex(ValueError, neg_exc_str):
            fset.beta = neg_beta
        with self.assertRaisesRegex(ValueError, big_exc_str):
            fset.beta = big_beta

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_compute_parameter_bounds(self):
        """
        If LP solver is available, test parameter bounds method
        for factor model set is correct (check against
        results from an LP solver).
        """
        solver = SolverFactory("baron")

        # cases where prior parameter bounds
        # approximations were probably too tight
        fset1 = FactorModelSet(
            origin=[0, 0],
            number_of_factors=3,
            psi_mat=[[1, -1, 1], [1, 0.1, 1]],
            beta=1 / 6,
        )
        fset2 = FactorModelSet(
            origin=[0], number_of_factors=3, psi_mat=[[1, 6, 8]], beta=1 / 2
        )
        fset3 = FactorModelSet(
            origin=[1], number_of_factors=2, psi_mat=[[1, 2]], beta=1 / 4
        )
        fset4 = FactorModelSet(
            origin=[1], number_of_factors=3, psi_mat=[[-1, -6, -8]], beta=1 / 2
        )
        fset5 = FactorModelSet(
            origin=[0], number_of_factors=3, psi_mat=[[-1.5, 3, 4]], beta=7 / 9
        )

        # check against hand-calculated bounds
        self.assertEqual(fset1.parameter_bounds, [(-2.5, 2.5), (-1.4, 1.4)])
        self.assertEqual(fset2.parameter_bounds, [(-13.5, 13.5)])
        self.assertEqual(fset3.parameter_bounds, [(-0.5, 2.5)])
        self.assertEqual(fset4.parameter_bounds, [(-12.5, 14.5)])
        self.assertEqual(fset5.parameter_bounds, [(-8.5, 8.5)])

        # check parameter bounds matches LP results
        # exactly for each case
        for fset in [fset1, fset2, fset3, fset4]:
            param_bounds = fset.parameter_bounds
            solver_param_bounds = fset._compute_parameter_bounds(solver)
            np.testing.assert_allclose(
                param_bounds,
                solver_param_bounds,
                err_msg=(
                    "Parameter bounds not consistent with LP values for "
                    "FactorModelSet with parameterization:\n"
                    f"F={fset.number_of_factors},\n"
                    f"beta={fset.beta},\n"
                    f"psi_mat={fset.psi_mat},\n"
                    f"origin={fset.origin}."
                ),
            )

    def test_set_as_constraint(self):
        """
        Test `set_as_constraint` works as expected.
        """
        fset = FactorModelSet(
            origin=[1, 2],
            number_of_factors=3,
            psi_mat=[[1, -1, 1], [1, 0.1, 1]],
            beta=1 / 6,
        )
        uq = fset.set_as_constraint(uncertain_params=None)

        self.assertEqual(len(uq.auxiliary_vars), 3)
        self.assertEqual(uq.auxiliary_vars[0].bounds, (-1, 1))
        self.assertEqual(uq.auxiliary_vars[1].bounds, (-1, 1))
        self.assertEqual(uq.auxiliary_vars[2].bounds, (-1, 1))

        *factor_model_matrix_cons, betaf_abs_val_con = uq.uncertainty_cons

        self.assertEqual(len(factor_model_matrix_cons), 2)
        assertExpressionsEqual(
            self,
            factor_model_matrix_cons[0].expr,
            (
                uq.auxiliary_vars[0]
                + (-1.0) * uq.auxiliary_vars[1]
                + uq.auxiliary_vars[2]
                + 1
                == uq.uncertain_param_vars[0]
            ),
        )
        assertExpressionsEqual(
            self,
            factor_model_matrix_cons[1].expr,
            (
                uq.auxiliary_vars[0]
                + 0.1 * uq.auxiliary_vars[1]
                + uq.auxiliary_vars[2]
                + 2
                == uq.uncertain_param_vars[1]
            ),
        )

        betaf_abs_val_con = uq.uncertainty_cons[-1]
        assertExpressionsEqual(
            self,
            betaf_abs_val_con.expr,
            RangedExpression((-0.5, sum(uq.auxiliary_vars), 0.5), False),
        )

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain parameters
        does not match the dimension.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        box_set = BoxSet(bounds=[[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, ".*dimension"):
            box_set.set_as_constraint(uncertain_params=[m.v1], block=m)

    def test_set_as_constraint_type_mismatch(self):
        """
        Check exception raised if uncertain parameter variables
        are of invalid type.
        """
        m = ConcreteModel()
        m.p1 = Param([0, 1], initialize=0, mutable=True)
        box_set = BoxSet(bounds=[[1, 2], [3, 4]])
        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            box_set.set_as_constraint(uncertain_params=[m.p1[0], m.p1[1]], block=m)

        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            box_set.set_as_constraint(uncertain_params=m.p1, block=m)

    def test_point_in_set_skinny_psi_matrix(self):
        """
        Test point in set check works if psi matrix is skinny.
        """
        fset = FactorModelSet(
            origin=[0, 0, 0, 0],
            number_of_factors=3,
            psi_mat=[[1, -1, 1], [1, 0.1, 1], [2, 0.3, 1], [4, 5, 1]],
            beta=1 / 6,
        )

        self.assertTrue(fset.point_in_set(fset.origin))

        for aux_space_pt in it.permutations([1, 0.5, -1]):
            fset_pt_from_crit = fset.origin + fset.psi_mat @ aux_space_pt
            self.assertTrue(
                fset.point_in_set(fset_pt_from_crit),
                msg=(
                    f"Point {fset_pt_from_crit} generated from critical point "
                    f"{aux_space_pt} of the auxiliary variable space "
                    "is not in the set."
                ),
            )

            fset_pt_from_neg_crit = fset.origin - fset.psi_mat @ aux_space_pt
            self.assertTrue(
                fset.point_in_set(fset_pt_from_neg_crit),
                msg=(
                    f"Point {fset_pt_from_neg_crit} generated from critical point "
                    f"{aux_space_pt} of the auxiliary variable space "
                    "is not in the set."
                ),
            )

        # some points transformed from hypercube vertices.
        # since F - k = 2 < 1 = k, no such point should be in the set
        self.assertFalse(fset.point_in_set(fset.origin + fset.psi_mat @ [1, 1, 1]))
        self.assertFalse(fset.point_in_set(fset.origin + fset.psi_mat @ [1, 1, -1]))
        self.assertFalse(fset.point_in_set(fset.origin + fset.psi_mat @ [1, -1, -1]))
        self.assertFalse(fset.point_in_set(fset.origin + fset.psi_mat @ [-1, -1, -1]))

    def test_point_in_set_nonskinny_psi_matrix(self):
        """
        Test point in set check works if psi matrix is not
        skinny.
        """
        fset = FactorModelSet(
            origin=[0, 0],
            number_of_factors=3,
            psi_mat=[[1, -1, 1], [1, 0.1, 1]],
            beta=1 / 6,
        )

        self.assertTrue(fset.point_in_set(fset.origin))

        for aux_space_pt in it.permutations([1, 0.5, -1]):
            fset_pt_from_crit = fset.origin + fset.psi_mat @ aux_space_pt
            self.assertTrue(
                fset.point_in_set(fset_pt_from_crit),
                msg=(
                    f"Point {fset_pt_from_crit} generated from critical point "
                    f"{aux_space_pt} of the auxiliary variable space "
                    "is not in the set."
                ),
            )

            fset_pt_from_neg_crit = fset.origin - fset.psi_mat @ aux_space_pt
            self.assertTrue(
                fset.point_in_set(fset_pt_from_neg_crit),
                msg=(
                    f"Point {fset_pt_from_neg_crit} generated from critical point "
                    f"{aux_space_pt} of the auxiliary variable space "
                    "is not in the set."
                ),
            )

        # some points transformed from hypercube vertices.
        # no such point should be in this instance of the set
        self.assertFalse(fset.point_in_set(fset.origin + fset.psi_mat @ [1, 1, 1]))
        self.assertFalse(fset.point_in_set(fset.origin + fset.psi_mat @ [1, 1, -1]))
        self.assertFalse(fset.point_in_set(fset.origin + fset.psi_mat @ [1, -1, -1]))
        self.assertFalse(fset.point_in_set(fset.origin + fset.psi_mat @ [-1, -1, -1]))

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var([0, 1], initialize=0)

        fset = FactorModelSet(
            origin=[0, 0],
            number_of_factors=3,
            psi_mat=[[1, -1, 1], [1, 0.1, 1]],
            beta=1 / 6,
        )
        fset.add_bounds_on_uncertain_parameters(
            config=Bunch(uncertainty_set=fset),
            uncertain_param_vars=m.uncertain_param_vars,
        )

        self.assertEqual(m.uncertain_param_vars[0].bounds, (-2.5, 2.5))
        self.assertEqual(m.uncertain_param_vars[1].bounds, (-1.4, 1.4))


if __name__ == "__main__":
    unittest.main()

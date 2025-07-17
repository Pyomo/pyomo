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

"""
Tests for the PyROS UncertaintySet class and subclasses.
"""

import itertools as it
import pyomo.common.unittest as unittest

from pyomo.common.collections import Bunch
from pyomo.common.dependencies import (
    attempt_import,
    numpy as np,
    numpy_available,
    scipy as sp,
    scipy_available,
)

from pyomo.environ import SolverFactory
from pyomo.core.base import ConcreteModel, Param, Var
from pyomo.core.expr import RangedExpression
from pyomo.core.expr.compare import assertExpressionsEqual

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
    UncertaintyQuantification,
    Geometry,
    _setup_standard_uncertainty_set_constraint_block,
)

from pyomo.contrib.pyros.config import pyros_config
import time

import logging

logger = logging.getLogger(__name__)

parameterized, param_available = attempt_import('parameterized')

if not (numpy_available and scipy_available and param_available):
    raise unittest.SkipTest(
        'PyROS preprocessor unit tests require parameterized, numpy, and scipy'
    )
parameterized = parameterized.parameterized

# === Config args for testing
global_solver = 'baron'
global_solver_args = dict()

_baron = SolverFactory('baron')
baron_available = _baron.available(exception_flag=False)
if baron_available:
    baron_license_is_valid = _baron.license_is_valid()
    baron_version = _baron.version()
else:
    baron_license_is_valid = False
    baron_version = (0, 0, 0)


def bounded_and_nonempty_check(test, unc_set):
    """
    All uncertainty sets should pass these checks,
    regardless of their custom `validate` method.
    """
    CONFIG = pyros_config()
    CONFIG.global_solver = global_solver

    # check is_bounded
    test.assertTrue(unc_set.is_bounded(config=CONFIG), "Set is not bounded.")

    # check is_nonempty
    test.assertTrue(unc_set.is_nonempty(config=CONFIG), "Set is empty.")


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
        Test method for setting up constraints works correctly.
        """
        m = ConcreteModel()
        box_set = BoxSet(bounds=[[1, 2], [3, 4]])
        uq = box_set.set_as_constraint(uncertain_params=None, block=m)

        self.assertEqual(uq.auxiliary_vars, [])
        self.assertIs(uq.block, m)
        con1, con2 = uq.uncertainty_cons
        var1, var2 = uq.uncertain_param_vars

        assertExpressionsEqual(
            self, con1.expr, RangedExpression((np.int_(1), var1, np.int_(2)), False)
        )
        assertExpressionsEqual(
            self, con2.expr, RangedExpression((np.int_(3), var2, np.int_(4)), False)
        )

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain param vars
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
    def test_compute_exact_parameter_bounds(self):
        """
        Test parameter bounds computations give expected results.
        """
        box_set = BoxSet([[1, 2], [3, 4]])
        computed_bounds = box_set._compute_exact_parameter_bounds(
            SolverFactory("baron")
        )
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
                msg=f"Point {point} should not be in uncertainty set {box_set}.",
            )
        for point in out_of_set_points:
            self.assertFalse(
                box_set.point_in_set(point),
                msg=f"Point {point} should not be in uncertainty set {box_set}.",
            )

        # check what happens if dimensions are off
        with self.assertRaisesRegex(ValueError, ".*to match the set dimension.*"):
            box_set.point_in_set([1, 2, 3])

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var([0, 1], initialize=0)
        box_set = BoxSet(bounds=[(1, 2), (3, 4)])

        box_set._add_bounds_on_uncertain_parameters(
            global_solver=None, uncertain_param_vars=m.uncertain_param_vars
        )
        self.assertEqual(m.uncertain_param_vars[0].bounds, (1, 2))
        self.assertEqual(m.uncertain_param_vars[1].bounds, (3, 4))

    def test_validate(self):
        """
        Test validate performs as expected.
        """
        CONFIG = Bunch()

        # construct valid box set
        box_set = BoxSet(bounds=[[1.0, 2.0], [3.0, 4.0]])

        # validate raises no issues on valid set
        box_set.validate(config=CONFIG)

    def test_validate_finiteness(self):
        """
        Test validate finiteness check performs as expected.
        """
        CONFIG = Bunch()

        # construct valid box set
        box_set = BoxSet(bounds=[[1.0, 2.0], [3.0, 4.0]])

        # check when values are not finite
        box_set.bounds[0][0] = np.nan
        exc_str = (
            r"Entry 'nan' of the argument `bounds` " r"is not a finite numeric value"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            box_set.validate(config=CONFIG)

    def test_validate_bounds(self):
        """
        Test validate bounds check performs as expected.
        """
        CONFIG = Bunch()

        # construct valid box set
        box_set = BoxSet(bounds=[[1.0, 2.0], [3.0, 4.0]])

        # check when LB >= UB
        box_set.bounds[0][0] = 5
        exc_str = r"Lower bound 5.0 exceeds upper bound 2.0"
        with self.assertRaisesRegex(ValueError, exc_str):
            box_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_bounded_and_nonempty(self):
        """
        Test `is_bounded` and `is_nonempty` for a valid box set.
        """
        box_set = BoxSet(bounds=[[1.0, 2.0], [3.0, 4.0]])
        bounded_and_nonempty_check(self, box_set)

    def test_fbbt_error(self):
        """
        Test that `_fbbt_parameter_bounds` error message with bad bounds.
        """
        CONFIG = pyros_config()

        # construct box set with invalid bounds
        box_set = BoxSet(bounds=[[2, 1]])
        exc_str = (
            "Encountered the following exception while "
            "computing parameter bounds with FBBT"
        )
        with self.assertLogs(CONFIG.progress_logger, level='ERROR') as cm:
            box_set._fbbt_parameter_bounds(config=CONFIG)
        self.assertIn(exc_str, cm.output[0])

    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        box_set = BoxSet(bounds=[(1, 2), (1, 1), (3, 4)])
        self.assertEqual(
            box_set._is_coordinate_fixed(config=Bunch()), [False, True, False]
        )

        # test the tolerance
        box_set.bounds = [(1, 2), (2, 2 + 5e-5), (3, 4)]
        self.assertEqual(
            box_set._is_coordinate_fixed(config=Bunch()), [False, True, False]
        )
        box_set.bounds = [(1, 2), (2, 2 + 1e-4), (3, 4)]
        self.assertEqual(
            box_set._is_coordinate_fixed(config=Bunch()), [False, False, False]
        )


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

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_compute_exact_parameter_bounds(self):
        """
        Test parameter bounds computations give expected results.
        """
        solver = SolverFactory("baron")

        buset1 = BudgetSet([[1, 1], [0, 1]], rhs_vec=[2, 3], origin=None)
        np.testing.assert_allclose(
            buset1.parameter_bounds, buset1._compute_exact_parameter_bounds(solver)
        )

        # this also checks that the list entries are tuples
        self.assertEqual(buset1.parameter_bounds, [(0, 2), (0, 2)])

        buset2 = BudgetSet([[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 2])
        self.assertEqual(
            buset2.parameter_bounds, buset2._compute_exact_parameter_bounds(solver)
        )
        np.testing.assert_allclose(
            buset2.parameter_bounds, buset2._compute_exact_parameter_bounds(solver)
        )
        self.assertEqual(buset2.parameter_bounds, [(1, 3), (2, 4)])

    def test_set_as_constraint(self):
        """
        Test method for setting up constraints works correctly.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        m.v2 = Var(initialize=0)
        buset = BudgetSet([[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 3])

        uq = buset.set_as_constraint(uncertain_params=[m.v1, m.v2], block=m)
        self.assertEqual(uq.auxiliary_vars, [])
        self.assertIs(uq.block, m)
        self.assertEqual(len(uq.uncertain_param_vars), 2)
        self.assertIs(uq.uncertain_param_vars[0], m.v1)
        self.assertIs(uq.uncertain_param_vars[1], m.v2)
        self.assertEqual(len(uq.uncertainty_cons), 4)

        assertExpressionsEqual(
            self, uq.uncertainty_cons[0].expr, m.v1 + np.float64(0) * m.v2 <= np.int_(4)
        )
        assertExpressionsEqual(
            self, uq.uncertainty_cons[1].expr, m.v1 + m.v2 <= np.int_(6)
        )
        assertExpressionsEqual(
            self,
            uq.uncertainty_cons[2].expr,
            -np.float64(1.0) * m.v1 - np.float64(0) * m.v2 <= np.int_(-1),
        )
        assertExpressionsEqual(
            self,
            uq.uncertainty_cons[3].expr,
            -np.float64(0) * m.v1 + np.float64(-1.0) * m.v2 <= np.int_(-3),
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

        # check what happens if dimensions are off
        with self.assertRaisesRegex(ValueError, ".*to match the set dimension.*"):
            buset.point_in_set([1, 2, 3, 4])

    def test_add_bounds_on_uncertain_parameters(self):
        """
        Test method for adding bounds on uncertain params
        works as expected.
        """
        m = ConcreteModel()
        m.v = Var([0, 1], initialize=0.5)
        buset = BudgetSet([[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 3])
        buset._add_bounds_on_uncertain_parameters(
            global_solver=None, uncertain_param_vars=m.v
        )
        self.assertEqual(m.v[0].bounds, (1, 3))
        self.assertEqual(m.v[1].bounds, (3, 5))

    def test_validate(self):
        """
        Test validate performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid budget set
        budget_mat = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        budget_rhs_vec = [1.0, 3.0]
        budget_set = BudgetSet(budget_mat, budget_rhs_vec)

        # validate raises no issues on valid set
        budget_set.validate(config=CONFIG)

    def test_validate_finiteness(self):
        """
        Test validate finiteness check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid budget set
        budget_mat = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        budget_rhs_vec = [1.0, 3.0]
        budget_set = BudgetSet(budget_mat, budget_rhs_vec)

        # check when values are not finite
        budget_set.origin[0] = np.nan
        exc_str = (
            r"Entry 'nan' of the argument `origin` " r"is not a finite numeric value"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            budget_set.validate(config=CONFIG)
        budget_set.origin[0] = 0

        budget_set.budget_rhs_vec[0] = np.nan
        exc_str = (
            r"Entry 'nan' of the argument `budget_rhs_vec` "
            r"is not a finite numeric value"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            budget_set.validate(config=CONFIG)
        budget_set.budget_rhs_vec[0] = 1

        budget_set.budget_membership_mat[0][0] = np.nan
        exc_str = (
            r"Entry 'nan' of the argument `budget_membership_mat` "
            r"is not a finite numeric value"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            budget_set.validate(config=CONFIG)
        budget_set.budget_membership_mat[0][0] = 1

    def test_validate_rhs(self):
        """
        Test validate RHS check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid budget set
        budget_mat = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        budget_rhs_vec = [1.0, 3.0]
        budget_set = BudgetSet(budget_mat, budget_rhs_vec)

        # check when rhs has negative element
        budget_set.budget_rhs_vec = [1, -1]
        exc_str = r"Entry -1 of.*'budget_rhs_vec' is negative*"
        with self.assertRaisesRegex(ValueError, exc_str):
            budget_set.validate(config=CONFIG)
        budget_set.budget_rhs_vec = budget_rhs_vec

    def test_validate_non_bool_budget_mat_entry(self):
        """
        Test validate LHS matrix 0-1 entries check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid budget set
        budget_mat = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        budget_rhs_vec = [1.0, 3.0]
        budget_set = BudgetSet(budget_mat, budget_rhs_vec)

        # check when not all lhs entries are 0-1
        budget_set.budget_membership_mat = [[1, 0, 1], [1, 1, 0.1]]
        exc_str = r"Attempting.*entries.*not 0-1 values \(example: 0.1\).*"
        with self.assertRaisesRegex(ValueError, exc_str):
            budget_set.validate(config=CONFIG)

    def test_validate_budget_mat_all_zero_rows(self):
        """
        Test validate LHS matrix all zero row check performs as expected.
        """
        CONFIG = Bunch()

        # check when row has all zeros
        invalid_row_mat = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        budget_rhs_vec = [1, 1, 2]
        budget_set = BudgetSet(invalid_row_mat, budget_rhs_vec)
        exc_str = r".*all entries zero in rows at indexes: 0, 2.*"
        with self.assertRaisesRegex(ValueError, exc_str):
            budget_set.validate(config=CONFIG)

    def test_validate_budget_mat_all_zero_columns(self):
        """
        Test validate LHS matrix all zero column check performs as expected.
        """
        CONFIG = Bunch()

        # check when column has all zeros
        invalid_col_mat = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        budget_rhs_vec = [1, 1, 2]
        budget_set = BudgetSet(invalid_col_mat, budget_rhs_vec)
        exc_str = r".*all entries zero in columns at indexes: 0, 1.*"
        with self.assertRaisesRegex(ValueError, exc_str):
            budget_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_bounded_and_nonempty(self):
        """
        Test `is_bounded` and `is_nonempty` for a valid budget set.
        """
        budget_mat = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        budget_rhs_vec = [1.0, 3.0]
        budget_set = BudgetSet(budget_mat, budget_rhs_vec)
        bounded_and_nonempty_check(self, budget_set)

    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        buset = BudgetSet(
            origin=np.zeros(3), budget_membership_mat=np.eye(3), rhs_vec=[1, 0, 2]
        )
        self.assertEqual(
            buset._is_coordinate_fixed(config=Bunch()), [False, True, False]
        )


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
            FactorModelSet(
                origin=[0], number_of_factors=-1, psi_mat=[[1, 2], [1, 1]], beta=0.1
            )

        fset = FactorModelSet(
            origin=[0, 1], number_of_factors=2, psi_mat=[[1, 2], [1, 1]], beta=0.1
        )

        exc_str = r".*'number_of_factors' is immutable"
        with self.assertRaisesRegex(AttributeError, exc_str):
            fset.number_of_factors = 3

    @parameterized.expand(
        [
            # map beta to expected parameter bounds
            ["beta0", 0, [(-2.0, 2.0), (0.1, 1.9), (-5.0, 9.0), (-4.0, 10.0)]],
            ["beta1ov6", 1 / 6, [(-2.5, 2.5), (-0.4, 2.4), (-8.0, 12.0), (-7.0, 13.0)]],
            [
                "beta1ov3",
                1 / 3,
                [(-3.0, 3.0), (-0.9, 2.9), (-11.0, 15.0), (-10.0, 16.0)],
            ],
            [
                "beta1ov2",
                1 / 2,
                [(-3.0, 3.0), (-0.95, 2.95), (-11.5, 15.5), (-10.5, 16.5)],
            ],
            [
                "beta2ov3",
                2 / 3,
                [(-3.0, 3.0), (-1.0, 3.0), (-12.0, 16.0), (-11.0, 17.0)],
            ],
            [
                "beta7ov9",
                7 / 9,
                [
                    (-3.0, 3.0),
                    (-31 / 30, 91 / 30),
                    (-37 / 3, 49 / 3),
                    (-34 / 3, 52 / 3),
                ],
            ],
            ["beta1", 1, [(-3.0, 3.0), (-1.1, 3.1), (-13.0, 17.0), (-12.0, 18.0)]],
        ]
    )
    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_compute_exact_parameter_bounds(self, name, beta, expected_param_bounds):
        """
        Test parameter bounds computations give expected results.
        """
        solver = SolverFactory("baron")

        fset = FactorModelSet(
            origin=[0, 1, 2, 3],
            number_of_factors=3,
            psi_mat=[[1, -1, 1], [1, 0.1, 1], [-1, -6, -8], [1, 6, 8]],
            beta=beta,
        )

        param_bounds = fset.parameter_bounds
        # won't be exactly equal,
        np.testing.assert_allclose(param_bounds, expected_param_bounds, atol=1e-13)

        # check parameter bounds matches LP results
        # exactly for each case
        solver_param_bounds = fset._compute_exact_parameter_bounds(solver)
        np.testing.assert_allclose(
            solver_param_bounds,
            param_bounds,
            err_msg=(
                "Parameter bounds not consistent with LP values for "
                "FactorModelSet with parameterization:\n"
                f"F={fset.number_of_factors},\n"
                f"beta={fset.beta},\n"
                f"psi_mat={fset.psi_mat},\n"
                f"origin={fset.origin}."
            ),
            # account for solver tolerances and numerical errors
            atol=1e-4,
        )

    def test_set_as_constraint(self):
        """
        Test method for setting up constraints works correctly.
        """
        fset = FactorModelSet(
            origin=[0, 1, 2, 3],
            number_of_factors=3,
            psi_mat=[[1, -1, 1], [1, 0.1, 1], [-1, -6, -8], [1, 6, 8]],
            beta=1 / 6,
        )
        uq = fset.set_as_constraint(uncertain_params=None)

        self.assertEqual(len(uq.auxiliary_vars), 3)
        self.assertEqual(uq.auxiliary_vars[0].bounds, (-1, 1))
        self.assertEqual(uq.auxiliary_vars[1].bounds, (-1, 1))
        self.assertEqual(uq.auxiliary_vars[2].bounds, (-1, 1))

        *factor_model_matrix_cons, betaf_abs_val_con = uq.uncertainty_cons

        self.assertEqual(len(factor_model_matrix_cons), 4)
        assertExpressionsEqual(
            self,
            factor_model_matrix_cons[0].expr,
            (
                uq.auxiliary_vars[0]
                + (-1.0) * uq.auxiliary_vars[1]
                + uq.auxiliary_vars[2]
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
                + 1
                == uq.uncertain_param_vars[1]
            ),
        )
        assertExpressionsEqual(
            self,
            factor_model_matrix_cons[2].expr,
            (
                (-1.0) * uq.auxiliary_vars[0]
                + (-6.0) * uq.auxiliary_vars[1]
                + (-8.0) * uq.auxiliary_vars[2]
                + 2
                == uq.uncertain_param_vars[2]
            ),
        )
        assertExpressionsEqual(
            self,
            factor_model_matrix_cons[3].expr,
            (
                (1.0) * uq.auxiliary_vars[0]
                + (6.0) * uq.auxiliary_vars[1]
                + (8.0) * uq.auxiliary_vars[2]
                + 3
                == uq.uncertain_param_vars[3]
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

    def test_point_in_set(self):
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

        # check what happens if dimensions are off
        with self.assertRaisesRegex(ValueError, ".*to match the set dimension.*"):
            fset.point_in_set([1, 2, 3])

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var(range(4), initialize=0)
        fset = FactorModelSet(
            origin=[0, 1, 2, 3],
            number_of_factors=3,
            psi_mat=[[1, -1, 1], [1, 0.1, 1], [-1, -6, -8], [1, 6, 8]],
            beta=1,
        )

        fset._add_bounds_on_uncertain_parameters(
            global_solver=None, uncertain_param_vars=m.uncertain_param_vars
        )
        self.assertEqual(m.uncertain_param_vars[0].bounds, (-3.0, 3.0))
        self.assertEqual(m.uncertain_param_vars[1].bounds, (-1.1, 3.1))
        self.assertEqual(m.uncertain_param_vars[2].bounds, (-13.0, 17.0))
        self.assertEqual(m.uncertain_param_vars[3].bounds, (-12.0, 18.0))

    def test_validate(self):
        """
        Test validate performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid factor model set
        origin = [0.0, 0.0, 0.0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        beta = 0.5
        factor_set = FactorModelSet(origin, number_of_factors, psi_mat, beta)

        # validate raises no issues on valid set
        factor_set.validate(config=CONFIG)

    def test_validate_finiteness(self):
        """
        Test validate finiteness check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid factor model set
        origin = [0.0, 0.0, 0.0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        beta = 0.5
        factor_set = FactorModelSet(origin, number_of_factors, psi_mat, beta)

        # check when values are not finite
        factor_set.origin[0] = np.nan
        exc_str = (
            r"Entry 'nan' of the argument `origin` " r"is not a finite numeric value"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            factor_set.validate(config=CONFIG)

    def test_validate_beta(self):
        """
        Test validate beta check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid factor model set
        origin = [0.0, 0.0, 0.0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        beta = 0.5
        factor_set = FactorModelSet(origin, number_of_factors, psi_mat, beta)

        # check when beta is invalid
        neg_beta = -0.5
        big_beta = 1.5
        neg_exc_str = (
            r".*must be a real number between 0 and 1.*\(provided value -0.5\)"
        )
        big_exc_str = r".*must be a real number between 0 and 1.*\(provided value 1.5\)"
        factor_set.beta = neg_beta
        with self.assertRaisesRegex(ValueError, neg_exc_str):
            factor_set.validate(config=CONFIG)
        factor_set.beta = big_beta
        with self.assertRaisesRegex(ValueError, big_exc_str):
            factor_set.validate(config=CONFIG)

    def test_validate_psi_matrix(self):
        """
        Test validate psi matrix check performs as expected.
        """
        CONFIG = Bunch()

        # check when psi matrix is rank defficient
        with self.assertRaisesRegex(ValueError, r"full column rank.*\(2, 3\)"):
            # more columns than rows
            factor_set = FactorModelSet(
                origin=[0, 0],
                number_of_factors=3,
                psi_mat=[[1, -1, 1], [1, 0.1, 1]],
                beta=1 / 6,
            )
            factor_set.validate(config=CONFIG)
        with self.assertRaisesRegex(ValueError, r"full column rank.*\(2, 2\)"):
            # linearly dependent columns
            factor_set = FactorModelSet(
                origin=[0, 0],
                number_of_factors=2,
                psi_mat=[[1, -1], [1, -1]],
                beta=1 / 6,
            )
            factor_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_bounded_and_nonempty(self):
        """
        Test `is_bounded` and `is_nonempty` for a valid factor model set.
        """
        origin = [0.0, 0.0, 0.0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        beta = 0.5
        factor_set = FactorModelSet(origin, number_of_factors, psi_mat, beta)
        bounded_and_nonempty_check(self, factor_set)

    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        fset = FactorModelSet(
            origin=np.zeros(3),
            number_of_factors=2,
            psi_mat=[[1, 0], [1, -1], [1, 1]],
            beta=1,
        )
        self.assertEqual(
            fset._is_coordinate_fixed(config=Bunch()), [False, False, False]
        )
        fset.beta = 0
        self.assertEqual(
            fset._is_coordinate_fixed(config=Bunch()), [False, False, True]
        )


class TestIntersectionSet(unittest.TestCase):
    """
    Tests for the IntersectionSet.
    """

    def test_normal_construction_and_update(self):
        """
        Test IntersectionSet constructor and setter
        work normally when arguments are appropriate.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])

        iset = IntersectionSet(box_set=bset, axis_aligned_set=aset)
        self.assertIn(
            bset,
            iset.all_sets,
            msg=(
                "IntersectionSet 'all_sets' attribute does not"
                "contain expected BoxSet"
            ),
        )
        self.assertIn(
            aset,
            iset.all_sets,
            msg=(
                "IntersectionSet 'all_sets' attribute does not"
                "contain expected AxisAlignedEllipsoidalSet"
            ),
        )

    def test_error_on_intersecting_wrong_dims(self):
        """
        Test ValueError raised if IntersectionSet sets
        are not of same dimension.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])
        wrong_aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])

        exc_str = r".*of dimension 2, but attempting to add set of dimension 3"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            IntersectionSet(box_set=bset, axis_set=aset, wrong_set=wrong_aset)

        # construct a valid intersection set
        iset = IntersectionSet(box_set=bset, axis_set=aset)
        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            iset.all_sets.append(wrong_aset)

    def test_type_error_on_invalid_arg(self):
        """
        Test TypeError raised if an argument not of type
        UncertaintySet is passed to the IntersectionSet
        constructor or appended to 'all_sets'.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])

        exc_str = (
            r"Entry '1' of the argument `all_sets` is not An `UncertaintySet` "
            r"object.*\(provided type 'int'\)"
        )

        # assert error on construction
        with self.assertRaisesRegex(TypeError, exc_str):
            IntersectionSet(box_set=bset, axis_set=aset, invalid_arg=1)

        # construct a valid intersection set
        iset = IntersectionSet(box_set=bset, axis_set=aset)

        # assert error on update
        with self.assertRaisesRegex(TypeError, exc_str):
            iset.all_sets.append(1)

    def test_error_on_intersection_dim_change(self):
        """
        IntersectionSet dimension is considered immutable.
        Test ValueError raised when attempting to set the
        constituent sets to a different dimension.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0], [2, 2])

        # construct the set
        iset = IntersectionSet(box_set=bset, axis_set=aset)

        exc_str = r"Attempting to set.*dimension 2 to a sequence.* of dimension 1"

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            # attempt to set to 1-dimensional sets
            iset.all_sets = [BoxSet([[1, 1]]), AxisAlignedEllipsoidalSet([0], [1])]

    def test_error_on_too_few_sets(self):
        """
        Check ValueError raised if too few sets are passed
        to the intersection set.
        """
        exc_str = r"Attempting.*minimum required length 2.*iterable of length 1"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            IntersectionSet(bset=BoxSet([[1, 2]]))

        # construct a valid intersection set
        iset = IntersectionSet(
            box_set=BoxSet([[1, 2]]), axis_set=AxisAlignedEllipsoidalSet([0], [1])
        )

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            # attempt to set to 1-dimensional sets
            iset.all_sets = [BoxSet([[1, 1]])]

    def test_intersection_uncertainty_set_list_behavior(self):
        """
        Test the 'all_sets' attribute of the IntersectionSet
        class behaves like a regular Python list.
        """
        iset = IntersectionSet(
            bset=BoxSet([[0, 2]]), aset=AxisAlignedEllipsoidalSet([0], [1])
        )

        # an UncertaintySetList of length 2.
        # should behave like a list of length 2
        all_sets = iset.all_sets

        # test append
        all_sets.append(BoxSet([[1, 2]]))
        del all_sets[2:]

        # test extend
        all_sets.extend([BoxSet([[1, 2]]), EllipsoidalSet([0], [[1]], 2)])
        del all_sets[2:]

        # index in range. Allow slicing as well
        # none of these should result in exception
        all_sets[0]
        all_sets[1]
        all_sets[100:]
        all_sets[0:2:20]
        all_sets[0:2:1]
        all_sets[-20:-1:2]

        # index out of range
        self.assertRaises(IndexError, lambda: all_sets[2])
        self.assertRaises(IndexError, lambda: all_sets[-3])

        # assert min length ValueError if attempting to clear
        # list to length less than 2
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            all_sets[:] = all_sets[0]
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            del all_sets[1]
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            del all_sets[1:]
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            del all_sets[:]
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            all_sets.clear()
        with self.assertRaisesRegex(ValueError, r"Length.* must be at least 2"):
            all_sets[0:] = []

        # assignment out of range
        with self.assertRaisesRegex(IndexError, r"assignment index out of range"):
            all_sets[-3] = BoxSet([[1, 1.5]])
        with self.assertRaisesRegex(IndexError, r"assignment index out of range"):
            all_sets[2] = BoxSet([[1, 1.5]])

        # assigning to slices should work fine
        all_sets[3:] = [BoxSet([[1, 1.5]]), BoxSet([[1, 3]])]

    def test_set_as_constraint(self):
        """
        Test method for setting up constraints works correctly.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        m.v2 = Var(initialize=0)

        i_set = IntersectionSet(
            set1=BoxSet([(-0.5, 0.5), (-0.5, 0.5)]),
            set2=FactorModelSet(
                origin=[0, 0], number_of_factors=2, beta=0.75, psi_mat=[[1, 1], [1, 2]]
            ),
            set3=CardinalitySet([-0.5, -0.5], [2, 2], 2),
            # ellipsoid. this is enclosed in all the other sets
            set4=AxisAlignedEllipsoidalSet([0, 0], [0.25, 0.25]),
        )

        uq = i_set.set_as_constraint(uncertain_params=[m.v1, m.v2], block=m)

        self.assertIs(uq.block, m)
        self.assertEqual(uq.uncertain_param_vars, [m.v1, m.v2])
        self.assertEqual(len(uq.auxiliary_vars), 4)
        self.assertEqual(len(uq.uncertainty_cons), 9)

        # box set constraints
        assertExpressionsEqual(
            self,
            uq.uncertainty_cons[0].expr,
            RangedExpression((np.float64(-0.5), m.v1, np.float64(0.5)), False),
        )
        assertExpressionsEqual(
            self,
            uq.uncertainty_cons[1].expr,
            RangedExpression((np.float64(-0.5), m.v2, np.float64(0.5)), False),
        )

        # factor model constraints
        aux_vars = uq.auxiliary_vars
        assertExpressionsEqual(
            self, uq.uncertainty_cons[2].expr, aux_vars[0] + aux_vars[1] == m.v1
        )
        assertExpressionsEqual(
            self, uq.uncertainty_cons[3].expr, aux_vars[0] + 2 * aux_vars[1] == m.v2
        )
        assertExpressionsEqual(
            self,
            uq.uncertainty_cons[4].expr,
            RangedExpression((-1.5, aux_vars[0] + aux_vars[1], 1.5), False),
        )
        self.assertEqual(aux_vars[0].bounds, (-1, 1))
        self.assertEqual(aux_vars[1].bounds, (-1, 1))

        # cardinality set constraints
        assertExpressionsEqual(
            self, uq.uncertainty_cons[5].expr, -0.5 + 2 * aux_vars[2] == m.v1
        )
        assertExpressionsEqual(
            self, uq.uncertainty_cons[6].expr, -0.5 + 2 * aux_vars[3] == m.v2
        )
        assertExpressionsEqual(
            self, uq.uncertainty_cons[7].expr, sum(aux_vars[2:4]) <= 2
        )
        self.assertEqual(aux_vars[2].bounds, (0, 1))
        self.assertEqual(uq.auxiliary_vars[3].bounds, (0, 1))

        # axis-aligned ellipsoid constraint
        assertExpressionsEqual(
            self,
            uq.uncertainty_cons[8].expr,
            m.v1**2 / np.float64(0.0625) + m.v2**2 / np.float64(0.0625) <= 1,
        )

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain parameters
        does not match the dimension.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        i_set = IntersectionSet(
            set1=BoxSet(bounds=[[1, 2], [3, 4]]),
            set2=AxisAlignedEllipsoidalSet([0, 1], [5, 5]),
        )
        with self.assertRaisesRegex(ValueError, ".*dimension"):
            i_set.set_as_constraint(uncertain_params=[m.v1], block=m)

    def test_set_as_constraint_type_mismatch(self):
        """
        Check exception raised if uncertain parameter variables
        are of invalid type.
        """
        m = ConcreteModel()
        m.p1 = Param([0, 1], initialize=0, mutable=True)
        i_set = IntersectionSet(
            set1=BoxSet(bounds=[[1, 2], [3, 4]]),
            set2=AxisAlignedEllipsoidalSet([0, 1], [5, 5]),
        )
        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            i_set.set_as_constraint(uncertain_params=[m.p1[0], m.p1[1]], block=m)

        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            i_set.set_as_constraint(uncertain_params=m.p1, block=m)

    @unittest.skipUnless(baron_available, "BARON is not available.")
    def test_compute_exact_parameter_bounds(self):
        """
        Test parameter bounds computations give expected results.
        """
        i_set = IntersectionSet(
            set1=BoxSet([(-0.5, 0.5), (-0.5, 0.5)]),
            set2=FactorModelSet(
                origin=[0, 0], number_of_factors=2, beta=0.75, psi_mat=[[1, 1], [1, 2]]
            ),
            # another origin-centered square
            set3=CardinalitySet([-0.5, -0.5], [2, 2], 2),
            # ellipsoid. this is enclosed in all the other sets
            set4=AxisAlignedEllipsoidalSet([0, 0], [0.25, 0.25]),
        )

        # ellipsoid is enclosed by everyone else, so
        # that determines the bounds
        computed_bounds = i_set._compute_exact_parameter_bounds(SolverFactory("baron"))
        np.testing.assert_allclose(computed_bounds, [[-0.25, 0.25], [-0.25, 0.25]])

        # returns empty list
        self.assertFalse(i_set.parameter_bounds)

    def test_point_in_set(self):
        """
        Test point in set check for intersection set.
        """
        i_set = IntersectionSet(
            set1=BoxSet([(-0.5, 0.5), (-0.5, 0.5)]),
            # this is just an origin-centered square
            set2=FactorModelSet(
                origin=[0, 0], number_of_factors=2, beta=0.75, psi_mat=[[1, 1], [1, 2]]
            ),
            set3=CardinalitySet([-0.5, -0.5], [2, 2], 2),
            # ellipsoid. this is enclosed in all the other sets
            set4=AxisAlignedEllipsoidalSet([0, 0], [0.25, 0.25]),
        )

        # ellipsoid points
        self.assertTrue(i_set.point_in_set([0, 0]))
        self.assertTrue(i_set.point_in_set([0, 0.25]))
        self.assertTrue(i_set.point_in_set([0, -0.25]))
        self.assertTrue(i_set.point_in_set([0.25, 0]))
        self.assertTrue(i_set.point_in_set([-0.25, 0]))

        # box vertex
        self.assertFalse(i_set.point_in_set([0.5, 0.5]))
        # cardinality set origin and vertex of the box
        # are outside the ellipse
        self.assertFalse(i_set.point_in_set([-0.5, -0.5]))

    @unittest.skipUnless(baron_available, "Global NLP solver is not available.")
    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var([0, 1], initialize=0)
        iset = IntersectionSet(
            set1=BoxSet([(-0.5, 0.5), (-0.5, 0.5)]),
            set2=FactorModelSet(
                origin=[0, 0], number_of_factors=2, beta=0.75, psi_mat=[[1, 1], [1, 2]]
            ),
            set3=CardinalitySet([-0.5, -0.5], [2, 2], 2),
            # ellipsoid. this is enclosed in all the other sets
            set4=AxisAlignedEllipsoidalSet([0, 0], [0.25, 0.25]),
        )

        iset._add_bounds_on_uncertain_parameters(
            global_solver=SolverFactory("baron"),
            uncertain_param_vars=m.uncertain_param_vars,
        )

        # account for imprecision
        np.testing.assert_allclose(m.uncertain_param_vars[0].bounds, (-0.25, 0.25))
        np.testing.assert_allclose(m.uncertain_param_vars[1].bounds, (-0.25, 0.25))

        # check what happens if dimensions are off
        with self.assertRaisesRegex(ValueError, ".*to match the set dimension.*"):
            iset.point_in_set([1, 2, 3])

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_validate(self):
        """
        Test validate checks perform as expected.
        """
        CONFIG = pyros_config()
        CONFIG.global_solver = global_solver

        # construct a valid intersection set
        bset = BoxSet(bounds=[[-1, 1], [-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])
        intersection_set = IntersectionSet(box_set=bset, axis_aligned_set=aset)

        # validate raises no issues on valid set
        intersection_set.validate(config=CONFIG)

        # check when individual sets fail validation method
        bset = BoxSet(bounds=[[-1, 1], [-1, 1], [-1, 1]])
        bset.bounds[0][0] = 2
        aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])
        intersection_set = IntersectionSet(box_set=bset, axis_aligned_set=aset)
        exc_str = r"Lower bound 2 exceeds upper bound 1"
        with self.assertRaisesRegex(ValueError, exc_str):
            intersection_set.validate(config=CONFIG)

        # check when individual sets are not actually intersecting
        bset1 = BoxSet(bounds=[[1, 2], [1, 2]])
        bset2 = BoxSet(bounds=[[-2, -1], [-2, -1]])
        intersection_set = IntersectionSet(box_set1=bset1, box_set2=bset2)
        exc_str = r"Could not compute.*bound in dimension.*Solver status summary:.*"
        with self.assertRaisesRegex(ValueError, exc_str):
            intersection_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_bounded_and_nonempty(self):
        """
        Test `is_bounded` and `is_nonempty` for a valid intersection set.
        """
        bset = BoxSet(bounds=[[-1, 1], [-1, 1], [-1, 1]])
        aset = AxisAlignedEllipsoidalSet([0, 0, 0], [1, 1, 1])
        intersection_set = IntersectionSet(box_set=bset, axis_aligned_set=aset)
        bounded_and_nonempty_check(self, intersection_set)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        iset = IntersectionSet(
            set1=BoxSet(bounds=[[0, 1], [0, 1]]), set2=BoxSet(bounds=[[1, 2], [0, 1]])
        )
        baron = SolverFactory("baron")
        self.assertEqual(
            iset._is_coordinate_fixed(config=Bunch(global_solver=baron)), [True, False]
        )


class TestCardinalitySet(unittest.TestCase):
    """
    Tests for the CardinalitySet.
    """

    def test_normal_cardinality_construction_and_update(self):
        """
        Test CardinalitySet constructor and setter work normally
        when bounds are appropriate.
        """
        # valid inputs
        cset = CardinalitySet(origin=[0, 0], positive_deviation=[1, 3], gamma=2)

        # check attributes are as expected
        np.testing.assert_allclose(cset.origin, [0, 0])
        np.testing.assert_allclose(cset.positive_deviation, [1, 3])
        np.testing.assert_allclose(cset.gamma, 2)
        self.assertEqual(cset.dim, 2)

        # update the set
        cset.origin = [1, 2]
        cset.positive_deviation = [3, 0]
        cset.gamma = 0.5

        # check updates work
        np.testing.assert_allclose(cset.origin, [1, 2])
        np.testing.assert_allclose(cset.positive_deviation, [3, 0])
        np.testing.assert_allclose(cset.gamma, 0.5)

    def test_error_on_cardinality_set_dim_change(self):
        """
        Dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        set dimension (i.e. number of entries of `origin`).
        """
        # construct a valid cardinality set
        cset = CardinalitySet(origin=[0, 0], positive_deviation=[1, 1], gamma=2)

        exc_str = r"Attempting to set.*dimension 2 to value of dimension 3"

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.origin = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.positive_deviation = [1, 1, 1]

    def test_set_as_constraint(self):
        """
        Test method for setting up constraints works correctly.
        """
        m = ConcreteModel()
        cset = CardinalitySet([-0.5, 1, 2], [2.5, 3, 0], 1.5)
        uq = cset.set_as_constraint(uncertain_params=None, block=m)

        self.assertEqual(len(uq.uncertainty_cons), 4)
        self.assertEqual(len(uq.auxiliary_vars), 3)
        self.assertEqual(len(uq.uncertain_param_vars), 3)
        self.assertIs(uq.block, m)

        *hadamard_cons, gamma_con = uq.uncertainty_cons
        var1, var2, var3 = uq.uncertain_param_vars
        auxvar1, auxvar2, auxvar3 = uq.auxiliary_vars

        assertExpressionsEqual(
            self, hadamard_cons[0].expr, -0.5 + 2.5 * auxvar1 == var1
        )
        assertExpressionsEqual(self, hadamard_cons[1].expr, 1.0 + 3.0 * auxvar2 == var2)
        assertExpressionsEqual(self, hadamard_cons[2].expr, 2.0 + 0.0 * auxvar3 == var3)
        assertExpressionsEqual(self, gamma_con.expr, auxvar1 + auxvar2 + auxvar3 <= 1.5)

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain parameters
        does not match the dimension.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        cset = CardinalitySet([-0.5, 1, 2], [2.5, 3, 0], 1.5)
        with self.assertRaisesRegex(ValueError, ".*dimension"):
            cset.set_as_constraint(uncertain_params=[m.v1], block=m)

    def test_set_as_constraint_type_mismatch(self):
        """
        Check exception raised if uncertain parameter variables
        are of invalid type.
        """
        m = ConcreteModel()
        m.p1 = Param([0, 1, 2], initialize=0, mutable=True)
        cset = CardinalitySet([-0.5, 1, 2], [2.5, 3, 0], 1.5)
        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            cset.set_as_constraint(uncertain_params=[m.p1[0], m.p1[1]], block=m)

        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            cset.set_as_constraint(uncertain_params=m.p1, block=m)

    def test_point_in_set(self):
        cset = CardinalitySet(
            origin=[-0.5, 1, 2], positive_deviation=[2.5, 3, 0], gamma=1.5
        )

        self.assertTrue(cset.point_in_set(cset.origin))

        # first param full deviation
        self.assertTrue(cset.point_in_set([-0.5, 4, 2]))
        # second param full deviation
        self.assertTrue(cset.point_in_set([2, 1, 2]))
        # one and a half deviations (max)
        self.assertTrue(cset.point_in_set([2, 2.5, 2]))

        # over one and a half deviations; out of set
        self.assertFalse(cset.point_in_set([2.05, 2.5, 2]))
        self.assertFalse(cset.point_in_set([2, 2.55, 2]))

        # deviation in dimension that has been fixed
        self.assertFalse(cset.point_in_set([-0.25, 4, 2.01]))

        # check what happens if dimensions are off
        with self.assertRaisesRegex(ValueError, ".*to match the set dimension.*"):
            cset.point_in_set([1, 2, 3, 4])

    @unittest.skipUnless(baron_available, "BARON is not available.")
    def test_compute_exact_parameter_bounds(self):
        """
        Test parameter bounds computations give expected results.
        """
        cset = CardinalitySet(
            origin=[-0.5, 1, 2], positive_deviation=[2.5, 3, 0], gamma=1.5
        )
        computed_bounds = cset._compute_exact_parameter_bounds(SolverFactory("baron"))
        np.testing.assert_allclose(computed_bounds, [[-0.5, 2], [1, 4], [2, 2]])
        np.testing.assert_allclose(computed_bounds, cset.parameter_bounds)

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var([0, 1, 2], initialize=0)
        cset = CardinalitySet(
            origin=[-0.5, 1, 2], positive_deviation=[2.5, 3, 0], gamma=1.5
        )

        cset._add_bounds_on_uncertain_parameters(
            global_solver=None, uncertain_param_vars=m.uncertain_param_vars
        )
        self.assertEqual(m.uncertain_param_vars[0].bounds, (-0.5, 2))
        self.assertEqual(m.uncertain_param_vars[1].bounds, (1, 4))
        self.assertEqual(m.uncertain_param_vars[2].bounds, (2, 2))

    def test_validate(self):
        """
        Test validate performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid cardinality set
        cardinality_set = CardinalitySet(
            origin=[0.0, 0.0], positive_deviation=[1.0, 1.0], gamma=2
        )

        # validate raises no issues on valid set
        cardinality_set.validate(config=CONFIG)

    def test_validate_finiteness(self):
        """
        Test validate finiteness check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid cardinality set
        cardinality_set = CardinalitySet(
            origin=[0.0, 0.0], positive_deviation=[1.0, 1.0], gamma=2
        )

        # check when values are not finite
        cardinality_set.origin[0] = np.nan
        exc_str = r"Entry 'nan' of the argument `origin` is not a finite numeric value"
        with self.assertRaisesRegex(ValueError, exc_str):
            cardinality_set.validate(config=CONFIG)

        cardinality_set.origin[0] = 0
        cardinality_set.positive_deviation[0] = np.nan
        exc_str = (
            r"Entry 'nan' of the argument `positive_deviation` "
            r"is not a finite numeric value"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            cardinality_set.validate(config=CONFIG)

    def test_validate_pos_deviation(self):
        """
        Test validate positive deviation check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid cardinality set
        cardinality_set = CardinalitySet(
            origin=[0.0, 0.0], positive_deviation=[1.0, 1.0], gamma=2
        )

        # check when deviation is negative
        cardinality_set.positive_deviation[0] = -2
        exc_str = r"Entry -2.0 of attribute 'positive_deviation' is negative value"
        with self.assertRaisesRegex(ValueError, exc_str):
            cardinality_set.validate(config=CONFIG)

    def test_validate_gamma(self):
        """
        Test validate gamma check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid cardinality set
        cardinality_set = CardinalitySet(
            origin=[0.0, 0.0], positive_deviation=[1.0, 1.0], gamma=2
        )

        # check when gamma is invalid
        cardinality_set.gamma = 3
        exc_str = (
            r".*attribute 'gamma' must be a real number "
            r"between 0 and dimension 2 \(provided value 3\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            cardinality_set.validate(config=CONFIG)

        cardinality_set.gamma = -1
        exc_str = (
            r".*attribute 'gamma' must be a real number "
            r"between 0 and dimension 2 \(provided value -1\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            cardinality_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_bounded_and_nonempty(self):
        """
        Test `is_bounded` and `is_nonempty` for a valid cardinality set.
        """
        cardinality_set = CardinalitySet(
            origin=[0, 0], positive_deviation=[1, 1], gamma=2
        )
        bounded_and_nonempty_check(self, cardinality_set)

    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        cset = CardinalitySet(origin=np.zeros(3), positive_deviation=[1, 1, 0], gamma=1)
        self.assertEqual(
            cset._is_coordinate_fixed(config=Bunch()), [False, False, True]
        )


class TestDiscreteScenarioSet(unittest.TestCase):
    """
    Tests for the DiscreteScenarioSet.
    """

    def test_normal_discrete_set_construction_and_update(self):
        """
        Test DiscreteScenarioSet constructor and setter work normally
        when scenarios are appropriate.
        """
        scenarios = [[0, 0, 0], [1, 2, 3]]

        # normal construction should work
        dset = DiscreteScenarioSet(scenarios)

        # check scenarios added appropriately
        np.testing.assert_allclose(scenarios, dset.scenarios)

        # check scenarios updated appropriately
        new_scenarios = [[0, 1, 2], [1, 2, 0], [3, 5, 4]]
        dset.scenarios = new_scenarios
        np.testing.assert_allclose(new_scenarios, dset.scenarios)

    def test_error_on_discrete_set_dim_change(self):
        """
        Test ValueError raised when attempting to update
        DiscreteScenarioSet dimension.
        """
        scenarios = [[1, 2], [3, 4]]
        dset = DiscreteScenarioSet(scenarios)  # 2-dimensional set

        exc_str = (
            r".*must have 2 columns.* to match set dimension "
            r"\(provided.*with 3 columns\)"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            dset.scenarios = [[1, 2, 3], [4, 5, 6]]

    def test_set_as_constraint(self):
        """
        Test method for setting up constraints works correctly.
        """
        m = ConcreteModel()
        m.v1 = Var([0, 1], initialize=0)
        dset = DiscreteScenarioSet([[1, 2], [3, 4]])
        uq = dset.set_as_constraint(block=m, uncertain_params=m.v1)
        self.assertEqual(uq.uncertain_param_vars, [m.v1[0], m.v1[1]])
        self.assertEqual(uq.uncertainty_cons, [])
        self.assertEqual(uq.auxiliary_vars, [])
        self.assertIs(uq.block, m)

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain parameters
        does not match the dimension.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        dset = DiscreteScenarioSet([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, ".*dimension"):
            dset.set_as_constraint(uncertain_params=[m.v1], block=m)

    def test_set_as_constraint_type_mismatch(self):
        """
        Check exception raised if uncertain parameter variables
        are of invalid type.
        """
        m = ConcreteModel()
        m.p1 = Param([0, 1], initialize=0, mutable=True)
        dset = DiscreteScenarioSet([[1, 2], [3, 4]])
        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            dset.set_as_constraint(uncertain_params=[m.p1[0], m.p1[1]], block=m)

        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            dset.set_as_constraint(uncertain_params=m.p1, block=m)

    def test_point_in_set(self):
        dset = DiscreteScenarioSet([(0, 0), (1.5, 0), (0, 1), (1, 1), (2, 0)])
        self.assertTrue(dset.point_in_set([0, 0]))
        self.assertTrue(dset.point_in_set([1.5, 0]))
        self.assertTrue(dset.point_in_set([0, 1.0]))
        self.assertTrue(dset.point_in_set([1, 1.0]))
        self.assertTrue(dset.point_in_set([2, 0]))
        self.assertFalse(dset.point_in_set([2, 2]))

        # check precision: slight deviations from (0, 0)
        self.assertTrue(dset.point_in_set([4.9e-9, 4.9e-9]))
        self.assertFalse(dset.point_in_set([5.1e-9, 5.1e-9]))
        self.assertFalse(dset.point_in_set([1e-7, 1e-7]))

        # check what happens if dimensions are off
        with self.assertRaisesRegex(ValueError, ".*to match the set dimension.*"):
            dset.point_in_set([1, 2, 3])

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var([0, 1], initialize=0)
        dset = DiscreteScenarioSet([(0, 0), (1.5, 0), (0, 1), (1, 1), (2, 0)])

        dset._add_bounds_on_uncertain_parameters(
            global_solver=None, uncertain_param_vars=m.uncertain_param_vars
        )
        self.assertEqual(m.uncertain_param_vars[0].bounds, (0, 2))
        self.assertEqual(m.uncertain_param_vars[1].bounds, (0, 1.0))

    def test_validate(self):
        """
        Test validate performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid discrete scenario set
        discrete_set = DiscreteScenarioSet([[1, 2], [3, 4]])

        # validate raises no issues on valid set
        discrete_set.validate(config=CONFIG)

    def test_validate_finiteness(self):
        """
        Test validate finiteness check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid discrete scenario set
        discrete_set = DiscreteScenarioSet([[1, 2], [3, 4]])

        # validate raises no issues on valid set
        discrete_set.validate(config=CONFIG)

        # check when not all scenarios are finite
        discrete_set = DiscreteScenarioSet([[1, 2], [3, 4]])
        for val_str in ["inf", "nan"]:
            exc_str = (
                fr"Entry '{val_str}' of the argument `scenarios` "
                r"is not a finite numeric value"
            )
            discrete_set.scenarios[0] = [1, float(val_str)]
            with self.assertRaisesRegex(ValueError, exc_str):
                discrete_set.validate(config=CONFIG)

    def test_validate_nonemptiness(self):
        """
        Test validate finiteness check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid discrete scenario set
        discrete_set = DiscreteScenarioSet([[1, 2], [3, 4]])

        # validate raises no issues on valid set
        discrete_set.validate(config=CONFIG)

        # check when scenario set is empty
        discrete_set = DiscreteScenarioSet([[0]])
        discrete_set.scenarios.pop(0)  # remove initial scenario
        discrete_set.scenarios.append([])  # add empty scenario
        exc_str = r".* argument `scenarios` must be non-empty"
        with self.assertRaisesRegex(ValueError, exc_str):
            discrete_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_bounded_and_nonempty(self):
        """
        Test `is_bounded` and `is_nonempty` for a valid
        discrete scenario set.
        """
        discrete_set = DiscreteScenarioSet([[1, 2], [3, 4]])
        bounded_and_nonempty_check(self, discrete_set)

    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        dset = DiscreteScenarioSet(scenarios=[[0, 0, 0], [0, 1, 2], [0, 3, 4]])
        self.assertEqual(
            dset._is_coordinate_fixed(config=Bunch()), [True, False, False]
        )


class TestAxisAlignedEllipsoidalSet(unittest.TestCase):
    """
    Tests for the AxisAlignedEllipsoidalSet.
    """

    def test_normal_construction_and_update(self):
        """
        Test AxisAlignedEllipsoidalSet constructor and setter
        work normally when bounds are appropriate.
        """
        center = [0, 0]
        half_lengths = [1, 3]
        aset = AxisAlignedEllipsoidalSet(center, half_lengths)
        np.testing.assert_allclose(
            center,
            aset.center,
            err_msg="AxisAlignedEllipsoidalSet center not as expected",
        )
        np.testing.assert_allclose(
            half_lengths,
            aset.half_lengths,
            err_msg="AxisAlignedEllipsoidalSet half-lengths not as expected",
        )

        # check attributes update
        new_center = [-1, -3]
        new_half_lengths = [0, 1]
        aset.center = new_center
        aset.half_lengths = new_half_lengths

        np.testing.assert_allclose(
            new_center,
            aset.center,
            err_msg="AxisAlignedEllipsoidalSet center update not as expected",
        )
        np.testing.assert_allclose(
            new_half_lengths,
            aset.half_lengths,
            err_msg=("AxisAlignedEllipsoidalSet half lengths update not as expected"),
        )

    def test_error_on_axis_aligned_dim_change(self):
        """
        AxisAlignedEllipsoidalSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        box set dimension (i.e. number of rows of `bounds`).
        """
        center = [0, 0]
        half_lengths = [1, 3]
        aset = AxisAlignedEllipsoidalSet(center, half_lengths)

        exc_str = r"Attempting to set.*dimension 2 to value of dimension 3"
        with self.assertRaisesRegex(ValueError, exc_str):
            aset.center = [0, 0, 1]

        with self.assertRaisesRegex(ValueError, exc_str):
            aset.half_lengths = [0, 0, 1]

    def test_set_as_constraint(self):
        """
        Test method for setting up constraints works correctly.
        """
        m = ConcreteModel()
        m.v = Var([0, 1, 2])
        aeset = AxisAlignedEllipsoidalSet(center=[0, 1.5, 1], half_lengths=[1.5, 2, 0])
        uq = aeset.set_as_constraint(uncertain_params=m.v, block=m)

        self.assertEqual(len(uq.uncertainty_cons), 2)
        self.assertEqual(len(uq.uncertain_param_vars), 3)
        self.assertEqual(uq.auxiliary_vars, [])
        self.assertIs(uq.block, m)

        con1, con2 = uq.uncertainty_cons

        assertExpressionsEqual(self, con1.expr, m.v[2] == np.float64(1.0))
        assertExpressionsEqual(
            self,
            con2.expr,
            m.v[0] ** 2 / np.float64(2.25)
            + (m.v[1] - np.float64(1.5)) ** 2 / np.float64(4)
            <= 1,
        )

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain parameters
        does not match the dimension.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        aeset = AxisAlignedEllipsoidalSet(center=[0, 1.5, 1], half_lengths=[1.5, 2, 0])
        with self.assertRaisesRegex(ValueError, ".*dimension"):
            aeset.set_as_constraint(uncertain_params=[m.v1], block=m)

    def test_set_as_constraint_type_mismatch(self):
        """
        Check exception raised if uncertain parameter variables
        are of invalid type.
        """
        m = ConcreteModel()
        m.p1 = Param([0, 1, 2], initialize=0, mutable=True)
        aeset = AxisAlignedEllipsoidalSet(center=[0, 1.5, 1], half_lengths=[1.5, 2, 0])
        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            aeset.set_as_constraint(uncertain_params=[m.p1[0], m.p1[1]], block=m)

        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            aeset.set_as_constraint(uncertain_params=m.p1, block=m)

    @unittest.skipUnless(baron_available, "BARON is not available.")
    def test_compute_exact_parameter_bounds(self):
        """
        Test parameter bounds computations give expected results.
        """
        aeset = AxisAlignedEllipsoidalSet(center=[0, 1.5, 1], half_lengths=[1.5, 2, 0])
        computed_bounds = aeset._compute_exact_parameter_bounds(SolverFactory("baron"))
        np.testing.assert_allclose(computed_bounds, [[-1.5, 1.5], [-0.5, 3.5], [1, 1]])
        np.testing.assert_allclose(computed_bounds, aeset.parameter_bounds)

    def test_point_in_set(self):
        aeset = AxisAlignedEllipsoidalSet(center=[0, 0, 1], half_lengths=[1.5, 2, 0])

        self.assertTrue(aeset.point_in_set([0, 0, 1]))
        self.assertTrue(aeset.point_in_set([0, 2, 1]))
        self.assertTrue(aeset.point_in_set([0, -2, 1]))
        self.assertTrue(aeset.point_in_set([1.5, 0, 1]))
        self.assertTrue(aeset.point_in_set([-1.5, 0, 1]))
        self.assertFalse(aeset.point_in_set([0, 0, 1.05]))
        self.assertFalse(aeset.point_in_set([1.505, 0, 1]))
        self.assertFalse(aeset.point_in_set([0, 2.05, 1]))

        # check what happens if dimensions are off
        with self.assertRaisesRegex(ValueError, ".*to match the set dimension.*"):
            aeset.point_in_set([1, 2, 3, 4])

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var([0, 1, 2], initialize=0)
        aeset = AxisAlignedEllipsoidalSet(center=[0, 1.5, 1], half_lengths=[1.5, 2, 0])
        aeset._add_bounds_on_uncertain_parameters(
            global_solver=None, uncertain_param_vars=m.uncertain_param_vars
        )
        self.assertEqual(m.uncertain_param_vars[0].bounds, (-1.5, 1.5))
        self.assertEqual(m.uncertain_param_vars[1].bounds, (-0.5, 3.5))
        self.assertEqual(m.uncertain_param_vars[2].bounds, (1, 1))

    def test_validate(self):
        """
        Test validate performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid axis aligned ellipsoidal set
        center = [0.0, 0.0]
        half_lengths = [1.0, 3.0]
        a_ellipsoid_set = AxisAlignedEllipsoidalSet(center, half_lengths)

        # validate raises no issues on valid set
        a_ellipsoid_set.validate(config=CONFIG)

    def test_validate_finiteness(self):
        """
        Test validate finiteness check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid axis aligned ellipsoidal set
        center = [0.0, 0.0]
        half_lengths = [1.0, 3.0]
        a_ellipsoid_set = AxisAlignedEllipsoidalSet(center, half_lengths)

        # check when values are not finite
        a_ellipsoid_set.center[0] = np.nan
        exc_str = r"Entry 'nan' of the argument `center` is not a finite numeric value"
        with self.assertRaisesRegex(ValueError, exc_str):
            a_ellipsoid_set.validate(config=CONFIG)
        a_ellipsoid_set.center[0] = 0

        a_ellipsoid_set.half_lengths[0] = np.nan
        exc_str = (
            r"Entry 'nan' of the argument `half_lengths` "
            r"is not a finite numeric value"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            a_ellipsoid_set.validate(config=CONFIG)
        a_ellipsoid_set.half_lengths[0] = 1

    def test_validate_half_length(self):
        """
        Test validate half-lengths check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid axis aligned ellipsoidal set
        center = [0.0, 0.0]
        half_lengths = [1.0, 3.0]
        a_ellipsoid_set = AxisAlignedEllipsoidalSet(center, half_lengths)

        # check when half lengths are negative
        a_ellipsoid_set.half_lengths = [1, -1]
        exc_str = r"Entry -1 of.*'half_lengths' is negative.*"
        with self.assertRaisesRegex(ValueError, exc_str):
            a_ellipsoid_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_bounded_and_nonempty(self):
        """
        Test `is_bounded` and `is_nonempty` for a valid
        axis aligned ellipsoidal set.
        """
        center = [0.0, 0.0]
        half_lengths = [1.0, 3.0]
        a_ellipsoid_set = AxisAlignedEllipsoidalSet(center, half_lengths)
        bounded_and_nonempty_check(self, a_ellipsoid_set)

    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        aeset = AxisAlignedEllipsoidalSet(center=np.zeros(3), half_lengths=[1, 2, 0])
        self.assertEqual(
            aeset._is_coordinate_fixed(config=Bunch()), [False, False, True]
        )


class TestEllipsoidalSet(unittest.TestCase):
    """
    Tests for the EllipsoidalSet.
    """

    def test_normal_construction_and_update(self):
        """
        Test EllipsoidalSet constructor and setter
        work normally when arguments are appropriate.
        """
        center = [0, 0]
        shape_matrix = [[1, 0], [0, 2]]
        scale = 2
        eset = EllipsoidalSet(center, shape_matrix, scale)
        np.testing.assert_allclose(
            center, eset.center, err_msg="EllipsoidalSet center not as expected"
        )
        np.testing.assert_allclose(
            shape_matrix,
            eset.shape_matrix,
            err_msg="EllipsoidalSet shape matrix not as expected",
        )
        np.testing.assert_allclose(
            scale, eset.scale, err_msg="EllipsoidalSet scale not as expected"
        )
        np.testing.assert_allclose(
            # evaluate chisquare CDF for 2 degrees of freedom
            # using simplified formula
            1 - np.exp(-scale / 2),
            eset.gaussian_conf_lvl,
            err_msg="EllipsoidalSet Gaussian confidence level not as expected",
        )

        # check attributes update
        new_center = [-1, -3]
        new_shape_matrix = [[2, 1], [1, 3]]
        new_scale = 1

        eset.center = new_center
        eset.shape_matrix = new_shape_matrix
        eset.scale = new_scale

        np.testing.assert_allclose(
            new_center,
            eset.center,
            err_msg="EllipsoidalSet center update not as expected",
        )
        np.testing.assert_allclose(
            new_shape_matrix,
            eset.shape_matrix,
            err_msg="EllipsoidalSet shape matrix update not as expected",
        )
        np.testing.assert_allclose(
            new_scale, eset.scale, err_msg="EllipsoidalSet scale update not as expected"
        )
        np.testing.assert_allclose(
            # evaluate chisquare CDF for 2 degrees of freedom
            # using simplified formula
            1 - np.exp(-new_scale / 2),
            eset.gaussian_conf_lvl,
            err_msg="EllipsoidalSet Gaussian confidence level update not as expected",
        )

    def test_normal_construction_and_update_gaussian_conf_lvl(self):
        """
        Test EllipsoidalSet constructor and setter
        work normally when arguments are appropriate.
        """
        init_conf_lvl = 0.95
        eset = EllipsoidalSet(
            center=[0, 0, 0],
            shape_matrix=np.eye(3),
            scale=None,
            gaussian_conf_lvl=init_conf_lvl,
        )

        self.assertEqual(eset.gaussian_conf_lvl, init_conf_lvl)
        np.testing.assert_allclose(
            sp.stats.chi2.isf(q=1 - init_conf_lvl, df=eset.dim),
            eset.scale,
            err_msg="EllipsoidalSet scale not as expected",
        )

        new_conf_lvl = 0.99
        eset.gaussian_conf_lvl = new_conf_lvl
        self.assertEqual(eset.gaussian_conf_lvl, new_conf_lvl)
        np.testing.assert_allclose(
            sp.stats.chi2.isf(q=1 - new_conf_lvl, df=eset.dim),
            eset.scale,
            err_msg="EllipsoidalSet scale not as expected",
        )

    def test_error_on_ellipsoidal_dim_change(self):
        """
        EllipsoidalSet dimension is considered immutable.
        Test ValueError raised when center size is not equal
        to set dimension.
        """
        shape_matrix = [[1, 0], [0, 1]]
        scale = 2

        eset = EllipsoidalSet([0, 0], shape_matrix, scale)

        exc_str = r"Attempting to set.*dimension 2 to value of dimension 3"

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.center = [0, 0, 0]

    def test_error_invalid_gaussian_conf_lvl(self):
        """
        Test error when attempting to initialize with Gaussian
        confidence level outside range.
        """
        center = [0, 0]
        shape_matrix = [[1, 0], [0, 2]]
        invalid_conf_lvl = 1.001

        exc_str = r"Ensure the confidence level is a value in \[0, 1\)."

        # error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            EllipsoidalSet(
                center=center,
                shape_matrix=shape_matrix,
                scale=None,
                gaussian_conf_lvl=invalid_conf_lvl,
            )

        # error on updating valid ellipsoid
        eset = EllipsoidalSet(center, shape_matrix, scale=None, gaussian_conf_lvl=0.95)
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.gaussian_conf_lvl = invalid_conf_lvl

        # negative confidence level
        eset = EllipsoidalSet(center, shape_matrix, scale=None, gaussian_conf_lvl=0.95)
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.gaussian_conf_lvl = -0.1

    def test_error_scale_gaussian_conf_lvl_construction(self):
        """
        Test exception raised if neither or both of
        `scale` and `gaussian_conf_lvl` are None.
        """
        exc_str = r"Exactly one of `scale` and `gaussian_conf_lvl` should be None"
        with self.assertRaisesRegex(ValueError, exc_str):
            EllipsoidalSet([0], [[1]], scale=None, gaussian_conf_lvl=None)

        with self.assertRaisesRegex(ValueError, exc_str):
            EllipsoidalSet([0], [[1]], scale=1, gaussian_conf_lvl=0.95)

    def test_error_on_shape_matrix_with_wrong_size(self):
        """
        Test error in event EllipsoidalSet shape matrix
        is not in accordance with set dimension.
        """
        center = [0, 0]
        invalid_shape_matrix = [[1, 0]]
        scale = 1

        exc_str = r".*must be a square matrix of size 2.*\(provided.*shape \(1, 2\)\)"

        # assert error on construction
        with self.assertRaisesRegex(ValueError, exc_str):
            EllipsoidalSet(center, invalid_shape_matrix, scale)

        # construct a valid EllipsoidalSet
        eset = EllipsoidalSet(center, [[1, 0], [0, 1]], scale)

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.shape_matrix = invalid_shape_matrix

    def test_set_as_constraint(self):
        """
        Test method for setting up constraints works correctly.
        """
        m = ConcreteModel()
        eset = EllipsoidalSet(
            center=[1, 1.5], shape_matrix=[[1, 0.5], [0.5, 1]], scale=2.5
        )
        uq = eset.set_as_constraint(uncertain_params=None, block=m)

        self.assertEqual(uq.auxiliary_vars, [])
        self.assertEqual(len(uq.uncertain_param_vars), 2)
        self.assertEqual(len(uq.uncertainty_cons), 1)
        self.assertIs(uq.block, m)

        var1, var2 = uq.uncertain_param_vars

        assertExpressionsEqual(
            self,
            uq.uncertainty_cons[0].expr,
            (
                np.float64(4 / 3) * (var1 - np.float64(1.0)) * (var1 - np.float64(1.0))
                + np.float64(-2 / 3)
                * (var1 - np.float64(1.0))
                * (var2 - np.float64(1.5))
                + np.float64(-2 / 3)
                * (var2 - np.float64(1.5))
                * (var1 - np.float64(1.0))
                + np.float64(4 / 3)
                * (var2 - np.float64(1.5))
                * (var2 - np.float64(1.5))
                <= 2.5
            ),
        )

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain parameters
        does not match the dimension.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        eset = EllipsoidalSet(
            center=[1, 1.5], shape_matrix=[[1, 0.5], [0.5, 1]], scale=2.5
        )
        with self.assertRaisesRegex(ValueError, ".*dimension"):
            eset.set_as_constraint(uncertain_params=[m.v1], block=m)

    def test_set_as_constraint_type_mismatch(self):
        """
        Check exception raised if uncertain parameter variables
        are of invalid type.
        """
        m = ConcreteModel()
        m.p1 = Param([0, 1], initialize=0, mutable=True)
        eset = EllipsoidalSet(
            center=[1, 1.5], shape_matrix=[[1, 0.5], [0.5, 1]], scale=2.5
        )
        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            eset.set_as_constraint(uncertain_params=[m.p1[0], m.p1[1]], block=m)

        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            eset.set_as_constraint(uncertain_params=m.p1, block=m)

    def test_point_in_set(self):
        eset = EllipsoidalSet(
            center=[1, 1.5], shape_matrix=[[1, 0.5], [0.5, 1]], scale=2.5
        )
        sqrt_mat = np.linalg.cholesky(eset.shape_matrix)
        sqrt_scale = eset.scale**0.5
        center = eset.center
        self.assertTrue(eset.point_in_set(eset.center))

        # some boundary points
        self.assertTrue(eset.point_in_set(center + sqrt_mat @ [0, sqrt_scale]))
        self.assertTrue(eset.point_in_set(center + sqrt_mat @ [sqrt_scale, 0]))
        self.assertTrue(eset.point_in_set(center + sqrt_mat @ [0, -sqrt_scale]))
        self.assertTrue(eset.point_in_set(center + sqrt_mat @ [-sqrt_scale, 0]))

        self.assertFalse(eset.point_in_set(center + sqrt_mat @ [0, sqrt_scale * 2]))
        self.assertFalse(eset.point_in_set(center + sqrt_mat @ [sqrt_scale * 2, 0]))
        self.assertFalse(eset.point_in_set(center + sqrt_mat @ [0, -sqrt_scale * 2]))
        self.assertFalse(eset.point_in_set(center + sqrt_mat @ [-sqrt_scale * 2, 0]))

        # test singleton
        eset.scale = 0
        self.assertTrue(eset.point_in_set(eset.center))
        self.assertTrue(eset.point_in_set(eset.center + [5e-9, 0]))
        self.assertFalse(eset.point_in_set(eset.center + [1e-4, 1e-4]))

        # check what happens if dimensions are off
        with self.assertRaisesRegex(ValueError, ".*to match the set dimension.*"):
            eset.point_in_set([1, 2, 3, 4])

    @unittest.skipUnless(baron_available, "BARON is not available.")
    def test_compute_exact_parameter_bounds(self):
        """
        Test parameter bounds computations give expected results.
        """
        baron = SolverFactory("baron")
        eset = EllipsoidalSet(
            center=[1, 1.5], shape_matrix=[[1, 0.5], [0.5, 1]], scale=0.25
        )
        computed_bounds = eset._compute_exact_parameter_bounds(baron)
        np.testing.assert_allclose(computed_bounds, [[0.5, 1.5], [1.0, 2.0]])
        np.testing.assert_allclose(computed_bounds, eset.parameter_bounds)

        eset2 = EllipsoidalSet(
            center=[1, 1.5], shape_matrix=[[1, 0.5], [0.5, 1]], scale=2.25
        )
        computed_bounds_2 = eset2._compute_exact_parameter_bounds(baron)

        # add absolute tolerance to account from
        # matrix inversion and roundoff errors
        np.testing.assert_allclose(computed_bounds_2, [[-0.5, 2.5], [0, 3]], atol=1e-8)
        np.testing.assert_allclose(computed_bounds_2, eset2.parameter_bounds, atol=1e-8)

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var([0, 1], initialize=0)
        eset = EllipsoidalSet(
            center=[1, 1.5], shape_matrix=[[1, 0.5], [0.5, 1]], scale=0.25
        )
        eset._add_bounds_on_uncertain_parameters(
            global_solver=None, uncertain_param_vars=m.uncertain_param_vars
        )
        self.assertEqual(m.uncertain_param_vars[0].bounds, (0.5, 1.5))
        self.assertEqual(m.uncertain_param_vars[1].bounds, (1, 2))

    def test_validate(self):
        """
        Test validate performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid ellipsoidal set
        center = [0.0, 0.0]
        shape_matrix = [[1.0, 0.0], [0.0, 2.0]]
        scale = 1
        ellipsoid_set = EllipsoidalSet(center, shape_matrix, scale)

        # validate raises no issues on valid set
        ellipsoid_set.validate(config=CONFIG)

    def test_validate_finiteness(self):
        """
        Test validate finiteness check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid ellipsoidal set
        center = [0.0, 0.0]
        shape_matrix = [[1.0, 0.0], [0.0, 2.0]]
        scale = 1
        ellipsoid_set = EllipsoidalSet(center, shape_matrix, scale)

        # check when values are not finite
        ellipsoid_set.center[0] = np.nan
        exc_str = r"Entry 'nan' of the argument `center` is not a finite numeric value"
        with self.assertRaisesRegex(ValueError, exc_str):
            ellipsoid_set.validate(config=CONFIG)

    def test_validate_scale(self):
        """
        Test validate scale check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid ellipsoidal set
        center = [0.0, 0.0]
        shape_matrix = [[1.0, 0.0], [0.0, 2.0]]
        scale = 1
        ellipsoid_set = EllipsoidalSet(center, shape_matrix, scale)

        # check when scale is not positive
        ellipsoid_set.scale = -1
        exc_str = r".*must be a non-negative real \(provided.*-1\)"
        with self.assertRaisesRegex(ValueError, exc_str):
            ellipsoid_set.validate(config=CONFIG)

    def test_validate_shape_matrix(self):
        """
        Test validate shape matrix check performs as expected.
        """
        CONFIG = Bunch()

        # construct a valid ellipsoidal set
        center = [0.0, 0.0]
        shape_matrix = [[1.0, 0.0], [0.0, 2.0]]
        scale = 1
        ellipsoid_set = EllipsoidalSet(center, shape_matrix, scale)

        # check when shape matrix is invalid
        center = [0, 0]
        scale = 3

        with self.assertRaisesRegex(
            ValueError,
            r"Shape matrix must be symmetric",
            msg="Asymmetric shape matrix test failed",
        ):
            ellipsoid_set = EllipsoidalSet(center, [[1, 1], [0, 1]], scale)
            ellipsoid_set.validate(config=CONFIG)
        with self.assertRaises(
            np.linalg.LinAlgError, msg="Singular shape matrix test failed"
        ):
            ellipsoid_set = EllipsoidalSet(center, [[0, 0], [0, 0]], scale)
            ellipsoid_set.validate(config=CONFIG)
        with self.assertRaisesRegex(
            ValueError,
            r"Non positive-definite.*",
            msg="Indefinite shape matrix test failed",
        ):
            ellipsoid_set = EllipsoidalSet(center, [[1, 0], [0, -2]], scale)
            ellipsoid_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_bounded_and_nonempty(self):
        """
        Test `is_bounded` and `is_nonempty` for a valid ellipsoidal set.
        """
        center = [0.0, 0.0]
        shape_matrix = [[1.0, 0.0], [0.0, 2.0]]
        scale = 1
        ellipsoid_set = EllipsoidalSet(center, shape_matrix, scale)
        bounded_and_nonempty_check(self, ellipsoid_set)

    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        eset = EllipsoidalSet(
            center=np.zeros(3), shape_matrix=np.diag([1, 2, 3]), scale=1
        )
        self.assertEqual(
            eset._is_coordinate_fixed(config=Bunch()), [False, False, False]
        )
        eset.scale = 0
        self.assertEqual(eset._is_coordinate_fixed(config=Bunch()), [True, True, True])


class TestPolyhedralSet(unittest.TestCase):
    """
    Tests for the PolyhedralSet.
    """

    def test_normal_construction_and_update(self):
        """
        Test PolyhedralSet constructor and attribute setters work
        appropriately.
        """
        lhs_coefficients_mat = [[1, 2, 3], [4, 5, 6]]
        rhs_vec = [1, 3]

        pset = PolyhedralSet(lhs_coefficients_mat, rhs_vec)

        # check attributes are as expected
        np.testing.assert_allclose(lhs_coefficients_mat, pset.coefficients_mat)
        np.testing.assert_allclose(rhs_vec, pset.rhs_vec)

        # update the set
        pset.coefficients_mat = [[1, 0, 1], [1, 1, 1.5]]
        pset.rhs_vec = [3, 4]

        # check updates work
        np.testing.assert_allclose([[1, 0, 1], [1, 1, 1.5]], pset.coefficients_mat)
        np.testing.assert_allclose([3, 4], pset.rhs_vec)

    def test_error_on_polyhedral_set_dim_change(self):
        """
        PolyhedralSet dimension (number columns of 'coefficients_mat')
        is considered immutable.
        Test ValueError raised if attempt made to change dimension.
        """
        # construct valid set
        pset = PolyhedralSet([[1, 2, 3], [4, 5, 6]], [1, 3])

        exc_str = (
            r".*must have 3 columns to match set dimension \(provided.*2 columns\)"
        )

        # assert error on update
        with self.assertRaisesRegex(ValueError, exc_str):
            pset.coefficients_mat = [[1, 2], [3, 4]]

    def test_error_on_inconsistent_rows(self):
        """
        Number of rows of budget membership mat is immutable.
        Similarly, size of rhs_vec is immutable.
        Check ValueError raised in event of attempted change.
        """
        coeffs_mat_exc_str = (
            r".*must have 2 rows to match shape of attribute 'rhs_vec' "
            r"\(provided.*3 rows\)"
        )
        rhs_vec_exc_str = (
            r".*must have 2 entries to match shape of attribute "
            r"'coefficients_mat' \(provided.*3 entries\)"
        )
        # assert error on construction
        with self.assertRaisesRegex(ValueError, rhs_vec_exc_str):
            PolyhedralSet([[1, 2], [3, 4]], rhs_vec=[1, 3, 3])

        # construct a valid polyhedral set
        # (2 x 2 coefficients, 2-vector for RHS)
        pset = PolyhedralSet([[1, 2], [3, 4]], rhs_vec=[1, 3])

        # assert error on update
        with self.assertRaisesRegex(ValueError, coeffs_mat_exc_str):
            # 3 x 2 matrix row mismatch
            pset.coefficients_mat = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaisesRegex(ValueError, rhs_vec_exc_str):
            # 3-vector mismatches 2 rows
            pset.rhs_vec = [1, 3, 2]

    def test_set_as_constraint(self):
        """
        Test method for setting up constraints works correctly.
        """
        m = ConcreteModel()
        pset = PolyhedralSet(
            lhs_coefficients_mat=[[1, 0], [-1, 1], [-1, -1]], rhs_vec=[2, -1, -1]
        )
        uq = pset.set_as_constraint(uncertain_params=None, block=m)

        self.assertEqual(uq.auxiliary_vars, [])
        self.assertEqual(len(uq.uncertain_param_vars), 2)
        self.assertEqual(len(uq.uncertainty_cons), 3)
        self.assertIs(uq.block, m)

        var1, var2 = uq.uncertain_param_vars

        assertExpressionsEqual(
            self, uq.uncertainty_cons[0].expr, var1 + np.int_(0) * var2 <= np.int_(2)
        )
        assertExpressionsEqual(
            self,
            uq.uncertainty_cons[1].expr,
            np.int_(-1) * var1 + np.int_(1) * var2 <= np.int_(-1),
        )
        assertExpressionsEqual(
            self,
            uq.uncertainty_cons[2].expr,
            np.int_(-1) * var1 + np.int_(-1) * var2 <= np.int_(-1),
        )

    def test_set_as_constraint_dim_mismatch(self):
        """
        Check exception raised if number of uncertain parameters
        does not match the dimension.
        """
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        pset = PolyhedralSet(
            lhs_coefficients_mat=[[1, 0], [-1, 1], [-1, -1]], rhs_vec=[2, -1, -1]
        )
        with self.assertRaisesRegex(ValueError, ".*dimension"):
            pset.set_as_constraint(uncertain_params=[m.v1], block=m)

    def test_set_as_constraint_type_mismatch(self):
        """
        Check exception raised if uncertain parameter variables
        are of invalid type.
        """
        m = ConcreteModel()
        m.p1 = Param([0, 1], initialize=0, mutable=True)
        pset = PolyhedralSet(
            lhs_coefficients_mat=[[1, 0], [-1, 1], [-1, -1]], rhs_vec=[2, -1, -1]
        )
        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            pset.set_as_constraint(uncertain_params=[m.p1[0], m.p1[1]], block=m)

        with self.assertRaisesRegex(TypeError, ".*valid component type"):
            pset.set_as_constraint(uncertain_params=m.p1, block=m)

    @unittest.skipUnless(baron_available, "BARON is not available.")
    def test_compute_exact_parameter_bounds(self):
        """
        Test parameter bounds computations give expected results.
        """
        pset = PolyhedralSet(
            lhs_coefficients_mat=[[1, 0], [-1, 1], [-1, -1]], rhs_vec=[2, -1, -1]
        )
        self.assertEqual(pset.parameter_bounds, [])
        computed_bounds = pset._compute_exact_parameter_bounds(SolverFactory("baron"))
        self.assertEqual(computed_bounds, [(1, 2), (-1, 1)])

    def test_point_in_set(self):
        """
        Test point in set checks work as expected.
        """
        pset = PolyhedralSet(
            lhs_coefficients_mat=[[1, 0], [-1, 1], [-1, -1]], rhs_vec=[2, -1, -1]
        )
        self.assertTrue(pset.point_in_set([1, 0]))
        self.assertTrue(pset.point_in_set([2, 1]))
        self.assertTrue(pset.point_in_set([2, -1]))
        self.assertFalse(pset.point_in_set([1, 1]))
        self.assertFalse(pset.point_in_set([-1, 0]))
        self.assertFalse(pset.point_in_set([0, 0]))

        # check what happens if dimensions are off
        with self.assertRaisesRegex(ValueError, ".*to match the set dimension.*"):
            pset.point_in_set([1, 2, 3, 4])

    @unittest.skipUnless(baron_available, "Global NLP solver is not available.")
    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.uncertain_param_vars = Var([0, 1], initialize=0)
        pset = PolyhedralSet(
            lhs_coefficients_mat=[[1, 0], [-1, 1], [-1, -1]], rhs_vec=[2, -1, -1]
        )
        pset._add_bounds_on_uncertain_parameters(
            global_solver=SolverFactory("baron"),
            uncertain_param_vars=m.uncertain_param_vars,
        )
        self.assertEqual(m.uncertain_param_vars[0].bounds, (1, 2))
        self.assertEqual(m.uncertain_param_vars[1].bounds, (-1, 1))

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_validate(self):
        """
        Test validate performs as expected.
        """
        CONFIG = pyros_config()
        CONFIG.global_solver = global_solver

        # construct a valid polyhedral set
        polyhedral_set = PolyhedralSet(
            lhs_coefficients_mat=[[1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]],
            rhs_vec=[2.0, -1.0, -1.0],
        )

        # validate raises no issues on valid set
        polyhedral_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_validate_finiteness(self):
        """
        Test validate finiteness check performs as expected.
        """
        CONFIG = pyros_config()
        CONFIG.global_solver = global_solver

        # construct a valid polyhedral set
        polyhedral_set = PolyhedralSet(
            lhs_coefficients_mat=[[1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]],
            rhs_vec=[2.0, -1.0, -1.0],
        )

        # check when values are not finite
        polyhedral_set.rhs_vec[0] = np.nan
        exc_str = r"Entry 'nan' of the argument `rhs_vec` is not a finite numeric value"
        with self.assertRaisesRegex(ValueError, exc_str):
            polyhedral_set.validate(config=CONFIG)

        polyhedral_set.rhs_vec[0] = 2
        polyhedral_set.coefficients_mat[0][0] = np.nan
        exc_str = (
            r"Entry 'nan' of the argument `coefficients_mat` "
            r"is not a finite numeric value"
        )
        with self.assertRaisesRegex(ValueError, exc_str):
            polyhedral_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_validate_full_column_rank(self):
        """
        Test validate full column rank check performs as expected.
        """
        CONFIG = pyros_config()
        CONFIG.global_solver = global_solver

        # construct a valid polyhedral set
        polyhedral_set = PolyhedralSet(
            lhs_coefficients_mat=[[1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]],
            rhs_vec=[2.0, -1.0, -1.0],
        )

        # check when LHS matrix is not full column rank
        polyhedral_set.coefficients_mat = [[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        exc_str = r".*all entries zero in columns at indexes: 0.*"
        with self.assertRaisesRegex(ValueError, exc_str):
            polyhedral_set.validate(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_bounded_and_nonempty(self):
        """
        Test `is_bounded` and `is_nonempty` for a valid polyhedral set.
        """
        polyhedral_set = PolyhedralSet(
            lhs_coefficients_mat=[[1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]],
            rhs_vec=[2.0, -1.0, -1.0],
        )
        bounded_and_nonempty_check(self, polyhedral_set)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        pset = PolyhedralSet(
            lhs_coefficients_mat=[[1, 0], [-1, 0], [0, 1], [0, -1]],
            rhs_vec=[1, 1, 1, -1],
        )
        baron = SolverFactory("baron")
        self.assertEqual(
            pset._is_coordinate_fixed(config=Bunch(global_solver=baron)), [False, True]
        )


class CustomUncertaintySet(UncertaintySet):
    """
    Test simple custom uncertainty set subclass.
    """

    def __init__(self, dim):
        self._dim = dim
        self._parameter_bounds = [(-1, 1)] * self.dim

    @property
    def geometry(self):
        self.geometry = Geometry.LINEAR

    @property
    def dim(self):
        return self._dim

    def set_as_constraint(self, uncertain_params=None, block=None):
        blk, param_var_list, conlist, aux_vars = (
            _setup_standard_uncertainty_set_constraint_block(
                block=block,
                uncertain_param_vars=uncertain_params,
                dim=self.dim,
                num_auxiliary_vars=None,
            )
        )
        conlist.add(sum(param_var_list) <= 0)
        for var in param_var_list:
            conlist.add(-1 <= var)

        return UncertaintyQuantification(
            block=blk,
            uncertainty_cons=list(conlist.values()),
            uncertain_param_vars=param_var_list,
            auxiliary_vars=aux_vars,
        )

    def point_in_set(self, point):
        point_arr = np.array(point)
        return point_arr.sum() <= 0 and np.all(-1 <= point_arr)

    @property
    def parameter_bounds(self):
        return self._parameter_bounds

    @parameter_bounds.setter
    def parameter_bounds(self, val):
        self._parameter_bounds = val


class TestCustomUncertaintySet(unittest.TestCase):
    """
    Test for a custom uncertainty set subclass.
    """

    def test_set_as_constraint(self):
        """
        Test method for setting up constraints works correctly.
        """
        m = ConcreteModel()
        custom_set = CustomUncertaintySet(dim=2)
        uq = custom_set.set_as_constraint(uncertain_params=None, block=m)

        con1, con2, con3 = uq.uncertainty_cons
        var1, var2 = uq.uncertain_param_vars
        self.assertEqual(uq.auxiliary_vars, [])
        self.assertIs(uq.block, m)
        self.assertEqual(len(uq.uncertainty_cons), 3)
        self.assertEqual(len(uq.uncertain_param_vars), 2)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_compute_exact_parameter_bounds(self):
        """
        Test parameter bounds computations give expected results.
        """
        baron = SolverFactory("baron")
        custom_set = CustomUncertaintySet(dim=2)
        self.assertEqual(custom_set.parameter_bounds, [(-1, 1)] * 2)
        self.assertEqual(
            custom_set._compute_exact_parameter_bounds(baron), [(-1, 1)] * 2
        )

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_solve_feasibility(self):
        """
        Test uncertainty set feasibility problem gives expected results.
        """
        # feasibility problem passes
        baron = SolverFactory("baron")
        custom_set = CustomUncertaintySet(dim=2)
        custom_set._solve_feasibility(baron)

        # feasibility problem fails
        custom_set.parameter_bounds = [[1, 2], [3, 4]]
        exc_str = r"Could not successfully solve feasibility problem. .*"
        with self.assertRaisesRegex(ValueError, exc_str):
            custom_set._solve_feasibility(baron)

    # test default is_bounded
    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_is_bounded(self):
        """
        Test boundedness check computations give expected results.
        """
        custom_set = CustomUncertaintySet(dim=2)
        CONFIG = pyros_config()
        CONFIG.global_solver = global_solver

        # using provided parameter_bounds
        self.assertTrue(custom_set.is_bounded(config=CONFIG), "Set is not bounded")

        # when parameter_bounds is not available
        custom_set.parameter_bounds = []
        self.assertTrue(custom_set.is_bounded(config=CONFIG), "Set is not bounded")

        # when bad bounds are provided
        for val_str in ["inf", "nan"]:
            bad_bounds = [[1, float(val_str)], [2, 3]]
            custom_set.parameter_bounds = bad_bounds
            self.assertFalse(custom_set.is_bounded(config=CONFIG), "Set is bounded")

    # test default is_nonempty
    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_is_nonempty(self):
        """
        Test nonemptiness check computations give expected results.
        """
        custom_set = CustomUncertaintySet(dim=2)
        CONFIG = pyros_config()
        CONFIG.global_solver = global_solver

        # constructing a feasibility problem
        self.assertTrue(custom_set.is_nonempty(config=CONFIG), "Set is empty")

        # using provided nominal point
        CONFIG.nominal_uncertain_param_vals = [0, 0]
        self.assertTrue(custom_set.is_nonempty(config=CONFIG), "Set is empty")

        # check when nominal point is not in set
        CONFIG.nominal_uncertain_param_vals = [-2, -2]
        self.assertFalse(
            custom_set.is_nonempty(config=CONFIG), "Nominal point is in set"
        )

        # check when feasibility problem fails
        CONFIG.nominal_uncertain_param_vals = None
        custom_set.parameter_bounds = [[1, 2], [3, 4]]
        exc_str = r"Could not successfully solve feasibility problem. .*"
        with self.assertRaisesRegex(ValueError, exc_str):
            custom_set.is_nonempty(config=CONFIG)

    @unittest.skipUnless(baron_available, "BARON is not available")
    def test_is_coordinate_fixed(self):
        """
        Test method for checking whether there are coordinates
        constrained to a single value.
        """
        custom_set = CustomUncertaintySet(dim=3)
        custom_set._PARAMETER_BOUNDS_EXACT = True
        self.assertEqual(
            custom_set._is_coordinate_fixed(config=Bunch()), [False, False, False]
        )

        custom_set._PARAMETER_BOUNDS_EXACT = False
        baron = SolverFactory("baron")
        self.assertEqual(
            custom_set._is_coordinate_fixed(config=Bunch(global_solver=baron)),
            [False, False, False],
        )


if __name__ == "__main__":
    unittest.main()

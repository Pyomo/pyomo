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

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import

np, np_available = attempt_import('numpy', minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy.sparse')
mumps, mumps_available = attempt_import('mumps')
if not np_available or not scipy_available:
    raise unittest.SkipTest('numpy and scipy are needed for interior point tests')
import numpy as np
from scipy.sparse import coo_matrix, tril
from pyomo.contrib import interior_point as ip
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus

if scipy_available:
    from pyomo.contrib.interior_point.linalg.scipy_interface import ScipyInterface
if mumps_available:
    from pyomo.contrib.interior_point.linalg.mumps_interface import MumpsInterface
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface

ma27_available = MA27Interface.available()
if ma27_available:
    from pyomo.contrib.interior_point.linalg.ma27_interface import (
        InteriorPointMA27Interface,
    )


def get_base_matrix(use_tril):
    if use_tril:
        row = [0, 1, 1, 2, 2]
        col = [0, 0, 1, 0, 2]
        data = [1, 7, 4, 3, 6]
    else:
        row = [0, 0, 0, 1, 1, 2, 2]
        col = [0, 1, 2, 0, 1, 0, 2]
        data = [1, 7, 3, 7, 4, 3, 6]
    mat = coo_matrix((data, (row, col)), shape=(3, 3), dtype=np.double)
    return mat


def get_base_matrix_wrong_order(use_tril):
    if use_tril:
        row = [1, 0, 1, 2, 2]
        col = [0, 0, 1, 0, 2]
        data = [7, 1, 4, 3, 6]
    else:
        row = [1, 0, 0, 0, 1, 2, 2]
        col = [0, 1, 2, 0, 1, 0, 2]
        data = [7, 7, 3, 1, 4, 3, 6]
    mat = coo_matrix((data, (row, col)), shape=(3, 3), dtype=np.double)
    return mat


class TestTrilBehavior(unittest.TestCase):
    """
    Some of the other tests in this file depend on
    the behavior of tril that is tested in this
    test, namely the tests in TestWrongNonzeroOrdering.
    """

    def test_tril_behavior(self):
        mat = get_base_matrix(use_tril=True)
        mat2 = tril(mat)
        self.assertTrue(np.all(mat.row == mat2.row))
        self.assertTrue(np.all(mat.col == mat2.col))
        self.assertTrue(np.allclose(mat.data, mat2.data))

        mat = get_base_matrix_wrong_order(use_tril=True)
        self.assertFalse(np.all(mat.row == mat2.row))
        self.assertFalse(np.allclose(mat.data, mat2.data))
        mat2 = tril(mat)
        self.assertTrue(np.all(mat.row == mat2.row))
        self.assertTrue(np.all(mat.col == mat2.col))
        self.assertTrue(np.allclose(mat.data, mat2.data))


class TestLinearSolvers(unittest.TestCase):
    def _test_linear_solvers(self, solver):
        mat = get_base_matrix(use_tril=False)
        zero_mat = mat.copy()
        zero_mat.data.fill(0)
        stat = solver.do_symbolic_factorization(zero_mat)
        self.assertEqual(stat.status, LinearSolverStatus.successful)
        stat = solver.do_numeric_factorization(mat)
        self.assertEqual(stat.status, LinearSolverStatus.successful)
        x_true = np.array([1, 2, 3], dtype=np.double)
        rhs = mat * x_true
        x, res = solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x, x_true))
        x_true = np.array([4, 2, 3], dtype=np.double)
        rhs = mat * x_true
        x, res = solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x, x_true))

    @unittest.skipIf(
        not scipy_available, 'scipy is needed for interior point scipy tests'
    )
    def test_scipy(self):
        solver = ScipyInterface()
        self._test_linear_solvers(solver)

    @unittest.skipIf(
        not mumps_available, 'mumps is needed for interior point mumps tests'
    )
    def test_mumps(self):
        solver = MumpsInterface()
        self._test_linear_solvers(solver)

    @unittest.skipIf(not ma27_available, 'MA27 is needed for interior point MA27 tests')
    def test_ma27(self):
        solver = InteriorPointMA27Interface()
        self._test_linear_solvers(solver)


@unittest.skip('This does not work yet')
class TestWrongNonzeroOrdering(unittest.TestCase):
    def _test_solvers(self, solver, use_tril):
        mat = get_base_matrix(use_tril=use_tril)
        wrong_order_mat = get_base_matrix_wrong_order(use_tril=use_tril)
        stat = solver.do_symbolic_factorization(mat)
        stat = solver.do_numeric_factorization(wrong_order_mat)
        x_true = np.array([1, 2, 3], dtype=np.double)
        rhs = mat * x_true
        x, res = solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x, x_true))

    @unittest.skipIf(
        not scipy_available, 'scipy is needed for interior point scipy tests'
    )
    def test_scipy(self):
        solver = ScipyInterface()
        self._test_solvers(solver, use_tril=False)

    @unittest.skipIf(
        not mumps_available, 'mumps is needed for interior point mumps tests'
    )
    def test_mumps(self):
        solver = MumpsInterface()
        self._test_solvers(solver, use_tril=True)

    @unittest.skipIf(not ma27_available, 'MA27 is needed for interior point MA27 tests')
    def test_ma27(self):
        solver = InteriorPointMA27Interface()
        self._test_solvers(solver, use_tril=True)

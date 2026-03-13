# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
#
#  Additional contributions Copyright (c) 2026 OLI Systems, Inc.
#  ___________________________________________________________________________________

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

import pyomo.contrib.pynumero.interfaces.utils as utils
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest("Pynumero ASL interface is not available")


class TestCondensedSparseSummation(unittest.TestCase):
    def test_condensed_sparse_summation(self):
        data = [1.0, 0.0]
        row = [1, 2]
        col = [2, 2]
        A = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))

        data = [3.0, 0.0]
        B = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))

        # By default, scipy will remove structural nonzeros that
        # have zero values
        C = A + B
        self.assertEqual(C.nnz, 1)

        # Our CondensedSparseSummation should not remove any
        # structural nonzeros
        sparse_sum = utils.CondensedSparseSummation([A, B])
        C = sparse_sum.sum([A, B])
        expected_data = np.asarray([4.0, 0.0], dtype=np.float64)
        expected_row = np.asarray([1, 2], dtype=np.int64)
        expected_col = np.asarray([2, 2], dtype=np.int64)
        self.assertTrue(np.array_equal(expected_data, C.data))
        self.assertTrue(np.array_equal(expected_row, C.row))
        self.assertTrue(np.array_equal(expected_col, C.col))

        B.data[1] = 5.0
        C = sparse_sum.sum([A, B])
        expected_data = np.asarray([4.0, 5.0], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_data, C.data))
        self.assertTrue(np.array_equal(expected_row, C.row))
        self.assertTrue(np.array_equal(expected_col, C.col))

        B.data[1] = 0.0
        C = sparse_sum.sum([A, B])
        expected_data = np.asarray([4.0, 0.0], dtype=np.float64)
        self.assertTrue(np.array_equal(expected_data, C.data))
        self.assertTrue(np.array_equal(expected_row, C.row))
        self.assertTrue(np.array_equal(expected_col, C.col))

    def test_repeated_row_col(self):
        data = [1.0, 0.0, 2.0]
        row = [1, 2, 1]
        col = [2, 2, 2]
        A = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))

        data = [3.0, 0.0]
        row = [1, 2]
        col = [2, 2]
        B = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))

        # Our CondensedSparseSummation should not remove any
        # structural nonzeros
        sparse_sum = utils.CondensedSparseSummation([A, B])
        C = sparse_sum.sum([A, B])
        expected_data = np.asarray([6.0, 0.0], dtype=np.float64)
        expected_row = np.asarray([1, 2], dtype=np.int64)
        expected_col = np.asarray([2, 2], dtype=np.int64)
        self.assertTrue(np.array_equal(expected_data, C.data))
        self.assertTrue(np.array_equal(expected_row, C.row))
        self.assertTrue(np.array_equal(expected_col, C.col))

    def test_empty_hessian_linear_model_does_not_crash(self):
        """
        Regression test for CondensedSparseSummation bug where an empty
        union of nonzeros caused unpack failure.

        Linear model -> Lagrangian Hessian has no nonzeros.
        Previously this crashed inside CondensedSparseSummation._build_maps().
        """

        m = pyo.ConcreteModel()

        # Linear model (same structure as user's example)
        m.v1 = pyo.Var(initialize=1e-8)
        m.v2 = pyo.Var(initialize=1)
        m.v3 = pyo.Var(initialize=1)

        m.c1 = pyo.Constraint(expr=m.v1 == m.v2)
        m.c2 = pyo.Constraint(expr=m.v1 == 1e-8 * m.v3)
        m.c3 = pyo.Constraint(expr=1e8 * m.v1 + 1e10 * m.v2 == 1e-6 * m.v3)

        # PyNumero requires an objective
        m.obj = pyo.Objective(expr=0)

        # This used to crash during initialization due to empty
        # union of nonzeros in the Hessian of the Lagrangian
        nlp = PyomoNLPWithGreyBoxBlocks(m)

        # Explicitly evaluate Hessian of Lagrangian
        hess = nlp.evaluate_hessian_lag()

        # Shape should match number of primals
        n = nlp.n_primals()
        assert hess.shape == (n, n)

        # For a purely linear model, Hessian must be structurally empty
        assert hess.nnz == 0

        # Also verify the sparse data vector is empty
        assert len(hess.data) == 0


if __name__ == '__main__':
    TestCondensedSparseSummation().test_condensed_sparse_summation()

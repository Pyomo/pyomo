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
from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

import pyomo.contrib.pynumero.interfaces.utils as utils


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


if __name__ == '__main__':
    TestCondensedSparseSummation().test_condensed_sparse_summation()

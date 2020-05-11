#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import pyutilib.th as unittest

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from scipy.sparse import coo_matrix, bmat

from pyomo.contrib.pynumero.sparse.utils import is_symmetric_dense, is_symmetric_sparse

class TestSparseUtils(unittest.TestCase):

    def setUp(self):

        row = np.array([0, 1, 4, 1, 2, 7, 2, 3, 5, 3, 4, 5, 4, 7, 5, 6, 6, 7])
        col = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7])
        data = np.array([27, 5, 12, 56, 66, 34, 94, 31, 41, 7, 98, 72, 24, 33, 78, 47, 98, 41])

        off_diagonal_mask = row != col
        new_row = np.concatenate([row, col[off_diagonal_mask]])
        new_col = np.concatenate([col, row[off_diagonal_mask]])
        new_data = np.concatenate([data, data[off_diagonal_mask]])
        m = coo_matrix((new_data, (new_row, new_col)), shape=(8, 8))

        self.block00 = m

        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4, 5, 7, 9])
        m = coo_matrix((data, (row, col)), shape=(4, 8))

        self.block10 = m

        row = np.array([0, 1, 2, 3])
        col = np.array([0, 1, 2, 3])
        data = np.array([1, 1, 1, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))

        self.block11 = m

    def test_is_symmetric_dense(self):

        m = self.block00.toarray()
        self.assertTrue(is_symmetric_dense(m))
        self.assertTrue(is_symmetric_dense(2))
        with self.assertRaises(Exception) as context:
            self.assertTrue(is_symmetric_dense(self.block00))

    def test_is_symmetric_sparse(self):
        m = self.block00
        self.assertTrue(is_symmetric_sparse(m))
        m = self.block00.toarray()
        self.assertTrue(is_symmetric_sparse(m))
        m = self.block11
        self.assertTrue(is_symmetric_sparse(m))
        m = self.block10
        self.assertFalse(is_symmetric_sparse(m))
        self.assertTrue(is_symmetric_sparse(2))

        row = np.array([0, 1, 2, 3])
        col = np.array([0, 1, 2, 3])
        data = np.array([1, 1, 1, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 6))
        self.assertFalse(is_symmetric_sparse(m))

        with self.assertRaises(Exception) as context:
            self.assertTrue(is_symmetric_sparse(range(5)))

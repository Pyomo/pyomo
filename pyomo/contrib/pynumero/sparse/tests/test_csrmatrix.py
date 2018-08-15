import unittest
import sys

try:
    from pyomo.contrib.pynumero.sparse import (COOMatrix,
                                               COOSymMatrix,
                                               SparseBase,
                                               IdentityMatrix,
                                               EmptyMatrix)

    from pyomo.contrib.pynumero.sparse.csr import CSRMatrix, CSRSymMatrix
    from pyomo.contrib.pynumero.sparse.csc import CSCMatrix, CSCSymMatrix

    from scipy.sparse.csr import csr_matrix
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.coo import coo_matrix
    import numpy as np
except:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")


class TestCSRMatrix(unittest.TestCase):

    def setUp(self):

        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4., 5., 7., 9.])
        m = CSRMatrix((data, (row, col)), shape=(4, 4))
        m.name = 'basic_matrix'
        self.basic_m = m

    def test_is_symmetric(self):
        self.assertFalse(self.basic_m.is_symmetric)

    def test_name(self):
        self.assertEqual(self.basic_m.name, 'basic_matrix')
        self.basic_m.name = 'hola'
        self.assertEqual(self.basic_m.name, 'hola')

    def test_shape(self):
        self.assertEqual(self.basic_m.shape[0], 4)
        self.assertEqual(self.basic_m.shape[1], 4)
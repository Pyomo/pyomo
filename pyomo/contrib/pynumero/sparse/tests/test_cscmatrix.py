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
    from pyomo.contrib.pynumero.sparse.utils import (_is_symmetric_numerically,
                                                     _convert_matrix_to_symmetric,
                                                     is_symmetric_dense)

    from scipy.sparse.csr import csr_matrix
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.coo import coo_matrix
    import numpy as np
except:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")


class TestCSCMatrix(unittest.TestCase):

    def setUp(self):

        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4., 5., 7., 9.])
        m = CSCMatrix((data, (row, col)), shape=(4, 4))
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

    def test_getrow(self):
        m = self.basic_m
        m_row = m.getrow(0)
        self.assertIsInstance(m_row, CSRMatrix)
        self.assertEqual(m_row.shape, (1, m.shape[1]))
        values = m_row.toarray()[0]
        tvalues = [4.0, 0.0, 9.0, 0.0]
        self.assertListEqual(values.tolist(), tvalues)

    def test_getcol(self):
        m = self.basic_m
        m_col = m.getcol(0)
        self.assertIsInstance(m_col, CSCMatrix)
        self.assertEqual(m_col.shape, (m.shape[1], 1))
        values = m_col.toarray().transpose()[0]
        tvalues = [4.0, 0.0, 0.0, 0.0]
        self.assertListEqual(values.tolist(), tvalues)

    def test_add_sparse(self):
        m = self.basic_m
        mm = m + m
        test_m = np.array([[4., 0., 9., 0.],
                           [0., 7., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 5.]])
        mm2 = test_m * 2
        self.assertIsInstance(mm, CSCMatrix)
        self.assertTrue(np.allclose(mm.toarray(), mm2))

        m2 = IdentityMatrix(4)
        mm = m + m2
        test_m = np.array([[5., 0., 9., 0.],
                           [0., 8., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 6.]])
        mm2 = test_m
        self.assertIsInstance(mm, CSCMatrix)
        self.assertTrue(np.allclose(mm.toarray(), mm2))

        mm = m2 + m
        self.assertIsInstance(mm, CSRMatrix)
        self.assertTrue(np.allclose(mm.toarray(), mm2))

    def test_sub_sparse(self):
        m = self.basic_m
        mm = m - m
        mm2 = np.zeros(m.shape, dtype=np.double)
        self.assertIsInstance(mm, CSCMatrix)
        self.assertTrue(np.allclose(mm.toarray(), mm2))

        m2 = IdentityMatrix(4)
        mm = m - m2
        test_m = np.array([[3., 0., 9., 0.],
                           [0., 6., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 0., 0., 4.]])
        mm2 = test_m
        self.assertIsInstance(mm, CSCMatrix)
        self.assertTrue(np.allclose(mm.toarray(), mm2))

        test_m = np.array([[-3., 0., -9., 0.],
                           [0., -6., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., -4.]])
        mm = m2 - m
        mm2 = test_m
        self.assertIsInstance(mm, CSRMatrix)
        self.assertTrue(np.allclose(mm.toarray(), mm2))

    def test_mul_sparse_matrix(self):

        # test unsymmetric times unsymmetric
        m = self.basic_m
        dense_m = m.toarray()
        res = m * m
        dense_res = np.matmul(dense_m, dense_m)
        self.assertFalse(res.is_symmetric)
        self.assertTrue(np.allclose(res.toarray(), dense_res))

        # test symmetric result
        m = self.basic_m
        dense_m = m.toarray()
        res = m.transpose() * m
        dense_res = np.matmul(dense_m.transpose(), dense_m)
        self.assertTrue(res.is_symmetric)
        self.assertTrue(np.allclose(res.toarray(), dense_res))

        # test unsymmetric with rectangular
        m = self.basic_m
        dense_m2 = np.array([[1.0, 2.0],
                             [3.0, 4.0],
                             [5.0, 6.0],
                             [7.0, 8.0]])

        m2 = CSCMatrix(dense_m2)
        res = m * m2
        dense_res = np.matmul(m.toarray(), dense_m2)
        self.assertFalse(res.is_symmetric)
        self.assertTrue(np.allclose(res.toarray(), dense_res))

        # test unsymmetric with rectangular scipycsr
        m = self.basic_m
        dense_m2 = np.array([[1.0, 2.0],
                             [3.0, 4.0],
                             [5.0, 6.0],
                             [7.0, 8.0]])

        m2 = csc_matrix(dense_m2)
        with self.assertRaises(Exception) as context:
            res = m * m2

    def test_is_symmetric_numerically(self):

        test_m = np.array([[2., 0., 0., 1.],
                           [0., 3., 0., 0.],
                           [0., 0., 4., 0.],
                           [1., 0., 0., 5.]])

        m = CSCMatrix(test_m)
        self.assertTrue(_is_symmetric_numerically(m))
        self.assertFalse(_is_symmetric_numerically(self.basic_m))

    def test_convert_matrix_to_symmetric(self):

        test_m = np.array([[2., 0., 0., 1.],
                           [0., 3., 0., 0.],
                           [0., 0., 4., 0.],
                           [1., 0., 0., 5.]])

        m = CSCMatrix(test_m)
        sm = _convert_matrix_to_symmetric(m)
        self.assertTrue(sm.is_symmetric)
        mm = sm.toarray()
        self.assertTrue(np.allclose(mm, test_m, atol=1e-6))
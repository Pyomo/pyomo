import sys
import os
import pyutilib.th as unittest

try:
    from scipy.sparse.csr import csr_matrix
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.coo import coo_matrix
    import numpy as np
except ImportError:
    raise unittest.SkipTest(
        "Pynumero needs scipy and numpy to run COO matrix tests")

from pyomo.contrib.pynumero.extensions.sparseutils import SparseLib
if not SparseLib.available():
    raise unittest.SkipTest(
        "Pynumero needs the SparseUtils extension to run COO matrix tests")


from pyomo.contrib.pynumero.sparse.base import SparseBase
from pyomo.contrib.pynumero.sparse.coo import (COOMatrix,
                                               COOSymMatrix,
                                               IdentityMatrix,
                                               EmptyMatrix)

from pyomo.contrib.pynumero.sparse.csr import CSRMatrix, CSRSymMatrix
from pyomo.contrib.pynumero.sparse.csc import CSCMatrix, CSCSymMatrix
from pyomo.contrib.pynumero.sparse.utils import (_is_symmetric_numerically,
                                                 _convert_matrix_to_symmetric,
                                                 is_symmetric_dense)


@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestCOOMatrix(unittest.TestCase):

    def setUp(self):

        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4., 5., 7., 9.])
        m = COOMatrix((data, (row, col)), shape=(4, 4))
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

    def test_tocsr(self):
        csrm = self.basic_m.tocsr()
        m = self.basic_m
        scipym = csr_matrix((m.data,
                             (m.row, m.col)),
                             shape=(4, 4))
        self.assertListEqual(csrm.indices.tolist(), scipym.indices.tolist())
        self.assertListEqual(csrm.indptr.tolist(), scipym.indptr.tolist())
        self.assertListEqual(csrm.data.tolist(), scipym.data.tolist())
        self.assertEqual(csrm.shape, scipym.shape)
        self.assertIsInstance(csrm, SparseBase)

    def test_tocsc(self):
        cscm = self.basic_m.tocsc()
        m = self.basic_m
        scipym = csc_matrix((m.data,
                             (m.row, m.col)),
                            shape=(4, 4))
        self.assertListEqual(cscm.indices.tolist(), scipym.indices.tolist())
        self.assertListEqual(cscm.indptr.tolist(), scipym.indptr.tolist())
        self.assertListEqual(cscm.data.tolist(), scipym.data.tolist())
        self.assertEqual(cscm.shape, scipym.shape)
        self.assertIsInstance(cscm, SparseBase)

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
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), mm2.flatten().tolist())

        m2 = coo_matrix((self.basic_m.data,
                        (self.basic_m.row, self.basic_m.col)),
                        shape=self.basic_m.shape)
        with self.assertRaises(Exception) as context:
            mm = m + m2

        m2 = IdentityMatrix(4)
        mm = m + m2
        test_m = np.array([[5., 0., 9., 0.],
                           [0., 8., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 6.]])
        mm2 = test_m
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), mm2.flatten().tolist())

        mm = m2 + m
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), mm2.flatten().tolist())

    def test_sub_sparse(self):
        m = self.basic_m
        mm = m - m
        mm2 = np.zeros(m.shape, dtype=np.double)
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), mm2.flatten().tolist())

        m2 = IdentityMatrix(4)
        mm = m - m2
        test_m = np.array([[3., 0., 9., 0.],
                           [0., 6., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 0., 0., 4.]])
        mm2 = test_m
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), mm2.flatten().tolist())

        test_m = np.array([[-3., 0., -9., 0.],
                           [0., -6., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., -4.]])
        mm = m2 - m
        mm2 = test_m
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), mm2.flatten().tolist())

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

        m2 = COOMatrix(dense_m2)
        res = m * m2
        dense_res = np.matmul(m.toarray(), dense_m2)
        self.assertFalse(res.is_symmetric)
        self.assertTrue(np.allclose(res.toarray(), dense_res))

        # test unsymmetric with rectangular scipycoo
        m = self.basic_m
        dense_m2 = np.array([[1.0, 2.0],
                             [3.0, 4.0],
                             [5.0, 6.0],
                             [7.0, 8.0]])

        m2 = coo_matrix(dense_m2)
        with self.assertRaises(Exception) as context:
            res = m * m2

    def test_is_symmetric_dense(self):

        test_m = np.array([[2., 0., 0., 1.],
                           [0., 3., 0., 0.],
                           [0., 0., 4., 0.],
                           [1., 0., 0., 5.]])

        flag = is_symmetric_dense(test_m)
        self.assertTrue(flag)

        usm = self.basic_m.toarray()
        flag = is_symmetric_dense(usm)
        self.assertFalse(flag)

    def test_is_symmetric_numerically(self):

        test_m = np.array([[2., 0., 0., 1.],
                           [0., 3., 0., 0.],
                           [0., 0., 4., 0.],
                           [1., 0., 0., 5.]])

        m = COOMatrix(test_m)
        self.assertTrue(_is_symmetric_numerically(m))
        self.assertFalse(_is_symmetric_numerically(self.basic_m))

    def test_convert_matrix_to_symmetric(self):

        test_m = np.array([[2., 0., 0., 1.],
                           [0., 3., 0., 0.],
                           [0., 0., 4., 0.],
                           [1., 0., 0., 5.]])

        m = COOMatrix(test_m)
        sm = _convert_matrix_to_symmetric(m)
        self.assertTrue(sm.is_symmetric)
        mm = sm.toarray()
        self.assertTrue(np.allclose(mm, test_m, atol=1e-6))


@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestCOOSymMatrix(unittest.TestCase):

    def setUp(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1., 3., 4., 5., 1.])
        m = COOMatrix((data, (row, col)), shape=(4, 4))
        m.name = 'basic_matrix'
        self.full_m = m

        row = np.array([0, 3, 1, 2, 3])
        col = np.array([0, 0, 1, 2, 3])
        data = np.array([2., 1., 3., 4., 5.])
        m = COOSymMatrix((data, (row, col)), shape=(4, 4))
        m.name = 'basic_sym_matrix'
        self.basic_m = m

        row = np.array([0, 0, 1, 2, 3])
        col = np.array([0, 3, 1, 2, 3])
        data = np.array([2., 1., 3., 4., 5.])
        m = COOSymMatrix((data, (col, row)), shape=(4, 4))
        m.name = 'basic_sym_matrix'
        self.transposed = m

        row = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5])
        col = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5])
        data = np.array([36., 17., 33., 19., 18., 43., 12., 11., 13., 18.,  8.,  7.,  8.,
                         6.,  9., 15., 14., 16., 11.,  8., 29.])
        m = COOSymMatrix((data, (row, col)), shape=(6, 6))
        m.name = 'G'
        self.g_matrix = m

    def test_is_symmetric(self):
        self.assertTrue(self.basic_m.is_symmetric)

    def test_name(self):
        self.assertEqual(self.basic_m.name, 'basic_sym_matrix')
        self.basic_m.name = 'hola'
        self.assertEqual(self.basic_m.name, 'hola')

    def test_shape(self):
        self.assertEqual(self.basic_m.shape[0], 4)
        self.assertEqual(self.basic_m.shape[1], 4)
        #self.assertRaises(RuntimeError, setattr, self.basic_m, 'shape', (3, 3))

    # ToDo: this should be creating CSRMATRIX to check later
    @unittest.skipIf(sys.version_info < (3, 0), "not supported in this version")
    def test_tocsr(self):

        csrm = self.basic_m.tocsr()
        m = self.basic_m
        scipym = csr_matrix((m.data,
                             (m.row, m.col)),
                            shape=(4, 4))
        self.assertListEqual(csrm.indices.tolist(), scipym.indices.tolist())
        self.assertListEqual(csrm.indptr.tolist(), scipym.indptr.tolist())
        self.assertListEqual(csrm.data.tolist(), scipym.data.tolist())
        self.assertEqual(csrm.shape, scipym.shape)
        self.assertIsInstance(csrm, SparseBase)

    # ToDo: this should be creating CSCMATRIX to check later
    @unittest.skipIf(sys.version_info < (3, 0), "not supported in this version")
    def test_tocsc(self):
        cscm = self.basic_m.tocsc()
        m = self.basic_m
        scipym = csc_matrix((m.data,
                             (m.row, m.col)),
                            shape=(4, 4))
        self.assertListEqual(cscm.indices.tolist(), scipym.indices.tolist())
        self.assertListEqual(cscm.indptr.tolist(), scipym.indptr.tolist())
        self.assertListEqual(cscm.data.tolist(), scipym.data.tolist())
        self.assertEqual(cscm.shape, scipym.shape)
        self.assertIsInstance(cscm, SparseBase)

    def test_tofullcoo(self):

        full_coo = self.basic_m.tofullcoo()
        full_row = np.sort(full_coo.row)
        full_col = np.sort(full_coo.col)
        full_data = np.sort(full_coo.data)
        full_row2 = np.sort(self.full_m.row)
        full_col2 = np.sort(self.full_m.col)
        full_data2 = np.sort(self.full_m.data)
        self.assertListEqual(full_row.tolist(), full_row2.tolist())
        self.assertListEqual(full_col.tolist(), full_col2.tolist())
        self.assertListEqual(full_data.tolist(), full_data2.tolist())

    def test_tofullcsr(self):
        csrm = self.basic_m.tofullcsr()
        m = self.full_m
        scipym = csr_matrix((m.data,
                             (m.row, m.col)),
                            shape=(4, 4))
        self.assertListEqual(csrm.indices.tolist(), scipym.indices.tolist())
        self.assertListEqual(csrm.indptr.tolist(), scipym.indptr.tolist())
        self.assertListEqual(csrm.data.tolist(), scipym.data.tolist())
        self.assertEqual(csrm.shape, scipym.shape)
        self.assertIsInstance(csrm, SparseBase)

    def test_tofullcsc(self):
        cscm = self.basic_m.tofullcsc()
        m = self.full_m
        scipym = csc_matrix((m.data,
                             (m.row, m.col)),
                            shape=(4, 4))
        self.assertListEqual(cscm.indices.tolist(), scipym.indices.tolist())
        self.assertListEqual(cscm.indptr.tolist(), scipym.indptr.tolist())
        self.assertListEqual(cscm.data.tolist(), scipym.data.tolist())
        self.assertEqual(cscm.shape, scipym.shape)
        self.assertIsInstance(cscm, SparseBase)

    def test_to_array(self):

        m1 = self.basic_m.tofullcoo()
        m2 = self.full_m
        arr1 = m1.toarray().flatten()
        arr2 = m2.toarray().flatten()
        self.assertListEqual(np.sort(arr1).tolist(), np.sort(arr2).tolist())

    def test_add_dense(self):

        m = self.basic_m
        m2 = self.basic_m.todense()
        m3 = self.basic_m.todense()*2
        result = m + m2
        flat_result = result.flatten()
        flat_compare = m3.flatten()
        self.assertListEqual(flat_result.tolist(), flat_compare.tolist())

    def test_dot(self):
        x = np.array([1, 2, 3, 4], dtype=np.float64)
        m = self.basic_m
        mfull = self.full_m
        res = m.dot(x)
        res_compare = mfull.dot(x)
        self.assertListEqual(res.tolist(), res_compare.tolist())

    def test_transpose(self):
        m = self.basic_m.transpose()
        m2 = self.transposed

        row = np.sort(m.row)
        col = np.sort(m.col)
        data = np.sort(m.data)
        trow = np.sort(m2.row)
        tcol = np.sort(m2.col)
        tdata = np.sort(m2.data)
        self.assertListEqual(row.tolist(), trow.tolist())
        self.assertListEqual(col.tolist(), tcol.tolist())
        self.assertListEqual(data.tolist(), tdata.tolist())

    def test_getrow(self):
        m = self.g_matrix
        m_row = m.getrow(0)
        self.assertIsInstance(m_row, CSRMatrix)
        self.assertEqual(m_row.shape, (1, m.shape[1]))
        values = m_row.toarray()[0]
        tvalues = [36., 17., 19., 12.,  8., 15.]
        self.assertListEqual(values.tolist(), tvalues)

    def test_getcol(self):
        m = self.g_matrix
        m_col = m.getcol(0)
        self.assertIsInstance(m_col, CSCMatrix)
        self.assertEqual(m_col.shape, (m.shape[1], 1))
        values = m_col.toarray().transpose()[0]
        tvalues = [36., 17., 19., 12., 8., 15.]
        self.assertListEqual(values.tolist(), tvalues)

    def test_add_sparse(self):
        m = self.basic_m
        mm = m + m
        test_m = np.array([[2., 0., 0., 1.],
                           [0., 3., 0., 0.],
                           [0., 0., 4., 0.],
                           [1., 0., 0., 5.]])
        mm2 = test_m * 2
        self.assertIsInstance(mm, CSRSymMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), mm2.flatten().tolist())

        row = np.array([0, 3, 2])
        col = np.array([0, 0, 2])
        data = np.array([2., 1., 4.])
        m2 = COOSymMatrix((data, (row, col)), shape=(4, 4))

        test_m = np.array([[4., 0., 0., 2.],
                           [0., 3., 0., 0.],
                           [0., 0., 8., 0.],
                           [2., 0., 0., 5.]])

        mm = m + m2
        self.assertIsInstance(mm, CSRSymMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), test_m.flatten().tolist())

        row = np.array([0, 3, 1, 0, 2])
        col = np.array([0, 3, 1, 2, 1])
        data = np.array([4., 5., 7., 9., 6.])
        m2 = COOMatrix((data, (row, col)), shape=(4, 4))

        mm = m + m2

        test_m = np.array([[6., 0., 9., 1.],
                           [0., 10., 0., 0.],
                           [0., 6., 4., 0.],
                           [1., 0., 0., 10.]])
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), test_m.flatten().tolist())

        mm = m2 + m
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), test_m.flatten().tolist())

    def test_sub_sparse(self):
        m = self.basic_m
        mm = m - m
        mm2 = np.zeros(m.shape, dtype=np.double)
        self.assertIsInstance(mm, CSRSymMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), mm2.flatten().tolist())

        row = np.array([0, 3, 2])
        col = np.array([0, 0, 2])
        data = np.array([2., 1., 4.])
        m2 = COOSymMatrix((data, (row, col)), shape=(4, 4))

        test_m = np.array([[0., 0., 0., 0.],
                           [0., 3., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 5.]])

        mm = m - m2
        self.assertIsInstance(mm, CSRSymMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), test_m.flatten().tolist())

        row = np.array([0, 3, 1, 0, 2])
        col = np.array([0, 3, 1, 2, 1])
        data = np.array([4., 5., 7., 9., 6.])
        m2 = COOMatrix((data, (row, col)), shape=(4, 4))

        mm = m - m2
        test_m = np.array([[-2., 0., -9., 1.],
                           [0., -4., 0., 0.],
                           [0., -6., 4., 0.],
                           [1., 0., 0., 0]])
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), test_m.flatten().tolist())

        test_m = np.array([[2., 0., 9., -1.],
                           [0., 4., 0., 0.],
                           [0., 6., -4., 0.],
                           [-1., 0., 0., 0]])

        mm = m2 - m
        self.assertIsInstance(mm, CSRMatrix)
        self.assertListEqual(mm.toarray().flatten().tolist(), test_m.flatten().tolist())

    def test_mul_sparse_matrix(self):

        # test symmetric times symmetric
        m = self.g_matrix
        dense_m = m.toarray()
        res = m * m
        dense_res = np.matmul(dense_m, dense_m)
        self.assertTrue(res.is_symmetric)
        self.assertTrue(np.allclose(res.toarray(), dense_res))

        # test symmetric times unsymmetric
        m = self.basic_m
        dense_m2 = np.array([[1.0, 2.0],
                             [3.0, 4.0],
                             [5.0, 6.0],
                             [7.0, 8.0]])

        m2 = COOMatrix(dense_m2)
        res = m * m2
        dense_res = np.matmul(m.toarray(), dense_m2)
        self.assertFalse(res.is_symmetric)
        self.assertTrue(np.allclose(res.toarray(), dense_res))

        # test symmetric times full symmetric
        m2 = self.full_m
        dense_m = m.toarray()
        dense_m2 = m2.toarray()
        res = m * m2
        dense_res = np.matmul(dense_m, dense_m2)
        self.assertTrue(res.is_symmetric)
        self.assertTrue(np.allclose(res.toarray(), dense_res))

        # test symmetric times full scipycoo
        m2 = coo_matrix((self.full_m.data,
                        (self.full_m.row, self.full_m.col)),
                        shape=self.full_m.shape)

        dense_m = m.toarray()
        dense_m2 = m2.toarray()
        with self.assertRaises(Exception) as context:
            res = m * m2


        m = self.basic_m
        dense_m2 = np.array([[1.0, 2.0],
                             [3.0, 4.0],
                             [5.0, 6.0],
                             [7.0, 8.0]])

        m2 = coo_matrix(dense_m2)
        with self.assertRaises(Exception) as context:
            res = m * m2

        # ToDo: add test with block matrix

    # ToDo: add tests for getallnnz

@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestEmptyMatrix(unittest.TestCase):

    def test_constructor(self):

        m = EmptyMatrix(3, 3)
        self.assertEqual(m.shape, (3, 3))
        self.assertTrue(m.is_symmetric)
        self.assertEqual(m.nnz, 0)
        self.assertEqual(m.getallnnz(), 0)

        m = EmptyMatrix(3, 1)
        self.assertFalse(m.is_symmetric)

    def test_tocsr(self):
        m = EmptyMatrix(3, 1)
        mcsr = m.tocsr()
        self.assertIsInstance(mcsr, CSRMatrix)

    def test_tocsc(self):
        m = EmptyMatrix(3, 1)
        mcsc = m.tocsc()
        self.assertIsInstance(mcsc, CSCMatrix)


@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestIdentityMatrix(unittest.TestCase):

    def test_constructor(self):

        m = IdentityMatrix(3)
        self.assertEqual(m.shape, (3, 3))
        self.assertTrue(m.is_symmetric)
        self.assertEqual(m.nnz, 3)
        self.assertEqual(m.getallnnz(), 3)
        self.assertListEqual(m.data.tolist(), [1.0]*3)





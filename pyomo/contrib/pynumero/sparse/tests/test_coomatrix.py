from pyomo.contrib.pynumero.sparse import (COOMatrix,
                             COOSymMatrix,
                             SparseBase)

from scipy.sparse.csr import csr_matrix
from scipy.sparse.csc import csc_matrix
from scipy.sparse.coo import coo_matrix
import numpy as np
import unittest
import sys

class TestCOOMatrix(unittest.TestCase):

    def setUp(self):

        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4, 5, 7, 9])
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


class TestCOOSymMatrix(unittest.TestCase):

    def setUp(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2, 1, 3, 4, 5, 1])
        m = COOMatrix((data, (row, col)), shape=(4, 4))
        m.name = 'basic_matrix'
        self.full_m = m

        row = np.array([0, 3, 1, 2, 3])
        col = np.array([0, 0, 1, 2, 3])
        data = np.array([2, 1, 3, 4, 5])
        m = COOSymMatrix((data, (row, col)), shape=(4, 4))
        m.name = 'basic_sym_matrix'
        self.basic_m = m

        row = np.array([0, 0, 1, 2, 3])
        col = np.array([0, 3, 1, 2, 3])
        data = np.array([2, 1, 3, 4, 5])
        m = COOSymMatrix((data, (col, row)), shape=(4, 4))
        m.name = 'basic_sym_matrix'
        self.transposed = m

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
    @unittest.skipIf(sys.version_info < (3, 0), "not supported in this veresion")
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
    @unittest.skipIf(sys.version_info < (3, 0), "not supported in this veresion")
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

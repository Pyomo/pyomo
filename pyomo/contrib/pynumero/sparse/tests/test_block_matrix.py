import pyutilib.th as unittest
try:
    from scipy.sparse import bmat
    import numpy as np
except ImportError:
    raise unittest.SkipTest(
        "Pynumero needs scipy and numpy to run block matrix tests")

from pyomo.contrib.pynumero.extensions.sparseutils import SparseLib
if not SparseLib.available():
    raise unittest.SkipTest(
        "Pynumero needs the SparseUtils extension to run block matrix tests")

from pyomo.contrib.pynumero.sparse import (COOMatrix,
                                           COOSymMatrix,
                                           BlockMatrix,
                                           BlockSymMatrix,
                                           SparseBase,
                                           BlockVector)

class TestBlockMatrix(unittest.TestCase):
    def setUp(self):
        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2, 1, 3, 4, 5, 1])
        m = COOMatrix((data, (row, col)), shape=(4, 4))

        self.block_m = m

        bm = BlockMatrix(2, 2)
        bm.name = 'basic_matrix'
        bm[0, 0] = m
        bm[1, 1] = m
        bm[0, 1] = m
        self.basic_m = bm

    def test_is_symmetric(self):
        self.assertFalse(self.basic_m.is_symmetric)

    def test_name(self):
        self.assertEqual(self.basic_m.name, 'basic_matrix')
        self.basic_m.name = 'hola'
        self.assertEqual(self.basic_m.name, 'hola')

    def test_bshape(self):
        self.assertRaises(RuntimeError, setattr, self.basic_m, 'bshape', (2, 2))

    def test_shape(self):
        shape = (self.block_m.shape[0]*2, self.block_m.shape[1]*2)
        self.assertRaises(RuntimeError, setattr, self.basic_m, 'shape', shape)

    def test_tocoo(self):

        block = self.block_m
        m = self.basic_m
        scipy_mat = bmat([[block, block], [None, block]], format='coo')
        dinopy_mat = m.tocoo()
        drow = np.sort(dinopy_mat.row)
        dcol = np.sort(dinopy_mat.col)
        ddata = np.sort(dinopy_mat.data)
        srow = np.sort(scipy_mat.row)
        scol = np.sort(scipy_mat.col)
        sdata = np.sort(scipy_mat.data)
        self.assertListEqual(drow.tolist(), srow.tolist())
        self.assertListEqual(dcol.tolist(), scol.tolist())
        self.assertListEqual(ddata.tolist(), sdata.tolist())

    def test_tocsr(self):

        block = self.block_m
        m = self.basic_m
        scipy_mat = bmat([[block, block], [None, block]], format='csr')
        dinopy_mat = m.tocsr()
        dindices = np.sort(dinopy_mat.indices)
        dindptr = np.sort(dinopy_mat.indptr)
        ddata = np.sort(dinopy_mat.data)
        sindices = np.sort(scipy_mat.indices)
        sindptr = np.sort(scipy_mat.indptr)
        sdata = np.sort(scipy_mat.data)
        self.assertListEqual(dindices.tolist(), sindices.tolist())
        self.assertListEqual(dindptr.tolist(), sindptr.tolist())
        self.assertListEqual(ddata.tolist(), sdata.tolist())

    def test_tocsc(self):
        block = self.block_m
        m = self.basic_m
        scipy_mat = bmat([[block, block], [None, block]], format='csc')
        dinopy_mat = m.tocsc()
        dindices = np.sort(dinopy_mat.indices)
        dindptr = np.sort(dinopy_mat.indptr)
        ddata = np.sort(dinopy_mat.data)
        sindices = np.sort(scipy_mat.indices)
        sindptr = np.sort(scipy_mat.indptr)
        sdata = np.sort(scipy_mat.data)
        self.assertListEqual(dindices.tolist(), sindices.tolist())
        self.assertListEqual(dindptr.tolist(), sindptr.tolist())
        self.assertListEqual(ddata.tolist(), sdata.tolist())

    def test_multiply(self):

        # check scalar multiplication
        block = self.block_m
        m = self.basic_m * 5.0
        scipy_mat = bmat([[block, block], [None, block]], format='coo')
        mulscipy_mat = scipy_mat * 5.0
        dinopy_mat = m.tocoo()
        drow = np.sort(dinopy_mat.row)
        dcol = np.sort(dinopy_mat.col)
        ddata = np.sort(dinopy_mat.data)
        srow = np.sort(mulscipy_mat.row)
        scol = np.sort(mulscipy_mat.col)
        sdata = np.sort(mulscipy_mat.data)
        self.assertListEqual(drow.tolist(), srow.tolist())
        self.assertListEqual(dcol.tolist(), scol.tolist())
        self.assertListEqual(ddata.tolist(), sdata.tolist())

        m = 5.0 * self.basic_m
        dinopy_mat = m.tocoo()
        drow = np.sort(dinopy_mat.row)
        dcol = np.sort(dinopy_mat.col)
        ddata = np.sort(dinopy_mat.data)
        self.assertListEqual(drow.tolist(), srow.tolist())
        self.assertListEqual(dcol.tolist(), scol.tolist())
        self.assertListEqual(ddata.tolist(), sdata.tolist())

        # check dot product with block vector
        block = self.block_m
        m = self.basic_m
        scipy_mat = bmat([[block, block], [None, block]], format='coo')
        x = BlockVector(2)
        x[0] = np.ones(block.shape[1], dtype=np.float64)
        x[1] = np.ones(block.shape[1], dtype=np.float64)

        res_scipy = scipy_mat.dot(x.flatten())
        res_dinopy = m * x
        res_dinopy_flat = m * x.flatten()

        self.assertListEqual(res_dinopy.tolist(), res_scipy.tolist())
        self.assertListEqual(res_dinopy_flat.tolist(), res_scipy.tolist())

    def test_getitem(self):

        m = BlockMatrix(3, 3)
        for i in range(3):
            for j in range(3):
                self.assertIsNone(m[i, j])

        m[0, 1] = self.block_m
        self.assertIsInstance(m[0, 1], SparseBase)
        self.assertEqual(m[0, 1].shape, self.block_m.shape)

    def test_setitem(self):

        m = BlockMatrix(2, 2)
        m[0, 1] = self.block_m
        self.assertFalse(m.is_empty_block(0, 1))
        self.assertEqual(m.row_block_sizes()[0], self.block_m.shape[0])
        self.assertEqual(m.col_block_sizes()[1], self.block_m.shape[1])
        self.assertIsInstance(m[0, 1], SparseBase)
        self.assertEqual(m[0, 1].shape, self.block_m.shape)

    def test_coo_data(self):
        m = self.basic_m.tocoo()
        data = self.basic_m.coo_data()
        self.assertListEqual(m.data.tolist(), data.tolist())

    # ToDo: add tests for block matrices with block matrices in it
    # ToDo: add tests for matrices with zeros in the diagonal
    # ToDo: add tests for getallnnz
    # ToDo: add tests for block matrices with coo and csc matrices

class TestSymBlockMatrix(unittest.TestCase):

    def setUp(self):

        row = np.array([0, 1, 4, 1, 2, 7, 2, 3, 5, 3, 4, 5, 4, 7, 5, 6, 6, 7])
        col = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7])
        data = np.array([27, 5, 12, 56, 66, 34, 94, 31, 41, 7, 98, 72, 24, 33, 78, 47, 98, 41])
        m = COOSymMatrix((data, (row, col)), shape=(8, 8))

        self.block00 = m

        row = np.array([0, 3, 1, 0])
        col = np.array([0, 3, 1, 2])
        data = np.array([4, 5, 7, 9])
        m = COOMatrix((data, (row, col)), shape=(4, 8))

        self.block10 = m

        row = np.array([0, 1, 2, 3])
        col = np.array([0, 1, 2, 3])
        data = np.array([1, 1, 1, 1])
        m = COOSymMatrix((data, (row, col)), shape=(4, 4))

        self.block11 = m

        bm = BlockSymMatrix(2)
        bm.name = 'basic_matrix'
        bm[0, 0] = self.block00
        bm[1, 0] = self.block10
        bm[1, 1] = self.block11
        self.basic_m = bm

    def test_is_symmetric(self):
        self.assertTrue(self.basic_m.is_symmetric)

    def test_getitem(self):
        self.assertRaises(RuntimeError, self.basic_m.__getitem__, (0, 1))

    def test_tofullmatrix(self):
        m = self.basic_m.tofullmatrix()
        a = m.toarray()
        self.assertTrue(np.allclose(a, a.T, atol=1e-3))

    def test_tocoo(self):
        m = self.basic_m.tocoo()
        a = m.toarray()
        self.assertTrue(np.allclose(a, a.T, atol=1e-3))

    def test_coo_data(self):
        m = self.basic_m.tocoo()
        data = self.basic_m.coo_data()
        self.assertListEqual(m.data.tolist(), data.tolist())

    def test_multiply(self):

        # test scalar multiplication
        m = self.basic_m * 5.0
        dense_m = m.todense()

        b00 = self.block00.tofullcoo()
        b11 = self.block11.tofullcoo()
        b10 = self.block10
        scipy_m = bmat([[b00, b10.transpose()], [b10, b11]], format='coo')
        dense_scipy_m = scipy_m.todense() * 5.0

        self.assertTrue(np.allclose(dense_scipy_m, dense_m, atol=1e-3))

        m = 5.0 * self.basic_m
        dense_m = m.todense()

        self.assertTrue(np.allclose(dense_scipy_m, dense_m, atol=1e-3))

        # test matrix vector product
        m = self.basic_m
        x = BlockVector(m.bshape[1])
        for i in range(m.bshape[1]):
            x[i] = np.ones(m.col_block_sizes()[i], dtype=np.float64)
        dinopy_res = m * x
        scipy_res = scipy_m * x.flatten()

        self.assertListEqual(dinopy_res.tolist(), scipy_res.tolist())

        dinopy_res = m * x.flatten()
        scipy_res = scipy_m * x.flatten()

        self.assertListEqual(dinopy_res.tolist(), scipy_res.tolist())

    # ToDo: Add test for transpose






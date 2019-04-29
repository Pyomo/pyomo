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

from pyomo.contrib.pynumero import numpy_available, scipy_available
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run BlockMatrix tests")

from scipy.sparse import coo_matrix, bmat
import numpy as np

from pyomo.contrib.pynumero.sparse import (BlockMatrix,
                                           BlockSymMatrix,
                                           BlockVector)
import warnings


class TestBlockMatrix(unittest.TestCase):
    def setUp(self):
        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))

        self.block_m = m

        bm = BlockMatrix(2, 2)
        bm.name = 'basic_matrix'
        bm[0, 0] = m
        bm[1, 1] = m
        bm[0, 1] = m
        self.basic_m = bm

        self.composed_m = BlockMatrix(2, 2)
        self.composed_m[0, 0] = self.block_m
        self.composed_m[1, 1] = self.basic_m

    def test_name(self):
        self.assertEqual(self.basic_m.name, 'basic_matrix')
        self.basic_m.name = 'hola'
        self.assertEqual(self.basic_m.name, 'hola')

    def test_bshape(self):
        self.assertEqual(self.basic_m.bshape, (2, 2))

    def test_shape(self):
        shape = (self.block_m.shape[0]*2, self.block_m.shape[1]*2)
        self.assertEqual(self.basic_m.shape, shape)

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

        dense_mat = dinopy_mat.toarray()
        self.basic_m *= 5.0
        self.assertTrue(np.allclose(dense_mat, self.basic_m.toarray()))

        flat_mat = self.basic_m.tocoo()
        result = flat_mat * flat_mat
        dense_result = result.toarray()
        mat = self.basic_m * self.basic_m.tocoo()
        dense_mat = mat.toarray()
        self.assertTrue(np.allclose(dense_mat, dense_result))

    def test_mul_sparse_matrix(self):
        m = self.basic_m

        flat_prod = m.tocoo() * m.tocoo()
        prod = m * m

        self.assertIsInstance(prod, BlockMatrix)
        self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))

        prod = m * m.tocoo()
        self.assertIsInstance(prod, BlockMatrix)
        self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))

        prod = m.tocoo() * m
        self.assertIsInstance(prod, BlockMatrix)
        self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))

        m2 = m.copy_structure()
        ones = np.ones(m.shape)
        m2.copyfrom(ones)
        flat_prod = m.tocoo() * m2.tocoo()
        prod = m * m2

        self.assertIsInstance(prod, BlockMatrix)
        self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))

        prod = m * m2.tocoo()
        self.assertIsInstance(prod, BlockMatrix)
        self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))

        prod = m.tocoo() * m2
        self.assertIsInstance(prod, BlockMatrix)
        self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))

    def test_getitem(self):

        m = BlockMatrix(3, 3)
        for i in range(3):
            for j in range(3):
                self.assertIsNone(m[i, j])

        m[0, 1] = self.block_m
        self.assertEqual(m[0, 1].shape, self.block_m.shape)

    def test_setitem(self):

        m = BlockMatrix(2, 2)
        m[0, 1] = self.block_m
        self.assertFalse(m.is_empty_block(0, 1))
        self.assertEqual(m.row_block_sizes()[0], self.block_m.shape[0])
        self.assertEqual(m.col_block_sizes()[1], self.block_m.shape[1])
        self.assertEqual(m[0, 1].shape, self.block_m.shape)

    def test_coo_data(self):
        m = self.basic_m.tocoo()
        data = self.basic_m.coo_data()
        self.assertListEqual(m.data.tolist(), data.tolist())

    # ToDo: add tests for block matrices with block matrices in it
    # ToDo: add tests for matrices with zeros in the diagonal
    # ToDo: add tests for block matrices with coo and csc matrices

    def test_nnz(self):
        self.assertEqual(self.block_m.nnz*3, self.basic_m.nnz)

    def test_block_shapes(self):
        shapes = self.basic_m.block_shapes()
        for i in range(self.basic_m.bshape[0]):
            for j in range(self.basic_m.bshape[1]):
                self.assertEqual(shapes[i][j], self.block_m.shape)

    def test_dot(self):
        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        x = np.ones(A_dense.shape[1])
        block_x = BlockVector(2)
        block_x[0] = np.ones(self.block_m.shape[1])
        block_x[1] = np.ones(self.block_m.shape[1])
        flat_res = A_block.dot(x).flatten()
        block_res = A_block.dot(block_x)
        self.assertTrue(np.allclose(A_dense.dot(x), flat_res))
        self.assertTrue(np.allclose(A_dense.dot(x), block_res.flatten()))
        self.assertEqual(block_res.bshape[0], 2)

    def test_reset_brow(self):
        self.basic_m.reset_brow(0)
        for j in range(self.basic_m.bshape[1]):
            self.assertIsNone(self.basic_m[0, j])

    def test_reset_bcol(self):
        self.basic_m.reset_bcol(0)
        for j in range(self.basic_m.bshape[0]):
            self.assertIsNone(self.basic_m[j, 0])

    def test_to_scipy(self):

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

    def test_has_empty_rows(self):
        self.assertFalse(self.basic_m.has_empty_rows())

    def test_has_empty_cols(self):
        self.assertFalse(self.basic_m.has_empty_cols())

    def test_transpose(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        A_dense_t = A_dense.transpose()
        A_block_t = A_block.transpose()
        self.assertTrue(np.allclose(A_dense_t, A_block_t.toarray()))

        A_dense = self.composed_m.toarray()
        A_block = self.composed_m
        A_dense_t = A_dense.transpose()
        A_block_t = A_block.transpose()
        self.assertTrue(np.allclose(A_dense_t, A_block_t.toarray()))

    def test_repr(self):
        self.assertEqual(len(self.basic_m.__repr__()), 17)

    def test_set_item(self):

        self.basic_m[1, 0] = None
        self.assertIsNone(self.basic_m[1, 0])
        self.basic_m[1, 1] = None
        self.assertIsNone(self.basic_m[1, 1])
        self.assertEqual(self.basic_m._brow_lengths[1], 0)
        self.basic_m[1, 1] = self.block_m
        self.assertEqual(self.basic_m._brow_lengths[1], self.block_m.shape[1])

    def test_add(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m

        aa = A_dense + A_dense
        mm = A_block + A_block

        self.assertTrue(np.allclose(aa, mm.toarray()))

        mm = A_block.__radd__(A_block)
        self.assertTrue(np.allclose(aa, mm.toarray()))

        r = A_block + A_block.tocoo()
        dense_res = A_block.toarray() + A_block.toarray()
        self.assertIsInstance(r, BlockMatrix)
        self.assertTrue(np.allclose(r.toarray(), dense_res))

        r = A_block.tocoo() + A_block
        dense_res = A_block.toarray() + A_block.toarray()
        #self.assertIsInstance(r, BlockMatrix)
        self.assertTrue(np.allclose(r.toarray(), dense_res))

        r = A_block + 2 * A_block.tocoo()
        dense_res = A_block.toarray() + 2 * A_block.toarray()
        self.assertIsInstance(r, BlockMatrix)
        self.assertTrue(np.allclose(r.toarray(), dense_res))

        r = 2 * A_block.tocoo() + A_block
        dense_res = 2 * A_block.toarray() + A_block.toarray()
        #self.assertIsInstance(r, BlockMatrix)
        self.assertTrue(np.allclose(r.toarray(), dense_res))

        r = A_block.T + A_block.tocoo()
        dense_res = A_block.toarray().T + A_block.toarray()
        self.assertIsInstance(r, BlockMatrix)
        self.assertTrue(np.allclose(r.toarray(), dense_res))

        with self.assertRaises(Exception) as context:
            mm = A_block.__radd__(A_block.toarray())

        with self.assertRaises(Exception) as context:
            mm = A_block + A_block.toarray()

        with self.assertRaises(Exception) as context:
            mm = A_block + 1.0

    def test_sub(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        A_block2 = 2 * self.basic_m

        aa = A_dense - A_dense
        mm = A_block - A_block

        self.assertTrue(np.allclose(aa, mm.toarray()))
        mm = A_block.__rsub__(A_block)
        self.assertTrue(np.allclose(aa, mm.toarray()))

        mm = A_block2 - A_block.tocoo()
        self.assertTrue(np.allclose(A_block.toarray(), mm.toarray()))

        mm = A_block2.tocoo() - A_block
        self.assertTrue(np.allclose(A_block.toarray(), mm.toarray()))

        mm = A_block2.T - A_block.tocoo()
        dense_r = A_block2.toarray().T - A_block.toarray()
        self.assertTrue(np.allclose(dense_r, mm.toarray()))

        with self.assertRaises(Exception) as context:
            mm = A_block.__rsub__(A_block.toarray())

        with self.assertRaises(Exception) as context:
            mm = A_block - A_block.toarray()

        with self.assertRaises(Exception) as context:
            mm = A_block - 1.0

        with self.assertRaises(Exception) as context:
            mm = 1.0 - A_block

    def test_neg(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m

        aa = -A_dense
        mm = -A_block

        self.assertTrue(np.allclose(aa, mm.toarray()))

    def test_copyfrom(self):
        A_dense = self.basic_m.toarray()
        A_block = 2 * self.basic_m
        A_block2 = 2 * self.basic_m

        A_block.copyfrom(self.basic_m.tocoo())

        dd = A_block.toarray()
        self.assertTrue(np.allclose(dd, A_dense))

        A_block = 2 * self.basic_m
        flat = np.ones(A_block.shape)
        A_block.copyfrom(flat)
        self.assertTrue(np.allclose(flat, A_block.toarray()))

        A_block = self.basic_m.copy_structure()
        to_copy = 2 * self.basic_m
        A_block.copyfrom(to_copy)

        dd = A_block.toarray()
        self.assertTrue(np.allclose(dd, A_block2.toarray()))

    def test_copy(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        clone = A_block.copy()
        self.assertTrue(np.allclose(clone.toarray(), A_dense))

    def test_iadd(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m.copy()
        A_dense += A_dense
        A_block += A_block

        self.assertTrue(np.allclose(A_block.toarray(), A_dense))

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m.copy()
        A_dense += A_dense
        A_block += A_block.tocoo()

        self.assertTrue(np.allclose(A_block.toarray(), A_dense))

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m.copy()
        A_block += 2 * A_block.tocoo()

        self.assertTrue(np.allclose(A_block.toarray(), 3 * A_dense))

        with self.assertRaises(Exception) as context:
            A_block += 1.0

    def test_isub(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        A_dense -= A_dense
        A_block -= A_block

        self.assertTrue(np.allclose(A_block.toarray(), A_dense))

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        A_dense -= A_dense
        A_block -= A_block.tocoo()

        self.assertTrue(np.allclose(A_block.toarray(), A_dense))

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m.copy()
        A_block -= 2 * A_block.tocoo()

        self.assertTrue(np.allclose(A_block.toarray(), -A_dense))

        with self.assertRaises(Exception) as context:
            A_block -= 1.0

    def test_imul(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        A_dense *= 3
        A_block *= 3.

        self.assertTrue(np.allclose(A_block.toarray(), A_dense))

        with self.assertRaises(Exception) as context:
            A_block *= A_block

        with self.assertRaises(Exception) as context:
            A_block *= A_block.tocoo()

        with self.assertRaises(Exception) as context:
            A_block *= A_block.toarray()

    @unittest.skip('Ignore this for now')
    def test_itruediv(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        A_dense /= 3
        A_block /= 3.

        self.assertTrue(np.allclose(A_block.toarray(), A_dense))

        with self.assertRaises(Exception) as context:
            A_block /= A_block

        with self.assertRaises(Exception) as context:
            A_block /= A_block.tocoo()

        with self.assertRaises(Exception) as context:
            A_block /= A_block.toarray()

    @unittest.skip('Ignore this for now')
    def test_truediv(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        B_block = A_block / 3.
        self.assertTrue(np.allclose(B_block.toarray(), A_dense/3.))

        with self.assertRaises(Exception) as context:
            b = A_block / A_block

        with self.assertRaises(Exception) as context:
            b = A_block / A_block.tocoo()

        with self.assertRaises(Exception) as context:
            b = A_block / A_block.toarray()

        with self.assertRaises(Exception) as context:
            B_block = 3./ A_block

    def test_eq(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_flat = self.basic_m.tocoo()
            A_block = self.basic_m

            A_bool_flat = A_flat == 2.0
            A_bool_block = A_block == 2.0

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = A_flat == A_flat
            A_bool_block = A_block == A_block

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))


            A_bool_flat = 2.0 != A_flat
            A_bool_block = 2.0 != A_block
            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

    def test_ne(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_flat = self.basic_m.tocoo()
            A_block = self.basic_m

            A_bool_flat = A_flat != 2.0
            A_bool_block = A_block != 2.0
            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = 2.0 != A_flat
            A_bool_block = 2.0 != A_block
            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = A_flat != A_flat
            A_bool_block = A_block != A_block

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

    def test_le(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_flat = self.basic_m.tocoo()
            A_block = self.basic_m

            A_bool_flat = A_flat <= 2.0
            A_bool_block = A_block <= 2.0
            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            # A_bool_flat = 2.0 <= A_flat
            # A_bool_block = 2.0 <= A_block
            # self.assertTrue(np.allclose(A_bool_flat.toarray(),
            #                             A_bool_block.toarray()))

            A_bool_flat = A_flat <= A_flat
            A_bool_block = A_block <= A_block

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = A_flat <= 2 * A_flat
            A_bool_block = A_block <= 2 * A_block

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = 2.0 >= A_flat
            A_bool_block = 2.0 >= A_block
            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

    def test_lt(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_flat = self.basic_m.tocoo()
            A_block = self.basic_m

            A_bool_flat = A_flat < 2.0
            A_bool_block = A_block < 2.0

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            # A_bool_flat = 2.0 <= A_flat
            # A_bool_block = 2.0 <= A_block
            # self.assertTrue(np.allclose(A_bool_flat.toarray(),
            #                             A_bool_block.toarray()))

            A_bool_flat = A_flat < A_flat
            A_bool_block = A_block < A_block

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = A_flat < 2 * A_flat
            A_bool_block = A_block < 2 * A_block

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = 2.0 > A_flat
            A_bool_block = 2.0 > A_block
            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

    def test_ge(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_flat = self.basic_m.tocoo()
            A_block = self.basic_m

            A_bool_flat = A_flat >= 2.0
            A_bool_block = A_block >= 2.0
            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = 2.0 <= A_flat
            A_bool_block = 2.0 <= A_block
            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = A_flat >= A_flat
            A_bool_block = A_block >= A_block

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

            A_bool_flat = A_flat >= 0.5 * A_flat
            A_bool_block = A_block >= 0.5 * A_block

            self.assertTrue(np.allclose(A_bool_flat.toarray(),
                                        A_bool_block.toarray()))

    def test_abs(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = -1.0 * np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))

        self.block_m = m

        bm = BlockMatrix(2, 2)
        bm.name = 'basic_matrix'
        bm[0, 0] = m
        bm[1, 1] = m
        bm[0, 1] = m

        abs_flat = abs(bm.tocoo())
        abs_mat = abs(bm)

        self.assertIsInstance(abs_mat, BlockMatrix)
        self.assertTrue(np.allclose(abs_flat.toarray(),
                                    abs_mat.toarray()))

    def test_getcol(self):

        m = self.basic_m

        flat_mat = m.tocoo()
        flat_col = flat_mat.getcol(2)
        block_col = m.getcol(2)
        self.assertTrue(np.allclose(flat_col.toarray(), block_col.toarray()))

    def test_getrow(self):

        m = self.basic_m

        flat_mat = m.tocoo()
        flat_row = flat_mat.getrow(2)
        block_row = m.getrow(2)
        self.assertTrue(np.allclose(flat_row.toarray(), block_row.toarray()))

    def test_nonzero(self):

        m = self.basic_m

        flat_mat = m.tocoo()
        flat_row, flat_col = flat_mat.nonzero()
        block_row, block_col = m.nonzero()
        self.assertTrue(np.allclose(flat_col, block_col))
        self.assertTrue(np.allclose(flat_row, block_row))

class TestSymBlockMatrix(unittest.TestCase):

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

        bm = BlockSymMatrix(2)
        bm.name = 'basic_matrix'
        bm[0, 0] = self.block00
        bm[1, 0] = self.block10
        bm[1, 1] = self.block11
        self.basic_m = bm

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
        dense_m = m.toarray()

        b00 = self.block00.tocoo()
        b11 = self.block11.tocoo()
        b10 = self.block10
        scipy_m = bmat([[b00, b10.transpose()], [b10, b11]], format='coo')
        dense_scipy_m = scipy_m.toarray() * 5.0

        self.assertTrue(np.allclose(dense_scipy_m, dense_m, atol=1e-3))

        m = 5.0 * self.basic_m
        dense_m = m.toarray()

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

        self.basic_m *= 5.0
        self.assertTrue(np.allclose(self.basic_m.toarray(), dense_m, atol=1e-3))

    def test_transpose(self):

        m = self.basic_m.transpose()
        dense_m = self.basic_m.toarray().transpose()
        self.assertTrue(np.allclose(m.toarray(), dense_m))

        m = self.basic_m.transpose(copy=True)
        dense_m = self.basic_m.toarray().transpose()
        self.assertTrue(np.allclose(m.toarray(), dense_m))

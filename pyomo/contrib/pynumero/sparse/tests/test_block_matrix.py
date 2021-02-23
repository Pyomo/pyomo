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
    numpy as np, numpy_available, scipy_sparse as sp, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest(
        "Pynumero needs scipy and numpy to run BlockMatrix tests")

from scipy.sparse import coo_matrix, bmat

from pyomo.contrib.pynumero.sparse import (BlockMatrix,
                                           BlockVector,
                                           NotFullyDefinedBlockMatrixError)
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
        bm.set_block(0, 0, m.copy())
        bm.set_block(1, 1, m.copy())
        bm.set_block(0, 1, m.copy())
        self.basic_m = bm
        self.dense = np.zeros((8, 8))
        self.dense[0:4, 0:4] = m.toarray()
        self.dense[0:4, 4:8] = m.toarray()
        self.dense[4:8, 4:8] = m.toarray()

        self.composed_m = BlockMatrix(2, 2)
        self.composed_m.set_block(0, 0, self.block_m.copy())
        self.composed_m.set_block(1, 1, self.basic_m.copy())

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
        x.set_block(0, np.ones(block.shape[1], dtype=np.float64))
        x.set_block(1, np.ones(block.shape[1], dtype=np.float64))

        res_scipy = scipy_mat.dot(x.flatten())
        res_dinopy = m * x
        res_dinopy_flat = m * x.flatten()

        self.assertListEqual(res_dinopy.tolist(), res_scipy.tolist())
        self.assertListEqual(res_dinopy_flat.tolist(), res_scipy.tolist())

        dense_mat = dinopy_mat.toarray()
        self.basic_m *= 5.0
        self.assertTrue(np.allclose(dense_mat, self.basic_m.toarray()))

    def test_mul_sparse_matrix(self):
        m = self.basic_m

        flat_prod = m.tocoo() * m.tocoo()
        prod = m * m

        self.assertIsInstance(prod, BlockMatrix)
        self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))

        m2 = m.copy_structure()
        ones = np.ones(m.shape)
        m2.copyfrom(ones)
        flat_prod = m.tocoo() * m2.tocoo()
        prod = m * m2

        self.assertIsInstance(prod, BlockMatrix)
        self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))

    def test_getitem(self):

        m = BlockMatrix(3, 3)
        for i in range(3):
            for j in range(3):
                self.assertIsNone(m.get_block(i, j))

        m.set_block(0, 1, self.block_m)
        self.assertEqual(m.get_block(0, 1).shape, self.block_m.shape)

    def test_setitem(self):

        m = BlockMatrix(2, 2)
        m.set_block(0, 1, self.block_m)
        self.assertFalse(m.is_empty_block(0, 1))
        self.assertEqual(m._brow_lengths[0], self.block_m.shape[0])
        self.assertEqual(m._bcol_lengths[1], self.block_m.shape[1])
        self.assertEqual(m.get_block(0, 1).shape, self.block_m.shape)

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
        block_x.set_block(0, np.ones(self.block_m.shape[1]))
        block_x.set_block(1, np.ones(self.block_m.shape[1]))
        flat_res = A_block.dot(x).flatten()
        block_res = A_block.dot(block_x)
        self.assertTrue(np.allclose(A_dense.dot(x), flat_res))
        self.assertTrue(np.allclose(A_dense.dot(x), block_res.flatten()))
        self.assertEqual(block_res.bshape[0], 2)

        m = BlockMatrix(2, 2)
        sub_m = np.array([[1, 0],
                          [0, 1]])
        sub_m = coo_matrix(sub_m)
        m.set_block(0, 1, sub_m.copy())
        m.set_block(1, 0, sub_m.copy())
        x = np.arange(4)
        res = m*x
        self.assertTrue(np.allclose(res.flatten(), np.array([2, 3, 0, 1])))

    def test_reset_brow(self):
        self.basic_m.reset_brow(0)
        for j in range(self.basic_m.bshape[1]):
            self.assertIsNone(self.basic_m.get_block(0, j))

    def test_reset_bcol(self):
        self.basic_m.reset_bcol(0)
        for j in range(self.basic_m.bshape[0]):
            self.assertIsNone(self.basic_m.get_block(j, 0))

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

    def test_has_undefined_row_sizes(self):
        self.assertFalse(self.basic_m.has_undefined_row_sizes())

    def test_has_undefined_col_sizes(self):
        self.assertFalse(self.basic_m.has_undefined_col_sizes())

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

        self.basic_m.set_block(1, 0, None)
        self.assertIsNone(self.basic_m.get_block(1, 0))
        self.basic_m.set_block(1, 1, None)
        self.assertIsNone(self.basic_m.get_block(1, 1))
        self.assertEqual(self.basic_m._brow_lengths[1], self.block_m.shape[0])
        self.basic_m.set_block(1, 1, self.block_m)
        self.assertEqual(self.basic_m._brow_lengths[1], self.block_m.shape[0])

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
            mm = A_block.toarray() + A_block

        with self.assertRaises(Exception) as context:
            mm = A_block + A_block.toarray()

        with self.assertRaises(Exception) as context:
            mm = A_block + 1.0

    def test_add_copy(self):
        """
        The purpose of this test is to ensure that copying happens correctly when block matrices are added.
        For example, when adding

        [A  B   +  [D  0
         0  C]      E  F]

        we want to make sure that E and B both get copied in the result rather than just placed in the result.
        """
        bm = self.basic_m.copy()
        bmT = bm.transpose()
        res = bm + bmT
        self.assertIsNot(res.get_block(1, 0), bmT.get_block(1, 0))
        self.assertIsNot(res.get_block(0, 1), bm.get_block(0, 1))
        self.assertTrue(np.allclose(res.toarray(), self.dense + self.dense.transpose()))

    def test_sub(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m
        A_block2 = 2 * self.basic_m

        aa = A_dense - A_dense
        mm = A_block - A_block

        self.assertTrue(np.allclose(aa, mm.toarray()))

        mm = A_block2 - A_block.tocoo()
        self.assertTrue(np.allclose(A_block.toarray(), mm.toarray()))

        mm = A_block2.tocoo() - A_block
        self.assertTrue(np.allclose(A_block.toarray(), mm.toarray()))

        mm = A_block2.T - A_block.tocoo()
        dense_r = A_block2.toarray().T - A_block.toarray()
        self.assertTrue(np.allclose(dense_r, mm.toarray()))

        with self.assertRaises(Exception) as context:
            mm = A_block - A_block.toarray()

        with self.assertRaises(Exception) as context:
            mm = A_block - 1.0

        with self.assertRaises(Exception) as context:
            mm = 1.0 - A_block

    def test_sub_copy(self):
        """
        The purpose of this test is to ensure that copying happens correctly when block matrices are subtracted.
        For example, when subtracting

        [A  B   -  [D  0
         0  C]      E  F]

        we want to make sure that E and B both get copied in the result rather than just placed in the result.
        """
        bm = self.basic_m.copy()
        bmT = 2 * bm.transpose()
        res = bm - bmT
        self.assertIsNot(res.get_block(1, 0), bmT.get_block(1, 0))
        self.assertIsNot(res.get_block(0, 1), bm.get_block(0, 1))
        self.assertTrue(np.allclose(res.toarray(), self.dense - 2 * self.dense.transpose()))

    def test_neg(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m

        aa = -A_dense
        mm = -A_block

        self.assertTrue(np.allclose(aa, mm.toarray()))

    def test_copyfrom(self):
        bm0 = self.basic_m.copy()
        bm = bm0.copy_structure()
        self.assertFalse(np.allclose(bm.toarray(), self.dense))
        bm.copyfrom(bm0.tocoo())
        self.assertTrue(np.allclose(bm.toarray(), self.dense))

        flat = np.ones((8, 8))
        bm.copyfrom(flat)
        self.assertTrue(np.allclose(flat, bm.toarray()))

        bm.copyfrom(bm0)
        self.assertTrue(np.allclose(bm.toarray(), self.dense))

        bm.get_block(0, 0).data.fill(1.0)
        self.assertAlmostEqual(bm0.toarray()[0, 0], 2)  # this tests that a deep copy was done
        self.assertAlmostEqual(bm.toarray()[0, 0], 1)

        bm.copyfrom(bm0, deep=False)
        bm.get_block(0, 0).data.fill(1.0)
        self.assertAlmostEqual(bm0.toarray()[0, 0], 1)  # this tests that a shallow copy was done
        self.assertAlmostEqual(bm.toarray()[0, 0], 1)

    def test_copyto(self):
        bm0 = self.basic_m.copy()
        coo = bm0.tocoo()
        coo.data.fill(1.0)
        csr = coo.tocsr()
        csc = coo.tocsc()
        self.assertFalse(np.allclose(coo.toarray(), self.dense))
        self.assertFalse(np.allclose(csr.toarray(), self.dense))
        self.assertFalse(np.allclose(csc.toarray(), self.dense))
        bm0.copyto(coo)
        bm0.copyto(csr)
        bm0.copyto(csc)
        self.assertTrue(np.allclose(coo.toarray(), self.dense))
        self.assertTrue(np.allclose(csr.toarray(), self.dense))
        self.assertTrue(np.allclose(csc.toarray(), self.dense))

        flat = np.ones((8, 8))
        bm0.copyto(flat)
        self.assertTrue(np.allclose(flat, self.dense))

        bm = bm0.copy_structure()
        bm0.copyto(bm)
        self.assertTrue(np.allclose(bm.toarray(), self.dense))

        bm.get_block(0, 0).data.fill(1.0)
        self.assertAlmostEqual(bm0.toarray()[0, 0], 2)  # this tests that a deep copy was done
        self.assertAlmostEqual(bm.toarray()[0, 0], 1)

        bm0.copyto(bm, deep=False)
        bm.get_block(0, 0).data.fill(1.0)
        self.assertAlmostEqual(bm0.toarray()[0, 0], 1)  # this tests that a shallow copy was done
        self.assertAlmostEqual(bm.toarray()[0, 0], 1)

    def test_copy(self):
        clone = self.basic_m.copy()
        self.assertTrue(np.allclose(clone.toarray(), self.dense))
        clone.get_block(0, 0).data.fill(1)
        self.assertAlmostEqual(clone.toarray()[0, 0], 1)
        self.assertAlmostEqual(self.basic_m.toarray()[0, 0], 2)

        bm = self.basic_m.copy()
        clone = bm.copy(deep=False)
        self.assertTrue(np.allclose(clone.toarray(), self.dense))
        clone.get_block(0, 0).data.fill(1)
        self.assertAlmostEqual(clone.toarray()[0, 0], 1)
        self.assertAlmostEqual(bm.toarray()[0, 0], 1)

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
        print(A_dense)
        print(A_block.toarray())
        A_dense *= 3
        print(A_dense)
        print(A_block.toarray())
        A_block *= 3.
        print(A_dense)
        print(A_block.toarray())

        self.assertTrue(np.allclose(A_block.toarray(), A_dense))

        with self.assertRaises(Exception) as context:
            A_block *= A_block

        with self.assertRaises(Exception) as context:
            A_block *= A_block.tocoo()

        with self.assertRaises(Exception) as context:
            A_block *= A_block.toarray()

    def test_itruediv(self):

        A_dense = self.basic_m.toarray()
        A_block = self.basic_m.copy()
        A_dense /= 3
        A_block /= 3.

        self.assertTrue(np.allclose(A_block.toarray(), A_dense))

        with self.assertRaises(Exception) as context:
            A_block /= A_block

        with self.assertRaises(Exception) as context:
            A_block /= A_block.tocoo()

        with self.assertRaises(Exception) as context:
            A_block /= A_block.toarray()

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

    def test_gt(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = self.basic_m.copy()
            B = 2 * A.transpose()

            res = A > B
            expected = A.toarray() > B.toarray()
            self.assertTrue(np.allclose(res.toarray(), expected))

    def test_abs(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = -1.0 * np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))

        self.block_m = m

        bm = BlockMatrix(2, 2)
        bm.set_block(0, 0, m)
        bm.set_block(1, 1, m)
        bm.set_block(0, 1, m)

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
        self.assertTrue(np.allclose(flat_col.toarray().flatten(),
                                    block_col.flatten()))

        flat_col = flat_mat.getcol(4)
        block_col = m.getcol(4)
        self.assertTrue(np.allclose(flat_col.toarray().flatten(),
                                    block_col.flatten()))

        flat_col = flat_mat.getcol(6)
        block_col = m.getcol(6)
        self.assertTrue(np.allclose(flat_col.toarray().flatten(),
                                    block_col.flatten()))

    def test_getrow(self):

        m = self.basic_m

        flat_mat = m.tocoo()
        flat_row = flat_mat.getrow(2)
        block_row = m.getrow(2)
        self.assertTrue(np.allclose(flat_row.toarray().flatten(),
                                    block_row.flatten()))

        flat_row = flat_mat.getrow(7)
        block_row = m.getrow(7)
        self.assertTrue(np.allclose(flat_row.toarray().flatten(),
                                    block_row.flatten()))

    def test_nonzero(self):

        m = self.basic_m
        flat_mat = m.tocoo()
        flat_row, flat_col = flat_mat.nonzero()
        with self.assertRaises(Exception) as context:
            block_row, block_col = m.nonzero()

    def test_get_block_column_index(self):

        m = BlockMatrix(2,4)
        m.set_block(0, 0, coo_matrix((3, 2)))
        m.set_block(0, 1, coo_matrix((3, 4)))
        m.set_block(0, 2, coo_matrix((3, 3)))
        m.set_block(0, 3, coo_matrix((3, 6)))
        m.set_block(1, 3, coo_matrix((5, 6)))

        bcol = m.get_block_column_index(8)
        self.assertEqual(bcol, 2)
        bcol = m.get_block_column_index(5)
        self.assertEqual(bcol, 1)
        bcol = m.get_block_column_index(14)
        self.assertEqual(bcol, 3)

    def test_get_block_row_index(self):

        m = BlockMatrix(2,4)
        m.set_block(0, 0, coo_matrix((3, 2)))
        m.set_block(0, 1, coo_matrix((3, 4)))
        m.set_block(0, 2, coo_matrix((3, 3)))
        m.set_block(0, 3, coo_matrix((3, 6)))
        m.set_block(1, 3, coo_matrix((5, 6)))

        brow = m.get_block_row_index(0)
        self.assertEqual(brow, 0)
        brow = m.get_block_row_index(6)
        self.assertEqual(brow, 1)

    def test_matrix_multiply(self):
        """
        Test

        [A  B  C   *  [G  J   = [A*G + B*H + C*I    A*J + B*K + C*L
         D  E  F]      H  K      D*G + E*H + F*I    D*J + E*K + F*L]
                       I  L]
        """
        np.random.seed(0)
        A = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        B = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        C = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        D = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        E = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        F = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        G = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        H = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        I = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        J = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        K = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))
        L = sp.csr_matrix(np.random.normal(0, 10, (2, 2)))

        bm1 = BlockMatrix(2, 3)
        bm2 = BlockMatrix(3, 2)

        bm1.set_block(0, 0, A)
        bm1.set_block(0, 1, B)
        bm1.set_block(0, 2, C)
        bm1.set_block(1, 0, D)
        bm1.set_block(1, 1, E)
        bm1.set_block(1, 2, F)

        bm2.set_block(0, 0, G)
        bm2.set_block(1, 0, H)
        bm2.set_block(2, 0, I)
        bm2.set_block(0, 1, J)
        bm2.set_block(1, 1, K)
        bm2.set_block(2, 1, L)

        got = (bm1 * bm2).toarray()
        exp00 = (A * G + B * H + C * I).toarray()
        exp01 = (A * J + B * K + C * L).toarray()
        exp10 = (D * G + E * H + F * I).toarray()
        exp11 = (D * J + E * K + F * L).toarray()
        exp = np.zeros((4, 4))
        exp[0:2, 0:2] = exp00
        exp[0:2, 2:4] = exp01
        exp[2:4, 0:2] = exp10
        exp[2:4, 2:4] = exp11

        self.assertTrue(np.allclose(got, exp))

    def test_dimensions(self):
        bm = BlockMatrix(2, 2)
        self.assertTrue(bm.has_undefined_row_sizes())
        self.assertTrue(bm.has_undefined_col_sizes())
        with self.assertRaises(NotFullyDefinedBlockMatrixError):
            shape = bm.shape
        with self.assertRaises(NotFullyDefinedBlockMatrixError):
            bm.set_block(0, 0, BlockMatrix(2, 2))
        with self.assertRaises(NotFullyDefinedBlockMatrixError):
            row_sizes = bm.row_block_sizes()
        with self.assertRaises(NotFullyDefinedBlockMatrixError):
            col_sizes = bm.col_block_sizes()
        bm2 = BlockMatrix(2, 2)
        bm2.set_block(0, 0, coo_matrix((2, 2)))
        bm2.set_block(1, 1, coo_matrix((2, 2)))
        bm3 = bm2.copy()
        bm.set_block(0, 0, bm2)
        bm.set_block(1, 1, bm3)
        self.assertFalse(bm.has_undefined_row_sizes())
        self.assertFalse(bm.has_undefined_col_sizes())
        self.assertEqual(bm.shape, (8, 8))
        bm.set_block(0, 0, None)
        self.assertFalse(bm.has_undefined_row_sizes())
        self.assertFalse(bm.has_undefined_col_sizes())
        self.assertEqual(bm.shape, (8, 8))
        self.assertTrue(np.all(bm.row_block_sizes() == np.ones(2)*4))
        self.assertTrue(np.all(bm.col_block_sizes() == np.ones(2)*4))
        self.assertTrue(np.all(bm.row_block_sizes(copy=False) == np.ones(2)*4))
        self.assertTrue(np.all(bm.col_block_sizes(copy=False) == np.ones(2)*4))

    def test_transpose_with_empty_rows(self):
        m = BlockMatrix(2, 2)
        m.set_row_size(0, 2)
        m.set_row_size(1, 2)
        m.set_col_size(0, 2)
        m.set_col_size(1, 2)
        mt = m.transpose()
        self.assertEqual(mt.get_row_size(0), 2)
        self.assertEqual(mt.get_row_size(1), 2)
        self.assertEqual(mt.get_col_size(0), 2)
        self.assertEqual(mt.get_col_size(1), 2)

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import warnings
import pyutilib.th as unittest

from pyomo.contrib.pynumero.dependencies import (
    numpy_available, scipy_available, numpy as np
)

SKIPTESTS=[]
if numpy_available and scipy_available:
    from scipy.sparse import coo_matrix, bmat
else:
    SKIPTESTS.append(
        "Pynumero needs scipy and numpy>=1.13.0 to run BlockMatrix tests"
    )

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 3:
        SKIPTESTS.append(
            "Pynumero needs at least 3 processes to run BlockMatrix MPI tests"
        )
except ImportError:
    SKIPTESTS.append("Pynumero needs mpi4py to run BlockMatrix MPI tests")

if not SKIPTESTS:
    from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
    from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
    from pyomo.contrib.pynumero.sparse.mpi_block_matrix import (
        MPIBlockMatrix, NotFullyDefinedBlockMatrixError
    )


@unittest.category("mpi")
class TestMPIBlockMatrix(unittest.TestCase):

    # Because the setUpClass is called before decorators around the
    # class itself, we need to put the skipIf on the class setup and not
    # the class.

    @classmethod
    @unittest.skipIf(SKIPTESTS, SKIPTESTS)
    def setUpClass(cls):
        # test problem 1

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))

        rank = comm.Get_rank()
        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
        if rank == 1:
            bm.set_block(1, 1, m)

        # create serial matrix image
        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m)
        serial_bm.set_block(1, 1, m)
        cls.square_serial_mat = serial_bm

        cls.square_mpi_mat = bm

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
        if rank == 1:
            bm.set_block(1, 1, m)

        cls.square_mpi_mat_no_broadcast = bm

        # create matrix with shared blocks
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
        if rank == 1:
            bm.set_block(1, 1, m)
        bm.set_block(0, 1, m)

        cls.square_mpi_mat2 = bm

        # create serial matrix image
        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m)
        serial_bm.set_block(1, 1, m)
        serial_bm.set_block(0, 1, m)
        cls.square_serial_mat2 = serial_bm

        row = np.array([0, 1, 2, 3])
        col = np.array([0, 1, 0, 1])
        data = np.array([1., 1., 1., 1.])
        m2 = coo_matrix((data, (row, col)), shape=(4, 2))

        rank_ownership = [[0, -1, 0], [-1, 1, -1]]
        bm = MPIBlockMatrix(2, 3, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
            bm.set_block(0, 2, m2)
        if rank == 1:
            bm.set_block(1, 1, m)
        cls.rectangular_mpi_mat = bm

        bm = BlockMatrix(2, 3)
        bm.set_block(0, 0, m)
        bm.set_block(0, 2, m2)
        bm.set_block(1, 1, m)
        cls.rectangular_serial_mat = bm

    def test_bshape(self):
        self.assertEqual(self.square_mpi_mat.bshape, (2, 2))
        self.assertEqual(self.rectangular_mpi_mat.bshape, (2, 3))

    def test_shape(self):
        self.assertEqual(self.square_mpi_mat.shape, (8, 8))
        self.assertEqual(self.rectangular_mpi_mat.shape, (8, 10))

    def test_tocoo(self):
        with self.assertRaises(Exception) as context:
            self.square_mpi_mat.tocoo()

    def test_tocsr(self):
        with self.assertRaises(Exception) as context:
            self.square_mpi_mat.tocsr()

    def test_tocsc(self):
        with self.assertRaises(Exception) as context:
            self.square_mpi_mat.tocsc()

    def test_todia(self):
        with self.assertRaises(Exception) as context:
            self.square_mpi_mat.todia()

    def test_tobsr(self):
        with self.assertRaises(Exception) as context:
            self.square_mpi_mat.tobsr()

    def test_toarray(self):
        with self.assertRaises(Exception) as context:
            self.square_mpi_mat.toarray()

    def test_coo_data(self):
        with self.assertRaises(Exception) as context:
            self.square_mpi_mat.coo_data()

    def test_getitem(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if rank == 0:
                self.assertTrue((m == self.square_mpi_mat.get_block(0, 0)).toarray().all())
            if rank == 1:
                self.assertTrue((m == self.square_mpi_mat.get_block(1, 1)).toarray().all())

            self.assertTrue((m == self.square_mpi_mat2.get_block(0, 1)).toarray().all())

    def test_setitem(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)

        bm.set_block(0, 1, m)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue((m == bm.get_block(0, 1)).toarray().all())

    def test_nnz(self):
        self.assertEqual(self.square_mpi_mat.nnz, 12)
        self.assertEqual(self.square_mpi_mat2.nnz, 18)
        self.assertEqual(self.rectangular_mpi_mat.nnz, 16)

    def test_block_shapes(self):

        m, n = self.square_mpi_mat.bshape
        mpi_shapes = self.square_mpi_mat.block_shapes()
        serial_shapes = self.square_serial_mat.block_shapes()
        for i in range(m):
            for j in range(n):
                self.assertEqual(serial_shapes[i][j], mpi_shapes[i][j])

    def test_reset_brow(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
        if rank == 1:
            bm.set_block(1, 1, m)

        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m)
        serial_bm.set_block(1, 1, m)

        self.assertTrue(np.allclose(serial_bm.row_block_sizes(),
                                    bm.row_block_sizes()))
        bm.reset_brow(0)
        serial_bm.reset_brow(0)
        self.assertTrue(np.allclose(serial_bm.row_block_sizes(),
                                    bm.row_block_sizes()))

        bm.reset_brow(1)
        serial_bm.reset_brow(1)
        self.assertTrue(np.allclose(serial_bm.row_block_sizes(),
                                    bm.row_block_sizes()))

    def test_reset_bcol(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
        if rank == 1:
            bm.set_block(1, 1, m)

        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m)
        serial_bm.set_block(1, 1, m)

        self.assertTrue(np.allclose(serial_bm.row_block_sizes(),
                                    bm.row_block_sizes()))
        bm.reset_bcol(0)
        serial_bm.reset_bcol(0)
        self.assertTrue(np.allclose(serial_bm.col_block_sizes(),
                                    bm.col_block_sizes()))

        bm.reset_bcol(1)
        serial_bm.reset_bcol(1)
        self.assertTrue(np.allclose(serial_bm.col_block_sizes(),
                                    bm.col_block_sizes()))

    def test_has_empty_rows(self):
        with self.assertRaises(Exception) as context:
            self.square_mpi_mat.has_empty_rows()

    def test_has_empty_cols(self):
        with self.assertRaises(Exception) as context:
            self.square_mpi_mat.has_empty_cols()

    def test_transpose(self):

        mat1 = self.square_mpi_mat
        mat2 = self.rectangular_mpi_mat

        res = mat1.transpose()
        self.assertIsInstance(res, MPIBlockMatrix)
        self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership.T))
        self.assertEqual(mat1.bshape[1], res.bshape[0])
        self.assertEqual(mat1.bshape[0], res.bshape[1])
        rows, columns = np.nonzero(res.ownership_mask)
        for i, j in zip(rows, columns):
            if res.get_block(i, j) is not None:
                self.assertTrue(np.allclose(res.get_block(i, j).toarray().T,
                                            mat1.get_block(j, i).toarray()))

        res = mat2.transpose()
        self.assertIsInstance(res, MPIBlockMatrix)
        self.assertTrue(np.allclose(mat2.rank_ownership, res.rank_ownership.T))
        self.assertEqual(mat2.bshape[1], res.bshape[0])
        self.assertEqual(mat2.bshape[0], res.bshape[1])
        rows, columns = np.nonzero(res.ownership_mask)
        for i, j in zip(rows, columns):
            if res.get_block(i, j) is not None:
                self.assertTrue(np.allclose(res.get_block(i, j).toarray().T,
                                            mat2.get_block(j, i).toarray()))

        res = mat1.transpose(copy=True)
        self.assertIsInstance(res, MPIBlockMatrix)
        self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership.T))
        self.assertEqual(mat1.bshape[1], res.bshape[0])
        self.assertEqual(mat1.bshape[0], res.bshape[1])
        rows, columns = np.nonzero(res.ownership_mask)
        for i, j in zip(rows, columns):
            if res.get_block(i, j) is not None:
                self.assertTrue(np.allclose(res.get_block(i, j).toarray().T,
                                            mat1.get_block(j, i).toarray()))

        res = mat2.transpose(copy=True)
        self.assertIsInstance(res, MPIBlockMatrix)
        self.assertTrue(np.allclose(mat2.rank_ownership, res.rank_ownership.T))
        self.assertEqual(mat2.bshape[1], res.bshape[0])
        self.assertEqual(mat2.bshape[0], res.bshape[1])
        rows, columns = np.nonzero(res.ownership_mask)
        for i, j in zip(rows, columns):
            if res.get_block(i, j) is not None:
                self.assertTrue(np.allclose(res.get_block(i, j).toarray().T,
                                            mat2.get_block(j, i).toarray()))

        res = mat1.T
        self.assertIsInstance(res, MPIBlockMatrix)
        self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership.T))
        self.assertEqual(mat1.bshape[1], res.bshape[0])
        self.assertEqual(mat1.bshape[0], res.bshape[1])
        rows, columns = np.nonzero(res.ownership_mask)
        for i, j in zip(rows, columns):
            if res.get_block(i, j) is not None:
                self.assertTrue(np.allclose(res.get_block(i, j).toarray().T,
                                            mat1.get_block(j, i).toarray()))

        res = mat2.T
        self.assertIsInstance(res, MPIBlockMatrix)
        self.assertTrue(np.allclose(mat2.rank_ownership, res.rank_ownership.T))
        self.assertEqual(mat2.bshape[1], res.bshape[0])
        self.assertEqual(mat2.bshape[0], res.bshape[1])
        rows, columns = np.nonzero(res.ownership_mask)
        for i, j in zip(rows, columns):
            if res.get_block(i, j) is not None:
                self.assertTrue(np.allclose(res.get_block(i, j).toarray().T,
                                            mat2.get_block(j, i).toarray()))

    def _compare_mpi_and_serial_block_matrices(self, mpi_mat, serial_mat):
        self.assertIsInstance(mpi_mat, MPIBlockMatrix)
        rows, columns = np.nonzero(mpi_mat.ownership_mask)
        for i, j in zip(rows, columns):
            if mpi_mat.get_block(i, j) is not None:
                self.assertTrue(np.allclose(mpi_mat.get_block(i, j).toarray(),
                                            serial_mat.get_block(i, j).toarray()))
            else:
                self.assertIsNone(serial_mat.get_block(i, j))

    def test_add(self):
        mat1 = self.square_mpi_mat
        mat2 = self.square_mpi_mat2

        serial_mat1 = self.square_serial_mat
        serial_mat2 = self.square_serial_mat2

        res = mat1 + mat2
        serial_res = serial_mat1 + serial_mat2
        self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
        self._compare_mpi_and_serial_block_matrices(res, serial_res)

        res = mat1 + serial_mat2
        self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
        self._compare_mpi_and_serial_block_matrices(res, serial_res)

        res = serial_mat2 + mat1
        self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
        self._compare_mpi_and_serial_block_matrices(res, serial_res)

        with self.assertRaises(Exception) as context:
            res = mat1 + serial_mat2.tocoo()

        with self.assertRaises(Exception) as context:
            res = serial_mat2.tocoo() + mat1

    def test_sub(self):

        mat1 = self.square_mpi_mat
        mat2 = self.square_mpi_mat2

        serial_mat1 = self.square_serial_mat
        serial_mat2 = self.square_serial_mat2

        res = mat1 - mat2
        serial_res = serial_mat1 - serial_mat2
        self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
        self._compare_mpi_and_serial_block_matrices(res, serial_res)

        res = mat1 - serial_mat2
        self._compare_mpi_and_serial_block_matrices(res, serial_res)

        res = mat2 - mat1
        serial_res = serial_mat2 - serial_mat1
        self._compare_mpi_and_serial_block_matrices(res, serial_res)

        res = serial_mat2 - mat1
        self._compare_mpi_and_serial_block_matrices(res, serial_res)

        with self.assertRaises(Exception) as context:
            res = mat1 - serial_mat2.tocoo()
        with self.assertRaises(Exception) as context:
            res = serial_mat2.tocoo() - mat1

    def test_div(self):

        mat1 = self.square_mpi_mat
        serial_mat1 = self.square_serial_mat

        res =  mat1 / 3.0
        serial_res = serial_mat1 / 3.0

        self.assertIsInstance(res, MPIBlockMatrix)
        rows, columns = np.nonzero(res.ownership_mask)
        for i, j in zip(rows, columns):
            if res.get_block(i, j) is not None:
                self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                            serial_res.get_block(i, j).toarray()))
            else:
                self.assertIsNone(serial_res.get_block(i, j))

    def test_iadd(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m.copy())
        if rank == 1:
            bm.set_block(1, 1, m.copy())

        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m.copy())
        serial_bm.set_block(1, 1, m.copy())

        bm += bm
        serial_bm += serial_bm

        rows, columns = np.nonzero(bm.ownership_mask)
        for i, j in zip(rows, columns):
            if bm.get_block(i, j) is not None:
                self.assertTrue(np.allclose(bm.get_block(i, j).toarray(),
                                            serial_bm.get_block(i, j).toarray()))

        bm += serial_bm
        serial_bm += serial_bm
        self._compare_mpi_and_serial_block_matrices(bm, serial_bm)

    def test_isub(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m.copy())
        if rank == 1:
            bm.set_block(1, 1, m.copy())

        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m.copy())
        serial_bm.set_block(1, 1, m.copy())

        bm -= bm
        serial_bm -= serial_bm

        rows, columns = np.nonzero(bm.ownership_mask)
        for i, j in zip(rows, columns):
            if bm.get_block(i, j) is not None:
                self.assertTrue(np.allclose(bm.get_block(i, j).toarray(),
                                            serial_bm.get_block(i, j).toarray()))

        bm -= serial_bm
        serial_bm -= serial_bm
        self._compare_mpi_and_serial_block_matrices(bm, serial_bm)

    def test_imul(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
        if rank == 1:
            bm.set_block(1, 1, m)

        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m)
        serial_bm.set_block(1, 1, m)

        bm *= 2.0
        serial_bm *= 2.0

        rows, columns = np.nonzero(bm.ownership_mask)
        for i, j in zip(rows, columns):
            if bm.get_block(i, j) is not None:
                self.assertTrue(np.allclose(bm.get_block(i, j).toarray(),
                                            serial_bm.get_block(i, j).toarray()))

    def test_idiv(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
        if rank == 1:
            bm.set_block(1, 1, m)

        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m)
        serial_bm.set_block(1, 1, m)

        bm /= 2.0
        serial_bm /= 2.0

        rows, columns = np.nonzero(bm.ownership_mask)
        for i, j in zip(rows, columns):
            if bm.get_block(i, j) is not None:
                self.assertTrue(np.allclose(bm.get_block(i, j).toarray(),
                                            serial_bm.get_block(i, j).toarray()))

    def test_neg(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
        if rank == 1:
            bm.set_block(1, 1, m)

        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m)
        serial_bm.set_block(1, 1, m)

        res = -bm
        serial_res = -serial_bm

        rows, columns = np.nonzero(bm.ownership_mask)
        for i, j in zip(rows, columns):
            if res.get_block(i, j) is not None:
                self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                            serial_res.get_block(i, j).toarray()))

    def test_abs(self):

        row = np.array([0, 3, 1, 2, 3, 0])
        col = np.array([0, 0, 1, 2, 3, 3])
        data = np.array([2., 1, 3, 4, 5, 1])
        m = coo_matrix((data, (row, col)), shape=(4, 4))
        rank = comm.Get_rank()

        # create mpi matrix
        rank_ownership = [[0, -1], [-1, 1]]
        bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
        if rank == 0:
            bm.set_block(0, 0, m)
        if rank == 1:
            bm.set_block(1, 1, m)

        serial_bm = BlockMatrix(2, 2)
        serial_bm.set_block(0, 0, m)
        serial_bm.set_block(1, 1, m)

        res = abs(bm)
        serial_res = abs(serial_bm)

        rows, columns = np.nonzero(bm.ownership_mask)
        for i, j in zip(rows, columns):
            if res.get_block(i, j) is not None:
                self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                            serial_res.get_block(i, j).toarray()))

    def test_eq(self):

        mat1 = self.square_mpi_mat
        mat2 = self.square_mpi_mat2

        serial_mat1 = self.square_serial_mat
        serial_mat2 = self.square_serial_mat2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 == mat2
            serial_res = serial_mat1 == serial_mat2

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 == serial_mat2

        mat1 = self.rectangular_mpi_mat
        serial_mat1 = self.rectangular_serial_mat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 == mat1
            serial_res = serial_mat1 == serial_mat1

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 == serial_mat1

    def test_ne(self):

        mat1 = self.square_mpi_mat
        mat2 = self.square_mpi_mat2

        serial_mat1 = self.square_serial_mat
        serial_mat2 = self.square_serial_mat2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 != mat2
            serial_res = serial_mat1 != serial_mat2

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 != serial_mat2

        mat1 = self.rectangular_mpi_mat
        serial_mat1 = self.rectangular_serial_mat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 != mat1
            serial_res = serial_mat1 != serial_mat1

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 != serial_mat1

        with self.assertRaises(Exception) as context:
            res = serial_mat1 != mat1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 != 2
            serial_res = serial_mat1 != 2

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

    def test_le(self):

        mat1 = self.square_mpi_mat
        mat2 = self.square_mpi_mat2

        serial_mat1 = self.square_serial_mat
        serial_mat2 = self.square_serial_mat2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 <= mat2
            serial_res = serial_mat1 <= serial_mat2

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 <= serial_mat2
            serial_res = serial_mat1 <= serial_mat2

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        mat1 = self.rectangular_mpi_mat
        serial_mat1 = self.rectangular_serial_mat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 <= mat1
            serial_res = serial_mat1 <= serial_mat1

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 <= serial_mat1

        with self.assertRaises(Exception) as context:
            res = serial_mat1 <= mat1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 <= 2
            serial_res = serial_mat1 <= 2

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

    def test_lt(self):

        mat1 = self.square_mpi_mat
        mat2 = self.square_mpi_mat2

        serial_mat1 = self.square_serial_mat
        serial_mat2 = self.square_serial_mat2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 < mat2
            serial_res = serial_mat1 < serial_mat2

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 < serial_mat2

        mat1 = self.rectangular_mpi_mat
        serial_mat1 = self.rectangular_serial_mat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 < mat1
            serial_res = serial_mat1 < serial_mat1

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 < serial_mat1

        with self.assertRaises(Exception) as context:
            res = serial_mat1 < mat1

    def test_ge(self):

        mat1 = self.square_mpi_mat
        mat2 = self.square_mpi_mat2

        serial_mat1 = self.square_serial_mat
        serial_mat2 = self.square_serial_mat2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 >= mat2
            serial_res = serial_mat1 >= serial_mat2

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 >= serial_mat2

        mat1 = self.rectangular_mpi_mat
        serial_mat1 = self.rectangular_serial_mat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 >= mat1
            serial_res = serial_mat1 >= serial_mat1

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 >= serial_mat1

        with self.assertRaises(Exception) as context:
            res = serial_mat1 >= mat1

    def test_gt(self):

        mat1 = self.square_mpi_mat
        mat2 = self.square_mpi_mat2

        serial_mat1 = self.square_serial_mat
        serial_mat2 = self.square_serial_mat2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 > mat2
            serial_res = serial_mat1 > serial_mat2

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 > serial_mat2

        mat1 = self.rectangular_mpi_mat
        serial_mat1 = self.rectangular_serial_mat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mat1 > mat1
            serial_res = serial_mat1 > serial_mat1

            self.assertIsInstance(res, MPIBlockMatrix)
            self.assertTrue(np.allclose(mat1.rank_ownership, res.rank_ownership))
            rows, columns = np.nonzero(res.ownership_mask)
            for i, j in zip(rows, columns):
                if res.get_block(i, j) is not None:
                    self.assertTrue(np.allclose(res.get_block(i, j).toarray(),
                                                serial_res.get_block(i, j).toarray()))
                else:
                    self.assertIsNone(serial_res.get_block(i, j))

        with self.assertRaises(Exception) as context:
            res = mat1 > serial_mat1

        with self.assertRaises(Exception) as context:
            res = serial_mat1 > mat1


@unittest.category("mpi")
class TestMPIMatVec(unittest.TestCase):

    @classmethod
    @unittest.skipIf(SKIPTESTS, SKIPTESTS)
    def setUpClass(cls):
        pass

    def test_get_block_vector_for_dot_product_1(self):
        rank = comm.Get_rank()

        rank_ownership = np.array([[0, 1, 2],
                                   [1, 1, 2],
                                   [0, 1, 2],
                                   [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0],
                          [0, 1]])
        sub_m = coo_matrix(sub_m)
        m.set_block(rank, rank, sub_m.copy())
        m.set_block(3, rank, sub_m.copy())

        rank_ownership = np.array([0, 1, 2])
        v = MPIBlockVector(3, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(rank, sub_v)

        res = m._get_block_vector_for_dot_product(v)

        self.assertIs(res, v)

    def test_get_block_vector_for_dot_product_2(self):
        rank = comm.Get_rank()

        rank_ownership = np.array([[1, 1, 2],
                                   [0, 1, 2],
                                   [0, 1, 2],
                                   [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0],
                          [0, 1]])
        sub_m = coo_matrix(sub_m)
        if rank == 0:
            m.set_block(3, rank, sub_m.copy())
        elif rank == 1:
            m.set_block(0, 0, sub_m.copy())
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        else:
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())

        rank_ownership = np.array([-1, 1, 2])
        v = MPIBlockVector(3, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(0, sub_v.copy())
        if rank != 0:
            v.set_block(rank, sub_v.copy())

        res = m._get_block_vector_for_dot_product(v)

        self.assertIs(res, v)

    def test_get_block_vector_for_dot_product_3(self):
        rank = comm.Get_rank()

        rank_ownership = np.array([[1, 1, 2],
                                   [0, 1, 2],
                                   [0, 1, 2],
                                   [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0],
                          [0, 1]])
        sub_m = coo_matrix(sub_m)
        if rank == 0:
            m.set_block(3, rank, sub_m.copy())
        elif rank == 1:
            m.set_block(0, 0, sub_m.copy())
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        else:
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())

        rank_ownership = np.array([0, 1, 2])
        v = MPIBlockVector(3, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(rank, sub_v.copy())

        res = m._get_block_vector_for_dot_product(v)

        self.assertIsNot(res, v)
        self.assertTrue(np.array_equal(res.get_block(0), sub_v))
        if rank == 0:
            self.assertIsNone(res.get_block(1))
            self.assertIsNone(res.get_block(2))
        elif rank == 1:
            self.assertTrue(np.array_equal(res.get_block(1), sub_v))
            self.assertIsNone(res.get_block(2))
        elif rank == 2:
            self.assertTrue(np.array_equal(res.get_block(2), sub_v))
            self.assertIsNone(res.get_block(1))

    def test_get_block_vector_for_dot_product_4(self):
        rank = comm.Get_rank()

        rank_ownership = np.array([[-1, 1, 2],
                                   [0, 1, 2],
                                   [0, 1, 2],
                                   [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0],
                          [0, 1]])
        sub_m = coo_matrix(sub_m)
        m.set_block(0, 0, sub_m.copy())
        if rank == 0:
            m.set_block(3, rank, sub_m.copy())
        else:
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())

        rank_ownership = np.array([0, 1, 2])
        v = MPIBlockVector(3, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(rank, sub_v.copy())

        res = m._get_block_vector_for_dot_product(v)

        self.assertIs(res, v)

    def test_get_block_vector_for_dot_product_5(self):
        rank = comm.Get_rank()

        rank_ownership = np.array([[1, 1, 2],
                                   [0, 1, 2],
                                   [0, 1, 2],
                                   [0, 1, 2]])
        m = MPIBlockMatrix(4, 3, rank_ownership, comm)
        sub_m = np.array([[1, 0],
                          [0, 1]])
        sub_m = coo_matrix(sub_m)
        if rank == 0:
            m.set_block(3, rank, sub_m.copy())
        elif rank == 1:
            m.set_block(0, 0, sub_m.copy())
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())
        else:
            m.set_block(rank, rank, sub_m.copy())
            m.set_block(3, rank, sub_m.copy())

        v = BlockVector(3)
        sub_v = np.ones(2)
        for ndx in range(3):
            v.set_block(ndx, sub_v.copy())

        res = m._get_block_vector_for_dot_product(v)

        self.assertIs(res, v)

        v_flat = v.flatten()
        res = m._get_block_vector_for_dot_product(v_flat)
        self.assertIsInstance(res, BlockVector)
        for ndx in range(3):
            block = res.get_block(ndx)
            self.assertTrue(np.array_equal(block, sub_v))

    def test_matvec_1(self):
        rank = comm.Get_rank()

        rank_ownership = np.array([[0, -1, -1, 0],
                                   [-1, 1, -1, 1],
                                   [-1, -1, 2, 2],
                                   [0, 1, 2, -1]])
        m = MPIBlockMatrix(4, 4, rank_ownership, comm)
        sub_m = np.array([[1, 0],
                          [0, 1]])
        sub_m = coo_matrix(sub_m)
        m.set_block(rank, rank, sub_m.copy())
        m.set_block(rank, 3, sub_m.copy())
        m.set_block(3, rank, sub_m.copy())
        m.set_block(3, 3, sub_m.copy())

        rank_ownership = np.array([0, 1, 2, -1])
        v = MPIBlockVector(4, rank_ownership, comm)
        sub_v = np.ones(2)
        v.set_block(rank, sub_v.copy())
        v.set_block(3, sub_v.copy())

        res = m.dot(v)
        self.assertIsInstance(res, MPIBlockVector)
        self.assertTrue(np.array_equal(res.get_block(rank), sub_v*2))
        self.assertTrue(np.array_equal(res.get_block(3), sub_v*4))
        self.assertTrue(np.array_equal(res.rank_ownership, np.array([0, 1, 2, -1])))
        self.assertFalse(res.has_none)

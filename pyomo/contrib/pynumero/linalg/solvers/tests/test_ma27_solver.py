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
try:
    import numpy as np
    from scipy.sparse import coo_matrix, tril
except ImportError:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

try:
    from pyomo.contrib.pynumero.linalg.solvers.ma27_solver import MA27LinearSolver
except ImportError:
    raise unittest.SkipTest("Pynumero needs libpynumero_MA27 to run MA27 solvers")

from pyomo.contrib.pynumero.linalg.solvers.ma27_solver import MA27LinearSolver
from pyomo.contrib.pynumero.sparse import (BlockSymMatrix,
                                           BlockMatrix,
                                           BlockVector,
                                           empty_matrix)


class TestMA27(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        row = np.array([0, 1, 2, 1, 2, 2])
        col = np.array([0, 0, 0, 1, 1, 2])
        data = np.array([1.0, 7.0, 3.0, 4.0, -5.0, 6.0])
        off_diagonal_mask = row != col
        new_row = np.concatenate([row, col[off_diagonal_mask]])
        new_col = np.concatenate([col, row[off_diagonal_mask]])
        new_data = np.concatenate([data, data[off_diagonal_mask]])
        cls.basic_m = coo_matrix((new_data, (new_row, new_col)), shape=(3, 3))
        cls.basic_rhs = np.array([1, 1, 0])

        # block algebra
        row = np.array([0, 1])
        col = np.array([0, 1])
        data = np.array([2, 8])
        H = coo_matrix((data, (row, col)), shape=(2, 2))

        row = np.array([0, 1, 0])
        col = np.array([0, 0, 1])
        data = np.array([1, 1, 1])
        A = coo_matrix((data, (row, col)), shape=(2, 2))

        cls.block_m = BlockSymMatrix(2)
        cls.block_m[0, 0] = H
        cls.block_m[1, 0] = A
        cls.block_rhs = BlockVector([np.array([8.0, 16.0]), np.array([-5.0, -3.0])])

        G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = np.array([[1, 0, 1], [0, 1, 1]])
        b = np.array([3, 0])
        c = np.array([-8, -3, -3])
        G_sparse = coo_matrix(G)
        A_sparse = coo_matrix(A)
        row = np.array([0])
        col = np.array([0])
        data = np.array([1.0])
        coupling_x = coo_matrix((data, (row, col)), shape=(1, 3))
        coupling_z = coo_matrix((-data, (row, col)), shape=(1, 1))
        H = BlockSymMatrix(3)
        H[0, 0] = G_sparse
        H[1, 1] = G_sparse
        H[2, 2] = empty_matrix(1, 1)

        A = BlockMatrix(4, 3)
        A[0, 0] = A_sparse
        A[2, 0] = coupling_x
        A[2, 2] = coupling_z
        A[1, 1] = A_sparse
        A[3, 1] = coupling_x
        A[3, 2] = coupling_z

        cls.block_m2 = BlockSymMatrix(2)
        cls.block_m2[0, 0] = H
        cls.block_m2[1, 0] = A
        rhsx = BlockVector([np.array([-8.0, -3.0, -3.0]), np.array([-8.0, -3.0, -3.0]), np.zeros(1)])
        rhsy = BlockVector([np.array([3.0, 0.0]), np.array([3.1, 0.1]), np.zeros(1), np.zeros(1)])
        cls.block_rhs2 = BlockVector([-rhsx, -rhsy])

    def test_do_symbolic_factorization(self):

        A = self.basic_m
        linear_solver = MA27LinearSolver()
        linear_solver.do_symbolic_factorization(A)
        ltr = tril(A)
        self.assertEqual(linear_solver._nnz, ltr.nnz)
        self.assertEqual(linear_solver._dim, A.shape[0])

        # checks for block matrices
        A = self.block_m
        linear_solver = MA27LinearSolver()
        linear_solver.do_symbolic_factorization(A)
        self.assertEqual(A.bshape[0], linear_solver._row_blocks)
        self.assertEqual(A.bshape[1], linear_solver._col_blocks)
        ltr = tril(A.tocoo())
        self.assertEqual(linear_solver._nnz, ltr.nnz)
        self.assertEqual(linear_solver._dim, A.shape[0])

    def test_do_numeric_factorization(self):

        linear_solver = MA27LinearSolver()
        A = self.basic_m
        linear_solver.do_symbolic_factorization(A)
        status = linear_solver.do_numeric_factorization(A)
        self.assertEqual(status, 0)

        # check block matrices
        linear_solver = MA27LinearSolver()
        A = self.block_m
        linear_solver.do_symbolic_factorization(A)
        status = linear_solver.do_numeric_factorization(A)
        self.assertEqual(status, 0)

    def test_do_back_solve(self):

        linear_solver = MA27LinearSolver()
        A = self.basic_m
        linear_solver.do_symbolic_factorization(A)
        linear_solver.do_numeric_factorization(A)
        x = linear_solver.do_back_solve(self.basic_rhs)
        x_np = np.linalg.solve(A.toarray(), self.basic_rhs)
        for i, v in enumerate(x_np):
            self.assertAlmostEqual(v, x[i])

        # check for block matrices
        linear_solver = MA27LinearSolver()
        A = self.block_m
        linear_solver.do_symbolic_factorization(A)
        linear_solver.do_numeric_factorization(A)
        x = linear_solver.do_back_solve(self.block_rhs)
        self.assertIsInstance(x, BlockVector)
        flat_x = x.flatten()
        x_np = np.linalg.solve(A.toarray(), self.block_rhs.flatten())
        for i, v in enumerate(x_np):
            self.assertAlmostEqual(v, flat_x[i])

        linear_solver2 = MA27LinearSolver()
        A = self.block_m.tocoo()
        linear_solver2.do_symbolic_factorization(A)
        linear_solver2.do_numeric_factorization(A)
        x = linear_solver.do_back_solve(self.block_rhs)
        self.assertListEqual(flat_x.tolist(), x.tolist())

        linear_solver2 = MA27LinearSolver()
        A = self.block_m2.tocoo()
        linear_solver2.do_symbolic_factorization(A)
        linear_solver2.do_numeric_factorization(A)
        x = linear_solver2.do_back_solve(self.block_rhs2.flatten())
        npx = np.linalg.solve(A.toarray(), self.block_rhs2.flatten())
        self.assertTrue(np.allclose(x, npx))

        linear_solver2 = MA27LinearSolver()
        A = self.block_m2
        linear_solver2.do_symbolic_factorization(A)
        linear_solver2.do_numeric_factorization(A)
        x = linear_solver2.do_back_solve(self.block_rhs2)
        npx = np.linalg.solve(A.toarray(), self.block_rhs2.flatten())
        self.assertTrue(np.allclose(x.flatten(), npx))

import pyutilib.th as unittest
try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.linalg.solvers import MA27LinearSolver
from pyomo.contrib.pynumero.sparse import (COOMatrix,
                                           COOSymMatrix,
                                           BlockSymMatrix,
                                           BlockMatrix,
                                           BlockVector)

@unittest.skip
class TestMA27(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        row = np.array([0, 1, 2, 1, 2, 2])
        col = np.array([0, 0, 0, 1, 1, 2])
        data = np.array([1, 7, 3, 4, -5, 6])
        cls.basic_m = COOSymMatrix((data, (row, col)), shape=(3, 3))
        cls.basic_rhs = np.array([1, 1, 0])

        # block algebra
        row = np.array([0, 1])
        col = np.array([0, 1])
        data = np.array([2, 8])
        H = COOSymMatrix((data, (row, col)), shape=(2, 2))

        row = np.array([0, 1, 0])
        col = np.array([0, 0, 1])
        data = np.array([1, 1, 1])
        A = COOMatrix((data, (row, col)), shape=(2, 2))

        cls.block_m = BlockSymMatrix(2)
        cls.block_m[0, 0] = H
        cls.block_m[1, 0] = A
        cls.block_rhs = BlockVector([np.array([8, 16]), np.array([-5, -3])])

    def test_do_symbolic_factorization(self):

        A = self.basic_m
        linear_solver = MA27LinearSolver()
        linear_solver.do_symbolic_factorization(A)

        self.assertEqual(linear_solver._nnz, A.nnz)
        self.assertEqual(linear_solver._dim, A.shape[0])

        fA = A.tofullmatrix()
        linear_solver = MA27LinearSolver()
        with self.assertRaises(RuntimeError):
            linear_solver.do_symbolic_factorization(fA)

        # checks for block matrices
        A = self.block_m
        linear_solver = MA27LinearSolver()
        linear_solver.do_symbolic_factorization(A)
        self.assertEqual(A.bshape[0], linear_solver._row_blocks)
        self.assertEqual(A.bshape[1], linear_solver._col_blocks)
        self.assertEqual(linear_solver._nnz, A.nnz)
        self.assertEqual(linear_solver._dim, A.shape[0])

        fA = A.tofullmatrix()
        linear_solver = MA27LinearSolver()
        with self.assertRaises(RuntimeError):
            linear_solver.do_symbolic_factorization(fA)

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
        x_np = np.linalg.solve(A.todense(), self.basic_rhs)
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
        x_np = np.linalg.solve(A.todense(), self.block_rhs.flatten())
        for i, v in enumerate(x_np):
            self.assertAlmostEqual(v, flat_x[i])

        linear_solver2 = MA27LinearSolver()
        A = self.block_m.tocoo()
        linear_solver2.do_symbolic_factorization(A)
        linear_solver2.do_numeric_factorization(A)
        x = linear_solver.do_back_solve(self.block_rhs)
        self.assertListEqual(flat_x.tolist(), x.tolist())





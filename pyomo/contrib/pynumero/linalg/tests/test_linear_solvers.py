from pyomo.common import unittest
from pyomo.contrib.pynumero.dependencies import numpy_available, scipy_available

if not numpy_available or not scipy_available:
    raise unittest.SkipTest("pynumero linear solver tests require numpy and scipy")
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from scipy.sparse import coo_matrix, spmatrix
import numpy as np
from pyomo.contrib.pynumero.linalg.base import LinearSolverInterface, LinearSolverStatus
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from pyomo.contrib.pynumero.linalg.ma57_interface import MA57
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU, ScipyIterative
from scipy.sparse.linalg import gmres
from pyomo.contrib.pynumero.linalg.mumps_interface import (
    mumps_available,
    MumpsCentralizedAssembledLinearSolver,
)


class TestLinearSolvers(unittest.TestCase):
    def create_blocks(self, m: np.ndarray, x: np.ndarray):
        m = coo_matrix(m)
        r = m * x
        bm = BlockMatrix(2, 2)
        bm.set_block(0, 0, m.copy())
        bm.set_block(1, 1, m.copy())
        br = BlockVector(2)
        br.set_block(0, r.copy())
        br.set_block(1, r.copy())
        bx = BlockVector(2)
        bx.set_block(0, x.copy())
        bx.set_block(1, x.copy())
        return bm, bx, br

    def solve_helper(self, m: np.ndarray, x: np.ndarray, solver: LinearSolverInterface):
        bm, bx, br = self.create_blocks(m, x)
        bx2, res = solver.solve(bm, br)
        self.assertEqual(res.status, LinearSolverStatus.successful)
        err = np.max(np.abs(bx - bx2))
        self.assertAlmostEqual(err, 0)

    def symmetric_helper(self, solver: LinearSolverInterface):
        m = np.array([[1, 2], [2, -1]], dtype=np.double)
        x = np.array([4, 7], dtype=np.double)
        self.solve_helper(m, x, solver)

    def unsymmetric_helper(self, solver: LinearSolverInterface):
        m = np.array([[1, 2], [0, -1]], dtype=np.double)
        x = np.array([4, 7], dtype=np.double)
        self.solve_helper(m, x, solver)

    def singular_helper(self, solver: LinearSolverInterface):
        m = np.array([[1, 1], [1, 1]], dtype=np.double)
        x = np.array([4, 7], dtype=np.double)
        bm, bx, br = self.create_blocks(m, x)
        br.get_block(0)[1] += 1
        br.get_block(1)[1] += 1
        bx2, res = solver.solve(bm, br, raise_on_error=False)
        self.assertNotEqual(res.status, LinearSolverStatus.successful)

    @unittest.skipIf(not MA27Interface.available(), reason="MA27 not available")
    def test_ma27(self):
        solver = MA27()
        self.symmetric_helper(solver)
        self.singular_helper(solver)

    @unittest.skipIf(not MA57Interface.available(), reason="MA57 not available")
    def test_ma57(self):
        solver = MA57()
        self.symmetric_helper(solver)
        self.singular_helper(solver)

    def test_scipy_direct(self):
        solver = ScipyLU()
        self.symmetric_helper(solver)
        self.unsymmetric_helper(solver)
        self.singular_helper(solver)

    def test_scipy_iterative(self):
        solver = ScipyIterative(gmres)
        solver.options["atol"] = 1e-8
        self.symmetric_helper(solver)
        self.unsymmetric_helper(solver)
        self.singular_helper(solver)

    @unittest.skipIf(not mumps_available, reason="mumps not available")
    def test_mumps(self):
        solver = MumpsCentralizedAssembledLinearSolver(sym=2)
        self.symmetric_helper(solver)
        self.singular_helper(solver)
        solver = MumpsCentralizedAssembledLinearSolver(sym=0)
        self.unsymmetric_helper(solver)

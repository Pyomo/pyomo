#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import pyomo.common.unittest as unittest

try:
    import numpy as np
    from scipy.sparse import coo_matrix, tril
except ImportError:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run linear solver tests")

from pyomo.contrib.pynumero.linalg.mumps_interface import (
    mumps_available,
    MumpsCentralizedAssembledLinearSolver,
)

from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector


@unittest.skipIf(
    not mumps_available, "Pynumero needs pymumps to run linear solver tests"
)
class TestMumpsLinearSolver(unittest.TestCase):
    def test_mumps_linear_solver(self):
        A = np.array([[1, 7, 3], [7, 4, -5], [3, -5, 6]], dtype=np.double)
        A = coo_matrix(A)
        A_lower = tril(A)
        x1 = np.arange(3) + 1
        b1 = A * x1
        x2 = np.array(list(reversed(x1)))
        b2 = A * x2

        solver = MumpsCentralizedAssembledLinearSolver()
        solver.do_symbolic_factorization(A)
        solver.do_numeric_factorization(A)
        x, res = solver.do_back_solve(b1)
        self.assertTrue(np.allclose(x, x1))
        x, res = solver.do_back_solve(b2)
        self.assertTrue(np.allclose(x, x2))

        solver = MumpsCentralizedAssembledLinearSolver(sym=2)
        x, res = solver.solve(A_lower, b1)
        self.assertTrue(np.allclose(x, x1))

        block_A = BlockMatrix(2, 2)
        block_A.set_row_size(0, 2)
        block_A.set_row_size(1, 1)
        block_A.set_col_size(0, 2)
        block_A.set_col_size(1, 1)
        block_A.copyfrom(A)

        block_b1 = BlockVector(2)
        block_b1.set_block(0, b1[0:2])
        block_b1.set_block(1, b1[2:])

        block_b2 = BlockVector(2)
        block_b2.set_block(0, b2[0:2])
        block_b2.set_block(1, b2[2:])

        solver = MumpsCentralizedAssembledLinearSolver(
            icntl_options={10: -3}, cntl_options={2: 1e-16}
        )
        solver.do_symbolic_factorization(block_A)
        solver.do_numeric_factorization(block_A)
        x, res = solver.do_back_solve(block_b1)
        self.assertTrue(np.allclose(x, x1))
        x, res = solver.do_back_solve(block_b2)
        self.assertTrue(np.allclose(x, x2))
        self.assertEqual(solver.get_infog(15), 3)

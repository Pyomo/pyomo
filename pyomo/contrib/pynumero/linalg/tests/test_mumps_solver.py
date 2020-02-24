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
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run linear solver tests")

try:
    from pyomo.contrib.pynumero.linalg.mumps_solver import MumpsCentralizedAssembledLinearSolver
except ImportError:
    raise unittest.SkipTest("Pynumero needs pymumps to run linear solver tests")

from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector


class TestMumpsLinearSolver(unittest.TestCase):
    def test_mumps_linear_solver(self):
        A = np.array([[ 1,  7,  3],
                      [ 7,  4, -5],
                      [ 3, -5,  6]], dtype=np.double)
        A = coo_matrix(A)
        A_lower = tril(A)
        x1 = np.arange(3) + 1
        b1 = A * x1
        x2 = np.array(list(reversed(x1)))
        b2 = A * x2

        solver = MumpsCentralizedAssembledLinearSolver()
        solver.do_symbolic_factorization(A)
        solver.do_numeric_factorization(A)
        x = solver.do_back_solve(b1)
        self.assertTrue(np.allclose(x, x1))
        x = solver.do_back_solve(b2)
        self.assertTrue(np.allclose(x, x2))

        solver = MumpsCentralizedAssembledLinearSolver(sym=2)
        solver.do_symbolic_factorization(A_lower)
        solver.do_numeric_factorization(A_lower)
        x = solver.do_back_solve(b1)
        self.assertTrue(np.allclose(x, x1))
        x = solver.do_back_solve(b2)
        self.assertTrue(np.allclose(x, x2))

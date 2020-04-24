from .base_linear_solver_interface import LinearSolverInterface
from scipy.sparse.linalg import splu
from scipy.linalg import eigvals
from scipy.sparse import isspmatrix_csc
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
import logging
import numpy as np


class ScipyInterface(LinearSolverInterface):
    def __init__(self, compute_inertia=False):
        self._lu = None
        self._inertia = None
        self.compute_inertia = compute_inertia

        self.logger = logging.getLogger('scipy')
        self.logger.propagate = False

    def do_symbolic_factorization(self, matrix):
        pass

    def do_numeric_factorization(self, matrix):
        if not isspmatrix_csc(matrix):
            matrix = matrix.tocsc()
        self._lu = splu(matrix)
        if self.compute_inertia:
            eig = eigvals(matrix.toarray())
            pos_eig = np.count_nonzero((eig > 0))
            neg_eigh = np.count_nonzero((eig < 0))
            zero_eig = np.count_nonzero(eig == 0)
            self._inertia = (pos_eig, neg_eigh, zero_eig)

    def try_factorization(self, matrix):
        error = None
        try:
            self.do_numeric_factorization(matrix)
        except RuntimeError as err:
            error = err
        finally:
            if self.compute_inertia:
                eig = eigvals(matrix.toarray())
                pos_eig = np.count_nonzero((eig > 0))
                neg_eigh = np.count_nonzero((eig < 0))
                zero_eig = np.count_nonzero(eig == 0)
                self._inertia = (pos_eig, neg_eigh, zero_eig)
        return error

    def is_numerically_singular(self, err=None, raise_if_not=True):
        if err:
            if 'Factor is exactly singular' in str(err):
                return True
            else:
                raise
        # Appears to be no way to query splu for info about the solve
        return False

    def do_back_solve(self, rhs):
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
        else:
            _rhs = rhs

        result = self._lu.solve(_rhs)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result
        
        return result

    def get_inertia(self):
        if self._inertia is None:
            raise RuntimeError('The intertia was not computed during do_numeric_factorization. Set compute_inertia to True.')
        return self._inertia

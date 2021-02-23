from .base_linear_solver_interface import LinearSolverInterface
from .results import LinearSolverStatus, LinearSolverResults
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

    def do_symbolic_factorization(self, matrix, raise_on_error=True):
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        return res

    def do_numeric_factorization(self, matrix, raise_on_error=True):
        if not isspmatrix_csc(matrix):
            matrix = matrix.tocsc()
        res = LinearSolverResults()
        try:
            self._lu = splu(matrix)
            res.status = LinearSolverStatus.successful
        except RuntimeError as err:
            if raise_on_error:
                raise err
            if 'Factor is exactly singular' in str(err):
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error

        if self.compute_inertia:
            eig = eigvals(matrix.toarray())
            pos_eig = np.count_nonzero((eig > 0))
            neg_eigh = np.count_nonzero((eig < 0))
            zero_eig = np.count_nonzero(eig == 0)
            self._inertia = (pos_eig, neg_eigh, zero_eig)

        return res

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

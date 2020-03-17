from .base_linear_solver_interface import LinearSolverInterface
from scipy.sparse.linalg import splu
from scipy.linalg import eigvalsh
from scipy.sparse import isspmatrix_csc
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector


class ScipyInterface(LinearSolverInterface):
    def __init__(self):
        self._lu = None
        self._inertia = None

    def do_symbolic_factorization(self, matrix):
        pass

    def do_numeric_factorization(self, matrix, compute_inertia=False):
        if not isspmatrix_csc(matrix):
            matrix = matrix.tocsc()
        self._lu = splu(matrix)
        if compute_inertia:
            eig = eigvalsh(matrix.toarray())
            pos_eig = (eig > 0).nonzero()[0]
            neg_eigh = (eig < 0).nonzero()[0]
            zero_eig = (eig == 0).nonzero()[0]
            self._inertia = (pos_eig, neg_eigh, zero_eig)

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
            raise RuntimeError('The intertia was not computed during do_numeric_factorization.')
        return self._inertia

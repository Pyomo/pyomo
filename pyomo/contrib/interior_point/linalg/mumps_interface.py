from .base_linear_solver_interface import LinearSolverInterface
from pyomo.contrib.pynumero.linalg.mumps_solver import MumpsCentralizedAssembledLinearSolver
from scipy.sparse import isspmatrix_coo, tril


class MumpsInterface(LinearSolverInterface):
    def __init__(self, par=1, comm=None, cntl_options=None, icntl_options=None):        
        self._mumps = MumpsCentralizedAssembledLinearSolver(sym=2,
                                                            par=par,
                                                            comm=comm)

        if cntl_options is None:
            cntl_options = dict()
        if icntl_options is None:
            icntl_options = dict()

        # These options are set in order to get the correct inertia.
        if 13 not in icntl_options:
            icntl_options[13] = 1
        if 24 not in icntl_options:
            icntl_options[24] = 0
            
        for k, v in cntl_options.items():
            self.set_cntl(k, v)
        for k, v in icntl_options.items():
            self.set_icntl(k, v)

        self._dim = None

    def do_symbolic_factorization(self, matrix):
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        nrows, ncols = matrix.shape
        self._dim = nrows
        self._mumps.do_symbolic_factorization(matrix)

    def do_numeric_factorization(self, matrix):
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        self._mumps.do_numeric_factorization(matrix)

    def do_back_solve(self, rhs):
        return self._mumps.do_back_solve(rhs)

    def get_inertia(self):
        num_negative_eigenvalues = self.mumps.get_infog(12)
        num_positive_eigenvalues = self._dim - num_negative_eigenvalues
        return (num_positive_eigenvalues, num_negative_eigenvalues, 0)

    def set_icntl(self, key, value):
        if key == 13:
            if value <= 0:
                raise ValueError('ICNTL(13) must be positive for the MumpsInterface.')
        elif key == 24:
            if value != 0:
                raise ValueError('ICNTL(24) must be 0 for the MumpsInterface.')
        self._mumps.set_icntl(key, value)

    def set_cntl(self, key, value):
        self._mumps.set_cntl(key, value)

    def get_info(self, key):
        return self._mumps.get_info(key)

    def get_infog(self, key):
        return self._mumps.get_infog(key)

    def get_rinfo(self, key):
        return self._mumps.get_rinfo(key)

    def get_rinfog(self, key):
        return self._mumps.get_rinfog(key)

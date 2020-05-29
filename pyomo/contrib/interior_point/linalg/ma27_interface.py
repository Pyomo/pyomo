from .base_linear_solver_interface import LinearSolverInterface
from .results import LinearSolverStatus, LinearSolverResults
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
from scipy.sparse import isspmatrix_coo, tril
from pyomo.contrib.pynumero.sparse import BlockVector


class InteriorPointMA27Interface(LinearSolverInterface):
    @classmethod
    def getLoggerName(cls):
        return 'ma27'

    def __init__(self, cntl_options=None, icntl_options=None, iw_factor=1.2, a_factor=2):
        self._ma27 = MA27Interface(iw_factor=iw_factor, a_factor=a_factor)

        if cntl_options is None:
            cntl_options = dict()
        if icntl_options is None:
            icntl_options = dict()

        for k, v in cntl_options.items():
            self.set_cntl(k, v)
        for k, v in icntl_options.items():
            self.set_icntl(k, v)

        self._dim = None
        self._num_status = None

    def do_symbolic_factorization(self, matrix, raise_on_error=True):
        self._num_status = None
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError('Matrix must be square')
        self._dim = nrows

        stat = self._ma27.do_symbolic_factorization(dim=self._dim, irn=matrix.row, icn=matrix.col)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError('Symbolic factorization was not successful; return code: ' + str(stat))
            if stat in {-3, -4}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-5, 3}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error
        return res

    def do_numeric_factorization(self, matrix, raise_on_error=True):
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError('Matrix must be square')
        if nrows != self._dim:
            raise ValueError('Matrix dimensions do not match the dimensions of '
                             'the matrix used for symbolic factorization')

        stat = self._ma27.do_numeric_factorization(irn=matrix.row, icn=matrix.col, dim=self._dim, entries=matrix.data)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError('Numeric factorization was not successful; return code: ' + str(stat))
            if stat in {-3, -4}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-5, 3}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error

        self._num_status = res.status

        return  res

    def increase_memory_allocation(self, factor):
        self._ma27.iw_factor *= factor
        self._ma27.a_factor *= factor

    def do_back_solve(self, rhs):
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
            result = _rhs
        else:
            result = rhs.copy()

        result = self._ma27.do_backsolve(result)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result

    def get_inertia(self):
        if self._num_status is None:
            raise RuntimeError('Must call do_numeric_factorization before inertia can be computed')
        if self._num_status != LinearSolverStatus.successful:
            raise RuntimeError('Can only compute inertia if the numeric factorization was successful.')
        num_negative_eigenvalues = self.get_info(15)
        num_positive_eigenvalues = self._dim - num_negative_eigenvalues
        return (num_positive_eigenvalues, num_negative_eigenvalues, 0)

    def set_icntl(self, key, value):
        self._ma27.set_icntl(key, value)

    def set_cntl(self, key, value):
        self._ma27.set_cntl(key, value)

    def get_icntl(self, key):
        return self._ma27.get_icntl(key)

    def get_cntl(self, key):
        return self._ma27.get_cntl(key)

    def get_info(self, key):
        return self._ma27.get_info(key)

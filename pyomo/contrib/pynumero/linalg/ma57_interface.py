from .base import DirectLinearSolverInterface, LinearSolverStatus, LinearSolverResults
from .ma57 import MA57Interface
from scipy.sparse import isspmatrix_coo, tril, spmatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from typing import Union, Tuple, Optional
import numpy as np


class MA57(DirectLinearSolverInterface):
    def __init__(
        self,
        cntl_options=None,
        icntl_options=None,
        work_factor=1.2,
        fact_factor=2,
        ifact_factor=2,
    ):
        self._ma57 = MA57Interface(
            work_factor=work_factor, fact_factor=fact_factor, ifact_factor=ifact_factor
        )

        if cntl_options is None:
            cntl_options = dict()
        if icntl_options is None:
            icntl_options = dict()

        for k, v in cntl_options.items():
            self.set_cntl(k, v)
        for k, v in icntl_options.items():
            self.set_icntl(k, v)

        self._dim = None
        self._row = None
        self._col = None

    def do_symbolic_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError("Matrix must be square")
        self._dim = nrows

        self._row = matrix.row
        self._col = matrix.col
        stat = self._ma57.do_symbolic_factorization(
            dim=self._dim, irn=matrix.row, jcn=matrix.col
        )
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError(
                    "Symbolic factorization was not successful; return code: "
                    + str(stat)
                )
            if stat in {-3, -4}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-5, 4}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error
        return res

    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError("Matrix must be square")
        if nrows != self._dim:
            raise ValueError(
                "Matrix dimensions do not match the dimensions of "
                "the matrix used for symbolic factorization"
            )

        if np.any(matrix.row != self._row) or np.any(matrix.col != self._col):
            raise ValueError("Matrix row or col ordering changed")
        stat = self._ma57.do_numeric_factorization(dim=self._dim, entries=matrix.data)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError(
                    "Numeric factorization was not successful; return code: "
                    + str(stat)
                )
            if stat in {-3, -4}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-5, 4}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error

        return res

    def do_back_solve(
        self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool = True
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
            result = _rhs
        else:
            result = rhs.copy()

        result = self._ma57.do_backsolve(result, copy=False)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result, LinearSolverResults(LinearSolverStatus.successful)

    def increase_memory_allocation(self, factor):
        self._ma57.work_factor *= factor
        self._ma57.fact_factor *= factor
        self._ma57.ifact_factor *= factor

    def set_icntl(self, key, value):
        self._ma57.set_icntl(key, value)

    def set_cntl(self, key, value):
        self._ma57.set_cntl(key, value)

    def get_icntl(self, key):
        return self._ma57.get_icntl(key)

    def get_cntl(self, key):
        return self._ma57.get_cntl(key)

    def get_info(self, key):
        return self._ma57.get_info(key)

    def get_rinfo(self, key):
        return self._ma57.get_rinfo(key)

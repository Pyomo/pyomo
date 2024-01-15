from .base import (
    DirectLinearSolverInterface,
    LinearSolverStatus,
    LinearSolverResults,
    LinearSolverInterface,
)
from scipy.sparse.linalg import splu, LinearOperator
from scipy.linalg import eigvals
from scipy.sparse import isspmatrix_csc, spmatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
import numpy as np
from typing import Union, Tuple, Optional, Callable


class ScipyLU(DirectLinearSolverInterface):
    def __init__(self):
        self._lu = None

    def do_symbolic_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        return res

    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        if not isspmatrix_csc(matrix):
            matrix = matrix.tocsc()
        res = LinearSolverResults()
        try:
            self._lu = splu(matrix)
            res.status = LinearSolverStatus.successful
        except RuntimeError as err:
            if raise_on_error:
                raise err
            if "Factor is exactly singular" in str(err):
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error

        return res

    def do_back_solve(
        self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool = True
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
        else:
            _rhs = rhs

        result = self._lu.solve(_rhs)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result, LinearSolverResults(LinearSolverStatus.successful)


class _LinearOperator(LinearOperator):
    def __init__(self, matrix: Union[spmatrix, BlockMatrix]):
        self._matrix = matrix
        shape = self._matrix.shape
        dtype = self._matrix.dtype
        super(_LinearOperator, self).__init__(shape=shape, dtype=dtype)

    def _matvec(self, x):
        return self._matrix * x

    def _adjoint(self):
        return _LinearOperator(self._matrix.transpose())


class ScipyIterative(LinearSolverInterface):
    def __init__(self, method: Callable, options=None):
        self.method = method
        if options is None:
            self.options = dict()
        else:
            self.options = dict(options)

    def solve(
        self,
        matrix: Union[spmatrix, BlockMatrix],
        rhs: Union[np.ndarray, BlockVector],
        raise_on_error: bool = True,
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        # eventually, we will want to remove the .tocoo(), but we would have to first
        # figure out how to deal with np.asarray for BlockVector and MPIBlockVector
        # np.asarray is used on rhs within the scipy iterative solvers
        linear_operator = _LinearOperator(matrix.tocoo())
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
        else:
            _rhs = rhs
        result, info = self.method(linear_operator, _rhs, **self.options)
        if info == 0:
            stat = LinearSolverStatus.successful
        else:
            stat = LinearSolverStatus.error

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result, LinearSolverResults(stat)

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.extensions.hsl import _MA57_LinearSolver
if not _MA57_LinearSolver.available():
    raise ImportError('MA57 library not available.'
                      ' Make sure libpynumero_MA57 is in pynumero.extensions.lib')

from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from pyomo.contrib.pynumero.sparse.utils import is_symmetric_sparse
from scipy.sparse import isspmatrix_coo, coo_matrix

import numpy as np


class MA57LinearSolver(object):
    """
    Interface to HSL MA27 sparse direct Solver

    Attributes
    ----------
    _ma27: _MA27_LinearSolver
        wrapper to ma27 interface

    Parameters
    ----------
    pivot_tol: float64, optional
        pivot tolerance used by mumps (default 1e-8)

    """
    def __init__(self, pivotol=1e-8, prealocate_factor=1.05):

        self._ma57 = _MA57_LinearSolver(pivotol, prealocate_factor)
        self._nnz = 0
        self._dim = 0
        self._row_blocks = -1
        self._col_blocks = -1
        self._col_sizes = None

    def _get_num_neg_evals(self):
        return self._ma57.get_num_neg_evals()

    def do_symbolic_factorization(self, matrix, include_diagonal=False, check_symmetry=True):
        """
        Identify pivots from the used for numerical factorization.

        Parameters
        ----------
        matrix: spmatrix or BlockMatrix
            matrix to be factorized
        include_diagonal: boolean, optional
            True if regularization is going to be used (default False)
        check_symmetry: boolean, optional
            True for checking symmetry in matrix (default True)

        """
        if check_symmetry:
            assert is_symmetric_sparse(matrix), 'Matrix is not symmetric'

        if isinstance(matrix, BlockMatrix):

            assert not matrix.has_empty_rows(), \
                "Block matrix has undefined row sizes"

            shape = matrix.bshape
            self._row_blocks = shape[0]
            self._col_blocks = shape[1]
            self._col_sizes = matrix.col_block_sizes()

            expanded_matrix = matrix.tocoo()
            lower_mask = expanded_matrix.row >= expanded_matrix.col
            irows = expanded_matrix.row[lower_mask]
            jcols = expanded_matrix.col[lower_mask]
            dim = expanded_matrix.shape[0]
            if include_diagonal:
                diag = np.arange(dim, dtype=np.intc)
                irows = np.concatenate((expanded_matrix.row[lower_mask], diag))
                jcols = np.concatenate((expanded_matrix.col[lower_mask], diag))

            irows += 1
            jcols += 1

            status = self._ma57.DoSymbolicFactorization(dim, irows, jcols)
            self._dim = dim
            self._nnz = irows.size

            irows -= 1
            jcols -= 1

        elif isspmatrix_coo(matrix):

            # get lower triangular entries
            lower_mask = matrix.row >= matrix.col
            irows = matrix.row[lower_mask]
            jcols = matrix.col[lower_mask]
            dim = matrix.shape[0]
            if include_diagonal:
                diag = np.arange(dim, dtype=np.intc)
                irows = np.concatenate((matrix.row[lower_mask], diag))
                jcols = np.concatenate((matrix.col[lower_mask], diag))

            # deals with off set
            irows += 1
            jcols += 1

            status = self._ma57.DoSymbolicFactorization(dim, irows, jcols)
            self._dim = dim
            self._nnz = irows.size
        else:
            raise RuntimeError("Matrix must be coo_matrix or a block_matrix")
        return status

    def do_numeric_factorization(self, matrix, diagonal=None, desired_num_neg_eval=-1):
        """
        Factorizes matrix using LU factorization from ma27

        Parameters
        ----------
        matrix: spmatrix or BlockMatrix
            matrix to be factorized
        diagonal: array_like, optional
            array with values to be added for regularization (default None)
        desired_num_neg_eval: int, optinal
            number of negative eigenvalues to check in regularization (default -1)

        """
        if isinstance(matrix, BlockMatrix):
            assert self._dim != 0 and self._nnz != 0, \
                'Call symbolic factorization first'
            assert self._dim == matrix.shape[0], \
                'Need to pass the same matrix that was factorized symbolically'
            assert self._col_blocks == matrix.bshape[1] and self._row_blocks == matrix.bshape[0], \
                'Need to pass the same block matrix that was factorized symbolically'

            expanded_matrix = matrix.tocoo()
            lower_mask = expanded_matrix.row >= expanded_matrix.col
            lower_rows = expanded_matrix.row[lower_mask]
            msg = 'Dimensions do not agree. Make sure diagonal is passed if symbolic factorization included diagonal'
            assert ((diagonal is None and lower_rows.size == self._nnz) or
                    (diagonal is not None and lower_rows.size + self._dim == self._nnz)), msg

            values = expanded_matrix.data[lower_mask]
            if diagonal is not None:
                values = np.concatenate((values, diagonal))

            return self._ma57.DoNumericFactorization(self._dim, values, desired_num_neg_eval)

        elif isspmatrix_coo(matrix):
            lower_mask = matrix.row >= matrix.col
            lower_rows = matrix.row[lower_mask]
            assert self._dim != 0 and self._nnz != 0, \
                'Call symbolic factorization first'
            assert self._dim == matrix.shape[0], \
                'Need to pass the same matrix that was factorized symbolically'

            assert ((diagonal is None and lower_rows.size == self._nnz) or
                    (diagonal is not None and lower_rows.size + self._dim == self._nnz)), \
                'Dimensions do not agree. Make sure diagonal is passed if symbolic factorization included diagonal'

            values = matrix.data[lower_mask]
            if diagonal is not None:
                values = np.concatenate((values, diagonal))

            return self._ma57.DoNumericFactorization(self._dim, values, desired_num_neg_eval)

        else:
            raise RuntimeError('Matrix must be coo_matrix or a block_matrix')

    def do_back_solve(self, rhs, **kwargs):
        """
        Computes solution

        Parameters
        ----------
        rhs: array_like
            right-hand-side vector

        """
        flat_solution = kwargs.pop('flat_solution', False)
        matrix = kwargs.pop('matrix', None)
        max_iter_ref = kwargs.pop('max_iter_ref', 1)
        tol_iter_ref = kwargs.pop('tol_iter_ref', 1e-8)

        assert self._dim == rhs.size, \
            'RHS dimension does not agree with matrix {} != {}'.format(self._dim,
                                                                       rhs.size)
        flat_rhs = rhs.flatten()
        x = np.zeros(self._dim)
        self._ma57.DoBacksolve(flat_rhs, x)

        if matrix is not None and max_iter_ref > 0:
            xr = x.flatten()
            flat_matrix = matrix.tocsr()

            for i in range(max_iter_ref):
                res = flat_rhs - flat_matrix.dot(xr)
                d = self.do_back_solve(res, flat_solution=True)
                xr += d
                if np.linalg.norm(res, ord=np.inf) <= tol_iter_ref:
                    break
            x = xr

        if flat_solution:
            return x

        if self._col_blocks == -1 and not isinstance(rhs, BlockVector):
            return x
        if isinstance(rhs, BlockVector):
            block_x = rhs.copy()
            block_x.copyfrom(x)
            return block_x
        arrays = [np.zeros(s) for s in self._col_sizes]
        block_x = BlockVector(arrays)
        block_x.copyfrom(x)
        return block_x

    def solve(self,
              matrix,
              rhs,
              diagonal=None,
              do_symbolic=True,
              check_symmetry=True,
              desired_num_neg_eval=-1,
              max_iter_ref=1,
              tol_iter_ref=1e-8):
        """
        Solve linear system with direct solver ma57

        Parameters
        ----------
        matrix: spmatrix or BlockMatrix
            matrix to be factorized
        rhs: array_like
            right-hand-side vector
        diagonal: array_like, optional
            array with values to be added for regularization (default None)
        do_symbolic: boolean, optional
            True if symbolic factorization needs to be run (default True).
        check_symmetry: boolean, optional
            True for checking symmetry in matrix (default True)
        desired_num_neg_eval: int, optinal
            number of negative eigenvalues to check in regularization (default -1)
        max_iter_ref: int, optional
            number of iterative refinement iterations
        tol_iter_ref: float64, optional
            tolerance for iterative refinement

        """
        include_diagonal = False
        if diagonal is not None:
            include_diagonal = True

        if do_symbolic:
            self.do_symbolic_factorization(matrix, include_diagonal, check_symmetry)

        self.do_numeric_factorization(matrix, diagonal, desired_num_neg_eval)

        if max_iter_ref>0:
            x = self.do_back_solve(rhs,
                                   matrix=matrix,
                                   max_iter_ref=max_iter_ref,
                                   tol_iter_ref=tol_iter_ref)
        else:
            x = self.do_back_solve(rhs)

        return x


if __name__ == "__main__":

    row = np.array([0, 1, 2, 1, 2, 2])
    col = np.array([0, 0, 0, 1, 1, 2])
    data = np.array([1, 7, 3, 4, -5, 6])
    off_diagonal_mask = row != col
    new_row = np.concatenate([row, col[off_diagonal_mask]])
    new_col = np.concatenate([col, row[off_diagonal_mask]])
    new_data = np.concatenate([data, data[off_diagonal_mask]])
    A = coo_matrix((new_data, (new_row, new_col)), shape=(3, 3))
    dense_A = A.toarray()
    print(dense_A)

    linear_solver = MA57LinearSolver()
    b = np.array([1, 1, 0])
    M = A.tocoo()
    x = linear_solver.solve(M, b)
    print(x)
    print(np.linalg.solve(dense_A, b))

    print(np.linalg.norm(b - M.dot(x), ord=np.inf))

    linear_solver.do_symbolic_factorization(M)
    status = linear_solver.do_numeric_factorization(M)
    print(status)

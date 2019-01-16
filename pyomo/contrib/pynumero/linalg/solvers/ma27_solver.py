
try:
    import pyomo.contrib.pynumero.extensions.hsl as _hsl
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing hsl while running ma27 linear solver. '
                      'Make sure libpynumero_HSL is installed and added to path.')

from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from pyomo.contrib.pynumero.sparse.utils import is_symmetric_sparse
from scipy.sparse import isspmatrix_coo, coo_matrix

import numpy as np


class MA27LinearSolver(object):

    def __init__(self, pivotol=1e-8):

        self._ma27 = _hsl.MA27_LinearSolver(pivotol)
        self._nnz = 0
        self._dim = 0
        self._row_blocks = -1
        self._col_blocks = -1
        self._col_sizes = None

    def _get_num_neg_evals(self):
        return self._ma27.get_num_neg_evals()

    def do_symbolic_factorization(self, matrix, include_diagonal=False, check_symmetry=True):

        if check_symmetry:
            assert is_symmetric_sparse(matrix), 'Matrix is not symmetric'

        if isinstance(matrix, BlockMatrix):

            assert not matrix.has_empty_rows(), "Block matrix has undefined row sizes"

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

            self._ma27.DoSymbolicFactorization(dim, irows, jcols)
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

            self._ma27.DoSymbolicFactorization(dim, irows, jcols)
            self._dim = dim
            self._nnz = irows.size

        else:
            raise RuntimeError("Matrix must be coo_matrix or a block_matrix")

    def do_numeric_factorization(self, matrix, diagonal=None, desired_num_neg_eval=-1):

        if isinstance(matrix, BlockMatrix):
            msg = 'Call symbolic factorization first'
            assert self._dim != 0 and self._nnz != 0, msg
            msg = 'Need to pass the same matrix that was factorized symbolically'
            assert self._dim == matrix.shape[0], msg
            msg = 'Need to pass the same block matrix that was factorized symbolically'
            assert self._col_blocks == matrix.bshape[1] and self._row_blocks == matrix.bshape[0], msg

            expanded_matrix = matrix.tocoo()
            lower_mask = expanded_matrix.row >= expanded_matrix.col
            lower_rows = expanded_matrix.row[lower_mask]
            msg = 'Dimensions do not agree. Make sure diagonal is passed if symbolic factorization included diagonal'
            assert ((diagonal is None and lower_rows.size == self._nnz) or
                    (diagonal is not None and lower_rows.size + self._dim == self._nnz)), msg

            values = expanded_matrix.data[lower_mask]
            if diagonal is not None:
                values = np.concatenate((values, diagonal))

            return self._ma27.DoNumericFactorization(self._dim, values, desired_num_neg_eval)

        elif isspmatrix_coo(matrix):
            lower_mask = matrix.row >= matrix.col
            lower_rows = matrix.row[lower_mask]
            msg = 'Call symbolic factorization first'
            assert self._dim != 0 and self._nnz != 0, msg
            msg = 'Need to pass the same matrix that was factorized symbolically'
            assert self._dim == matrix.shape[0], msg

            msg = 'Dimensions do not agree. Make sure diagonal is passed if symbolic factorization included diagonal'
            assert ((diagonal is None and lower_rows.size == self._nnz) or
                    (diagonal is not None and lower_rows.size + self._dim == self._nnz)), msg

            values = matrix.data[lower_mask]
            if diagonal is not None:
                values = np.concatenate((values, diagonal))

            return self._ma27.DoNumericFactorization(self._dim, values, desired_num_neg_eval)

        else:
            raise RuntimeError('Matrix must be coo_matrix or a block_matrix')

    def do_back_solve(self, rhs):

        msg = 'RHS dimension does not agree with matrix {} != {}'.format(self._dim,
                                                                         rhs.size)
        assert self._dim == rhs.size, msg
        flat_rhs = rhs.flatten()
        x = np.zeros(self._dim)
        self._ma27.DoBacksolve(flat_rhs, x)
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

    def solve(self, matrix, rhs, diagonal=None, do_symbolic=True, check_symmetry=True, desired_num_neg_eval=-1):

        include_diagonal = False
        if diagonal is not None:
            include_diagonal = True

        if do_symbolic:
            self.do_symbolic_factorization(matrix, include_diagonal, check_symmetry)

        self.do_numeric_factorization(matrix, diagonal, desired_num_neg_eval)
        return self.do_back_solve(rhs)


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

    linear_solver = MA27LinearSolver()
    b = np.array([1, 1, 0])
    M = A.tocoo()
    x = linear_solver.solve(M, b)
    print(x)
    print(np.linalg.solve(dense_A, b))

    linear_solver.do_symbolic_factorization(M)
    status = linear_solver.do_numeric_factorization(M)
    print(status)
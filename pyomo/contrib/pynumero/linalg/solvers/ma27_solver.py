
try:
    import pyomo.contrib.pynumero.extensions.hsl as _hsl
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing hsl while running ma27 linear solver. '
                      'Make sure libdinopy_HSL is installed and added to path.')

from pyomo.contrib.pynumero.sparse import (COOMatrix,
                             COOSymMatrix,
                             BlockMatrix,
                             BlockSymMatrix,
                             BlockVector)
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

    def do_symbolic_factorization(self, matrix, include_diagonal=False):

        if isinstance(matrix, COOSymMatrix):
            # deals with off set

            irows = matrix.row
            jcols = matrix.col
            dim = matrix.shape[0]
            if include_diagonal:
                diag = np.arange(dim, dtype=np.intc)
                irows = np.concatenate((matrix.row, diag))
                jcols = np.concatenate((matrix.col, diag))

            irows += 1
            jcols += 1

            self._ma27.DoSymbolicFactorization(dim, irows, jcols)
            self._dim = dim
            self._nnz = irows.size

            irows -= 1
            jcols -= 1

        elif isinstance(matrix, BlockSymMatrix):

            assert not matrix.has_empty_rows(), "Block matrix has undefined row sizes"

            shape = matrix.bshape
            self._row_blocks = shape[0]
            self._col_blocks = shape[1]
            self._col_sizes = matrix.col_block_sizes()

            unblocked_matrix = matrix.tocoo()
            irows = unblocked_matrix.row
            jcols = unblocked_matrix.col
            dim = unblocked_matrix.shape[0]
            if include_diagonal:
                diag = np.arange(dim, dtype=np.intc)
                irows = np.concatenate((unblocked_matrix.row, diag))
                jcols = np.concatenate((unblocked_matrix.col, diag))

            irows += 1
            jcols += 1

            self._ma27.DoSymbolicFactorization(dim, irows, jcols)
            self._dim = dim
            self._nnz = irows.size

            irows -= 1
            jcols -= 1

        else:
            raise RuntimeError("Matrix must be COOSymMatrix or a block symmetric matrix")

    def do_numeric_factorization(self, matrix, diagonal=None, desired_num_neg_eval=-1):

        if isinstance(matrix, COOSymMatrix):
            msg = 'Call symbolic factorization first'
            assert self._dim != 0 and self._nnz != 0, msg
            msg = 'Need to pass the same matrix that was factorized symbolically'
            assert self._dim == matrix.shape[0], msg
            msg = 'Need to pass the same block matrix that was factorized symbolically'
            assert self._col_blocks == -1 and self._row_blocks == -1, msg

            msg = 'Dimensions do not agree. Make sure diagonal is passed if symbolic factorization included diagonal'
            assert ((diagonal is None and matrix.nnz == self._nnz) or
                    (diagonal is not None and matrix.nnz + self._dim == self._nnz)), msg

            values = matrix.data
            if diagonal is not None:
                values = np.concatenate((values, diagonal))

            return self._ma27.DoNumericFactorization(self._dim, values, desired_num_neg_eval)

        elif isinstance(matrix, BlockSymMatrix):
            msg = 'Call symbolic factorization first'
            assert self._dim != 0 and self._nnz != 0, msg
            msg = 'Need to pass the same matrix that was factorized symbolically'
            assert self._dim == matrix.shape[0], msg
            msg = 'Need to pass the same block matrix that was factorized symbolically'
            assert self._col_blocks == matrix.bshape[1] and self._row_blocks == matrix.bshape[0], msg

            msg = 'Dimensions do not agree. Make sure diagonal is passed if symbolic factorization included diagonal'
            assert ((diagonal is None and matrix.nnz == self._nnz) or
                    (diagonal is not None and matrix.nnz + self._dim == self._nnz)), msg

            values = matrix.coo_data()

            if diagonal is not None:
                values = np.concatenate((values, diagonal))

            return self._ma27.DoNumericFactorization(self._dim, values, desired_num_neg_eval)

        else:
            raise RuntimeError('Matrix must be COOSymMatrix or a block symmetric matrix')

    def do_back_solve(self, rhs):

        msg = 'RHS dimension does not agree with matrix'
        assert self._dim == rhs.size, msg
        flat_rhs = rhs.flatten()
        x = np.zeros(self._dim)
        self._ma27.DoBacksolve(flat_rhs, x)
        if self._col_blocks == -1:
            return x
        arrays = [np.zeros(s) for s in self._col_sizes]
        block_x = BlockVector(arrays)
        block_x.copyfrom(x)
        return block_x

    def solve(self, matrix, rhs, diagonal=None, do_symbolic=True, desired_num_neg_eval=-1):

        include_diagonal = False
        if diagonal is not None:
            include_diagonal = True

        if do_symbolic:
            self.do_symbolic_factorization(matrix, include_diagonal)

        self.do_numeric_factorization(matrix, diagonal, desired_num_neg_eval)
        return self.do_back_solve(rhs)


if __name__ == "__main__":

    row = np.array([0, 1, 2, 1, 2, 2])
    col = np.array([0, 0, 0, 1, 1, 2])
    data = np.array([1, 7, 3, 4, -5, 6])
    A = COOSymMatrix((data, (row, col)), shape=(3, 3))
    dense_A = A.toarray()
    print(dense_A)

    linear_solver = MA27LinearSolver()
    b = np.array([1, 1, 0])
    x = linear_solver.solve(A, b)
    print(x)
    print(np.linalg.solve(dense_A, b))

    linear_solver.do_symbolic_factorization(A)
    status = linear_solver.do_numeric_factorization(A)
    print(status)
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from scipy.sparse import isspmatrix_coo, coo_matrix
import numpy as np

try:
    import mumps
except ImportError as e:
    raise ImportError('Error importing mumps. Install pymumps '
                      'conda install -c conda-forge pymumps')

from pyomo.contrib.pynumero.sparse.utils import is_symmetric_sparse
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector


class MUMPSSymLinearSolver(object):

    def __init__(self, pivotol=1e-8, **kwargs):

        pivtolmax = kwargs.pop('pivtolmax', 0.1)
        mumps_mem_percent = kwargs.pop('mumps_mem_percent', 1000)
        mumps_permuting_scaling = kwargs.pop('mumps_permuting_scaling', 7)
        mumps_pivot_order = kwargs.pop('mumps_pivot_order', 7)
        mumps_scaling = kwargs.pop('mumps_scaling', 77)
        mumps_dep_tol = kwargs.pop('mumps_dep_tol', 0.0)

        self._pivtol = pivotol
        self._mumps_mem_percent = mumps_mem_percent
        self._mumps_permuting_scaling = mumps_permuting_scaling
        self._mumps_pivot_order = mumps_pivot_order
        self._mumps_scaling = mumps_scaling
        self._mumps_dep_tol = mumps_dep_tol

        self.ctx = mumps.DMumpsContext(sym=2, par=1)
        self.ctx.id.cntl[0] = self._pivtol
        self.ctx.set_silent()
        self.ctx.id.icntl[5] = self._mumps_permuting_scaling
        self.ctx.id.icntl[6] = self._mumps_pivot_order
        self.ctx.id.icntl[7] = self._mumps_scaling
        self.ctx.id.icntl[13] = int(self._mumps_mem_percent * 2.0)
        self.ctx.id.icntl[12] = 1 # to get correct negative eigenvalues

        self._nnz = 0
        self._dim = 0

        self._row_blocks = -1
        self._col_blocks = -1
        self._col_sizes = None

    def __del__(self):
        self.ctx.destroy()

    def _get_num_neg_evals(self):
        n_negevals = self.ctx.id.infog[11]
        return n_negevals

    def do_symbolic_factorization(self, matrix, include_diagonal=False, check_symmetry=True):

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

            self.ctx.set_shape(dim)
            self.ctx.set_centralized_assembled_rows_cols(irows, jcols)
            self.ctx.run(job=1)
            status = self.ctx.id.infog[0]
            if status != 0:
                raise RuntimeError("MUMPS returned INFO(1) = {} fatal error".format(status))
            self._dim = dim
            self._nnz = irows.size

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

            self.ctx.set_shape(dim)
            self.ctx.set_centralized_assembled_rows_cols(irows, jcols)
            self.ctx.run(job=1)
            status = self.ctx.id.infog[0]
            if status != 0:
                raise RuntimeError("MUMPS returned INFO(1) = {} fatal error".format(status))
            self._dim = dim
            self._nnz = irows.size
        else:
            raise RuntimeError("Matrix must be coo_matrix or a block_matrix")

    def do_numeric_factorization(self, matrix, diagonal=None, desired_num_neg_eval=-1):

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
            assert ((diagonal is None and lower_rows.size == self._nnz) or
                    (diagonal is not None and lower_rows.size + self._dim == self._nnz)), \
                'Dimensions do not agree. Make sure diagonal is passed if symbolic factorization included diagonal'

            values = expanded_matrix.data[lower_mask]
            if diagonal is not None:
                values = np.concatenate((values, diagonal))
            values = values.astype('d')
            self.ctx.set_centralized_assembled_values(values)
            status = self.ctx.id.infog[0]
            try:
                self.ctx.run(job=2)  # Factorization
                status = self.ctx.id.infog[0]
            except:
                status = self.ctx.id.infog[0]

            if status == -8 or status == -9:
                # need more memory
                n_updates = 1
                for i in range(20):
                    print("MUMPS returned INFO(1) = {} and "
                          "requires more memory, reallocating.  Attempt {}\n".format(status, i + 1)
                          )
                    self.ctx.id.icntl[13] = int(self.ctx.id.icntl[13] * 2.0)
                    try:
                        self.ctx.run(job=2)  # Factorization
                        status = self.ctx.id.infog[0]
                    except:
                        status = self.ctx.id.infog[0]

                    if status != -8 and status != -9:
                        break
                if n_updates >= 20:
                    raise RuntimeError("MUMPS returned INFO(1) = {} fatal error".format(status))
                # reset memory
                self.ctx.id.icntl[13] = int(self._mumps_mem_percent * 2.0)
            if status == 0:
                if desired_num_neg_eval != -1:
                    n_negevals = self.ctx.id.infog[11]
                    if desired_num_neg_eval != n_negevals:
                        return 2
                    return 0
                return 0
            if status == -10:
                return 1  # singular matrix
            if status < 0:
                raise RuntimeError("MUMPS returned INFO(1) = {} fatal error".format(status))

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

            values = values.astype('d')
            self.ctx.set_centralized_assembled_values(values)
            status = self.ctx.id.infog[0]
            try:
                self.ctx.run(job=2)  # Factorization
                status = self.ctx.id.infog[0]
            except:
                status = self.ctx.id.infog[0]

            if status == -8 or status == -9:
                # need more memory
                n_updates = 1
                for i in range(20):
                    print("MUMPS returned INFO(1) = {} and "
                          "requires more memory, reallocating.  Attempt {}\n".format(status, i+1)
                          )
                    self.ctx.id.icntl[13] = int(self.ctx.id.icntl[13] * 2.0)
                    try:
                        self.ctx.run(job=2)  # Factorization
                        status = self.ctx.id.infog[0]
                    except:
                        status = self.ctx.id.infog[0]

                    if status != -8 and status != -9:
                        break
                if n_updates >= 20:
                    raise RuntimeError("MUMPS returned INFO(1) = {} fatal error".format(status))
                # reset memory
                self.ctx.id.icntl[13] = int(self._mumps_mem_percent * 2.0)
            if status == 0:
                if desired_num_neg_eval != -1:
                    n_negevals = self.ctx.infog[11]
                    if desired_num_neg_eval != n_negevals:
                        return 2
                    return 0
                return 0
            if status == -10:
                return 1  # singular matrix
            if status < 0:
                raise RuntimeError("MUMPS returned INFO(1) = {} fatal error".format(status))
        else:
            raise RuntimeError('Matrix must be coo_matrix or a block_matrix')

    def do_back_solve(self, rhs, **kwargs):

        flat_solution = kwargs.pop('flat_solution', False)
        matrix = kwargs.pop('matrix', None)
        max_iter_ref = kwargs.pop('max_iter_ref', 1)
        tol_iter_ref = kwargs.pop('tol_iter_ref', 1e-8)

        assert self._dim == rhs.size, \
            'RHS dimension does not agree with matrix'
        flat_rhs = rhs.flatten()
        x = flat_rhs.copy()
        self.ctx.set_rhs(x)
        status = self.ctx.id.infog[0]
        try:
            self.ctx.run(job=3)  # Solve
            status = self.ctx.id.infog[0]
        except:
            status = self.ctx.id.infog[0]
        if status < 0:
            raise RuntimeError("MUMPS returned INFO(1) = {} fatal error".format(status))

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

        include_diagonal = False
        if diagonal is not None:
            include_diagonal = True

        if do_symbolic:
            self.do_symbolic_factorization(matrix, include_diagonal, check_symmetry)

        status = self.do_numeric_factorization(matrix, diagonal, desired_num_neg_eval)
        if status == 1:
            raise RuntimeError('Matrix is singular')

        # if max_iterative_refinement >0:
        #     self.ctx.id.icntl[9] = max_iterative_refinement
        # x = self.do_back_solve(rhs)
        # self.ctx.id.icntl[9] = 0

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
    data = np.array([1.0, 7.0, 3.0, 4.0, -5.0, 6.0], dtype='d')
    off_diagonal_mask = row != col
    new_row = np.concatenate([row, col[off_diagonal_mask]])
    new_col = np.concatenate([col, row[off_diagonal_mask]])
    new_data = np.concatenate([data, data[off_diagonal_mask]])
    A = coo_matrix((data, (row, col)), shape=(3, 3))
    b = np.array([1, 1, 0], dtype='d')
    print(A.toarray())

    linear_solver = MUMPSSymLinearSolver()
    A = coo_matrix((new_data, (new_row, new_col)), shape=(3, 3))
    linear_solver.do_symbolic_factorization(A)
    status = linear_solver.do_numeric_factorization(A)
    x = linear_solver.do_back_solve(b)
    print(x)
    print(linear_solver.solve(A, b))
    print(np.linalg.norm(b - A.dot(x), ord=np.inf))
    linear_solver2 = MUMPSSymLinearSolver()

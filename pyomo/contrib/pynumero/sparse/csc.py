from scipy.sparse import csc_matrix as scipy_csc_matrix
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse import issparse
from scipy.sparse._sparsetools import csc_tocsr
from scipy.sparse.sputils import (upcast,
                                  upcast_char,
                                  get_index_dtype)

from scipy.sparse import _sparsetools

try:
    from pyomo.contrib.pynumero.extensions.sparseutils import (csr_matvec_no_diag,
                                                               csc_matvec_no_diag)
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing sparseutils while running coo interface. '
                      'Make sure libpynumero_SPARSE is installed and added to path.')

from pyomo.contrib.pynumero.sparse.base import SparseBase
from pyomo.contrib.pynumero.sparse.utils import (is_symmetric_dense,
                                                 _convert_matrix_to_symmetric,
                                                 _is_symmetric_numerically)
import numpy as np

__all__ = ['CSCMatrix', 'CSCSymMatrix']


class CSCMatrix(SparseBase, scipy_csc_matrix):
    """
        Compressed Sparse Column matrix

        | This can be instantiated in several ways
        |     CSCMatrix(D)
        |         with a dense matrix D
        |     CSCMatrix(S)
        |         with another sparse matrix S (equivalent to S.tocoo())
        |     CSCMatrix((M, N), [dtype])
        |         to construct an empty matrix with shape (M, N) dtype is optional, defaulting to dtype=d.
        |     CSCMatrix((data, (i, j)), [shape=(M, N)])
        |         to construct from three arrays:
        |             data[:] the entries of the matrix, in any order
        |             i[:] the row indices of the matrix entries
        |             j[:] the column indices of the matrix entries
        |         Where A[i[k], j[k]] = data[k]. When shape is not specified, it is inferred from the index arrays
        |     CSCMatrix((data, indices, indptr), [shape=(M, N)])
        |         to construct from three arrays:
        |             is the standard CSC representation where the row indices for column i are stored in
        |             indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in
        |             data[indptr[i]:indptr[i+1]]. If the shape parameter is not supplied, the matrix dimensions are inferred from
        |             the index arrays.
    """
    def __init__(self, arg1, shape=None, dtype=None, copy=False, **kwargs):

        # include upper triangular if arg1 is symmetric
        expand_symmetry = kwargs.pop('expand_symmetry', True)
        if expand_symmetry and isinstance(arg1, SparseBase):
            if arg1.is_symmetric:
                arg1 = arg1.tofullmatrix().tocsc()
        scipy_csc_matrix.__init__(self, arg1, shape=shape, dtype=dtype, copy=copy)
        SparseBase.__init__(self)

    def tosymcsc(self, copy=False):
        raise NotImplementedError('Not supported')

    def tocsr(self, copy=False):
        M, N = self.shape
        idx_dtype = get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, N))
        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csc_tocsr(M, N,
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        from pyomo.contrib.pynumero.sparse.csr import CSRMatrix
        A = CSRMatrix((data, indices, indptr), shape=self.shape, copy=False)
        A.has_sorted_indices = True
        return A

    def tocoo(self, copy=True):
        major_dim, minor_dim = self._swap(self.shape)
        minor_indices = self.indices
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
        _sparsetools.expandptr(major_dim, self.indptr, major_indices)
        row, col = self._swap((major_indices, minor_indices))

        from pyomo.contrib.pynumero.sparse.coo import COOMatrix
        return COOMatrix((self.data, (row, col)), self.shape, copy=copy,
                         dtype=self.dtype)

    def tocsc(self, copy=False):
        # copy only there to agree with the signature
        return self

    def todok(self, copy=False):
        raise NotImplementedError('Not supported')

    def todia(self, copy=False):
        raise NotImplementedError('Not supported')

    def tolil(self, copy=False):
        raise NotImplementedError('Not supported')

    def toscipy(self):
        return scipy_csc_matrix(self)

    def tofullmatrix(self):
        return self

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape

        from pyomo.contrib.pynumero.sparse.csr import CSRMatrix
        return CSRMatrix((self.data, self.indices,
                          self.indptr), (N, M), copy=copy)

    def _with_data(self, data,copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        if copy:
            return self.__class__((data, self.indices.copy(), self.indptr.copy()),
                                  shape=self.shape, dtype=data.dtype)
        else:
            return self.__class__((data, self.indices, self.indptr),
                                  shape=self.shape, dtype=data.dtype)

    def _add_sparse(self, other):
        if isinstance(other, SparseBase):
            if other.is_symmetric:
                return super(CSCMatrix, self)._add_sparse(other.tofullmatrix())
            return super(CSCMatrix, self)._add_sparse(other)
        if issparse(other):
            raise RuntimeError("Addition not supported with scipy matrices")
        raise RuntimeError("Sparse format not recognized {}".format(type(other)))

    def _sub_sparse(self, other):
        if isinstance(other, SparseBase):
            if other.is_symmetric:
                return super(CSCMatrix, self)._sub_sparse(other.tofullmatrix())
            return super(CSCMatrix, self)._sub_sparse(other)
        if issparse(other):
            raise RuntimeError("Subtraction not supported with scipy matrices")
        raise RuntimeError("Sparse format not recognized {}".format(type(other)))

    def _mul_sparse_matrix(self, other):

        if isinstance(other, SparseBase):

            if other.is_symmetric:
                expanded_other = other.tofullmatrix()
                result = super()._mul_sparse_matrix(expanded_other)
                if self.shape[0] == expanded_other.shape[1]:
                    if _is_symmetric_numerically(result):
                        return _convert_matrix_to_symmetric(result, check_symmetry=False)
                return result
            from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
            if isinstance(other, BlockMatrix):
                raise NotImplementedError("Not supported yet")
                expanded_other = other.tocsc()
                result = super()._mul_sparse_matrix(expanded_other)
                if self.shape[0] == expanded_other.shape[1]:
                    if _is_symmetric_numerically(result):
                        return _convert_matrix_to_symmetric(result, check_symmetry=False)
                return result

            result = super()._mul_sparse_matrix(other)
            if self.shape[0] == other.shape[1]:
                if _is_symmetric_numerically(result):
                    return _convert_matrix_to_symmetric(result, check_symmetry=False)
            return result
        if issparse(other):
            raise RuntimeError("Multiplication not supported with scipy matrices")
        raise RuntimeError("Format not recognized {}".format(type(other)))

    def getcol(self, j):
        return CSCMatrix(self.toscipy().getcol(j))

    def getrow(self, i):
        from pyomo.contrib.pynumero.sparse.csr import CSRMatrix
        return CSRMatrix(super(CSCMatrix, self).getrow(i))

    def __repr__(self):
        return 'CSCMatrix{}'.format(self.shape)


# this matrix will only store the lower triangular indices
class CSCSymMatrix(CSCMatrix):

    def __init__(self, arg1, shape=None, dtype=None, copy=False, **kwargs):

        # check if dense matrix is symmetric
        if isinstance(arg1, np.ndarray):
            if not is_symmetric_dense(arg1):
                raise RuntimeError("ndarray is not symmetric")
            # keep only lower triangular
            arg1 = np.tril(arg1)

        # symmetric matrices don't expand symmetry
        expand_symmetry = kwargs.pop('expand_symmetry', False)

        error_msg = "Symmetric matrices only store lower triangular"
        assert not expand_symmetry, error_msg
        super().__init__(arg1,
                         shape=shape,
                         dtype=dtype,
                         copy=copy,
                         expand_symmetry=expand_symmetry,
                         **kwargs)

        # add check to verify square matrix
        if self.shape[0] != self.shape[1]:
            raise RuntimeError('A rectangular matrix is not symmetric')

        # check nnz is less than the full lower triangular
        if self.nnz > self.shape[0] * (self.shape[0] + 1) / 2:
            raise RuntimeError('CSCSymMatrix only store lower triangular entries. Too many nnz')

        # TODO: check only lower triangular entries

        # makes sparse matrix symmetric
        self._symmetric = True

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape

        return CSCSymMatrix((self.data, self.indices,
                             self.indptr), (N, M), copy=copy)

    def toarray(self, order=None, out=None):
        m = self.tofullcoo()
        return m.toarray(order=order, out=out)

    def todense(self, order=None, out=None):
        return np.asmatrix(self.toarray(order=order, out=out))

    def tocsr(self, copy=False):
        M, N = self.shape
        idx_dtype = get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, N))
        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csc_tocsr(M, N,
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        from pyomo.contrib.pynumero.sparse.csr import CSRSymMatrix
        A = CSRSymMatrix((data, indices, indptr), shape=self.shape, copy=False)
        A.has_sorted_indices = True
        return A

    def tocoo(self, copy=True):
        major_dim, minor_dim = self._swap(self.shape)
        minor_indices = self.indices
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)
        _sparsetools.expandptr(major_dim, self.indptr, major_indices)
        row, col = self._swap((major_indices, minor_indices))

        from pyomo.contrib.pynumero.sparse.coo import COOSymMatrix
        return COOSymMatrix((self.data, (row, col)), self.shape, copy=copy,
                            dtype=self.dtype)

    def tofullcoo(self):
        return self.tocoo().tofullcoo()

    def tofullcsr(self):
        return self.tocoo().tofullcsr()

    def tofullcsc(self):
        return self.tocoo().tofullcsc()

    def tofullmatrix(self):
        return self.tofullcsc()

    def todok(self, copy=False):
        raise NotImplementedError('Not supported')

    def todia(self, copy=False):
        raise NotImplementedError('Not supported')

    def tolil(self, copy=False):
        raise NotImplementedError('Not supported')

    def toscipy(self):
        return scipy_csc_matrix(self.tofullmatrix())

    def _add_sparse(self, other):
        if isinstance(other, SparseBase):
            if other.is_symmetric:
                return self.tocsr()._add_sparse(other)
            return self.tofullmatrix()._add_sparse(other)
        if issparse(other):
            raise RuntimeError("Addition not supported with scipy matrices")
        raise RuntimeError("Sparse format not recognized {}".format(type(other)))

    def _sub_sparse(self, other):
        if isinstance(other, SparseBase):
            if other.is_symmetric:
                # ToDo: check if binopt works here directly
                return self.tocsr()._sub_sparse(other)
            return self.tofullmatrix()._sub_sparse(other)
        if issparse(other):
            raise RuntimeError("Subtraction not supported with scipy matrices")
        raise RuntimeError("Sparse format not recognized {}".format(type(other)))

    def _add_dense(self, other):
        return self.tofullcoo()._add_dense(other)

    def _mul_vector(self, other):

        M, N = self.shape

        # output array
        resultl = np.zeros(M, dtype=upcast_char(self.dtype.char,
                                                other.dtype.char))

        resultu = np.zeros(M, dtype=upcast_char(self.dtype.char,
                                                other.dtype.char))

        # csr_matvec or csc_matvec
        fnl = getattr(_sparsetools, self.format + '_matvec')
        fnl(M, N, self.indptr, self.indices, self.data, other, resultl)
        upper = self.transpose()
        csr_matvec_no_diag(M, upper.indptr, upper.indices, upper.data, other, resultu)
        return resultl + resultu

    def _mul_multivector(self, other):
        raise NotImplementedError('Not supported')

    def _mul_sparse_matrix(self, other):

        expanded_sym = self.tofullcsc()
        if isinstance(other, SparseBase):
            if other.is_symmetric:
                expanded_other = other.tofullcsc()
                result = expanded_sym * expanded_other
                return _convert_matrix_to_symmetric(result, check_symmetry=False)
            from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
            if isinstance(other, BlockMatrix):
                raise NotImplementedError("Not supported yet")
                expanded_other = other.tocsc()
                result = expanded_sym * expanded_other
                if expanded_sym.shape[0] == expanded_other.shape[1]:
                    if _is_symmetric_numerically(result):
                        return _convert_matrix_to_symmetric(result, check_symmetry=False)
                return result

            result = expanded_sym * other
            if expanded_sym.shape[0] == other.shape[1]:
                if _is_symmetric_numerically(result):
                    return _convert_matrix_to_symmetric(result, check_symmetry=False)
            return result
        if issparse(other):
            raise RuntimeError("Multiplication not supported with scipy matrices")
        raise RuntimeError("Sparse format not recognized {}".format(type(other)))

    def getcol(self, j):
        return self.tofullmatrix().getcol(j)

    def getrow(self, i):
        return self.tofullmatrix().getrow(i)

    def getallnnz(self):
        # ToDo: add support for this
        raise NotImplementedError("Operation not supported yet")

    def __repr__(self):
        return 'CSCSymMatrix{}'.format(self.shape)



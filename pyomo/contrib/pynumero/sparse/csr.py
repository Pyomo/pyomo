from scipy.sparse import csr_matrix as scipy_csr_matrix
from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse.sputils import (upcast,
                                  upcast_char,
                                  get_index_dtype,
                                  )

from scipy.sparse._sparsetools import csr_tocsc
from scipy.sparse import issparse

try:
    from pyomo.contrib.pynumero.extensions.sparseutils import (sym_csr_matvec,
                                                               csr_matvec_no_diag,
                                                               csc_matvec_no_diag)
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing sparseutils while running coo interface. '
                      'Make sure libpynumero_SPARSE is installed and added to path.')

from scipy.sparse import _sparsetools
from pyomo.contrib.pynumero.sparse.base import SparseBase
from pyomo.contrib.pynumero.sparse.utils import (is_symmetric_dense,
                                                 _convert_matrix_to_symmetric,
                                                 _is_symmetric_numerically)
import numpy as np

__all__ = ['CSRMatrix', 'CSRSymMatrix']


class CSRMatrix(SparseBase, scipy_csr_matrix):

    def __init__(self, arg1, shape=None, dtype=None, copy=False, **kwargs):

        # include upper triangular if arg1 is symmetric
        expand_symmetry = kwargs.pop('expand_symmetry', True)
        if expand_symmetry and isinstance(arg1, SparseBase):
            if arg1.is_symmetric:
                arg1 = arg1.tofullmatrix().tocsr()

        scipy_csr_matrix.__init__(self, arg1, shape=shape, dtype=dtype, copy=copy)
        SparseBase.__init__(self)

    def tosymcsr(self):
        raise NotImplementedError('Not supported')

    def tocsc(self, copy=False):
        idx_dtype = get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, self.shape[0]))
        indptr = np.empty(self.shape[1] + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csr_tocsc(self.shape[0], self.shape[1],
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        from pyomo.contrib.pynumero.sparse.csc import CSCMatrix
        A = CSCMatrix((data, indices, indptr), shape=self.shape)
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

    def tocsr(self, copy=False):
        # copy only there to agree with the signature
        return self

    def tofullmatrix(self):
        return self

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape

        from pyomo.contrib.pynumero.sparse.csc import CSCMatrix
        return CSCMatrix((self.data, self.indices,
                          self.indptr), shape=(N, M), copy=copy)

    def _with_data(self, data, copy=True):
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
                return super(CSRMatrix, self)._add_sparse(other.tofullmatrix())
            return super(CSRMatrix, self)._add_sparse(other)
        if issparse(other):
            raise RuntimeError("Addition not supported with scipy matrices")
        raise RuntimeError("Sparse format not recognized {}".format(type(other)))

    def _sub_sparse(self, other):
        if isinstance(other, SparseBase):
            if other.is_symmetric:
                return super(CSRMatrix, self)._sub_sparse(other.tofullmatrix())
            return super(CSRMatrix, self)._sub_sparse(other)
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
                expanded_other = other.tocsr()
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
        raise RuntimeError("Sparse format not recognized {}".format(type(other)))

    def getcol(self, j):
        from pyomo.contrib.pynumero.sparse.csc import CSCMatrix
        return CSCMatrix(self.toscipy().getcol(j))

    def getrow(self, i):
        return CSRMatrix(super(CSRMatrix, self).getrow(i))

    def toscipy(self):
        return scipy_csr_matrix(self)

    def __repr__(self):
        return 'CSRMatrix{}'.format(self.shape)


# this matrix will only store the lower triangular indices
class CSRSymMatrix(CSRMatrix):

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


        # add check to veryfy square matrix
        if self.shape[0] != self.shape[1]:
            raise RuntimeError('A rectangular matrix is not symmetric')

        # check nnz is less than the full lower triangular
        if self.nnz > self.shape[0]*(self.shape[0] + 1)/2:
            raise RuntimeError('CSRSymMatrix only store lower triangular entries. Too many nnz')

        # TODO: check only lower triangular entries

        # makes sparse matrix symmetric
        self._symmetric = True

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape

        return CSRSymMatrix((self.data, self.indices,
                             self.indptr), shape=(N, M), copy=copy)

    def toarray(self, order=None, out=None):
        m = self.tofullcoo()
        return m.toarray(order=order, out=out)

    def todense(self, order=None, out=None):
        return np.asmatrix(self.toarray(order=order, out=out))

    def tocsc(self, copy=False):
        idx_dtype = get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, self.shape[0]))
        indptr = np.empty(self.shape[1] + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csr_tocsc(self.shape[0], self.shape[1],
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        from pyomo.contrib.pynumero.sparse.csc import CSCSymMatrix
        A = CSCSymMatrix((data, indices, indptr), shape=self.shape)
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
        return self.tofullcsr()

    def todia(self, copy=False):
        raise NotImplementedError('Not supported')

    def tolil(self, copy=False):
        raise NotImplementedError('Not supported')

    def toscipy(self):
        return scipy_csr_matrix(self.tofullmatrix())

    def _add_sparse(self, other):
        if isinstance(other, SparseBase):
            if other.is_symmetric:
                return self._binopt(other, '_plus_')
            return self.tofullmatrix()._add_sparse(other)
        if issparse(other):
            raise RuntimeError("Addition not supported with scipy matrices")
        raise RuntimeError("Sparse format not recognized {}".format(type(other)))

    def _sub_sparse(self, other):
        if isinstance(other, SparseBase):
            if other.is_symmetric:
                return self._binopt(other, '_minus_')
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
        # fnu = getattr(_sparsetools, upper.format + '_matvec')
        # fnu(M, N, upper.indptr, upper.indices, upper.data, other, resultu)
        csc_matvec_no_diag(N, upper.indptr, upper.indices, upper.data, other, resultu)

        # diagonal = self.diagonal()

        # result = np.zeros(M, dtype=upcast_char(self.dtype.char, other.dtype.char))
        # sym_csr_matvec(M, self.indptr, self.indices, self.data, other, result)
        # return result

        return resultl + resultu # - np.multiply(other, diagonal)

    def _mul_multivector(self, other):
        raise NotImplementedError('Not supported')

    def _mul_sparse_matrix(self, other):

        expanded_sym = self.tofullcsr()
        if isinstance(other, SparseBase):
            if other.is_symmetric:
                expanded_other = other.tofullcsr()
                result = expanded_sym * expanded_other
                return _convert_matrix_to_symmetric(result, check_symmetry=False)
            from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
            if isinstance(other, BlockMatrix):
                # this will also need a check for symmetric block matrices
                raise NotImplementedError("Not supported yet")
                expanded_other = other.tocsr()
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
        return 'CSRSymMatrix{}'.format(self.shape)

if __name__ == "__main__":

    row = np.array([0, 3, 1, 0])
    col = np.array([0, 3, 1, 2])
    data = np.array([4, 5, 7, 9])
    m = CSRMatrix((data, (row, col)), shape=(4, 4))
    print(m.toarray())
    print(m.is_symmetric)

    row = np.array([0, 3, 1, 2, 3])
    col = np.array([0, 0, 1, 2, 3])
    data = np.array([2, 1, 3, 4, 5])
    m = CSRSymMatrix((data, (row, col)), shape=(4, 4))
    print(m.toarray())
    print(m.is_symmetric)

    mcsr = m.tofullcsr()
    print(mcsr.toarray())
    print(mcsr.is_symmetric)

    x = np.ones(m.shape[0])
    print(mcsr.dot(x))
    print(m.dot(x))

    row = np.array([0, 1, 4, 1, 2, 7, 2, 3, 5, 3, 4, 5, 4, 7, 5, 6, 6, 7])
    col = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7])
    data = np.array([27, 5, 12, 56, 66, 34, 94, 31, 41, 7, 98, 72, 24, 33, 78, 47, 98, 41])
    big_m = CSRSymMatrix((data, (row, col)), shape=(8, 8))
    print(big_m.toarray())
    print(big_m.is_symmetric)
    x = np.ones(big_m.shape[0])
    print(big_m.tofullcsr().dot(x))
    print(big_m.dot(x))
    print(big_m.toarray())

"""
The pyomo.contrib.pynumero.sparse.coo module includes methods that extend
linear algebra operations in scipy.sparse. In particular pynumero
adds functionality for dealing efficiently with symmetric matrices

All classes in this module subclass from the corresponding scipy.sparse
class. Hence, scipy documentation is the same for the methods here, unless
explicitly stated.

.. rubric:: Contents

"""

from scipy.sparse import coo_matrix as scipy_coo_matrix
from scipy.sparse._sparsetools import coo_tocsr, coo_todense
from scipy.sparse.sputils import (upcast,
                                  isdense,
                                  isscalarlike,
                                  get_index_dtype)

try:
    from pyomo.contrib.pynumero.extensions.sparseutils import sym_coo_matvec
except ImportError as e:
    print('{}'.format(e))
    raise ImportError('Error importing sparseutils while running coo interface. '
                      'Make sure libpynumero_SPARSE is installed and added to path.')
from pyomo.contrib.pynumero.sparse import SparseBase
import numpy as np


__all__ = ['COOMatrix', 'COOSymMatrix', 'EmptyMatrix', 'IdentityMatrix']


class COOMatrix(SparseBase, scipy_coo_matrix):
    """
    A sparse matrix in COOrdinate format.

    Also known as the  ijv or triplet format.

    This can be instantiated in several ways:
    COOMatrix(D)
    with a dense matrix D
    COOMatrix(S)
    with another sparse matrix S (equivalent to S.tocoo())
    COOMatrix((M, N), [dtype])
    to construct an empty matrix with shape (M, N) dtype is optional, defaulting to dtype=d.
    COOMatrix((data, (i, j)), [shape=(M, N)])
    to construct from three arrays:
    data[:] the entries of the matrix, in any order
    i[:] the row indices of the matrix entries
    j[:] the column indices of the matrix entries
    Where A[i[k], j[k]] = data[k]. When shape is not specified, it is inferred from the index arrays
    """
    def __init__(self, arg1, shape=None, dtype=None, copy=False):

        scipy_coo_matrix.__init__(self, arg1, shape=shape, dtype=dtype, copy=copy)
        SparseBase.__init__(self)

    """
    def tosymmatrix(self):
        if not self.is_symmetric:
            if self.shape[0] == self.shape[1]:
                lt = tril(self)
                ut = triu(self)
                if norm((lt-ut.transpose()).toarray()) == 0:
                    return COOSymMatrix(lt)
                raise RuntimeError('Matrix is not symmetric')
            raise RuntimeError('Matrix is not symmetric')
    """

    def tocsc(self, copy=False):
        """
        Convert this matrix to Compressed Sparse Column format

        Returns
        -------
        CSCMatrix

        """
        from .csc import CSCMatrix
        if self.nnz == 0:
            return CSCMatrix(self.shape, dtype=self.dtype)
        else:
            M, N = self.shape
            idx_dtype = get_index_dtype((self.col, self.row),
                                        maxval=max(self.nnz, M))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(N + 1, dtype=idx_dtype)
            indices = np.empty_like(row, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            # TODO: check why scipy does this and not coo_tocsc
            coo_tocsr(N, M, self.nnz, col, row, self.data,
                      indptr, indices, data)

            x = CSCMatrix((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocsr(self, copy=False):
        """
        Convert this matrix to Compressed Sparse Row format

        Returns
        -------
        CSRMatrix

        """
        from pyomo.contrib.pynumero.sparse.csr import CSRMatrix
        if self.nnz == 0:
            return CSRMatrix(self.shape, dtype=self.dtype)
        else:
            M, N = self.shape
            idx_dtype = get_index_dtype((self.row, self.col),
                                        maxval=max(self.nnz, N))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(M + 1, dtype=idx_dtype)
            indices = np.empty_like(col, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            coo_tocsr(M, N, self.nnz, row, col, self.data,
                      indptr, indices, data)

            x = CSRMatrix((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def todok(self, copy=False):
        # ToDo: decide if we should suppert this
        raise NotImplementedError('Not supported')

    def todia(self, copy=False):
        # ToDo: decide if we should suppert this
        raise NotImplementedError('Not supported')

    def tolil(self, copy=False):
        # ToDo: decide if we should suppert this
        raise NotImplementedError('Not supported')

    def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the sparse matrix.

        Parameters
        ----------
        axes: None, optional
        This argument is in the signature solely for NumPy compatibility reasons. Do not pass in
        anything except for the default value.
        copy: bool, optional
        Indicates whether or not attributes of self should be copied whenever possible.
        The degree to which attributes are copied varies depending on the type of sparse matrix being used.

        Returns
        -------
        COOMatrix with dimensions reversed

        """
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape
        return COOMatrix((self.data, (self.col, self.row)),
                         shape=(N, M), copy=copy)

    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the index arrays
        (i.e. .row and .col) are copied.
        """
        if copy:
            return COOMatrix((data, (self.row.copy(), self.col.copy())),
                             shape=self.shape, dtype=data.dtype)
        else:
            return COOMatrix((data, (self.row, self.col)),
                             shape=self.shape, dtype=data.dtype)

    def _mul_vector(self, other):
        from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
        if isinstance(other, BlockVector):
            raise RuntimeError('multiplication of COOMatrix with BlockVector not supported ')
        return super(COOMatrix, self)._mul_vector(other)

    def __repr__(self):
        return 'COOMatrix{}'.format(self.shape)


# this matrix will only store the lower triangular indices
class COOSymMatrix(COOMatrix):
    """
    A symmetric sparse matrix in COOrdinate format.

    Also known as the ijv or triplet format.

    This can be instantiated in several ways:
    COOSymMatrix(D)
    with a dense matrix D
    COOSymMatrix(S)
    with another sparse matrix S (equivalent to S.tocoo())
    COOSymMatrix((M, N), [dtype])
    to construct an empty matrix with shape (M, N) dtype is optional, defaulting to dtype=d.
    COOSymMatrix((data, (i, j)), [shape=(M, N)])
    to construct from three arrays:
    data[:] the entries of the matrix, in any order
    i[:] the row indices of the matrix entries
    j[:] the column indices of the matrix entries
    Where A[i[k], j[k]] = data[k]. When shape is not specified, it is inferred from the index arrays
    """
    def __init__(self, arg1, shape=None, dtype=None, copy=False):

        super(COOSymMatrix, self).__init__(arg1, shape=shape, dtype=dtype, copy=copy)

        # add check to veryfy square matrix
        if self.shape[0] != self.shape[1]:
            raise RuntimeError('A rectangular matrix is not symmetric')

        # check nnz is less than the full lower triangular
        if self.nnz > self.shape[0]*(self.shape[0] + 1)/2:
            raise RuntimeError('COOSymMatrix only store lower triangular entries. Too many nnz')

        # check only lower triangular entries
        diff = self.row - self.col
        if np.any(diff < 0):
            raise RuntimeError('COOSymMatrix only store lower triangular entries. j>i not allowed in irow jcol')

        # makes sparse matrix symmetric
        self._symmetric = True

    def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the sparse matrix.

        Parameters
        ----------
        axes: None, optional
        This argument is in the signature solely for NumPy compatibility reasons. Do not pass in
        anything except for the default value.
        copy: bool, optional
        Indicates whether or not attributes of self should be copied whenever possible.
        The degree to which attributes are copied varies depending on the type of sparse matrix being used.

        Returns
        -------
        COOSymMatrix with dimensions reversed

        """
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape
        return COOSymMatrix((self.data, (self.row, self.col)),
                            shape=(N, M), copy=copy)

    def toarray(self, order=None, out=None):
        """
        Return a dense ndarray representation of this matrix.

        Returns
        -------
        arr : ndarray, 2-dimensional
        An array with the same shape and containing the same data
        represented by the block matrix.

        """
        return self.tofullcoo().toarray(order=order, out=out)

    def todense(self, order=None, out=None):
        """
        Return a dense ndarray representation of this matrix.

        Returns
        -------
        arr : ndarray, 2-dimensional
        An array with the same shape and containing the same data
        represented by the block matrix.

        """
        return np.asmatrix(self.toarray(order=order, out=out))

    def tofullcoo(self):
        """
        Convert this matrix to COOMatrix format.

        Returns
        -------
        COOMatrix

        """
        diff = self.row - self.col
        lower = np.where(diff != 0)

        irow = np.concatenate((self.row, self.col[lower]))
        jcol = np.concatenate((self.col, self.row[lower]))
        values = np.concatenate((self.data, self.data[lower]))

        m = scipy_coo_matrix((values, (irow, jcol)), shape=self.shape)
        return COOMatrix(m)

    def tofullcsc(self):
        """
        Convert this matrix to CSCSymMatrix format. Remains symmetric

        Returns
        -------
        CSCSymMatrix

        """
        return self.tofullcoo().tocsc()

    def tofullcsr(self):
        """
        Convert this matrix to CSRSymMatrix format. Remains symmetric

        Returns
        -------
        CSRSymMatrix

        """
        return self.tofullcoo().tocsr()

    def tofullmatrix(self):
        """
        Convert this matrix to COOMatrix format.

        Returns
        -------
        COOMatrix

        """
        return self.tofullcoo()

    def tocsc(self, copy=False):
        """
        Convert this matrix to CSCSymMatrix format. Remains symmetric

        Returns
        -------
        CSCSymMatrix

        """
        from pyomo.contrib.pynumero.sparse.csc import CSCSymMatrix
        if self.nnz == 0:
            return CSCSymMatrix(self.shape, dtype=self.dtype)
        else:
            M, N = self.shape
            idx_dtype = get_index_dtype((self.col, self.row),
                                        maxval=max(self.nnz, M))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(N + 1, dtype=idx_dtype)
            indices = np.empty_like(row, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            coo_tocsr(N, M, self.nnz, col, row, self.data,
                      indptr, indices, data)

            x = CSCSymMatrix((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocsr(self, copy=False):
        """
        Convert this matrix to CSRSymMatrix format. Remains symmetric

        Returns
        -------
        CSRSymMatrix

        """
        from pyomo.contrib.pynumero.sparse.csr import CSRSymMatrix
        if self.nnz == 0:
            return CSRSymMatrix(self.shape, dtype=self.dtype)
        else:
            M, N = self.shape
            idx_dtype = get_index_dtype((self.row, self.col),
                                        maxval=max(self.nnz, N))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(M + 1, dtype=idx_dtype)
            indices = np.empty_like(col, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            coo_tocsr(M, N, self.nnz, row, col, self.data,
                      indptr, indices, data)

            x = CSRSymMatrix((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def todok(self, copy=False):
        # ToDo: decide if we should suppert this
        raise NotImplementedError('Not supported')

    def todia(self, copy=False):
        # ToDo: decide if we should suppert this
        raise NotImplementedError('Not supported')

    def tolil(self, copy=False):
        # ToDo: decide if we should suppert this
        raise NotImplementedError('Not supported')

    def _add_sparse(self, other):
        raise RuntimeError('COOSymMatrix does not support addition with other sparse matrix yet')

    def _sub_sparse(self, other):
        raise RuntimeError('COOSymMatrix does not support substraction with other sparse matrix yet')

    def _add_dense(self, other):
        return self.tofullcoo()._add_dense(other)

    def _mul_vector(self, other):
        #resultl = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
        #                                                   other.dtype.char))
        #resultu = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
        #                                                    other.dtype.char))
        #coo_matvec(self.nnz, self.row, self.col, self.data, other, resultl)
        #coo_matvec(self.nnz, self.col, self.row, self.data, other, resultu)
        #diagonal = self.diagonal()

        from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
        if isinstance(other, BlockVector):
            #ToDo: need to add support for this
            raise RuntimeError('multiplication of COOMatrix with BlockVector not supported yet')

        sol = np.zeros(self.shape[0], dtype=np.float64)
        sym_coo_matvec(self.row, self.col, self.data, other, sol)
        return sol

    def _mul_multivector(self, other):
        raise NotImplementedError('Not supported')

    def _mul_sparse_matrix(self, other):
        # ToDo: need to add support for this
        raise NotImplementedError('Not supported')

    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the index arrays
        (i.e. .row and .col) are copied.
        """
        if copy:
            return COOSymMatrix((data, (self.row.copy(), self.col.copy())),
                                shape=self.shape, dtype=data.dtype)
        else:
            return COOSymMatrix((data, (self.row, self.col)),
                                shape=self.shape, dtype=data.dtype)

    def getcol(self, j):
        # ToDo: implement this in the sparse library with ctypes
        raise NotImplementedError('Not supported')

    def getrow(self, i):
        # ToDo: implement this in the sparse library with ctypes
        raise NotImplementedError('Not supported')

    def getallnnz(self):
        """
        Return total number of nonzero values in the matrix
        """
        d = self.diagonal()
        nnz_diag = len(d[d!=0])
        lnnz = self.nnz - nnz_diag
        return lnnz*2 + nnz_diag

    def __repr__(self):
        return 'COOSymMatrix{}'.format(self.shape)

    """
    def __add__(self, other):  # self + other

        if isinstance(other, SparseBase):
            raise RuntimeError('not implemented yet')
        super().__add__(other)

    def __radd__(self, other):  # other + self
        return self.__add__(other)
        
    """


# this mimics an empty matrix
class EmptyMatrix(COOMatrix):

    def __init__(self, nrows, ncols):

        data = np.zeros(0)
        irows = np.zeros(0)
        jcols = np.zeros(0)
        arg1 = (data, (irows, jcols))
        super(EmptyMatrix, self).__init__(arg1, shape=(nrows, ncols), dtype=np.double, copy=False)

        # makes sparse matrix symmetric
        if nrows == ncols:
            self._symmetric = True

    def getallnnz(self):
        """
        Return total number of nonzero values in the matrix
        """
        return 0

    def tofullcoo(self):
        return self

    def tofullmatrix(self):
        return self

    def tocsr(self, copy=False):
        return self.tocsr(copy=copy)

    def tocsc(self, copy=False):
        return self.tocsc(copy=copy)


class IdentityMatrix(COOSymMatrix):

    def __init__(self, nrowcols):
        data = np.ones(nrowcols)
        irows = np.arange(0, nrowcols)
        jcols = np.arange(0, nrowcols)
        arg1 = (data, (irows, jcols))
        super(IdentityMatrix, self).__init__(arg1, shape=(nrowcols, nrowcols), dtype=np.double, copy=False)


if __name__ == "__main__":

    row = np.array([0, 3, 1, 0])
    col = np.array([0, 3, 1, 2])
    data = np.array([4, 5, 7, 9])
    m = COOMatrix((data, (row, col)), shape=(4, 4))
    print(m.toarray())
    print(m.is_symmetric)
    d = m.diagonal()
    print(d[d!=0])

    print(m.__repr__())

    row = np.array([0, 3, 1, 2, 3])
    col = np.array([0, 0, 1, 2, 3])
    data = np.array([2, 1, 3, 4, 5])
    m = COOSymMatrix((data, (row, col)), shape=(4, 4))
    print(m.toarray())
    print(m.is_symmetric)
    print(m.__repr__())
    mcoo = m.tofullcoo()
    print(mcoo.toarray())
    print(mcoo.is_symmetric)

    x = np.ones(m.shape[0])
    print(mcoo.dot(x))
    print(m.dot(x))


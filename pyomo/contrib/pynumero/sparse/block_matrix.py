#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
The pyomo.contrib.pynumero.sparse.block_matrix module includes methods that extend
linear algebra operations in scipy for case of structured problems
where linear algebra operations present an inherent block structure.
This interface consider matrices of the form:

m = [[m11, m12],[m21, m22], ..]

where m_{i,j} are sparse matrices

.. rubric:: Contents

"""

from scipy.sparse.sputils import upcast, isscalarlike, get_index_dtype
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import isspmatrix
from pyomo.contrib.pynumero.sparse.utils import is_symmetric_sparse
from pyomo.contrib.pynumero.sparse import empty_matrix
from .base_block import BaseBlockMatrix
from scipy.sparse.base import spmatrix
import numpy as np
import six
import abc

__all__ = ['BlockMatrix', 'BlockSymMatrix']


def assert_block_structure(mat):
    msgr = 'Operation not allowed with None rows. ' \
           'Specify at least one block in every row'
    assert not mat.has_empty_rows(), msgr
    msgc = 'Operation not allowed with None columns. ' \
           'Specify at least one block every column'
    assert not mat.has_empty_cols(), msgc


class BlockMatrix(BaseBlockMatrix):
    """
    Structured Matrix interface

    Parameters
    -------------------
    nbrows: number of block-rows in the matrix
    nbcols: number of block-columns in the matrix

    """
    format = 'block_matrix'

    def __init__(self, nbrows, nbcols):

        shape = (nbrows, nbcols)

        blocks = []
        for i in range(shape[0]):
            blocks.append([None for j in range(shape[1])])

        self._blocks = np.asarray(blocks, dtype='object')

        self._name = None
        self._bshape = shape

        self._block_mask = np.zeros(shape, dtype=bool)
        self._brow_lengths = np.zeros(nbrows, dtype=np.int64)
        self._bcol_lengths = np.zeros(nbcols, dtype=np.int64)

        #super(BlockMatrix, self).__init__()

    @property
    def bshape(self):
        """
        Returns the block-shape of the matrix
        """
        return self._bshape

    @property
    def shape(self):
        """
        Returns tuple with total number of rows and columns
        """
        return np.sum(self._brow_lengths), np.sum(self._bcol_lengths)

    @property
    def nnz(self):
        """
        Returns total number of nonzero values in the matrix
        """
        return sum(blk.nnz for blk in self._blocks[self._block_mask])

    @property
    def dtype(self):
        """
        Returns data type of the matrix.
        """
        # ToDo: decide if this is the right way of doing this
        all_dtypes = [blk.dtype for blk in self._blocks[self._block_mask]]
        dtype = upcast(*all_dtypes) if all_dtypes else None
        return dtype

    @property
    def T(self):
        """
        Transpose matrix
        """
        return self.transpose()

    def row_block_sizes(self, copy=True):
        """
        Returns row-block sizes

        Returns
        -------
        ndarray

        """
        if copy:
            return np.copy(self._brow_lengths)
        return self._brow_lengths

    def col_block_sizes(self, copy=True):
        """
        Returns col-block sizes

        Returns
        -------
        ndarray

        """
        if copy:
            return np.copy(self._bcol_lengths)
        self._bcol_lengths

    def block_shapes(self):
        """
        Returns shapes of blocks in BlockMatrix

        Returns
        -------
        list
        """
        bm, bn = self.bshape
        sizes = [list() for i in range(bm)]
        for i in range(bm):
            sizes[i] = list()
            for j in range(bn):
                shape = self._brow_lengths[i], self._bcol_lengths[j]
                sizes[i].append(shape)
        return sizes

    def dot(self, other):
        """
        Ordinary dot product
        """
        return self * other

    def reset_brow(self, idx):
        """
        Resets all blocks in selected row to None

        Parameters
        ----------
        idx: integer
            row index to be reset

        Returns
        -------
        None

        """
        assert 0 <= idx < self.bshape[0], 'Index out of bounds'
        self._brow_lengths[idx] = 0
        self._block_mask[idx, :] = False
        self._blocks[idx, :] = None

    def reset_bcol(self, jdx):
        """
        Resets all blocks in selected column to None

        Parameters
        ----------
        idx: integer
            column index to be reset

        Returns
        -------
        None

        """
        assert 0 <= jdx < self.bshape[1], 'Index out of bounds'
        self._bcol_lengths[jdx] = 0
        self._block_mask[:, jdx] = False
        self._blocks[:, jdx] = None

    def coo_data(self):
        """
        Returns data values of matrix in coo format

        Returns
        -------
        ndarray with values of all entries in the matrix

        """
        assert_block_structure(self)

        nonzeros = self.nnz
        data = np.empty(nonzeros, dtype=self.dtype)

        nnz = 0
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            B = self._blocks[i, j].tocoo()
            idx = slice(nnz, nnz + B.nnz)
            data[idx] = B.data
            nnz += B.nnz

        return data

    def tocoo(self):
        """
        Converts this matrix to coo_matrix format.

        Returns
        -------
        coo_matrix

        """
        # ToDo: copy argument to match scipy?
        assert_block_structure(self)

        dtype = self.dtype

        row_offsets = np.append(0, np.cumsum(self._brow_lengths))
        col_offsets = np.append(0, np.cumsum(self._bcol_lengths))

        shape = (row_offsets[-1], col_offsets[-1])

        nonzeros = self.nnz

        data = np.empty(nonzeros, dtype=dtype)
        idx_dtype = get_index_dtype(maxval=max(shape))
        row = -np.ones(nonzeros, dtype=idx_dtype)
        col = -np.ones(nonzeros, dtype=idx_dtype)

        nnz = 0
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            B = self[i, j].tocoo()
            idx = slice(nnz, nnz + B.nnz)
            data[idx] = B.data
            row[idx] = B.row + row_offsets[i]
            col[idx] = B.col + col_offsets[j]
            nnz += B.nnz

        return coo_matrix((data, (row, col)), shape=shape)

    def tocsr(self):
        """
        Converts this matrix to csr format.

        Returns
        -------
        csr_matrix

        """
        return self.tocoo().tocsr()

    def tocsc(self):
        """
        Converts this matrix to csc format.

        Returns
        -------
        csc_matrix

        """
        return self.tocoo().tocsc()

    def tolil(self, copy=False):
        BaseBlockMatrix.tolil(self, copy=copy)

    def todia(self, copy=False):
        BaseBlockMatrix.todia(self, copy=copy)

    def tobsr(self, blocksize=None, copy=False):
        BaseBlockMatrix.tobsr(self, blocksize=blocksize, copy=copy)

    def toarray(self):
        """
        Returns a dense ndarray representation of this matrix.

        Returns
        -------
        arr : ndarray, 2-dimensional
            An array with the same shape and containing the same data
            represented by the block matrix.

        """
        return self.tocoo().toarray()

    def _mul_sparse_matrix(self, other):

        if isinstance(other, BlockMatrix):
            assert other.bshape[0] == self.bshape[1], "Dimension mismatch"
            result = BlockMatrix(self.bshape[0], self.bshape[1])
            m, n = self.bshape
            for i in range(m):
                for j in range(n):
                    accum = empty_matrix(self._brow_lengths[i],
                                         self._bcol_lengths[j])
                    for k in range(n):
                        if self._blocks[i,k] is not None and other[k, j] is not None:
                            accum = accum + self._blocks[i,k] * other[k, j]
                    result[i, j] = accum
            return result
        elif isspmatrix(other):
            assert other.shape[0] == self.shape[1], "Dimension mismatch"
            result = BlockMatrix(self.bshape[0], self.bshape[1])
            mat = self.copy_structure()
            mat.copyfrom(other)
            m, n = self.bshape
            for i in range(m):
                for j in range(n):
                    accum = empty_matrix(self._brow_lengths[i],
                                         self._bcol_lengths[j])
                    for k in range(n):
                        if self._blocks[i,k] is not None and mat[k, j] is not None:
                            accum = accum + self._blocks[i,k] * mat[k, j]
                    result[i, j] = accum
            return result
        else:
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the block matrix.

        Parameters
        ----------
        axes: None, optional
            This argument is in the signature solely for NumPy compatibility reasons. Do not pass in
            anything except for the default value.
        copy: bool, optional
            Indicates whether or not attributes of self should be copied whenever possible.

        Returns
        -------
        BlockMatrix with dimensions reversed
        """
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        m = self.bshape[0]
        n = self.bshape[1]
        mat = BlockMatrix(n, m)
        for i in range(m):
            for j in range(n):
                if not self.is_empty_block(i, j):
                    mat[j, i] = self[i, j].transpose(copy=copy)
                else:
                    mat[j, i] = None
        return mat

    def is_empty_block(self, idx, jdx):
        """
        Indicates if a block is None

        Parameters
        ----------
        idx: int
            block-row index
        jdx: int
            block-column index

        Returns
        -------
        boolean

        """
        return not self._block_mask[idx, jdx]

    def has_empty_rows(self):
        """
        Indicates if the matrix has block-rows that are empty

        Returns
        -------
        boolean

        """
        bm, bn = self.bshape

        empty_rows = []
        for idx in range(bm):
            row_bool = np.logical_not(self._block_mask[idx, :])
            if np.all(row_bool):
                empty_rows.append(idx)

        return len(empty_rows) > 0

    def has_empty_cols(self):
        """
        Indicates if the matrix has block-columns that are empty

        Returns
        -------
        boolean

        """
        bm, bn = self.bshape

        empty_cols = []
        for jdx in range(bn):
            col_bool = np.logical_not(self._block_mask[:, jdx])
            if np.all(col_bool):
                empty_cols.append(jdx)

        return len(empty_cols) > 0

    def copyfrom(self, other):
        """
        Copies entries of other matrix into this matrix

        Parameters
        ----------
        other: BlockMatrix or sparse_matrix

        Returns
        -------
        None
        """
        assert_block_structure(self)
        m, n = self.bshape
        assert other.shape == self.shape, \
            'dimensions mismatch {} != {}'.format(self.shape, other.shape)
        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            for i in range(m):
                for j in range(n):
                    self[i, j] = other[i, j]

        elif isspmatrix(other) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                m = coo_matrix(other)
            else:
                m = other.tocoo()
            row_offsets = np.append(0, np.cumsum(self._brow_lengths))
            col_offsets = np.append(0, np.cumsum(self._bcol_lengths))

            for i in range(self.bshape[0]):
                for j in range(self.bshape[1]):
                    if i < self.bshape[0] - 1 and j < self.bshape[1] - 1:
                        row_indices1 = row_offsets[i] <= m.row
                        row_indices2 = m.row < row_offsets[i + 1]
                        row_indices = np.multiply(row_indices1, row_indices2)
                        col_indices1 = col_offsets[j] <= m.col
                        col_indices2 = m.col < col_offsets[j + 1]
                        col_indices = np.multiply(col_indices1, col_indices2)
                        bool_entries = np.multiply(row_indices, col_indices)

                    elif i < self.bshape[0] - 1 and j == self.bshape[1] - 1:

                        row_indices1 = row_offsets[i] <= m.row
                        row_indices2 = m.row < row_offsets[i + 1]
                        row_indices = np.multiply(row_indices1, row_indices2)
                        col_indices1 = col_offsets[j] <= m.col
                        col_indices2 = m.col < self.shape[1]
                        col_indices = np.multiply(col_indices1, col_indices2)
                        bool_entries = np.multiply(row_indices, col_indices)
                    elif i == self.bshape[0] - 1 and j < self.bshape[1] - 1:

                        row_indices1 = row_offsets[i] <= m.row
                        row_indices2 = m.row < self.shape[0]
                        row_indices = np.multiply(row_indices1, row_indices2)
                        col_indices1 = col_offsets[j] <= m.col
                        col_indices2 = m.col < col_offsets[j + 1]
                        col_indices = np.multiply(col_indices1, col_indices2)
                        bool_entries = np.multiply(row_indices, col_indices)
                    else:

                        row_indices1 = row_offsets[i] <= m.row
                        row_indices2 = m.row < self.shape[0]
                        row_indices = np.multiply(row_indices1, row_indices2)
                        col_indices1 = col_offsets[j] <= m.col
                        col_indices2 = m.col < self.shape[1]
                        col_indices = np.multiply(col_indices1, col_indices2)
                        bool_entries = np.multiply(row_indices, col_indices)

                    sub_row = np.compress(bool_entries, m.row)
                    sub_col = np.compress(bool_entries, m.col)
                    sub_data = np.compress(bool_entries, m.data)
                    sub_row -= row_offsets[i]
                    sub_col -= col_offsets[j]

                    shape = (self._brow_lengths[i], self._bcol_lengths[j])
                    mm = csr_matrix((sub_data, (sub_row, sub_col)), shape=shape)

                    if self.is_empty_block(i, j) and mm.nnz == 0:
                        self[i, j] = None
                    else:
                        self[i, j] = mm

        else:
            raise NotImplementedError()

    def copy(self):
        result = BlockMatrix(self.bshape[0], self.bshape[1])
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            result[i, j] = self._blocks[i, j].copy()
        return result

    def copy_structure(self):
        result = BlockMatrix(self.bshape[0], self.bshape[1])
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            if isinstance(self._blocks[i, j], BlockMatrix):
                result[i, j] = self._blocks[i, j].copy_structure()
            else:
                nrows, ncols = self._blocks[i, j].shape
                result[i, j] = empty_matrix(nrows, ncols)
        return result

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.bshape)

    def __str__(self):
        msg = '{}{}\n'.format(self.__class__.__name__, self.bshape)
        for idx in range(self.bshape[0]):
            for jdx in range(self.bshape[1]):
                repn = self._blocks[idx, jdx].__repr__() if self._block_mask[idx, jdx] else None
                msg += '({}, {}): {}\n'.format(idx, jdx, repn)
        return msg

    def __getitem__(self, item):

        assert isinstance(item, tuple), 'Indices must be tuples (i,j)'

        idx, jdx = item
        assert idx >= 0 and jdx >= 0, 'indices must be positive'
        assert idx < self.bshape[0] and \
               jdx < self.bshape[1], 'Indices out of range'
        return self._blocks[item]

    def __setitem__(self, key, value):

        assert not isinstance(key, slice), 'Slices not supported in BlockMatrix'
        assert isinstance(key, tuple), 'Indices must be tuples (i,j)'

        idx, jdx = key
        assert idx >= 0 and \
               jdx >= 0, 'Indices must be positive'

        assert idx < self.bshape[0] and \
               jdx < self.bshape[1], 'Indices out of range'

        if value is None:
            self._blocks[idx, jdx] = None
            self._block_mask[idx, jdx] = False
            all_none_rows = True
            for blk in self._blocks[:, jdx]:
                if blk is not None:
                    all_none_rows = False
                    break

            all_none_cols = True
            for blk in self._blocks[idx, :]:
                if blk is not None:
                    all_none_cols = False
                    break

            if all_none_cols:
                self._brow_lengths[idx] = 0
            if all_none_rows:
                self._bcol_lengths[jdx] = 0
        else:
            assert isinstance(value, BaseBlockMatrix) or isspmatrix(value), \
                'blocks need to be sparse matrices or BlockMatrices'
            if self._brow_lengths[idx] == 0 and self._bcol_lengths[jdx] == 0:
                self._blocks[idx, jdx] = value
                self._brow_lengths[idx] = value.shape[0]
                self._bcol_lengths[jdx] = value.shape[1]
                self._block_mask[idx, jdx] = True
            elif self._brow_lengths[idx] != 0 and self._bcol_lengths[jdx] == 0:
                assert self._brow_lengths[idx] == value.shape[0],\
                    'Incompatible row dimensions for block ({i},{j}) ' \
                    'got {got}, expected {exp}.'.format(i=idx,
                                                        j=jdx,
                                                        exp=self._brow_lengths[idx],
                                                        got=value.shape[0])

                self._blocks[idx, jdx] = value
                self._block_mask[idx, jdx] = True
                self._bcol_lengths[jdx] = value.shape[1]
            elif self._brow_lengths[idx] == 0 and self._bcol_lengths[jdx] != 0:
                assert self._bcol_lengths[jdx] == value.shape[1], \
                    'Incompatible col dimensions for block ({i},{j}) ' \
                    'got {got}, expected {exp}.'.format(i=idx,
                                                        j=jdx,
                                                        exp=self._bcol_lengths[jdx],
                                                        got=value.shape[1])

                self._blocks[idx, jdx] = value
                self._block_mask[idx, jdx] = True
                self._brow_lengths[idx] = value.shape[0]
            else:
                assert self._brow_lengths[idx] == value.shape[0], \
                    'Incompatible row dimensions for block ({i},{j}) ' \
                    'got {got}, expected {exp}.'.format(i=idx,
                                                        j=jdx,
                                                        exp=self._brow_lengths[idx],
                                                        got=value.shape[0])

                assert self._bcol_lengths[jdx] == value.shape[1], \
                    'Incompatible col dimensions for block ({i},{j}) ' \
                    'got {got}, expected {exp}.'.format(i=idx,
                                                        j=jdx,
                                                        exp=self._bcol_lengths[jdx],
                                                        got=value.shape[1])

                self._blocks[idx, jdx] = value
                self._block_mask[idx, jdx] = True

    def __add__(self, other):

        assert_block_structure(self)

        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape

        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            assert other.shape == self.shape, \
                'dimensions mismatch {} != {}'.format(self.shape, other.shape)
            assert_block_structure(other)

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j] + other[i, j]
                    elif not self.is_empty_block(i, j) and other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j]
                    elif self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = other[i, j]
                    else:
                        result[i, j] = None
            return result
        elif isspmatrix(other):
            # Note: this is not efficient but is just for flexibility.
            mat = self.copy_structure()
            mat.copyfrom(other)
            return self.__add__(mat)
        else:
            if other.__class__.__name__ == 'MPIBlockMatrix':
                raise RuntimeError('Operation not supported by BlockMatrix')

            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        assert_block_structure(self)
        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape

        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            assert other.shape == self.shape, \
                'dimensions mismatch {} != {}'.format(self.shape, other.shape)
            assert_block_structure(other)
            for i in range(m):
                for j in range(n):
                    if self._block_mask[i, j] and other._block_mask[i, j]:
                        result[i, j] = self._blocks[i, j] - other[i, j]
                    elif self._block_mask[i, j] and not other._block_mask[i, j]:
                        result[i, j] = self._blocks[i, j]
                    elif not self._block_mask[i, j] and other._block_mask[i, j]:
                        result[i, j] = -other[i, j]
                    else:
                        result[i, j] = None
            return result
        elif isspmatrix(other):
            # Note: this is not efficient but is just for flexibility.
            mat = self.copy_structure()
            mat.copyfrom(other)
            return self.__sub__(mat)
        else:
            if other.__class__.__name__ == 'MPIBlockMatrix':
                raise RuntimeError('Operation not supported by BlockMatrix')
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __rsub__(self, other):
        assert_block_structure(self)
        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape
        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            assert other.shape == self.shape, \
                'dimensions mismatch {} != {}'.format(self.shape, other.shape)
            assert_block_structure(other)
            for i in range(m):
                for j in range(n):
                    if self._block_mask[i, j] and other._block_mask[i, j]:
                        result[i, j] = other[i, j] - self._blocks[i, j]
                    elif self._block_mask[i, j] and not other._block_mask[i, j]:
                        result[i, j] = -self._blocks[i, j]
                    elif not self._block_mask[i, j] and other._block_mask[i, j]:
                        result[i, j] = other[i, j]
                    else:
                        result[i, j] = None
            return result
        elif isspmatrix(other):
            # Note: this is not efficient but is just for flexibility.
            mat = self.copy_structure()
            mat.copyfrom(other)
            return self.__rsub__(mat)
        else:
            from .mpi_block_matrix import MPIBlockMatrix
            if isinstance(other, MPIBlockMatrix):
                other._assert_broadcasted_sizes()

                assert other.bshape == self.bshape, \
                    'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

                result = other.copy_structure()

                ii, jj = np.nonzero(other._owned_mask)
                for i, j in zip(ii, jj):
                    mat1 = self[i, j]
                    mat2 = other[i, j]
                    if mat1 is not None and mat2 is not None:
                        result[i, j] = mat2 - mat1
                    elif mat1 is not None and mat2 is None:
                        result[i, j] = -mat1
                    elif mat1 is None and mat2 is not None:
                        result[i, j] = mat2
                    else:
                        result[i, j] = None
                return result
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __mul__(self, other):

        bm, bn = self.bshape
        if np.isscalar(other):
            result = BlockMatrix(bm, bn)
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                    scaled = self._blocks[i, j] * other
                    result[i, j] = scaled
            return result
        elif isinstance(other, BlockVector):
            assert bn == other.bshape[0], 'Dimension mismatch'
            assert self.shape[1] == other.shape[0], 'Dimension mismatch'
            assert not other.has_none, 'Block vector must not have none entries'
            assert_block_structure(self)

            nblocks = self.bshape[0]
            result = BlockVector(nblocks)
            for i in range(bm):
                result[i] = np.zeros(self._brow_lengths[i])
                for j in range(bn):
                    x = other[j]  # this flattens block vectors that are within block vectors
                    if not self.is_empty_block(i, j):
                        A = self._blocks[i, j]
                        result[i] += A * x
            return result
        elif isinstance(other, np.ndarray):

            if other.ndim == 2:
                raise NotImplementedError('Operation not supported by BlockMatrix')

            assert self.shape[1] == other.shape[0], \
                'Dimension mismatch {}!={}'.format(self.shape[1],
                                                   other.shape[0])
            assert_block_structure(self)

            nblocks = self.bshape[0]
            result = BlockVector(nblocks)
            for i in range(bm):
                result[i] = np.zeros(self._brow_lengths[i])
                counter = 0
                for j in range(bn):
                    if not self.is_empty_block(i, j):
                        A = self._blocks[i, j]
                        x = other[counter: counter + A.shape[1]]
                        result[i] += A * x
                        counter += A.shape[0]
            return result
        elif isinstance(other, BlockMatrix) or isspmatrix(other):
            assert_block_structure(self)
            return self._mul_sparse_matrix(other)
        else:
            raise NotImplementedError('input not recognized for multiplication')

    def __truediv__(self, other):
        bm, bn = self.bshape
        if np.isscalar(other):
            result = BlockMatrix(bm, bn)
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                    scaled = self._blocks[i, j] / other
                    result[i, j] = scaled
            return result
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __rtruediv__(self, other):
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __rmul__(self, other):
        assert_block_structure(self)
        bm, bn = self.bshape
        if np.isscalar(other):
            result = BlockMatrix(bm, bn)
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                    scaled = self._blocks[i, j] * other
                    result[i, j] = scaled
            return result
        elif isinstance(other, BlockMatrix):
            assert_block_structure(self)
            return other._mul_sparse_matrix(self)
        elif isspmatrix(other):
            mat = self.copy_structure()
            mat.copyfrom(other)
            return mat._mul_sparse_matrix(self)
        else:
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __pow__(self, other):
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __abs__(self):
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            self._blocks[i, j] = self._blocks[i, j].__abs__()
        return self

    def __iadd__(self, other):

        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            assert other.shape == self.shape, \
                'dimensions mismatch {} != {}'.format(self.shape, other.shape)
            assert_block_structure(other)

            m, n = self.bshape
            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        self[i, j] = self._blocks[i, j] + other[i, j]
                    elif not self.is_empty_block(i, j) and other.is_empty_block(i, j):
                        self[i, j] = self._blocks[i, j]
                    elif self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        self[i, j] = other[i, j]
                    else:
                        self[i, j] = None

            return self
        elif isspmatrix(other):
            # Note: this is not efficient but is just for flexibility.
            mat = self.copy_structure()
            mat.copyfrom(other)
            return self.__iadd__(mat)
        else:
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __isub__(self, other):

        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            assert other.shape == self.shape, \
                'dimensions mismatch {} != {}'.format(self.shape, other.shape)
            assert_block_structure(other)

            m, n = self.bshape
            for i in range(m):
                for j in range(n):
                    if self._block_mask[i, j] and other._block_mask[i, j]:
                        self[i, j] = self._blocks[i, j] - other[i, j]
                    elif self._block_mask[i, j] and not other._block_mask[i, j]:
                        self[i, j] = self._blocks[i, j]
                    elif not self._block_mask[i, j] and other._block_mask[i, j]:
                        self[i, j] = -other[i, j]
                    else:
                        self[i, j] = None
            return self
        elif isspmatrix(other):
            # Note: this is not efficient but is just for flexibility.
            mat = self.copy_structure()
            mat.copyfrom(other)
            return self.__isub__(mat)
        else:
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __imul__(self, other):
        if np.isscalar(other):
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                self._blocks[i, j] = self._blocks[i, j] * other
            return self
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __itruediv__(self, other):
        if np.isscalar(other):
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                self._blocks[i, j] = self._blocks[i, j] / other
            return self
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __ifloordiv__(self, other):
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __neg__(self):
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            self._blocks[i, j] = -self._blocks[i, j]
        return self

    def __eq__(self, other):

        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape

        if isinstance(other, BlockMatrix) and other.bshape == self.bshape:

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__eq__(other[i, j])
                    elif not self.is_empty_block(i, j) and other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__eq__(0.0)
                    elif self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = other[i, j].__eq__(0.0)
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        mat = empty_matrix(nrows, ncols)
                        result[i, j] = mat.__eq__(0.0)
            return result
        elif isinstance(other, BlockMatrix) or isspmatrix(other):

            if isinstance(other, BlockMatrix):
                raise NotImplementedError('Operation supported with same block structure only')
            else:
                raise NotImplementedError('Operation not supported by BlockMatrix')

        elif np.isscalar(other):
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                result[i, j] = self._blocks[i, j].__eq__(other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockMatrix':
                raise RuntimeError('Operation not supported by BlockMatrix')
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __ne__(self, other):

        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape
        if isinstance(other, BlockMatrix) and other.bshape == self.bshape:

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__ne__(other[i, j])
                    elif not self.is_empty_block(i, j) and other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__ne__(0.0)
                    elif self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = other[i, j].__ne__(0.0)
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        mat = empty_matrix(nrows, ncols)
                        result[i, j] = mat.__ne__(0.0)
            return result
        elif isinstance(other, BlockMatrix) or isspmatrix(other):

            if isinstance(other, BlockMatrix):
                raise NotImplementedError('Operation supported with same block structure only')
            else:
                raise NotImplementedError('Operation not supported by BlockMatrix')

        elif np.isscalar(other):

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__ne__(other)
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        matc = empty_matrix(nrows, ncols)
                        result[i, j] = matc.__ne__(other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockMatrix':
                raise RuntimeError('Operation not supported by BlockMatrix')
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __le__(self, other):

        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape
        if isinstance(other, BlockMatrix) and other.bshape == self.bshape:

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__le__(other[i, j])
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        mat = empty_matrix(nrows, ncols)
                        if not self.is_empty_block(i, j) and other.is_empty_block(i, j):
                            result[i, j] = self._blocks[i, j].__le__(mat)
                        elif self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                            result[i, j] = other[i, j].__le__(mat)
                        else:
                            result[i, j] = mat.__le__(mat)
            return result
        elif isinstance(other, BlockMatrix) or isspmatrix(other):

            if isinstance(other, BlockMatrix):
                raise NotImplementedError('Operation supported with same block structure only')
            else:
                raise NotImplementedError('Operation not supported by BlockMatrix')

        elif np.isscalar(other):

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__le__(other)
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        matc = empty_matrix(nrows, ncols)
                        result[i, j] = matc.__le__(other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockMatrix':
                raise RuntimeError('Operation not supported by BlockMatrix')
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __lt__(self, other):

        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape
        if isinstance(other, BlockMatrix) and other.bshape == self.bshape:

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__lt__(other[i, j])
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        mat = empty_matrix(nrows, ncols)
                        if not self.is_empty_block(i, j) and other.is_empty_block(i, j):
                            result[i, j] = self._blocks[i, j].__lt__(mat)
                        elif self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                            result[i, j] = other[i, j].__lt__(mat)
                        else:
                            result[i, j] = mat.__lt__(mat)
            return result
        elif isinstance(other, BlockMatrix) or isspmatrix(other):

            if isinstance(other, BlockMatrix):
                raise NotImplementedError('Operation supported with same block structure only')
            else:
                raise NotImplementedError('Operation not supported by BlockMatrix')

        elif np.isscalar(other):

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__lt__(other)
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[i]
                        matc = empty_matrix(nrows, ncols)
                        result[i, j] = matc.__lt__(other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockMatrix':
                raise RuntimeError('Operation not supported by BlockMatrix')
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __ge__(self, other):

        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape
        if isinstance(other, BlockMatrix) and other.bshape == self.bshape:

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__ge__(other[i, j])
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        mat = empty_matrix(nrows, ncols)
                        if not self.is_empty_block(i, j) and other.is_empty_block(i, j):
                            result[i, j] = self._blocks[i, j].__ge__(mat)
                        elif self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                            result[i, j] = other[i, j].__ge__(mat)
                        else:
                            result[i, j] = mat.__ge__(mat)
            return result
        elif isinstance(other, BlockMatrix) or isspmatrix(other):

            if isinstance(other, BlockMatrix):
                raise NotImplementedError('Operation supported with same block structure only')
            else:
                raise NotImplementedError('Operation not supported by BlockMatrix')

        elif np.isscalar(other):

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__ge__(other)
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[i]
                        matc = empty_matrix(nrows, ncols)
                        result[i, j] = matc.__ge__(other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockMatrix':
                raise RuntimeError('Operation not supported by BlockMatrix')
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __gt__(self, other):

        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape
        if isinstance(other, BlockMatrix) and other.bshape == self.bshape:

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__gt__(other[i, j])
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        mat = empty_matrix(nrows, ncols)
                        if not self.is_empty_block(i, j) and other.is_empty_block(i, j):
                            result[i, j] = self._blocks[i, j].__gt__(mat)
                        elif self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                            result[i, j] = other[i, j].__gt__(mat)
                        else:
                            result[i, j] = mat.__gt__(mat)
            return result
        elif isinstance(other, BlockMatrix) or isspmatrix(other):

            if isinstance(other, BlockMatrix):
                raise NotImplementedError('Operation supported with same block structure only')
            else:
                raise NotImplementedError('Operation not supported by BlockMatrix')

        elif np.isscalar(other):

            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j):
                        result[i, j] = self._blocks[i, j].__ge__(other)
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[i]
                        matc = empty_matrix(nrows, ncols)
                        result[i, j] = matc.__gt__(other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockMatrix':
                raise RuntimeError('Operation not supported by BlockMatrix')
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __len__(self):
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __matmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                            "use '*' instead")
        return self.__mul__(other)

    def __rmatmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        return self.__rmul__(other)

    def sum(self, axis=None, dtype=None, out=None):
        BaseBlockMatrix.sum(self, axis=axis, dtype=dtype, out=out)

    def mean(self, axis=None, dtype=None, out=None):
        BaseBlockMatrix.mean(self, axis=axis, dtype=dtype, out=out)

    def diagonal(self, k=0):
        BaseBlockMatrix.diagonal(self, k=k)

    def nonzero(self):
        BaseBlockMatrix.nonzero(self)

    def setdiag(self, values, k=0):
        BaseBlockMatrix.setdiag(self, value, k=k)

    def get_block_column(index):

        msgc = 'Operation not allowed with None columns. ' \
               'Specify at least one block every column'
        assert not mat.has_empty_cols(), msgc

        bm, bn = self.bshape
        # get cummulative sum of block sizes
        cum = self._bcol_lengths.cumsum()
        assert index >=0, 'index out of bounds'
        assert index < cum[bn-1], 'index out of bounds'

        # exits if only has one column
        if bn <= 1:
            return 0

        ge = cum >= index
        # find first entry that is greater or equal
        block_index = np.argmax(ge)

        if cum[block_index] == index:
            return block_index + 1
        return block_index

    def get_block_row(index):

        msgr = 'Operation not allowed with None rows. ' \
               'Specify at least one block in every row'
        assert not mat.has_empty_rows(), msgr

        bm, bn = self.bshape
        # get cummulative sum of block sizes
        cum = self._brow_lengths.cumsum()
        assert index >=0, 'index out of bounds'
        assert index < cum[bm-1], 'index out of bounds'

        # exits if only has one column
        if bm <= 1:
            return 0

        ge = cum >= index
        # find first entry that is greater or equal
        block_index = np.argmax(ge)

        if cum[block_index] == index:
            return block_index + 1
        return block_index

    def getcol(self, j):
        """Returns a copy of column j of the matrix, as an (m x 1) sparse
        matrix (column vector).
        """
        # ToDo: this can be done more efficiently by copying structure
        # and populating only the blocks that have col j
        return self.tocsr().getcol(j)

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n) sparse
        matrix (row vector).
        """
        # ToDo: this can be done more efficiently by copying structure
        # and populating only the blocks that have row i
        return self.tocsr().getrow(i)

class BlockSymMatrix(BlockMatrix):

    def __init__(self, nrowcols):

        super(BlockSymMatrix, self).__init__(nrowcols, nrowcols)

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.bshape)

    def __str__(self):
        msg = '{}{}\n'.format(self.__class__.__name__, self.bshape)
        for idx in range(self.bshape[0]):
            for jdx in range(self.bshape[1]):
                if idx >= jdx:
                    if self._blocks[idx, jdx] is not None:
                        repn = self._blocks[idx, jdx].__repr__() if self._block_mask[idx, jdx] else None
                        msg += '({}, {}): {}\n'.format(idx, jdx, repn)
        return msg

    def __getitem__(self, item):

        if isinstance(item, slice):
            raise NotImplementedError

        if isinstance(item, tuple):
            idx, jdx = item
            assert idx >= 0 and jdx >= 0, 'indices must be positive'
            return self._blocks[item]
        else:
            raise RuntimeError('Wrong index: need a tuple')

    def __setitem__(self, key, value):

        if isinstance(key, slice):
            raise NotImplementedError

        if not isinstance(key, tuple):
            raise RuntimeError('Wrong index: need a tuple')

        idx, jdx = key

        assert idx >= 0 and jdx >= 0, 'indices must be positive'
        assert idx >= jdx, 'symmetric block matrices only set lower triangular entries idx >= jdx'
        if idx == jdx:
            assert is_symmetric_sparse(value), 'Matrix is not symmetric'
        super(BlockSymMatrix, self).__setitem__(key, value)
        super(BlockSymMatrix, self).__setitem__((jdx, idx), value.transpose())

    def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the block matrix.

        Parameters
        ----------
        axes: None, optional
            This argument is in the signature solely for NumPy compatibility reasons. Do not pass in
            anything except for the default value.
        copy: bool, optional
            Indicates whether or not attributes of self should be copied whenever possible.

        Returns
        -------
        BlockMatrix with dimensions reversed
        """
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))
        if copy:
            return self.copy()
        return self

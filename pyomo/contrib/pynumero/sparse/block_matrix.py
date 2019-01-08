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
from scipy.sparse import coo_matrix
from scipy.sparse import isspmatrix
from pyomo.contrib.pynumero.sparse.utils import is_symmetric_sparse
import numpy as np

__all__ = ['BlockMatrix', 'BlockSymMatrix']


# ToDo: better exception handling
class BlockMatrix(object):
    """
    Structured Matrix interface

    Parameters
    -------------------
    nbrows: number of block-rows in the matrix
    nbcols: number of block-columns in the matrix

    """

    def __init__(self, nbrows, nbcols):

        super(BlockMatrix, self).__init__()
        shape = (nbrows, nbcols)

        blocks = []
        for i in range(shape[0]):
            blocks.append([None for j in range(shape[1])])

        self._blocks = np.asarray(blocks, dtype='object')

        self._symmetric = False
        self._name = None

        self._bshape = shape

        self._block_mask = np.zeros(shape, dtype=bool)
        self._brow_lengths = np.zeros(nbrows, dtype=np.int64)
        self._bcol_lengths = np.zeros(nbcols, dtype=np.int64)

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

    def row_block_sizes(self):
        """
        Returns row-block sizes

        Returns
        -------
        ndarray

        """
        return np.copy(self._brow_lengths)

    def col_block_sizes(self):
        """
        Returns col-block sizes

        Returns
        -------
        narray

        """
        return np.copy(self._bcol_lengths)

    def block_shapes(self):
        """
        Returns shapes of blocks in BlockMatrix

        Returns
        -------
        list
        """
        bm, bn =self.bshape
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
            row index to be reseted

        Returns
        -------
        None

        """
        assert 0 <= idx < self.bshape[0], "index must be less than {}".format(self.bshape[0])
        self._brow_lengths[idx] = 0
        self._block_mask[idx, :] = False
        self._blocks[idx, :] = None

    def reset_bcol(self, jdx):
        """
        Resets all blocks in selected column to None

        Parameters
        ----------
        idx: integer
            column index to be reseted

        Returns
        -------
        None

        """
        assert 0 <= jdx < self.bshape[1], "index must be less than {}".format(self.bshape[1])
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
        self._check_mask()

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
        self._check_mask()

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
            #row[idx] = (B.row + row_offsets[i]).astype(idx_dtype, copy=False)
            #col[idx] = (B.col + col_offsets[j]).astype(idx_dtype, copy=False)
            row[idx] = B.row + row_offsets[i]
            col[idx] = B.col + col_offsets[j]
            nnz += B.nnz

        return coo_matrix((data, (row, col)), shape=shape)

    def tocsr(self):
        """
        Converts this matrix to csr format.

        Returns
        -------
        CSRMatrix

        """
        return self.tocoo().tocsr()

    def tocsc(self):
        """
        Converts this matrix to csc format.

        Returns
        -------
        CSCMatrix

        """
        return self.tocoo().tocsc()

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

    def todense(self):
        """
        Returns a dense matrix representation of this matrix.

        Returns
        -------
        arr : ndarray, 2-dimensional
            An array with the same shape and containing the same data
            represented by the block matrix.

        """
        return np.asmatrix(self.toarray())

    def _mul_sparse_matrix(self, other):

        assert other.shape == self.shape, "Dimension mismatch"

        if not isinstance(other, BlockMatrix):
            return self.tocsr()._mul_sparse_matrix(other)
        else:
            raise NotImplementedError("Not supported yet")

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
                    mat[j, i] = self[i, j].transpose()
        return mat

    def is_empty_block(self, idx, jdx):
        """
        Indicates if a block is empty

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

    def _check_mask(self):

        bm, bn = self.bshape

        empty_rows = []
        for idx in range(bm):
            row_bool = np.logical_not(self._block_mask[idx, :])
            if np.all(row_bool):
                empty_rows.append(idx)
        empty_cols = []
        for jdx in range(bn):
            col_bool = np.logical_not(self._block_mask[:, jdx])
            if np.all(col_bool):
                empty_cols.append(jdx)

        if len(empty_rows) > 0:
            msg = 'Operation not allowed with None rows. Specify at least one block in rows:\n'
            msg += '{} of BlockMatrix'.format(empty_rows)
            raise RuntimeError(msg)

        if len(empty_cols)>0:
            msg = 'Operation not allowed with None columns. Specify at least one block in columns:\n'
            msg += '{} of BlockMatrix'.format(empty_cols)
            raise RuntimeError(msg)

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

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.shape)

    def __str__(self):
        msg = ''
        for idx in range(self.bshape[0]):
            for jdx in range(self.bshape[1]):
                repn = self._blocks[idx, jdx].__repr__() if self._block_mask[idx, jdx] else None
                msg += '({}, {}): {}\n'.format(idx, jdx, repn)
        return msg

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, tuple):
            idx, jdx = item
            assert idx >= 0 and jdx >= 0, 'indices must be positive'
            return self._blocks[item]
        else:
            raise RuntimeError('Wrong index')

    def __setitem__(self, key, value):

        if isinstance(key, slice):
            raise NotImplementedError('slices not supported for BlockMatrix')

        if not isinstance(key, tuple):
            raise RuntimeError('Wrong index')

        idx, jdx = key
        assert idx >= 0 and jdx >= 0, 'indices must be positive'
        assert idx < self.bshape[0] and jdx < self.bshape[1], 'indices out of range'
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
            msg = 'blocks need to be sparse matrices'
            assert isinstance(value, BlockMatrix) or isspmatrix(value), msg
            if self._brow_lengths[idx] == 0 and self._bcol_lengths[jdx] == 0:
                self._blocks[idx, jdx] = value
                self._brow_lengths[idx] = value.shape[0]
                self._bcol_lengths[jdx] = value.shape[1]
                self._block_mask[idx, jdx] = True
            elif self._brow_lengths[idx] != 0 and self._bcol_lengths[jdx] == 0:
                if self._brow_lengths[idx] != value.shape[0]:
                    msg = ('Incompatible row dimensions for block ({i},{j}) '
                           'Got {got}, '
                           'expected {exp}.'.format(i=idx, j=jdx,
                                                    exp=self._brow_lengths[idx],
                                                    got=value.shape[0]))
                    raise RuntimeError(msg)
                self._blocks[idx, jdx] = value
                self._block_mask[idx, jdx] = True
                self._bcol_lengths[jdx] = value.shape[1]
            elif self._brow_lengths[idx] == 0 and self._bcol_lengths[jdx] != 0:
                if self._bcol_lengths[jdx] != value.shape[1]:
                    msg = ('Incompatible col dimensions for block ({i},{j}) '
                           'Got {got}, '
                           'expected {exp}.'.format(i=idx, j=jdx,
                                                    exp=self._bcol_lengths[jdx],
                                                    got=value.shape[1]))
                    raise RuntimeError(msg)
                self._blocks[idx, jdx] = value
                self._block_mask[idx, jdx] = True
                self._brow_lengths[idx] = value.shape[0]
            else:
                if self._brow_lengths[idx] != value.shape[0]:
                    msg = ('Incompatible row dimensions for block ({i},{j}) '
                           'Got {got}, '
                           'expected {exp}.'.format(i=idx, j=jdx,
                                                    exp=self._brow_lengths[idx],
                                                    got=value.shape[0]))
                    raise RuntimeError(msg)
                if self._bcol_lengths[jdx] != value.shape[1]:
                    msg = ('Incompatible col dimensions for block ({i},{j}) '
                           'Got {got}, '
                           'expected {exp}.'.format(i=idx, j=jdx,
                                                    exp=self._bcol_lengths[jdx],
                                                    got=value.shape[1]))
                    raise RuntimeError(msg)
                self._blocks[idx, jdx] = value
                self._block_mask[idx, jdx] = True

    def __add__(self, other):

        self._check_mask()

        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape
        assert other.shape == self.shape, \
            'dimensions mismatch {} != {}'.format(self.shape, other.shape)
        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            other._check_mask()
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
            raise NotImplementedError('Sparse Matrix with BlockMatrix addition not supported')
        elif np.isscalar(other):
            raise NotImplementedError('Scalar with BlockMatrix addition not supported')
        else:
            raise NotImplementedError('input not recognized for addition')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):

        self._check_mask()
        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape
        assert other.shape == self.shape, \
            'dimensions mismatch {} != {}'.format(self.shape, other.shape)
        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            other._check_mask()
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
            raise NotImplementedError('Sparse Matrix with BlockMatrix subtraction not supported')
        elif np.isscalar(other):
            raise NotImplementedError('Scalar with BlockMatrix subtraction not supported')
        else:
            raise NotImplementedError('input not recognized for subtraction')

    def __rsub__(self, other):
        self._check_mask()
        result = BlockMatrix(self.bshape[0], self.bshape[1])
        m, n = self.bshape
        assert other.shape == self.shape, \
            'dimensions mismatch {} != {}'.format(self.shape, other.shape)
        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            other._check_mask()
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
            raise NotImplementedError('Sparse Matrix with BlockMatrix subtraction not supported')
        elif np.isscalar(other):
            raise NotImplementedError('Scalar with BlockMatrix subtraction not supported')
        else:
            raise NotImplementedError('input not recognized for subtraction')

    def __mul__(self, other):
        self._check_mask()
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
            other._check_mask()
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

            assert self.shape[1] == other.shape[0], 'Dimension mismatch {}!={}'.format(self.shape[1],
                                                                                       other.shape[0])
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
            return self._mul_sparse_matrix(other)
        else:
            raise NotImplementedError('input not recognized for multiplication')

    def __rmul__(self, other):
        self._check_mask()
        bm, bn = self.bshape
        if np.isscalar(other):
            result = BlockMatrix(bm, bn)
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                    scaled = self._blocks[i, j] * other
                    result[i, j] = scaled
            return result
        else:
            raise NotImplementedError('Not implemented yet')

    def __iadd__(self, other):
        raise NotImplementedError('implicit add not supported for BlockMatrix')

    def __isub__(self, other):
        raise NotImplementedError('implicit sub not supported for BlockMatrix')

    def __imul__(self, other):
        self._check_mask()
        if np.isscalar(other):
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                self._blocks[i, j] = self._blocks[i, j] * other
            return self
        raise NotImplementedError('only scalar support for implicit multiplication')

    def __itruediv__(self, other):
        raise NotImplementedError('implicit divide not supported yet')


class BlockSymMatrix(BlockMatrix):

    def __init__(self, nrowcols):

        super(BlockSymMatrix, self).__init__(nrowcols, nrowcols)
        self._symmetric = True

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.shape)

    def __str__(self):
        msg = ''
        for idx in range(self.bshape[0]):
            for jdx in range(self.bshape[1]):
                if idx >= jdx:
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




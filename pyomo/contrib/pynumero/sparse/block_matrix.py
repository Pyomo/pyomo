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

from scipy.sparse.sputils import get_index_dtype
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings

__all__ = ['BlockMatrix', 'NotFullyDefinedBlockMatrixError']


logger = logging.getLogger(__name__)


class NotFullyDefinedBlockMatrixError(Exception):
    pass


def assert_block_structure(mat):
    if mat.has_undefined_row_sizes():
        msgr = 'Operation not allowed with None rows. ' \
               'Specify at least one block in every row'
        raise NotFullyDefinedBlockMatrixError(msgr)
    if mat.has_undefined_col_sizes():
        msgc = 'Operation not allowed with None columns. ' \
               'Specify at least one block every column'
        raise NotFullyDefinedBlockMatrixError(msgc)


class BlockMatrix(BaseBlockMatrix):
    """
    Structured Matrix interface

    Attributes
    ----------
    _blocks: numpy.ndarray
        2D-array where submatrices are stored
    _bshape: tuple
        number of block-rows and block-columns
    _block_mask: numpy.ndarray
        2D-array with booleans that indicates if block is not empty.
        Empty blocks are represented with None
    _brow_lengths: numpy.ndarray
        1D-array with sizes of block-rows
    _bcol_lengths: numpy.ndarray
        1D-array with sizes of block-columns
    _undefined_brows: set
        set of block row indices with undefined dimensions
    _undefined_bcols: set
        set of block column indices with undefined dimensions

    Parameters
    -------------------
    nbrows: int
        number of block-rows in the matrix
    nbcols: int
        number of block-columns in the matrix

    """
    format = 'block_matrix'

    def __init__(self, nbrows, nbcols):

        shape = (nbrows, nbcols)

        self._blocks = np.empty(shape, dtype='object')
        self._bshape = shape
        self._block_mask = np.zeros(shape, dtype=bool)

        # _brow_lengths and _bcol_lengths get converted to dtype=np.int64 as soon as
        # all of the dimensions are defined. Until then, users do not have access
        # to these. See __setitem__, has_undefined_row_sizes, has_undefined_col_sizes,
        # row_block_sizes, col_block_sizes, and assert_block_structure
        self._brow_lengths = np.empty(nbrows, dtype=np.float64)
        self._bcol_lengths = np.empty(nbcols, dtype=np.float64)
        self._brow_lengths.fill(np.nan)
        self._bcol_lengths.fill(np.nan)
        self._undefined_brows = set(range(nbrows))
        self._undefined_bcols = set(range(nbcols))

    @property
    def bshape(self):
        """
        Returns tuple with the block-shape of the matrix
        """
        return self._bshape

    @property
    def shape(self):
        """
        Returns tuple with total number of rows and columns
        """
        assert_block_structure(self)
        nrows = np.sum(self._brow_lengths)
        ncols = np.sum(self._bcol_lengths)
        return nrows, ncols

    @property
    def nnz(self):
        """
        Returns total number of nonzero values in this matrix
        """
        return sum(blk.nnz for blk in self._blocks[self._block_mask])

    @property
    def dtype(self):
        """
        Returns data type of the matrix.
        """
        all_dtypes = [blk.dtype for blk in self._blocks[self._block_mask]]
        if len(all_dtypes) == 0:
            ref_dtype = np.double
        else:
            ref_dtype = all_dtypes[0]
        if all(ref_dtype is i for i in all_dtypes):
            return ref_dtype
        else:
            raise ValueError('Multiple dtypes found: {0}'.format(str(all_dtypes)))

    @property
    def T(self):
        """
        Transpose matrix
        """
        return self.transpose()

    def row_block_sizes(self, copy=True):
        """
        Returns array with row-block sizes

        Parameters
        ----------
        copy: bool
            If False, then the internal array which stores the row block sizes will be returned without being copied.
            Setting copy to False is risky and should only be done with extreme care.

        Returns
        -------
        numpy.ndarray

        """
        if self.has_undefined_row_sizes():
            raise NotFullyDefinedBlockMatrixError('Some block row lengths are not defined: {0}'.format(str(self._brow_lengths)))
        if copy:
            return self._brow_lengths.copy()
        else:
            return self._brow_lengths

    def col_block_sizes(self, copy=True):
        """
        Returns array with col-block sizes

        Parameters
        ----------
        copy: bool
            If False, then the internal array which stores the column block sizes will be returned without being copied.
            Setting copy to False is risky and should only be done with extreme care.

        Returns
        -------
        numpy.ndarray

        """
        if self.has_undefined_col_sizes():
            raise NotFullyDefinedBlockMatrixError('Some block column lengths are not defined: {0}'.format(str(self._bcol_lengths)))
        if copy:
            return self._bcol_lengths.copy()
        else:
            return self._bcol_lengths

    def get_row_size(self, row):
        if row in self._undefined_brows:
            raise NotFullyDefinedBlockMatrixError('The dimensions of the requested row are not defined.')
        return int(self._brow_lengths[row])

    def get_col_size(self, col):
        if col in self._undefined_bcols:
            raise NotFullyDefinedBlockMatrixError('The dimensions of the requested column are not defined.')
        return int(self._bcol_lengths[col])

    def set_row_size(self, row, size):
        if row in self._undefined_brows:
            self._undefined_brows.remove(row)
            self._brow_lengths[row] = size
            if len(self._undefined_brows) == 0:
                self._brow_lengths = np.asarray(self._brow_lengths, dtype=np.int64)
        else:
            if self._brow_lengths[row] != size:
                raise ValueError('Incompatible row dimensions for '
                                 'row {row}; got {got}; '
                                 'expected {exp}'.format(row=row,
                                                         got=size,
                                                         exp=self._brow_lengths[row]))

    def set_col_size(self, col, size):
        if col in self._undefined_bcols:
            self._undefined_bcols.remove(col)
            self._bcol_lengths[col] = size
            if len(self._undefined_bcols) == 0:
                self._bcol_lengths = np.asarray(self._bcol_lengths, dtype=np.int64)
        else:
            if self._bcol_lengths[col] != size:
                raise ValueError('Incompatible column dimensions for '
                                 'column {col}; got {got}; '
                                 'expected {exp}'.format(col=col,
                                                         got=size,
                                                         exp=self._bcol_lengths[col]))

    def is_row_size_defined(self, row):
        return row not in self._undefined_brows

    def is_col_size_defined(self, col):
        return col not in self._undefined_bcols

    def block_shapes(self):
        """
        Returns list with shapes of blocks in this BlockMatrix

        Notes
        -----
        For a BlockMatrix with 2 block-rows and 2 block-cols
        this method returns [[Block_00.shape, Block_01.shape],[Block_10.shape, Block_11.shape]]

        Returns
        -------
        list

        """
        assert_block_structure(self)
        bm, bn = self.bshape
        sizes = [list() for i in range(bm)]
        for i in range(bm):
            sizes[i] = list()
            for j in range(bn):
                shape = self._brow_lengths[i], self._bcol_lengths[j]
                sizes[i].append(shape)
        return sizes

    def get_block_mask(self, copy=True):
        if copy:
            return self._block_mask.copy()
        else:
            return self._block_mask

    def dot(self, other):
        """
        Ordinary dot product
        """
        return self * other

    def reset_brow(self, idx):
        """
        Resets all blocks in selected block-row to None

        Parameters
        ----------
        idx: int
            block-row index to be reset

        Returns
        -------
        None

        """
        assert 0 <= idx < self.bshape[0], 'Index out of bounds'
        self._block_mask[idx, :] = False
        self._blocks[idx, :] = None

    def reset_bcol(self, jdx):
        """
        Resets all blocks in selected block-column to None

        Parameters
        ----------
        jdx: int
            block-column index to be reset

        Returns
        -------
        None

        """
        assert 0 <= jdx < self.bshape[1], 'Index out of bounds'
        self._block_mask[:, jdx] = False
        self._blocks[:, jdx] = None

    def coo_data(self):
        """
        Returns data array of matrix. The array corresponds to
        the data pointer in COOrdinate matrix format.

        Returns
        -------
        numpy.ndarray with values of all entries in the matrix

        """
        assert_block_structure(self)

        nonzeros = self.nnz
        data = np.empty(nonzeros, dtype=self.dtype)

        nnz = 0

        # get row col indices of blocks that are not none
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            # transform block to coo
            B = self._blocks[i, j].tocoo()
            idx = slice(nnz, nnz + B.nnz)
            # populate coo_data array
            data[idx] = B.data
            nnz += B.nnz

        return data

    def tocoo(self, copy=True):
        """
        Converts this matrix to COOrdinate format.

        Parameters
        ----------
        copy: bool, optional
            This argument is in the signature solely for Scipy compatibility
            reasons. It does not do anything. The data is always copied.

        Returns
        -------
        scipy.sparse.coo_matrix

        """
        assert_block_structure(self)

        dtype = self.dtype

        # Determine offsets for rows
        # e.g. row_offset[1] = block_00.shape[0]
        # e.g. row_offset[2] = block_00.shape[0] + block_10.shape[0]
        row_offsets = np.append(0, np.cumsum(self._brow_lengths))
        # Determine offsets for columns
        col_offsets = np.append(0, np.cumsum(self._bcol_lengths))

        # stores shape of resulting "flattened" matrix
        shape = (row_offsets[-1], col_offsets[-1])

        # total number of nonzeros
        nonzeros = self.nnz

        # create pointers for COO matrix (row, col, data)
        data = np.empty(nonzeros, dtype=dtype)
        idx_dtype = get_index_dtype(maxval=max(shape))
        row = -np.ones(nonzeros, dtype=idx_dtype)
        col = -np.ones(nonzeros, dtype=idx_dtype)

        # populate COO pointers
        nnz = 0
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):

            B = self.get_block(i, j).tocoo()
            # get slice that contains all elements in current block
            idx = slice(nnz, nnz + B.nnz)

            # append B.nnz elements to COO pointers using the slice
            data[idx] = B.data
            row[idx] = B.row + row_offsets[i]
            col[idx] = B.col + col_offsets[j]
            nnz += B.nnz

        return coo_matrix((data, (row, col)), shape=shape)

    def tocsr(self, copy=True):
        """
        Converts this matrix to Compressed Sparse Row format.

        Parameters
        ----------
        copy: bool, optional
            This argument is in the signature solely for Scipy compatibility
            reasons. It does not do anything. The data is always copied.

        Returns
        -------
        scipy.sparse.csr_matrix

        """

        return self.tocoo().tocsr()

    def tocsc(self, copy=True):
        """
        Converts this matrix to Compressed Sparse Column format.

        Parameters
        ----------
        copy: bool, optional
            This argument is in the signature solely for Scipy compatibility
            reasons. It does not do anything. The data is always copied.

        Returns
        -------
        scipy.sparse.csc_matrix

        """
        return self.tocoo().tocsc()

    def toarray(self, order=None, out=None):
        """
        Returns a numpy.ndarray representation of this matrix.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', indicating the NumPy default of C-ordered.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-dimensional, optional
            If specified, uses this array as the output buffer
            instead of allocating a new array to return. The provided
            array must have the same shape and dtype as the sparse
            matrix on which you are calling the method. For most
            sparse types, `out` is required to be memory contiguous
            (either C or Fortran ordered).

        Returns
        -------
        arr : ndarray, 2-dimensional
            An array with the same shape and containing the same data
            represented by the BlockMatrix.

        """
        return self.tocoo().toarray(order=order, out=out)

    def _mul_sparse_matrix(self, other):
        """
        Perform self * other where other is a block matrix

        Parameters
        ----------
        other: BlockMatrix

        Returns
        -------
        BlockMatrix
        """

        if isinstance(other, BlockMatrix):
            assert other.bshape[0] == self.bshape[1], "Dimension mismatch"
            result = BlockMatrix(self.bshape[0], other.bshape[1])

            # get dimenions from the other matrix
            other_col_sizes = other.col_block_sizes(copy=False)

            # compute result
            for i in range(self.bshape[0]):
                for j in range(other.bshape[1]):
                    accum = coo_matrix((self._brow_lengths[i],
                                        other_col_sizes[i]))
                    for k in range(self.bshape[1]):
                        if self._block_mask[i, k] and not other.is_empty_block(k, j):
                            prod = self._blocks[i,k] * other.get_block(k, j)
                            accum = accum + prod
                    result.set_block(i, j, accum)
            return result
        elif isspmatrix(other):
            raise NotImplementedError('BlockMatrix multiply with spmatrix not supported. Multiply a BlockMatrix '
                                      'with another BlockMatrix of compatible dimensions.')
        else:
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def transpose(self, axes=None, copy=True):
        """
        Creates a transpose copy of the BlockMatrix.

        Parameters
        ----------
        axes: None, optional
            This argument is in the signature solely for NumPy compatibility reasons. Do not pass in
            anything except for the default value.
        copy: bool
            This argument is in the signature solely for scipy compatibility reasons. Do not pass in
            anything except for the default value.

        Returns
        -------
        BlockMatrix with dimensions reversed

        """
        """
        It is difficult to support transpose without copying. A "TransposeView" object might be a better approach.
        """
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))
        if not copy:
            raise ValueError('BlockMatrix only supports transpose with copy=True')

        m, n = self.bshape
        mat = BlockMatrix(n, m)
        mat._brow_lengths = self._bcol_lengths.copy()
        mat._bcol_lengths = self._brow_lengths.copy()
        mat._undefined_brows = set(self._undefined_bcols)
        mat._undefined_bcols = set(self._undefined_brows)
        for i, j in zip(*np.nonzero(self._block_mask)):
            mat.set_block(j, i, self.get_block(i, j).transpose(copy=True))
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
        bool

        """
        return not self._block_mask[idx, jdx]

    def has_undefined_row_sizes(self):
        """
        Indicates if the matrix has block-rows with undefined dimensions

        Returns
        -------
        bool

        """
        return len(self._undefined_brows) != 0

    def has_undefined_col_sizes(self):
        """
        Indicates if the matrix has block-columns with undefined dimensions

        Returns
        -------
        bool

        """
        return len(self._undefined_bcols) != 0

    def copyfrom(self, other, deep=True):
        """
        Copies entries of other matrix into this matrix. This method provides
        an easy way to populate a BlockMatrix from scipy.sparse matrices. It also
        intended to facilitate copying values from other BlockMatrix to this BlockMatrix

        Parameters
        ----------
        other: BlockMatrix or scipy.spmatrix
        deep: bool
            If deep is True and other is a BlockMatrix, then the blocks in other are copied. If deep is False
            and other is a BlockMatrix, then the blocks in other are not copied.

        Returns
        -------
        None

        """
        assert_block_structure(self)
        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            m, n = self.bshape
            if deep:
                for i in range(m):
                    for j in range(n):
                        if not other.is_empty_block(i, j):
                            self.set_block(i, j, other.get_block(i, j).copy())
                        else:
                            self.set_block(i, j, None)
            else:
                for i in range(m):
                    for j in range(n):
                        self.set_block(i, j, other.get_block(i, j))

        elif isspmatrix(other) or isinstance(other, np.ndarray):
            assert other.shape == self.shape, \
                'dimensions mismatch {} != {}'.format(self.shape, other.shape)
            if isinstance(other, np.ndarray):
                # cast numpy.array to coo_matrix for ease of manipulation
                m = csr_matrix(other)
            else:
                m = other.tocsr()

            # determine offsets for each block
            row_offsets = np.append(0, np.cumsum(self._brow_lengths))
            col_offsets = np.append(0, np.cumsum(self._bcol_lengths))

            # maps 'flat' matrix to the block structure of this matrix
            # csr row slicing is fast
            # csc column slicing is fast
            # therefore, we do the row slice once for each row, then we convert to csc for the column slicing
            for i in range(self.bshape[0]):
                mm = m[row_offsets[i]:row_offsets[i+1], :].tocsc()
                for j in range(self.bshape[1]):
                    mmm = mm[:, col_offsets[j]:col_offsets[j+1]]

                    if self.is_empty_block(i, j) and mmm.nnz == 0:
                        self.set_block(i, j, None)
                    else:
                        self.set_block(i, j, mmm)

        else:
            raise NotImplementedError("Format not supported. BlockMatrix can only copy data from another BlockMatrix, "
                                      "a numpy array, or a scipy sparse matrix.")

    def copyto(self, other, deep=True):
        """
        Copies entries of this BlockMatrix into other. This method provides
        an easy way to copy values of this matrix into another format.

        Parameters
        ----------
        other: BlockMatrix or scipy.spmatrix
        deep: bool
            If deep is True and other is a BlockMatrix, then the blocks in this BlockMatrix are copied. If deep is
            False and other is a BlockMatrix, then the blocks in this BlockMatrix are not copied.

        Returns
        -------
        None

        """
        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            if deep:
                m, n = self.bshape
                for i in range(m):
                    for j in range(n):
                        if self.is_empty_block(i, j):
                            other.set_block(i, j, None)
                        else:
                            other.set_block(i, j, self.get_block(i, j).copy())
            else:
                m, n = self.bshape
                for i in range(m):
                    for j in range(n):
                        other.set_block(i, j, self.get_block(i, j))
        elif isspmatrix(other) or isinstance(other, np.ndarray):
            assert other.shape == self.shape, \
                'dimensions mismatch {} != {}'.format(self.shape, other.shape)

            # create temporary matrix to copy
            tmp_matrix = self.tocoo()
            if isinstance(other, coo_matrix):
                np.copyto(other.data, tmp_matrix.data)
                np.copyto(other.row, tmp_matrix.row)
                np.copyto(other.col, tmp_matrix.col)
            elif isinstance(other, csr_matrix):
                tmp_matrix2 = tmp_matrix.tocsr()
                np.copyto(other.data, tmp_matrix2.data)
                np.copyto(other.indices, tmp_matrix2.indices)
                np.copyto(other.indptr, tmp_matrix2.indptr)
            elif isinstance(other, csc_matrix):
                tmp_matrix2 = tmp_matrix.tocsc()
                np.copyto(other.data, tmp_matrix2.data)
                np.copyto(other.indices, tmp_matrix2.indices)
                np.copyto(other.indptr, tmp_matrix2.indptr)
            elif isinstance(other, np.ndarray):
                np.copyto(other, tmp_matrix.toarray())
            else:
                raise NotImplementedError("Format not supported. BlockMatrix can only copy data to another BlockMatrix, "
                                          "a numpy array, or a scipy sparse coo, csr, or csc matrix.")
        else:
            raise NotImplementedError("Format not supported. BlockMatrix can only copy data to another BlockMatrix, "
                                      "a numpy array, or a scipy sparse coo, csr, or csc matrix.")

    def copy(self, deep=True):
        """
        Makes a copy of this BlockMatrix

        Parameters
        ----------
        deep: bool
            If deep is True, then the blocks in this BlockMatrix are copied

        Returns
        -------
        BlockMatrix

        """
        result = BlockMatrix(self.bshape[0], self.bshape[1])
        result._brow_lengths = self._brow_lengths.copy()
        result._bcol_lengths = self._bcol_lengths.copy()
        result._undefined_brows = set(self._undefined_brows)
        result._undefined_bcols = set(self._undefined_bcols)
        ii, jj = np.nonzero(self._block_mask)
        if deep:
            for i, j in zip(ii, jj):
                result.set_block(i, j, self._blocks[i, j].copy())
        else:
            for i, j in zip(ii, jj):
                result.set_block(i, j, self._blocks[i, j])
        return result

    def copy_structure(self):
        """
        Makes a copy of the structure of this BlockMatrix. This proivides a
        light-weighted copy of each block in this BlockMatrix. The blocks in the
        resulting matrix have the same shape as in the original matrices but not
        the same number of nonzeros.

        Returns
        -------
        BlockMatrix

        """
        m, n = self.bshape
        result = BlockMatrix(m, n)
        for row in range(m):
            if self.is_row_size_defined(row):
                result.set_row_size(row, self.get_row_size(row))
        for col in range(n):
            if self.is_col_size_defined(col):
                result.set_col_size(col, self.get_col_size(col))
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            if isinstance(self._blocks[i, j], BlockMatrix):
                result.set_block(i, j, self._blocks[i, j].copy_structure())
            else:
                nrows, ncols = self._blocks[i, j].shape
                result.set_block(i, j, coo_matrix((nrows, ncols)))
        return result

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.bshape)

    def _print(self, indent):
        msg = ''
        for idx in range(self.bshape[0]):
            for jdx in range(self.bshape[1]):
                if self.is_empty_block(idx, jdx):
                    msg += indent + str((idx, jdx)) + ': ' + str(None) + '\n'
                else:
                    block = self.get_block(idx, jdx)
                    if isinstance(block, BlockMatrix):
                        msg += indent + str((idx, jdx)) + ': ' + block.__class__.__name__ + str(block.bshape) + '\n'
                        msg += block._print(indent=indent+'   ')
                    else:
                        msg += indent + str((idx, jdx)) + ': ' + block.__class__.__name__ + str(block.shape) + '\n'
        return msg

    def __str__(self):
        return self._print(indent='')

    def get_block(self, row, col):
        assert row >= 0 and col >= 0, 'indices must be positive'
        assert row < self.bshape[0] and \
               col < self.bshape[1], 'Indices out of range'
        return self._blocks[row, col]

    def set_block(self, row, col, value):
        assert row >= 0 and col >= 0, 'Indices must be positive'

        assert row < self.bshape[0] and col < self.bshape[1], 'Indices out of range'

        if value is None:
            self._blocks[row, col] = None
            self._block_mask[row, col] = False
        else:
            if isinstance(value, BaseBlockMatrix):
                assert_block_structure(value)
            elif isinstance(value, np.ndarray):
                if value.ndim != 2:
                    msg = 'blocks need to be sparse matrices or BlockMatrices'
                    raise ValueError(msg)
                msg = 'blocks need to be sparse matrices or BlockMatrices; a numpy array was given; copying the numpy array to a coo_matrix'
                logger.warning(msg)
                warnings.warn(msg)
                value = coo_matrix(value)
            else:
                assert isspmatrix(value), 'blocks need to be sparse matrices or BlockMatrices'

            nrows, ncols = value.shape
            self.set_row_size(row, nrows)
            self.set_col_size(col, ncols)
            self._blocks[row, col] = value
            self._block_mask[row, col] = True

    def __getitem__(self, item):
        raise NotImplementedError('BlockMatrix does not support __getitem__. '
                                  'Use get_block or set_block to access sub-blocks.')

    def __setitem__(self, item, val):
        raise NotImplementedError('BlockMatrix does not support __setitem__. '
                                  'Use get_block or set_block to access sub-blocks.')

    def _binary_operation_helper(self, other, operation):
        assert_block_structure(self)
        result = BlockMatrix(self.bshape[0], self.bshape[1])

        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            assert other.shape == self.shape, \
                'dimensions mismatch {} != {}'.format(self.shape, other.shape)
            assert_block_structure(other)

            block_indices = np.bitwise_or(self.get_block_mask(copy=False), other.get_block_mask(copy=False))
            for i, j in zip(*np.nonzero(block_indices)):
                mat1 = self.get_block(i, j)
                mat2 = other.get_block(i, j)
                if mat1 is not None and mat2 is not None:
                    result.set_block(i, j, operation(mat1, mat2))
                elif mat1 is not None:
                    result.set_block(i, j, operation(mat1, 0))
                elif mat2 is not None:
                    result.set_block(i, j, operation(0, mat2))
            return result
        elif isspmatrix(other):
            # Note: this is not efficient but is just for flexibility.
            mat = self.copy_structure()
            mat.copyfrom(other)
            return operation(self, mat)
        elif np.isscalar(other):
            for i, j in zip(*np.nonzero(self.get_block_mask(copy=False))):
                result.set_block(i, j, operation(self.get_block(i, j), other))
            return result
        else:
            return NotImplemented

    def __add__(self, other):
        return self._binary_operation_helper(other, operator.add)

    def __radd__(self, other):
        return self._binary_operation_helper(other, operator.add)

    def __sub__(self, other):
        return self._binary_operation_helper(other, operator.sub)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        """
        When doing A*B with numpy arrays, element-by-element multiplication is done. However, when doing
        A*B with scipy sparse matrices, a matrix-matrix dot product is performed. We are following the
        scipy sparse matrix API.
        """

        bm, bn = self.bshape
        if np.isscalar(other):
            return self._binary_operation_helper(other, operator.mul)
        elif isinstance(other, BlockVector):
            assert bn == other.bshape[0], 'Dimension mismatch'
            assert self.shape[1] == other.shape[0], 'Dimension mismatch'
            assert not other.has_none, 'Block vector must not have none entries'
            assert_block_structure(self)

            nblocks = self.bshape[0]
            result = BlockVector(nblocks)
            for i in range(bm):
                result.set_block(i, np.zeros(self._brow_lengths[i]))
            for i, j in zip(*np.nonzero(self._block_mask)):
                    x = other.get_block(j)
                    A = self._blocks[i, j]
                    blk = result.get_block(i)
                    _tmp = A*x
                    _tmp += blk
                    result.set_block(i, _tmp)
            return result
        elif isinstance(other, np.ndarray):

            if other.ndim != 1:
                raise NotImplementedError('Operation not supported by BlockMatrix')

            assert self.shape[1] == other.shape[0], \
                'Dimension mismatch {}!={}'.format(self.shape[1],
                                                   other.shape[0])
            assert_block_structure(self)

            nblocks = self.bshape[0]
            result = BlockVector(nblocks)
            for i in range(bm):
                result.set_block(i, np.zeros(self._brow_lengths[i]))
                counter = 0
                for j in range(bn):
                    if not self.is_empty_block(i, j):
                        A = self._blocks[i, j]
                        x = other[counter: counter + A.shape[1]]
                        blk = result.get_block(i)
                        blk += A * x
                    counter += self.get_col_size(j)
            return result
        elif isinstance(other, BlockMatrix) or isspmatrix(other):
            assert_block_structure(self)
            return self._mul_sparse_matrix(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if np.isscalar(other):
            return self._binary_operation_helper(other, operator.truediv)
        return NotImplemented

    def __rtruediv__(self, other):
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __rmul__(self, other):
        """
        When doing A*B with numpy arrays, element-by-element multiplication is done. However, when doing
        A*B with scipy sparse matrices, a matrix-matrix dot product is performed. We are following the
        scipy sparse matrix API.
        """
        bm, bn = self.bshape
        if np.isscalar(other):
            result = BlockMatrix(bm, bn)
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                result.set_block(i, j, self._blocks[i, j] * other)
            return result
        elif isspmatrix(other):
            raise NotImplementedError('sparse matrix times block matrix is not supported.')
        else:
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __pow__(self, other):
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __abs__(self):
        res = BlockMatrix(*self.bshape)
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            res.set_block(i, j, abs(self._blocks[i, j]))
        return res

    def __iadd__(self, other):

        if isinstance(other, BlockMatrix):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
            assert other.shape == self.shape, \
                'dimensions mismatch {} != {}'.format(self.shape, other.shape)

            iterator = set(zip(*np.nonzero(self._block_mask)))
            iterator.update(zip(*np.nonzero(other._block_mask)))
            for i, j in iterator:
                if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                    self._blocks[i, j] += other.get_block(i, j)
                elif not other.is_empty_block(i, j):
                    self.set_block(i, j, other.get_block(i, j).copy())
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

            iterator = set(zip(*np.nonzero(self._block_mask)))
            iterator.update(zip(*np.nonzero(other._block_mask)))
            for i, j in iterator:
                if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                    self._blocks[i, j] -= other.get_block(i, j)
                elif not other.is_empty_block(i, j):
                    self.set_block(i, j, -other.get_block(i, j))  # the copy happens in __neg__ of other.get_block(i, j)
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
                self._blocks[i, j] *= other
            return self
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __itruediv__(self, other):
        if np.isscalar(other):
            ii, jj = np.nonzero(self._block_mask)
            for i, j in zip(ii, jj):
                self._blocks[i, j] /= other
            return self
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __ifloordiv__(self, other):
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __neg__(self):
        res = BlockMatrix(*self.bshape)
        ii, jj = np.nonzero(self._block_mask)
        for i, j in zip(ii, jj):
            res.set_block(i, j, -self._blocks[i, j])
        return res

    def _comparison_helper(self, operation, other):
        result = BlockMatrix(self.bshape[0], self.bshape[1])

        if isinstance(other, BlockMatrix) and other.bshape == self.bshape:
            m, n = self.bshape
            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j) and not other.is_empty_block(i, j):
                        result.set_block(i, j, operation(self._blocks[i, j], other.get_block(i, j)))
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        mat = coo_matrix((nrows, ncols))
                        if not self.is_empty_block(i, j):
                            result.set_block(i, j, operation(self._blocks[i, j], mat))
                        elif not other.is_empty_block(i, j):
                            result.set_block(i, j, operation(mat, other.get_block(i, j)))
                        else:
                            result.set_block(i, j, operation(mat, mat))
            return result
        elif isinstance(other, BlockMatrix) or isspmatrix(other):
            if isinstance(other, BlockMatrix):
                raise NotImplementedError('Operation supported with same block structure only')
            else:
                raise NotImplementedError('Operation not supported by BlockMatrix')
        elif np.isscalar(other):
            m, n = self.bshape
            for i in range(m):
                for j in range(n):
                    if not self.is_empty_block(i, j):
                        result.set_block(i, j, operation(self._blocks[i, j], other))
                    else:
                        nrows = self._brow_lengths[i]
                        ncols = self._bcol_lengths[j]
                        matc = coo_matrix((nrows, ncols))
                        result.set_block(i, j, operation(matc, other))
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockMatrix':
                raise RuntimeError('Operation not supported by BlockMatrix')
            raise NotImplementedError('Operation not supported by BlockMatrix')

    def __eq__(self, other):
        return self._comparison_helper(operation=operator.eq, other=other)

    def __ne__(self, other):
        return self._comparison_helper(operation=operator.ne, other=other)

    def __le__(self, other):
        return self._comparison_helper(operation=operator.le, other=other)

    def __lt__(self, other):
        return self._comparison_helper(operation=operator.lt, other=other)

    def __ge__(self, other):
        return self._comparison_helper(operation=operator.ge, other=other)

    def __gt__(self, other):
        return self._comparison_helper(operation=operator.gt, other=other)

    def __len__(self):
        raise NotImplementedError('Operation not supported by BlockMatrix')

    def __matmul__(self, other):
        return self.__mul__(other)

    def __rmatmul__(self, other):
        return self.__rmul__(other)

    def pprint(self):
        """Prints BlockMatrix in pretty format"""
        print(str(self))

    def get_block_column_index(self, index):
        """
        Returns block-column idx from matrix column index.

        Parameters
        ----------
        index: int
            Column index

        Returns
        -------
        int

        """
        msgc = 'Operation not allowed with None columns. ' \
               'Specify at least one block in every column'
        assert not self.has_undefined_col_sizes(), msgc

        bm, bn = self.bshape
        # get cummulative sum of block sizes
        cum = self._bcol_lengths.cumsum()
        assert index >= 0, 'index out of bounds'
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

    def get_block_row_index(self, index):
        """
        Returns block-row idx from matrix row index.

        Parameters
        ----------
        index: int
            Row index

        Returns
        -------
        int

        """
        msgr = 'Operation not allowed with None rows. ' \
               'Specify at least one block in every row'
        assert not self.has_undefined_row_sizes(), msgr

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
        """
        Returns vector of column j

        Parameters
        ----------
        j: int
            Column index

        Returns
        -------
        pyomo.contrib.pynumero.sparse BlockVector

        """
        # Note: this method is slightly different than the sparse_matrix
        # from scipy. It returns an array always instead of returning
        # an sparse matrix with a single column

        # get block column index
        bcol = self.get_block_column_index(j)
        bm, bn = self.bshape

        # compute offset columns
        offset = 0
        if bcol > 0:
            cum_sum = self._bcol_lengths.cumsum()
            offset = cum_sum[bcol-1]

        # build block vector
        result = BlockVector(bm)
        for i in range(bm):
            mat = self.get_block(i, bcol)
            if self.is_empty_block(i, bcol):
                v = np.zeros(self._brow_lengths[i])
            elif isinstance(mat, BaseBlockMatrix):
                # this will return a block vector
                v = mat.getcol(j-offset)
            else:
                # if it is sparse matrix transform array to vector
                v = mat.getcol(j-offset).toarray().flatten()
            result.set_block(i, v)
        return result

    def getrow(self, i):
        """
        Returns vector of column i

        Parameters
        ----------
        i: int
            Row index

        Returns
        -------
        pyomo.contrib.pynumero.sparse BlockVector

        """
        # Note: this method is slightly different than the sparse_matrix
        # from scipy. It returns an array always instead of returning
        # an sparse matrix with a single row

        # get block column index
        brow = self.get_block_row_index(i)
        bm, bn = self.bshape

        # compute offset columns
        offset = 0
        if brow > 0:
            cum_sum = self._brow_lengths.cumsum()
            offset = cum_sum[brow-1]

        # build block vector
        result = BlockVector(bn)
        for j in range(bn):
            mat = self.get_block(brow, j)
            if self.is_empty_block(brow, j):
                v = np.zeros(self._bcol_lengths[j])
            elif isinstance(mat, BaseBlockMatrix):
                # this will return a block vector
                v = mat.getcol(i-offset)
            else:
                # if it is sparse matrix transform array to vector
                v = mat.getcol(i-offset).toarray().flatten()
            result.set_block(j, v)
        return result

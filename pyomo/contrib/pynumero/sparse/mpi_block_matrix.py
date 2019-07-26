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

from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.contrib.pynumero.sparse.utils import is_symmetric_sparse
from pyomo.contrib.pynumero.sparse import empty_matrix
from pyomo.contrib.pynumero.sparse.warnings import MPISpaceWarning
from .base_block import BaseBlockMatrix
from warnings import warn
from mpi4py import MPI
import numpy as np

# Array classifiers
SINGLE_OWNER=1
MULTIPLE_OWNER=2
ALL_OWN_IT=0


# ALL_OWNED = -1

class MPIBlockMatrix(BaseBlockMatrix):
    """
    Parallel Structured Matrix interface

    Attributes
    ----------
    _rank_owner: numpy.ndarray
        2D-array with processor ownership of each block. A block can be own by a
        single processor or by all processors. Blocks own by all processors have
        ownership -1. Blocks own by a single processor have ownership rank. where
        rank=MPI.COMM_WORLD.Get_rank()
    _mpiw: MPI communicator
        A communicator from the MPI space. Typically MPI.COMM_WORLD
    _block_matrix: BlockMatrix
        Internal BlockMatrix. Blocks that belong to this processor are stored
        in _block_matrix.
    _owned_mask: numpy.ndarray bool
        2D-array that indicates if a block belongs to this processor. While
        _rank_owner tells which processor(s) owns each block, _owned_mask tells
        if a block is owned by this processor. Blocks that are owned by everyone
        (i.e. ownership = -1) are True in _owned_mask
    _unique_owned_mask: numpy.ndarray bool
        2D-array that indicates if a block belongs to this processor. While
        _rank_owner tells which processor(s) owns each block, _unique_owned_mask tells
        if a block is owned by this processor. Blocks that are owned by everyone
        (i.e. ownership = -1) are False in _unique_owned_mask
    _row_type: numpy.ndarray int
        1D-Array that classify the type of row. Rows can be of three types. If
        all the row blocks have the same owner, the row is said to be SINGLE_OWNER=1.
        If all the blocks in the row have ownership equals -1, the row is said
        to be ALL_OWN_IT=0. Finally if two or more blocks have different owner in
        a row, the row is said to be MULTIPLE_OWNER=2. This information is useful
        when performing matrix-vector and matrix-matrix products
    _col_type: numpy.ndarray int
        1D-Array that classify the type of column. Columns can be of three types. If
        all the column blocks have the same owner, the column is said to be SINGLE_OWNER=1.
        If all the blocks in the column have ownership equals -1, the column is said
        to be ALL_OWN_IT=0. Finally if two or more blocks have different owner in
        a column, the column is said to be MULTIPLE_OWNER=2. This information is useful
        when performing matrix-vector and matrix-matrix products
    _brow_lengths: numpy.ndarray
        1D-array with sizes of block-rows
    _bcol_lengths: numpy.ndarray
        1D-array with sizes of block-columns
    _need_broadcast_sizes: bool
        True if length of any block changed. If true user will need to call
        broadcast_block_sizes in the future before performing any operation.
        Users will be notified if that is the case.
    _done_first_broadcast_sizes: bool
        True if broadcast_block_sizes has been called and the length of any
        block changed since. If true user will need to call
        broadcast_block_sizes in the future before performing any operation.
        Users will be notified if that is the case.

    Parameters
    -------------------
    nbrows : int
             number of block-rows in the matrix
    nbcols : int
             number of block-columns in the matrix
    rank_ownership: array_like
                    integer 2D array that specifies the rank of process
                    owner of each block in the matrix. For blocks that are
                    owned by all processes the rank is -1. Blocks that are
                    None should be owned by all processes.
    mpi_comm : MPI communicator
    row_block_sizes: array_like, optional
        Array_like of size nbrows. This specifies the length of each row in
        the MPIBlockMatrix.
    col_block_sizes: array_like, optional
        Array_like of size nbcols. This specifies the length of each column in
        the MPIBlockMatrix.
    """

    def __init__(self,
                 nbrows,
                 nbcols,
                 rank_ownership,
                 mpi_comm,
                 row_block_sizes=None,
                 col_block_sizes=None):

        shape = (nbrows, nbcols)
        self._block_matrix = BlockMatrix(nbrows, nbcols)
        self._mpiw = mpi_comm
        self._rank_owner = np.zeros(shape, dtype=np.int64)
        self._owned_mask = np.zeros(shape, dtype=bool)
        self._unique_owned_mask = np.zeros(shape, dtype=bool)

        rank = self._mpiw.Get_rank()

        if isinstance(rank_ownership, list):
            rank_owner_format = 1
        elif isinstance(rank_ownership, np.ndarray):
            rank_owner_format = 2
            assert rank_ownership.ndim == 2, 'rank_ownership must be of size 2'
        else:
            raise RuntimeError('rank_ownership must be a list of lists or a numpy array')

        for i in range(nbrows):
            for j in range(nbcols):

                if rank_owner_format == 1:
                    owner = rank_ownership[i][j]
                else:
                    owner = rank_ownership[i, j]
                assert owner < self._mpiw.Get_size(), \
                    'rank owner out of range'
                self._rank_owner[i, j] = owner
                if rank == owner or owner < 0:
                    self._owned_mask[i, j] = True
                    if owner == rank:
                        self._unique_owned_mask[i, j] = True

        # Note: this requires communication but is disabled when assertions
        # are turned off
        assert self._assert_correct_owners(), \
            'rank_owner must be the same in all processors'

        # classify row ownership
        self._row_type = np.empty(nbrows, dtype=np.int64)
        for i in range(nbrows):
            self._row_type[i] = ALL_OWN_IT
            last_owner = -1
            for j in range(nbcols):
                owner = self._rank_owner[i, j]
                if owner >= 0:
                    if self._row_type[i] == ALL_OWN_IT:
                        last_owner = owner
                        self._row_type[i] = SINGLE_OWNER
                    elif self._row_type[i] == SINGLE_OWNER and owner != last_owner:
                        self._row_type[i] = MULTIPLE_OWNER
                        break

        # classify column ownership
        self._column_type = np.empty(nbcols, dtype=np.int64)
        for j in range(nbcols):
            self._column_type[j] = ALL_OWN_IT
            last_owner = -1
            for i in range(nbrows):
                owner = self._rank_owner[i, j]
                if owner >= 0:
                    if self._column_type[j] == ALL_OWN_IT:
                        last_owner = owner
                        self._column_type[j] = SINGLE_OWNER
                    elif self._column_type[j] == SINGLE_OWNER and owner != last_owner:
                        self._column_type[j] = MULTIPLE_OWNER
                        break

        if row_block_sizes is None and col_block_sizes is None:
            self._need_broadcast_sizes = True
            self._done_first_broadcast_sizes = False
            self._brow_lengths = np.zeros(nbrows, dtype=np.int64)
            self._bcol_lengths = np.zeros(nbcols, dtype=np.int64)
        else:
            if row_block_sizes is not None and col_block_sizes is not None:
                self._need_broadcast_sizes = False
                self._done_first_broadcast_sizes = True
                self._brow_lengths = np.array(row_block_sizes, dtype=np.int64)
                self._bcol_lengths = np.array(col_block_sizes, dtype=np.int64)
            elif row_block_sizes is None and col_block_sizes is not None:
                raise RuntimeError('Specify row_block_sizes')
            else:
                raise RuntimeError('Specify col_block_sizes')

        # make some of the pointers unmutable
        self._rank_owner.flags.writeable = False
        self._owned_mask.flags.writeable = False

    @property
    def bshape(self):
        """
        Returns tuple with the block-shape of the matrix
        """
        return self._block_matrix.bshape

    @property
    def shape(self):
        """
        Returns tuple with total number of rows and columns
        """
        self._assert_broadcasted_sizes()
        return np.sum(self._brow_lengths), np.sum(self._bcol_lengths)

    @property
    def nnz(self):
        """
        Returns total number of nonzero values in this matrix
        """
        local_nnz = 0
        rank = self._mpiw.Get_rank()
        block_indices = self._unique_owned_mask if rank!=0 else self._owned_mask

        # this is an easy and efficient way to loop though owned blocks
        ii, jj = np.nonzero(block_indices)
        for i, j in zip(ii, jj):
            if not self._block_matrix.is_empty_block(i, j):
                local_nnz += self._block_matrix[i, j].nnz

        return self._mpiw.allreduce(local_nnz, op=MPI.SUM)

    @property
    def owned_blocks(self):
        """
        Returns list with inidices of blocks owned by this processor.
        """
        bm, bn = self.bshape
        owned_blocks = []
        for i in range(bm):
            for j in range(bn):
                if self._owned_mask[i, j]:
                    owned_blocks.append((i,j))
        return owned_blocks

    @property
    def shared_blocks(self):
        """
        Returns list with inidices of blocks shared by all processors
        """
        bm, bn = self.bshape
        owned_blocks = []
        for i in range(bm):
            for j in range(bn):
                if self._owned_mask[i, j] and self._rank_owner[i, j]<0:
                    owned_blocks.append((i,j))
        return owned_blocks

    @property
    def rank_ownership(self):
        """
        Returns 2D array that specifies process rank that owns each blocks. If
        a block is owned by all the ownership=-1.
        """
        return self._rank_owner

    @property
    def ownership_mask(self):
        """
        Returns boolean 2D-Array that indicates which blocks are owned by
        this processor
        """
        return self._owned_mask

    @property
    def mpi_comm(self):
        """Returns MPI communicator"""
        return self._mpiw

    @property
    def T(self):
        """
        Transpose matrix
        """
        return self.transpose()

    def dot(self, other):
        """
        Ordinary dot product
        """
        return self * other

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
        MPIBlockMatrix with dimensions reversed
        """
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        m = self.bshape[0]
        n = self.bshape[1]
        if not self._need_broadcast_sizes:
            result = MPIBlockMatrix(n, m, self._rank_owner.T, self._mpiw,
                                    row_block_sizes=self._bcol_lengths.copy(),
                                    col_block_sizes=self._brow_lengths.copy())
        else:
            raise RuntimeError('Call broadcast_block_sizes() before transposing')

        rows, columns = np.nonzero(self.ownership_mask)
        for i, j in zip(rows, columns):
            if self[i, j] is not None:
                result[j, i] = self[i, j].transpose(copy=copy)
        return result

    def tocoo(self):
        """
        Converts this matrix to coo_matrix format.

        Returns
        -------
        coo_matrix

        """
        raise RuntimeError('Operation not supported by MPIBlockMatrix')

    def tocsr(self):
        """
        Converts this matrix to csr format.

        Returns
        -------
        csr_matrix

        """
        raise RuntimeError('Operation not supported by MPIBlockMatrix')

    def tocsc(self):
        """
        Converts this matrix to csc format.

        Returns
        -------
        csc_matrix

        """
        raise RuntimeError('Operation not supported by MPIBlockMatrix')

    def tolil(self, copy=False):
        BaseBlockMatrix.tolil(self, copy=copy)

    def todia(self, copy=False):
        BaseBlockMatrix.todia(self, copy=copy)

    def tobsr(self, blocksize=None, copy=False):
        BaseBlockMatrix.tobsr(self, blocksize=blocksize, copy=copy)

    def coo_data(self):
        raise RuntimeError('Operation not supported by MPIBlockMatrix')

    def toarray(self):
        """
        Returns a dense ndarray representation of this matrix.

        Returns
        -------
        arr : ndarray, 2-dimensional
            An array with the same shape and containing the same data
            represented by the block matrix.

        """
        raise RuntimeError('Operation not supported by MPIBlockMatrix')

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
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    # Note: this requires communication
    def broadcast_block_sizes(self):
        """
        Send sizes of all blocks to all processors. After this method is called
        this MPIBlockMatrix knows it's dimensions of all rows and columns. This method
        must be called before running any operations with the MPIBlockVector.
        """
        rank = self._mpiw.Get_rank()
        num_processors = self._mpiw.Get_size()

        local_row_data = self._block_matrix.row_block_sizes()
        local_col_data = self._block_matrix.col_block_sizes()
        send_data = np.concatenate([local_row_data, local_col_data])

        receive_data = np.empty(num_processors * (self.bshape[0] + self.bshape[1]),
                                dtype=np.int64)

        self._mpiw.Allgather(send_data, receive_data)

        proc_dims = np.split(receive_data, num_processors)
        m, n = self.bshape

        # check the rows
        for i in range(m):
            rows_length = set()
            for k in range(num_processors):
                row_sizes, col_sizes = np.split(proc_dims[k],
                                                [self.bshape[0]])
                rows_length.add(row_sizes[i])
            if len(rows_length)>2:
                msg = 'Row {} has more than one dimension accross processors'.format(i)
                raise RuntimeError(msg)
            elif len(rows_length) == 2:
                if 0 not in rows_length:
                    msg = 'Row {} has more than one dimension accross processors'.format(i)
                    raise RuntimeError(msg)
                rows_length.remove(0)

            # here rows_length must only have one element
            self._brow_lengths[i] = rows_length.pop()

        # check columns
        for i in range(n):
            cols_length = set()
            for k in range(num_processors):
                rows_sizes, col_sizes = np.split(proc_dims[k],
                                                 [self.bshape[0]])
                cols_length.add(col_sizes[i])
            if len(cols_length)>2:
                msg = 'Column {} has more than one dimension accross processors'.format(i)
                raise RuntimeError(msg)
            elif len(cols_length) == 2:
                if 0 not in cols_length:
                    msg = 'Column {} has more than one dimension accross processors'.format(i)
                    raise RuntimeError(msg)
                cols_length.remove(0)

            # here rows_length must only have one element
            self._bcol_lengths[i] = cols_length.pop()

        self._need_broadcast_sizes = False
        self._done_first_broadcast_sizes = True

    def row_block_sizes(self, copy=True):
        """
        Returns row-block sizes

        Returns
        -------
        numpy.ndarray

        """
        self._assert_broadcasted_sizes()
        if copy:
            return np.copy(self._brow_lengths)
        return self._brow_lengths

    def col_block_sizes(self, copy=True):
        """
        Returns col-block sizes

        Returns
        -------
        numpy.narray

        """
        self._assert_broadcasted_sizes()
        if copy:
            return np.copy(self._bcol_lengths)
        return self._bcol_lengths

    def block_shapes(self):
        """
        Returns list with shapes of blocks in this BlockMatrix

        Notes
        -----
        For an MPIBlockMatrix with 2 block-rows and 2 block-cols
        this method returns [[Block_00.shape, Block_01.shape],[Block_10.shape, Block_11.shape]]

        Returns
        -------
        list

        """
        self._assert_broadcasted_sizes()
        bm, bn = self.bshape
        sizes = [list() for i in range(bm)]
        for i in range(bm):
            sizes[i] = list()
            for j in range(bn):
                shape = self._brow_lengths[i], self._bcol_lengths[j]
                sizes[i].append(shape)
        return sizes

    def has_empty_rows(self):
        """
        Indicates if the matrix has block-rows that are empty

        Returns
        -------
        boolean

        """
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def has_empty_cols(self):
        """
        Indicates if the matrix has block-columns that are empty

        Returns
        -------
        boolean

        """
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

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
        self._block_matrix.reset_bcol(jdx)
        self._bcol_lengths[jdx] = 0

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
        self._block_matrix.reset_brow(idx)
        self._brow_lengths[idx] = 0

    def copy(self):
        """
        Makes a copy of this MPIBlockMatrix

        Returns
        -------
        MPIBlockMatrix

        """
        m, n = self.bshape
        result = MPIBlockMatrix(m, n, self._rank_owner, self._mpiw)
        result._block_matrix = self._block_matrix.copy()
        result._need_broadcast_sizes = self._need_broadcast_sizes
        result._done_first_broadcast_sizes = self._done_first_broadcast_sizes
        result._brow_lengths = self._brow_lengths.copy()
        result._bcol_lengths = self._bcol_lengths.copy()
        return result

    def copy_structure(self):
        """
        Makes a copy of the structure of this MPIBlockMatrix. This proivides a
        light-weighted copy of each block in this MPIBlockMatrix. The blocks in the
        resulting matrix have the same shape as in the original matrices but not
        the same number of nonzeros.

        Returns
        -------
        MPIBlockMatrix

        """
        m, n = self.bshape
        result = MPIBlockMatrix(m, n, self._rank_owner, self._mpiw)
        result._block_matrix = self._block_matrix.copy_structure()
        result._need_broadcast_sizes = self._need_broadcast_sizes
        result._done_first_broadcast_sizes = self._done_first_broadcast_sizes
        result._brow_lengths = self._brow_lengths.copy()
        result._bcol_lengths = self._bcol_lengths.copy()
        return result

    # ToDo: need support for copy from and copy to

    def _assert_broadcasted_sizes(self):

        if not self._done_first_broadcast_sizes:
            assert not self._need_broadcast_sizes, \
                'First need to call broadcast_block_sizes()'
        else:
            assert not self._need_broadcast_sizes, \
                'Changes in structure. Need to recall broadcast_block_sizes()'

    # Note: this requires communication
    def _assert_correct_owners(self, root=0):

        rank = self._mpiw.Get_rank()
        num_processors = self._mpiw.Get_size()

        if num_processors == 1:
            return True

        local_owners = self._rank_owner.flatten()
        flat_size = self.bshape[0] * self.bshape[1]
        receive_data = None
        if rank == root:
            receive_data = np.empty(flat_size * num_processors, dtype=np.int64)
        self._mpiw.Gather(local_owners, receive_data, root=root)

        if rank == root:
            owners_in_processor = np.split(receive_data, num_processors)
            root_rank_owners = owners_in_processor[root]
            for i in range(flat_size):
                for k in range(num_processors):
                    if k != root:
                        if owners_in_processor[k][i] != root_rank_owners[i]:
                            return False
        return True

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.bshape)

    def __str__(self):
        msg = '{}{}\n'.format(self.__class__.__name__, self.bshape)
        for idx in range(self.bshape[0]):
            for jdx in range(self.bshape[1]):
                rank = self._rank_owner[idx, jdx] if self._rank_owner[idx, jdx] >= 0 else 'A'
                msg += '({}, {}): Owned by processor{}\n'.format(idx, jdx, rank)
        return msg

    def pprint(self, root=0):
        """Prints MPIBlockMatrix in pretty format"""
        self._assert_broadcasted_sizes()
        msg = self.__repr__() + '\n'
        num_processors = self._mpiw.Get_size()
        # figure out which ones are none
        local_mask = self._block_matrix._block_mask.flatten()
        receive_data = np.empty(num_processors * local_mask.size,
                                dtype=np.bool)

        self._mpiw.Allgather(local_mask, receive_data)
        all_masks = np.split(receive_data, num_processors)
        m, n = self.bshape
        matrix_maks = [mask.reshape(m, n) for mask in all_masks]

        global_mask = np.zeros((m, n), dtype=np.bool)
        for k in range(num_processors):
            for idx in range(m):
                for jdx in range(n):
                    global_mask[idx, jdx] += matrix_maks[k][idx, jdx]

        for idx in range(m):
            for jdx in range(n):
                rank = self._rank_owner[idx, jdx] if self._rank_owner[idx, jdx] >= 0 else 'A'
                row_size = self._brow_lengths[idx]
                col_size = self._bcol_lengths[jdx]
                is_none = '' if global_mask[idx, jdx] else '*'
                repn = 'Owned by {} Shape({},{}){}'.format(rank,
                                                           row_size,
                                                           col_size,
                                                           is_none)
                msg += '({}, {}): {}\n'.format(idx, jdx, repn)
        if self._mpiw.Get_rank() == root:
            print(msg)

    def __getitem__(self, item):

        block = self._block_matrix[item]
        owner = self._rank_owner[item]
        rank = self._mpiw.Get_rank()
        assert owner == rank or \
               owner < 0, \
               'Block {} not owned by processor {}'.format(item, rank)

        return block

    def __setitem__(self, key, value):

        assert not isinstance(key, slice), \
            'Slices not supported in MPIBlockMatrix'
        assert isinstance(key, tuple), \
            'Indices must be tuples (i,j)'

        idx, jdx = key
        assert idx >= 0 and \
               jdx >= 0, 'Indices must be positive'

        assert idx < self.bshape[0] and \
               jdx < self.bshape[1], 'Indices out of range'

        owner = self._rank_owner[key]
        rank = self._mpiw.Get_rank()
        assert owner == rank or \
               owner < 0, \
               'Block {} not owned by processor {}'.format(key, rank)

        # Flag broadcasting if needed
        if value is None:
            if self._block_matrix[key] is not None:
                if self._brow_lengths[idx] != 0 or self._bcol_lengths[jdx] != 0:
                    self._need_broadcast_sizes = True
        else:
            m, n  = value.shape
            if self._brow_lengths[idx] != m or self._bcol_lengths[jdx] != n:
                self._need_broadcast_sizes = True

        self._block_matrix[key] = value

    def __add__(self, other):

        # ToDo: this might not be needed
        self._assert_broadcasted_sizes()
        m, n = self.bshape
        result = self.copy_structure()

        rank = self._mpiw.Get_rank()

        if isinstance(other, MPIBlockMatrix):

            # ToDo: this might not be needed
            other._assert_broadcasted_sizes()

            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]
                if mat1 is not None and mat2 is not None:
                    result[i, j] = mat1 + mat2
                elif mat1 is not None and mat2 is None:
                    result[i, j] = mat1
                elif mat1 is None and mat2 is not None:
                    result[i, j] = mat2
                else:
                    result[i, j] = None
            return result

        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if isspmatrix(other):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):

        self._assert_broadcasted_sizes()
        m, n = self.bshape
        result = self.copy_structure()
        rank = self._mpiw.Get_rank()

        if isinstance(other, MPIBlockMatrix):

            # ToDo: this might not be needed
            other._assert_broadcasted_sizes()

            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]
                if mat1 is not None and mat2 is not None:
                    result[i, j] = mat1 - mat2
                elif mat1 is not None and mat2 is None:
                    result[i, j] = mat1
                elif mat1 is None and mat2 is not None:
                    result[i, j] = -mat2
                else:
                    result[i, j] = None
            return result

        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if isspmatrix(other):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __rsub__(self, other):

        self._assert_broadcasted_sizes()
        m, n = self.bshape
        result = self.copy_structure()

        rank = self._mpiw.Get_rank()

        if isinstance(other, MPIBlockMatrix):

            other._assert_broadcasted_sizes()
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
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

        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if isspmatrix(other):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __mul__(self, other):

        self._assert_broadcasted_sizes()
        m, n = self.bshape
        result = self.copy_structure()

        if isinstance(other, MPIBlockVector):

            other._assert_broadcasted_sizes()
            rank = self._mpiw.Get_rank()
            m, n = self.bshape
            assert n == other.nblocks, 'Dimension mismatch'
            assert not other.has_none, 'Block vector must not have none entries' # this check is expensive
            assert np.compress(self._row_type == MULTIPLE_OWNER,
                               self._row_type).size == 0, \
                'Matrix-vector multiply only supported for ' \
                'matrices with single rank owner in each row.' \
                'Call pynumero.matvec_multiply instead or modify matrix ownership'

            # Note: this relies on the assertion above
            row_rank_ownership = np.empty(m, dtype=np.int64)
            for i in range(m):
                row_rank_ownership[i] = -1
                if self._row_type[i] != ALL_OWN_IT:
                    for j in range(n):
                        owner = self._rank_owner[i, j]
                        if owner != row_rank_ownership[i]:
                            row_rank_ownership[i] = owner
                            break

            result = MPIBlockVector(m,
                                    row_rank_ownership,
                                    self._mpiw,
                                    block_sizes=self._brow_lengths.copy())

            # check same mpi spaces in matrix and vector
            owners_match = True
            for i in range(m):
                for j in range(n):
                    mat_owner = self._rank_owner[i, j]
                    vector_owner = other.rank_ownership[j]
                    if mat_owner != vector_owner:
                        if mat_owner >= 0 and vector_owner >= 0:
                            owners_match = False
                            break

            if owners_match:
                for i in range(m):
                    local_sum = np.zeros(self._brow_lengths[i])
                    for j in range(n):
                        mat_owner = self._rank_owner[i, j]
                        vector_owner = other.rank_ownership[j]
                        if (mat_owner == vector_owner and rank == mat_owner) or \
                            (mat_owner == rank and vector_owner < 0) or \
                            (vector_owner == rank and mat_owner < 0):
                            x = other[j]
                            if self[i, j] is not None:
                                local_sum += self[i, j] * x

                    row_owner = row_rank_ownership[i]
                    if row_owner < 0:
                        global_sum = self._mpiw.allreduce(local_sum, op=MPI.SUM)
                    else:
                        global_sum = self._mpiw.reduce(local_sum,
                                                       op=MPI.SUM,
                                                       root=row_owner)
                    if row_owner == rank  or row_owner < 0:
                        result[i] = global_sum
                return result
            else:
                if rank == 0:
                    msg = "Matrix-vector multiply with blocks in different MPI spaces is inefficient."
                    warn(msg, MPISpaceWarning)
                return self.__mul__(other.make_local_copy())

        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    result[i, j] = self[i, j] * other
            return result

        if isinstance(other, MPIBlockMatrix):
            raise NotImplementedError('Matrix-Matrix multiply not supported yet')
        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Matrix-Matrix multiply not supported yet')

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __rmul__(self, other):

        self._assert_broadcasted_sizes()
        m, n = self.bshape
        result = self.copy_structure()

        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    result[i, j] = self[i, j] * other
            return result

        if isinstance(other, MPIBlockVector):
            raise NotImplementedError('Vector-Matrix multiply not supported yet')
        if isinstance(other, BlockVector):
            raise NotImplementedError('Vector-Matrix multiply not supported yet')

        if isinstance(other, MPIBlockMatrix):
            raise NotImplementedError('Matrix-Matrix multiply not supported yet')
        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Matrix-Matrix multiply not supported yet')


        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __pow__(self, other):
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __truediv__(self, other):

        self._assert_broadcasted_sizes()
        m, n = self.bshape
        result = self.copy_structure()

        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    result[i, j] = self[i, j] / other
            return result
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __floordiv__(self, other):
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __iadd__(self, other):

        # ToDo: this might not be needed
        self._assert_broadcasted_sizes()
        m, n = self.bshape

        if isinstance(other, MPIBlockMatrix):
            # ToDo: this might not be needed
            other._assert_broadcasted_sizes()

            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]
                if mat1 is not None and mat2 is not None:
                    self[i, j] += mat2
                elif mat1 is None and mat2 is not None:
                    self[i, j] = mat2
            return self

        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if isspmatrix(other):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __isub__(self, other):

        # ToDo: this might not be needed
        self._assert_broadcasted_sizes()
        m, n = self.bshape

        if isinstance(other, MPIBlockMatrix):
            # ToDo: this might not be needed
            other._assert_broadcasted_sizes()

            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]
                if mat1 is not None and mat2 is not None:
                    self[i, j] -= mat2
                elif mat1 is None and mat2 is not None:
                    self[i, j] = -mat2
            return self

        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if isspmatrix(other):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __imul__(self, other):

        self._assert_broadcasted_sizes()
        m, n = self.bshape

        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    self[i, j] = self[i, j] * other
            return self
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __itruediv__(self, other):

        self._assert_broadcasted_sizes()
        m, n = self.bshape

        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    self[i, j] = self[i, j] / other
            return self
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __neg__(self):

        ii, jj = np.nonzero(self._owned_mask)
        for i, j in zip(ii, jj):
            if not self._block_matrix.is_empty_block(i, j):
                self[i, j] = -self[i, j]
        return self

    def __abs__(self):

        ii, jj = np.nonzero(self._owned_mask)
        for i, j in zip(ii, jj):
            if not self._block_matrix.is_empty_block(i, j):
                self[i, j] = self[i, j].__abs__()
        return self

    def __eq__(self, other):

        self._assert_broadcasted_sizes() # needed for the nones
        m, n = self.bshape
        result = self.copy_structure()

        if isinstance(other, MPIBlockMatrix):
            other._assert_broadcasted_sizes()
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]

                if mat1 is not None and mat2 is not None:
                    result[i, j] = mat1.__eq__(mat2)
                elif mat1 is not None and mat2 is None:
                    result[i, j] = mat1.__eq__(0.0)
                elif mat1 is None and mat2 is not None:
                    result[i, j] = mat2.__eq__(0.0)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    result[i, j] = mat.__eq__(0.0)
            return result

        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    result[i, j] = self[i, j].__eq__(other)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    result[i, j] = mat.__eq__(other)
            return result

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __ne__(self, other):

        self._assert_broadcasted_sizes() # needed for the nones
        m, n = self.bshape
        result = self.copy_structure()

        if isinstance(other, MPIBlockMatrix):
            other._assert_broadcasted_sizes()
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]

                if mat1 is not None and mat2 is not None:
                    result[i, j] = mat1.__ne__(mat2)
                elif mat1 is not None and mat2 is None:
                    result[i, j] = mat1.__ne__(0.0)
                elif mat1 is None and mat2 is not None:
                    result[i, j] = mat2.__ne__(0.0)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    result[i, j] = mat.__ne__(0.0)
            return result

        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    result[i, j] = self[i, j].__ne__(other)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    result[i, j] = mat.__ne__(other)
            return result

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __le__(self, other):

        self._assert_broadcasted_sizes() # needed for the nones
        m, n = self.bshape
        result = self.copy_structure()

        if isinstance(other, MPIBlockMatrix):
            other._assert_broadcasted_sizes()
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]

                if mat1 is not None and mat2 is not None:
                    result[i, j] = mat1.__le__(mat2)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    if mat1 is not None and mat2 is None:
                        result[i, j] = mat1.__le__(mat)
                    elif mat1 is None and mat2 is not None:
                        result[i, j] = mat2.__le__(mat)
                    else:
                        result[i, j] = mat.__le__(mat)
            return result
        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    result[i, j] = self[i, j].__le__(other)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    result[i, j] = mat.__le__(other)
            return result

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __lt__(self, other):

        self._assert_broadcasted_sizes() # needed for the nones
        m, n = self.bshape
        result = self.copy_structure()

        if isinstance(other, MPIBlockMatrix):

            other._assert_broadcasted_sizes()
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]

                if mat1 is not None and mat2 is not None:
                    result[i, j] = mat1.__lt__(mat2)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    if mat1 is not None and mat2 is None:
                        result[i, j] = mat1.__lt__(mat)
                    elif mat1 is None and mat2 is not None:
                        result[i, j] = mat2.__lt__(mat)
                    else:
                        result[i, j] = mat.__lt__(mat)
            return result

        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    result[i, j] = self[i, j].__lt__(other)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    result[i, j] = mat.__lt__(other)
            return result

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __ge__(self, other):

        self._assert_broadcasted_sizes() # needed for the nones
        m, n = self.bshape
        result = self.copy_structure()

        if isinstance(other, MPIBlockMatrix):
            other._assert_broadcasted_sizes()
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]

                if mat1 is not None and mat2 is not None:
                    result[i, j] = mat1.__ge__(mat2)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    if mat1 is not None and mat2 is None:
                        result[i, j] = mat1.__ge__(mat)
                    elif mat1 is None and mat2 is not None:
                        result[i, j] = mat2.__ge__(mat)
                    else:
                        result[i, j] = mat.__ge__(mat)
            return result
        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    result[i, j] = self[i, j].__ge__(other)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    result[i, j] = mat.__ge__(other)
            return result

        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __gt__(self, other):

        self._assert_broadcasted_sizes() # needed for the nones
        m, n = self.bshape
        result = self.copy_structure()

        if isinstance(other, MPIBlockMatrix):
            other._assert_broadcasted_sizes()
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockMatrices must be distributed in same processors'

            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                mat1 = self[i, j]
                mat2 = other[i, j]

                if mat1 is not None and mat2 is not None:
                    result[i, j] = mat1.__gt__(mat2)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    if mat1 is not None and mat2 is None:
                        result[i, j] = mat1.__gt__(mat)
                    elif mat1 is None and mat2 is not None:
                        result[i, j] = mat2.__gt__(mat)
                    else:
                        result[i, j] = mat.__gt__(mat)
            return result
        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        if np.isscalar(other):
            ii, jj = np.nonzero(self._owned_mask)
            for i, j in zip(ii, jj):
                if not self._block_matrix.is_empty_block(i, j):
                    result[i, j] = self[i, j].__gt__(other)
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = empty_matrix(nrows, ncols)
                    result[i, j] = mat.__gt__(other)
            return result
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

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
        self._assert_broadcasted_sizes()

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
        self._assert_broadcasted_sizes()

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
        Returns MPIBlockVector of column j

        Parameters
        ----------
        j: int
            Column index

        Returns
        -------
        pyomo.contrib.pynumero.sparse MPIBlockVector

        """
        # get size of the blocks to input in the vector
        # this implicitly checks that sizes have been broadcasted beforehand
        block_sizes = self.row_block_sizes()
        # get block column index
        bcol = self.get_block_column_index(j)
        # get rank ownership
        col_ownership = []
        bm, bn = self.bshape
        for i in range(bm):
             col_ownership.append(self._rank_owner[i, bcol])
        # create vector
        bv = MPIBlockVector(bm,
                           col_ownership,
                           self._mpiw,
                           block_sizes=block_sizes)

        # compute offset columns
        offset = 0
        if bcol > 0:
            cum_sum = self._bcol_lengths.cumsum()
            offset = cum_sum[bcol-1]

        # populate vector
        rank = self._mpiw.Get_rank()
        for row_bid, owner in enumerate(col_ownership):
            if rank == owner or owner<0:
                sub_matrix = self._block_matrix[row_bid, bcol]
                if self._block_matrix.is_empty_block(row_bid, bcol):
                    v = np.zeros(self._brow_lengths[row_bid])
                elif isinstance(sub_matrix, BaseBlockMatrix):
                    v = sub_matrix.getcol(j-offset)
                else:
                    # if it is sparse matrix transform array to vector
                    v = sub_matrix.getcol(j-offset).toarray().flatten()
                bv[row_bid] = v
        return bv

    def getrow(self, i):
        """
        Returns MPIBlockVector of column i

        Parameters
        ----------
        i: int
            Row index

        Returns
        -------
        pyomo.contrib.pynumero.sparse MPIBlockVector

        """
        # get size of the blocks to input in the vector
        # this implicitly checks that sizes have been broadcasted beforehand
        block_sizes = self.col_block_sizes()
        # get block column index
        brow = self.get_block_row_index(i)
        # get rank ownership
        row_ownership = []
        bm, bn = self.bshape
        for j in range(bn):
             row_ownership.append(self._rank_owner[brow, j])
        # create vector
        bv = MPIBlockVector(bn,
                            row_ownership,
                            self._mpiw,
                            block_sizes=block_sizes)
        # compute offset columns
        offset = 0
        if brow > 0:
            cum_sum = self._brow_lengths.cumsum()
            offset = cum_sum[brow-1]
        # populate vector
        rank = self._mpiw.Get_rank()
        for col_bid, owner in enumerate(row_ownership):
            if rank == owner or owner<0:
                sub_matrix = self._block_matrix[brow, col_bid]
                if self._block_matrix.is_empty_block(brow, col_bid):
                    v = np.zeros(self._bcol_lengths[col_bid])
                elif isinstance(sub_matrix, BaseBlockMatrix):
                    v = sub_matrix.getrow(i-offset)
                else:
                    # if it is sparse matrix transform array to vector
                    v = sub_matrix.getrow(i-offset).toarray().flatten()
                bv[col_bid] = v
        return bv

class MPIBlockSymMatrix(MPIBlockMatrix):
    """
    Parallel Structured Symmetric Matrix interface

    Parameters
    -------------------
    nbrowcols : int
             number of block-rows and block-columns in the matrix
    rank_ownership: array_like
                    integer 2D array that specifies the rank of process
                    owner of each block in the matrix. For blocks that are
                    owned by all processes the rank is -1. Blocks that are
                    None should be owned by all processes. Must be Symmetric.
    mpi_comm : communicator
    """
    def __init__(self,
                 nbrowcols,
                 rank_ownership,
                 mpi_comm,
                 block_sizes=None):

        super(MPIBlockSymMatrix, self).__init__(nbrowcols,
                                                nbrowcols,
                                                rank_ownership,
                                                mpi_comm,
                                                row_block_sizes=block_sizes,
                                                col_block_sizes=block_sizes)
        assert np.allclose(self.rank_ownership, self.rank_ownership.T), \
            'The rank ownership of a symmetric matrix must be symmetric. ' + \
            'If processor k owns (i, j) must also own (j, i)'

    def __setitem__(self, key, value):

        assert not isinstance(key, slice), \
            'Slices not supported in MPIBlockMatrix'
        assert isinstance(key, tuple), \
            'Indices must be tuples (i,j)'

        idx, jdx = key
        assert idx >= 0 and \
               jdx >= 0, 'Indices must be positive'

        assert idx < self.bshape[0] and \
               jdx < self.bshape[1], 'Indices out of range'

        owner = self._rank_owner[key]
        rank = self._mpiw.Get_rank()
        assert owner == rank or \
               owner < 0, \
               'Block {} not owned by processor {}'.format(key, rank)

        assert idx >= jdx, 'MPIBlockSymMatrix only sets lower triangular entries idx >= jdx'

        if idx == jdx:
            assert is_symmetric_sparse(value), 'Matrix is not symmetric'

        # Flag broadcasting if needed
        if value is None:
            if self._block_matrix[key] is not None:
                if self._brow_lengths[idx] != 0 or self._bcol_lengths[jdx] != 0:
                    self._need_broadcast_sizes = True
        else:
            m, n  = value.shape
            if self._brow_lengths[idx] != m or self._bcol_lengths[jdx] != n:
                self._need_broadcast_sizes = True

        self._block_matrix[key] = value
        self._block_matrix[jdx, idx] = value.T

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

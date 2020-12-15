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

from .mpi_block_vector import MPIBlockVector
from .block_vector import BlockVector
from .block_matrix import BlockMatrix, NotFullyDefinedBlockMatrixError
from .block_matrix import assert_block_structure as block_matrix_assert_block_structure
from .base_block import BaseBlockMatrix
from mpi4py import MPI
import numpy as np
from scipy.sparse import coo_matrix
import operator

__all__ = ['MPIBlockMatrix']


def assert_block_structure(mat):
    if mat.has_undefined_row_sizes() or mat.has_undefined_col_sizes():
        mat.broadcast_block_sizes()
    if mat.has_undefined_row_sizes() or mat.has_undefined_col_sizes():
        msg = 'Call MPIBlockMatrix.broadcast_block_sizes() first. '
        raise NotFullyDefinedBlockMatrixError(msg)


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
    _mpiw: MPI.Comm
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
    assert_correct_owners: bool
                           If True, then checks will be performed to ensure
                           that processor owners are consistent. This check
                           requires communication. If False, this check is 
                           skipped.
    """

    def __init__(self,
                 nbrows,
                 nbcols,
                 rank_ownership,
                 mpi_comm,
                 assert_correct_owners=False):

        shape = (nbrows, nbcols)
        self._block_matrix = BlockMatrix(nbrows, nbcols)
        self._mpiw = mpi_comm
        rank = self._mpiw.Get_rank()
        self._rank_owner = np.asarray(rank_ownership, dtype=np.int)
        self._owned_mask = np.bitwise_or(self._rank_owner == rank, self._rank_owner < 0)
        self._unique_owned_mask = self._rank_owner == rank

        assert self._rank_owner.ndim == 2, 'rank_ownership must be of size 2'

        # Note: this requires communication but is disabled when assertions
        # are turned off
        if assert_correct_owners:
            assert self._assert_correct_owners(), \
                'rank_owner must be the same in all processors'

        # make some of the pointers unmutable
        self._rank_owner.flags.writeable = False
        self._owned_mask.flags.writeable = False
        self._unique_owned_mask.flags.writeable = False

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
        assert_block_structure(self)
        return self._block_matrix.shape

    @property
    def nnz(self):
        """
        Returns total number of nonzero values in this matrix
        """
        local_nnz = 0
        rank = self._mpiw.Get_rank()
        block_indices = self._unique_owned_mask if rank != 0 else self._owned_mask

        # this is an easy and efficient way to loop though owned blocks
        ii, jj = np.nonzero(block_indices)
        for i, j in zip(ii, jj):
            if not self._block_matrix.is_empty_block(i, j):
                local_nnz += self._block_matrix.get_block(i, j).nnz

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
        Returns list of 2-tuples with inidices of blocks shared by all processors
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

    def get_row_size(self, row):
        return self._block_matrix.get_row_size(row)

    def get_col_size(self, col):
        return self._block_matrix.get_col_size(col)

    def set_row_size(self, row, size):
        self._block_matrix.set_row_size(row, size)

    def set_col_size(self, col, size):
        self._block_matrix.set_col_size(col, size)

    def is_row_size_defined(self, row):
        return self._block_matrix.is_row_size_defined(row)

    def is_col_size_defined(self, col):
        return self._block_matrix.is_col_size_defined(col)

    def get_block_mask(self, copy=True):
        return self._block_matrix.get_block_mask(copy=copy)

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

    def transpose(self, axes=None, copy=True):
        """
        Reverses the dimensions of the block matrix.

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
        MPIBlockMatrix with dimensions reversed
        """
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))
        if not copy:
            raise ValueError('MPIBlockMatrix only supports transpose with copy=True')

        m = self.bshape[0]
        n = self.bshape[1]
        result = MPIBlockMatrix(n, m, self._rank_owner.T, self._mpiw, assert_correct_owners=False)
        result._block_matrix = self._block_matrix.transpose()
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

    def to_local_array(self):
        """
        This method is only for testing/debugging

        Returns
        -------
        result: np.ndarray
        """
        assert_block_structure(self)
        local_result = self._block_matrix.copy_structure()
        rank = self._mpiw.Get_rank()
        block_indices = self._unique_owned_mask if rank != 0 else self._owned_mask

        ii, jj = np.nonzero(block_indices)
        for i, j in zip(ii, jj):
            if not self._block_matrix.is_empty_block(i, j):
                local_result.set_block(i, j, self.get_block(i, j))
        local_result = local_result.toarray()
        global_result = np.zeros(shape=self.shape, dtype=local_result.dtype)
        self._mpiw.Allreduce(local_result, global_result)
        return global_result

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
        must be called before running any operations with the MPIBlockMatrix.
        """
        rank = self._mpiw.Get_rank()
        num_processors = self._mpiw.Get_size()

        local_row_data = np.zeros(self.bshape[0], dtype=np.int64)
        local_col_data = np.zeros(self.bshape[1], dtype=np.int64)
        local_row_data.fill(-1)
        local_col_data.fill(-1)
        for row_ndx in range(self.bshape[0]):
            if self._block_matrix.is_row_size_defined(row_ndx):
                local_row_data[row_ndx] = self._block_matrix.get_row_size(row_ndx)
        for col_ndx in range(self.bshape[1]):
            if self._block_matrix.is_col_size_defined(col_ndx):
                local_col_data[col_ndx] = self._block_matrix.get_col_size(col_ndx)

        send_data = np.concatenate([local_row_data, local_col_data])

        receive_data = np.empty(num_processors * (self.bshape[0] + self.bshape[1]),
                                dtype=np.int64)
        self._mpiw.Allgather(send_data, receive_data)

        proc_dims = np.split(receive_data, num_processors)
        m, n = self.bshape

        brow_lengths = np.zeros(m, dtype=np.int64)
        bcol_lengths = np.zeros(n, dtype=np.int64)

        # check the rows
        for i in range(m):
            rows_length = set()
            for k in range(num_processors):
                row_sizes, col_sizes = np.split(proc_dims[k],
                                                [self.bshape[0]])
                rows_length.add(row_sizes[i])
            if len(rows_length) > 2:
                msg = 'Row {} has more than one dimension accross processors'.format(i)
                raise RuntimeError(msg)
            elif len(rows_length) == 2:
                if -1 not in rows_length:
                    msg = 'Row {} has more than one dimension accross processors'.format(i)
                    raise RuntimeError(msg)
                rows_length.remove(-1)
            elif -1 in rows_length:
                msg = 'The dimensions of block row {} were not defined in any process'.format(i)
                raise NotFullyDefinedBlockMatrixError(msg)

            # here rows_length must only have one element
            brow_lengths[i] = rows_length.pop()

        # check columns
        for i in range(n):
            cols_length = set()
            for k in range(num_processors):
                rows_sizes, col_sizes = np.split(proc_dims[k],
                                                 [self.bshape[0]])
                cols_length.add(col_sizes[i])
            if len(cols_length) > 2:
                msg = 'Column {} has more than one dimension accross processors'.format(i)
                raise RuntimeError(msg)
            elif len(cols_length) == 2:
                if -1 not in cols_length:
                    msg = 'Column {} has more than one dimension accross processors'.format(i)
                    raise RuntimeError(msg)
                cols_length.remove(-1)
            elif -1 in cols_length:
                msg = 'The dimensions of block column {} were not defined in any process'.format(i)
                raise NotFullyDefinedBlockMatrixError(msg)

            # here rows_length must only have one element
            bcol_lengths[i] = cols_length.pop()

        for row_ndx, row_size in enumerate(brow_lengths):
            self.set_row_size(row_ndx, row_size)
        for col_ndx, col_size in enumerate(bcol_lengths):
            self.set_col_size(col_ndx, col_size)

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
        assert_block_structure(self)
        return self._block_matrix.row_block_sizes(copy=copy)

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
        assert_block_structure(self)
        return self._block_matrix.col_block_sizes(copy=copy)

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
        assert_block_structure(self)
        return self._block_matrix.block_shapes()

    def has_undefined_row_sizes(self):
        """
        Indicates if the matrix has block-rows with undefined dimensions

        Returns
        -------
        bool

        """
        return self._block_matrix.has_undefined_row_sizes()

    def has_undefined_col_sizes(self):
        """
        Indicates if the matrix has block-columns with undefined dimensions

        Returns
        -------
        bool

        """
        return self._block_matrix.has_undefined_col_sizes()

    def reset_bcol(self, jdx):
        """
        Resets all blocks in selected column to None (0 nonzero entries)

        Parameters
        ----------
        jdx: integer
            column index to be reset

        Returns
        -------
        None

        """
        self._block_matrix.reset_bcol(jdx)

    def reset_brow(self, idx):
        """
        Resets all blocks in selected row to None (0 nonzero entries)

        Parameters
        ----------
        idx: integer
            row index to be reset

        Returns
        -------
        None

        """
        self._block_matrix.reset_brow(idx)

    def copy(self):
        """
        Makes a copy of this MPIBlockMatrix

        Returns
        -------
        MPIBlockMatrix

        """
        m, n = self.bshape
        result = MPIBlockMatrix(m, n, self._rank_owner, self._mpiw, assert_correct_owners=False)
        result._block_matrix = self._block_matrix.copy()
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
        result = MPIBlockMatrix(m, n, self._rank_owner, self._mpiw, assert_correct_owners=False)
        result._block_matrix = self._block_matrix.copy_structure()
        return result

    # ToDo: need support for copy from and copy to

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
            for k in range(num_processors):
                if k != root:
                    if not np.array_equal(owners_in_processor[k], root_rank_owners):
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
        assert_block_structure(self)
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
                row_size = self.get_row_size(idx)
                col_size = self.get_col_size(jdx)
                is_none = '' if global_mask[idx, jdx] else '*'
                repn = 'Owned by {} Shape({},{}){}'.format(rank,
                                                           row_size,
                                                           col_size,
                                                           is_none)
                msg += '({}, {}): {}\n'.format(idx, jdx, repn)
        if self._mpiw.Get_rank() == root:
            print(msg)

    def get_block(self, row, col):
        block = self._block_matrix.get_block(row, col)
        owner = self._rank_owner[row, col]
        rank = self._mpiw.Get_rank()
        assert owner == rank or \
               owner < 0, \
               'Block {} not owned by processor {}'.format((row, col), rank)

        return block

    def set_block(self, row, col, value):
        assert row >= 0 and \
               col >= 0, 'Indices must be positive'

        assert row < self.bshape[0] and \
               col < self.bshape[1], 'Indices out of range'

        owner = self._rank_owner[row, col]
        rank = self._mpiw.Get_rank()
        assert owner == rank or \
               owner < 0, \
               'Block {} not owned by processor {}'.format((row, col), rank)

        self._block_matrix.set_block(row, col, value)

    def __getitem__(self, item):
        raise NotImplementedError('MPIBlockMatrix does not support __getitem__.')

    def __setitem__(self, item, val):
        raise NotImplementedError('MPIBlockMatrix does not support __setitem__.')

    def _binary_operation_helper(self, other, operation):
        result = self.copy_structure()
        if isinstance(other, (MPIBlockMatrix, BlockMatrix)):
            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            if isinstance(other, MPIBlockMatrix):
                assert np.array_equal(self._rank_owner, other._rank_owner), \
                    'MPIBlockMatrices must be distributed in same processors'

            block_indices = np.bitwise_or(self.get_block_mask(copy=False), other.get_block_mask(copy=False))
            block_indices = np.bitwise_and(block_indices, self._owned_mask)
            ii, jj = np.nonzero(block_indices)
            for i, j in zip(ii, jj):
                mat1 = self.get_block(i, j)
                mat2 = other.get_block(i, j)
                if mat1 is not None and mat2 is not None:
                    result.set_block(i, j, operation(mat1, mat2))
                elif mat1 is not None and mat2 is None:
                    result.set_block(i, j, operation(mat1, 0))
                elif mat1 is None and mat2 is not None:
                    result.set_block(i, j, operation(0, mat2))
                else:
                    raise ValueError('This is unexpected. Please report to the developers.')
        elif np.isscalar(other):
            block_indices = np.bitwise_and(self.get_block_mask(copy=False), self._owned_mask)
            for i, j in zip(*np.nonzero(block_indices)):
                result.set_block(i, j, operation(self.get_block(i, j), other))
        else:
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        return result

    def _inplace_binary_operation_helper(self, other, operation):
        if isinstance(other, (MPIBlockMatrix, BlockMatrix)):
            assert operation in {operator.iadd, operator.isub}

            assert other.bshape == self.bshape, \
                'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)

            if isinstance(other, MPIBlockMatrix):
                assert np.array_equal(self._rank_owner, other._rank_owner), \
                    'MPIBlockMatrices must be distributed in same processors'

            block_indices = other.get_block_mask(copy=False)
            block_indices = np.bitwise_and(block_indices, self._owned_mask)
            ii, jj = np.nonzero(block_indices)
            for i, j in zip(ii, jj):
                mat1 = self.get_block(i, j)
                mat2 = other.get_block(i, j)
                if mat1 is not None and mat2 is not None:
                    mat1 = operation(mat1, mat2)
                    self.set_block(i, j, mat1)
                elif mat1 is None and mat2 is not None:
                    if operation is operator.iadd:
                        sub_res = mat2.copy()
                    else:
                        sub_res = -mat2
                    self.set_block(i, j, sub_res)
                else:
                    raise RuntimeError('Please report this to the developers.')
        elif np.isscalar(other):
            block_indices = np.bitwise_and(self.get_block_mask(copy=False), self._owned_mask)
            for i, j in zip(*np.nonzero(block_indices)):
                blk = self.get_block(i, j)
                blk = operation(blk, other)
                self.set_block(i, j, blk)
        else:
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')
        return self

    def __add__(self, other):
        return self._binary_operation_helper(other, operator.add)

    def __radd__(self, other):  # other + self
        return self._binary_operation_helper(other, operator.add)

    def __sub__(self, other):
        return self._binary_operation_helper(other, operator.sub)

    def __rsub__(self, other):
        return (-self) + other

    def _get_block_vector_for_dot_product(self, x):
        if isinstance(x, MPIBlockVector):
            """
            Consider a non-empty block m_{i, j} from the mpi block matrix with rank owner r_m and the 
            corresponding block v_{j} from the mpi block vector with rank owner r_v. There are 4 cases:
              1. r_m = r_v
                 In this case, all is good.
              2. r_v = -1
                 In this case, all is good.
              3. r_m = -1 and r_v = 0
                 All is good
              4. If none of the above cases hold, then v_{j} must be broadcast
            """
            n_block_rows, n_block_cols = self.bshape
            blocks_needing_broadcast = np.zeros(n_block_cols, dtype=np.int64)  # a value > 0 means broadcast
            x_rank_ownership = x.rank_ownership
            comm = self._mpiw
            rank = comm.Get_rank()

            if rank == 0:
                block_indices = self._owned_mask
            else:
                block_indices = self._unique_owned_mask
            block_indices = np.bitwise_and(block_indices, self.get_block_mask(copy=False))
            for i, j in zip(*np.nonzero(block_indices)):
                r_m = self._rank_owner[i, j]
                r_v = x_rank_ownership[j]
                if r_m == r_v:
                    pass
                elif r_v == -1:
                    pass
                elif r_m == -1 and r_v == 0:
                    pass
                else:
                    blocks_needing_broadcast[j] = 1

            global_blocks_needing_broadcast = np.zeros(n_block_cols, dtype=np.int64)
            comm.Allreduce(blocks_needing_broadcast, global_blocks_needing_broadcast)
            indices_needing_broadcast = np.nonzero(global_blocks_needing_broadcast)[0]
            if len(indices_needing_broadcast) == 0:
                return x
            else:
                res = BlockVector(n_block_cols)
                for ndx in np.nonzero(x.ownership_mask)[0]:
                    res.set_block(ndx, x.get_block(ndx))
                for j in indices_needing_broadcast:
                    j_owner = x_rank_ownership[j]
                    if rank == j_owner:
                        j_size = x.get_block_size(j)
                    else:
                        j_size = None
                    j_size = comm.bcast(j_size, j_owner)
                    if rank == j_owner:
                        data = x.get_block(j).flatten()
                    else:
                        data = np.empty(j_size)
                    comm.Bcast(data, j_owner)
                    res.set_block(j, data)
                return res
        elif isinstance(x, BlockVector):
            return x
        elif isinstance(x, np.ndarray):
            y = BlockVector(self.bshape[1])
            for ndx, size in enumerate(self.col_block_sizes(copy=False)):
                y.set_block(ndx, np.zeros(size))
            y.copyfrom(x)
            return y
        else:
            raise NotImplementedError('Dot product is not yet supported for MPIBlockMatrix*'+str(type(x)))

    def _block_vector_multiply(self, x):
        """
        In this method, we assume that we can access the correct blocks from x. This means that
        _get_block_vector_for_dot_product should be called first.

        For a given block row, if there are multiple non-empty blocks with different rank owners,
        then the result for that row is owned by all, and we need to do an Allreduce. Otherwise the
        rank owner of the resulting block is the rank owner of the non-empty blocks in the block row.
        """
        n_block_rows, n_block_cols = self.bshape
        comm = self._mpiw
        rank = comm.Get_rank()

        blocks_that_need_reduced = np.zeros(n_block_rows, dtype=np.int64)
        res_rank_owner = np.zeros(n_block_rows, dtype=np.int64)
        for i, j in zip(*np.nonzero(self._block_matrix._block_mask)):
            blocks_that_need_reduced[i] = 1
            res_rank_owner[i] = self._rank_owner[i, j]

        # we need some special handling to determine the owner of empty rows
        local_empty_rows = self._block_matrix._block_mask.any(axis=1)
        local_empty_rows = np.array(local_empty_rows, dtype=np.int64)
        global_empty_rows = np.empty(local_empty_rows.size, dtype=np.int64)
        comm.Allreduce(local_empty_rows, global_empty_rows)
        empty_rows = np.nonzero(global_empty_rows == 0)[0]

        global_blocks_that_need_reduced = np.zeros(n_block_rows, dtype=np.int64)
        comm.Allreduce(blocks_that_need_reduced, global_blocks_that_need_reduced)
        block_indices_that_need_reduced = np.nonzero(global_blocks_that_need_reduced > 1)[0]
        global_res_rank_owner = np.zeros(n_block_rows, dtype=np.int64)
        comm.Allreduce(res_rank_owner, global_res_rank_owner)
        global_res_rank_owner[block_indices_that_need_reduced] = -1
        for ndx in empty_rows:
            row_owners = set(self._rank_owner[ndx, :])
            if len(row_owners) == 1:
                global_res_rank_owner[ndx] = row_owners.pop()
            elif len(row_owners) == 2 and -1 in row_owners:
                tmp = row_owners.pop()
                if tmp == -1:
                    global_res_rank_owner[ndx] = row_owners.pop()
                else:
                    global_res_rank_owner[ndx] = tmp
            else:
                global_res_rank_owner[ndx] = -1

        res = MPIBlockVector(nblocks=n_block_rows,
                             rank_owner=global_res_rank_owner,
                             mpi_comm=comm,
                             assert_correct_owners=False)
        for ndx in np.nonzero(res.ownership_mask)[0]:
            res.set_block(ndx, np.zeros(self.get_row_size(ndx)))
        if rank == 0:
            block_indices = self._owned_mask
        else:
            block_indices = self._unique_owned_mask
        block_indices = np.bitwise_and(block_indices, self._block_matrix._block_mask)
        for row_ndx, col_ndx in zip(*np.nonzero(block_indices)):
            res_blk = res.get_block(row_ndx)
            tmp = self.get_block(row_ndx, col_ndx) * x.get_block(col_ndx)
            tmp += res_blk
            res.set_block(row_ndx, tmp)

        for ndx in block_indices_that_need_reduced:
            local = res.get_block(ndx)
            flat_local = local.flatten()
            flat_global = np.zeros(flat_local.size)
            comm.Allreduce(flat_local, flat_global)
            if isinstance(local, BlockVector):
                local.copyfrom(flat_global)
            else:
                res.set_block(ndx, flat_global)

        return res

    def __mul__(self, other):
        """
        When doing A*B with numpy arrays, element-by-element multiplication is done. However, when doing
        A*B with scipy sparse matrices, a matrix-matrix dot product is performed. We are following the
        scipy sparse matrix API.
        """
        if np.isscalar(other):
            return self._binary_operation_helper(other, operator.mul)
        else:
            x = self._get_block_vector_for_dot_product(other)
            return self._block_vector_multiply(x)

    def __rmul__(self, other):
        """
        When doing A*B with numpy arrays, element-by-element multiplication is done. However, when doing
        A*B with scipy sparse matrices, a matrix-matrix dot product is performed. We are following the
        scipy sparse matrix API.
        """
        if np.isscalar(other):
            return self._binary_operation_helper(other, operator.mul)

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
        if np.isscalar(other):
            return self._binary_operation_helper(other, operator.truediv)
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __rtruediv__(self, other):
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __floordiv__(self, other):
        if np.isscalar(other):
            return self._binary_operation_helper(other, operator.floordiv)
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __rfloordiv__(self, other):
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __iadd__(self, other):
        return self._inplace_binary_operation_helper(other, operator.iadd)

    def __isub__(self, other):
        return self._inplace_binary_operation_helper(other, operator.isub)

    def __imul__(self, other):
        if np.isscalar(other):
            return self._inplace_binary_operation_helper(other, operator.imul)
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __itruediv__(self, other):
        if np.isscalar(other):
            return self._inplace_binary_operation_helper(other, operator.itruediv)
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __neg__(self):
        result = self.copy_structure()
        block_indices = np.bitwise_and(self.get_block_mask(copy=False), self._owned_mask)
        for i, j in zip(*np.nonzero(block_indices)):
            result.set_block(i, j, -self.get_block(i, j))
        return result

    def __abs__(self):
        result = self.copy_structure()
        block_indices = np.bitwise_and(self.get_block_mask(copy=False), self._owned_mask)
        for i, j in zip(*np.nonzero(block_indices)):
            result.set_block(i, j, abs(self.get_block(i, j)))
        return result

    def _comparison_helper(self, operation, other):
        assert_block_structure(self)
        result = self.copy_structure()

        if isinstance(other, MPIBlockMatrix):
            assert other.bshape == self.bshape, 'dimension mismatch {} != {}'.format(self.bshape, other.bshape)
            assert np.array_equal(self.rank_ownership, other.rank_ownership), 'MPIBlockMatrices must be distributed in ' \
                                                                              'the same processors'

            for i, j in zip(*np.nonzero(self.ownership_mask)):
                mat1 = self.get_block(i, j)
                mat2 = other.get_block(i, j)

                if mat1 is not None and mat2 is not None:
                    result.set_block(i, j, operation(mat1, mat2))
                else:
                    nrows = self.get_row_size(i)
                    ncols = self.get_col_size(j)
                    mat = coo_matrix((nrows, ncols))
                    if mat1 is not None:
                        result.set_block(i, j, operation(mat1, mat))
                    elif mat2 is not None:
                        result.set_block(i, j, operation(mat, mat2))
                    else:
                        result.set_block(i, j, operation(mat, mat))
            return result
        elif np.isscalar(other):
            for i, j in zip(*np.nonzero(self.ownership_mask)):
                if not self._block_matrix.is_empty_block(i, j):
                    result.set_block(i, j, operation(self.get_block(i, j), other))
                else:
                    nrows = self.get_row_size(i)
                    ncols = self.get_col_size(j)
                    mat = coo_matrix((nrows, ncols))
                    result.set_block(i, j, operation(mat, other))
            return result
        else:
            raise NotImplementedError('Operation not supported by MPIBlockMatrix')

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
        assert_block_structure(self)

        bm, bn = self.bshape
        # get cummulative sum of block sizes
        cum = self.col_block_sizes(copy=False).cumsum()
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
        assert_block_structure(self)

        bm, bn = self.bshape
        # get cummulative sum of block sizes
        cum = self.row_block_sizes(copy=False).cumsum()
        assert index >= 0, 'index out of bounds'
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
                            assert_correct_owners=False)

        # compute offset columns
        offset = 0
        if bcol > 0:
            cum_sum = self.col_block_sizes(copy=False).cumsum()
            offset = cum_sum[bcol-1]

        # populate vector
        rank = self._mpiw.Get_rank()
        for row_bid, owner in enumerate(col_ownership):
            if rank == owner or owner < 0:
                sub_matrix = self._block_matrix.get_block(row_bid, bcol)
                if self._block_matrix.is_empty_block(row_bid, bcol):
                    v = np.zeros(self.get_row_size(row_bid))
                elif isinstance(sub_matrix, BaseBlockMatrix):
                    v = sub_matrix.getcol(j-offset)
                else:
                    # if it is sparse matrix transform array to vector
                    v = sub_matrix.getcol(j-offset).toarray().flatten()
                bv.set_block(row_bid, v)
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
                            assert_correct_owners=False)
        # compute offset columns
        offset = 0
        if brow > 0:
            cum_sum = self.row_block_sizes(copy=False).cumsum()
            offset = cum_sum[brow-1]
        # populate vector
        rank = self._mpiw.Get_rank()
        for col_bid, owner in enumerate(row_ownership):
            if rank == owner or owner<0:
                sub_matrix = self._block_matrix.get_block(brow, col_bid)
                if self._block_matrix.is_empty_block(brow, col_bid):
                    v = np.zeros(self.get_col_size(col_bid))
                elif isinstance(sub_matrix, BaseBlockMatrix):
                    v = sub_matrix.getrow(i-offset)
                else:
                    # if it is sparse matrix transform array to vector
                    v = sub_matrix.getrow(i-offset).toarray().flatten()
                bv.set_block(col_bid, v)
        return bv

    @staticmethod
    def fromBlockMatrix(block_matrix, rank_ownership, mpi_comm, assert_correct_owners=False):
        """
        Creates a parallel MPIBlockMatrix from blockmatrix

        Parameters
        ----------
        block_matrix: BlockMatrix
            The block matrix to use to create the MPIBlockMatrix
        rank_ownership: array_like
            2D-array with processor ownership of each block. A block can be own by a
            single processor or by all processors. Blocks own by all processors have
            ownership -1. Blocks own by a single processor have ownership rank. where
            rank=MPI.COMM_WORLD.Get_rank()
        mpi_comm: MPI communicator
            An MPI communicator. Tyically MPI.COMM_WORLD
        """
        block_matrix_assert_block_structure(block_matrix)

        # create mpi matrix
        bm, bn = block_matrix.bshape
        mat = MPIBlockMatrix(bm,
                             bn,
                             rank_ownership,
                             mpi_comm,
                             assert_correct_owners=assert_correct_owners)

        # populate matrix
        for i in range(bm):
            mat.set_row_size(i, block_matrix.get_row_size(i))

        for j in range(bn):
            mat.set_col_size(j, block_matrix.get_col_size(j))

        for i, j in mat.owned_blocks:
            mat.set_block(i, j, block_matrix.get_block(i, j))

        return mat

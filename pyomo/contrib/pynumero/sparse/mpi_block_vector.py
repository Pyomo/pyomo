#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.contrib.pynumero.sparse import BlockVector
from .base_block import BaseBlockVector
from .block_vector import NotFullyDefinedBlockVectorError
from .block_vector import assert_block_structure as block_vector_assert_block_structure
from mpi4py import MPI
import numpy as np
import operator

__all__ = ['MPIBlockVector']


def assert_block_structure(vec):
    if vec.has_none:
        msg = 'Call MPIBlockVector.broadcast_block_sizes() first.'
        raise NotFullyDefinedBlockVectorError(msg)


class MPIBlockVector(np.ndarray, BaseBlockVector):
    """
    Parallel structured vector interface. This interface can be used to
    perform parallel operations on vectors composed by vectors. The main
    idea is to allocate vectors in different processors and make the corresponding
    parallel calls when necessary.

    Attributes
    ----------
    _rank_owner: numpy.ndarray
        1D-array with processor ownership of each block. A block can be own by a
        single processor or by all processors. Blocks own by all processors have
        ownership -1. Blocks own by a single processor have ownership rank. where
        rank=MPI.COMM_WORLD.Get_rank()
    _mpiw: MPI communicator
        A communicator from the MPI space. Typically MPI.COMM_WORLD
    _block_vector: BlockVector
        Internal BlockVector. Blocks that belong to this processor are stored
        in _block_vector. Blocks that do not belong to this proceesor are empty
        and store as numpy.zeros(0)
    _owned_mask: numpy.ndarray bool
        1D-array that indicates if a block belongs to this processor. While
        _rank_owner tells which processor(s) owns each block, _owned_mask tells
        if a block is owned by this processor. Blocks that are owned by everyone
        (i.e. ownership = -1) are True in _owned_mask
    _owned_blocks: numpy.ndarray
        1D-array with block indices owned by this processor. This includes blocks
        with ownership -1.
    _unique_owned_blocks: numpy.ndarray
        1D-array with block indices owned only by this processor. This does not
        include blocks with ownership -1.
    _brow_lengths: numpy.ndarray
        1D-Array of size nblocks that specifies the length of each entry
        in the MPIBlockVector. This is the same accross all processors.
    _undefined_brows: set
        A set of block indices for which the blocks are still None (i.e., the dimensions
        have not yet ben set). Operations with BlockVectors require all entries to be
        different than None.

    Notes
    ------
    This is the parallel implementation of pyomo.contrib.pynumero.sparse.BlockVector

    Parameters
    -------------------
    nblocks: int
        number of blocks contained in the block vector
    rank_owner: array_like
        Array_like of size nblocks. Each entry defines ownership of each block.
        There are two types of ownership. Block that are owned by all processor,
        and blocks owned by a single processor. If a block is owned by all
        processors then its ownership is -1. Otherwise, if a block is owned by
        a single processor, then its ownership is equal to the rank of the
        processor.
    mpi_com: MPI communicator
        An MPI communicator. Tyically MPI.COMM_WORLD
    """

    def __new__(cls, nblocks, rank_owner, mpi_comm, assert_correct_owners=False):

        assert isinstance(nblocks, int)
        assert len(rank_owner) == nblocks

        blocks = [None for i in range(nblocks)]
        arr = np.asarray(blocks, dtype='object')
        obj = arr.view(cls)

        obj._rank_owner = np.array([i for i in rank_owner])
        obj._mpiw = mpi_comm
        obj._block_vector = BlockVector(nblocks)

        rank = obj._mpiw.Get_rank()
        comm_size = obj._mpiw.Get_size()
        assert np.all(obj._rank_owner < comm_size)

        # Determine which blocks are owned by this processor
        obj._owned_mask = np.bitwise_or(obj._rank_owner == rank, obj._rank_owner < 0)
        unique_owned_mask = obj._rank_owner == rank
        obj._unique_owned_blocks = unique_owned_mask.nonzero()[0]
        obj._owned_blocks = obj._owned_mask.nonzero()[0]

        # containers that facilitate looping
        obj._brow_lengths = np.empty(nblocks, dtype=np.float64)
        obj._brow_lengths.fill(np.nan)
        obj._undefined_brows = set(obj._owned_blocks)

        # make some pointers unmutable. These arrays don't change after
        # MPIBlockVector has been created
        obj._rank_owner.flags.writeable = False
        obj._owned_blocks.flags.writeable = False
        obj._owned_mask.flags.writeable = False
        obj._unique_owned_blocks.flags.writeable = False

        obj._broadcasted = False

        return obj

    def __init__(self, nblocks, rank_owner, mpi_comm, assert_correct_owners=False):
        # Note: this requires communication but is disabled when assertions
        # are turned off
        if assert_correct_owners:
            assert self._assert_correct_owners(), \
                'rank_owner must be the same in all processors'

    def __array_prepare__(self, out_arr, context=None):
        return super(MPIBlockVector, self).__array_prepare__(self, out_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        return super(MPIBlockVector, self).__array_wrap__(self, out_arr, context)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Runs ufuncs speciallizations to MPIBlockVector"""
        # functions that take one vector
        unary_funcs = [np.log10, np.sin, np.cos, np.exp, np.ceil,
                       np.floor, np.tan, np.arctan, np.arcsin,
                       np.arccos, np.sinh, np.cosh, np.abs,
                       np.tanh, np.arccosh, np.arcsinh, np.arctanh,
                       np.fabs, np.sqrt, np.log, np.log2, np.absolute,
                       np.isfinite, np.isinf, np.isnan, np.log1p,
                       np.logical_not, np.expm1, np.exp2, np.sign,
                       np.rint, np.square, np.positive, np.negative,
                       np.rad2deg, np.deg2rad, np.conjugate, np.reciprocal,
                       np.signbit]
        # functions that take two vectors
        binary_funcs = [np.add, np.multiply, np.divide, np.subtract,
                        np.greater, np.greater_equal, np.less, np.less_equal,
                        np.not_equal, np.maximum, np.minimum, np.fmax,
                        np.fmin, np.equal, np.logical_and,
                        np.logical_or, np.logical_xor, np.logaddexp,
                        np.logaddexp2, np.remainder, np.heaviside,
                        np.hypot]

        outputs = kwargs.pop('out', None)
        if outputs is not None:
            raise NotImplementedError(str(ufunc) + ' cannot be used with MPIBlockVector if the out keyword argument is given.')

        if ufunc in unary_funcs:
            results = self._unary_operation(ufunc, method, *inputs, **kwargs)
            return results
        elif ufunc in binary_funcs:
            results = self._binary_operation(ufunc, method, *inputs, **kwargs)
            return results
        else:
            raise NotImplementedError(str(ufunc) + "not supported for MPIBlockVector")

    def _unary_operation(self, ufunc, method, *args, **kwargs):
        """Run recursion to perform unary_funcs on MPIBlockVector"""
        # ToDo: deal with out
        x = args[0]

        if isinstance(x, MPIBlockVector):
            rank = self._mpiw.Get_rank()
            v = x.copy_structure()
            for i in self._owned_blocks:
                _args = [x.get_block(i)] + [args[j] for j in range(1, len(args))]
                v.set_block(i, self._unary_operation(ufunc, method, *_args, **kwargs))
            return v
        elif isinstance(x, BlockVector):
            v = BlockVector(x.nblocks)
            for i in range(x.nblocks):
                _args = [x.get_block(i)] + [args[j] for j in range(1, len(args))]
                v.set_block(i, self._unary_operation(ufunc, method, *_args, **kwargs))
            return v
        elif type(x) == np.ndarray:
            return super(MPIBlockVector, self).__array_ufunc__(ufunc, method,
                                                               *args, **kwargs)
        else:
            raise NotImplementedError()

    def _binary_operation(self, ufunc, method, *args, **kwargs):
        """Run recursion to perform binary_funcs on MPIBlockVector"""
        # ToDo: deal with out
        x1 = args[0]
        x2 = args[1]
        if isinstance(x1, MPIBlockVector) and isinstance(x2, MPIBlockVector):

            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(x1._rank_owner, x2._rank_owner) or self._mpiw.Get_size() == 1, msg
            assert x1._mpiw == x2._mpiw, 'Need to have same communicator'

            res = x1.copy_structure()
            for i in x1._owned_blocks:
                _args = [x1.get_block(i)] + [x2.get_block(i)] + [args[j] for j in range(2, len(args))]
                res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
            return res
        elif isinstance(x1, BlockVector) and isinstance(x2, MPIBlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(x1, MPIBlockVector) and isinstance(x2, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(x1, MPIBlockVector) and np.isscalar(x2):
            res = x1.copy_structure()
            for i in x1._owned_blocks:
                _args = [x1.get_block(i)] + [x2] + [args[j] for j in range(2, len(args))]
                res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
            return res
        elif isinstance(x2, MPIBlockVector) and np.isscalar(x1):
            res = x2.copy_structure()
            for i in x2._owned_blocks:
                _args = [x1] + [x2.get_block(i)] + [args[j] for j in range(2, len(args))]
                res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
            return res
        elif isinstance(x1, MPIBlockVector) and type(x2)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(x2, MPIBlockVector) and type(x1)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
            # this will take care of blockvector and ndarrays
            return self._block_vector.__array_ufunc__(ufunc, method, *args, **kwargs)
        elif (type(x1)==BlockVector or np.isscalar(x1)) and (type(x2)==BlockVector or np.isscalar(x2)):
            return self._block_vector.__array_ufunc__(ufunc, method, *args, **kwargs)
        elif (type(x1)==np.ndarray or np.isscalar(x1)) and (type(x2)==np.ndarray or np.isscalar(x2)):
            return super(MPIBlockVector, self).__array_ufunc__(ufunc, method,
                                                               *args, **kwargs)
        else:
            raise NotImplementedError()

    @property
    def nblocks(self):
        """
        Returns the number of blocks.
        """
        return self._block_vector.nblocks

    @property
    def bshape(self):
        """
        Returns the number of blocks in this MPIBlockVector in a tuple.
        """
        return self.nblocks,

    @property
    def shape(self):
        """
        Returns total number of elements in the MPIBlockVector
        """
        return (self.size,)

    @property
    def size(self):
        """
        Returns total number of elements in this MPIBlockVector
        """
        assert_block_structure(self)
        comm = self._mpiw
        rank = comm.Get_rank()
        if rank == 0:
            indices = self._owned_blocks
        else:
            indices = self._unique_owned_blocks
        local_size = np.sum(self._brow_lengths[indices])
        size = comm.allreduce(local_size)
        assert int(size) == size
        return int(size)

    @property
    def ndim(self):
        """
        Returns dimension of this MPIBlockVector
        """
        return 1

    @property
    def has_none(self):
        """
        Returns True if block vector has none entry
        """
        return len(self._undefined_brows) != 0

    @property
    def owned_blocks(self):
        """
        Returns list with inidices of blocks owned by this processor.
        """
        return self._owned_blocks

    @property
    def shared_blocks(self):
        """
        Returns list with inidices of blocks shared by all processors
        """
        return np.array([i for i in range(self.nblocks) if self._rank_owner[i] < 0])

    @property
    def rank_ownership(self):
        """
        Returns 1D-Array with processor ranks that own each block. The ownership
        of blocks that are owned by all processors is -1.
        """
        return self._rank_owner

    @property
    def ownership_mask(self):
        """
        Returns boolean 1D-Array that indicates which blocks are owned by
        this processor
        """
        return self._owned_mask

    @property
    def mpi_comm(self):
        """Returns MPI communicator"""
        return self._mpiw

    def is_broadcasted(self):
        return self._broadcasted

    def block_sizes(self, copy=True):
        """
        Returns 1D-Array with sizes of individual blocks in this MPIBlockVector
        """
        if not self._broadcasted:
            self.broadcast_block_sizes()
        if copy:
            return self._brow_lengths.copy()
        return self._brow_lengths

    def get_block_size(self, ndx):
        res = self._brow_lengths[ndx]
        if np.isnan(res):
            raise NotFullyDefinedBlockVectorError('The dimensions of the requested block are not defined.')
        res = int(res)
        return res

    def _set_block_size(self, ndx, size):
        if ndx in self._undefined_brows:
            self._undefined_brows.remove(ndx)
            self._brow_lengths[ndx] = size
        else:
            if self._brow_lengths[ndx] != size:
                raise ValueError('Incompatible dimensions for block {ndx}; '
                                 'got {got}; expected {exp}'.format(ndx=ndx,
                                                                    got=size,
                                                                    exp=self._brow_lengths[ndx]))

    # Note: this operation requires communication
    def broadcast_block_sizes(self):
        """
        Send sizes of all blocks to all processors. After this method is called
        this MPIBlockVector knows it's dimensions across all blocks. This method
        must be called before running any operations with the MPIBlockVector.
        """
        assert_block_structure(self)
        rank = self._mpiw.Get_rank()
        num_processors = self._mpiw.Get_size()

        local_length_data = np.empty(self.nblocks, dtype=np.int64)
        local_length_data.fill(-1)
        for ndx in self.owned_blocks:
            if ndx in self._undefined_brows:
                raise NotFullyDefinedBlockVectorError('Block {ndx} is owned by rank {rank}, '
                                                      'but the dimensions for block {ndx} '
                                                      'have not yet been specified in rank {rank}. '
                                                      'Please specify all owned blocks.'.format(ndx=ndx,
                                                                                                rank=rank))
            local_length_data[ndx] = self.get_block_size(ndx)
        receive_data = np.empty(num_processors * self.nblocks, dtype=np.int64)
        self._mpiw.Allgather(local_length_data, receive_data)

        proc_dims = np.split(receive_data, num_processors)

        for i in range(self.nblocks):
            block_length = set()
            for k in range(num_processors):
                processor_sizes = proc_dims[k]
                block_length.add(processor_sizes[i])
            if len(block_length) > 2:
                msg = 'Block {} has more than one dimension accross processors'.format(i)
                raise RuntimeError(msg)
            elif len(block_length) == 2:
                if -1 not in block_length:
                    msg = 'Block {} has more than one dimension accross processors'.format(i)
                    raise RuntimeError(msg)
                block_length.remove(-1)
            elif -1 in block_length:
                msg = 'The dimension of block {} was not specified in any process'.format(i)

            # here block_length must only have one element
            self._brow_lengths[i] = block_length.pop()

        self._brow_lengths = np.asarray(self._brow_lengths, dtype=np.int64)
        self._broadcasted = True

    def finalize_block_sizes(self, broadcast=True, block_sizes=None):
        """
        Only set broadcast=False if you know what you are doing!

        Parameters
        ----------
        broadcast: bool
        block_sizes: None or np.ndarray
        """
        if broadcast:
            self.broadcast_block_sizes()
        else:
            self._undefined_brows = set()
            self._brow_lengths = block_sizes
            self._broadcasted = True

    # Note: this requires communication but is only run in __new__
    def _assert_correct_owners(self, root=0):

        rank = self._mpiw.Get_rank()
        num_processors = self._mpiw.Get_size()

        if num_processors == 1:
            return True

        local_owners = self._rank_owner.copy()
        receive_data = None
        if rank == root:
            receive_data = np.empty(self.nblocks * num_processors,
                                    dtype=np.int64)

        self._mpiw.Gather(local_owners, receive_data, root=root)

        if rank == root:
            owners_in_processor = np.split(receive_data, num_processors)
            root_rank_owners = owners_in_processor[root]
            for i in range(self.nblocks):
                for k in range(num_processors):
                    if k != root:
                        if owners_in_processor[k][i] != root_rank_owners[i]:
                            return False
        return True

    def all(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.
        """
        assert out is None, 'Out keyword not supported'
        assert_block_structure(self)
        local = 1
        for i in self._owned_blocks:
            local *= self._block_vector.get_block(i).all()

        return bool(self._mpiw.allreduce(local, op=MPI.PROD))

    def any(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.
        """
        assert out is None, 'Out keyword not supported'
        assert_block_structure(self)
        local = 0
        for i in self._owned_blocks:
            local += self._block_vector.get_block(i).any()

        return bool(self._mpiw.allreduce(local, op=MPI.SUM))

    def min(self, axis=None, out=None, keepdims=False):
        """
        Returns the smallest value stored in the vector
        """
        assert out is None, 'Out keyword not supported'
        assert_block_structure(self)
        local_min = np.inf
        for i in self._owned_blocks:
            lmin = self._block_vector.get_block(i).min()
            if lmin <= local_min:
                local_min = lmin
        return self._mpiw.allreduce(local_min, op=MPI.MIN)

    def max(self, axis=None, out=None, keepdims=False):
        """
        Returns the largest value stored in this MPIBlockVector
        """
        assert out is None, 'Out keyword not supported'
        assert_block_structure(self)
        local_max = -np.inf
        for i in self._owned_blocks:
            lmax = self._block_vector.get_block(i).max()
            if lmax >= local_max:
                local_max = lmax
        return self._mpiw.allreduce(local_max, op=MPI.MAX)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the sum of all entries in this MPIBlockVector
        """
        assert out is None, 'Out keyword not supported'
        assert_block_structure(self)
        rank = self._mpiw.Get_rank()
        indices = self._unique_owned_blocks if rank != 0 else self._owned_blocks

        local_sum = 0.0
        for i in indices:
            local_sum += self._block_vector.get_block(i).sum(axis=axis, dtype=dtype)

        return self._mpiw.allreduce(local_sum, op=MPI.SUM)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the product of all entries in this MPIBlockVector
        """
        assert out is None, 'Out keyword not supported'
        assert_block_structure(self)
        rank = self._mpiw.Get_rank()
        indices = self._unique_owned_blocks if rank != 0 else self._owned_blocks

        local_prod = 1.0
        for i in indices:
            local_prod *= self._block_vector.get_block(i).prod(axis=axis, dtype=dtype)
        return self._mpiw.allreduce(local_prod, op=MPI.PROD)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the average of all entries in this MPIBlockVector
        """
        return self.sum(out=out)/self.size

    def conj(self):
        """
        Complex-conjugate all elements.
        """
        assert_block_structure(self)
        result = self.copy_structure()
        for i in self._owned_blocks:
            result.set_block(i, self.get_block(i).conj())
        return result

    def conjugate(self):
        """
        Complex-conjugate all elements.
        """
        return self.conj()

    def nonzero(self):
        """
        Returns the indices of the elements that are non-zero.
        """
        result = MPIBlockVector(nblocks=self.nblocks, rank_owner=self.rank_ownership,
                                mpi_comm=self.mpi_comm, assert_correct_owners=False)
        assert_block_structure(self)
        for i in self._owned_blocks:
            result.set_block(i, self._block_vector.get_block(i).nonzero()[0])
        return (result,)

    def round(self, decimals=0, out=None):
        """
        Return MPIBlockVector with each element rounded to the given number of decimals
        """
        assert out is None, 'Out keyword not supported'
        assert_block_structure(self)
        result = self.copy_structure()
        for i in self._owned_blocks:
            result.set_block(i, self._block_vector.get_block(i).round(decimals=decimals))
        return result

    def clip(self, min=None, max=None, out=None):
        """
        Return MPIBlockVector whose values are limited to [min, max].
        One of max or min must be given.

        Parameters
        ----------
        min: scalar_like, optional
            Minimum value. If None, clipping is not performed on lower interval edge.
        max: scalar_like, optional
            Maximum value. If None, clipping is not performed on upper interval edge.

        Returns
        -------
        MPIBlockVector

        """
        assert out is None, 'Out keyword not supported'
        assert_block_structure(self)
        result = self.copy_structure()
        for i in self._owned_blocks:
            result.set_block(i, self._block_vector.get_block(i).clip(min=min, max=max))
        return result

    def compress(self, condition, axis=None, out=None):
        """
        Return selected slices of each subblock.

        Parameters
        ----------
        condition: MPIBlockVector that selects which entries to return.
            Determines to select (evaluate True in condition)

        Returns
        -------
        MPIBlockVector

        """
        assert out is None, 'Out keyword not supported'
        assert_block_structure(self)
        result = MPIBlockVector(nblocks=self.nblocks, rank_owner=self.rank_ownership,
                                mpi_comm=self.mpi_comm, assert_correct_owners=False)
        if isinstance(condition, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, condition._rank_owner), msg
            assert self._mpiw == condition._mpiw, 'Need to have same communicator'
            for i in self._owned_blocks:
                result.set_block(i, self.get_block(i).compress(condition.get_block(i)))
            return result
        if isinstance(condition, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(condition, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        else:
            raise NotImplementedError('Operation not supported by MPIBlockVector')

    def copyfrom(self, other):
        """
        Copy entries of other into this MPIBlockVector

        Parameters
        ----------
        other: MPIBlockVector or BlockVector

        Returns
        -------
        None
        """
        if isinstance(other, MPIBlockVector):
            assert_block_structure(other)
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            assert self.nblocks == other.nblocks, msg
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other.rank_ownership), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                self.set_block(i, other.get_block(i).copy())

        elif isinstance(other, BlockVector):
            block_vector_assert_block_structure(other)
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                self.set_block(i, other.get_block(i).copy())
        elif isinstance(other, np.ndarray):
            assert_block_structure(self)
            if not self.is_broadcasted():
                self.broadcast_block_sizes()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            offset = 0
            for idx in range(self.nblocks):
                if self._owned_mask[idx]:
                    subarray = other[offset: offset + self.get_block_size(idx)]
                    if isinstance(self.get_block(idx), BlockVector):
                        self.get_block(idx).copyfrom(subarray)
                    else:
                        np.copyto(self.get_block(idx), subarray)
                offset += self.get_block_size(idx)
        else:
            raise NotImplementedError('Operation not supported by MPIBlockVector')

    def copyto(self, other):
        """
        Copy entries of this MPIBlockVector into other

        Parameters
        ----------
        other: MPIBlockVector or BlockVector

        Returns
        -------
        None

        """
        if isinstance(other, MPIBlockVector):
            other.copyfrom(self)
        else:
            raise NotImplementedError('Operation not supported by MPIBlockVector')

    def set_blocks(self, blocks):
        """
        Assigns vectors in blocks

        Parameters
        ----------
        blocks: list
            list of vectors

        Returns
        -------
        None
        """
        raise NotImplementedError('Operation not supported by MPIBlockVector')

    def clone(self, value=None, copy=True):
        """
        Returns a copy of this MPIBlockVector

        Parameters
        ----------
        value: scalar, optional
            all entries of the cloned vector are set to this value
        copy: bool, optinal
            if set to true makes a deepcopy of each block in this vector. default False

        Returns
        -------
        MPIBlockVector
        """
        result = MPIBlockVector(self.nblocks, self.rank_ownership, self.mpi_comm, assert_correct_owners=False)
        result._block_vector = self._block_vector.clone(copy=copy)
        result._brow_lengths = self._brow_lengths.copy()
        result._undefined_brows = set(self._undefined_brows)
        if value is not None:
            result.fill(value)
        return result

    def copy(self, order='C'):
        """
        Returns a copy of the MPIBlockVector
        """
        result = MPIBlockVector(self.nblocks, self.rank_ownership, self.mpi_comm, assert_correct_owners=False)
        result._block_vector = self._block_vector.copy(order=order)
        result._brow_lengths = self._brow_lengths.copy()
        result._undefined_brows = set(self._undefined_brows)
        return result

    def copy_structure(self):
        """
        Returns a copy of the MPIBlockVector structure filled with zeros
        """
        result = MPIBlockVector(self.nblocks, self.rank_ownership, self.mpi_comm, assert_correct_owners=False)
        if self.is_broadcasted():
            result.finalize_block_sizes(broadcast=False, block_sizes=self.block_sizes(copy=False))
        for bid in self.owned_blocks:
            block = self.get_block(bid)
            if block is not None:
                if isinstance(block, BlockVector):
                    result.set_block(bid, block.copy_structure())
                elif type(block) == np.ndarray:
                    result.set_block(bid, np.zeros(block.size))
                else:
                    raise NotImplementedError('Should never get here')
        return result

    def fill(self, value):
        """
        Fills the MPIBLockVector with a scalar value.

        Parameters
        ----------
        value : scalar
            All elements in the vector will be assigned this value

        Returns
        -------
        None

        """
        assert_block_structure(self)
        for idx in self.owned_blocks:
            self.get_block(idx).fill(value)

    def dot(self, other, out=None):
        """
        Returns dot product

        Parameters
        ----------
        other : MPIBlockVector

        Returns
        -------
        float

        """
        assert_block_structure(self)
        assert out is None
        if isinstance(other, MPIBlockVector):
            assert_block_structure(other)
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self.rank_ownership, other.rank_ownership), msg
            assert self.mpi_comm == other.mpi_comm, 'Need to have same communicator'

            rank = self._mpiw.Get_rank()
            indices = self._unique_owned_blocks if rank != 0 else self._owned_blocks
            local_dot_prod = 0.0
            for i in indices:
                local_dot_prod += self._block_vector.get_block(i).dot(other.get_block(i))

            return self._mpiw.allreduce(local_dot_prod, op=MPI.SUM)
        elif isinstance(other, BlockVector):
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch: {} != {}'.format(self.nblocks, other.nblocks)
            return self.dot(other.toMPIBlockVector(self.rank_ownership, self.mpi_comm, assert_correct_owners=False))
        elif isinstance(other, np.ndarray):
            other_bv = self.copy_structure()
            other_bv.copyfrom(other)
            return self.dot(other_bv)
        else:
            raise NotImplementedError('Operation not supported by MPIBlockVector')

    @staticmethod
    def _serialize_structure(block_vector):
        """
        Parameters
        ----------
        block_vector: BlockVector

        Returns
        -------
        list
        """
        serialized_structure = list()
        for ndx in range(block_vector.nblocks):
            blk = block_vector.get_block(ndx)
            if isinstance(blk, BlockVector):
                serialized_structure.append(-1)
                serialized_structure.append(blk.nblocks)
                serialized_structure.extend(MPIBlockVector._serialize_structure(blk))
            elif isinstance(blk, MPIBlockVector):
                raise NotImplementedError('Operation not supported for MPIBlockVectors containing other MPIBlockVectors')
            elif isinstance(blk, np.ndarray):
                serialized_structure.append(-2)
                serialized_structure.append(blk.size)
            else:
                raise NotImplementedError('Unrecognized input.')
        return serialized_structure

    @staticmethod
    def _create_from_serialized_structure(serialized_structure, structure_ndx, result):
        """
        Parameters
        ----------
        serialized_structure: np.ndarray
        structure_ndx: int
        result: BlockVector

        Returns
        -------
        structure_ndx: int
        """
        for ndx in range(result.nblocks):
            if serialized_structure[structure_ndx] == -1:
                structure_ndx += 1
                block = BlockVector(serialized_structure[structure_ndx])
                structure_ndx += 1
                structure_ndx = MPIBlockVector._create_from_serialized_structure(serialized_structure,
                                                                                 structure_ndx,
                                                                                 block)
                result.set_block(ndx, block)
            elif serialized_structure[structure_ndx] == -2:
                structure_ndx += 1
                result.set_block(ndx, np.zeros(serialized_structure[structure_ndx]))
                structure_ndx += 1
            else:
                raise ValueError('Unrecognized structure')
        return structure_ndx

    def make_local_structure_copy(self):
        """
        Creates a BlockVector with the same structure as the MPIBlockVector

        Returns
        -------
        BlockVector
        """
        """
        We do this by serializing the structure, then gathering it.
        To serialize the structure, we use an array. The first number indicates if the first block is a numpy array 
        or a BlockVector. We use -1 to indicate a BlockVector and -2 to indicate a numpy array. If the block is a
        BlockVector, then the next number is a positive integer specifying the number of blocks in the block vector.
        If the block is a numpy array, then the next number is a positive integer specifying the size of the array. 
        After the number of blocks in a BlockVector is specified, we follow the same procedure to specify the 
        structure of that BlockVector. 
        """
        assert_block_structure(self)
        serialized_structure_by_block = dict()
        length_per_block = np.zeros(self.nblocks, dtype=np.int64)
        rank = self._mpiw.Get_rank()
        if rank == 0:
            block_indices = self._owned_blocks
        else:
            block_indices = self._unique_owned_blocks
        for ndx in block_indices:
            blk = self.get_block(ndx)
            blk_structure = list()
            if isinstance(blk, BlockVector):
                blk_structure.append(-1)
                blk_structure.append(blk.nblocks)
                blk_structure.extend(self._serialize_structure(blk))
            elif isinstance(blk, MPIBlockVector):
                raise NotImplementedError('Operation not supported for MPIBlockVectors containing other MPIBlockVectors')
            elif isinstance(blk, np.ndarray):
                blk_structure.append(-2)
                blk_structure.append(blk.size)
            else:
                raise NotImplementedError('Unrecognized input.')
            length_per_block[ndx] = len(blk_structure)
            serialized_structure_by_block[ndx] = np.asarray(blk_structure, dtype=np.int64)

        global_length_per_block = np.zeros(self.nblocks, dtype=np.int64)
        self._mpiw.Allreduce(length_per_block, global_length_per_block)
        local_serialized_structure = np.zeros(global_length_per_block.sum(), dtype=np.int64)

        offset = 0
        block_indices_set = set(block_indices)
        for ndx in range(self.nblocks):
            if ndx in block_indices_set:
                local_serialized_structure[offset: offset+global_length_per_block[ndx]] = serialized_structure_by_block[ndx]
            offset += global_length_per_block[ndx]
        global_serialized_structure = np.zeros(global_length_per_block.sum(), dtype=np.int64)
        self._mpiw.Allreduce(local_serialized_structure, global_serialized_structure)

        result = BlockVector(self.nblocks)
        structure_ndx = 0
        self._create_from_serialized_structure(global_serialized_structure, structure_ndx, result)

        return result

    def make_local_copy(self):
        """
        Copies the MPIBlockVector into a BlockVector

        Returns
        -------
        BlockVector
        """
        assert_block_structure(self)
        if not self.is_broadcasted():
            self.broadcast_block_sizes()
        result = self.make_local_structure_copy()

        local_data = np.zeros(self.size)
        global_data = np.zeros(self.size)

        offset = 0
        rank = self._mpiw.Get_rank()
        if rank == 0:
            block_indices = set(self._owned_blocks)
        else:
            block_indices = set(self._unique_owned_blocks)
        for ndx in range(self.nblocks):
            if ndx in block_indices:
                blk = self.get_block(ndx)
                if isinstance(blk, BlockVector):
                    local_data[offset: offset + self.get_block_size(ndx)] = blk.flatten()
                elif isinstance(blk, np.ndarray):
                    local_data[offset: offset + self.get_block_size(ndx)] = blk
                else:
                    raise ValueError('Unrecognized block type')
            offset += self.get_block_size(ndx)

        self._mpiw.Allreduce(local_data, global_data)
        result.copyfrom(global_data)

        return result

    def _binary_operation_helper(self, other, operation):
        assert_block_structure(self)
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector) or isinstance(other, BlockVector):
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch: {} != {}'.format(self.nblocks, other.nblocks)
            if isinstance(other, MPIBlockVector):
                assert np.array_equal(self._rank_owner, other._rank_owner) or self._mpiw.Get_size() == 1, \
                    'MPIBlockVectors must be distributed in same processors'
                assert self._mpiw == other._mpiw, 'Need to have same communicator'
            for i in self._owned_blocks:
                result.set_block(i, operation(self.get_block(i), other.get_block(i)))
            return result
        elif isinstance(other, np.ndarray):
            _tmp = self.copy_structure()
            _tmp.copyfrom(other)
            return self._binary_operation_helper(_tmp, operation)
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, operation(self.get_block(i), other))
            return result
        else:
            raise NotImplementedError('Operation not supported by MPIBlockVector')

    def _reverse_binary_operation_helper(self, other, operation):
        assert_block_structure(self)
        result = self.copy_structure()
        if isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i,  operation(other, self.get_block(i)))
            return result
        else:
            raise NotImplementedError('Operation not supported by MPIBlockVector')

    def _inplace_binary_operation_helper(self, other, operation):
        assert_block_structure(self)
        if isinstance(other, MPIBlockVector) or isinstance(other, BlockVector):
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch: {} != {}'.format(self.nblocks, other.nblocks)
            if isinstance(other, MPIBlockVector):
                assert np.array_equal(self._rank_owner, other._rank_owner) or self._mpiw.Get_size() == 1, \
                    'MPIBlockVectors must be distributed in same processors'
                assert self._mpiw == other._mpiw, 'Need to have same communicator'
                assert_block_structure(other)
            else:
                block_vector_assert_block_structure(other)

            for i in self._owned_blocks:
                blk = self.get_block(i)
                operation(blk, other.get_block(i))
                self.set_block(i, blk)
            return self
        elif isinstance(other, np.ndarray):
            _tmp = self.copy_structure()
            _tmp.copyfrom(other)
            return self._inplace_binary_operation_helper(_tmp, operation)
        elif np.isscalar(other):
            for i in self._owned_blocks:
                blk = self.get_block(i)
                operation(blk, other)
                self.set_block(i, blk)
            return self
        else:
            raise NotImplementedError('Operation not supported by MPIBlockVector')

    def __add__(self, other):
        return self._binary_operation_helper(other, operator.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_operation_helper(other, operator.sub)

    def __rsub__(self, other):
        return self._reverse_binary_operation_helper(other, operator.sub)

    def __mul__(self, other):
        return self._binary_operation_helper(other, operator.mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_operation_helper(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._reverse_binary_operation_helper(other, operator.truediv)

    def __floordiv__(self, other):
        return self._binary_operation_helper(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self._reverse_binary_operation_helper(other, operator.floordiv)

    def __neg__(self):
        assert_block_structure(self)
        result = self.copy_structure()
        for ndx in self._owned_blocks:
            result.set_block(ndx, -self.get_block(ndx))
        return result

    def __iadd__(self, other):
        return self._inplace_binary_operation_helper(other, operator.iadd)

    def __isub__(self, other):
        return self._inplace_binary_operation_helper(other, operator.isub)

    def __imul__(self, other):
        return self._inplace_binary_operation_helper(other, operator.imul)

    def __itruediv__(self, other):
        return self._inplace_binary_operation_helper(other, operator.itruediv)

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def _comparison_helper(self, other, operation):
        assert_block_structure(self)
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            assert_block_structure(other)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch: {} != {}'.format(self.nblocks, other.nblocks)
            assert np.array_equal(self._rank_owner, other._rank_owner) or self._mpiw.Get_size() == 1, \
                'MPIBlockVectors must be distributed in same processors'
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, operation(self.get_block(i), other.get_block(i)))
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, operation(self.get_block(i), other))
            return result
        else:
            raise NotImplementedError('Operation not supported by MPIBlockVector')

    def __le__(self, other):
        return self._comparison_helper(other, operator.le)

    def __lt__(self, other):
        return self._comparison_helper(other, operator.lt)

    def __ge__(self, other):
        return self._comparison_helper(other, operator.ge)

    def __gt__(self, other):
        return self._comparison_helper(other, operator.gt)

    def __eq__(self, other):
        return self._comparison_helper(other, operator.eq)

    def __ne__(self, other):
        return self._comparison_helper(other, operator.ne)

    def __contains__(self, item):
        other = item
        assert_block_structure(self)
        if np.isscalar(other):
            contains = False
            for i in self._owned_blocks:
                if other in self.get_block(i):
                    contains = True
            return bool(self._mpiw.allreduce(contains, op=MPI.SUM))
        else:
            raise NotImplementedError('Operation not supported by MPIBlockVector')

    def get_block(self, key):
        owner = self._rank_owner[key]
        rank = self._mpiw.Get_rank()
        assert owner == rank or owner < 0, 'Block {} not own by processor {}'.format(key, rank)
        return self._block_vector.get_block(key)

    def set_block(self, key, value):
        owner = self._rank_owner[key]
        rank = self._mpiw.Get_rank()
        assert owner == rank or owner < 0, \
            'Block {} not owned by processor {}'.format(key, rank)

        self._block_vector.set_block(key, value)
        self._set_block_size(key, value.size)

    def _has_equal_structure(self, other):
        if not (isinstance(other, MPIBlockVector) or isinstance(other, BlockVector)):
            return False
        if self.nblocks != other.nblocks:
            return False
        if isinstance(other, MPIBlockVector):
            if (self.owned_blocks != other.owned_blocks).any() and self._mpiw.Get_size() != 1:
                return False
        for ndx in self.owned_blocks:
            block1 = self.get_block(ndx)
            block2 = other.get_block(ndx)
            if isinstance(block1, BlockVector):
                if not isinstance(block2, BlockVector):
                    return False
                if not block1._has_equal_structure(block2):
                    return False
            elif isinstance(block2, BlockVector):
                return False
        return True

    def __getitem__(self, item):
        if not self._has_equal_structure(item):
            raise ValueError('MIPBlockVector.__getitem__ only accepts slices in the form of MPIBlockVectors of the same structure')
        res = self.copy_structure()
        for ndx in self.owned_blocks:
            block = self.get_block(ndx)
            res.set_block(ndx, block[item.get_block(ndx)])

    def __setitem__(self, key, value):
        if not (self._has_equal_structure(key) and (self._has_equal_structure(value) or np.isscalar(value))):
            raise ValueError(
                'MPIBlockVector.__setitem__ only accepts slices in the form of MPIBlockVectors of the same structure')
        if np.isscalar(value):
            for ndx in self.owned_blocks:
                block = self.get_block(ndx)
                block[key.get_block(ndx)] = value
        else:
            for ndx in self.owned_blocks:
                block = self.get_block(ndx)
                block[key.get_block(ndx)] = value.get_block(ndx)

    def __str__(self):
        msg = '{}{}:\n'.format(self.__class__.__name__, self.bshape)
        for idx in range(self.nblocks):
            msg += '{}: Owned by processor {}\n'.format(idx, self._rank_owner[idx])

        return msg

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.bshape)

    def pprint(self, root=0):
        """Prints BlockVector in pretty format"""
        assert_block_structure(self)
        msg = self.__repr__() + '\n'
        num_processors = self._mpiw.Get_size()
        local_mask = self._owned_mask.flatten()
        receive_data = np.empty(num_processors * self.nblocks,
                                dtype=np.bool)
        self._mpiw.Allgather(local_mask, receive_data)
        processor_to_mask = np.split(receive_data, num_processors)

        global_mask = np.zeros(self.nblocks, dtype=np.bool)

        for bid in range(self.nblocks):
            owner = self._rank_owner[bid]
            if owner >= 0:
                global_mask[bid] = processor_to_mask[owner][bid]
            else:
                # checks only the mask of one of them since all must have the same
                global_mask[bid] = processor_to_mask[0][bid]

            disp_owner = self._rank_owner[bid] if self._rank_owner[bid] >= 0 else 'All'
            is_none = '' if global_mask[bid] else 'None'
            repn = 'Owned by {} Shape({},){}'.format(disp_owner,
                                                     self._brow_lengths[bid],
                                                     is_none)
            msg += '{}: {}\n'.format(bid, repn)
        if self._mpiw.Get_rank() == root:
            print(msg)

    def __len__(self):
        raise NotImplementedError('Use size or nblocks')

    def __iter__(self):
        raise NotImplementedError('Not supported by MPIBlockVector')

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def cumprod(self, axis=None, dtype=None, out=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def cumsum(self, axis=None, dtype=None, out=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def tolist(self):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def flatten(self, order='C'):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def ravel(self, order='C'):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def argpartition(self, kth, axis=-1, kind='introselect', order=None):
        BaseBlockVector.argpartition(self, kth, axis=axis, kind=kind, order=order)

    def argsort(self, axis=-1, kind='quicksort', order=None):
        BaseBlockVector.argsort(self, axis=axis, kind=kind, order=order)

    def byteswap(self, inplace=False):
        BaseBlockVector.byteswap(self, inplace=inplace)

    def choose(self, choices, out=None, mode='raise'):
        BaseBlockVector.choose(self, choices, out=out, mode=mode)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        BaseBlockVector.diagonal(self, offset=offset, axis1=axis1, axis2=axis2)

    def dump(self, file):
        BaseBlockVector.dump(self, file)

    def dumps(self):
        BaseBlockVector.dumps(self)

    def getfield(self, dtype, offset=0):
        BaseBlockVector.getfield(self, dtype, offset=offset)

    def item(self, *args):
        BaseBlockVector.item(self, *args)

    def itemset(self, *args):
        BaseBlockVector.itemset(self, *args)

    def newbyteorder(self, new_order='S'):
        BaseBlockVector.newbyteorder(self, new_order=new_order)

    def put(self, indices, values, mode='raise'):
        BaseBlockVector.put(self, indices, values, mode=mode)

    def partition(self, kth, axis=-1, kind='introselect', order=None):
        BaseBlockVector.partition(self, kth, axis=axis, kind=kind, order=order)

    def repeat(self, repeats, axis=None):
        BaseBlockVector.repeat(self, repeats, axis=axis)

    def reshape(self, shape, order='C'):
        BaseBlockVector.reshape(self, shape, order=order)

    def resize(self, new_shape, refcheck=True):
        BaseBlockVector.resize(self, new_shape, refcheck=refcheck)

    def searchsorted(self, v, side='left', sorter=None):
        BaseBlockVector.searchsorted(self, v, side=side, sorter=sorter)

    def setfield(self, val, dtype, offset=0):
        BaseBlockVector.setfield(self, val, dtype, offset=offset)

    def setflags(self, write=None, align=None, uic=None):
        BaseBlockVector.setflags(self, write=write, align=align, uic=uic)

    def sort(self, axis=-1, kind='quicksort', order=None):
        BaseBlockVector.sort(self, axis=axis, kind=kind, order=order)

    def squeeze(self, axis=None):
        BaseBlockVector.squeeze(self, axis=axis)

    def swapaxes(self, axis1, axis2):
        BaseBlockVector.swapaxes(self, axis1, axis2)

    def tobytes(self, order='C'):
        BaseBlockVector.tobytes(self, order=order)

    def argmax(self, axis=None, out=None):
        BaseBlockVector.argmax(self, axis=axis, out=out)

    def argmin(self, axis=None, out=None):
        BaseBlockVector.argmax(self, axis=axis, out=out)

    def take(self, indices, axis=None, out=None, mode='raise'):
        BaseBlockVector.take(self, indices, axis=axis, out=out, mode=mode)

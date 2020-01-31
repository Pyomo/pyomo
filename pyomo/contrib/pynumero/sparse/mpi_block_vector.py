#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from __future__ import division
from pyomo.contrib.pynumero.sparse import BlockVector
from .base_block import BaseBlockVector
from mpi4py import MPI
import numpy as np
import copy as cp

__all__ = ['MPIBlockVector']

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
    _need_broadcast_sizes: bool
        True if length of any block changed. If true user will need to call
        broadcast_block_sizes in the future before performing any operation.
        Users will be notified if that is the case.
    _done_first_broadcast_sizes: bool
        True if broadcast_block_sizes has been called and the length of any
        block changed since. If true user will need to call
        broadcast_block_sizes in the future before performing any operation.
        Users will be notified if that is the case.

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
    block_sizes: array_like, optional
        Array_like of size nblocks. This specifies the length of each entry in
        the MPIBlockVector.

    """

    def __new__(cls, nblocks, rank_owner, mpi_comm, block_sizes=None):

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
        obj._owned_blocks = list()
        obj._unique_owned_blocks = list()
        obj._owned_mask = np.zeros(nblocks, dtype=bool)
        for i, owner in enumerate(obj._rank_owner):
            if owner != rank and owner >= 0:
                # empty the blocks that are not owned by this processor
                # blocks that are not owned by this proceesor are set
                # to numpy.zeros(0) in _block_vector
                obj._block_vector.set_block(i, np.zeros(0))
            else:
                obj._owned_blocks.append(i)
                obj._owned_mask[i] = True
                if owner == rank:
                    obj._unique_owned_blocks.append(i)

        # containers that facilitate looping
        obj._owned_blocks = np.array(obj._owned_blocks)
        obj._unique_owned_blocks = np.array(obj._unique_owned_blocks)

        # make some pointers unmutable. These arrays don't change after
        # MPIBlockVector has been created
        obj._rank_owner.flags.writeable = False
        obj._owned_blocks.flags.writeable = False
        obj._owned_mask.flags.writeable = False
        obj._unique_owned_blocks.flags.writeable = False
        return obj

    def __init__(self, nblocks, rank_owner, mpi_comm, block_sizes=None):

        # keep track of global sizes
        if block_sizes is None:
            self._need_broadcast_sizes = True
            self._done_first_broadcast_sizes = False
            self._brow_lengths = np.zeros(nblocks, dtype=np.int64)
        else:
            sizes = np.array(block_sizes, dtype=np.int64)
            assert sizes.size == self.nblocks
            assert (sizes >= 0).size == self.nblocks, \
                'Blocks must have positive sizes'
            self._need_broadcast_sizes = False
            self._done_first_broadcast_sizes = True
            self._brow_lengths = sizes

        # Note: this requires communication but is disabled when assertions
        # are turned off
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
                       ]
        # functions that take two vectors
        binary_funcs = [np.add, np.multiply, np.divide, np.subtract,
                        np.greater, np.greater_equal, np.less, np.less_equal,
                        np.not_equal, np.maximum, np.minimum, np.fmax,
                        np.fmin, np.equal, np.logical_and,
                        np.logical_or, np.logical_xor, np.logaddexp,
                        np.logaddexp2, np.remainder, np.heaviside,
                        np.hypot]

        args = [input_ for i, input_ in enumerate(inputs)]

        outputs = kwargs.pop('out', None)
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, BlockVector):
                    raise NotImplementedError(str(ufunc))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        if ufunc in unary_funcs:
            results = self._unary_operation(ufunc, method, *args, **kwargs)
            return results
        elif ufunc in binary_funcs:
            results = self._binary_operation(ufunc, method, *args, **kwargs)
            return results
        else:
            raise NotImplementedError(str(ufunc) + "not supported for MPIBlockVector")

    def _unary_operation(self, ufunc, method, *args, **kwargs):
        """Run recursion to perform unary_funcs on MPIBlockVector"""
        # ToDo: deal with out
        x = args[0]

        if isinstance(x, MPIBlockVector):
            rank = self._mpiw.Get_rank()
            v = MPIBlockVector(self.nblocks, self._rank_owner, self._mpiw)
            for i in self._owned_blocks:
                _args = [x[i]] + [args[j] for j in range(1, len(args))]
                v.set_block(i, self._unary_operation(ufunc, method, *_args, **kwargs))
            return v
        elif isinstance(x, BlockVector):
            v = BlockVector(x.nblocks)
            for i in range(x.nblocks):
                _args = [x[i]] + [args[j] for j in range(1, len(args))]
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
            assert np.array_equal(x1._rank_owner, x2._rank_owner), msg
            assert x1._mpiw == x2._mpiw, 'Need to have same communicator'

            res = MPIBlockVector(x1.nblocks, x1._rank_owner, self._mpiw)
            for i in x1._owned_blocks:
                _args = [x1[i]] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
            return res
        elif isinstance(x1, BlockVector) and isinstance(x2, MPIBlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(x1, MPIBlockVector) and isinstance(x2, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(x1, MPIBlockVector) and np.isscalar(x2):
            res = MPIBlockVector(x1.nblocks, x1._rank_owner, self._mpiw)
            for i in x1._owned_blocks:
                _args = [x1[i]] + [x2] + [args[j] for j in range(2, len(args))]
                res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
            return res
        elif isinstance(x2, MPIBlockVector) and np.isscalar(x1):
            res = MPIBlockVector(x2.nblocks, x2._rank_owner, self._mpiw)
            for i in x2._owned_blocks:
                _args = [x1] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res.set_block(i, self._binary_operation(ufunc, method, *_args, **kwargs))
            return res
        elif isinstance(x1, MPIBlockVector) and type(x2)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(x2, MPIBlockVector) and type(x1)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
            # this will take care of blockvector and ndarrays
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
        return self.size,

    @property
    def size(self):
        """
        Returns total number of elements in this MPIBlockVector
        """
        self._assert_broadcasted_sizes()
        return np.sum(self._brow_lengths)

    @property
    def ndim(self):
        """
        Returns dimension of this MPIBlockVector
        """
        return 1

    # Note: this operation requires communication
    @property
    def has_none(self):
        """
        Returns True if block vector has none entry

        Notes
        -----
        This operation is expensive as it requires communication of all
        processors. Mostly for debugging purposes.
        Also, This only checks if all entries at the BlockVector are
        different than none. It does not check recursively for subvectors
        to not have nones.

        """
        return self._mpiw.allreduce(self._block_vector.has_none, op=MPI.SUM)

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
        return np.array([i for i in range(self.nblocks) if self._rank_owner[i]<0])

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

    def block_sizes(self, copy=True):
        """
        Returns 1D-Array with sizes of individual blocks in this MPIBlockVector
        """
        self._assert_broadcasted_sizes()
        if copy:
            return self._brow_lengths.copy()
        return self._brow_lengths

    # Note: this operation requires communication
    def broadcast_block_sizes(self):
        """
        Send sizes of all blocks to all processors. After this method is called
        this MPIBlockVector knows it's dimensions across all blocks. This method
        must be called before running any operations with the MPIBlockVector.
        """
        rank = self._mpiw.Get_rank()
        num_processors = self._mpiw.Get_size()

        local_length_data = self._block_vector.block_sizes()
        receive_data = np.empty(num_processors * self.nblocks, dtype=np.int64)

        self._mpiw.Allgather(local_length_data, receive_data)

        proc_dims = np.split(receive_data, num_processors)

        for i in range(self.nblocks):
            block_length = set()
            for k in range(num_processors):
                processor_sizes = proc_dims[k]
                block_length.add(processor_sizes[i])
            if len(block_length)>2:
                msg = 'Block {} has more than one dimension accross processors'.format(i)
                raise RuntimeError(msg)
            elif len(block_length) == 2:
                if 0 not in block_length:
                    msg = 'Block {} has more than one dimension accross processors'.format(i)
                    raise RuntimeError(msg)
                block_length.remove(0)

            # here block_length must only have one element
            self._brow_lengths[i] = block_length.pop()

        self._need_broadcast_sizes = False
        self._done_first_broadcast_sizes = True

    def _assert_broadcasted_sizes(self):
        """
        Checks if this MPIBlockVector needs to boradcast sizes. This is needed if
        there has been changes in sizes blocks
        """
        if not self._done_first_broadcast_sizes:
            assert not self._need_broadcast_sizes, \
                'First need to call broadcast_block_sizes()'
        else:
            assert not self._need_broadcast_sizes, \
                'Structure changed. Need to recall broadcast_block_sizes()'

    # Note: this requires communication but is only runned in __new__
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
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        local = 1
        for i in self._owned_blocks:
            local *= self._block_vector[i].all()

        return bool(self._mpiw.allreduce(local, op=MPI.PROD))

    def any(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.
        """
        assert out is None, 'Out keyword not supported'
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        local = 0
        for i in self._owned_blocks:
            local += self._block_vector[i].any()

        return bool(self._mpiw.allreduce(local, op=MPI.SUM))

    def min(self, axis=None, out=None, keepdims=False):
        """
        Returns the smallest value stored in the vector
        """
        assert out is None, 'Out keyword not supported'
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        local_min = np.inf
        for i in self._owned_blocks:
            lmin = self._block_vector[i].min()
            if lmin <= local_min:
                local_min = lmin
        return self._mpiw.allreduce(local_min, op=MPI.MIN)

    def max(self, axis=None, out=None, keepdims=False):
        """
        Returns the largest value stored in this MPIBlockVector
        """
        assert out is None, 'Out keyword not supported'
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        local_max = -np.inf
        for i in self._owned_blocks:
            lmax = self._block_vector[i].max()
            if lmax >= local_max:
                local_max = lmax
        return self._mpiw.allreduce(local_max, op=MPI.MAX)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the sum of all entries in this MPIBlockVector
        """
        assert out is None, 'Out keyword not supported'
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        rank = self._mpiw.Get_rank()
        indices = self._unique_owned_blocks if rank != 0 else self._owned_blocks

        local_sum = 0.0
        for i in indices:
            local_sum += self._block_vector[i].sum(axis=axis, dtype=dtype)

        return self._mpiw.allreduce(local_sum, op=MPI.SUM)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the product of all entries in this MPIBlockVector
        """
        assert out is None, 'Out keyword not supported'
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        rank = self._mpiw.Get_rank()
        indices = self._unique_owned_blocks if rank != 0 else self._owned_blocks

        local_prod = 1.0
        for i in indices:
            local_prod *= self._block_vector[i].prod(axis=axis, dtype=dtype)
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
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        for i in self._owned_blocks:
            result.set_block(i, self._block_vector[i].conj())
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
        result = self.copy_structure()
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        for i in self._owned_blocks:
            result.set_block(i, self._block_vector[i].nonzero()[0])
        return (result,)

    def round(self, decimals=0, out=None):
        """
        Return MPIBlockVector with each element rounded to the given number of decimals
        """
        assert out is None, 'Out keyword not supported'
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        result = self.copy_structure()
        for i in self._owned_blocks:
            result.set_block(i, self._block_vector[i].round(decimals=decimals))
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
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        result = self.copy_structure()
        for i in self._owned_blocks:
            result.set_block(i, self._block_vector[i].clip(min=min, max=max))
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
        assert not self._block_vector.has_none, 'Operations not allowed with None blocks.'
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(condition, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, condition._rank_owner), msg
            assert self._mpiw == condition._mpiw, 'Need to have same communicator'
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].compress(condition[i]))
            return result
        if isinstance(condition, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(condition, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        else:
            raise NotImplementedError()

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
        rank = self._mpiw.Get_rank()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            if not other._need_broadcast_sizes:
                self._brow_lengths = other._brow_lengths.copy()
                self._need_broadcast_sizes = False
                self._done_first_broadcast_sizes = True

            for i in self._owned_blocks:
                if isinstance(self._block_vector[i], BlockVector):
                    self._block_vector[i].copyfrom(other[i])
                elif type(self._block_vector[i]) == np.ndarray:
                    if isinstance(other[i], BlockVector):
                        self._block_vector.set_block(i, other[i].copy())
                    elif type(other[i])==np.ndarray:
                        np.copyto(self._block_vector[i], other[i])
                    elif other[i] is None:
                        self._block_vector.set_block(i, None)
                    else:
                        raise RuntimeError('Input not recognized')
                elif self._block_vector[i] is None:
                    if isinstance(other[i], np.ndarray):
                        self._block_vector.set_block(i, other[i].copy())
                    elif other[i] is None:
                        self._block_vector.set_block(i, None)
                    else:
                        raise RuntimeError('Input not recognized')
                else:
                    raise RuntimeError('Should never get here')

        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                if isinstance(self._block_vector[i], BlockVector):
                    self._block_vector[i].copyfrom(other[i])
                elif isinstance(self._block_vector[i], np.ndarray):
                    if isinstance(other[i], BlockVector):
                        self._block_vector.set_block(i, other[i].copy())
                    elif isinstance(other[i], np.ndarray):
                        np.copyto(self._block_vector[i], other[i])
                    elif other[i] is None:
                        self._block_vector.set_block(i, None)
                    else:
                        raise RuntimeError('Input not recognized')
                elif self._block_vector[i] is None:
                    if isinstance(other[i], np.ndarray):
                        self._block_vector.set_block(i, other[i].copy())
                    elif other[i] is None:
                        self._block_vector.set_block(i, None)
                    else:
                        raise RuntimeError('Input not recognized')
                else:
                    raise RuntimeError('Should never get here')

        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        else:
            raise NotImplementedError('Input not recognized')

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
        rank = self._mpiw.Get_rank()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other.rank_ownership), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            if not self._need_broadcast_sizes:
                other._brow_lengths = self._brow_lengths.copy()
                other._need_broadcast_sizes = False
                other._done_first_broadcast_sizes = True

            for i in self._owned_blocks:
                blk = self._block_vector[i]
                if isinstance(other[i], BlockVector):
                    other[i].copyfrom(blk)
                elif isinstance(other[i], np.ndarray):
                    if blk is not None:
                        np.copyto(other[i], blk.flatten())
                    else:
                        other.set_block(i, None)
                elif other[i] is None:
                    if blk is not None:
                        other.set_block(i, blk.copy())
                    else:
                        other.set_block(i, None)
                else:
                    raise RuntimeError('Should never get here')
        elif isinstance(other, BlockVector):
            assert other.nblocks == self.nblocks
            if self._need_broadcast_sizes:
                # need to add warning here
                self.broadcast_block_sizes()

            # determine size sent by each processor
            num_processors = self._mpiw.Get_size()
            nblocks = self.nblocks
            rank = self._mpiw.Get_rank()
            chunk_size_per_processor = np.zeros(num_processors, dtype=np.int64)
            sizes_within_processor = [np.zeros(nblocks, dtype=np.int64) for k in range(num_processors)]
            for i in range(nblocks):
                owner = self._rank_owner[i]
                if owner >= 0:
                    chunk_size = self._brow_lengths[i]
                    sizes_within_processor[owner][i] = chunk_size
                    chunk_size_per_processor[owner] += chunk_size

            receive_size = sum(chunk_size_per_processor)
            send_data = np.concatenate([self._block_vector[bid] for bid in self._unique_owned_blocks])
            receive_data = np.empty(receive_size, dtype=send_data.dtype)

            # communicate data to all
            self._mpiw.Allgatherv(send_data, (receive_data, chunk_size_per_processor))

            # split data by processor
            proc_dims = np.split(receive_data, chunk_size_per_processor.cumsum())

            # split data within processor
            splitted_data = []
            for k in range(num_processors):
                splitted_data.append(np.split(proc_dims[k],
                                            sizes_within_processor[k].cumsum()))
            # populate block vector
            for bid in range(nblocks):
                owner = self._rank_owner[bid]
                if owner >= 0:
                    block_data = splitted_data[owner][bid]
                else:
                    block_data = self._block_vector[bid]
                other.set_block(bid, block_data)

        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        else:
            raise NotImplementedError()

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
        msg = 'blocks should be passed in ordered list'
        assert isinstance(blocks, list), msg
        msgj = 'More blocks passed than allocated {} != {}'.format(len(blocks),
                                                                   self.nblocks)
        assert len(blocks) == self.nblocks, msg

        for i in self._owned_blocks:
            self.set_block(i, blocks[i])

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
        result = MPIBlockVector(self.nblocks, self._rank_owner, self._mpiw)
        result._block_vector = self._block_vector.clone(value=value, copy=copy)
        result._need_broadcast_sizes = self._need_broadcast_sizes
        result._done_first_broadcast_sizes = self._done_first_broadcast_sizes
        result._brow_lengths = self._brow_lengths.copy()
        return result

    def copy(self, order='C'):
        """
        Returns a copy of the MPIBlockVector
        """
        v = MPIBlockVector(self.nblocks, self._rank_owner, self._mpiw)
        v._block_vector = self._block_vector.copy()
        v._need_broadcast_sizes = self._need_broadcast_sizes
        v._done_first_broadcast_sizes = self._done_first_broadcast_sizes
        v._brow_lengths = self._brow_lengths.copy()
        return v

    def copy_structure(self, order='C'):
        """
        Returns a copy of the MPIBlockVector structure filled with zeros
        """
        v = MPIBlockVector(self.nblocks, self._rank_owner, self._mpiw)
        v._block_vector = self._block_vector.copy_structure()
        v._need_broadcast_sizes = self._need_broadcast_sizes
        v._done_first_broadcast_sizes = self._done_first_broadcast_sizes
        v._brow_lengths = self._brow_lengths.copy()
        return v

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
        self._block_vector.fill(value)

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
        assert out is None
        rank = self._mpiw.Get_rank()
        indices = self._unique_owned_blocks if rank != 0 else self._owned_blocks
        if isinstance(other, MPIBlockVector):
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            local_dot_prod = 0.0
            for i in indices:
                local_dot_prod += self._block_vector[i].dot(other[i])

            return self._mpiw.allreduce(local_dot_prod, op=MPI.SUM)
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        else:
            raise NotImplementedError()

    def make_local_copy(self):
        """
        Creates copy of this MPIBlockVector but with all blocks owned by all
        processors (i.e. rank_ownership= -np.ones(nblocks))

        Returns
        -------
        MPIBLockVector

        """
        # TODO: This only works for MPIBLockVectors that have np.arrays in blocks
        # but not with blocks with BlockVectors. Need to add support for this
        new_ownership = -np.ones(self.nblocks, dtype=np.int64)
        if np.array_equal(self.rank_ownership, new_ownership):
            return self.copy()

        new_MPIBlockVector = MPIBlockVector(self.nblocks,
                                            new_ownership,
                                            self._mpiw)

        # determine size sent by each processor
        num_processors = self._mpiw.Get_size()
        nblocks = self.nblocks
        rank = self._mpiw.Get_rank()
        chunk_size_per_processor = np.zeros(num_processors, dtype=np.int64)
        sizes_within_processor = [np.zeros(nblocks, dtype=np.int64) for k in range(num_processors)]
        for i in range(nblocks):
            owner = self._rank_owner[i]
            if owner >= 0:
                chunk_size = self._brow_lengths[i]
                sizes_within_processor[owner][i] = chunk_size
                chunk_size_per_processor[owner] += chunk_size

        receive_size = sum(chunk_size_per_processor)
        send_data = np.concatenate([self._block_vector.get_block(bid) for bid in self._unique_owned_blocks])
        receive_data = np.empty(receive_size, dtype=send_data.dtype)

        # communicate data to all
        self._mpiw.Allgatherv(send_data, (receive_data, chunk_size_per_processor))

        # split data by processor
        proc_dims = np.split(receive_data, chunk_size_per_processor.cumsum())

        # split data within processor
        splitted_data = []
        for k in range(num_processors):
            splitted_data.append(np.split(proc_dims[k],
                                          sizes_within_processor[k].cumsum()))
        # populate new vector
        for bid in range(nblocks):
            owner = self._rank_owner[bid]
            if owner >= 0:
                block_data = splitted_data[owner][bid]
            else:
                block_data = self._block_vector[bid]
            new_MPIBlockVector.set_block(bid, block_data)

        # no need to broadcast sizes coz all have the same 
        new_MPIBlockVector._done_first_broadcast_sizes = True
        new_MPIBlockVector._need_broadcast_sizes = False

        return new_MPIBlockVector

    def make_new_MPIBlockVector(self, rank_ownership):
        """
        Creates copy of this MPIBlockVector in a different MPI space. If
        rank_ownership is the same as in this MPIBlockVector a copy of this
        MPIBlockVector is returned.

        Parameters
        ----------
        rank_ownership: array_like
            Array_like of size nblocks. Each entry defines ownership of each block.
            There are two types of ownership. Block that are owned by all processor,
            and blocks owned by a single processor. If a block is owned by all
            processors then its ownership is -1. Otherwise, if a block is owned by
            a single processor, then its ownership is equal to the rank of the
            processor.

        Returns
        -------
        MPIBLockVector

        """
        self._assert_broadcasted_sizes()
        new_ownership = np.array(rank_ownership)
        if np.array_equal(self.rank_ownership, new_ownership):
            return self.copy()

        new_MPIBlockVector = MPIBlockVector(self.nblocks,
                                            new_ownership,
                                            self._mpiw,
                                            block_sizes=self.block_sizes())
        rank = self._mpiw.Get_rank()
        for bid in range(self.nblocks):
            src_owner = self.rank_ownership[bid]
            dest_owner = new_ownership[bid]

            # first check if block is owned by everyone in source
            if src_owner < 0:
                if rank == dest_owner:
                    new_MPIBlockVector.set_block(bid, self[bid])
            # then check if it is the same owner to just copy without any mpi call
            elif src_owner == dest_owner:
                if src_owner == rank:
                    new_MPIBlockVector.set_block(bid, self[bid])
            else:
                # if destination is in different space
                if dest_owner >= 0:
                    # point to point communication
                    if rank == src_owner:
                        data = self[bid]
                        self._mpiw.Send([data, MPI.DOUBLE], dest=dest_owner)
                    elif rank == dest_owner:
                        data = np.empty(self._brow_lengths[bid], dtype=np.float64)
                        self._mpiw.Recv([data, MPI.DOUBLE], source=src_owner)
                        new_MPIBlockVector.set_block(bid, data)
                # if destination is all processors
                else:
                    # broadcast from source to all
                    if rank == src_owner:
                        data = self[bid]
                    else:
                        data = np.empty(self._brow_lengths[bid], dtype=np.float64)

                    self._mpiw.Bcast(data, root=src_owner)
                    new_MPIBlockVector.set_block(bid, data)

        return new_MPIBlockVector

    def __add__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockVectors must be distributed in same processors'
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i] + other[i])
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif type(other)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i] + other)
            return result
        else:
            raise NotImplementedError()

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i] - other[i])
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif type(other)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i] - other)
            return result
        else:
            raise NotImplementedError()

    def __rsub__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i,  other[i] - self._block_vector[i])
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif type(other)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i,  other - self._block_vector[i])
            return result
        else:
            raise NotImplementedError()

    def __mul__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__mul__(other[i]))
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__mul__(other))
            return result
        else:
            raise NotImplementedError()

    def __rmul__(self, other):  # other + self
        return self.__mul__(other)

    def __truediv__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i] / other[i])
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i] / other)
            return result
        else:
            raise NotImplementedError()

    def __rtruediv__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, other[i] / self._block_vector[i])
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, other / self._block_vector[i])
            return result
        else:
            raise NotImplementedError()

    def __floordiv__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'
            result._rank_owner = self._rank_owner

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i] // other[i])
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i] // other)
            return result
        else:
            raise NotImplementedError()

    def __rfloordiv__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, other[i] // self._block_vector[i])
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif type(other)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, other // self._block_vector[i])
            return result
        else:
            raise NotImplementedError()

    def __iadd__(self, other):
        rank = self._mpiw.Get_rank()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                self._block_vector[i] += other[i]
            return self
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif type(other)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                self._block_vector[i] += other
            return self
        else:
            raise NotImplementedError()

    def __isub__(self, other):
        rank = self._mpiw.Get_rank()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                self._block_vector[i] -= other[i]
            return self
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                self._block_vector[i] -= other
            return self
        else:
            raise NotImplementedError()

    def __imul__(self, other):
        rank = self._mpiw.Get_rank()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                self._block_vector[i] *= other[i]
            return self
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                self._block_vector[i] *= other
            return self
        else:
            raise NotImplementedError()

    def __itruediv__(self, other):
        rank = self._mpiw.Get_rank()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                self._block_vector.set_block(i, self._block_vector[i] / other[i])
            return self
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                self._block_vector.set_block(i, self._block_vector[i] / other)
            return self
        else:
            raise NotImplementedError()

    def __le__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__le__(other[i]))
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__le__(other))
            return result
        else:
            raise NotImplementedError()

    def __lt__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__lt__(other[i]))
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__lt__(other))
            return result
        else:
            raise NotImplementedError()

    def __ge__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__ge__(other[i]))
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__ge__(other))
            return result
        else:
            raise NotImplementedError()

    def __gt__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__gt__(other[i]))
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__gt__(other))
            return result
        else:
            raise NotImplementedError()

    def __eq__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__eq__(other[i]))
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__eq__(other))
            return result
        else:
            raise NotImplementedError()

    def __ne__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, other._rank_owner), msg
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__ne__(other[i]))
            return result
        elif isinstance(other, BlockVector):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result.set_block(i, self._block_vector[i].__ne__(other))
            return result
        else:
            raise NotImplementedError()

    def __contains__(self, item):
        other = item
        if np.isscalar(other):
            contains = False
            for i in self._owned_blocks:
                if self._block_vector[i].__contains__(other):
                    contains = True
            return bool(self._mpiw.allreduce(contains, op=MPI.SUM))
        else:
            raise NotImplementedError()

    def get_block(self, key):

        owner = self._rank_owner[key]
        rank = self._mpiw.Get_rank()
        assert owner == rank or \
               owner < 0, 'Block {} not own by processor {}'.format(key, rank)
        return self._block_vector.get_block(key)

    def set_block(self, key, value):

        owner = self._rank_owner[key]
        rank = self._mpiw.Get_rank()
        assert owner == rank or \
               owner < 0, 'Block {} not owned by processor {}'.format(key, rank)
        if value is None:
            if self._block_vector[key] is not None:
                self._need_broadcast_sizes = True
        else:
            new_size = value.size
            if self._brow_lengths[key] != new_size:
                self._need_broadcast_sizes = True

        self._block_vector.set_block(key, value)

    def __getitem__(self, item):
        raise NotImplementedError('MPIBlockVector does not support __getitem__.')

    def __setitem__(self, key, value):
        raise NotImplementedError('MPIBlockVector does not support __setitem__.')

    def __str__(self):
        msg = '{}{}:\n'.format(self.__class__.__name__, self.bshape)
        for idx in range(self.nblocks):
            msg += '{}: Owned by processor {}\n'.format(idx, self._rank_owner[idx])

        return msg

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.bshape)

    def pprint(self, root=0):
        """Prints BlockVector in pretty format"""
        self._assert_broadcasted_sizes()
        msg = self.__repr__() + '\n'
        num_processors = self._mpiw.Get_size()
        local_mask = self._block_vector._block_mask.flatten()
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

            disp_owner = self._rank_owner[bid] if self._rank_owner[bid] >= 0 else 'A'
            is_none = '' if global_mask[bid] else '*'
            repn = 'Owned by {} Shape({},){}'.format(disp_owner,
                                                     self._brow_lengths[bid],
                                                     is_none)
            msg += '{}: {}\n'.format(bid, repn)
        if self._mpiw.Get_rank() == root:
            print(msg)

    def __len__(self):
        return self.nblocks

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

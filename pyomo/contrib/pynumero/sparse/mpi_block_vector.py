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
from mpi4py import MPI
import numpy as np
import copy as cp

__all__ = ['MPIBlockVector']

class MPIBlockVector(np.ndarray):
    """
    Parallel Structured Vector interface

    Parameters
    -------------------
    nblocks: int
    number of blocks contained in the block vector
    rank_owner: list
    list of ranks that own blocks. negative entries mean its own by all.
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

        # empty the blocks that are not owned by this processor
        obj._owned_blocks = list()
        obj._unique_owned_blocks = list()
        obj._owned_mask = np.zeros(nblocks, dtype=bool)
        for i, owner in enumerate(obj._rank_owner):
            if owner != rank and owner >= 0:
                obj._block_vector[i] = np.zeros(0)
            else:
                obj._owned_blocks.append(i)
                obj._owned_mask[i] = True
                if owner == rank:
                    obj._unique_owned_blocks.append(i)

        # containers that facilitate looping
        obj._owned_blocks = np.array(obj._owned_blocks)
        obj._unique_owned_blocks = np.array(obj._unique_owned_blocks)

        # make some pointer unmutable
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
        # ToDo: deal with out
        x = args[0]

        if isinstance(x, MPIBlockVector):
            rank = self._mpiw.Get_rank()
            v = MPIBlockVector(self.nblocks, self._rank_owner, self._mpiw)
            for i in self._owned_blocks:
                _args = [x[i]] + [args[j] for j in range(1, len(args))]
                v[i] = self._unary_operation(ufunc, method, *_args, **kwargs)
            return v
        elif isinstance(x, BlockVector):
            v = BlockVector(x.nblocks)
            for i in range(x.nblocks):
                _args = [x[i]] + [args[j] for j in range(1, len(args))]
                v[i] = self._unary_operation(ufunc, method, *_args, **kwargs)
            return v
        elif type(x) == np.ndarray:
            return super(MPIBlockVector, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        else:
            raise NotImplementedError()

    def _binary_operation(self, ufunc, method, *args, **kwargs):
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
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif isinstance(x1, BlockVector) and isinstance(x2, MPIBlockVector):
            assert x1.nblocks == x2.nblocks, 'Need to have same number of blocks'

            res = MPIBlockVector(x2.nblocks, x2._rank_owner, self._mpiw)
            for i in x2._owned_blocks:
                _args = [x1[i]] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif isinstance(x1, MPIBlockVector) and isinstance(x2, BlockVector):
            assert x1.nblocks == x2.nblocks, 'Need to have same number of blocks'

            res = MPIBlockVector(x1.nblocks, x1._rank_owner, self._mpiw)
            for i in x1._owned_blocks:
                _args = [x1[i]] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif isinstance(x1, MPIBlockVector) and np.isscalar(x2):
            res = MPIBlockVector(x1.nblocks, x1._rank_owner, self._mpiw)
            for i in x1._owned_blocks:
                _args = [x1[i]] + [x2] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif isinstance(x2, MPIBlockVector) and np.isscalar(x1):
            res = MPIBlockVector(x2.nblocks, x2._rank_owner, self._mpiw)
            for i in x2._owned_blocks:
                _args = [x1] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
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
        Returns the number of blocks.
        """
        return self.nblocks,

    @property
    def shape(self):
        """
        Returns total number of elements in the block vector
        """
        return self.size,

    @property
    def size(self):
        """
        Returns total number of elements in the block vector
        """
        if self._need_broadcast_sizes:
            rank = self._mpiw.Get_rank()
            indices = self._unique_owned_blocks if rank != 0 else self._owned_blocks
            local_size = 0
            for i in indices:
                assert self._block_vector[i] is not None
                local_size += self._block_vector[i].size
            return self._mpiw.allreduce(local_size, op=MPI.SUM)
        else:
            self._assert_broadcasted_sizes()
            return np.sum(self._brow_lengths)

    @property
    def ndim(self):
        """
        Returns dimension of the block vector
        """
        return 1

    @property
    def has_none(self):
        """
        Returns True if block vector has none entry
        """
        return self._mpiw.allreduce(self._block_vector.has_none, op=MPI.SUM)

    @property
    def owned_blocks(self):
        """
        Returns array of inidices of blocks owned by this processor
        """
        return self._owned_blocks

    @property
    def shared_blocks(self):
        """
        Returns array of inidices blocks shared by all processors
        """
        return np.array([i for i in range(self.nblocks) if self._rank_owner[i]<0])

    @property
    def rank_ownership(self):
        """
        Returns array of processor rank that own blocks
        """
        return self._rank_owner

    @property
    def ownership_mask(self):
        """
        Returns boolean array that indicates which blocks are owned by this process
        """
        return self._owned_mask

    @property
    def mpi_comm(self):
        return self._mpiw

    def block_sizes(self, copy=True):
        self._assert_broadcasted_sizes()
        if copy:
            return self._brow_lengths.copy()
        return self._brow_lengths

    def broadcast_block_sizes(self):
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

        if not self._done_first_broadcast_sizes:
            assert not self._need_broadcast_sizes, \
                'First need to call broadcast_block_sizes()'
        else:
            assert not self._need_broadcast_sizes, \
                'Structure changed. Need to recall broadcast_block_sizes()'

    # Note: this requires communication
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
        assert out is None
        local = 1
        for i in self._owned_blocks:
            if self._block_vector[i] is not None:
                local *= self._block_vector[i].all()
            else:
                local = 0
        return bool(self._mpiw.allreduce(local, op=MPI.PROD))

    def any(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.
        """
        assert out is None
        local = 0
        for i in self._owned_blocks:
            if self._block_vector[i] is not None:
                local += self._block_vector[i].any()

        return bool(self._mpiw.allreduce(local, op=MPI.SUM))

    def min(self, axis=None, out=None, keepdims=False):
        """
        Returns the smallest value stored in the vector
        """
        assert out is None
        local_min = np.inf
        for i in self._owned_blocks:
            assert self._block_vector[i] is not None
            lmin = self._block_vector[i].min()
            if lmin <= local_min:
                local_min = lmin
        return self._mpiw.allreduce(local_min, op=MPI.MIN)

    def max(self, axis=None, out=None, keepdims=False):
        """
        Returns the largest value stored in the vector
        """
        assert out is None
        local_max = -np.inf
        for i in self._owned_blocks:
            assert self._block_vector[i] is not None
            lmax = self._block_vector[i].max()
            if lmax >= local_max:
                local_max = lmax
        return self._mpiw.allreduce(local_max, op=MPI.MAX)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the sum of all entries in the block vector
        """
        assert out is None
        rank = self._mpiw.Get_rank()
        indices = self._unique_owned_blocks if rank != 0 else self._owned_blocks

        local_sum = 0.0
        for i in indices:
            assert self._block_vector[i] is not None
            local_sum += self._block_vector[i].sum(axis=axis, dtype=dtype)

        return self._mpiw.allreduce(local_sum, op=MPI.SUM)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the product of all entries in the vector
        """
        assert out is None
        rank = self._mpiw.Get_rank()
        indices = self._unique_owned_blocks if rank != 0 else self._owned_blocks

        local_prod = 1.0
        for i in indices:
            assert self._block_vector[i] is not None
            local_prod *= self._block_vector[i].prod(axis=axis, dtype=dtype)
        return self._mpiw.allreduce(local_prod, op=MPI.PROD)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the average of all entries in the vector
        """
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def conj(self):
        """
        Complex-conjugate all elements.
        """
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        for i in self._owned_blocks:
            if self._block_vector[i] is not None:
                result[i] = self._block_vector[i].conj()
            else:
                result[i] = None
        return result

    def conjugate(self):
        """
        Complex-conjugate all elements.
        """
        return self.conj()

    def nonzero(self):
        """
        Return the indices of the elements that are non-zero.
        """
        result = self.copy_structure()

        for i in self._owned_blocks:
            if self._block_vector[i] is not None:
                result[i] = self._block_vector[i].nonzero()[0]
            else:
                result[i] = None
        return (result,)

    def round(self, decimals=0, out=None):
        """
        Return a vector with each element rounded to the given number of decimals
        """
        assert out is None
        result = self.copy_structure()
        for i in self._owned_blocks:
            if self._block_vector[i] is not None:
                result[i] = self._block_vector[i].round(decimals=decimals)
            else:
                result[i] = None
        return result

    def clip(self, min=None, max=None, out=None):
        assert out is None
        result = self.copy_structure()
        for i in self._owned_blocks:
            if self._block_vector[i] is not None:
                result[i] = self._block_vector[i].clip(min=min, max=max)
            else:
                result[i] = None
        return result

    def compress(self, condition, axis=None, out=None):
        assert out is None
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(condition, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            msg = 'BlockVectors must be distributed in same processors'
            assert np.array_equal(self._rank_owner, condition._rank_owner), msg
            assert self._mpiw == condition._mpiw, 'Need to have same communicator'
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].compress(condition[i])
            return result
        if isinstance(condition, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              condition.nblocks)
            assert self.nblocks == condition.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].compress(condition[i])
            return result
        elif isinstance(condition, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        else:
            raise NotImplementedError()

    def copyfrom(self, other):
        """
        Copies entries of other vector into this vector

        Parameters
        ----------
        other: BlockVector or ndarray

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
                        self._block_vector[i] = other[i].copy()
                    elif type(other[i])==np.ndarray:
                        np.copyto(self._block_vector[i], other[i])
                    elif other[i] is None:
                        self._block_vector[i] = None
                    else:
                        raise RuntimeError('Input not recognized')
                elif self._block_vector[i] is None:
                    if isinstance(other[i], np.ndarray):
                        self._block_vector[i] = other[i].copy()
                    elif other[i] is None:
                        self._block_vector[i] = None
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
                        self._block_vector[i] = other[i].copy()
                    elif isinstance(other[i], np.ndarray):
                        np.copyto(self._block_vector[i], other[i])
                    elif other[i] is None:
                        self._block_vector[i] = None
                    else:
                        raise RuntimeError('Input not recognized')
                elif self._block_vector[i] is None:
                    if isinstance(other[i], np.ndarray):
                        self._block_vector[i] = other[i].copy()
                    elif other[i] is None:
                        self._block_vector[i] = None
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
        Copies entries of this vector into other

        Parameters
        ----------
        other: BlockVector or ndarray

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
                        other[i] = None
                elif other[i] is None:
                    if blk is not None:
                        other[i] = blk.copy()
                    else:
                        other[i] = None
                else:
                    raise RuntimeError('Should never get here')
        elif isinstance(other, BlockVector):
            assert other.nblocks == self.nblocks
            if self._need_broadcast_sizes:
                # need to add warning here
                self.broadcast_block_sizes()

            # figure size sent by each processor
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
                other[bid] = block_data

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
            self[i] = blocks[i]

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def take(self, indices, axis=None, out=None, mode='raise'):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def ptp(self, axis=None, out=None, keepdims=False):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def diagonal(self, offset=0, axis1=0, axis2=1):
        raise ValueError('diag requires an array of at least two dimensions')

    def byteswap(self, inplace=False):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def choose(self, choices, out=None, mode='raise'):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def dump(self, file):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def dumps(self):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def getfield(self, dtype, offset=0):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def item(self, *args):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def itemset(self, *args):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def newbyteorder(self, new_order='S'):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def put(self, indices, values, mode='raise'):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def argmax(self, axis=None, out=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def argmin(self, axis=None, out=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def cumprod(self, axis=None, dtype=None, out=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def cumsum(self, axis=None, dtype=None, out=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def clone(self, value=None, copy=True):
        """
        Returns a copy of the block vector

        Parameters
        ----------
        value: scalar (optional)
            all entries of the cloned vector are set to this value
        copy: bool (optinal)
            if set to true makes a deepcopy of each block in this vector. default False

        Returns
        -------
        BlockVector
        """
        result = MPIBlockVector(self.nblocks, self._rank_owner, self._mpiw)
        result._block_vector = self._block_vector.clone(value=value, copy=copy)
        result._need_broadcast_sizes = self._need_broadcast_sizes
        result._done_first_broadcast_sizes = self._done_first_broadcast_sizes
        result._brow_lengths = self._brow_lengths.copy()
        return result

    def copy(self, order='C'):
        v = MPIBlockVector(self.nblocks, self._rank_owner, self._mpiw)
        v._block_vector = self._block_vector.copy()
        v._need_broadcast_sizes = self._need_broadcast_sizes
        v._done_first_broadcast_sizes = self._done_first_broadcast_sizes
        v._brow_lengths = self._brow_lengths.copy()
        return v

    def copy_structure(self, order='C'):
        v = MPIBlockVector(self.nblocks, self._rank_owner, self._mpiw)
        v._block_vector = self._block_vector.copy_structure()
        v._need_broadcast_sizes = self._need_broadcast_sizes
        v._done_first_broadcast_sizes = self._done_first_broadcast_sizes
        v._brow_lengths = self._brow_lengths.copy()
        return v

    def fill(self, value):
        """
        Fills the array with a scalar value.

        Parameters
        ----------
        value : scalar
            All elements in the vector will be assigned this value

        Returns
        -------
        None

        """
        self._block_vector.fill(value)

    def searchsorted(self, v, side='left', sorter=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def setfield(self, val, dtype, offset=0):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def setflags(self, write=None, align=None, uic=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def sort(self, axis=-1, kind='quicksort', order=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def squeeze(self, axis=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def swapaxes(self, axis1, axis2):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def tobytes(self, order='C'):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def partition(self, kth, axis=-1, kind='introselect', order=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def argpartition(self, kth, axis=-1, kind='introselect', order=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def argsort(self, axis=-1, kind='quicksort', order=None):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def tolist(self):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def flatten(self, order='C'):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def ravel(self, order='C'):
        raise RuntimeError('Operation not supported by MPIBlockVector')

    def dot(self, other, out=None):
        """
        Returns dot product

        Parameters
        ----------
        other : ndarray or BlockVector

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
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg

            local_dot_prod = 0.0
            for i in indices:
                local_dot_prod += self._block_vector[i].dot(other[i])
            return self._mpiw.allreduce(local_dot_prod, op=MPI.SUM)
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        else:
            raise NotImplementedError()

    def __add__(self, other):
        rank = self._mpiw.Get_rank()
        result = self.copy_structure()
        if isinstance(other, MPIBlockVector):
            # Note: do not need to check same size? this is checked implicitly
            assert np.array_equal(self._rank_owner, other._rank_owner), \
                'MPIBlockVectors must be distributed in same processors'
            assert self._mpiw == other._mpiw, 'Need to have same communicator'

            for i in self._owned_blocks:
                result[i] = self._block_vector[i] + other[i]
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i] + other[i]
            return result
        elif type(other)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i] + other
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
                result[i] = self._block_vector[i] - other[i]
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i] - other[i]
            return result
        elif type(other)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i] - other
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
                result[i] =  other[i] - self._block_vector[i]
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] =  other[i] - self._block_vector[i]
            return result
        elif type(other)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] =  other - self._block_vector[i]
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
                result[i] = self._block_vector[i].__mul__(other[i])
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__mul__(other[i])
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__mul__(other)
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
                result[i] = self._block_vector[i] / other[i]
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i] / other[i]
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i] / other
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
                result[i] = other[i] / self._block_vector[i]
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = other[i] / self._block_vector[i]
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = other / self._block_vector[i]
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
                result[i] = self._block_vector[i] // other[i]
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i] // other[i]
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i] // other
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
                result[i] = other[i] // self._block_vector[i]
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = other[i] // self._block_vector[i]
            return result
        elif type(other)==np.ndarray:
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = other // self._block_vector[i]
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
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                self._block_vector[i] += other[i]
            return self
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
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                self._block_vector[i] -= other[i]
            return self
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
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                self._block_vector[i] *= other[i]
            return self
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
                self._block_vector[i] = self._block_vector[i] / other[i]
            return self
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                self._block_vector[i] = self._block_vector[i] / other[i]
            return self
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                self._block_vector[i] = self._block_vector[i] / other
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
                result[i] = self._block_vector[i].__le__(other[i])
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__le__(other[i])
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__le__(other)
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
                result[i] = self._block_vector[i].__lt__(other[i])
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__lt__(other[i])
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__lt__(other)
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
                result[i] = self._block_vector[i].__ge__(other[i])
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__ge__(other[i])
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__ge__(other)
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
                result[i] = self._block_vector[i].__gt__(other[i])
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__gt__(other[i])
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__gt__(other)
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
                result[i] = self._block_vector[i].__eq__(other[i])
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__eq__(other[i])
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__eq__(other)
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
                result[i] = self._block_vector[i].__ne__(other[i])
            return result
        elif isinstance(other, BlockVector):
            msg = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                              other.nblocks)
            assert self.nblocks == other.nblocks, msg
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__ne__(other[i])
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Operation not supported by MPIBlockVector')
        elif np.isscalar(other):
            for i in self._owned_blocks:
                result[i] = self._block_vector[i].__ne__(other)
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

    def __getitem__(self, key):

        owner = self._rank_owner[key]
        rank = self._mpiw.Get_rank()
        assert owner == rank or \
               owner < 0, 'Block {} not own by processor {}'.format(key, rank)
        return self._block_vector[key]

    def __setitem__(self, key, value):

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

        self._block_vector[key] = value

    def __iter__(self):
        for j in self._block_vector:
            yield j

    def __str__(self):
        msg = '{}{}:\n'.format(self.__class__.__name__, self.bshape)
        for idx in range(self.nblocks):
            msg += '{}: Owned by processor {}\n'.format(idx, self._rank_owner[idx])

        return msg

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.bshape)

    def pprint(self, root=0):
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

            is_none = '' if global_mask[bid] else '*'
            repn = 'Owned by {} Shape({},){}'.format(owner,
                                                     self._brow_lengths[bid],
                                                     is_none)
            msg += '{}: {}\n'.format(bid, repn)
        if self._mpiw.Get_rank() == root:
            print(msg)



    def __len__(self):
        return self.size

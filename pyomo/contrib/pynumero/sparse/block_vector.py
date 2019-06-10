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
The pyomo.contrib.pynumero.sparse.block_vector module includes methods that extend
linear algebra operations in numpy for case of structured problems
where linear algebra operations present an inherent block structure.
This interface consider vectors of the form:

v = [v_1, v_2, v_3, ... , v_n]

where v_i are numpy arrays of dimension 1

.. rubric:: Contents

"""
from __future__ import division
from .base_block import BaseBlockVector
import numpy as np
import copy as cp

__all__ = ['BlockVector']


class BlockVector(np.ndarray, BaseBlockVector):
    """
    Structured Vector interface

    Parameters
    -------------------
    vectors: int or list of 1d-arrays
    number of blocks contained in the block vector
    if a list is passed the block vector is initialized from
    the list of 1d-arrays

    """

    def __new__(cls, vectors):

        if isinstance(vectors, int):
            blocks = [None for i in range(vectors)]
            block_mask = np.zeros(vectors, dtype=bool)
            brow_lengths = np.zeros(vectors, dtype=np.int64)
            arr = np.asarray(blocks, dtype='object')
            obj = arr.view(cls)
            obj._brow_lengths = np.array(brow_lengths, dtype=np.int64)
            obj._block_mask = block_mask
            obj._nblocks = len(brow_lengths)
            obj._has_none = True
            return obj
        elif isinstance(vectors, list):
            nblocks = len(vectors)
            blocks = [None for i in range(nblocks)]
            block_mask = np.zeros(nblocks, dtype=bool)
            brow_lengths = np.zeros(nblocks, dtype=np.int64)
            arr = np.asarray(blocks, dtype='object')
            obj = arr.view(cls)
            obj._brow_lengths = np.array(brow_lengths, dtype=np.int64)
            obj._block_mask = block_mask
            obj._nblocks = len(brow_lengths)
            obj._has_none = True
            for idx, blk in enumerate(vectors):
                obj[idx] = blk
            return obj
        else:
            raise RuntimeError('Vectors must be a list of an integer')

    def __init__(self, vectors):
        pass

    def __array_finalize__(self, obj):

        if obj is None:
            return
        self._brow_lengths = getattr(obj, '_brow_lengths', None)
        self._nblocks = getattr(obj, '_nblocks', 0)
        self._found_none = getattr(obj, '_has_none', True)

    def __array_prepare__(self, out_arr, context=None):
        return super(BlockVector, self).__array_prepare__(self, out_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        return super(BlockVector, self).__array_wrap__(self, out_arr, context)

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
            raise NotImplementedError(str(ufunc) + "not supported for BlockVector")

    def _unary_operation(self, ufunc, method, *args, **kwargs):
        # ToDo: deal with out
        x = args[0]
        if isinstance(x, BlockVector):
            v = BlockVector(x.nblocks)
            for i in range(x.nblocks):
                _args = [x[i]] + [args[j] for j in range(1, len(args))]
                v[i] = self._unary_operation(ufunc, method, *_args, **kwargs)
            return v
        elif type(x) == np.ndarray:
            return super(BlockVector, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        else:
            raise NotImplementedError()

    def _binary_operation(self, ufunc, method, *args, **kwargs):

        # ToDo: deal with out
        x1 = args[0]
        x2 = args[1]
        if isinstance(x1, BlockVector) and isinstance(x2, BlockVector):
            assert not x1.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert not x2.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert x1.nblocks == x2.nblocks, \
                'Operation on BlockVectors need the same number of blocks on each operand'
            assert x1.size == x2.size, \
                'Dimension missmatch {}!={}'.format(x1.size, x2.size)
            res = BlockVector(x1.nblocks)

            for i in range(x1.nblocks):
                _args = [x1[i]] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif type(x1)==np.ndarray and isinstance(x2, BlockVector):
            assert not x2.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert x1.size == x2.size, \
                'Dimension missmatch {}!={}'.format(x1.size, x2.size)
            res = BlockVector(x2.nblocks)
            accum = 0
            for i in range(x2.nblocks):
                nelements = x2._brow_lengths[i]
                _args = [x1[accum: accum + nelements]] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
                accum += nelements
            return res
        elif type(x2)==np.ndarray and isinstance(x1, BlockVector):
            assert not x1.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert x1.size == x2.size, \
                'Dimension missmatch {}!={}'.format(x1.size, x2.size)
            res = BlockVector(x1.nblocks)
            accum = 0
            for i in range(x1.nblocks):
                nelements = x1._brow_lengths[i]
                _args = [x1[i]] + [x2[accum: accum + nelements]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
                accum += nelements
            return res
        elif np.isscalar(x1) and isinstance(x2, BlockVector):
            assert not x2.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            res = BlockVector(x2.nblocks)
            for i in range(x2.nblocks):
                _args = [x1] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif np.isscalar(x2) and isinstance(x1, BlockVector):
            assert not x1.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            res = BlockVector(x1.nblocks)
            for i in range(x1.nblocks):
                _args = [x1[i]] + [x2] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif (type(x1)==np.ndarray or np.isscalar(x1)) and (type(x2)==np.ndarray or np.isscalar(x2)):
            return super(BlockVector, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(x1, MPIBlockVector):
                return x1.__array_ufunc__(ufunc, method, *args, **kwargs)
            elif isinstance(x2, MPIBlockVector):
                return x2.__array_ufunc__(ufunc, method, *args, **kwargs)
            else:
                raise NotImplementedError()

    @property
    def nblocks(self):
        """
        Returns the number of blocks.
        """
        return self._nblocks

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
        return np.sum(self._brow_lengths),

    @property
    def size(self):
        """
        Returns total number of elements in the block vector
        """
        return np.sum(self._brow_lengths)

    @property
    def ndim(self):
        """
        Returns dimension of the block vector
        """
        return 1

    @property
    def has_none(self):
        # this flag is updated in __setattr__
        return self._has_none

    def block_sizes(self, copy=True):
        """
        Returns array with sizes of individual blocks
        """
        if copy:
            return np.copy(self._brow_lengths)
        return self._brow_lengths

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
        assert not self.has_none, 'Operations not allowed with None blocks.'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operations not allowed with None blocks.'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            return sum(self[i].dot(other[i]) for i in range(self.nblocks))
        elif type(other)==np.ndarray:
            bv = self.flatten()
            return bv.dot(other)
        else:
            raise NotImplementedError()

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the sum of all entries in the block vector
        """
        assert not self.has_none, 'Operations not allowed with None blocks.'
        results = np.array([self[i].sum() for i in range(self.nblocks)])
        return results.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def all(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.
        """
        assert not self.has_none, 'Operations not allowed with None blocks.'
        results = np.array([self[i].all() for i in range(self.nblocks)],
                            dtype=np.bool)
        return results.all(axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.
        """
        assert not self.has_none, 'Operations not allowed with None blocks.'
        results = np.array([self[i].any() for i in range(self.nblocks)],
                            dtype=np.bool)
        return results.any(axis=axis, out=out, keepdims=keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        """
        Returns the largest value stored in the vector
        """
        assert not self.has_none, 'Operations not allowed with None blocks.'
        results = np.array([self[i].max() for i in range(self.nblocks) if self[i].size > 0])
        return results.max(axis=axis, out=out, keepdims=keepdims)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):

        if copy:
            bv = BlockVector(self.nblocks)
            for bid, vv in enumerate(self):
                if self._block_mask[bid]:
                    bv[bid] = vv.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
                else:
                    bv[bid] = None
            return bv
        raise NotImplementedError("astype not implemented for copy=False")

    def clip(self, min=None, max=None, out=None):

        assert not self.has_none, 'Operations not allowed with None blocks.'
        assert out is None, 'Out keyword not supported'

        bv = BlockVector(self.nblocks)
        for bid in range(self.nblocks):
            bv[bid] = self[bid].clip(min=min, max=max, out=None)
        return bv

    def compress(self, condition, axis=None, out=None):

        assert not self._has_none, 'Operations not allowed with None blocks.'
        assert out is None, 'Out keyword not supported'
        result = BlockVector(self.nblocks)

        if isinstance(condition, BlockVector):
            assert not condition.has_none, 'Operations not allowed with None blocks.'
            assert self.shape == condition.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, condition.shape)
            assert self.nblocks == condition.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            condition.nblocks)
            for idx in range(self.nblocks):
                result[idx] = self[idx].compress(condition[idx])
            return result
        elif type(condition)==np.ndarray:
            assert self.shape == condition.shape, \
                'Dimension mismatch {} != {}'.format(self.shape,
                                                     condition.shape)
            accum = 0
            for idx in range(self.nblocks):
                nelements = self._brow_lengths[idx]
                result[idx] = self[idx].compress(condition[accum: accum + nelements])
                accum += nelements
            return result
        else:
            raise NotImplementedError()

    def conj(self):
        """
        Complex-conjugate all elements.
        """
        assert not self._has_none, 'Operations not allowed with None blocks.'
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            result[idx] = self[idx].conj()
        return result

    def conjugate(self):
        """
        Complex-conjugate all elements.
        """
        assert not self._has_none, 'Operations not allowed with None blocks.'
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            result[idx] = self[idx].conjugate()
        return result

    def nonzero(self):
        """
        Return the indices of the elements that are non-zero.
        """
        assert not self._has_none, 'Operations not allowed with None blocks.'
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            result[idx] = self[idx].nonzero()[0]
        return (result,)

    def ptp(self, axis=None, out=None, keepdims=False):
        """
        Peak to peak (maximum - minimum) value along a given axis.
        """
        assert not self._has_none, 'Operations not allowed with None blocks.'
        assert out is None, 'Out keyword not supported'
        return self.max()-self.min()

    def round(self, decimals=0, out=None):
        """
        Return a with each element rounded to the given number of decimals
        """
        assert not self._has_none, 'Operations not allowed with None blocks.'
        assert out is None, 'Out keyword not supported'
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            result[idx] = self[idx].round(decimals=decimals)
        return result

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """
        Returns the standard deviation of the array elements along given axis.
        """
        return self.flatten().std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

    def tofile(self, fid, sep="", format="%s"):
        """
        Write array to a file as text or binary (default).
        """
        self.flatten().tofile(fid, sep=sep, format=format)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        raise NotImplementedError('trace not implemented for BlockVector')

    def min(self, axis=None, out=None, keepdims=False):
        """
        Returns the smallest value stored in the vector
        """
        assert not self._has_none, 'Operations not allowed with None blocks.'
        results = np.array([self[i].min() for i in range(self.nblocks)])
        return results.min(axis=axis, out=out, keepdims=keepdims)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the average of all entries in the vector
        """
        n = self.size
        if n == 0:
            return 0.0
        return self.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)/n

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the product of all entries in the vector
        """
        assert not self._has_none, 'Operations not allowed with None blocks.'
        results = np.array([self[i].prod() for i in range(self.nblocks)])
        return results.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

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
        assert not self._has_none, 'Operations not allowed with None blocks.'
        for i in range(self.nblocks):
            self[i].fill(value)

    def tolist(self):
        """
        Return the vector as a list.

        Returns
        -------
        list

        """
        return self.flatten().tolist()

    def flatten(self, order='C'):
        """
        Returns a copy of the array collapsed into one dimension.
        Parameters
        ----------
        order: : {C, F, A, K}, optional
            C means to flatten in row-major (C-style) order. F means to flatten in column-major (Fortran- style)
            order. A means to flatten in column-major order if a is Fortran contiguous in memory, row-major
            order otherwise. K means to flatten a in the order the elements occur in memory. The default is C.

        Returns
        -------
        ndarray

        """
        assert not self._has_none, 'Operations not allowed with None blocks.'
        all_blocks = tuple(self[i].flatten(order=order) for i in range(self.nblocks))
        return np.concatenate(all_blocks)

    def ravel(self, order='C'):
        """
        Returns a copy of the array collapsed into one dimension.
        Parameters
        ----------
        order: : {C, F, A, K}, optional
            C means to flatten in row-major (C-style) order. F means to flatten in column-major (Fortran- style)
            order. A means to flatten in column-major order if a is Fortran contiguous in memory, row-major
            order otherwise. K means to flatten a in the order the elements occur in memory. The default is C.

        Returns
        -------
        ndarray

        """
        assert not self._has_none, 'Operations not allowed with None blocks.'
        all_blocks = tuple(self[i].ravel(order=order) for i in range(self.nblocks))
        return np.concatenate(all_blocks)

    def cumprod(self, axis=None, dtype=None, out=None):
        """
        Returns the cumulative product of the elements along the given axis.
        """
        flat = self.flatten().cumprod(axis=axis, dtype=dtype, out=out)
        v = self.clone()
        v.copyfrom(flat)
        return v

    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Returns the cumulative sum of the elements along the given axis.
        """
        flat = self.flatten().cumsum(axis=axis, dtype=dtype, out=out)
        v = self.clone()
        v.copyfrom(flat)
        return v

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
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            if copy:
                result[idx] = cp.deepcopy(self[idx])
            else:
                result[idx] = self[idx]
            result._block_mask[idx] = self._block_mask[idx]
            result._brow_lengths[idx] = self._brow_lengths[idx]
        if value is not None:
            result.fill(value)
        return result

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

        if isinstance(other, BlockVector):
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx in range(other.nblocks):
                if isinstance(self[idx], BlockVector):
                    self[idx].copyfrom(other[idx])
                elif isinstance(self[idx], np.ndarray):
                    if isinstance(other[idx], BlockVector):
                        self[idx] = other[idx].copy()
                    elif isinstance(other[idx], np.ndarray):
                        np.copyto(self[idx], other[idx])
                    elif blk is None:
                        self[idx] = None
                    else:
                        raise RuntimeError('Input not recognized')
                elif self[idx] is None:
                    if isinstance(other[idx], np.ndarray):
                        # this inlcude block vectors too
                        self[idx] = other[idx].copy()
                    elif blk is None:
                        self[idx] = None
                    else:
                        raise RuntimeError('Input not recognized')
                else:
                    raise RuntimeError('Should never get here')
        elif isinstance(other, np.ndarray):

            assert not self.has_none, \
                'Operation not allowed with None blocks. Specify all blocks'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)

            offset = 0
            for idx in range(self.nblocks):
                subarray = other[offset: offset + self[idx].size]
                if isinstance(self[idx], BlockVector):
                    self[idx].copyfrom(subarray)
                else:
                    np.copyto(self[idx], subarray)
                offset += self[idx].size
        else:
            raise NotImplementedError()

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

        if isinstance(other, BlockVector):
            msgj = 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                               other.nblocks)
            assert self.nblocks == other.nblocks, msgj
            for idx in range(self.nblocks):
                if isinstance(other[idx], BlockVector):
                    other[idx].copyfrom(self[idx])
                elif isinstance(other[idx], np.ndarray):
                    if self[idx] is not None:
                        np.copyto(other[idx], self[idx].flatten())
                    else:
                        other[idx] = None
                elif other[idx] is None:
                    if self[idx] is not None:
                        other[idx] = self[idx].copy()
                    else:
                        other[idx] = None
                else:
                    raise RuntimeError('Should never get here')

        elif isinstance(other, np.ndarray):
            np.copyto(other, self.flatten())
        else:
            raise NotImplementedError()

    def copy(self, order='C'):
        bv = BlockVector(self.nblocks)
        for bid in range(self.nblocks):
            if self._block_mask[bid]:
                bv[bid] = self[bid].copy(order=order)
        return bv

    def copy_structure(self):
        bv = BlockVector(self.nblocks)
        for bid in range(self.nblocks):
            if self[bid] is not None:
                if isinstance(self._block_mask[bid], BlockVector):
                    bv[bid] = self[bid].copy_structure()
                elif type(self[bid]) == np.ndarray:
                    bv[bid] = np.zeros(self[bid].size, dtype=self[bid].dtype)
                else:
                    raise NotImplementedError('Should never get here')
        return bv

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
        assert isinstance(blocks, list), \
            'blocks should be passed in ordered list'
        assert len(blocks) == self.nblocks, \
            'More blocks passed than allocated {} != {}'.format(len(blocks),
                                                                self.nblocks)
        for idx, blk in enumerate(blocks):
            self[idx] = blk

    def __add__(self, other):

        result = BlockVector(self.nblocks)
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk + other[idx]
            return result
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk + other[accum: accum + nelements]
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = blk + other
            return result
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)
                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)
                for i in other.owned_blocks:
                    result[i] = self[i] + other[i]
                return result
            raise NotImplementedError()

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):
        result = BlockVector(self.nblocks)
        assert not self.has_none, \
            'Operation not allowed with None blocks. ' \
            'Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk - other[idx]
            return result
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk - other[accum: accum + nelements]
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = blk - other
            return result
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)

                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)
                for i in other.owned_blocks:
                    result[i] = self[i] - other[i]
                return result
            raise NotImplementedError()

    def __rsub__(self, other):  # other - self

        result = BlockVector(self.nblocks)
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = other[idx] - blk
            return result

        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = other[accum: accum + nelements] - blk
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = other - blk
            return result
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)

                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)
                for i in other.owned_blocks:
                    result[i] = other[i] - self[i]
                return result
            raise NotImplementedError()

    def __mul__(self, other):

        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk .__mul__(other[idx])
            return result
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.__mul__(other[accum: accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = blk.__mul__(other)
            return result
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)

                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)

                for i in other.owned_blocks:
                    result[i] = other[i].__mul__(self[i])
                return result
            raise NotImplementedError()

    def __rmul__(self, other):  # other + self
        return self.__mul__(other)

    def __truediv__(self, other):
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk / other[idx]
            return result
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk / other[accum: accum + nelements]
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = blk / other
            return result
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)

                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)

                for i in other.owned_blocks:
                    result[i] = self[i] / other[i]
                return result
            raise NotImplementedError()

    def __rtruediv__(self, other):
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = other[idx] / blk
            return result
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = other[accum: accum + nelements] / blk
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = other / blk
            return result
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)

                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)

                for i in other.owned_blocks:
                    result[i] = other[i] / self[i]
                return result
            raise NotImplementedError()

    def __floordiv__(self, other):
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk // other[idx]
            return result
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk // other[accum: accum + nelements]
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = blk // other
            return result
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)

                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)

                for i in other.owned_blocks:
                    result[i] = self[i] // other[i]
                return result
            raise NotImplementedError()

    def __rfloordiv__(self, other):
        assert not self.has_none, \
            'Operation not allowed with None blocks. '\
            'Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = other[idx] // blk
            return result
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = other[accum: accum + nelements] // blk
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = other // blk
            return result
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)

                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)

                for i in other.owned_blocks:
                    result[i] =  other[i] // self[i]
                return result
            raise NotImplementedError()

    def __iadd__(self, other):
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                self[idx] = self[idx] + other # maybe it suffice with doing self[idx] = self[idf] + other
            return self
        elif isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                self[idx] = self[idx] + other[idx]
            return self
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                self[idx] = blk + other[accum: accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __isub__(self, other):
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                self[idx] = self[idx] - other
            return self
        elif isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                self[idx] = self[idx] - other[idx]
            return self
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                self[idx] = blk - other[accum: accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __imul__(self, other):
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                self[idx] = self[idx] * other
            return self
        elif isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                self[idx] = self[idx] * other[idx]
            return self
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                self[idx] = blk * other[accum: accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __itruediv__(self, other):
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                self[idx] = self[idx] / other
            return self
        elif isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                'Specify all blocks in BlockVector'
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, \
                'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                            other.nblocks)
            for idx, blk in enumerate(self):
                self[idx] = self[idx] / other[idx]
            return self
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                self[idx] = blk / other[accum: accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __str__(self):
        msg = ''
        for idx in range(self.bshape[0]):
            if isinstance(self[idx], BlockVector):
                repn = self[idx].__repr__()
                repn += '\n'
                for j, vv in enumerate(self[idx]):
                    if isinstance(vv, BlockVector):
                        repn += '   {}: {}\n'.format(j, vv.__repr__())
                        repn += '\n'
                        for jj, vvv in enumerate(vv):
                            if isinstance(vv, BlockVector):
                                repn += '      {}: {}\n'.format(jj, vvv.__repr__())
                            else:
                                repn += '      {}: array({})\n'.format(jj, vvv.size)
                    else:
                        repn += '   {}: array({})\n'.format(j, vv.size)
            elif isinstance(self[idx], np.ndarray):
                repn = "array({})".format(self[idx].size)
            elif self[idx] is None:
                repn = None
            else:
                raise NotImplementedError("Should not get here")
            msg += '{}: {}\n'.format(idx, repn)
        return msg

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.bshape)

    def __getitem__(self, item):

        assert not isinstance(item, slice), 'Slicing not supported for BlockVector'
        return super(BlockVector, self).__getitem__(item)

    def __setitem__(self, key, value):

        assert not isinstance(key, slice), 'Slicing not supported for BlockVector'
        assert -self.nblocks < key < self.nblocks, 'out of range'
        if value is None:
            super(BlockVector, self).__setitem__(key, None)
            self._block_mask[key] = False
            self._brow_lengths[key] = 0
            self._has_none = True
        else:
            assert isinstance(value, np.ndarray) or \
                isinstance(value, BaseBlockVector), \
                    'Blocks need to be numpy arrays or BlockVectors'
            assert value.ndim == 1, 'Blocks need to be 1D'
            super(BlockVector, self).__setitem__(key, value)
            self._block_mask[key] = True
            self._brow_lengths[key] = value.size

            # check if needs to update has none flag
            if self._has_none:
                # if there was a single none and after setting 'value'
                # that none is gone the flag is updated to false.
                self._has_none = not self._block_mask.all()

    def __le__(self, other):

        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            flags = [vv.__le__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            result = BlockVector(self.nblocks)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.__le__(other[accum: accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            flags = [vv.__le__(other) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)
                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)
                for i in other.owned_blocks:
                    result[i] = self[i].__le__(other[i])
                return result
            raise NotImplementedError()

    def __lt__(self, other):

        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                'Specify all blocks in BlockVector'
            flags = [vv.__lt__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            result = BlockVector(self.nblocks)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.__lt__(other[accum: accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            flags = [vv.__lt__(other) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)
                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)
                for i in other.owned_blocks:
                    result[i] = self[i].__lt__(other[i])
                return result
            raise NotImplementedError()

    def __ge__(self, other):

        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            flags = [vv.__ge__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            result = BlockVector(self.nblocks)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.__ge__(other[accum: accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            flags = [vv.__ge__(other) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)
                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)
                for i in other.owned_blocks:
                    result[i] = self[i].__ge__(other[i])
                return result
            raise NotImplementedError()

    def __gt__(self, other):

        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            flags = [vv.__gt__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            result = BlockVector(self.nblocks)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.__gt__(other[accum: accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            flags = [vv.__gt__(other) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)
                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)
                for i in other.owned_blocks:
                    result[i] = self[i].__gt__(other[i])
                return result
            raise NotImplementedError()

    def __eq__(self, other):

        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            flags = [vv.__eq__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            result = BlockVector(self.nblocks)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.__eq__(other[accum: accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            flags = [vv.__eq__(other) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)
                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)
                for i in other.owned_blocks:
                    result[i] = other[i].__eq__(self[i])
                return result
            raise NotImplementedError()

    def __ne__(self, other):

        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, \
                'Operation not allowed with None blocks.' \
                ' Specify all blocks in BlockVector'
            flags = [vv.__ne__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif type(other)==np.ndarray:
            assert self.shape == other.shape, \
                'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            result = BlockVector(self.nblocks)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.__ne__(other[accum: accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            flags = [vv.__ne__(other) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        else:
            from .mpi_block_vector import MPIBlockVector
            if isinstance(other, MPIBlockVector):
                result = MPIBlockVector(other.nblocks, other.rank_ownership, other._mpiw)
                assert self.nblocks == other.nblocks, \
                    'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                other.nblocks)
                for i in other.owned_blocks:
                    result[i] = other[i].__ne__(self[i])
                return result
            raise NotImplementedError()

    def __contains__(self, item):
        other = item
        assert not self.has_none, \
            'Operation not allowed with None blocks.' \
            ' Specify all blocks in BlockVector'
        if np.isscalar(other):
            contains = False
            for idx, blk in enumerate(self):
                if blk.__contains__(other):
                    return True
            return contains
        else:
            raise NotImplementedError()

    def __len__(self):
        return self.nblocks

    # the following methods are not supported by blockvector

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

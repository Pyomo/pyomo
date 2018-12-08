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
import numpy as np
import copy as cp

__all__ = ['BlockVector']


class BlockVector(np.ndarray):
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

        # Note: this for now just flatten the inputs and call super

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
        # args = []
        # for i, input_ in enumerate(inputs):
        #     if isinstance(input_, BlockVector):
        #         args.append(input_.flatten())
        #     else:
        #         args.append(input_)
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
            #results = super(BlockVector, self).__array_ufunc__(ufunc, method,
            #                                                   *args, **kwargs)
        # if results is NotImplemented:
        #     return NotImplemented
        #
        # if method == 'at':
        #     raise NotImplementedError()
        #
        # if ufunc.nout == 1:
        #     results = (results,)
        #
        # return results

    def _unary_operation(self, ufunc, method, *args, **kwargs):
        # ToDo: deal with out
        x = args[0]
        if isinstance(x, BlockVector):
            v = BlockVector(x.nblocks)
            for i in range(x.nblocks):
                _args = [x[i]] + [args[j] for j in range(1, len(args))]
                v[i] = self._unary_operation(ufunc, method, *_args, **kwargs)
            return v
        elif isinstance(x, np.ndarray):
            return super(BlockVector, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
        else:
            raise NotImplementedError()

    def _binary_operation(self, ufunc, method, *args, **kwargs):
        # ToDo: deal with out
        x1 = args[0]
        x2 = args[1]
        if isinstance(x1, BlockVector) and isinstance(x2, BlockVector):
            assert not x1.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert not x2.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert x1.nblocks == x2.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
            assert x1.size == x2.size, 'Dimension missmatch {}!={}'.format(x1.size, x2.size)
            res = BlockVector(x1.nblocks)

            for i in range(x1.nblocks):
                _args = [x1[i]] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif isinstance(x1, np.ndarray) and isinstance(x2, BlockVector):
            assert not x2.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert x1.size == x2.size, 'Dimension missmatch {}!={}'.format(x1.size, x2.size)
            res = BlockVector(x2.nblocks)
            accum = 0
            for i in range(x2.nblocks):
                nelements = x2._brow_lengths[i]
                _args = [x1[accum: accum + nelements]] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
                accum += nelements
            return res
        elif isinstance(x2, np.ndarray) and isinstance(x1, BlockVector):
            assert not x1.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert x1.size == x2.size, 'Dimension missmatch {}!={}'.format(x1.size, x2.size)
            res = BlockVector(x1.nblocks)
            accum = 0
            for i in range(x1.nblocks):
                nelements = x1._brow_lengths[i]
                _args = [x1[i]] + [x2[accum: accum + nelements]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
                accum += nelements
            return res
        elif np.isscalar(x1) and isinstance(x2, BlockVector):
            assert not x2.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            res = BlockVector(x2.nblocks)
            for i in range(x2.nblocks):
                _args = [x1] + [x2[i]] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif np.isscalar(x2) and isinstance(x1, BlockVector):
            assert not x1.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            res = BlockVector(x1.nblocks)
            for i in range(x1.nblocks):
                _args = [x1[i]] + [x2] + [args[j] for j in range(2, len(args))]
                res[i] = self._binary_operation(ufunc, method, *_args, **kwargs)
            return res
        elif (isinstance(x1, np.ndarray) or np.isscalar(x1)) and (isinstance(x2, np.ndarray) or np.isscalar(x2)):
            return super(BlockVector, self).__array_ufunc__(ufunc, method,
                                                            *args, **kwargs)
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

    @shape.setter
    def shape(self, new_shape):
        raise NotImplementedError("BlockVector does not support reshaping")

    @property
    def size(self):
        """
        Returns total number of elements in the block vector
        """
        return np.sum(self._brow_lengths)

    @size.setter
    def size(self, new_size):
        raise NotImplementedError("BlockVector does not support resizing")

    @property
    def ndim(self):
        """
        Returns dimension of the block vector
        """
        return 1

    @property
    def has_none(self):
        if not self._has_none:
            return False
        if not np.all(self._block_mask):
            return True

        block_arr = np.array([blk.has_none for blk in self if isinstance(blk, BlockVector)], dtype=bool)
        it_has = np.any(block_arr)
        self._has_none = it_has
        return it_has

    def block_sizes(self):
        """
        Returns array with sizes of individual blocks
        """
        return np.copy(self._brow_lengths)

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
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            return sum(self[i].dot(other[i]) for i in range(self.nblocks))
        elif isinstance(other, np.ndarray):
            bv = self.flatten()
            return bv.dot(other)
        else:
            raise NotImplementedError()

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the sum of all entries in the block vector
        """
        return sum(self[i].sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
                   for i in range(self.nblocks) if self._block_mask[i])

    def all(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.
        """
        d = tuple(v for v in self if v is not None)
        arr = np.concatenate(d)
        return arr.all(axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.
        """
        d = tuple(v for v in self if v is not None)
        arr = np.concatenate(d)
        return arr.any(axis=axis, out=out, keepdims=keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        """
        Returns the largest value stored in the vector
        """
        return max([self[i].max(axis=axis, out=None, keepdims=keepdims)
                   for i in range(self.nblocks) if self._block_mask[i]])

    def argpartition(self, kth, axis=-1, kind='introselect', order=None):
        raise NotImplementedError("argpartition not implemented for BlockVector")

    def argsort(self, axis=-1, kind='quicksort', order=None):
        raise NotImplementedError("argsort not implemented for BlockVector")

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

    def byteswap(self, inplace=False):
        raise NotImplementedError("byteswap not implemented for BlockVector")

    def choose(self, choices, out=None, mode='raise'):
        raise NotImplementedError("choose not implemented for BlockVector")

    def clip(self, min=None, max=None, out=None):

        if out is not None:
            raise NotImplementedError()

        bv = BlockVector(self.nblocks)
        for bid, vv in enumerate(self):
            if self._block_mask[bid]:
                bv[bid] = vv.clip(min=min, max=max, out=None)
            else:
                bv[bid] = None
        return bv

    def compress(self, condition, axis=None, out=None):
        if out is not None:
            raise NotImplementedError('compress not supported with out')
        result = BlockVector(self.nblocks)
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(condition, BlockVector):
            assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == condition.shape, 'Dimension mismatch {} != {}'.format(self.shape, condition.shape)
            assert self.nblocks == condition.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                                  condition.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk.compress(condition[idx])
            return result
        elif isinstance(condition, np.ndarray):
            assert self.shape == condition.shape, 'Dimension mismatch {} != {}'.format(self.shape,
                                                                                       condition.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.compress(condition[accum: accum + nelements])
                accum += nelements
            return result
        else:
            raise NotImplementedError()

    def conj(self):
        """
        Complex-conjugate all elements.
        """
        result = BlockVector(self.nblocks)
        for idx, blk in enumerate(self):
            if self._block_mask[idx]:
                result[idx] = blk.conj()
            else:
                result[idx] = None
        return result

    def conjugate(self):
        """
        Complex-conjugate all elements.
        """
        result = BlockVector(self.nblocks)
        for idx, blk in enumerate(self):
            if self._block_mask[idx]:
                result[idx] = blk.conjugate()
            else:
                result[idx] = None
        return result

    def diagonal(self, offset=0, axis1=0, axis2=1):
        raise ValueError('diag requires an array of at least two dimensions')

    def dump(self, file):
        raise NotImplementedError('TODO')

    def dumps(self):
        raise NotImplementedError('TODO')

    def getfield(self, dtype, offset=0):
        raise NotImplementedError('getfield not implemented for BlockVector')

    def item(self, *args):
        raise NotImplementedError('item not implemented for BlockVector')

    def itemset(self, *args):
        raise NotImplementedError('itemset not implemented for BlockVector')

    def newbyteorder(self, new_order='S'):
        raise NotImplementedError('newbyteorder not implemented for BlockVector')

    def nonzero(self):
        """
        Return the indices of the elements that are non-zero.
        """
        result = BlockVector(self.nblocks)
        for idx, blk in enumerate(self):
            if self._block_mask[idx]:
                result[idx] = blk.nonzero()[0]
            else:
                result[idx] = None
        return (result,)

    def ptp(self, axis=None, out=None, keepdims=False):
        """
        Peak to peak (maximum - minimum) value along a given axis.
        """
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        return self.flatten().ptp(axis=axis, out=out)

    def put(self, indices, values, mode='raise'):
        raise NotImplementedError('TODO')

    def partition(self, kth, axis=-1, kind='introselect', order=None):
        raise NotImplementedError('partition not implemented for BlockVector')

    def repeat(self, repeats, axis=None):
        raise NotImplementedError('repeat not implemented for BlockVector')

    def reshape(self, shape, order='C'):
        raise NotImplementedError('reshape not implemented for BlockVector')

    def resize(self, new_shape, refcheck=True):
        raise NotImplementedError('resize not implemented for BlockVector')

    def round(self, decimals=0, out=None):
        """
        Return a with each element rounded to the given number of decimals
        """
        if out is not None:
            raise NotImplementedError('round not implemented with out input')
        result = BlockVector(self.nblocks)
        for idx, blk in enumerate(self):
            if self._block_mask[idx]:
                result[idx] = blk.round(decimals=0, out=None)
            else:
                result[idx] = None
        return result

    def searchsorted(self, v, side='left', sorter=None):
        raise NotImplementedError('searchsorted not implemented for BlockVector')

    def setfield(self, val, dtype, offset=0):
        raise NotImplementedError('setfield not implemented for BlockVector')

    def setflags(self, write=None, align=None, uic=None):
        raise NotImplementedError('setflags not implemented for BlockVector')

    def sort(self, axis=-1, kind='quicksort', order=None):
        raise NotImplementedError('sort not implemented for BlockVector')

    def squeeze(self, axis=None):
        raise NotImplementedError('squeeze not implemented for BlockVector')

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """
        Returns the standard deviation of the array elements along given axis.
        """
        return self.flatten().std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

    def swapaxes(self, axis1, axis2):
        raise NotImplementedError('swapaxes not implemented for BlockVector')

    def take(self, indices, axis=None, out=None, mode='raise'):
        """
        Return an array formed from the elements of a at the given indices.
        """
        return self.flatten().take(indices, axis=axis, out=out, mode=mode)

    def tobytes(self, order='C'):
        raise NotImplementedError('tobytes not implemented for BlockVector')

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
        return min([self[i].min(axis=axis, out=None, keepdims=keepdims)
                   for i in range(self.nblocks) if self._block_mask[i]])

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
        arr = [self[i].prod(axis=axis, dtype=dtype, out=None, keepdims=keepdims)
               for i in range(self.nblocks) if self._block_mask[i]]
        return np.prod(arr)

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
        for i in range(self.nblocks):
            if self._block_mask[i]:
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
        all_blocks = tuple(v.flatten(order=order) for v in self)
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
        all_blocks = tuple(v.ravel(order=order) for v in self)
        return np.concatenate(all_blocks)

    def argmax(self, axis=None, out=None):
        """
        Returns the index of the largest element.
        """
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        return self.flatten().argmax(axis=axis, out=out)

    def argmin(self, axis=None, out=None):
        """
        Returns the index of the smallest element.
        """
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        return self.flatten().argmin(axis=axis, out=out)

    def cumprod(self, axis=None, dtype=None, out=None):
        """
        Returns the cumulative product of the elements along the given axis.
        """
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        flat = self.flatten().cumprod(axis=axis, dtype=dtype, out=out)
        v = self.clone()
        v.copyfrom(flat)
        return v

    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Returns the cumulative sum of the elements along the given axis.
        """
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
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
        for idx, blk in enumerate(self):
            if copy:
                result[idx] = cp.deepcopy(blk)
            else:
                result[idx] = blk
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
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(other):
                if isinstance(blk, BlockVector) or isinstance(self[idx], BlockVector):
                    self[idx].copyfrom(blk)
                else:
                    np.copyto(self[idx], blk)
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)

            offset = 0
            for idx, blk in enumerate(self):
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
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                if other[idx] is not None:
                    msgi = 'Dimension mismatch in subblock {} {} != {}'
                    assert other[idx].shape == blk.shape, msgi.format(idx,
                                                                      blk.shape,
                                                                      other[idx].shape)
                if isinstance(blk, BlockVector):
                    other[idx] = blk.clone(copy=True)
                else:
                    other[idx] = cp.deepcopy(blk)

        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            np.copyto(other, self.flatten())

        else:
            raise NotImplementedError()

    def copy(self, order='C'):
        bv = BlockVector(self.nblocks)
        for bid, vv in enumerate(self):
            if self._block_mask[bid]:
                bv[bid] = vv.copy(order=order)
            else:
                bv[bid] = None
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
        assert isinstance(blocks, list), 'blocks should be passed in ordered list'
        msg = 'More blocks passed than allocated {} != {}'.format(len(blocks), self.nblocks)
        assert len(blocks) == self.nblocks, msg
        for idx, blk in enumerate(blocks):
            self[idx] = blk

    def _check_mask(self):
        msg = 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        msg += '\n{}'.format(self.__str__())
        if not np.all(self._block_mask):
            raise RuntimeError(msg)
        for idx, blk in enumerate(self):
            if isinstance(blk, BlockVector):
                blk._check_mask()

    def __add__(self, other):
        result = BlockVector(self.nblocks)
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk + other[idx]
            return result
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):
        result = BlockVector(self.nblocks)
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk - other[idx]
            return result
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __rsub__(self, other):  # other - self
        result = BlockVector(self.nblocks)
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = other[idx] - blk
            return result
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __mul__(self, other):
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk .__mul__(other[idx])
            return result
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __rmul__(self, other):  # other + self
        return self.__mul__(other)

    def __truediv__(self, other):
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk.__truediv__(other[idx])
            return result
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.__truediv__(other[accum: accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = blk.__truediv__(other)
            return result
        else:
            raise NotImplementedError()

    def __rtruediv__(self, other):
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = other[idx].__rtruediv__(blk)
            return result
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = other[accum: accum + nelements].__rtruediv__(blk)
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = other.__rtruediv__(blk)
            return result
        else:
            raise NotImplementedError()

    def __floordiv__(self, other):
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = blk.__floordiv__(other[idx])
            return result
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = blk.__floordiv__(other[accum: accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = blk.__floordiv__(other)
            return result
        else:
            raise NotImplementedError()

    def __rfloordiv__(self, other):
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                result[idx] = other[idx].__rfloordiv__(blk)
            return result
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result[idx] = other[accum: accum + nelements].__rfloordiv__(blk)
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result[idx] = other.__rfloordiv__(blk)
            return result
        else:
            raise NotImplementedError()

    def __iadd__(self, other):
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                self[idx] = self[idx] + other # maybe it suffice with doing self[idx] = self[idf] + other
            return self
        elif isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                self[idx] = self[idx] + other[idx]
            return self
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                self[idx] = blk + other[accum: accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __isub__(self, other):
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                self[idx] = self[idx] - other
            return self
        elif isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                self[idx] = self[idx] - other[idx]
            return self
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                self[idx] = blk - other[accum: accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __imul__(self, other):
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                self[idx] = self[idx] * other
            return self
        elif isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                self[idx] = self[idx] * other[idx]
            return self
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                self[idx] = blk * other[accum: accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __itruediv__(self, other):
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                self[idx] = self[idx] / other
            return self
        elif isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self):
                self[idx] = self[idx] / other[idx]
            return self
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            elif isinstance(self[idx], np.ndarray):
                repn = "array({})".format(self[idx].size)
            elif self[idx] is None:
                repn = None
            else:
                raise NotImplementedError("Should not get here")
            msg += '{}: {}\n'.format(idx, repn)
        return msg

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.shape)

    def __getitem__(self, item):

        if np.isscalar(item):
            return super(BlockVector, self).__getitem__(item)

        # deal with slices
        arr = self.flatten()
        return arr[item]

    def __setitem__(self, key, value):

        if isinstance(key, slice):
            raise NotImplementedError()

        assert -self.nblocks < key < self.nblocks, 'out of range'
        if value is None:
            super(BlockVector, self).__setitem__(key, None)
            self._block_mask[key] = False
            self._brow_lengths[key] = 0
            self._has_none = True
        else:
            msg = 'Blocks need to be numpy arrays or BlockVectors'
            assert isinstance(value, np.ndarray) or isinstance(value, BlockVector), msg
            assert value.ndim == 1, 'Blocks need to be 1D'
            super(BlockVector, self).__setitem__(key, value)
            self._block_mask[key] = True
            self._brow_lengths[key] = value.size

    def __le__(self, other):

        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            flags = [vv.__le__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __lt__(self, other):

        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            flags = [vv.__lt__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __ge__(self, other):

        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            flags = [vv.__ge__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __gt__(self, other):

        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            flags = [vv.__gt__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __eq__(self, other):

        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            flags = [vv.__eq__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __ne__(self, other):

        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if isinstance(other, BlockVector):
            assert not other.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            flags = [vv.__ne__(other[bid]) for bid, vv in enumerate(self)]
            bv = BlockVector(flags)
            return bv
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
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
            raise NotImplementedError()

    def __contains__(self, item):
        other = item
        assert not self.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        if np.isscalar(other):
            contains = False
            for idx, blk in enumerate(self):
                if blk.__contains__(other):
                    return True
            return contains
        else:
            raise NotImplementedError()

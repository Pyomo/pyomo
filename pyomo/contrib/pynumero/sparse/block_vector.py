#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""Implementation of a general "block vector"


The `pyomo.contrib.pynumero.sparse.block_vector` module includes methods
that extend linear algebra operations in numpy for case of structured
problems where linear algebra operations present an inherent block
structure.  This interface consider vectors of the form:

.. math::

   v = [v_1, v_2, v_3, ... , v_n]

where `v_i` are numpy arrays of dimension 1

.. rubric:: Contents

Methods specific to :py:class:`BlockVector`:

   * :py:meth:`~BlockVector.set_block`
   * :py:meth:`~BlockVector.get_block`
   * :py:meth:`~BlockVector.block_sizes`
   * :py:meth:`~BlockVector.get_block_size`
   * :py:meth:`~BlockVector.is_block_defined`
   * :py:meth:`~BlockVector.copyfrom`
   * :py:meth:`~BlockVector.copyto`
   * :py:meth:`~BlockVector.copy_structure`
   * :py:meth:`~BlockVector.set_blocks`
   * :py:meth:`~BlockVector.pprint`

Attributes specific to :py:class:`BlockVector`:

   * :py:attr:`~BlockVector.nblocks`
   * :py:attr:`~BlockVector.bshape`
   * :py:attr:`~BlockVector.has_none`


NumPy compatible methods:

   * :py:meth:`~numpy.ndarray.dot`
   * :py:meth:`~numpy.ndarray.sum`
   * :py:meth:`~numpy.ndarray.all`
   * :py:meth:`~numpy.ndarray.any`
   * :py:meth:`~numpy.ndarray.max`
   * :py:meth:`~numpy.ndarray.astype`
   * :py:meth:`~numpy.ndarray.clip`
   * :py:meth:`~numpy.ndarray.compress`
   * :py:meth:`~numpy.ndarray.conj`
   * :py:meth:`~numpy.ndarray.conjugate`
   * :py:meth:`~numpy.ndarray.nonzero`
   * :py:meth:`~numpy.ndarray.ptp` (NumPy 1.x only)
   * :py:meth:`~numpy.ndarray.round`
   * :py:meth:`~numpy.ndarray.std`
   * :py:meth:`~numpy.ndarray.var`
   * :py:meth:`~numpy.ndarray.tofile`
   * :py:meth:`~numpy.ndarray.min`
   * :py:meth:`~numpy.ndarray.mean`
   * :py:meth:`~numpy.ndarray.prod`
   * :py:meth:`~numpy.ndarray.fill`
   * :py:meth:`~numpy.ndarray.tolist`
   * :py:meth:`~numpy.ndarray.flatten`
   * :py:meth:`~numpy.ndarray.ravel`
   * :py:meth:`~numpy.ndarray.argmax`
   * :py:meth:`~numpy.ndarray.argmin`
   * :py:meth:`~numpy.ndarray.cumprod`
   * :py:meth:`~numpy.ndarray.cumsum`
   * :py:meth:`~numpy.ndarray.copy`

For example,

.. code-block:: python

   >>> import numpy as np
   >>> from pyomo.contrib.pynumero.sparse import BlockVector
   >>> v = BlockVector(2)
   >>> v.set_block(0, np.random.normal(size=100))
   >>> v.set_block(1, np.random.normal(size=30))
   >>> avg = v.mean()

NumPy compatible functions:

   * :py:func:`~numpy.log10`
   * :py:func:`~numpy.sin`
   * :py:func:`~numpy.cos`
   * :py:func:`~numpy.exp`
   * :py:func:`~numpy.ceil`
   * :py:func:`~numpy.floor`
   * :py:func:`~numpy.tan`
   * :py:func:`~numpy.arctan`
   * :py:func:`~numpy.arcsin`
   * :py:func:`~numpy.arccos`
   * :py:func:`~numpy.sinh`
   * :py:func:`~numpy.cosh`
   * :py:func:`~numpy.abs`
   * :py:func:`~numpy.tanh`
   * :py:func:`~numpy.arccosh`
   * :py:func:`~numpy.arcsinh`
   * :py:func:`~numpy.arctanh`
   * :py:func:`~numpy.fabs`
   * :py:func:`~numpy.sqrt`
   * :py:func:`~numpy.log`
   * :py:func:`~numpy.log2`
   * :py:func:`~numpy.absolute`
   * :py:func:`~numpy.isfinite`
   * :py:func:`~numpy.isinf`
   * :py:func:`~numpy.isnan`
   * :py:func:`~numpy.log1p`
   * :py:func:`~numpy.logical_not`
   * :py:func:`~numpy.expm1`
   * :py:func:`~numpy.exp2`
   * :py:func:`~numpy.sign`
   * :py:func:`~numpy.rint`
   * :py:func:`~numpy.square`
   * :py:func:`~numpy.positive`
   * :py:func:`~numpy.negative`
   * :py:func:`~numpy.rad2deg`
   * :py:func:`~numpy.deg2rad`
   * :py:func:`~numpy.conjugate`
   * :py:func:`~numpy.reciprocal`
   * :py:func:`~numpy.signbit`
   * :py:func:`~numpy.add`
   * :py:func:`~numpy.multiply`
   * :py:func:`~numpy.divide`
   * :py:func:`~numpy.subtract`
   * :py:func:`~numpy.greater`
   * :py:func:`~numpy.greater_equal`
   * :py:func:`~numpy.less`
   * :py:func:`~numpy.less_equal`
   * :py:func:`~numpy.not_equal`
   * :py:func:`~numpy.maximum`
   * :py:func:`~numpy.minimum`
   * :py:func:`~numpy.fmax`
   * :py:func:`~numpy.fmin`
   * :py:func:`~numpy.equal`
   * :py:func:`~numpy.logical_and`
   * :py:func:`~numpy.logical_or`
   * :py:func:`~numpy.logical_xor`
   * :py:func:`~numpy.logaddexp`
   * :py:func:`~numpy.logaddexp2`
   * :py:func:`~numpy.remainder`
   * :py:func:`~numpy.heaviside`
   * :py:func:`~numpy.hypot`

For example,

.. code-block:: python

   >>> import numpy as np
   >>> from pyomo.contrib.pynumero.sparse import BlockVector
   >>> v = BlockVector(2)
   >>> v.set_block(0, np.random.normal(size=100))
   >>> v.set_block(1, np.random.normal(size=30))
   >>> inf_norm = np.max(np.abs(v))

.. autosummary::

   BlockVector
   BlockVector.set_block
   BlockVector.get_block
   BlockVector.block_sizes
   BlockVector.get_block_size
   BlockVector.is_block_defined
   BlockVector.copyfrom
   BlockVector.copyto
   BlockVector.copy_structure
   BlockVector.set_blocks
   BlockVector.pprint
   BlockVector.nblocks
   BlockVector.bshape
   BlockVector.has_none

"""

import operator

from ..dependencies import numpy as np
from .base_block import (
    BaseBlockVector,
    vec_unary_ufuncs,
    vec_binary_ufuncs,
    vec_associative_reductions,
)


class NotFullyDefinedBlockVectorError(Exception):
    pass


def assert_block_structure(vec):
    if vec.has_none:
        msg = 'Operation not allowed with None blocks.'
        raise NotFullyDefinedBlockVectorError(msg)


class BlockVector(BaseBlockVector, np.ndarray):
    """
    Structured vector interface. This interface can be used to
    perform operations on vectors composed by vectors. For example,

    >>> import numpy as np
    >>> from pyomo.contrib.pynumero.sparse import BlockVector
    >>> bv = BlockVector(3)
    >>> v0 = np.ones(3)
    >>> v1 = v0*2
    >>> v2 = np.random.normal(size=4)
    >>> bv.set_block(0, v0)
    >>> bv.set_block(1, v1)
    >>> bv.set_block(2, v2)
    >>> bv2 = BlockVector(2)
    >>> bv2.set_block(0, v0)
    >>> bv2.set_block(1, bv)

    Attributes
    ----------
    _nblocks: int
        number of blocks
    _brow_lengths: numpy.ndarray
        1D-Array of size nblocks that specifies the length of each entry
        in the block vector
    _undefined_brows: set
        A set of block indices for which the blocks are still None (i.e., the dimensions
        have not yet ben set). Operations with BlockVectors require all entries to be
        different than None.

    Parameters
    ----------
    nblocks: int
        The number of blocks in the BlockVector

    """

    def __new__(cls, nblocks):
        blocks = [None for i in range(nblocks)]
        arr = np.asarray(blocks, dtype='object')
        obj = arr.view(cls)
        obj._nblocks = nblocks

        obj._brow_lengths = np.empty(nblocks, dtype=np.float64)
        obj._brow_lengths.fill(np.nan)
        obj._undefined_brows = set(range(nblocks))

        return obj

    def __init__(self, nblocks):
        pass

    def __array_finalize__(self, obj):
        """This method is required to subclass from numpy array"""
        if obj is None:
            return
        self._brow_lengths = getattr(obj, '_brow_lengths', None)
        self._nblocks = getattr(obj, '_nblocks', 0)
        self._undefined_brows = getattr(obj, '_undefined_brows', None)

    def __array_prepare__(self, out_arr, context=None):
        """This method is required to subclass from numpy array"""
        return super(BlockVector, self).__array_prepare__(self, out_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        """This method is required to subclass from numpy array"""
        return super(BlockVector, self).__array_wrap__(self, out_arr, context)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Runs ufuncs specializations to BlockVector"""
        if kwargs.get('out', None) is not None:
            return NotImplemented
        if method == 'reduce' and ufunc in vec_associative_reductions:
            (arg,) = inputs
            return self._reduction_operation(ufunc, method, arg, kwargs)
        if method == '__call__':
            if ufunc in vec_unary_ufuncs:
                (arg,) = inputs
                return self._unary_operation(ufunc, method, arg, kwargs)
            if ufunc in vec_binary_ufuncs:
                return self._binary_operation(ufunc, method, inputs, kwargs)
        return NotImplemented

    def _reduction_operation(self, ufunc, method, x, kwargs):
        results = [
            self._unary_operation(ufunc, method, x.get_block(i), kwargs)
            for i in range(x.nblocks)
        ]
        if len(results) == 1:
            return results[0]
        else:
            return super().__array_ufunc__(ufunc, method, np.array(results), **kwargs)

    def _unary_operation(self, ufunc, method, x, kwargs):
        """Run recursion to perform unary_funcs on BlockVector"""
        # ToDo: deal with out
        if isinstance(x, BlockVector):
            v = BlockVector(x.nblocks)
            for i in range(x.nblocks):
                v.set_block(
                    i, self._unary_operation(ufunc, method, x.get_block(i), kwargs)
                )
            return v
        elif type(x) == np.ndarray:
            return super().__array_ufunc__(ufunc, method, x, **kwargs)
        else:
            return NotImplemented

    def _binary_operation(self, ufunc, method, args, kwargs):
        """Run recursion to perform binary_funcs on BlockVector"""
        # ToDo: deal with out
        x1, x2 = args
        if isinstance(x1, BlockVector) and isinstance(x2, BlockVector):
            assert_block_structure(x1)
            assert_block_structure(x2)
            assert (
                x1.nblocks == x2.nblocks
            ), 'Operation on BlockVectors need the same number of blocks on each operand'
            assert x1.size == x2.size, 'Dimension mismatch {}!={}'.format(
                x1.size, x2.size
            )
            res = BlockVector(x1.nblocks)

            for i in range(x1.nblocks):
                _args = (x1.get_block(i), x2.get_block(i))
                res.set_block(i, self._binary_operation(ufunc, method, _args, kwargs))
            return res
        elif type(x1) == np.ndarray and isinstance(x2, BlockVector):
            assert_block_structure(x2)
            assert x1.size == x2.size, 'Dimension mismatch {}!={}'.format(
                x1.size, x2.size
            )
            res = BlockVector(x2.nblocks)
            accum = 0
            for i in range(x2.nblocks):
                nelements = x2._brow_lengths[i]
                _args = (x1[accum : accum + nelements], x2.get_block(i))
                res.set_block(i, self._binary_operation(ufunc, method, _args, kwargs))
                accum += nelements
            return res
        elif type(x2) == np.ndarray and isinstance(x1, BlockVector):
            assert_block_structure(x1)
            assert x1.size == x2.size, 'Dimension mismatch {}!={}'.format(
                x1.size, x2.size
            )
            res = BlockVector(x1.nblocks)
            accum = 0
            for i in range(x1.nblocks):
                nelements = x1._brow_lengths[i]
                _args = (x1.get_block(i), x2[accum : accum + nelements])
                res.set_block(i, self._binary_operation(ufunc, method, _args, kwargs))
                accum += nelements
            return res
        elif np.isscalar(x1) and isinstance(x2, BlockVector):
            assert_block_structure(x2)
            res = BlockVector(x2.nblocks)
            for i in range(x2.nblocks):
                _args = (x1, x2.get_block(i))
                res.set_block(i, self._binary_operation(ufunc, method, _args, kwargs))
            return res
        elif np.isscalar(x2) and isinstance(x1, BlockVector):
            assert_block_structure(x1)
            res = BlockVector(x1.nblocks)
            for i in range(x1.nblocks):
                _args = (x1.get_block(i), x2)
                res.set_block(i, self._binary_operation(ufunc, method, _args, kwargs))
            return res
        elif (type(x1) == np.ndarray or np.isscalar(x1)) and (
            type(x2) == np.ndarray or np.isscalar(x2)
        ):
            return super(BlockVector, self).__array_ufunc__(
                ufunc, method, *args, **kwargs
            )
        else:
            if x1.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            if x2.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            return NotImplemented

    @property
    def nblocks(self):
        """
        Returns the number of blocks.
        """
        return self._nblocks

    @property
    def bshape(self):
        """
        Returns the number of blocks in this BlockVector in a tuple.
        """
        return (self.nblocks,)

    @property
    def shape(self):
        """
        Returns total number of elements in this BlockVector
        """
        assert_block_structure(self)
        return (np.sum(self._brow_lengths),)

    @property
    def size(self):
        """
        Returns total number of elements in this BlockVector
        """
        assert_block_structure(self)
        return np.sum(self._brow_lengths)

    @property
    def ndim(self):
        """
        Returns dimension of this BlockVector
        """
        return 1

    @property
    def has_none(self):
        """
        Indicate if this BlockVector has any none entries.
        """
        # this flag is updated in __setattr__
        return len(self._undefined_brows) != 0

    def block_sizes(self, copy=True):
        """
        Returns 1D-Array with sizes of individual blocks in this BlockVector
        """
        assert_block_structure(self)
        if copy:
            return self._brow_lengths.copy()
        return self._brow_lengths

    def get_block_size(self, ndx):
        if ndx in self._undefined_brows:
            raise NotFullyDefinedBlockVectorError(
                'The dimensions of the requested block are not defined.'
            )
        return int(self._brow_lengths[ndx])

    def _set_block_size(self, ndx, size):
        if ndx in self._undefined_brows:
            self._undefined_brows.remove(ndx)
            self._brow_lengths[ndx] = size
            if len(self._undefined_brows) == 0:
                self._brow_lengths = np.asarray(self._brow_lengths, dtype=np.int64)
        else:
            if self._brow_lengths[ndx] != size:
                raise ValueError(
                    'Incompatible dimensions for '
                    'block {ndx}; got {got}; '
                    'expected {exp}'.format(
                        ndx=ndx, got=size, exp=self._brow_lengths[ndx]
                    )
                )

    def is_block_defined(self, ndx):
        return ndx not in self._undefined_brows

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
        assert out is None, 'Operation not supported with out keyword'
        assert_block_structure(self)
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            return sum(
                self.get_block(i).dot(other.get_block(i)) for i in range(self.nblocks)
            )
        elif type(other) == np.ndarray:
            bv = self.flatten()
            return bv.dot(other)
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the sum of all entries in this BlockVector
        """
        assert_block_structure(self)
        results = np.array([self.get_block(i).sum() for i in range(self.nblocks)])
        return results.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def all(self, axis=None, out=None, keepdims=False):
        """
        Returns True if all elements evaluate to True.
        """
        assert_block_structure(self)
        results = np.array(
            [self.get_block(i).all() for i in range(self.nblocks)], dtype=bool
        )
        return results.all(axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        """
        Returns True if any element evaluate to True.
        """
        assert_block_structure(self)
        results = np.array(
            [self.get_block(i).any() for i in range(self.nblocks)], dtype=bool
        )
        return results.any(axis=axis, out=out, keepdims=keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        """
        Returns the largest value stored in this BlockVector
        """
        assert_block_structure(self)
        results = list()
        for block in self:
            if block.size > 0:
                results.append(block.max())
        return max(results)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        """Copy of the array, cast to a specified type"""
        if copy:
            bv = BlockVector(self.nblocks)
            for bid, vv in enumerate(self):
                if bid not in self._undefined_brows:
                    bv.set_block(
                        bid,
                        vv.astype(
                            dtype, order=order, casting=casting, subok=subok, copy=copy
                        ),
                    )
            return bv
        raise NotImplementedError("astype not implemented for copy=False")

    def clip(self, min=None, max=None, out=None):
        """
        Return BlockVector whose values are limited to [min, max].
        One of max or min must be given.

        Parameters
        ----------
        min: scalar_like, optional
            Minimum value. If None, clipping is not performed on lower interval edge.
        max: scalar_like, optional
            Maximum value. If None, clipping is not performed on upper interval edge.

        Returns
        -------
        BlockVector

        """
        assert_block_structure(self)
        assert out is None, 'Out keyword not supported'

        bv = BlockVector(self.nblocks)
        for bid in range(self.nblocks):
            bv.set_block(bid, self.get_block(bid).clip(min=min, max=max, out=None))
        return bv

    def compress(self, condition, axis=None, out=None):
        """
        Return selected slices of each subblock.

        Parameters
        ----------
        condition: Array or BlockVector that selects which entries to return.
            Determines to select (evaluate True in condition)

        Returns
        -------
        BlockVector

        """
        assert_block_structure(self)
        assert out is None, 'Out keyword not supported'
        result = BlockVector(self.nblocks)

        if isinstance(condition, BlockVector):
            assert_block_structure(condition)
            assert self.shape == condition.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, condition.shape
            )
            assert (
                self.nblocks == condition.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(
                self.nblocks, condition.nblocks
            )
            for idx in range(self.nblocks):
                result.set_block(
                    idx, self.get_block(idx).compress(condition.get_block(idx))
                )
            return result
        elif type(condition) == np.ndarray:
            assert self.shape == condition.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, condition.shape
            )
            accum = 0
            for idx in range(self.nblocks):
                nelements = self._brow_lengths[idx]
                result.set_block(
                    idx,
                    self.get_block(idx).compress(condition[accum : accum + nelements]),
                )
                accum += nelements
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def conj(self):
        """
        Complex-conjugate all elements.
        """
        assert_block_structure(self)
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            result.set_block(idx, self.get_block(idx).conj())
        return result

    def conjugate(self):
        """
        Complex-conjugate all elements.
        """
        assert_block_structure(self)
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            result.set_block(idx, self.get_block(idx).conjugate())
        return result

    def nonzero(self):
        """
        Return the indices of the elements that are non-zero.
        """
        assert_block_structure(self)
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            result.set_block(idx, self.get_block(idx).nonzero()[0])
        return (result,)

    if np.__version__[0] < '2':

        def ptp(self, axis=None, out=None, keepdims=False):
            """
            Peak to peak (maximum - minimum) value along a given axis.
            """
            assert_block_structure(self)
            assert out is None, 'Out keyword not supported'
            return self.max() - self.min()

    def round(self, decimals=0, out=None):
        """
        Return BlockVector with each element rounded to the given number of decimals
        """
        assert_block_structure(self)
        assert out is None, 'Out keyword not supported'
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            result.set_block(idx, self.get_block(idx).round(decimals=decimals))
        return result

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """
        Returns the standard deviation of the BlockVector elements.
        """
        return self.flatten().std(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims
        )

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
        """
        Returns the variance of the BlockVector elements.
        """
        return self.flatten().var(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims
        )

    def tofile(self, fid, sep="", format="%s"):
        """
        Writes flat version of BlockVector to a file as text or binary (default).
        """
        self.flatten().tofile(fid, sep=sep, format=format)

    def min(self, axis=None, out=None, keepdims=False):
        """
        Returns the smallest value stored in the vector
        """
        assert_block_structure(self)
        results = list()
        for block in self:
            if block.size > 0:
                results.append(block.min())
        return min(results)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the average of all entries in this BlockVector
        """
        n = self.size
        if n == 0:
            return 0.0
        return self.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims) / n

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        """
        Returns the product of all entries in this BlockVector
        """
        assert_block_structure(self)
        results = np.array([self.get_block(i).prod() for i in range(self.nblocks)])
        return results.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def fill(self, value):
        """
        Fills the BlockVector with a scalar value.

        Parameters
        ----------
        value : scalar
            All elements in the vector will be assigned this value

        Returns
        -------
        None

        """
        assert_block_structure(self)
        for i in range(self.nblocks):
            self.get_block(i).fill(value)

    def tolist(self):
        """
        Return the BlockVector flattened as a list.

        Returns
        -------
        list

        """
        return self.flatten().tolist()

    def flatten(self, order='C'):
        """
        Converts the BlockVector to a NumPy array. This will also call flatten on the underlying NumPy arrays in
        the BlockVector.

        Parameters
        ----------
        order: str: {C, F, A, K}, optional
            See NumPy array documentation.

        Returns
        -------
        flat_array: numpy.ndarray
            The NumPy array resulting from concatenating all of the blocks
        """
        assert_block_structure(self)
        all_blocks = tuple(
            self.get_block(i).flatten(order=order) for i in range(self.nblocks)
        )
        return np.concatenate(all_blocks)

    def ravel(self, order='C'):
        """
        Converts the BlockVector into a NumPy array. Note that ravel is also called
        on all of the NumPy arrays in the BlockVector before concatenating them.

        Parameters
        ----------
        order: str
            See NumPy documentation.

        Returns
        -------
        res: numpy.ndarray
        """
        assert_block_structure(self)
        all_blocks = tuple(
            self.get_block(i).ravel(order=order) for i in range(self.nblocks)
        )
        return np.concatenate(all_blocks)

    def argmax(self, axis=None, out=None):
        """
        Returns the index of the largest element.
        """
        assert_block_structure(self)
        return self.flatten().argmax(axis=axis, out=out)

    def argmin(self, axis=None, out=None):
        """
        Returns the index of the smallest element.
        """
        assert_block_structure(self)
        return self.flatten().argmin(axis=axis, out=out)

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
        Returns a copy of this BlockVector

        Parameters
        ----------
        value: scalar (optional)
            all entries of the cloned vector are set to this value
        copy: bool (optional)
            if True makes a deepcopy of each block in this vector. default True

        Returns
        -------
        BlockVector

        """
        result = BlockVector(self.nblocks)
        for idx in range(self.nblocks):
            if idx not in self._undefined_brows:
                if copy:
                    result.set_block(idx, self.get_block(idx).copy())
                else:
                    result.set_block(idx, self.get_block(idx))
        if value is not None:
            result.fill(value)
        return result

    def copyfrom(self, other):
        """
        Copy entries of other vector into this vector

        Parameters
        ----------
        other: BlockVector or numpy.ndarray
            vector to be copied to this BlockVector

        Returns
        -------
        None

        """
        assert_block_structure(self)

        if isinstance(other, BlockVector):
            assert_block_structure(other)

            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx in range(other.nblocks):
                if isinstance(self.get_block(idx), BlockVector):
                    self.get_block(idx).copyfrom(other.get_block(idx))
                elif isinstance(self.get_block(idx), np.ndarray):
                    if isinstance(other.get_block(idx), BlockVector):
                        self.set_block(idx, other.get_block(idx).copy())
                    elif isinstance(other.get_block(idx), np.ndarray):
                        np.copyto(self.get_block(idx), other.get_block(idx))
                    else:
                        raise RuntimeError('Input not recognized')
                elif self.get_block(idx) is None:
                    if isinstance(other.get_block(idx), np.ndarray):
                        # this include block vectors too
                        self.set_block(idx, other.get_block(idx).copy())
                    else:
                        raise RuntimeError('Input not recognized')
                else:
                    raise RuntimeError('Input not recognized')
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )

            offset = 0
            for idx in range(self.nblocks):
                subarray = other[offset : offset + self.get_block(idx).size]
                if isinstance(self.get_block(idx), BlockVector):
                    self.get_block(idx).copyfrom(subarray)
                else:
                    np.copyto(self.get_block(idx), subarray)
                offset += self.get_block(idx).size
        else:
            raise NotImplementedError('Operation not supported by BlockVector')

    def copyto(self, other):
        """
        Copy entries of this BlockVector into other

        Parameters
        ----------
        other: BlockVector or numpy.ndarray

        Returns
        -------
        None

        """

        if isinstance(other, BlockVector):
            msgj = 'Number of blocks mismatch {} != {}'.format(
                self.nblocks, other.nblocks
            )
            assert self.nblocks == other.nblocks, msgj
            for idx in range(self.nblocks):
                if isinstance(other.get_block(idx), BlockVector):
                    other.get_block(idx).copyfrom(self.get_block(idx))
                elif isinstance(other.get_block(idx), np.ndarray):
                    if self.get_block(idx) is not None:
                        np.copyto(other.get_block(idx), self.get_block(idx).flatten())
                    else:
                        other.set_block(idx, None)
                elif other.get_block(idx) is None:
                    if self.get_block(idx) is not None:
                        other.set_block(idx, self.get_block(idx).copy())
                    else:
                        other.set_block(idx, None)
                else:
                    raise RuntimeError('Should never get here')

        elif isinstance(other, np.ndarray):
            np.copyto(other, self.flatten())
        else:
            raise NotImplementedError()

    def copy(self, order='C'):
        """
        Returns a copy of the BlockVector
        """
        bv = BlockVector(self.nblocks)
        for bid in range(self.nblocks):
            if bid not in self._undefined_brows:
                bv.set_block(bid, self.get_block(bid).copy(order=order))
        return bv

    def copy_structure(self):
        """
        Returns a copy of the BlockVector structure filled with zeros
        """
        bv = BlockVector(self.nblocks)
        for bid in range(self.nblocks):
            if self.get_block(bid) is not None:
                if isinstance(self.get_block(bid), BlockVector):
                    bv.set_block(bid, self.get_block(bid).copy_structure())
                elif type(self.get_block(bid)) == np.ndarray:
                    bv.set_block(
                        bid,
                        np.zeros(
                            self.get_block(bid).size, dtype=self.get_block(bid).dtype
                        ),
                    )
                else:
                    raise NotImplementedError('Should never get here')
        return bv

    def set_blocks(self, blocks):
        """
        Assigns vectors in blocks

        Parameters
        ----------
        blocks: list
            list of numpy.ndarrays and/or BlockVectors

        Returns
        -------
        None

        """
        assert isinstance(blocks, list), 'blocks should be passed in ordered list'
        assert (
            len(blocks) == self.nblocks
        ), 'More blocks passed than allocated {} != {}'.format(
            len(blocks), self.nblocks
        )
        for idx, blk in enumerate(blocks):
            self.set_block(idx, blk)

    def __iter__(self):
        for ndx in range(self._nblocks):
            yield self.get_block(ndx)

    def __add__(self, other):
        # add this BlockVector with other vector
        # supports addition with scalar, numpy.ndarray and BlockVectors
        # returns BlockVector
        result = BlockVector(self.nblocks)
        assert_block_structure(self)
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                result.set_block(idx, blk + other.get_block(idx))
            return result
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result.set_block(idx, blk + other[accum : accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result.set_block(idx, blk + other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):
        # subtract this BlockVector with other vector
        # supports subtraction with scalar, numpy.ndarray and BlockVectors
        # returns BlockVector
        result = BlockVector(self.nblocks)
        assert_block_structure(self)
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                result.set_block(idx, blk - other.get_block(idx))
            return result
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result.set_block(idx, blk - other[accum : accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result.set_block(idx, blk - other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def __rsub__(self, other):  # other - self
        result = BlockVector(self.nblocks)
        assert_block_structure(self)
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                result.set_block(idx, other.get_block(idx) - blk)
            return result

        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result.set_block(idx, other[accum : accum + nelements] - blk)
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result.set_block(idx, other - blk)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def __mul__(self, other):
        # elementwise multiply this BlockVector with other vector
        # supports multiplication with scalar, numpy.ndarray and BlockVectors
        # returns BlockVector
        assert_block_structure(self)
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                result.set_block(idx, blk.__mul__(other.get_block(idx)))
            return result
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result.set_block(idx, blk.__mul__(other[accum : accum + nelements]))
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result.set_block(idx, blk.__mul__(other))
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def __rmul__(self, other):  # other + self
        return self.__mul__(other)

    def __truediv__(self, other):
        # elementwise divide this BlockVector with other vector
        # supports division with scalar, numpy.ndarray and BlockVectors
        # returns BlockVector
        assert_block_structure(self)
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                result.set_block(idx, blk / other.get_block(idx))
            return result
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result.set_block(idx, blk / other[accum : accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result.set_block(idx, blk / other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def __rtruediv__(self, other):
        assert_block_structure(self)
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                result.set_block(idx, other.get_block(idx) / blk)
            return result
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result.set_block(idx, other[accum : accum + nelements] / blk)
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result.set_block(idx, other / blk)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def __floordiv__(self, other):
        assert_block_structure(self)
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                result.set_block(idx, blk // other.get_block(idx))
            return result
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result.set_block(idx, blk // other[accum : accum + nelements])
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result.set_block(idx, blk // other)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def __rfloordiv__(self, other):
        assert_block_structure(self)
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                result.set_block(idx, other.get_block(idx) // blk)
            return result
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                result.set_block(idx, other[accum : accum + nelements] // blk)
                accum += nelements
            return result
        elif np.isscalar(other):
            for idx, blk in enumerate(self):
                result.set_block(idx, other // blk)
            return result
        else:
            if other.__class__.__name__ == 'MPIBlockVector':
                raise RuntimeError('Operation not supported by BlockVector')
            raise NotImplementedError()

    def __iadd__(self, other):
        # elementwise inplace addition to this BlockVector with other vector
        # supports addition with scalar, numpy.ndarray and BlockVectors
        assert_block_structure(self)
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                blk += other
            return self
        elif isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                blk += other.get_block(idx)
            return self
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                blk += other[accum : accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __isub__(self, other):
        # elementwise inplace subtraction to this BlockVector with other vector
        # supports subtraction with scalar, numpy.ndarray and BlockVectors
        assert_block_structure(self)
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                blk -= other
            return self
        elif isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                blk -= other.get_block(idx)
            return self
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                blk -= other[accum : accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __imul__(self, other):
        # elementwise inplace multiplication to this BlockVector with other vector
        # supports multiplication with scalar, numpy.ndarray and BlockVectors
        assert_block_structure(self)
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                blk *= other
            return self
        elif isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                blk *= other.get_block(idx)
            return self
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                blk *= other[accum : accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __itruediv__(self, other):
        # elementwise inplace division to this BlockVector with other vector
        # supports division with scalar, numpy.ndarray and BlockVectors
        assert_block_structure(self)
        if np.isscalar(other):
            for idx, blk in enumerate(self):
                blk /= other
            return self
        elif isinstance(other, BlockVector):
            assert_block_structure(other)
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            assert (
                self.nblocks == other.nblocks
            ), 'Number of blocks mismatch {} != {}'.format(self.nblocks, other.nblocks)
            for idx, blk in enumerate(self):
                blk /= other.get_block(idx)
            return self
        elif type(other) == np.ndarray:
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for idx, blk in enumerate(self):
                nelements = self._brow_lengths[idx]
                blk /= other[accum : accum + nelements]
                accum += nelements
            return self
        else:
            raise NotImplementedError()

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def _print(self, indent):
        msg = ''
        for ndx, block in enumerate(self):
            if isinstance(block, BlockVector):
                msg += (
                    indent
                    + str(ndx)
                    + ': '
                    + block.__class__.__name__
                    + str(block.bshape)
                    + '\n'
                )
                msg += block._print(indent=indent + '   ')
            else:
                msg += (
                    indent
                    + str(ndx)
                    + ': '
                    + block.__class__.__name__
                    + str(block.shape)
                    + '\n'
                )
        return msg

    def __str__(self):
        return self._print(indent='')

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.bshape)

    def get_block(self, key):
        """
        Access a block.

        Parameters
        ----------
        key: int
            This is the block index

        Returns
        -------
        block: np.ndarray or BlockVector
            The block corresponding to the index key.
        """
        return super(BlockVector, self).__getitem__(key)

    def set_block(self, key, value):
        """
        Set a block. The value can be a NumPy array or another BlockVector.

        Parameters
        ----------
        key: int
            This is the block index
        value:
            This is the block. It can be a NumPy array or another BlockVector.
        """
        assert -self.nblocks < key < self.nblocks, 'out of range'
        assert isinstance(value, np.ndarray) or isinstance(
            value, BaseBlockVector
        ), 'Blocks need to be numpy arrays or BlockVectors'
        assert value.ndim == 1, 'Blocks need to be 1D'

        if isinstance(value, BaseBlockVector):
            assert_block_structure(value)
        self._set_block_size(key, value.size)
        super(BlockVector, self).__setitem__(key, value)

    def _has_equal_structure(self, other):
        """
        Parameters
        ----------
        other: BlockVector

        Returns
        -------
        equal_structure: bool
            True if self and other have the same block structure (recursive). False otherwise.
        """
        if not isinstance(other, BlockVector):
            return False
        if self.nblocks != other.nblocks:
            return False
        for ndx, block1 in enumerate(self):
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
        # numpy: __getitem__[()] is identity
        if item.__class__ is tuple and not item:
            return self
        if not self._has_equal_structure(item):
            raise ValueError(
                'BlockVector.__getitem__ only accepts slices in the form of BlockVectors of the same structure'
            )
        res = BlockVector(self.nblocks)
        for ndx, block in self:
            res.set_block(ndx, block[item.get_block(ndx)])

    def __setitem__(self, key, value):
        if not (
            self._has_equal_structure(key)
            and (self._has_equal_structure(value) or np.isscalar(value))
        ):
            raise ValueError(
                'BlockVector.__setitem__ only accepts slices in the form of BlockVectors of the same structure'
            )
        if np.isscalar(value):
            for ndx, block in enumerate(self):
                block[key.get_block(ndx)] = value
        else:
            for ndx, block in enumerate(self):
                block[key.get_block(ndx)] = value.get_block(ndx)

    def _comparison_helper(self, other, operation):
        assert_block_structure(self)
        result = self.copy_structure()
        if isinstance(other, BlockVector):
            assert_block_structure(other)
            for ndx in range(self.nblocks):
                result.set_block(
                    ndx, operation(self.get_block(ndx), other.get_block(ndx))
                )
            return result
        elif isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(
                self.shape, other.shape
            )
            accum = 0
            for ndx in range(self.nblocks):
                result.set_block(
                    ndx,
                    operation(
                        self.get_block(ndx),
                        other[accum : accum + self.get_block_size(ndx)],
                    ),
                )
                accum += self.get_block_size(ndx)
            return result
        elif np.isscalar(other):
            for ndx in range(self.nblocks):
                result.set_block(ndx, operation(self.get_block(ndx), other))
            return result
        else:
            raise NotImplementedError('Operation not supported by BlockVector')

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

    def __neg__(self):
        # elementwise negate this BlockVector
        assert_block_structure(self)
        bv = BlockVector(self.nblocks)
        for bid in range(self.nblocks):
            bv.set_block(bid, self.get_block(bid).__neg__())
        return bv

    def __contains__(self, item):
        other = item
        assert_block_structure(self)
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

    def pprint(self):
        """Prints BlockVector in pretty format"""
        msg = self.__repr__()
        msg += '\n'
        msg += self.__str__()
        print(msg)

    def toMPIBlockVector(self, rank_ownership, mpi_comm, assert_correct_owners=False):
        """
        Creates a parallel MPIBlockVector from this BlockVector

        Parameters
        ----------
        rank_ownership: array_like
            Array_like of size nblocks. Each entry defines ownership of each block.
            There are two types of ownership. Block that are owned by all processor,
            and blocks owned by a single processor. If a block is owned by all
            processors then its ownership is -1. Otherwise, if a block is owned by
            a single processor, then its ownership is equal to the rank of the
            processor.
        mpi_comm: MPI communicator
            An MPI communicator. Tyically MPI.COMM_WORLD

        """
        from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector

        assert_block_structure(self)
        assert (
            len(rank_ownership) == self.nblocks
        ), 'rank_ownership must be of size {}'.format(self.nblocks)

        mpi_bv = MPIBlockVector(
            self.nblocks,
            rank_ownership,
            mpi_comm,
            assert_correct_owners=assert_correct_owners,
        )

        # populate blocks in the right spaces
        for bid in mpi_bv.owned_blocks:
            mpi_bv.set_block(bid, self.get_block(bid))

        return mpi_bv

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


class BlockVector(object):
    """
    Structured Vector interface

    Parameters
    -------------------
    vectors: int or list of 1d-arrays
    number of blocks contained in the block vector
    if a list is passed the block vector is initialized from
    the list of 1d-arrays

    """

    def __init__(self, vectors):

        if isinstance(vectors, int):

            blocks = [None for i in range(vectors)]

            self._blocks = np.asarray(blocks, dtype='object')
            self._block_mask = np.zeros(vectors, dtype=bool)
            self._brow_lengths = np.zeros(vectors, dtype=np.int64)
            self._nblocks = vectors

        elif isinstance(vectors, list):

            nblocks = len(vectors)
            blocks = [None for i in range(nblocks)]
            self._blocks = np.asarray(blocks, dtype='object')
            self._block_mask = np.zeros(nblocks, dtype=bool)
            self._brow_lengths = np.zeros(nblocks, dtype=np.int64)
            self._nblocks = nblocks
            self.set_blocks(vectors)
        else:
            raise RuntimeError("BlockVector constructor takes an integer or a list of np.ndarrays")

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

    def block_sizes(self):
        """
        Returns array with sizes of individual blocks
        """
        return np.copy(self._brow_lengths)

    def dot(self, other):
        """
        Returns dot product

        Parameters
        ----------
        other : ndarray or BlockVector

        Returns
        -------
        float

        """
        self._check_mask()
        if isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            return sum(self[i].dot(other[i]) for i in range(self.nblocks))
        else:
            return NotImplemented

    def sum(self):
        """
        Returns the sum of all entries in the block vector
        """
        self._check_mask()
        return sum(self[i].sum() for i in range(self.nblocks))

    def max(self):
        """
        Returns the largest value stored in the vector
        """
        self._check_mask()
        return np.array([self[i].max() for i in range(self.nblocks)]).max()

    def min(self):
        """
        Returns the smallest value stored in the vector
        """
        self._check_mask()
        return np.array([self[i].min() for i in range(self.nblocks)]).min()

    def mean(self):
        """
        Returns the average of all entries in the vector
        """
        self._check_mask()
        return sum(self[i].mean() for i in range(self.nblocks))/self.nblocks

    def prod(self):
        """
        Returns the product of all entries in the vector
        """
        self._check_mask()
        return np.array([self[i].prod() for i in range(self.nblocks)]).prod()

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

    # TODO: this functions need to be implemented more efficiently
    ##############################################################
    def norm(self, ord=None, axis=None, keepdims=False):
        """
        Returns norm of block vector
        """
        return np.linalg.norm(self.flatten(), ord=ord, axis=axis, keepdims=keepdims)

    def argmax(self, axis=None, out=None):
        """
        Returns the index of the largest element.
        """
        msg = 'Operation not allowed with None blocks. Specify all blocks'
        msg += '\n{}'.format(self.__str__())
        assert np.all(self._block_mask), msg
        return self.flatten().argmax(axis=axis, out=out)

    def argmin(self, axis=None, out=None):
        """
        Returns the index of the smallest element.
        """
        msg = 'Operation not allowed with None blocks. Specify all blocks'
        msg += '\n{}'.format(self.__str__())
        assert np.all(self._block_mask), msg
        return self.flatten().argmin(axis=axis, out=out)

    def cumprod(self, axis=None, dtype=None, out=None):
        """
        Returns the cumulative product of the elements along the given axis.
        """
        msg = 'Operation not allowed with None blocks. Specify all blocks'
        msg += '\n{}'.format(self.__str__())
        assert np.all(self._block_mask), msg
        return self.flatten().cumprod(axis=axis, dtype=dtype, out=out)

    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Returns the cumulative sum of the elements along the given axis.
        """
        msg = 'Operation not allowed with None blocks. Specify all blocks'
        msg += '\n{}'.format(self.__str__())
        assert np.all(self._block_mask), msg
        return self.flatten().cumsum(axis=axis, dtype=dtype, out=out)

    ##############################################################

    def scale(self, value):
        """
        Scales all entries in the vector

        Parameters
        ----------
        value: scalar
            all elements in vector are multiply by this value

        Returns
        -------
        None
        """
        self._check_mask()
        for blk in self._blocks:
            blk *= value

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
        for idx, blk in enumerate(self._blocks):
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
        self._check_mask()
        if isinstance(other, BlockVector):
            other._check_mask()
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
            raise NotImplementedError

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
        self._check_mask()
        if isinstance(other, np.ndarray):
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            np.copyto(other, self.flatten())
        elif isinstance(other, BlockVector):
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                if other[idx] is not None:
                    msgi = 'Dimension mismatch in subblock {} {} != {}'
                    assert other[idx].shape == blk.shape, msgi.format(idx,
                                                                      blk.shape,
                                                                      other[idx].shape)
                if isinstance(blk, BlockVector):
                    other[idx] = blk.clone(copy=True)
                else:
                    other[idx] = cp.deepcopy(blk)

    def copy(self):
        bv = BlockVector(self.nblocks)
        for bid, vv in enumerate(self):
            bv[bid] = vv.copy()
        return bv

    def _check_mask(self):
        msg = 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        msg += '\n{}'.format(self.__str__())
        if not np.all(self._block_mask):
            print(self)
            raise RuntimeError(msg)
        for idx, blk in enumerate(self._blocks):
            if isinstance(blk, BlockVector):
                blk._check_mask()

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

    def __iter__(self):
        return (blk for blk in self._blocks)

    def __add__(self, other):
        result = BlockVector(self.nblocks)
        self._check_mask()
        if isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                result[idx] = blk + other[idx]
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported addition of block vector with numpy array. Use non-member function')
        elif np.isscalar(other):
            msg = 'Not supported addition of block vector with scalar. Use non-member function'
            msg += ' or clone the block vector, fill it and add it'
            raise RuntimeError(msg)
        else:
            return NotImplemented

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __sub__(self, other):
        result = BlockVector(self.nblocks)
        self._check_mask()
        if isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                result[idx] = blk - other[idx]
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported substraction of block vector with numpy array. Use non-member function')
        elif np.isscalar(other):
            msg = 'Not supported substraction of block vector with scalar. Use non-member function'
            msg += ' or clone the block vector, fill it and add it'
            raise RuntimeError(msg)
        else:
            raise NotImplemented

    def __rsub__(self, other):  # other - self
        result = BlockVector(self.nblocks)
        self._check_mask()
        if isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                result[idx] = other[idx] - blk
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported substraction of block vector with numpy array. Use non-member function')
        elif np.isscalar(other):
            msg = 'Not supported substraction of block vector with scalar. Use non-member function'
            msg += ' or clone the block vector, fill it and substract it'
            raise RuntimeError(msg)
        else:
            return NotImplemented

    def __mul__(self, other):
        self._check_mask()
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                result[idx] = blk * other[idx]
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported multiplication of block vector with numpy array. Use non-member function')
        elif np.isscalar(other):
            msg = 'Not supported multiplication of block vector with scalar. Use non-member function'
            msg += ' or clone the block vector, fill it and multiply it'
            raise RuntimeError(msg)
        else:
            return NotImplemented

    def __rmul__(self, other):  # other + self
        return self.__mul__(other)

    def __truediv__(self, other):
        self._check_mask()
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                result[idx] = blk / other[idx]
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported division of block vector with numpy array. Use non-member function')
        elif np.isscalar(other):
            msg = 'Not supported division of block vector with scalar. Use non-member function'
            msg += ' or clone the block vector, fill it and divide it'
            raise RuntimeError(msg)
        else:
            raise NotImplemented

    def __rtruediv__(self, other):
        self._check_mask()
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                result[idx] = other[idx] / blk
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported division of block vector with numpy array. Use non-member function')
        elif np.isscalar(other):
            msg = 'Not supported division of block vector with scalar. Use non-member function'
            msg += ' or clone the block vector, fill it and divide it'
            raise RuntimeError(msg)
        else:
            raise NotImplemented

    def __floordiv__(self, other):
        self._check_mask()
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                result[idx] = blk // other[idx]
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported division of block vector with numpy array. Use non-member function')
        elif np.isscalar(other):
            msg = 'Not supported division of block vector with scalar. Use non-member function'
            msg += ' or clone the block vector, fill it and divide it'
            raise RuntimeError(msg)
        else:
            return NotImplemented

    def __rfloordiv__(self, other):
        self._check_mask()
        result = BlockVector(self.nblocks)
        if isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                result[idx] = other[idx] // blk
            return result
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported division of block vector with numpy array. Use non-member function')
        elif np.isscalar(other):
            msg = 'Not supported division of block vector with scalar. Use non-member function'
            msg += ' or clone the block vector, fill it and divide it'
            raise RuntimeError(msg)
        else:
            return NotImplemented

    def __iadd__(self, other):
        self._check_mask()
        if np.isscalar(other):
            for idx, blk in enumerate(self._blocks):
                self[idx] = self[idx] + other # maybe it suffice with doing self[idx] = self[idf] + other
            return self
        elif isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                self[idx] = self[idx] + other[idx]
            return self
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported addition of block vector with numpy array. Use non-member function')
        else:
            return NotImplemented

    def __isub__(self, other):
        self._check_mask()
        if np.isscalar(other):
            for idx, blk in enumerate(self._blocks):
                self[idx] = self[idx] - other
            return self
        elif isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                self[idx] = self[idx] - other[idx]
            return self
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported substraction of block vector with numpy array. Use non-member function')
        else:
            return NotImplemented

    def __imul__(self, other):
        self._check_mask()
        if np.isscalar(other):
            for idx, blk in enumerate(self._blocks):
                self[idx] = self[idx] * other
            return self
        elif isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                self[idx] = self[idx] * other[idx]
            return self
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported multiplication of block vector with numpy array. Use non-member function')
        else:
            return NotImplemented

    def __itruediv__(self, other):
        self._check_mask()
        if np.isscalar(other):
            for idx, blk in enumerate(self._blocks):
                self[idx] = self[idx] / other
            return self
        elif isinstance(other, BlockVector):
            other._check_mask()
            assert self.shape == other.shape, 'Dimension mismatch {} != {}'.format(self.shape, other.shape)
            assert self.nblocks == other.nblocks, 'Number of blocks mismatch {} != {}'.format(self.nblocks,
                                                                                              other.nblocks)
            for idx, blk in enumerate(self._blocks):
                self[idx] = self[idx] / other[idx]
            return self
        elif isinstance(other, np.ndarray):
            raise RuntimeError('Not supported division of block vector with numpy array. Use non-member function')
        else:
            return NotImplemented

    def __getitem__(self, item):
        if isinstance(item, slice):
            raise NotImplementedError
        return self._blocks[item]

    def __setitem__(self, key, value):

        if isinstance(key, slice):
            raise NotImplementedError

        assert -self.nblocks < key < self.nblocks, 'out of range'
        if value is None:
            self._blocks[key] = None
            self._block_mask[key] = False
            self._brow_lengths[key] = 0
        else:
            msg = 'Blocks need to be numpy arrays or BlockVectors'
            assert isinstance(value, np.ndarray) or isinstance(value, BlockVector), msg
            assert value.ndim == 1, 'Blocks need to be 1D'
            self._blocks[key] = value
            self._block_mask[key] = True
            self._brow_lengths[key] = value.size

    def __neg__(self):

        v = BlockVector(self.nblocks)
        for bid in range(self.nblocks):
            v[bid] = -self[bid]
        return v

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__,self.shape)

    def __str__(self):
        msg = ''
        for idx in range(self.bshape[0]):
            if isinstance(self._blocks[idx], BlockVector):
                repn = self._blocks[idx].__repr__()
            elif isinstance(self._blocks[idx], np.ndarray):
                repn = "array({})".format(self._blocks[idx].size)
            elif self[idx] is None:
                repn = None
            else:
                raise NotImplementedError("Should not ge here")
            msg += '{}: {}\n'.format(idx, repn)
        return msg

    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memodict={}):
        raise NotImplementedError()

if __name__ == "__main__":

    print('done')
    v = BlockVector(4)
    print(v)
    print(v.__repr__())
    for a in v:
        print(a)

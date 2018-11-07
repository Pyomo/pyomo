#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
import numpy as np

__all__ = ['allclose', 'concatenate', 'where', 'isin']


def allclose(x1, x2, rtol, atol):
    x1_flat = x1.flatten()
    x2_flat = x2.flatten()
    return np.allclose(x1_flat, x2_flat, rtol=rtol, atol=atol)


def concatenate(arrays):
    return np.concatenate([a.flatten() for a in arrays])


def where(*args):

    condition = args[0]

    if len(args) == 2:
        raise ValueError('either both or neither of x and y should be given')
    if len(args) > 3:
        raise TypeError('where() takes at most 3 arguments ({} given)'.format(len(args)))

    n_args = len(args)
    if isinstance(condition, BlockVector):
        if n_args == 1:
            assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            res = BlockVector(condition.nblocks)
            for i in range(condition.nblocks):
                _args = [condition[i]]
                res[i] = where(*_args)[0]
            return (res,)
        else:
            x = args[1]
            y = args[2]
            if isinstance(x, BlockVector) and isinstance(y, BlockVector):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == x.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.nblocks == y.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = BlockVector(condition.nblocks)
                for i in range(condition.nblocks):
                    _args = [condition[i], x[i], y[i]]
                    res[i] = where(*_args)
                return res
            elif isinstance(x, np.ndarray) and isinstance(y, BlockVector):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == y.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition[i], x[accum: accum + nelements], y[i]]
                    res[i] = where(*_args)
                    accum += nelements
                return res

            elif isinstance(x, BlockVector) and isinstance(y, np.ndarray):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == x.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition[i], x[i], y[accum: accum + nelements]]
                    res[i] = where(*_args)
                    accum += nelements
                return res

            elif np.isscalar(x) and isinstance(y, BlockVector):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == y.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert condition.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition[i], x, y[i]]
                    res[i] = where(*_args)
                    accum += nelements
                return res

            elif isinstance(x, BlockVector) and np.isscalar(y):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == x.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition[i], x[i], y]
                    res[i] = where(*_args)
                    accum += nelements
                return res

            elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition[i], x[accum: accum + nelements], y[accum: accum + nelements]]
                    res[i] = where(*_args)
                    accum += nelements
                return res

            elif isinstance(x, np.ndarray) and np.isscalar(y):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition[i], x[accum: accum + nelements], y]
                    res[i] = where(*_args)
                    accum += nelements
                return res

            elif np.isscalar(x) and isinstance(y, np.ndarray):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition[i], x, y[accum: accum + nelements]]
                    res[i] = where(*_args)
                    accum += nelements
                return res

            elif np.isscalar(x) and np.isscalar(y):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                res = BlockVector(condition.nblocks)
                for i in range(condition.nblocks):
                    _args = [condition[i], x, y]
                    res[i] = where(*_args)
                return res

            else:
                raise NotImplementedError()
    else:
        if n_args == 1:
            return np.where(*args)
        else:

            x = args[1]
            y = args[2]
            if isinstance(x, BlockVector):
                # ToDo: add logger to give warning here
                x = x.flatten()
            if isinstance(y, BlockVector):
                # ToDo: add logger to give warning here
                y = y.flatten()
            _args = [condition, x, y]
            return np.where(*_args)


def isin(element, test_elements, assume_unique=False, invert=False):

    if isinstance(element, BlockVector) and isinstance(test_elements, BlockVector):
        assert not element.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        assert not test_elements.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        assert element.nblocks == test_elements.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
        res = BlockVector(element.nblocks)
        for i in range(element.nblocks):
            res[i] = isin(element[i],
                          test_elements[i],
                          assume_unique=assume_unique,
                          invert=invert)
        return res

    elif isinstance(element, BlockVector) and isinstance(test_elements, np.ndarray):

        assert not element.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        res = BlockVector(element.nblocks)
        for i in range(element.nblocks):
            res[i] = isin(element[i],
                          test_elements,
                          assume_unique=assume_unique,
                          invert=invert)
        return res

    elif isinstance(element, np.ndarray) and isinstance(test_elements, np.ndarray):

        return np.isin(element,
                       test_elements,
                       assume_unique=assume_unique,
                       invert=invert)

    else:
        raise NotImplementedError()
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import numpy as np, attempt_import

block_vector = attempt_import('pyomo.contrib.pynumero.sparse.block_vector',
                              defer_check=True)[0]

def norm(x, ord=None):

    f = np.linalg.norm
    if isinstance(x, np.ndarray):
        return f(x, ord=ord)
    elif isinstance(x, block_vector.BlockVector):
        flat_x = x.flatten()
        return f(flat_x, ord=ord)
    else:
        raise NotImplementedError()

def allclose(x1, x2, rtol, atol):
    # this needs to be implemented for parallel
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
    if isinstance(condition, block_vector.BlockVector):
        if n_args == 1:
            assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
            res = block_vector.BlockVector(condition.nblocks)
            for i in range(condition.nblocks):
                _args = [condition.get_block(i)]
                res.set_block(i, where(*_args)[0])
            return (res,)
        else:
            x = args[1]
            y = args[2]
            if isinstance(x, block_vector.BlockVector) and isinstance(y, block_vector.BlockVector):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == x.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.nblocks == y.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = block_vector.BlockVector(condition.nblocks)
                for i in range(condition.nblocks):
                    _args = [condition.get_block(i), x.get_block(i), y.get_block(i)]
                    res.set_block(i, where(*_args))
                return res
            elif isinstance(x, np.ndarray) and isinstance(y, block_vector.BlockVector):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == y.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = block_vector.BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition.get_block(i), x[accum: accum + nelements], y.get_block(i)]
                    res.set_block(i, where(*_args))
                    accum += nelements
                return res

            elif isinstance(x, block_vector.BlockVector) and isinstance(y, np.ndarray):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == x.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = block_vector.BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition.get_block(i), x.get_block(i), y[accum: accum + nelements]]
                    res.set_block(i, where(*_args))
                    accum += nelements
                return res

            elif np.isscalar(x) and isinstance(y, block_vector.BlockVector):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == y.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert condition.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = block_vector.BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition.get_block(i), x, y.get_block(i)]
                    res.set_block(i, where(*_args))
                    accum += nelements
                return res

            elif isinstance(x, block_vector.BlockVector) and np.isscalar(y):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.nblocks == x.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = block_vector.BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition.get_block(i), x.get_block(i), y]
                    res.set_block(i, where(*_args))
                    accum += nelements
                return res

            elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                assert x.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = block_vector.BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition.get_block(i), x[accum: accum + nelements], y[accum: accum + nelements]]
                    res.set_block(i, where(*_args))
                    accum += nelements
                return res

            elif isinstance(x, np.ndarray) and np.isscalar(y):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert x.size == condition.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = block_vector.BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition.get_block(i), x[accum: accum + nelements], y]
                    res.set_block(i, where(*_args))
                    accum += nelements
                return res

            elif np.isscalar(x) and isinstance(y, np.ndarray):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                assert condition.size == y.size, 'Operation on BlockVectors need the same number of blocks on each operand'
                res = block_vector.BlockVector(condition.nblocks)
                accum = 0
                for i in range(condition.nblocks):
                    nelements = condition._brow_lengths[i]
                    _args = [condition.get_block(i), x, y[accum: accum + nelements]]
                    res.set_block(i, where(*_args))
                    accum += nelements
                return res

            elif np.isscalar(x) and np.isscalar(y):
                assert not condition.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
                res = block_vector.BlockVector(condition.nblocks)
                for i in range(condition.nblocks):
                    _args = [condition.get_block(i), x, y]
                    res.set_block(i, where(*_args))
                return res

            else:
                raise NotImplementedError()
    else:
        if n_args == 1:
            return np.where(*args)
        else:

            x = args[1]
            y = args[2]
            if isinstance(x, block_vector.BlockVector):
                # ToDo: add logger to give warning here
                x = x.flatten()
            if isinstance(y, block_vector.BlockVector):
                # ToDo: add logger to give warning here
                y = y.flatten()
            _args = [condition, x, y]
            return np.where(*_args)


def isin(element, test_elements, assume_unique=False, invert=False):

    if isinstance(element, block_vector.BlockVector) and isinstance(test_elements, block_vector.BlockVector):
        assert not element.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        assert not test_elements.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        assert element.nblocks == test_elements.nblocks, 'Operation on BlockVectors need the same number of blocks on each operand'
        res = block_vector.BlockVector(element.nblocks)
        for i in range(element.nblocks):
            res.set_block(i, isin(element.get_block(i),
                                  test_elements.get_block(i),
                                  assume_unique=assume_unique,
                                  invert=invert))
        return res

    elif isinstance(element, block_vector.BlockVector) and isinstance(test_elements, np.ndarray):

        assert not element.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        res = block_vector.BlockVector(element.nblocks)
        for i in range(element.nblocks):
            res.set_block(i, isin(element.get_block(i),
                                  test_elements,
                                  assume_unique=assume_unique,
                                  invert=invert))
        return res

    elif isinstance(element, np.ndarray) and isinstance(test_elements, np.ndarray):

        return np.isin(element,
                       test_elements,
                       assume_unique=assume_unique,
                       invert=invert)

    else:
        raise NotImplementedError()


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):

    if return_indices:
        raise NotImplementedError()

    if isinstance(ar1, tuple) and len(ar1) == 1:
        x = ar1[0]
    elif isinstance(ar1, np.ndarray) or isinstance(ar1, block_vector.BlockVector):
        x = ar1
    else:
        raise RuntimeError('ar1 type not recognized. Needs to be np.ndarray or BlockVector')

    if isinstance(ar2, tuple) and len(ar2) == 1:
        y = ar2[0]
    elif isinstance(ar2, np.ndarray) or isinstance(ar1, block_vector.BlockVector):
        y = ar2
    else:
        raise RuntimeError('ar2 type not recognized. Needs to be np.ndarray or BlockVector')

    if isinstance(x, block_vector.BlockVector) and isinstance(y, block_vector.BlockVector):

        assert x.nblocks == y.nblocks, "Number of blocks does not match"
        assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'

        res = block_vector.BlockVector(x.nblocks)
        for i in range(x.nblocks):
            res.set_block(i, intersect1d(x.get_block(i), y.get_block(i), assume_unique=assume_unique))
        return res
    elif isinstance(x, block_vector.BlockVector) and isinstance(y, np.ndarray):
        assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'

        res = block_vector.BlockVector(x.nblocks)
        for i in range(x.nblocks):
            res.set_block(i, np.intersect1d(x.get_block(i), y, assume_unique=assume_unique))
        return res
    elif isinstance(x, np.ndarray) and isinstance(y, block_vector.BlockVector):

        assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'

        res = block_vector.BlockVector(y.nblocks)
        for i in range(y.nblocks):
            res.set_block(i, np.intersect1d(x, y.get_block(i), assume_unique=assume_unique))
        return res
    else:
        return np.intersect1d(x, y, assume_unique=assume_unique)


def setdiff1d(ar1, ar2, assume_unique=False):

    if isinstance(ar1, tuple) and len(ar1) == 1:
        x = ar1[0]
    elif isinstance(ar1, np.ndarray) or isinstance(ar1, block_vector.BlockVector):
        x = ar1
    else:
        raise RuntimeError('ar1 type not recognized. Needs to be np.ndarray or BlockVector')

    if isinstance(ar2, tuple) and len(ar2) == 1:
        y = ar2[0]
    elif isinstance(ar2, np.ndarray) or isinstance(ar1, block_vector.BlockVector):
        y = ar2
    else:
        raise RuntimeError('ar2 type not recognized. Needs to be np.ndarray or BlockVector')

    if isinstance(x, block_vector.BlockVector) and isinstance(y, block_vector.BlockVector):

        assert x.nblocks == y.nblocks, "Number of blocks does not match"
        assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'

        res = block_vector.BlockVector(x.nblocks)
        for i in range(x.nblocks):
            res.set_block(i, setdiff1d(x.get_block(i), y.get_block(i), assume_unique=assume_unique))
        return res
    elif isinstance(x, block_vector.BlockVector) and isinstance(y, np.ndarray):
        assert not x.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'
        res = block_vector.BlockVector(x.nblocks)
        for i in range(x.nblocks):
            res.set_block(i, np.setdiff1d(x.get_block(i), y, assume_unique=assume_unique))
        return res
    elif isinstance(x, np.ndarray) and isinstance(y, block_vector.BlockVector):

        assert not y.has_none, 'Operation not allowed with None blocks. Specify all blocks in BlockVector'

        res = block_vector.BlockVector(y.nblocks)
        for i in range(y.nblocks):
            res.set_block(i, np.setdiff1d(x, y.get_block(i), assume_unique=assume_unique))
        return res
    else:
        return np.setdiff1d(x, y, assume_unique=assume_unique)

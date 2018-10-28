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

__all__ = ['ceil', 'floor', 'exp', 'log', 'log10', 'sqrt', 'sin',
           'conjugate', 'cos', 'tan', 'arctan', 'arcsinh', 'sinh',
           'cosh', 'abs', 'absolute', 'fabs', 'around', 'log2',
           'arcsin', 'arccos', 'arccosh', 'sum', 'min', 'max', 'mean',
           'prod', 'add', 'subtract', 'divide', 'multiply']


def _unary_operation(x, fname):
    if isinstance(x, np.ndarray):
        f = getattr(np, fname)
        return f(x)
    elif isinstance(x, BlockVector):
        v = BlockVector(x.nblocks)
        for i in range(x.nblocks):
            v[i] = _unary_operation(x[i], fname)
        return v
    else:
        raise NotImplementedError()


def _reduce_operation(x, fname):
    f = getattr(np, fname)
    if isinstance(x, np.ndarray):
        return f(x)
    elif isinstance(x, BlockVector):
        flat_x = x.flatten()
        return f(flat_x)
    else:
        raise NotImplementedError()
    
    
def _binary_operation(x1, x2, fname, keep_blocks=False):
    f = getattr(np, fname)
    if (isinstance(x1, np.ndarray) or np.isscalar(x1)) and (isinstance(x2, np.ndarray)or np.isscalar(x2)):
        return f(x1, x2)
    elif (isinstance(x1, np.ndarray) or np.isscalar(x1)) and isinstance(x2, BlockVector):
        flat_x = x2.flatten()
        res = f(x1, flat_x)
        if not keep_blocks:
            return res
        v = x2.clone()
        v.copyfrom(res)
        return res
    elif (isinstance(x2, np.ndarray)or np.isscalar(x2)) and isinstance(x1, BlockVector):
        flat_x = x1.flatten()
        res = f(flat_x, x2)
        if not keep_blocks:
            return res
        v = x1.clone()
        v.copyfrom(res)
        return res
    elif isinstance(x1, BlockVector) and isinstance(x2, BlockVector):
        flat_x1 = x1.flatten()
        flat_x2 = x2.flatten()
        res = f(flat_x1, flat_x2)
        if not keep_blocks:
            return res
        v = x1.clone()
        v.copyfrom(res)
        return res
    else:
        raise NotImplementedError()


def ceil(x):
    return _unary_operation(x, 'ceil')


def floor(x):
    return _unary_operation(x, 'floor')


def exp(x):
    return _unary_operation(x, 'exp')


def conjugate(x):
    return _unary_operation(x, 'conjugate')


def sin(x):
    return _unary_operation(x, 'sin')


def cos(x):
    return _unary_operation(x, 'cos')


def tan(x):
    return _unary_operation(x, 'tan')


def arctan(x):
    return _unary_operation(x, 'arctan')


def arcsinh(x):
    return _unary_operation(x, 'arcsinh')


def sinh(x):
    return _unary_operation(x, 'sinh')


def cosh(x):
    return _unary_operation(x, 'cosh')


def abs(x):
    return _unary_operation(x, 'abs')


def absolute(x):
    return _unary_operation(x, 'absolute')


def fabs(x):
    return _unary_operation(x, 'fabs')


def around(x):
    return _unary_operation(x, 'around')


def sqrt(x):
    return _unary_operation(x, 'sqrt')


def log(x):
    return _unary_operation(x, 'log')


def log2(x):
    return _unary_operation(x, 'log2')


def log10(x):
    return _unary_operation(x, 'log10')


def arcsin(x):
    return _unary_operation(x, 'arcsin')


def arccos(x):
    return _unary_operation(x, 'arccos')


def arccosh(x):
    return _unary_operation(x, 'arccosh')


# reduce operations
def sum(x):
    return _reduce_operation(x, 'sum')


def mean(x):
    return _reduce_operation(x, 'mean')


def max(x):
    return _reduce_operation(x, 'max')


def min(x):
    return _reduce_operation(x, 'min')


def prod(x):
    return _reduce_operation(x, 'prod')


# binary operations
def add(x1, x2, keep_blocks=False):
    return _binary_operation(x1, x2, 'add', keep_blocks=keep_blocks)


def subtract(x1, x2, keep_blocks=False):
    return _binary_operation(x1, x2, 'subtract', keep_blocks=keep_blocks)


def multiply(x1, x2, keep_blocks=False):
    return _binary_operation(x1, x2, 'multiply', keep_blocks=keep_blocks)


def divide(x1, x2, keep_blocks=False):
    return _binary_operation(x1, x2, 'divide', keep_blocks=keep_blocks)


def allclose(x1, x2, rtol, atol):
    x1_flat = x1.flatten()
    x2_flat = x2.flatten()
    return np.allclose(x1_flat, x2_flat, rtol=rtol, atol=atol)


def concatenate(arrays):
    return np.concatenate([a.flatten() for a in arrays])







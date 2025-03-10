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

# These classes are for checking types consistently and raising errors

from ..dependencies import numpy as np


class BaseBlockVector(object):
    """Base class for block vectors"""

    def __init__(self):
        pass

    # We do not expect classes derived from BaseBlockVector to support
    # the methods below.
    def argpartition(self, kth, axis=-1, kind='introselect', order=None):
        msg = "argpartition not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def argsort(self, axis=-1, kind='quicksort', order=None):
        msg = "argsort not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def byteswap(self, inplace=False):
        msg = "byteswap not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def choose(self, choices, out=None, mode='raise'):
        msg = "choose not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        msg = "diagonal not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def getfield(self, dtype, offset=0):
        msg = "getfield not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def item(self, *args):
        msg = "item not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def itemset(self, *args):
        msg = "itemset not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def newbyteorder(self, new_order='S'):
        msg = "newbyteorder not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def put(self, indices, values, mode='raise'):
        msg = "put not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def partition(self, kth, axis=-1, kind='introselect', order=None):
        msg = "partition not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def repeat(self, repeats, axis=None):
        msg = "repeat not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def reshape(self, shape, order='C'):
        msg = "reshape not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def resize(self, new_shape, refcheck=True):
        msg = "resize not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def searchsorted(self, v, side='left', sorter=None):
        msg = "searchsorted not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def setfield(self, val, dtype, offset=0):
        msg = "setfield not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def setflags(self, write=None, align=None, uic=None):
        msg = "setflags not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def sort(self, axis=-1, kind='quicksort', order=None):
        msg = "sort not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def squeeze(self, axis=None):
        msg = "squeeze not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def swapaxes(self, axis1, axis2):
        msg = "swapaxes not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        msg = "trace not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def argmax(self, axis=None, out=None):
        msg = "argmax not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def argmin(self, axis=None, out=None):
        msg = "argmin not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def take(self, indices, axis=None, out=None, mode='raise'):
        msg = "take not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    # The following vectors are to be supported at some point
    def dump(self, file):
        msg = "dump not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def dumps(self):
        msg = "dumps not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def tobytes(self, order='C'):
        msg = "tobytes not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)


class BaseBlockMatrix(object):
    """Base class for block matrices"""

    def __init__(self):
        pass

    # We do not expect classes derived from BaseBlockVector to support
    # the methods below.
    def tolil(self, copy=False):
        msg = "tolil not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def todia(self, copy=False):
        msg = "todia not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def tobsr(self, blocksize=None, copy=False):
        msg = "tobsr not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def sum(self, axis=None, dtype=None, out=None):
        msg = "sum not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def mean(self, axis=None, dtype=None, out=None):
        msg = "mean not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def diagonal(self, k=0):
        msg = "diagonal not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def nonzero(self):
        msg = "nonzero not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def setdiag(self, values, k=0):
        msg = "setdiag not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def transpose(self, *axes):
        msg = "transpose not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def tostring(self, order='C'):
        msg = "tostring not implemented for {}".format(self.__class__.__name__)
        raise NotImplementedError(msg)


#: NumPy ufuncs that take one vector and are compatible with pyNumero vectors
vec_unary_ufuncs = {
    ## MATH ufuncs
    np.negative,
    np.positive,
    np.absolute,
    np.fabs,
    np.rint,
    np.sign,
    np.conj,
    np.conjugate,
    np.exp,
    np.exp2,
    np.log,
    np.log2,
    np.log10,
    np.expm1,
    np.log1p,
    np.sqrt,
    np.square,
    np.cbrt,
    np.reciprocal,
    ## TRIG ufuncs
    np.sin,
    np.cos,
    np.tan,
    np.arcsin,
    np.arccos,
    np.arctan,
    np.sinh,
    np.cosh,
    np.tanh,
    np.arcsinh,
    np.arccosh,
    np.arctanh,
    np.degrees,
    np.radians,
    np.deg2rad,
    np.rad2deg,
    ## COMPARISON ufuncs
    np.logical_not,
    ## BIT-TWIDDLING ufuncs
    np.invert,
    ## FLOATING ufuncs
    np.isfinite,
    np.isinf,
    np.isnan,
    # np.isnat,  # only defined for datetime
    np.fabs,  # numpy docs list here and in MATH
    np.signbit,
    np.spacing,
    # np.modf, # disabled because shape is not preserved
    # np.frexp, # disabled because shape is not preserved
    np.floor,
    np.ceil,
    np.trunc,
    # OTHER (not listed in ufuncs docs)
    np.abs,
}

#: NumPy ufuncs that take two vectors and are compatible with pyNumero vectors
vec_binary_ufuncs = {
    ## MATH ufuncs
    np.add,
    np.subtract,
    np.multiply,
    # np.matmult, # disabled because shape is not preserved
    np.divide,
    np.logaddexp,
    np.logaddexp2,
    np.true_divide,
    np.floor_divide,
    np.power,
    np.float_power,
    np.remainder,
    np.mod,
    np.fmod,
    # np.divmod,  # disabled because shape is not preserved
    np.heaviside,
    np.gcd,
    np.lcm,
    ## TRIG ufuncs
    np.arctan2,
    np.hypot,
    ## BIT-TWIDDLING ufuncs
    np.bitwise_and,
    np.bitwise_or,
    np.bitwise_xor,
    np.left_shift,
    np.right_shift,
    ## COMPARISON ufuncs
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
    np.not_equal,
    np.equal,
    np.logical_and,
    np.logical_or,
    np.logical_xor,
    np.maximum,
    np.minimum,
    np.fmax,
    np.fmin,
    ## FLOATING ufincs
    np.copysign,
    np.nextafter,
    np.ldexp,
    np.fmod,  # numpy docs list here and in MATH
}

#: NumPy ufuncs can be used as reductions for pyNumero vectors
vec_associative_reductions = {
    np.add,
    np.multiply,
    np.bitwise_and,
    np.bitwise_or,
    np.bitwise_xor,
    np.maximum,
    np.minimum,
    np.fmax,
    np.fmin,
}

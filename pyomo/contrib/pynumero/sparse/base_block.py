#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# These classes are for checking types consistently and raising errors


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

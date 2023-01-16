#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import Any, NonNegativeIntegers
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.expression import Expression
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import UnindexedComponent_set

class PiecewiseLinearFunctionData(_BlockData):
    _Block_reserved_words = Any

    def __init__(self, component=None):
        _BlockData.__init__(self, component)

        with self._declare_reserved_components():
            self._expressions = Expression(NonNegativeIntegers)


@ModelComponentFactory.register("Multidimensional piecewise linear function")
class PiecewiseLinearFunction(Block):
    """A piecewise linear function, which may be defined over an index.

    Can be specified in one of several ways:
        1) List of points and a nonlinear function to approximate. In
           this case, the points will be used to derive a triangulation
           of the part of the domain of interest, and a linear function
           approximating the given function will be calculated for each
           of the simplices in the triangulation. In this case, scipy is
           required.
        2) List of breakpoints along each dimension (variable) in the 
           function.

    Args:
        function: Nonlinear function to approximate, given as a Pyomo
            expression
        points: List of points in the same dimension as the domain of the 
            function being approximated. Note that if the pieces of the 
            function are specified this way, we require scipy.
        breakpoints: A ComponentMap mapping each variable in the function
            to a list of breakpoints along the corresponding dimension. If
            the pieces of  
    """
    _ComponentDataClass = PiecewiseLinearFunctionData

    def __new__(cls, *args, **kwds):
        if cls != PiecewiseLinearFunction:
            return super(PiecewiseLinearFunction, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return PiecewiseLinearFunction.__new__(
                ScalarPiecewiseLinearFunction)
        else:
            return IndexedPiecewiseLinearFunction.__new__(
                IndexedPiecewiseLinearFunction)

    def __init__(self, *args, **kwargs):
        _func_arg = kwargs.pop('function', None)
        _points_arg = kwargs.pop('points', None)

        kwargs.setdefault('ctype', PiecewiseLinearFunction)
        Block.__init__(self, *args, **kwargs)

        # args: points, f
        # Or domains, linear functions
        # Or simplices, f
        pass
        # TODO

    def _getitem_when_not_present(self, index):
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)

        obj._index = index

        return obj


class ScalarPiecewiseLinearFunction(PiecewiseLinearFunctionData,
                                    PiecewiseLinearFunction):
    def __init__(self, *args, **kwds):
        self._suppress_ctypes = set()

        PiecewiseLinearFunctionData.__init__(self, self)
        PiecewiseLinearFunction.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index


class IndexedPiecewiseLinearFunction(PiecewiseLinearFunction):
    pass

class PiecewiseLinearExpression():
    # This needs to be an expression node, it is what the _expressions above are
    # going to store.
    pass
    # example : m.c = Constraint(expr=m.pw(m.x, m.y) <= 0)

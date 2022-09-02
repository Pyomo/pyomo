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

from pyomo.common.modeling import NOTSET
from pyomo.common.pyomo_typing import overload
from pyomo.contrib.cp.scheduling_expr import AlwaysIn
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
    StepFunction)

from pyomo.core.base.component import ComponentData
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index

class CumulativeFunctionExpressionData(ComponentData):
    """
    An object that defines a named cumulative function, that is a step function
    over time whose value is a sum of step functions that depend on when
    IntervalVars are scheduled.

    Public Class Attributes
        expr: The expression
    """
    def __init__(self, expr=None, component=None):
        ComponentData.__init__(self, component)
        print(expr)
        self._expr = expr

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, expr):
        self.set_value(expr)

    def set_value(self, expr):
        """Set the expression on this CumulativeFunctionExpression."""
        if expr is None:
            self._expr = None
            return
        if not isinstance(expr, StepFunction):
            raise ValueError("CumulativeFunctionExpression exprs must be "
                             "sums of elementary step function expression "
                             "such as 'Pulse' and 'Step'. Recieved type: %s"
                             % type(expr))
        self._expr = expr

    def within(self, bounds, times):
        return AlwaysIn(self._expr, bounds, times)


@ModelComponentFactory.register(
    "Variable-valued step functions over time for use in scheduling")
class CumulativeFunctionExpression(IndexedComponent):
    """
    An object that defines a named cumulative function, that is a step function
    over time whose value is a sum of step functions that depend on when
    IntervalVars are scheduled. It may be defined over an index.

    Constructor arguments:
        expr: The expression (some of step functions)
        rule: A rule function used to initialize this object
        name: A name for this component
        doc: Text describing this component.
    """
    _ComponentDataClass = CumulativeFunctionExpressionData

    def __new__(cls, *args, **kwds):
        if cls != CumulativeFunctionExpression:
            return super(CumulativeFunctionExpression, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return ScalarCumulativeFunctionExpression.__new__(
                ScalarCumulativeFunctionExpression)
        else:
            return IndexedCumulativeFunctionExpression.__new__(
                IndexedCumulativeFunctionExpression)

    @overload
    def __init__(self, *indices, rule=None, expr=None, name=None,
                 doc=None): ...

    def __init__(self, *args, **kwds):
        _init = self._pop_from_kwargs('CumulativeFunctionExpression', kwds,
                                      ('rule', 'expr'), None)
        self._rule = Initializer(_init, arg_not_specified=NOTSET)

        kwds.setdefault('ctype', CumulativeFunctionExpression)
        IndexedComponent.__init__(self, *args, **kwds)


class ScalarCumulativeFunctionExpression(CumulativeFunctionExpressionData,
                                         CumulativeFunctionExpression):
    def __init__(self, *args, **kwds):
        CumulativeFunctionExpressionData.__init__(self, expr=None,
                                                  component=self)
        CumulativeFunctionExpression.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index


class IndexedCumulativeFunctionExpression(CumulativeFunctionExpression):
    pass

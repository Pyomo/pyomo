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


# pulse
# always_in: (sum of pulses).within(Set) and use get_interval() on the set for error checking. (returns (start, end, step_length)). within will be defined on the root node of the expression tree that is the sum of pulses (they should support + and -)

# element

from pyomo.common.collections import ComponentSet
from pyomo.common.pyomo_typing import overload
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
    BeforeExpression, AtExpression)

from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.initializer import BoundInitializer, Initializer
from pyomo.core.base import ScalarVar, ScalarBooleanVar
from pyomo.core import Integers

from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set)

class IntervalVarTimePoint(ScalarVar):
    """This class defines the abstract interface for a single variable
    denoting a start or end time point of an IntervalVar"""

    __slots__ = ()

    def __init__(self, component=None):
        super().__init__(domain=Integers, ctype=IntervalVarTimePoint)

    def before(self, time, delay=0):
        # These return logical constraint expressions. A node in a logical
        # expression tree.
        return BeforeExpression(self, time, delay)

    def after(self, time, delay=0):
        return BeforeExpression(time, self, -delay)

    def at(self, time, delay=0):
        return AtExpression(self, time, delay)


class IntervalVarLength(ScalarVar):
    """This class defines the abstract interface for a single variable
    denoting a start or end time point of an IntervalVar"""

    __slots__ = ('_before', '_after', '_at')

    def __init__(self, component=None):
        super().__init__(domain=Integers, ctype=IntervalVarLength)


class IntervalVarPresence(ScalarBooleanVar):
    """This class defines the abstract interface for a single variable
    denoting a start or end time point of an IntervalVar"""
    def __init__(self, component=None):
        super().__init__(ctype = IntervalVarPresence)


class IntervalVarData(_BlockData):
    """This class defines the abstract interface for a single interval variable.
    """
    def __init__(self, component=None):
        _BlockData.__init__(self, component)

        self.is_present = IntervalVarPresence()
        self.start_time = IntervalVarTimePoint()
        self.end_time = IntervalVarTimePoint()
        self.length = IntervalVarLength()

    @property
    def optional(self):
        # We only store this information in one place, but it's kind of annoying
        # to have to check if the BooleanVar is fixed, so this way you can ask
        # the IntervalVar directly.
        return not self.is_present.fixed

    @optional.setter
    def optional(self, val):
        if type(val) is not bool:
            raise ValueError(
                "Cannot set 'optional' to %s: Must be True or False." % val)
        if val:
            self.is_present.unfix()
        else:
            self.is_present.fix(True)


@ModelComponentFactory.register("Interval variables for scheduling.")
class IntervalVar(Block):
    """An interval variable, which may be defined over an index.

    Args:
        start (tuple of two integers): Feasible range for the
            interval variable's start time
        end (tuple of two integers, optional): Feasible range for the
            interval variables end time
        length (integer or tuple of two integers, optional): Feasible
            range for the length of the interval variable
        optional (boolean, optional) : If False, the interval variable must
            be scheduled. Otherwise the interval variable is optionally
            present. Default behavior is 'False'
        name (str, optional): Name for this component.
        doc (str, optional): Text describing this component.
    """

    _ComponentDataClass = IntervalVarData

    def __new__(cls, *args, **kwds):
        if cls != IntervalVar:
            return super(IntervalVar, cls).__new__(cls)
        if args == ():
            return ScalarIntervalVar.__new__(ScalarIntervalVar)
        else:
            return IndexedIntervalVar.__new__(IndexedIntervalVar)

    @overload
    def __init__(self, *indices, start=None, end=None, length=None,
                 optional=False, name=None, doc=None): ...

    def __init__(self, *args, **kwargs):
        _start_arg = kwargs.pop('start', None)
        _end_arg = kwargs.pop('end', None)
        _length_arg = kwargs.pop('length', None)
        _optional_arg = kwargs.pop('optional', False)

        kwargs.setdefault('ctype', IntervalVar)
        Block.__init__(self, *args, **kwargs)

        self._start_bounds = BoundInitializer(self, _start_arg)
        self._end_bounds = BoundInitializer(self, _end_arg)
        self._length_bounds = BoundInitializer(self, _length_arg)
        self._optional = Initializer(_optional_arg)

    def _getitem_when_not_present(self, index):
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        parent = obj.parent_block()
        obj._index = index
        if not self._optional:
            obj.is_present.fix(True)

        obj.start_time.bounds = self._start_bounds(parent, index)
        obj.end_time.bounds = self._end_bounds(parent, index)
        obj.length.bounds = self._length_bounds(parent, index)
        # hit the setter so I get error checking
        obj.optional = self._optional(parent, index)

        return obj


class ScalarIntervalVar(IntervalVarData, IntervalVar):
    def __init__(self, *args, **kwds):
        # TODO: John, it really does fail without this, in _BlockData's
        # implementation of __getattr__
        self._suppress_ctypes = set()

        IntervalVarData.__init__(self, self)
        IntervalVar.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index


class IndexedIntervalVar(IntervalVar):
    pass

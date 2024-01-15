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

from pyomo.common.collections import ComponentSet
from pyomo.common.pyomo_typing import overload
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
    BeforeExpression,
    AtExpression,
)

from pyomo.core import Integers, value
from pyomo.core.base import Any, ScalarVar, ScalarBooleanVar
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.initializer import BoundInitializer, Initializer
from pyomo.core.expr import GetItemExpression


class IntervalVarTimePoint(ScalarVar):
    """This class defines the abstract interface for a single variable
    denoting a start or end time point of an IntervalVar"""

    __slots__ = ()

    def get_associated_interval_var(self):
        return self.parent_block()

    def before(self, time, delay=0):
        return BeforeExpression((self, time, delay))

    def after(self, time, delay=0):
        return BeforeExpression((time, self, delay))

    def at(self, time, delay=0):
        return AtExpression((self, time, delay))


class IntervalVarStartTime(IntervalVarTimePoint):
    """This class defines a single variable denoting a start time point
    of an IntervalVar"""

    def __init__(self):
        super().__init__(domain=Integers, ctype=IntervalVarStartTime)


class IntervalVarEndTime(IntervalVarTimePoint):
    """This class defines a single variable denoting an end time point
    of an IntervalVar"""

    def __init__(self):
        super().__init__(domain=Integers, ctype=IntervalVarEndTime)


class IntervalVarLength(ScalarVar):
    """This class defines the abstract interface for a single variable
    denoting the length of an IntervalVar"""

    __slots__ = ()

    def __init__(self):
        super().__init__(domain=Integers, ctype=IntervalVarLength)

    def get_associated_interval_var(self):
        return self.parent_block()


class IntervalVarPresence(ScalarBooleanVar):
    """This class defines the abstract interface for a single Boolean variable
    denoting whether or not an IntervalVar is scheduled"""

    __slots__ = ()

    def __init__(self):
        super().__init__(ctype=IntervalVarPresence)

    def get_associated_interval_var(self):
        return self.parent_block()


class IntervalVarData(_BlockData):
    """This class defines the abstract interface for a single interval variable."""

    # We will put our four variables on this, and everything else is off limits.
    _Block_reserved_words = Any

    def __init__(self, component=None):
        _BlockData.__init__(self, component)

        with self._declare_reserved_components():
            self.is_present = IntervalVarPresence()
            self.start_time = IntervalVarStartTime()
            self.end_time = IntervalVarEndTime()
            self.length = IntervalVarLength()

    @property
    def optional(self):
        # We only store this information in one place, but it's kind of annoying
        # to have to check if the BooleanVar is fixed, so this way you can ask
        # the IntervalVar directly.
        return not self.is_present.fixed or (
            self.is_present.fixed and not value(self.is_present)
        )

    @optional.setter
    def optional(self, val):
        if type(val) is not bool:
            raise ValueError(
                "Cannot set 'optional' to %s: Must be True or False." % val
            )
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
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return ScalarIntervalVar.__new__(ScalarIntervalVar)
        else:
            return IndexedIntervalVar.__new__(IndexedIntervalVar)

    @overload
    def __init__(
        self,
        *indices,
        start=None,
        end=None,
        length=None,
        optional=False,
        name=None,
        doc=None
    ):
        ...

    def __init__(self, *args, **kwargs):
        _start_arg = kwargs.pop('start', None)
        _end_arg = kwargs.pop('end', None)
        _length_arg = kwargs.pop('length', None)
        _optional_arg = kwargs.pop('optional', False)

        kwargs.setdefault('ctype', IntervalVar)
        Block.__init__(self, *args, **kwargs)

        self._start_bounds = BoundInitializer(_start_arg, self)
        self._end_bounds = BoundInitializer(_end_arg, self)
        self._length_bounds = BoundInitializer(_length_arg, self)
        self._optional = Initializer(_optional_arg)

    def _getitem_when_not_present(self, index):
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        parent = obj.parent_block()
        obj._index = index

        if self._start_bounds is not None:
            obj.start_time.bounds = self._start_bounds(parent, index)
        if self._end_bounds is not None:
            obj.end_time.bounds = self._end_bounds(parent, index)
        if self._length_bounds is not None:
            obj.length.bounds = self._length_bounds(parent, index)
        # hit the setter so I get error checking
        obj.optional = self._optional(parent, index)

        return obj


class ScalarIntervalVar(IntervalVarData, IntervalVar):
    def __init__(self, *args, **kwds):
        self._suppress_ctypes = set()

        IntervalVarData.__init__(self, self)
        IntervalVar.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index


class IndexedIntervalVar(IntervalVar):
    # We allow indexing IntervalVars by expressions (including Vars).
    def __getitem__(self, args):
        tmp = args if args.__class__ is tuple else (args,)
        if any(
            hasattr(arg, 'is_potentially_variable') and arg.is_potentially_variable()
            for arg in tmp
        ):
            return GetItemExpression((self,) + tmp)
        return super().__getitem__(args)

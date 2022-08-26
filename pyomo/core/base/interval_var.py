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

import logging

from pyomo.common.collections import Sequence
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer

from pyomo.core.base.block import _BlockData, Block
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.initializer import BoundInitializer, Initializer
from pyomo.core.base import Var, BooleanVar
from pyomo.core import Integers

from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set)

logger = logging.getLogger('pyomo.core')

class _IntervalVarData(_BlockData):
    """This class defines the abstract interface for a single interval variable.
    """
    def __init__(self, component=None):
        _BlockData.__init__(self, component)

        self.is_present = BooleanVar()
        self.start_time = Var(domain=Integers)
        self.end_time = Var(domain=Integers)
        self.length = Var(domain=Integers)

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
            self.is_present.fixed = False
        else:
            self.is_present.fix(True)

@ModelComponentFactory.register("Interval variables for scheduling.")
class IntervalVar(Block):
    """And interval variable, which may be defined over an index.

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

    _ComponentDataClass = _IntervalVarData

    def __new__(cls, *args, **kwds):
        if cls != IntervalVar:
            return super(IntervalVar, cls).__new__(cls)
        if args == ():
            return ScalarIntervalVar.__new__(ScalarIntervalVar)
        else:
            return IndexedIntervalVar.__new__(IndexedIntervalVar)

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

        if self._start_bounds is not None:
            start_bounds = self._start_bounds(parent, index)
            if not isinstance(start_bounds, Sequence):
                start_bounds = (start_bounds, start_bounds)
            obj.start_time.lower, obj.start_time.upper = start_bounds
        if self._end_bounds is not None:
            end_bounds = self._end_bounds(parent, index)
            if not isinstance(end_bounds, Sequence):
                end_bounds = (end_bounds, end_bounds)
            obj.end_time.lower, obj.end_time.upper = end_bounds
        if self._length_bounds is not None:
            length_bounds = self._length_bounds(parent, index)
            if not isinstance(length_bounds, Sequence):
                length_bounds = (length_bounds, length_bounds)
            obj.length.lower, obj.length.upper = length_bounds
        if self._optional is not None:
            # hit the setter so I get error checking
            obj.optional = self._optional(parent, index)

        return obj

class ScalarIntervalVar(_IntervalVarData, IntervalVar):
    def __init__(self, *args, **kwds):
        self._suppress_ctypes = set()

        _IntervalVarData.__init__(self, self)
        IntervalVar.__init__(self, *args, **kwds)
        self._data[None] = self
        self._index = UnindexedComponent_index

class IndexedIntervalVar(IntervalVar):
    pass

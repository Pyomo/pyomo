#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['RangeSet']

import logging
import math
from six.moves import xrange

from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import value
from pyomo.core.base.sets import OrderedSimpleSet
from pyomo.core.base.set_types import Integers, Reals
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.plugin import ModelComponentFactory

logger = logging.getLogger('pyomo.core')


@ModelComponentFactory.register("A sequence of numeric values.  RangeSet(start,end,step) is a sequence starting a value 'start', and increasing in values by 'step' until a value greater than or equal to 'end' is reached.")
class RangeSet(OrderedSimpleSet):
    """
    A set that represents a list of numeric values.
    """

    def __init__(self, *args, **kwds):
        """
        Construct a list of integers
        """
        if len(args) == 0:
            raise RuntimeError("Attempting to construct a RangeSet object with no arguments!")
        super(RangeSet, self).__init__(**kwds)
        self._type=RangeSet
        #
        if len(args) == 1:
            #
            # RangeSet(end) generates the set: 1 ... end
            #
            self._start=1
            self._end=args[0]
            self._step=1
        elif len(args) == 2:
            #
            # RangeSet(start,end) generates the set: start ... end
            #
            self._start=args[0]
            self._end=args[1]
            self._step=1
        else:
            #
            # RangeSet(start,end,step) generates the set: start, start+step, start+2*step, ... end
            #
            self._start=args[0]
            self._end=args[1]
            self._step=args[2]
        #
        self.ordered = True     # This is an ordered set
        self.value = None       # No internal set data
        self.virtual = True     # This is a virtual set
        self.concrete = True    # This is a concrete set
        self._len = 0           # This is set by the construct() method

    def construct(self, values=None):
        """
        Initialize set data
        """
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True
        #
        # We call value() here for cases like Expressions, mutable
        # Params and the like
        #
        self._start_val = value(self._start)
        self._end_val = value(self._end)
        self._step_val = value(self._step)
        #
        # The set generates integer values if the starting value,
        # step and end value are all integers.  Otherwise, the set
        # generates real values.
        #
        if type(self._start_val) is int and type(self._step) is int and type(self._end_val) is int:
            self.domain = Integers
        else:
            self.domain = Reals
        #
        # Compute the set length and upper bound
        #
        if self.filter is None and self.validate is None:
            #
            # Directly compute the number of elements in the set, from
            # which the upper-bound is computed.
            #
            self._len = int(math.floor((self._end_val-self._start_val+self._step_val+1e-7)//self._step_val))
            ub = self._start_val + (self._len-1)*self._step_val
        else:
            #
            # Iterate through the set to compute the upper bound
            # and number of elements.
            #
            ub = self._start_val
            ctr=0
            for i in self:
                ub = i
                ctr += 1
            self._len = ctr
        #
        # Set the bounds information
        #
        self._bounds = (self._start_val, ub)
        timer.report()

    def __len__(self):
        """
        Return the pre-computed set length
        """
        return self._len

    def __iter__(self):
        if not self._constructed:
            raise RuntimeError(
                "Cannot iterate over abstract RangeSet '%s' before it has "
                "been constructed (initialized)." % (self.name,) )
        if self.filter is None and self.validate is None:
            #
            # Iterate through all set elements
            #
            for i in xrange(self._len):
                yield self._start_val + i*self._step_val
        else:
            #
            # Iterate through all set elements and filter
            # and/or validate the element values.
            #
            for i in xrange(int((self._end_val-self._start_val+self._step_val+1e-7)//self._step_val)):
                val = self._start_val + i*self._step_val
                if not self.filter is None and not apply_indexed_rule(self, self.filter, self._parent(), val):
                    continue
                if not self.validate is None and not apply_indexed_rule(self, self.validate, self._parent(), val):
                    continue
                yield val

    def data(self):
        """The underlying set data."""
        return set(self)

    def first(self):
        """The first element is the lower bound"""
        return self._bounds[0]

    def last(self):
        """The last element is the upper bound"""
        return self._bounds[1]

    def member(self, key):
        """
        Return the value associated with this key.
        """
        logger.warning("DEPRECATED: The RangeSet method \"x.member(idx)\" "
                       "is deprecated and will be removed in Pyomo 5.0.  "
                       "Use x[idx] instead.")
        return self.__getitem__(key)

    def __getitem__(self, key):
        """
        Return the value associated with this key.  Valid
        index values are 1 .. len(set), or -1 .. -len(set).
        Negative key values index from the end of the set.
        """
        if key >= 1:
            if key > self._len:
                raise IndexError("Cannot index a RangeSet past the last element")
            return self._start_val + (key-1)*self._step_val
        elif key < 0:
            if self._len+key < 0:
                raise IndexError("Cannot index a RangeSet past the first element")
            return self._start_val + (self._len+key)*self._step_val
        else:
            raise IndexError("Valid index values for sets are 1 .. len(set) or -1 .. -len(set)")

    def _set_contains(self, element):
        """
        Test if the specified element in this set.
        """
        try:
            x = element - self._start_val
            if x % self._step_val != 0:
                #
                # If we are doing floating-point arithmetic, there is a
                # chance that we are seeing roundoff error...
                #
                if math.fabs((x + 1e-7) % self._step_val) > 2e-7:
                    return False
            if element < self._bounds[0] or element > self._bounds[1]:
                return False
        except:
            #
            # This exception is triggered when type(element) is not int or float.
            #
            return False
        #
        # Now see if the element if filtered or invalid.
        #
        if self.filter is not None and not self.filter(element):
            return False
        if self.validate is not None and not self.validate(self, element):
            return False
        return True




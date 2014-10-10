#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['RangeSet']

import math
from six.moves import xrange

from pyomo.core.base.sets import SimpleSet
from pyomo.core.base.expr import _ExpressionBase
from pyomo.core.base.set_types import Integers, Reals
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.numvalue import value
from pyomo.core.base.component import register_component


class RangeSet(SimpleSet):
    """
    A set that represents a list of numeric values.
    """

    def __init__(self, *args, **kwds):
        """
        Construct a list of integers
        """
        if len(args) == 0:
            raise RuntimeError("Attempting to construct a RangeSet object with no arguments!")
        SimpleSet.__init__(self, **kwds)
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
        self._constructed=True
        #
        # Set the value of the set parameters using
        # expressions
        #
        if isinstance(self._start,_ExpressionBase):
            self._start_val = self._start()
        else:
            self._start_val = value(self._start)
        if isinstance(self._end,_ExpressionBase):
            self._end_val = self._end()
        else:
            self._end_val = value(self._end)
        if isinstance(self._step,_ExpressionBase):
            self._step_val = self._step()
        else:
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

    def __len__(self):
        """
        Return the pre-computed set length
        """
        return self._len

    def __iter__(self):
        if not self._constructed:
            raise RuntimeError("A RangeSet component must be initialized before a user can iterate over its elements.")
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

    def __eq__(self, other):
        """
        Equality comparison

        RangeSet inherits from SimpleSet, so we need to
        replicate this logic from SetOf.__eq__
        """
        if other is None:
            return False
        other = self._set_repn(other)
        if self.dimen != other.dimen:
            return False
        if other.concrete and len(self) != len(other):
            return False
        ctr = 0
        for i in other:
            if not i in self:
                return False
            ctr += 1
        return other.concrete or ctr == len(self)

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


register_component(RangeSet, "A sequence of numeric values.  RangeSet(start,end,step) is a sequence starting a value 'start', and increasing in values by 'step' until a value greater than or equal to 'end' is reached.")


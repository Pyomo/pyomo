#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import itertools
import logging

from math import copysign
try:
    from math import remainder
except ImportError:
    def remainder(a,b):
        ans = a % b
        if ans > abs(b/2.):
            ans -= b
        return ans
from six import iteritems
from six.moves import xrange
from sys import exc_info

from pyutilib.misc.misc import flatten_tuple

from pyomo.common.deprecation import deprecated
from pyomo.common.errors import DeveloperError
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
    native_types, native_numeric_types, as_numeric
)
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set
)
from pyomo.core.base.misc import sorted_robust, apply_indexed_rule

logger = logging.getLogger('pyomo.core')

def process_setarg(arg):
    """
    Process argument and return an associated set object.

    This method is used by IndexedComponent
    """
    if isinstance(arg,_SetData):
        # Argument is a non-indexed Set instance
        return arg
    elif isinstance(arg,IndexedSet):
        # Argument is an indexed Set instance
        raise TypeError("Cannot index a component with an indexed set")
    elif isinstance(arg,Component):
        # Argument is some other component
        raise TypeError("Cannot index a component with a non-set "
                        "component: %s" % (arg.name))
    else:
        try:
            #
            # If the argument has a set_options attribute, then use
            # it to initialize a set
            #
            options = getattr(arg,'set_options')
            options['initialize'] = arg
            return Set(**options)
        except:
            pass
    # Argument is assumed to be an initialization function
    return Set(initialize=arg)


@deprecated('The set_options decorator seems nonessential and is deprecated')
def set_options(**kwds):
    """
    This is a decorator for set initializer functions.  This
    decorator allows an arbitrary dictionary of values to passed
    through to the set constructor.

    Examples:
        @set_options(dimen=3)
        def B_index(model):
            return [(i,i+1,i*i) for i in model.A]

        @set_options(domain=Integers)
        def B_index(model):
            return range(10)
    """
    def decorator(func):
        func.set_options = kwds
        return func
    return decorator


def simple_set_rule( fn ):
    """
    This is a decorator that translates None into Set.End.
    This supports a simpler syntax in set rules, though these can be
    more difficult to debug when errors occur.

    Example:

    @simple_set_rule
    def A_rule(model, i, j):
        ...
    """

    def wrapper_function ( *args, **kwargs ):
        value = fn( *args, **kwargs )
        if value is None:
            return Set.End
        return value
    return wrapper_function


class _UnknownSetDimen(object): pass

# What do sets do?
#
# ALL:
#   __contains__
#   __len__ (Note: None for all infinite sets)
#
# Note: FINITE implies DISCRETE. Infinite discrete sets cannot be iterated
#
# FINITE: ALL +
#   __iter__, __reversed__
#   add()
#   sorted(), ordered()
#
# ORDERED: FINITE +
#   __getitem__
#   next(), prev(), first(), last()
#   ord()
#
# When we do math, the least specific set dictates the API of the resulting set.


class _ClosedNumericRange(object):
    """A representation of a closed numeric range.

    This class represents a contiguous range of numbers.  The class
    mimics the Pyomo (*not* Python) `range` API, with a Start, End, and
    Step.  The Step is a signed int.  If the Step is 0, the range is
    continuous.  The End *is* included in the range.

    While the class name implies that the range is always closed, it is
    not strictly finite, as None is allowed for then End value (and the
    Start value, for continuous ranges only).

    """
    __slots__ = ('start','end','step')
    _EPS = 1e-15

    def __init__(self, start, end, step):
        if int(step) != step:
            raise ValueError(
                "_ClosedNumericRange step must be int (got %s)" % (step,))
        step = int(step)
        if start is None:
            if step:
                raise ValueError("_ClosedNumericRange: start must not be None "
                                 "for non-continuous steps")
        elif end is not None:
            if step == 0 and end < start:
                raise ValueError(
                    "_ClosedNumericRange: start must be <= end for "
                    "continuous ranges (got %s..%s)" % (start,end)
                )
            elif (end-start)*step < 0:
                raise ValueError(
                    "_ClosedNumericRange: start, end ordering incompatible "
                    "with step direction (got [%s:%s:%s])" % (start,end,step)
                )
            if step:
                n = int( (end - start) / step )
                end = start + n*step
        if start == end:
            # If this is a scalar, we will force the step to be 0 (so that
            # things like [1:5:10] == [1:50:100] are easier to validate)
            step = 0
        self.start = start
        self.end = end
        self.step = step

    def __getstate__(self):
        """
        Retrieve the state of this object as a dictionary.

        This method must be defined because this class uses slots.
        """
        state = {} #super(_ClosedNumericRange, self).__getstate__()
        for i in _ClosedNumericRange.__slots__:
            state[i] = getattr(self, i)
        return state

    def __setstate__(self, state):
        """
        Set the state of this object using values from a state dictionary.

        This method must be defined because this class uses slots.
        """
        for key, val in iteritems(state):
            # Note: per the Python data model docs, we explicitly
            # set the attribute using object.__setattr__() instead
            # of setting self.__dict__[key] = val.
            object.__setattr__(self, key, val)

    def __str__(self):
        if self.step == 0:
            if self.start == self.end:
                return "[%s]" % (self.start, )
            else:
                return "[%s,%s]" % (self.start, self.end)
        elif self.step == 1:
            return "[%s:%s]" % (self.start, self.end)
        else:
            return "[%s:%s:%s]" % (self.start, self.end, self.step)

    __repr__ = __str__

    def __eq__(self, other):
        assert type(other) is _ClosedNumericRange
        return self.start == other.start \
            and self.end == other.end \
            and self.step == other.step

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        # NumericRanges must hold items that are comparable to ints
        try:
            if value.__class__(0) != 0:
                return False
        except:
            return False

        if self.step:
            _dir = copysign(1, self.step)
            return (
                (value - self.start) * copysign(1, self.step) >= 0
                and (self.end is None or
                     _dir*(self.end - self.start) >= _dir*(value - self.start))
                and abs(remainder(value - self.start, self.step)) <= self._EPS
            )
        else:
            return (self.start is None or value >= self.start) \
                   and (self.end is None or value <= self.end)

    @staticmethod
    def _continuous_discrete_disjoint(cont, disc):
        # At this point, we know the ranges overlap (tested for at the
        # beginning of isdisjoint()
        d_lb = disc.start if disc.step > 0 else disc.end
        d_ub = disc.end if disc.step > 0 else disc.start
        if cont.start is None or (
                d_lb is not None and cont.start <= d_lb):
            return False
        if cont.end is None or (
                d_ub is not None and cont.end >= d_ub):
            return False
        if cont.end - cont.start >= abs(disc.step):
            return False
        # At this point, the continuous set is shorter than the discrete
        # step.  We need to see if the continuous set ovverlaps one of the
        # points, or lies completely between two.
        #
        # Note, taking the absolute value of the step is safe because we are
        # seeing if the continuous range overlaps a discrete point in the
        # underlying (unbounded) discrete sequence grounded by disc.start
        rStart = remainder(cont.start - disc.start, abs(disc.step))
        rEnd = remainder(cont.end - disc.start, abs(disc.step))
        EPS = _ClosedNumericRange._EPS
        return abs(rStart) > EPS and abs(rEnd) > EPS \
               and rStart - rEnd > 0

    @staticmethod
    def _firstNonNull(minimize, *args):
        ans = None
        for x in args:
            if ans is None:
                ans = x
            elif minimize:
                if x is not None and x < ans:
                    ans = x
            else:
                if x is not None and x > ans:
                    ans = x
        return ans

    def is_finite(self):
        return self.start is not None and self.end is not None \
            and self.step != 0

    def isdisjoint(self, other):
        # First, do a simple sanity check on the endpoints
        s1, e1 = self.start, self.end
        if self.step < 0:
            s1, e1 = e1, s1
        s2, e2 = other.start, other.end
        if other.step < 0:
            s2, e2 = e2, s2
        if (e1 is not None and s2 is not None and e1 < s2) \
                or (e2 is not None and s1 is not None and e2 < s1):
            return True
        # Now, check continuous sets
        if not self.step or not other.step:
            # We now need to check a continuous set is a subset of a discrete
            # set and the continuous set sits between discrete points
            if self.step:
                return _ClosedNumericRange._continuous_discrete_disjoint(
                    other, self)
            elif other.step:
                return _ClosedNumericRange._continuous_discrete_disjoint(
                    self, other)
            else:
                # 2 continuous sets, with overlapping end points: not disjoint
                return False
        # both sets are discrete
        if self.step == other.step:
            return abs(remainder(other.start-self.start, self.step)) \
                   > self._EPS
        # Two infinite discrete sets will *eventually* have a common point.
        # This is trivial for integer steps.  It is not true for float steps
        # with infinite precision (think a step of PI).  However, for finite
        # precision maths, the "float" times a sufficient power of two is an
        # integer.  Is this a distinction we want to make?  Personally,
        # anyone making a discrete set with a non-integer step is asking for
        # trouble.  Maybe the better solution is to require that the step be
        # integer.
        elif self.end is None and other.end is None \
                and self.step*other.step > 0:
            return False
        # OK - just check all the members of one set against the other
        end = _ClosedNumericRange._firstNonNull(
            self.step > 0,
            self.end,
            _ClosedNumericRange._firstNonNull(
                self.step < 0, other.start, other.end)
        )
        i = 0
        item = self.start
        while (self.step>0 and item <= end) or (self.step<0 and item >= end):
            if item in other:
                return False
            i += 1
            item = self.start + self.step*i
        return True

    def issubset(self, other):
        # First, do a simple sanity check on the endpoints
        s1, e1 = self.start, self.end
        if self.step < 0:
            s1, e1 = e1, s1
        s2, e2 = other.start, other.end
        if other.step < 0:
            s2, e2 = e2, s2
        # Checks for unbounded ranges and to make sure self's endpoints are
        # within other's endpoints.
        if s1 is None:
            if s2 is not None:
                return False
        elif s2 is not None and s1 < s2:
            return False
        if e1 is None:
            if e2 is not None:
                return False
        elif e2 is not None and e1 > e2:
            return False
        # If other is continuous, then by definition, self is a subset (
        # regardless of step)
        if other.step == 0:
            return True
        # If other is discrete and self is continuous, then self can't be a
        # subset
        elif self.step == 0:
            return False
        # At this point, both sets are discrete.  Self's period must be a
        # positive integer multiple of other's ...
        EPS = _ClosedNumericRange._EPS
        if abs(remainder(self.step, other.step)) > EPS:
            return False
        # ...and they must shart a point in common
        return abs(remainder(other.start-self.start, other.step)) <= EPS

    @staticmethod
    def _lt(a,b):
        "Return True if a is strictly less than b, with None == -inf"
        if a is None:
            return b is not None
        if b is None:
            return False
        return a < b

    @staticmethod
    def _nooverlap(a,b):
        """Return True if a is strictly before b.

        Note: a(None) == +inf and b(None) == -inf

        """
        if a is None or b is None:
            return False
        return a < b

    @staticmethod
    def _gt(a,b):
        "Return True if a is strictly greater than b, with None == +inf"
        if a is None:
            return b is not None
        if b is None:
            return False
        return a > b

    @staticmethod
    def _min(a,b):
        """Modified implementation of min() with special None handling

        In _ClosedNumericRange objects, None can represent {positive,
        negative} infintiy.  In the context that this method is used,
        None will always be negative infinity, so None is less than any
        non-None value.

        """
        if a is None:
            return b
        elif b is None:
            return a
        return min(a, b)

    @staticmethod
    def _max(a,b):
        """Modified implementation of max() with special None handling

        In _ClosedNumericRange objects, None can represent {positive,
        negative} infintiy.  In the context that this method is used,
        None will always be positive infinity, so None is greater than
        any non-None value.

        """
        if a is None:
            return b
        elif b is None:
            return a
        return max(a, b)

    @staticmethod
    def _split_ranges(cnr, new_step):
        """Split a discrete range into a list of ranges using a new step.

        This takes a single _ClosedNumericRange and splits it into a set
        of new ranges, all of which use a new step.  The new_step must
        be a multiple of the current step.  CNR objects with a step of 0
        are returned unchanged.

        Parameters
        ----------
            cnr: `_ClosedNumericRange`
                The range to split
            new_step: `int`
                The new step to use for returned ranges

        """
        if cnr.step == 0 or new_step == 0:
            return [cnr]

        assert new_step >= abs(cnr.step)
        assert new_step % cnr.step == 0
        _dir = copysign(1, cnr.step)
        _subranges = []
        for i in range(abs(new_step // cnr.step)):
            if ( cnr.end is not None
                 and _dir*(cnr.start + i*cnr.step) > _dir*cnr.end ):
                # Once we walk past the end of the range, we are done
                # (all remaining offsets will be farther past the end)
                break

            _subranges.append(_ClosedNumericRange(
                cnr.start + i*cnr.step, cnr.end, _dir*new_step
            ))
        return _subranges

    def _lcm(self,other_ranges):
        """This computes an approximate Least Common Multiple step"""
        if self.step:
            steps = {abs(self.step)}
            for s in other_ranges:
                steps.add(abs(s.step))
            if 0 in steps:
                steps.remove(0)
            for step1 in sorted(steps):
                for step2 in sorted(steps):
                    if step1 % step2 == 0 and step1 > step2:
                        steps.remove(step2)
            lcm = steps.pop()
            for step in steps:
                lcm *= step
        else:
            lcm = 0
        return lcm

    def range_difference(self, other_ranges):
        """Return the difference between this range and a set of other ranges.

        FIXME: There is a known limitation with range_difference():
        Subtracting a range from another continuous closed range should
        result in an open range.  However, at this moment Open ranges
        aren't supported and this method returns a closed range that
        includes endpoints that mathematically should have been removed.

        Paramters
        ---------
            other_ranges: `iterable`
                An iterable of other range objects to subtract from this range

        """
        other_ranges = list(other_ranges)
        # Find the Least Common Multiple of all the range steps.  We
        # will split discrete ranges into separate ranges with this step
        # so that we can more easily compare them.
        lcm = self._lcm(other_ranges)
        if self.step == 0:
            logger.warn(
                "_ClosedNumericRange.range_difference() does not fully "
                "support closed continuous ranges and gives mathematically "
                "invalid answers (the set should be open, but the endpoints "
                "from the subtracted sets are still present in the result.")

        ans = []
        # Split this range into subranges
        _this = _ClosedNumericRange._split_ranges(self, lcm)
        # Split the other range(s) into subranges
        _other = []
        for s in other_ranges:
            _other.extend(_ClosedNumericRange._split_ranges(s, lcm))
        # For each lhs subrange, t
        for t in _this:
            # Compare it against each rhs range and only keep the
            # subranges of this range that are outside the lhs range
            _subranges = [t]
            for s in _other:
                if s.step and (lcm == 0 or (s.start-t.start) % lcm != 0 ):
                    continue
                tmp = []
                for ref in _subranges:
                    if ref.step >= 0:
                        r_min, r_max = ref.start, ref.end
                    else:
                        r_min, r_max = ref.end, ref.start
                    if s.step >= 0:
                        s_min, s_max = s.start, s.end
                    else:
                        s_min, s_max = s.end, s.start

                    if _ClosedNumericRange._lt(r_min, s_min):
                        if s_min is not None and lcm:
                            s_min -= lcm

                        if r_min is None:
                            tmp.append(_ClosedNumericRange(
                                _ClosedNumericRange._min(r_max, s_min),
                                r_min,
                                -lcm
                            ))
                        else:
                            tmp.append(_ClosedNumericRange(
                                r_min,
                                _ClosedNumericRange._min(r_max, s_min),
                                lcm
                            ))

                    if _ClosedNumericRange._gt(r_max, s_max):
                        if s_max is not None and lcm:
                            s_max += lcm
                        tmp.append(_ClosedNumericRange(
                            _ClosedNumericRange._max(r_min, s_max),
                            r_max,
                            lcm
                        ))
                _subranges = tmp
            ans.extend(_subranges)
        return ans

    def range_intersection(self, other_ranges):
        """Return the intersection between this range and a set of other ranges.

        Paramters
        ---------
            other_ranges: `iterable`
                An iterable of other range objects to intersect with this range

        """
        other_ranges = list(other_ranges)
        # Find the Least Common Multiple of all the range steps.  We
        # will split discrete ranges into separate ranges with this step
        # so that we can more easily compare them.
        lcm = self._lcm(other_ranges)

        ans = []
        # Split this range into subranges
        _this = _ClosedNumericRange._split_ranges(self, lcm)
        # Split the other range(s) into subranges
        _other = []
        for s in other_ranges:
            _other.extend(_ClosedNumericRange._split_ranges(s, lcm))
        # For each lhs subrange, t
        for t in _this:
            # Compare it against each rhs range and only keep the
            # subranges of this range that are inside the lhs range
            for s in _other:
                if s.step and t.step and (s.start-t.start) % lcm != 0:
                    continue
                if t.step >= 0:
                    t_min, t_max = t.start, t.end
                else:
                    t_min, t_max = t.end, t.start
                if s.step >= 0:
                    s_min, s_max = s.start, s.end
                else:
                    s_min, s_max = s.end, s.start

                if _ClosedNumericRange._nooverlap(s_max, t_min):
                    continue
                if _ClosedNumericRange._nooverlap(t_max, s_min):
                    continue

                step = abs(t.step if t.step else s.step)
                intersect_start = _ClosedNumericRange._max(t_min, s_min)
                if step and intersect_start is None:
                    ans.append(_ClosedNumericRange(
                        _ClosedNumericRange._min(t_max, s_max),
                        intersect_start,
                        -step
                    ))
                else:
                    ans.append(_ClosedNumericRange(
                        intersect_start,
                        _ClosedNumericRange._min(t_max, s_max),
                        step
                    ))
        return ans


class _AnyRange(object):
    """A range object for representing Any sets"""

    def __init__(self):
        self.start = None
        self.end = None
        self.step = 0

    def __str__(self):
        return "[*]"

    def __eq__(self, other):
        return isinstance(other, _AnyRange)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        return True

    def isdisjoint(self, other):
        return False

    def issubset(self, other):
        return isinstance(other, _AnyRange)

    def range_difference(self, other):
        for o in other:
            if isinstance(o, _AnyRange):
                return []
        else:
            return [self]

    def range_intersection(self, other):
        return other


# A trivial class that we can use to test if an object is a "legitimate"
# set (either SimpleSet, or a member of an IndexedSet)
class _SetDataBase(ComponentData):
    """The base for all objects that can be used as a component indexing set.
    """
    __slots__ = ()


class _SetData(_SetDataBase):
    """The base for all Pyomo AML objects that can be used as a component
    indexing set.

    Derived versions of this class can be used as the Index for any
    IndexedComponent (including IndexedSet)."""
    __slots__ = tuple()

    def __contains__(self, value):
        raise DeveloperError("Derived set class (%s) failed to "
                             "implement __contains__" % (type(self).__name__,))

    def __len__(self):
        raise DeveloperError("Derived set class (%s) failed to "
                             "implement __len__" % (type(self).__name__,))

    def is_finite(self):
        """Returns True if this is a finite discrete (iterable) Set"""
        return False

    def is_ordered(self):
        """Returns True if this is an ordered finite discrete (iterable) Set"""
        return False

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, _SetData):
            other_is_finite = other.is_finite()
        else:
            other_is_finite = True
            try:
                other = SetOf(other)
            except:
                pass
        if self.is_finite():
            if not other_is_finite:
                return False
            if len(self) != len(other):
                return False
            for x in self:
                if x not in other:
                    return False
            return True
        elif other_is_finite:
            return False
        return self.issubset(other) and other.issubset(self)

    def __ne__(self, other):
        return not self.__eq__(other)

    def isdisjoint(self, other):
        # For efficiency, if the other is not a Set, we will try converting
        # it to a Python set() for efficient lookup.
        if isinstance(other, _SetData):
            other_is_finite = other.is_finite()
        else:
            other_is_finite = True
            try:
                other = set(other)
            except:
                pass
        if self.is_finite():
            for x in self:
                if x in other:
                    return False
            return True
        elif other_is_finite:
            for x in other:
                if x in self:
                    return False
            return True
        else:
            all(r.isdisjoint(s) for r in self.ranges() for s in other.ranges())

    def issubset(self, other):
        if isinstance(other, _SetData):
            other_is_finite = other.is_finite()
        else:
            other_is_finite = True
            try:
                other = set(other)
            except:
                pass
        if self.is_finite():
            for x in self:
                if x not in other:
                    return False
            return True
        elif other_is_finite:
            return False
        else:
            for r in self.ranges():
                if r.range_difference(other.ranges()):
                    return False
            return True

    def issuperset(self, other):
        # For efficiency, if the other is not a Set, we will try converting
        # it to a Python set() for efficient lookup.
        if isinstance(other, _SetData):
            other_is_finite = other.is_finite()
        else:
            other_is_finite = True
            try:
                other = set(other)
            except:
                pass
        if other_is_finite:
            for x in other:
                if x not in self:
                    return False
            return True
        elif self.is_finite():
            return False
        else:
            return other.issubset(self)

    def union(self, *args):
        """
        Return the union of this set with one or more sets.
        """
        tmp = self
        for arg in args:
            tmp = _SetUnion(tmp, arg)
        return tmp

    def intersection(self, *args):
        """
        Return the intersection of this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = _SetIntersection(tmp, arg)
        return tmp

    def difference(self, *args):
        """
        Return the difference between this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = _SetDifference(tmp, arg)
        return tmp

    def symmetric_difference(self, *args):
        """
        Return the symmetric difference of this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = _SetSymmetricDifference(tmp, arg)
        return tmp

    def cross(self, *args):
        """
        Return the cross-product between this set and one or more sets
        """
        tmp = self
        for arg in args:
            tmp = _SetProduct(tmp, arg)
        return tmp

    # <= is equivalent to issubset
    # >= is equivalent to issuperset
    # |  is equivalent to union
    # &  is equivalent to intersection
    # -  is equivalent to difference
    # ^  is equivalent to symmetric_difference
    # *  is equivalent to cross

    __le__  = issubset
    __ge__  = issuperset
    __or__  = union
    __and__ = intersection
    __sub__ = difference
    __xor__ = symmetric_difference
    __mul__ = cross

    def __ror__(self, other):
        return Set(initialize=other) | self

    def __rand__(self, other):
        return Set(initialize=other) & self

    def __rsub__(self, other):
        return Set(initialize=other) - self

    def __rxor__(self, other):
        return Set(initialize=other) ^ self

    def __rmul__(self, other):
        return Set(initialize=other) * self

    def __lt__(self,other):
        """
        Return True if the set is a strict subset of 'other'

        TODO: verify that this is sufficiently efficient
             (vs. an explicit implimentation).
        """
        return self <= other and not self == other

    def __gt__(self,other):
        """
        Return True if the set is a strict superset of 'other'

        TODO: verify that this is sufficiently efficient
             (vs. an explicit implimentation).
        """
        return self >= other and not self == other


class _FiniteSetMixin(object):
    __slots__ = ()

    def __reversed__(self):
        return reversed(self.__iter__())

    def is_finite(self):
        """Returns True if this is a finite discrete (iterable) Set"""
        return True

    def data(self):
        return tuple(self)

    def sorted(self):
        return sorted_robust(self.data())

    def ordered(self):
        return self.sorted()

    def bounds(self):
        try:
            lb = min(self)
        except:
            lb = None
        try:
            ub = max(self)
        except:
            ub = None
        return lb,ub

    def ranges(self):
        # This is way inefficient, but should always work: the ranges in a
        # Finite set is the list of scalars
        for i in self:
            if i.__class__ in native_numeric_types:
                yield _ClosedNumericRange(i,i,0)
            elif i.__class__ in native_types:
                yield _NonNumericRange(i)
            else:
                try:
                    as_numeric(i)
                    yield _ClosedNumericRange(i,i,0)
                except:
                    yield _NonNumericRange(i)

class _FiniteSetData(_FiniteSetMixin, _SetData):
    """A general unordered iterable Set"""
    __slots__ = ('_values', '_domain', '_validate', '_dimen')

    def __init__(self, component, domain=None):
        super(_FiniteSetData, self).__init__(component)
        self._values = set()
        if domain is None:
            self._domain = Any
        else:
            self._domain = domain
        self._dimen = _UnknownSetDimen
        self._validate = None

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_FiniteSetData, self).__getstate__()
        for i in _FiniteSetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because None of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __contains__(self, value):
        """
        Return True if the set contains a given value.
        """
        return value in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return len(self._values)

    def data(self):
        return tuple(self._values)

    def _verify(self, value):
        if value not in self._domain:
            raise ValueError("Cannot add value %s to set %s.\n"
                             "\tThe value is not in the Set's domain"
                             % (value, self.name,))
        if type(value) is tuple:
            value = flatten_tuple(value)
        # We wrap this check in a try-except because some values (like lists)
        #  are not hashable and can raise exceptions.
        try:
            if value in self._values:
                logger.warning(
                    "Element %s already exists in set %s; no action taken"
                    % (value, self.name))
                return False
        except:
            exc = exc_info()
            raise TypeError("Unable to insert '%s' into set %s:\n\t%s: %s"
                            % (value, self.name, exc[0].__name__, exc[1]))
        if self._validate is not None:
            flag = False
            try:
                flag = apply_indexed_rule(
                    self, self._validate, self.parent(), value)
            except:
                logger.error(
                    "Exception raised while validating element '%s' for Set %s"
                    % (value, self.name))
                raise
            if not flag:
                raise ValueError(
                    "The value=%s violates the validation rule of set=%s"
                    % (value, self.name))
        # If the Set has a fixed dimension, checck that this element is
        # compatible.
        if self._dimen is not None:
            if type(value) is tuple:
                _d = len(value)
            else:
                _d = 1
            if self._dimen is _UnknownSetDimen:
                # The first thing added to a Set with unknown dimension sets
                # its dimension
                self._dimen = _d
            elif _d != self._dimen:
                raise ValueError(
                    "The value=%s has dimension %s and is not valid for "
                    "Set %s which has dimen=%s"
                    % (value, _d, self.name, self._dimen))
        return True

    def add(self, value):
        if self._verify(value):
            self._values.add(value)

    def remove(self, val):
        self._values.remove(val)

    def discard(self, val):
        self._values.discard(val)

    def clear(self):
        self._values.clear()

    def set_value(self, val):
        self.clear()
        for x in val:
            self.add(x)

class _OrderedSetMixin(object):
    __slots__ = ()

    def is_ordered(self):
        """Returns True if this is an ordered finite discrete (iterable) Set"""
        return True

    def ordered(self):
        return self.data()

    def sorted(self):
        return sorted_robust(self.data())

    def first(self):
        return self[1]

    def last(self):
        return self[len(self)]

    def next(self, item, step=1):
        """
        Return the next item in the set.

        The default behavior is to return the very next element. The `step`
        option can specify how many steps are taken to get the next element.

        If the search item is not in the Set, or the next element is beyond
        the end of the set, then an IndexError is raised.
        """
        position = self.ord(item)
        return self[position+step]

    def nextw(self, item, step=1):
        """
        Return the next item in the set with wrapping if necessary.

        The default behavior is to return the very next element. The `step`
        option can specify how many steps are taken to get the next element.
        If the next element is past the end of the Set, the search wraps back
        to the beginning of the Set.

        If the search item is not in the Set an IndexError is raised.
        """
        position = self.ord(item)
        return self[(position+step-1) % len(self) + 1]

    def prev(self, item, step=1):
        """Return the previous item in the set.

        The default behavior is to return the immediately previous
        element. The `step` option can specify how many steps are taken
        to get the previous element.

        If the search item is not in the Set, or the previous element is
        before the beginning of the set, then an IndexError is raised.
        """
        return self.next(item, -step)

    def prevw(self, item, step=1):
        """Return the previous item in the set with wrapping if necessary.

        The default behavior is to return the immediately
        previouselement. The `step` option can specify how many steps
        are taken to get the previous element. If the previous element
        is past the end of the Set, the search wraps back to the end of
        the Set.

        If the search item is not in the Set an IndexError is raised.
        """
        return self.nextw(item, -step)

    def _to_0_based_index(self, item):
        if item >= 1:
            if item > len(self):
                raise IndexError("Cannot index a Set past the last element")
            return item - 1
        elif item < 0:
            item += len(self)
            if item < 0:
                raise IndexError("Cannot index a Set before the first element")
            return item
        else:
            raise IndexError(
                "Valid index values for sets are 1 .. len(set) or "
                "-1 .. -len(set)")


class _OrderedSetData(_OrderedSetMixin, _FiniteSetData):
    """
    This class defines the base class for an ordered set of concrete data.

    In older Pyomo terms, this defines a "concrete" ordered set - that is,
    a set that "owns" the list of set members.  While this class actually
    implements a set ordered by insertion order, we make the "official"
    _IndertionOrderSetData an empty derivative class, so that

         issubclass(_SortedSetData, _InsertionOrderSetData) == False

    Constructor Arguments:
        component   The Set object that owns this data.

    Public Class Attributes:
    """

    __slots__ = ('_ordered_values',)

    def __init__(self, component):
        super(_OrderedSetData, self).__init__(component)
        self._values = {}
        self._ordered_values = []

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_OrderedSetData, self).__getstate__()
        for i in _OrderedSetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because None of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __iter__(self):
        """
        Return an iterator for the set.
        """
        return iter(self._ordered_values)

    def data(self):
        return tuple(self._ordered_values)

    def add(self, value):
        if self._verify(value):
            self._values[value] = len(self._values)
            self._ordered_values.append(value)

    def remove(self, val):
        idx = self._values.pop(val)
        self._ordered_values.pop(idx)
        for i in xrange(idx, len(self._ordered_values)):
            self._values[self._ordered_values[i]] -= 1

    def discard(self, val):
        try:
            self.remove(val)
        except KeyError:
            pass

    def clear(self):
        self._values.clear()
        self._ordered_values.clear()

    def __getitem__(self, item):
        """
        Return the specified member of the set.

        The public Set API is 1-based, even though the
        internal _lookup and _values are (pythonically) 0-based.
        """
        return self._ordered_values[self._to_0_based_index(item)]

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        try:
            return self._values[item] + 1
        except KeyError:
            raise IndexError(
                "Cannot identify position of %s in Set %s: item not in Set"
                % (item, self.name))

    def sorted(self):
        return sorted_robust(self.data())


class _InsertionOrderSetData(_OrderedSetData):
    """
    This class defines the data for a ordered set where the items are ordered
    in insertion order (similar to Python's OrderedSet.

    Constructor Arguments:
        component   The Set object that owns this data.

    Public Class Attributes:
    """
    __slots__ = ()


class _SortedSetMixin(object):
    ""
    __slots__ = ()


class _SortedSetData(_OrderedSetData, _SortedSetMixin):
    """
    This class defines the data for a sorted set.

    Constructor Arguments:
        component   The Set object that owns this data.

    Public Class Attributes:
    """

    __slots__ = ('_is_sorted',)

    def __init__(self, component):
        super(_SortedSetData, self).__init__(component)
        # An empty set is sorted...
        self._is_sorted = True

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_SortedSetData, self).__getstate__()
        for i in _SortedSetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because None of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __iter__(self):
        """
        Return an iterator for the set.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).__iter__()

    def data(self):
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).data()

    def add(self, value):
        # Note that the sorted status has no bearing on insertion,
        # so there is no reason to check if the data is correctly sorted
        if self._verify(value):
            self._values[value] = len(self._values)
            self._ordered_values.append(value)
            self._is_sorted = False

    # Note: removing data does not affect the sorted flag
    #def remove(self, val):

    def clear(self):
        super(_SortedSetData, self).clear()
        self._is_sorted = True

    def __getitem__(self, item):
        """
        Return the specified member of the set.

        The public Set API is 1-based, even though the
        internal _lookup and _values are (pythonically) 0-based.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).__getitem__(item)

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).ord(item)

    def sorted(self):
        return self.data()

    def _sort(self):
        self._ordered_values = sorted_robust(self._ordered_values)
        self._values = dict(
            (j, i) for i, j in enumerate(self._ordered_values) )
        self._is_sorted = True


############################################################################

class Set(IndexedComponent):
    """
    A set object that is used to index other Pyomo objects.

    This class has a similar look-and-feel as a Python set class.
    However, the set operations defined in this class return another
    abstract Set object. This class contains a concrete set, which
    can be initialized by the load() method.

    Constructor Arguments:
        name            The name of the set
        doc             A text string describing this component
        within          A set that defines the type of values that can
                            be contained in this set
        domain          A set that defines the type of values that can
                            be contained in this set
        initialize      A dictionary or rule for setting up this set
                            with existing model data
        validate        A rule for validating membership in this set. This has
                            the functional form:
                                f: data -> bool
                            and returns true if the data belongs in the set
        dimen           Specify the set's arity, or None if no arity is enforced
        virtual         If true, then this is a virtual set that does not
                            store data using the class dictionary
        bounds          A 2-tuple that specifies the range of possible set values.
        ordered         Specifies whether the set is ordered. Possible values are:
                            False           Unordered
                            True            Ordered by insertion order
                            InsertionOrder  Ordered by insertion order
                            SortedOrder     Ordered by sort order
                            <function>      Ordered with this comparison function
        filter          A function that is used to filter set entries.

    Public class attributes:
        concrete        If True, then this set contains elements.(TODO)
        dimen           The dimension of the data in this set.
        doc             A text string describing this component
        domain          A set that defines the type of values that can
                            be contained in this set
        filter          A function that is used to filter set entries.
        initialize      A dictionary or rule for setting up this set
                            with existing model data
        ordered         Specifies whether the set is ordered.
        validate        A rule for validating membership in this set.
        virtual         If True, then this set does not store data using the class
                             dictionary
    """

    class InsertionOrder(object): pass
    class SortedOrder(object): pass
    _ValidOrderedAuguments = {None, False, InsertionOrder, SortedOrder}

    def __new__(cls, *args, **kwds):
        if cls != Set:
            return super(Set, cls).__new__(cls)

        ordered = kwds.pop('ordered', False)
        if ordered is True:
            ordered = Set.InsertionOrder
        if ordered not in Set._ValidOrderedAuguments:
            raise TypeError(
                "Set 'ordered' argument is not valid (must be one of {%s})" % (
                    ', '.join(sorted(
                        'Set.'+x.__name__ if isinstance(x,type) else str(x)
                        for x in Set._ValidOrderedAuguments))))
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            if ordered is Set.InsertionOrder:
                return OrderedSimpleSet.__new__(OrderedSimpleSet)
            elif ordered is Set.SortedOrder:
                return SortedSimpleSet.__new__(SortedSimpleSet)
            else:
                return FiniteSimpleSet.__new__(FiniteSimpleSet)
        else:
            newObj = IndexedSet.__new__(IndexedSet)
            if ordered is Set.InsertionOrder:
                newObj._ComponentDataClass = _OrderedSetData
            elif ordered is Set.SortedOrder:
                newObj._ComponentDataClass = _SortedSetData
            else:
                newObj._ComponentDataClass = _FiniteSetData
            return newObj

    def __init__(self, *args, **kwds):
        kwds.setdefault('ctype', Set)
        # Drop the ordered flag: this was processed by __new__
        kwds.pop('ordered',None)
        IndexedComponent.__init__(self, *args, **kwds)

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing Set, name=%s, from data=%r"
                             % (self.name, data))
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True
        timer.report()

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Dim", self.dim()),
             ("Size", len(self)),
             ("Bounds", self.bounds())],
            iteritems(self),
            ("Finite","Ordered","Sorted","Domain","Members",),
            lambda k, v: [
                v.is_finite(),#isinstance(v, _FiniteSetMixin),
                v.is_ordered(), #isinstance(v, _OrderedSetMixin),
                isinstance(v, _SortedSetMixin),
                v._domain,
                v.ordered(),
            ])


class FiniteSimpleSet(_FiniteSetData, Set):
    def __init__(self, **kwds):
        _FiniteSetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

class OrderedSimpleSet(_OrderedSetData, Set):
    def __init__(self, **kwds):
        _OrderedSetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

class SortedSimpleSet(_SortedSetData, Set):
    def __init__(self, **kwds):
        _SortedSetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

class IndexedSet(Set):
    pass

############################################################################

class _InfiniteRangeSetData(_SetData):
    """Data class for a infinite set.

    This Set implements an interface to an *infinite set* defined by one
    or more _ClosedNumericRange objeccts.  As there are an infinite
    number of members, Infinite Range Sets are not iterable.

    """

    __slots__ = ('_ranges',)

    def __init__(self, component, ranges=None):
        super(_InfiniteRangeSetData, self).__init__(component)
        ranges = tuple(ranges)
        for r in ranges:
            if not isinstance(r, _ClosedNumericRange):
                raise TypeError(
                    "_InfiniteRangeSetData range argument must be an "
                    "interable of _ClosedNumericRange objects")
        self._ranges = ranges

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_InfiniteRangeSetData, self).__getstate__()
        for i in _InfiniteRangeSetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because None of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __contains__(self, val):
        return any(val in r for r in self._ranges)

    def ranges(self):
        return iter(self._ranges)

    def bounds(self):
        _bnds = list((r.start, r.end) if r.step >= 0 else (r.end, r.start)
                     for r in self._ranges)
        lb, ub = _bnds.pop()
        for _lb, _ub in _bnds:
            if lb is not None:
                if _lb is None:
                    lb = None
                else:
                    lb = min(lb, _lb)
            if ub is not None:
                if _ub is None:
                    ub = None
                else:
                    ub = max(ub, _ub)
        return lb, ub


class _FiniteRangeSetData( _InfiniteRangeSetData, _SortedSetMixin,
                           _OrderedSetMixin, _FiniteSetMixin):
    def __iter__(self):
        def _range_gen(r):
            start, end = (r.start, r.end) if r.step > 0 else (r.end, r.start)
            step = abs(r.step)
            n = start
            i = 0
            while n <= end:
                yield n
                i += 1
                n = start + i*step

        if len(self._ranges) == 1:
            for x in _range_gen(self._ranges[0]):
                yield x
            return
        iters = []
        for r in self._ranges:
            try:
                i = _range_gen(r)
                iters.append([next(i), i])
            except StopIteration:
                pass
        if not iters:
            return
        iters.sort()
        n = None
        while iters:
            if n != iters[0][0]:
                n = iters[0][0]
                yield n
            try:
                iters[0][0] = next(iters[0][1])
                iters.sort()
            except StopIteration:
                iters.pop(0)

    def __len__(self):
        if len(self._ranges) == 1:
            r = self._ranges[0]
            return (r.end - r.start) // r.step + 1
        else:
            return sum(1 for _ in self)

    def __getitem__(self, item):
        assert int(item) == item
        if len(self._ranges) == 1:
            r = self._ranges[0]
            ans = r.start + (item-1)*r.step
            if ans <= r.end:
                return ans
        else:
            try:
                return self.data()[item]
            except IndexError:
                pass
        raise IndexError("sorted set index out of range")

    def is_finite(self):
        return True

    def ord(self, item):
        if len(self._ranges) == 1:
            i = float(item - r.start) / r.step
            if i - int(i+0.5) < EPS:
                return int(i+0.5) + 1
        else:
            try:
                return self.data.index(item) + 1
            except ValueError:
                pass
        raise IndexError(
            "Cannot identify position of %s in Set %s: item not in Set"
            % (item, self.name))

    def sorted(self):
        return self.data()



class RangeSet(Component):
    """
    A set object that represents a set of numeric values

    """

    def __new__(cls, *args, **kwds):
        if cls != RangeSet:
            return super(RangeSet, cls).__new__(cls)

        if 'ranges' in kwds:
            if any(not r.is_finite() for r in kwds['ranges']):
                return InfiniteSimpleRangeSet.__new__(InfiniteSimpleRangeSet)
        if None in args or (len(args) > 2 and args[2] == 0):
            return InfiniteSimpleRangeSet.__new__(InfiniteSimpleRangeSet)
        else:
            return FiniteSimpleRangeSet.__new__(FiniteSimpleRangeSet)


    def __init__(self, *args, **kwds):
        kwds.setdefault('ctype', RangeSet)
        ranges = kwds.pop('ranges', [])
        _init_class = kwds.pop('_init_class')
        Component.__init__(self, **kwds)

        if type(ranges) is tuple:
            ranges = list(ranges)
        elif type(ranges) is not list:
            ranges = [ranges]
        if len(args) == 1:
            ranges.append(_ClosedNumericRange(1,args[0],1))
        elif len(args) == 2:
            ranges.append(_ClosedNumericRange(args[0],args[1],1))
        elif len(args) == 3:
            ranges.append(_ClosedNumericRange(*args))
        elif args:
            raise ValueError("RangeSet expects 3 or fewer positional "
                             "arguments (received %s)" % (len(args),))
        _init_class.__init__(self, component=self, ranges=ranges)

    def construct(self, data=None):
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing RangeSet, name=%s, from data=%r"
                             % (self.name, data))
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed=True
        timer.report()

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Dim", 0),
             ("Dimen", 1),
             ("Size", len(self) if self.is_finite() else 'Inf'),
             ("Bounds", self.bounds())],
            iteritems( {None: self} ),
            ("Finite","Members",),
            lambda k, v: [
                v.is_finite(),#isinstance(v, _FiniteSetMixin),
                ', '.join(str(r) for r in self.ranges()),
            ])


class InfiniteSimpleRangeSet(_InfiniteRangeSetData, RangeSet):
    def __init__(self, *args, **kwds):
        kwds.setdefault('_init_class', _InfiniteRangeSetData)
        RangeSet.__init__(self, *args, **kwds)

class FiniteSimpleRangeSet(_FiniteRangeSetData, RangeSet):
    def __init__(self, *args, **kwds):
        kwds.setdefault('_init_class', _FiniteRangeSetData)
        RangeSet.__init__(self, *args, **kwds)


############################################################################
# Set Operators
############################################################################

class _SetOperator(_SetData):
    __slots__ = ('_sets','_implicit_subsets')

    def __init__(self, set0, set1):
        self._sets, self._implicit_subsets = self._processArgs(set0, set1)

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_SetOperator, self).__getstate__()
        for i in _SetOperator.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because None of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    @staticmethod
    def _processArgs(self, *sets):
        implicit = []
        ans = []
        for s in sets:
            if isinstance(s, _SetDataBase):
                ans.append(s)
                if s.parent_block() is None:
                    implicit.append(s)
            else:
                ans.append(Set(initialize=s))
                implicit.append(ans[-1])
        return tuple(ans), tuple(implicit)

############################################################################

class _SetUnion(_SetOperator):
    __slots__ = tuple()

    def __new__(cls, set0, set1):
        if cls != _SetUnion:
            return super(_SetUnion, cls).__new__(cls)

        (set0, set1), implicit = _SetOperator._processArgs(set0, set1)
        if set0.is_ordered() and set1.is_ordered():
            cls = _SetUnion_OrderedSet
        elif set0.is_finite() and set1.is_finite():
            cls = _SetUnion_FiniteSet
        else:
            cls = _SetUnion_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        return itertools.chain(self._sets[i].ranges() for i in (0,1))


class _SetUnion_InfiniteSet(_SetUnion):
    __slots__ = tuple()

    def __contains__(self, val):
        return any(val in s for s in self._sets)


class _SetUnion_FiniteSet(_SetUnion_InfiniteSet, _FiniteSetMixin):
    __slots__ = tuple()

    def __iter__(self):
        set0 = self._sets[0]
        return itertools.chain(
            set0,
            (_ for _ in self._sets[1] if _ not in set0)
        )

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        # There is no easy way to tell how many duplicates there are in
        # the second set.  Our only choice is to count them.  We will
        # try and be a little efficient by using len() for the first
        # set, though.
        set0, set1 = self._sets
        return len(set0) + sum(1 for s in set1 if s not in set0)


class _SetUnion_OrderedSet(_SetUnion_FiniteSet, _OrderedSetMixin):
    __slots__ = tuple()

    def __getitem__(self, item):
        idx = self._to_0_based_index(item)
        set0_len = len(self._sets[0])
        if idx < set0_len:
            return self._sets[0][idx]
        else:
            idx -= set0_len
            set1_iter = iter(self._sets[1])
            while idx:
                next(set1_iter)
                idx -= 1
            return next(set1_iter)

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        if item in self._sets[0]:
            return self._sets[0].ord(item)
        if item not in self._sets[1]:
            raise IndexError(
                "Cannot identify position of %s in Set %s: item not in Set"
                % (item, self.name))
        idx = len(self._sets[0]) + 1
        _iter = iter(self._sets[1])
        while next(_iter) != item:
            idx += 1
        return idx + 1


############################################################################

class _SetIntersection(_SetData):
    __slots__ = tuple()

    def __new__(cls, set0, set1):
        if cls != _SetUnion:
            return super(_SetUnion, cls).__new__(cls)

        (set0, set1), implicit = _SetOperator._processArgs(set0, set1)
        if set0.is_ordered() or set1.is_ordered():
            cls = _SetIntersection_OrderedSet
        elif set0.is_finite() or set1.is_finite():
            cls = _SetIntersection_FiniteSet
        else:
            cls = _SetIntersection_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        for a in self._sets[0].ranges():
            for r in a.range_intersection(self._sets[1].ranges()):
                yield r


class _SetIntersection_InfiniteSet(_SetIntersection):
    __slots__ = tuple()

    def __contains__(self, val):
        return all(val in s for s in self._sets)


class _SetIntersection_FiniteSet(_SetIntersection_InfiniteSet, _FiniteSetMixin):
    __slots__ = tuple()

    def __iter__(self):
        set0, set1 = self._sets
        if set1.is_ordered() and not set0.is_ordered():
            set0, set1 = set1, set0
        elif set1.is_finite() and not set0.is_finite():
            set0, set1 = set1, set0
        return (s for s in set0 if s in set1)

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        # There is no easy way to tell how many duplicates there are in
        # the second set.  Our only choice is to count them.  We will
        # try and be a little efficient by using len() for the first
        # set, though.
        return sum(1 for _ in self)


class _SetIntersection_OrderedSet(_SetIntersection_FiniteSet, _OrderedSetMixin):
    __slots__ = tuple()

    def __getitem__(self, item):
        idx = self._to_0_based_index(item)
        _iter = iter(self)
        while idx:
            next(_iter)
            idx -= 1
        return next(_iter)

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        if item not in self._sets[0] or item not in self._sets[1]:
            raise IndexError(
                "Cannot identify position of %s in Set %s: item not in Set"
                % (item, self.name))
        idx = 0
        _iter = iter(self)
        while next(_iter) != item:
            idx += 1
        return idx + 1

############################################################################

class _SetDifference(_SetOperator):
    __slots__ = tuple()

    def __new__(cls, set0, set1):
        if cls != _SetDifference:
            return super(_SetDifference, cls).__new__(cls)

        (set0, set1), implicit = _SetOperator._processArgs(set0, set1)
        if set0.is_ordered():
            cls = _SetDifference_OrderedSet
        elif set0.is_finite():
            cls = _SetDifference_FiniteSet
        else:
            cls = _SetDifference_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        for a in self._sets[0].ranges():
            for r in a.range_difference(self._sets[1].ranges()):
                yield r


class _SetDifference_InfiniteSet(_SetDifference):
    __slots__ = tuple()

    def __contains__(self, val):
        return val in self._sets[0] and not val in self._sets[1]


class _SetDifference_FiniteSet(_SetDifference_InfiniteSet, _FiniteSetMixin):
    __slots__ = tuple()

    def __iter__(self):
        set0, set1 = self._sets
        return (_ for _ in set0 if _ not in set1)

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return sum(1 for _ in self)


class _SetDifference_OrderedSet(_SetDifference_FiniteSet, _OrderedSetMixin):
    __slots__ = tuple()

    def __getitem__(self, item):
        idx = self._to_0_based_index(item)
        _iter = iter(self)
        while idx:
            next(_iter)
            idx -= 1
        return next(_iter)

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        if item not in self:
            raise IndexError(
                "Cannot identify position of %s in Set %s: item not in Set"
                % (item, self.name))
        idx = 0
        _iter = iter(self)
        while next(_iter) != item:
            idx += 1
        return idx + 1


############################################################################

class _SetSymmetricDifference(_SetOperator):
    __slots__ = tuple()

    def __new__(cls, set0, set1):
        if cls != _SetSymmetricDifference:
            return super(_SetSymmetricDifference, cls).__new__(cls)

        (set0, set1), implicit = _SetOperator._processArgs(set0, set1)
        if set0.is_ordered() and set1.is_ordered():
            cls = _SetSymmetricDifference_OrderedSet
        elif set0.is_finite() and set1.is_finite():
            cls = _SetSymmetricDifference_FiniteSet
        else:
            cls = _SetSymmetricDifference_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        # Note: the following loop implements for (a,b), (b,a)
        assert len(self._sets) == 2
        for set_a, set_b in (self._sets, reversed(self._sets)):
            for a in set_a:
                for r in set_a.range_difference(set_b.ranges()):
                    yield r

class _SetSymmetricDifference_InfiniteSet(_SetSymmetricDifference):
    __slots__ = tuple()

    def __contains__(self, val):
        return (val in self._sets[0]) ^ (val in self._sets[1])


class _SetSymmetricDifference_FiniteSet(_SetSymmetricDifference_InfiniteSet,
                                        _FiniteSetMixin):
    __slots__ = tuple()

    def __iter__(self):
        set0, set1 = self._sets
        return itertools.chain(
            (_ for _ in set0 if _ not in set1),
            (_ for _ in set1 if _ not in set0),
        )

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return sum(1 for _ in self)


class _SetSymmetricDifference_OrderedSet(_SetSymmetricDifference_FiniteSet,
                                         _OrderedSetMixin):
    __slots__ = tuple()

    def __getitem__(self, item):
        idx = self._to_0_based_index(item)
        _iter = iter(self)
        while idx:
            next(_iter)
            idx -= 1
        return next(_iter)

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        if item not in self:
            raise IndexError(
                "Cannot identify position of %s in Set %s: item not in Set"
                % (item, self.name))
        idx = 0
        _iter = iter(self)
        while next(_iter) != item:
            idx += 1
        return idx + 1


############################################################################

class _SetProduct(_SetOperator):
    __slots__ = tuple()

    def __new__(cls, set0, set1):
        if cls != _SetProduct:
            return super(_SetProduct, cls).__new__(cls)

        (set0, set1), implicit = _SetOperator._processArgs(set0, set1)
        if set0.is_ordered() and set1.is_ordered():
            cls = _SetProduct_OrderedSet
        elif set0.is_finite() and set1.is_finite():
            cls = _SetProduct_FiniteSet
        else:
            cls = _SetProduct_InfiniteSet
        return cls.__new__(cls)

    def expand_crossproduct(self):
        for s in self._sets:
            if isinstance(s, _SetProduct):
                for ss in s.expand_crossproduct():
                    yield ss
            else:
                yield s

class _SetProduct_InfiniteSet(_SetProduct):
    __slots__ = tuple()

    def __contains__(self, val):
        set0, set1 = self._sets
        if len(val) == 2 and val[0] in set0 and val[1] in set1:
            return True
        val = flatten_tuple(val)
        if self._sets[0].dimen:
            return val[:set0.dimen] in set0 and val[set0.dimen:] in set1
        if self._sets[1].dimen:
            return val[:-set1.dimen] in set0 and val[-set1.dimen:] in set1
        # At this point, neither base set has a fixed dimention.  The
        # only thing we can do is test all possible split points.
        for i in xrange(len(val)):
            if val[:i] in set0 and val[i:] in set1:
                return True
        return False


class _SetProduct_FiniteSet(_SetProduct_InfiniteSet, _FiniteSetMixin):
    __slots__ = tuple()

    def __iter__(self):
        return itertools.product(self._sets[0], self._sets[1])

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return len(self._sets[0]) * len(self._sets[1])


class _SetProduct_OrderedSet(_SetProduct_FiniteSet, _OrderedSetMixin):
    __slots__ = tuple()

    def __getitem__(self, item):
        idx = self._to_0_based_index(item)
        I_len = len(self._sets[0])
        i = idx // I_len
        a = self._sets[0][i]
        if type(a) is not tuple:
            a = (a,)
        b = self._sets[1][idx - i*I_len]
        if type(b) is not tuple:
            b = (b,)
        return a + b

    def ord(self, val):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        set0, set1 = self._sets
        if len(val) == 2 and val[0] in set0 and val[1] in set1:
            return (set0.ord(val[0])-1) * len(set0) + set1.ord(val[1])

        val = flatten_tuple(val)
        _idx = None
        if set0.dimen \
           and val[:set0.dimen] in set0 and val[set0.dimen:] in set1:
            _idx = set0.dimen
        elif set1.dimen \
             and val[:-set1.dimen] in set1 and val[-set1.dimen:] in set1:
            _idx = -set1.dimen
        else:
            # At this point, neither base set has a fixed dimention.  The
            # only thing we can do is test all possible split points.
            for i in xrange(len(val)):
                if val[:i] in self._sets[0] and val[i:] in self._sets[1]:
                    _idx = i
                    break

        if _idx is None:
            raise IndexError(
                "Cannot identify position of %s in Set %s: item not in Set"
                % (val, self.name))
        return (set0.ord(val[:_idx])-1) * len(set0) + set1.ord(val[_idx:])

############################################################################

class _AnySet(_SetData, Set):
    def __init__(self, **kwds):
        _SetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

    def __contains__(self, val):
        return True

    def ranges(self):
        yield _AnyRange()

Any = _AnySet(name='Any', doc="A global Pyomo Set that admits any value")

Reals = RangeSet(
    name='Reals',
    doc='A global Pyomo Set that admits any real (floating point) value',
    ranges=(_ClosedNumericRange(None,None,0),))

NonNegativeReals = RangeSet(
    name='NonNegativeReals',
    doc='A global Pyomo Set admitting any real value in [0, +inf)',
    ranges=(_ClosedNumericRange(0,None,0),))

NonPositiveReals = RangeSet(
    name='NonPositiveReals',
    doc='A global Pyomo Set admitting any real value in (-inf, 0]',
    ranges=(_ClosedNumericRange(None,0,0),))

Integers = RangeSet(
    name='Integers',
    doc='A global Pyomo Set admitting any integer value',
    ranges=(_ClosedNumericRange(0,None,1), _ClosedNumericRange(0,None,-1)))

NonNegativeIntegers = RangeSet(
    name='NonNegativeIntegers',
    doc='A global Pyomo Set admitting any integer value in [0, +inf)',
    ranges=(_ClosedNumericRange(0,None,1),))

NonPositiveIntegers = RangeSet(
    name='NonPositiveIntegers',
    doc='A global Pyomo Set admitting any integer value in (-inf, 0]',
    ranges=(_ClosedNumericRange(0,None,-1),))

NegativeIntegers = RangeSet(
    name='NegativeIntegers',
    doc='A global Pyomo Set admitting any integer value in (-inf, -1]',
    ranges=(_ClosedNumericRange(-1,None,-1),))

PositiveIntegers = RangeSet(
    name='PositiveIntegers',
    doc='A global Pyomo Set admitting any integer value in [1, +inf)',
    ranges=(_ClosedNumericRange(1,None,1),))

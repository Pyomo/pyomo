#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import collections
import inspect
import itertools
import logging
import math
import six
import sys
import weakref

from six import iteritems, iterkeys
from six.moves import xrange

from pyutilib.misc.misc import flatten_tuple

from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
    native_types, native_numeric_types, as_numeric, value,
)
from pyomo.core.base.util import (
    disable_methods, InitializerBase, Initializer, ConstantInitializer,
    CountedCallInitializer, ItemInitializer, IndexedCallInitializer,
)
from pyomo.core.base.range import (
    NumericRange, NonNumericRange, AnyRange, RangeProduct,
    RangeDifferenceError,
)
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set, normalize_index,
)
from pyomo.core.base.misc import sorted_robust

logger = logging.getLogger('pyomo.core')

_prePython37 = sys.version_info[:2] < (3,7)

FLATTEN_CROSS_PRODUCT = True


def process_setarg(arg):
    if isinstance(arg, _SetDataBase):
        return arg
    elif isinstance(arg, IndexedComponent):
        raise TypeError("Cannot apply a Set operator to an "
                        "indexed %s component (%s)"
                        % (arg.type().__name__, arg.name,))
    elif isinstance(arg, Component):
        raise TypeError("Cannot apply a Set operator to a non-Set "
                        "%s component (%s)"
                        % (arg.__class__.__name__, arg.name,))
    elif isinstance(arg, ComponentData):
        raise TypeError("Cannot apply a Set operator to a non-Set "
                        "component data (%s)" % (arg.name,))

    # TODO: DEPRECATE this functionality? It has never been documented,
    # and I don't know of a use of it in the wild.
    try:
        # If the argument has a set_options attribute, then use
        # it to initialize a set
        args = getattr(arg,'set_options')
        args.setdefault('initialize', arg)
        args.setdefault('ordered', type(arg) not in Set._UnorderedInitializers)
        ans = Set(**args)
        ans.construct()
        return ans
    except AttributeError:
        pass

    # TBD: should lists/tuples be copied into Sets, or
    # should we preserve the reference using SetOf?
    # Historical behavior is to *copy* into a Set.
    #
    # ans.append(Set(initialize=arg,
    #               ordered=type(arg) in {tuple, list}))
    # ans.construct()
    #
    # But this causes problems, especially because Set()'s
    # constructor needs to know if the object is ordered
    # (Set defaults to ordered, and will toss a warning if
    # the underlying data is not ordered)).  While we could
    # add checks where we create the Set (like here and in
    # the __r*__ operators) and pass in a reasonable value
    # for ordered, it is starting to make more sense to use
    # SetOf (which has that logic).  Alternatively, we could
    # use SetOf to create the Set:
    #
    tmp = SetOf(arg)
    ans = Set(initialize=tmp, ordered=tmp.isordered())
    ans.construct()
    #
    # Or we can do the simple thing and just use SetOf:
    #
    # ans = SetOf(arg)
    return ans


@deprecated('The set_options decorator seems nonessential and is deprecated',
            version='TBD')
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

class UnknownSetDimen(object): pass

class SetInitializer(InitializerBase):
    __slots__ = ('_set','verified')

    def __init__(self, init, allow_generators=True):
        self.verified = False
        if init is None:
            self._set = None
        else:
            self._set = Initializer(
                init, allow_generators=allow_generators,
                treat_sequences_as_mappings=False)

    def intersect(self, other):
        if self._set is None:
            if type(other) is SetInitializer:
                self._set = other._set
            else:
                self._set = other
        elif type(other) is SetInitializer:
            if other._set is not None:
                self._set = SetIntersectInitializer(self._set, other._set)
        else:
            self._set = SetIntersectInitializer(self._set, other)

    def __call__(self, parent, idx):
        if self._set is None:
            return Any
        else:
            return self._set(parent, idx)

    def constant(self):
        return self._set is None or self._set.constant()

    def setdefault(self, val):
        if self._set is None:
            self._set = ConstantInitializer(val)

class SetIntersectInitializer(InitializerBase):
    __slots__ = ('_A','_B',)
    def __init__(self, setA, setB):
        self._A = setA
        self._B = setB

    def __call__(self, parent, idx):
        return SetIntersection(self._A(parent, idx), self._B(parent, idx))

    def constant(self):
        return self._A.constant() and self._B.constant()

class RangeSetInitializer(InitializerBase):
    __slots__ = ('_init', 'default_step',)
    def __init__(self, init, default_step=1):
        self._init = Initializer(init, treat_sequences_as_mappings=False)
        self.default_step = default_step

    def __call__(self, parent, idx):
        val = self._init(parent, idx)
        if not isinstance(val, collections.Sequence):
            val = (1, val, self.default_step)
        if len(val) < 3:
            val = tuple(val) + (self.default_step,)
        ans = RangeSet(*tuple(val))
        ans.construct()
        return ans

    def constant(self):
        return self._init.constant()

    def setdefault(self, val):
        # This is a real range set... there is no default to set
        pass

#
# DESIGN NOTES
#
# What do sets do?
#
# ALL:
#   __contains__
#
# Note: FINITE implies DISCRETE. Infinite discrete sets cannot be iterated
#
# FINITE: ALL +
#   __len__ (Note: Python len() requires __len__ to return non-negative int)
#   __iter__, __reversed__
#   add()
#   sorted(), ordered_data()
#
# ORDERED: FINITE +
#   __getitem__
#   next(), prev(), first(), last()
#   ord()
#
# When we do math, the least specific set dictates the API of the resulting set.
#
# Note that isfinite and isordered must be resolvable when the class
# is instantiated (*before* construction).  We will key off these fields
# when performing set operations to know what type of operation to
# create, and we will allow set operations in Abstract before
# construction.

#
# Set rewrite TODOs:
#
#   - Test index/ord for equivalence of 1 and (1,)
#
#   - Make sure that all classes implement the appropriate methods
#     (e.g., bounds)
#
#   - Sets created with Set.Skip should produce intelligible errors
#
#   - Resolve nonnumeric range operations on tuples of numeric ranges
#
#   - Ensure the range operators raise exeptions for unexpected
#     (non-range/non list arguments.
#


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
    __slots__ = ()

    def __contains__(self, value):
        raise DeveloperError("Derived set class (%s) failed to "
                             "implement __contains__" % (type(self).__name__,))

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return False

    def isfinite(self):
        """Returns True if this is a finite discrete (iterable) Set"""
        return False

    def isordered(self):
        """Returns True if this is an ordered finite discrete (iterable) Set"""
        return False

    def __eq__(self, other):
        if self is other:
            return True
        try:
            other_isfinite = other.isfinite()
        except:
            # we assume that everything that does not implement
            # isfinite() is a discrete set.
            other_isfinite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
            except:
                pass
        if self.isfinite():
            if not other_isfinite:
                return False
            if len(self) != len(other):
                return False
            for x in self:
                if x not in other:
                    return False
            return True
        elif other_isfinite:
            return False
        return self.issubset(other) and other.issubset(self)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        raise DeveloperError("Derived set class (%s) failed to "
                             "implement __str__" % (type(self).__name__,))

    @property
    def dimen(self):
        raise DeveloperError("Derived set class (%s) failed to "
                             "implement dimen" % (type(self).__name__,))

    def ranges(self):
        raise DeveloperError("Derived set class (%s) failed to "
                             "implement ranges" % (type(self).__name__,))

    def bounds(self):
        try:
            _bnds = list((r.start, r.end) if r.step >= 0 else (r.end, r.start)
                         for r in self.ranges())
        except AttributeError:
            return None, None
        if not _bnds:
            return None, None

        lb, ub = _bnds.pop()
        for _lb, _ub in _bnds:
            if lb is not None:
                if _lb is None:
                    lb = None
                    if ub is None:
                        break
                else:
                    lb = min(lb, _lb)
            if ub is not None:
                if _ub is None:
                    ub = None
                    if lb is None:
                        break
                else:
                    ub = max(ub, _ub)
        return lb, ub

    def get_interval(self):
        """Return the interval for this Set as (start, end, step)

        Returns the effective interval for this Set as a (start, end,
        step) tuple.  Start and End are the same as returned by
        `bounds()`.  Step is 0 for continuous ranges, a positive value
        for regular discrete sets (e.g., 1 for Integers), or `None` for
        Sets that do not have a regular interval (e.g., semicontinuous
        sets, mixed type sets, sets with dimen != 1, etc).

        """
        if self.dimen != 1:
            return self.bounds() + (None,)
        if self.isdiscrete():
            return self._get_discrete_interval()
        else:
            return self._get_continuous_interval()

    def _get_discrete_interval(self):
        #
        # Note: I'd like to use set() for ranges, since we will be
        # randomly removing elelments from the list; however, since we
        # do it by enumerating over ranges, using set() would make this
        # routine nondeterministic.  Not a hoge issue for the result,
        # but problemmatic for code coverage.
        ranges = list(self.ranges())
        try:
            step = min(abs(r.step) for r in ranges if r.step != 0)
        except ValueError:
            # If all the ranges are single points, we will just
            # brute-force it: sort the values and ensure that the step
            # is consistent.  Note that we know at this point ranges
            # only contains NumericRange objects (or at least that they
            # all have a `step` attribute).
            vals = sorted(self)
            if len(vals) < 2:
                return (vals[0], vals[0], 0)
            step = vals[1]-vals[0]
            for i in xrange(2, len(vals)):
                if step != vals[i] - vals[i-1]:
                    return self.bounds() + (None,)
            return (vals[0], vals[-1], step)
        except AttributeError:
            # Catching Any, NonNumericRange, RangeProduct, etc...
            return self.bounds() + (None,)

        nRanges = len(ranges)
        r = ranges.pop()
        _rlen = len(ranges)
        ref = r.start
        if r.step >= 0:
            start, end = r.start, r.end
        else:
            end, start = r.start, r.end
        if r.step % step:
            return self.bounds() + (None,)
        # Catch misaligned ranges
        for r in ranges:
            if ( r.start - ref ) % step:
                return self.bounds() + (None,)
            if r.step % step:
                return self.bounds() + (None,)

        # This loop terminates when we have a complete pass that doesn't
        # remove any ranges from the ranges list.
        while nRanges > _rlen:
            nRanges = _rlen
            for i,r in enumerate(ranges):
                if r.step > 0:
                    rstart, rend = r.start, r.end
                else:
                    rend, rstart = r.start, r.end
                if not r.step or abs(r.step) == step:
                    if ( start is None or rend is None or
                         start <= rend+step ) and (
                             end is None or rstart is None or
                             rstart <= end+step ):
                        ranges[i] = None
                        if rstart is None:
                            start = None
                        elif start is not None and start > rstart:
                            start = rstart
                        if rend is None:
                            end = None
                        elif end is not None and end < rend:
                            end = rend
                else:
                    # The range has a step bigger than the base
                    # interval we are building.  For us to absorb
                    # it, it has to be contained within the current
                    # interval +/- step.
                    if (start is None or ( rstart is not None and
                                           start <= rstart + step ))\
                        and (end is None or ( rend is not None and
                                              end >= rend - step )):
                        ranges[i] = None
                        if start is not None and start > rstart:
                            start = rstart
                        if end is not None and end < rend:
                            end = rend

            ranges = list(_ for _ in ranges if _ is not None)
            _rlen = len(ranges)
        if ranges:
            return self.bounds() + (None,)
        return (start, end, step)


    def _get_continuous_interval(self):
        # Note: this method assumes that at least one range is continuous.
        #
        # Note: I'd like to use set() for ranges, since we will be
        # randomly removing elelments from the list; however, since we
        # do it by enumerating over ranges, using set() would make this
        # routine nondeterministic.  Not a hoge issue for the result,
        # but problemmatic for code coverage.
        #
        # Note: We do not need to trap non-NumericRange objects:
        # RangeProduct and AnyRange will be caught by the dimen test in
        # get_interval(), and NonNumericRange objects are cleanly
        # handled as if they were regular discrete ranges.
        ranges = []
        discrete = []

        # Pull out the discrete intervals (for checking later), and
        # copy the continuous ranges (so we can later make them
        # closed, if applicable)
        for r in self.ranges():
            if r.isdiscrete():
                discrete.append(r)
            else:
                ranges.append(
                    NumericRange(r.start, r.end, r.step, r.closed))

        # There is a particular edge case where we could get 2 disjoint
        # continuous ranges that are joined by a discrete range...  When
        # we encounter an open range, check to see if the endpoint is
        # in the discrete set, and if so, convert it to a closed range.
        for r in ranges:
            if not r.closed[0]:
                for d in discrete:
                    if r.start in d:
                        r.closed = (True, r.closed[1])
                        break
            if not r.closed[1]:
                for d in discrete:
                    if r.end in d:
                        r.closed = (r.closed[0], True)
                        break

        nRanges = len(ranges)
        r = ranges.pop()
        interval = NumericRange(r.start, r.end, r.step, r.closed)
        _rlen = len(ranges)

        # This loop terminates when we have a complete pass that doesn't
        # remove any ranges from the ranges list.
        while _rlen and nRanges > _rlen:
            nRanges = _rlen
            for i, r in enumerate(ranges):
                if interval.isdisjoint(r):
                    continue
                # r and interval overlap: merge r into interval
                ranges[i] = None
                if r.start is None:
                    interval.start = None
                    interval.closed = (True, interval.closed[1])
                elif interval.start is not None \
                     and r.start < interval.start:
                    interval.start = r.start
                    interval.closed = (r.closed[0], interval.closed[1])

                if r.end is None:
                    interval.end = None
                    interval.closed = (interval.closed[0], True)
                elif interval.end is not None and r.end > interval.end:
                    interval.end = r.end
                    interval.closed = (interval.closed[0], r.closed[1])

            ranges = list(_ for _ in ranges if _ is not None)
            _rlen = len(ranges)
        if ranges:
            # The continuous ranges are disjoint
            return self.bounds() + (None,)
        for r in discrete:
            if not r.issubset(interval):
                # The discrete range extends outside the continuous
                # interval
                return self.bounds() + (None,)
        return (interval.start, interval.end, interval.step)

    @property
    @deprecated("The 'virtual' flag is no longer supported", version='TBD')
    def virtual(self):
        return False

    @property
    @deprecated("The 'concrete' flag is no longer supported.  "
                "Use isdiscrete() or isfinite()", version='TBD')
    def concrete(self):
        return self.isfinite()

    def isdisjoint(self, other):
        """Test if this Set is disjoint from `other`

        Parameters
        ----------
            other : ``Set`` or ``iterable``
                The Set or iterable object to compare this Set against

        Returns
        -------
        bool : True if this set is disjoint from `other`
        """
        try:
            other_isfinite = other.isfinite()
        except:
            # we assume that everything that does not implement
            # isfinite() is a discrete set.
            other_isfinite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
            except:
                pass
        if self.isfinite():
            for x in self:
                if x in other:
                    return False
            return True
        elif other_isfinite:
            for x in other:
                if x in self:
                    return False
            return True
        else:
            all(r.isdisjoint(s) for r in self.ranges() for s in other.ranges())

    def issubset(self, other):
        """Test if this Set is a subset of `other`

        Parameters
        ----------
            other : ``Set`` or ``iterable``
                The Set or iterable object to compare this Set against

        Returns
        -------
        bool : True if this set is a subset of `other`
        """
        try:
            other_isfinite = other.isfinite()
        except:
            # we assume that everything that does not implement
            # isfinite() is a discrete set.
            other_isfinite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
            except:
                pass
        if self.isfinite():
            for x in self:
                if x not in other:
                    return False
            return True
        elif other_isfinite:
            return False
        else:
            for r in self.ranges():
                try:
                    if r.range_difference(other.ranges()):
                        return False
                except RangeDifferenceError:
                    # This only occurs when subtracting an infinite
                    # discrete set from an infinite continuous set, so r
                    # (and hence self) cannot be a subset of other
                    return False
            return True

    def issuperset(self, other):
        try:
            other_isfinite = other.isfinite()
        except:
            # we assume that everything that does not implement
            # isfinite() is a discrete set.
            other_isfinite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
            except:
                pass
        if other_isfinite:
            for x in other:
                # Other may contain elements that are not representable
                # in self.  Trap that error (a TypeError due to hashing)
                # and return False
                try:
                    if x not in self:
                        return False
                except TypeError:
                    return False
            return True
        elif self.isfinite():
            return False
        else:
            return other.issubset(self)

    def union(self, *args):
        """
        Return the union of this set with one or more sets.
        """
        tmp = self
        for arg in args:
            tmp = SetUnion(tmp, arg)
        return tmp

    def intersection(self, *args):
        """
        Return the intersection of this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = SetIntersection(tmp, arg)
        return tmp

    def difference(self, *args):
        """
        Return the difference between this set with one or more sets
        """
        tmp = self
        for arg in args:
            tmp = SetDifference(tmp, arg)
        return tmp

    def symmetric_difference(self, other):
        """
        Return the symmetric difference of this set with another set
        """
        return SetSymmetricDifference(self, other)

    def cross(self, *args):
        """
        Return the cross-product between this set and one or more sets
        """
        return SetProduct(self, *args)

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
        # See the discussion of Set vs SetOf in _processArgs below
        #
        # return SetOf(other) | self
        tmp = SetOf(other)
        ans = Set(initialize=tmp, ordered=tmp.isordered())
        ans.construct()
        return ans | self

    def __rand__(self, other):
        # See the discussion of Set vs SetOf in _processArgs below
        #
        # return SetOf(other) & self
        tmp = SetOf(other)
        ans = Set(initialize=tmp, ordered=tmp.isordered())
        ans.construct()
        return ans & self

    def __rsub__(self, other):
        # See the discussion of Set vs SetOf in _processArgs below
        #
        # return SetOf(other) - self
        tmp = SetOf(other)
        ans = Set(initialize=tmp, ordered=tmp.isordered())
        ans.construct()
        return ans - self

    def __rxor__(self, other):
        # See the discussion of Set vs SetOf in _processArgs below
        #
        # return SetOf(other) ^ self
        tmp = SetOf(other)
        ans = Set(initialize=tmp, ordered=tmp.isordered())
        ans.construct()
        return ans ^ self

    def __rmul__(self, other):
        # See the discussion of Set vs SetOf in _processArgs below
        #
        # return SetOf(other) * self
        tmp = SetOf(other)
        ans = Set(initialize=tmp, ordered=tmp.isordered())
        ans.construct()
        return ans * self

    def __lt__(self,other):
        """
        Return True if the set is a strict subset of 'other'
        """
        return self <= other and not self == other

    def __gt__(self,other):
        """
        Return True if the set is a strict superset of 'other'
        """
        return self >= other and not self == other


class _FiniteSetMixin(object):
    __slots__ = ()

    def __len__(self):
        raise DeveloperError("Derived finite set class (%s) failed to "
                             "implement __len__" % (type(self).__name__,))

    def __iter__(self):
        raise DeveloperError("Derived finite set class (%s) failed to "
                             "implement __iter__" % (type(self).__name__,))

    def __reversed__(self):
        return reversed(self.data())

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return True

    def isfinite(self):
        """Returns True if this is a finite discrete (iterable) Set"""
        return True

    def data(self):
        return tuple(self)

    def sorted_data(self):
        return tuple(sorted_robust(self.data()))

    def ordered_data(self):
        return self.sorted_data()

    def bounds(self):
        try:
            lb = min(self)
        except:
            lb = None
        try:
            ub = max(self)
        except:
            ub = None
        # Python2/3 consistency: We will follow the Python3 convention
        # and not assume numeric/nonnumeric types are comparable.  If a
        # set is mixed non-numeric type, then we will report the bounds
        # as None.
        if type(lb) is not type(ub) and (
                type(lb) not in native_numeric_types
                or type(ub) not in native_numeric_types):
            return None,None
        else:
            return lb,ub

    def ranges(self):
        # This is way inefficient, but should always work: the ranges in a
        # Finite set is the list of scalars
        for i in self:
            if i.__class__ in native_numeric_types:
                yield NumericRange(i,i,0)
            elif i.__class__ in native_types:
                yield NonNumericRange(i)
            else:
                # Because of things like SetOf, self could contain types
                # we have never seen before.
                try:
                    as_numeric(i)
                    yield NumericRange(i,i,0)
                except:
                    yield NonNumericRange(i)


class _FiniteSetData(_FiniteSetMixin, _SetData):
    """A general unordered iterable Set"""
    __slots__ = ('_values', '_domain', '_validate', '_filter', '_dimen')

    def __init__(self, component):
        _SetData.__init__(self, component=component)
        # Derived classes (like _OrderedSetData) may want to change the
        # storage
        if not hasattr(self, '_values'):
            self._values = set()
        self._domain = None
        self._validate = None
        self._filter = None
        self._dimen = UnknownSetDimen

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_FiniteSetData, self).__getstate__()
        for i in _FiniteSetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because none of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __contains__(self, value):
        """
        Return True if the set contains a given value.

        This method will raise TypeError for unhashable types.
        """
        if normalize_index.flatten:
            value = normalize_index(value)

        return value in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return len(self._values)

    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        if not self.parent_component()._constructed:
            return type(self).__name__
        return "{" + (', '.join(str(_) for _ in self)) + "}"

    @property
    def dimen(self):
        return self._dimen

    def add(self, value):
        if normalize_index.flatten:
            _value = normalize_index(value)
            if _value.__class__ is tuple:
                _d = len(_value)
            else:
                _d = 1
        else:
            # If we are not normalizing indices, then we cannot reliably
            # infer the set dimen
            _value = value
            _d = None

        if _value not in self._domain:
            raise ValueError("Cannot add value %s to Set %s.\n"
                             "\tThe value is not in the domain %s"
                             % (value, self.name, self._domain))

        # We wrap this check in a try-except because some values (like lists)
        #  are not hashable and can raise exceptions.
        try:
            if _value in self:
                logger.warning(
                    "Element %s already exists in Set %s; no action taken"
                    % (value, self.name))
                return False
        except:
            exc = sys.exc_info()
            raise TypeError("Unable to insert '%s' into Set %s:\n\t%s: %s"
                            % (value, self.name, exc[0].__name__, exc[1]))

        if self._filter is not None:
            if not self._filter(self, _value):
                return False

        if self._validate is not None:
            try:
                flag = self._validate(self, _value)
            except:
                logger.error(
                    "Exception raised while validating element '%s' for Set %s"
                    % (value, self.name))
                raise
            if not flag:
                raise ValueError(
                    "The value=%s violates the validation rule of Set %s"
                    % (value, self.name))

        # If the Set has a fixed dimension, check that this element is
        # compatible.
        if self._dimen is not None:
            if _d != self._dimen:
                if self._dimen is UnknownSetDimen:
                    # The first thing added to a Set with unknown
                    # dimension sets its dimension
                    self._dimen = _d
                else:
                    raise ValueError(
                        "The value=%s has dimension %s and is not valid for "
                        "Set %s which has dimen=%s"
                        % (value, _d, self.name, self._dimen))

        # Add the value to this object (this last redirection allows
        # derived classes to implement a different storage mmechanism)
        self._add_impl(_value)
        return True

    def _add_impl(self, value):
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

    def update(self, values):
        for v in values:
            if v not in self:
                self.add(v)

    def pop(self):
        return self._values.pop()


class _OrderedSetMixin(object):
    __slots__ = ()

    def __getitem__(self, index):
        raise DeveloperError("Derived ordered set class (%s) failed to "
                             "implement __getitem__" % (type(self).__name__,))

    def ord(self, val):
        raise DeveloperError("Derived ordered set class (%s) failed to "
                             "implement ord" % (type(self).__name__,))

    def isordered(self):
        """Returns True if this is an ordered finite discrete (iterable) Set"""
        return True

    def ordered_data(self):
        return self.data()

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
        position = self.ord(item)+step
        if position < 1:
            raise IndexError("Cannot advance before the beginning of the Set")
        if position > len(self):
            raise IndexError("Cannot advance past the end of the Set")
        return self[position]

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
        # Efficiency note: unlike older Set implementations, this
        # implementation does not guarantee that the index is valid (it
        # could be outside of abs(i) <= len(self)).
        try:
            if item != int(item):
                raise IndexError(
                    "%s indices must be integers, not %s"
                    % (self.name, type(item).__name__,))
            item = int(item)
        except:
            raise IndexError(
                "%s indices must be integers, not %s"
                % (self.name, type(item).__name__,))

        if item >= 1:
            return item - 1
        elif item < 0:
            item += len(self)
            if item < 0:
                raise IndexError("%s index out of range" % (self.name,))
            return item
        else:
            raise IndexError(
                "Pyomo Sets are 1-indexed: valid index values for Sets are "
                "[1 .. len(Set)] or [-1 .. -len(Set)]")


class _OrderedSetData(_OrderedSetMixin, _FiniteSetData):
    """
    This class defines the base class for an ordered set of concrete data.

    In older Pyomo terms, this defines a "concrete" ordered set - that is,
    a set that "owns" the list of set members.  While this class actually
    implements a set ordered by insertion order, we make the "official"
    _InsertionOrderSetData an empty derivative class, so that

         issubclass(_SortedSetData, _InsertionOrderSetData) == False

    Constructor Arguments:
        component   The Set object that owns this data.

    Public Class Attributes:
    """

    __slots__ = ('_ordered_values',)

    def __init__(self, component):
        self._values = {}
        self._ordered_values = []
        _FiniteSetData.__init__(self, component=component)

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_OrderedSetData, self).__getstate__()
        for i in _OrderedSetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because none of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __iter__(self):
        """
        Return an iterator for the set.
        """
        return iter(self._ordered_values)

    def __reversed__(self):
        return reversed(self._ordered_values)

    def _add_impl(self, value):
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
        self._ordered_values = []

    def pop(self):
        try:
            ans = self.last()
        except IndexError:
            # Map the index error to a KeyError for consistency with
            # set().pop()
            raise KeyError('pop from an empty set')
        self.discard(ans)
        return ans

    def __getitem__(self, index):
        """
        Return the specified member of the set.

        The public Set API is 1-based, even though the
        internal _lookup and _values are (pythonically) 0-based.
        """
        i = self._to_0_based_index(index)
        try:
            return self._ordered_values[i]
        except IndexError:
            raise IndexError("%s index out of range" % (self.name))

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        # The bulk of single-value set members are stored as scalars.
        # However, we are now being more careful about matching tuples
        # when they are actually put as Set members.  So, we will look
        # for the exact thing that the user sent us and then fall back
        # on the scalar.
        try:
            return self._values[item] + 1
        except KeyError:
            if item.__class__ is not tuple or len(item) > 1:
                raise ValueError(
                    "%s.ord(x): x not in %s" % (self.name, self.name))
        try:
            return self._values[item[0]] + 1
        except KeyError:
            raise ValueError(
                "%s.ord(x): x not in %s" % (self.name, self.name))


class _InsertionOrderSetData(_OrderedSetData):
    """
    This class defines the data for a ordered set where the items are ordered
    in insertion order (similar to Python's OrderedSet.

    Constructor Arguments:
        component   The Set object that owns this data.

    Public Class Attributes:
    """
    __slots__ = ()

    def set_value(self, val):
        if type(val) in self._UnorderedInitializers:
            logger.warning(
                "Calling set_value() on an insertion order Set with "
                "a fundamentally unordered data source (type: %s).  "
                "This WILL potentially lead to nondeterministic behavior "
                "in Pyomo" % (type(val).__name__,))
        super(_InsertionOrderSetData, self).set_value(val)

    def update(self, values):
        if type(values) in self._UnorderedInitializers:
            logger.warning(
                "Calling update() on an insertion order Set with "
                "a fundamentally unordered data source (type: %s).  "
                "This WILL potentially lead to nondeterministic behavior "
                "in Pyomo" % (type(values).__name__,))
        super(_InsertionOrderSetData, self).update(values)


class _SortedSetMixin(object):
    ""
    __slots__ = ()


class _SortedSetData(_SortedSetMixin, _OrderedSetData):
    """
    This class defines the data for a sorted set.

    Constructor Arguments:
        component   The Set object that owns this data.

    Public Class Attributes:
    """

    __slots__ = ('_is_sorted',)

    def __init__(self, component):
        # An empty set is sorted...
        self._is_sorted = True
        _OrderedSetData.__init__(self, component=component)

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_SortedSetData, self).__getstate__()
        for i in _SortedSetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because none of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __iter__(self):
        """
        Return an iterator for the set.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).__iter__()

    def __reversed__(self):
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).__reversed__()

    def _add_impl(self, value):
        # Note that the sorted status has no bearing on insertion,
        # so there is no reason to check if the data is correctly sorted
        self._values[value] = len(self._values)
        self._ordered_values.append(value)
        self._is_sorted = False

    # Note: removing data does not affect the sorted flag
    #def remove(self, val):
    #def discard(self, val):

    def clear(self):
        super(_SortedSetData, self).clear()
        self._is_sorted = True

    def __getitem__(self, index):
        """
        Return the specified member of the set.

        The public Set API is 1-based, even though the
        internal _lookup and _values are (pythonically) 0-based.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).__getitem__(index)

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).ord(item)

    def sorted_data(self):
        return self.data()

    def _sort(self):
        self._ordered_values = list(self.parent_component()._sort_fcn(
            self._ordered_values))
        self._values = dict(
            (j, i) for i, j in enumerate(self._ordered_values) )
        self._is_sorted = True


############################################################################

_SET_API = (
    ('__contains__', 'test membership in'),
    'ranges', 'bounds',
)
_FINITESET_API = _SET_API + (
    ('__iter__', 'iterate over'),
    '__reversed__', '__len__', 'data', 'sorted_data', 'ordered_data',
)
_ORDEREDSET_API = _FINITESET_API + (
    '__getitem__', 'ord',
)
_SETDATA_API = (
    'set_value', 'add', 'remove', 'discard', 'clear', 'update', 'pop',
)

class Set(IndexedComponent):
    """
    A component used to index other Pyomo components.

    This class provides a Pyomo component that is API-compatible with
    Python `set` objects, with additional features, including:
        1. Member validation and filtering.  The user can declare
           domains and provide callback functions to validate set
           members and to filter (ignore) potential members.
        2. Set expressions.  Operations on Set objects (&,|,*,-,^)
           produce Set expressions taht preserve their references to the
           original Set objects so that updating the argument Sets
           implicitly updates the Set operator instance.
        3. Support for set operations with RangeSet instances (both
           finite and non-finite ranges).

    Parameters
    ----------
        name : str, optional
            The name of the set
        doc : str, optional
            A text string describing this component
        initialize : initializer(iterable), optional
            The initial values to store in the Set when it is
            constructed.  Values passed to `initialize` may be
            overridden by `data` passed to the :py:meth:`construct`
            method.
        dimen : initializer(int)
            Specify the Set's arity, or None if no arity is enforced
        ordered : bool or Set.InsertionOrder or Set.SortedOrder or function
            Specifies whether the set is ordered. Possible values are:
                False               Unordered
                True                Ordered by insertion order
                Set.InsertionOrder  Ordered by insertion order [default]
                Set.SortedOrder     Ordered by sort order
                <function>          Ordered with this comparison function
        within : initialiser(set), optional
            A set that defines the valid values that can be contained
            in this set
        domain : initializer(set), optional
            A set that defines the valid values that can be contained
            in this set
        bounds : initializer(tuple), optional
            A 2-tuple that specifies the lower and upper bounds for
            valid Set values
        filter : initializer(rule), optional
            A rule for determining membership in this set. This has the
            functional form:

                ``f: Block, *data -> bool``

            and returns True if the data belongs in the set.  Set will
            quietly ignore any values where `filter` returns False.
        validate : initializer(rule), optional
            A rule for validating membership in this set. This has the
            functional form:

                ``f: Block, *data -> bool``

            and returns True if the data belongs in the set.  Set will
            raise a ``ValueError`` for any values where `validate`
            returns False.

    Notes
    -----
        `domain`, `within`, and `bounds` all provide restrictions on the
        valid set values.  If more than one is specified, Set values
        will be restricted to the intersection of `domain`, `within`,
        and `bounds`.
    """

    class End(object): pass
    class Skip(object): pass
    class InsertionOrder(object): pass
    class SortedOrder(object): pass
    _ValidOrderedAuguments = {True, False, InsertionOrder, SortedOrder}
    _UnorderedInitializers = {set}
    if _prePython37:
        _UnorderedInitializers.add(dict)

    def __new__(cls, *args, **kwds):
        if cls is not Set:
            return super(Set, cls).__new__(cls)

        # TBD: Should ordered be allowed to vary across an IndexedSet?
        #
        # Many things are easier by forcing it to be consistent across
        # the set (namely, the _ComponentDataClass is constant).
        # However, it is a bit off that 'ordered' it the only arg NOT
        # processed by Initializer.  We can mock up a _SortedSetData
        # sort function that preserves Insertion Order (lambda x: x), but
        # the unsorted is harder (it would effectively be insertion
        # order, but ordered() may not be deterministic based on how the
        # set was populated - and we could not issue a warning?)
        #
        # JDS [5/2019]: Until someone demands otherwise, I think we
        # should leave it constant across an IndexedSet
        ordered = kwds.get('ordered', Set.InsertionOrder)
        if ordered is True:
            ordered = Set.InsertionOrder
        if ordered not in Set._ValidOrderedAuguments:
            if inspect.isfunction(ordered):
                ordered = Set.SortedOrder
            else:
                # We want the list to be deterministic, but not
                # alphabetical, so we first sort by type and then
                # convert evetything to string.  Note that we have to
                # convert *types* to string early, as the default
                # ordering of types is random: so InsertionOrder and
                # SortedOrder would occasionally swap places.
                raise TypeError(
                    "Set 'ordered' argument is not valid (must be one of {%s})"
                    % ( ', '.join(str(_) for _ in sorted_robust(
                        'Set.'+x.__name__ if isinstance(x,type) else x
                        for x in Set._ValidOrderedAuguments.union(
                                {'<function>',})
                    ))))
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            if ordered is Set.InsertionOrder:
                return super(Set, cls).__new__(AbstractOrderedSimpleSet)
            elif ordered is Set.SortedOrder:
                return super(Set, cls).__new__(AbstractSortedSimpleSet)
            else:
                return super(Set, cls).__new__(AbstractFiniteSimpleSet)
        else:
            newObj = super(Set, cls).__new__(IndexedSet)
            if ordered is Set.InsertionOrder:
                newObj._ComponentDataClass = _InsertionOrderSetData
            elif ordered is Set.SortedOrder:
                newObj._ComponentDataClass = _SortedSetData
            else:
                newObj._ComponentDataClass = _FiniteSetData
            return newObj

    def __init__(self, *args, **kwds):
        kwds.setdefault('ctype', Set)

        # The ordered flag was processed by __new__, but if this is a
        # sorted set, then we need to set the sorting function
        _ordered = kwds.pop('ordered',None)
        if _ordered and _ordered is not Set.InsertionOrder \
                and _ordered is not True:
            if inspect.isfunction(_ordered):
                self._sort_fcn = _ordered
            else:
                self._sort_fcn = sorted_robust

        # 'domain', 'within', and 'bounds' are synonyms, in that they
        # restrict the set of valid set values.  If more than one is
        # specified, we will restrict the Set values to the intersection
        # of the individual arguments
        self._init_domain = SetInitializer(None)
        _domain = kwds.pop('domain', None)
        if _domain is not None:
            self._init_domain.intersect(SetInitializer(_domain))
        _within = kwds.pop('within', None)
        if _within is not None:
            self._init_domain.intersect(SetInitializer(_within))
        _bounds = kwds.pop('bounds', None)
        if _bounds is not None:
            self._init_domain.intersect(RangeSetInitializer(
                _bounds, default_step=0))

        self._init_dimen = Initializer(
            kwds.pop('dimen', UnknownSetDimen),
            arg_not_specified=UnknownSetDimen)
        self._init_values = Initializer(
            kwds.pop('initialize', ()),
            treat_sequences_as_mappings=False, allow_generators=True)
        self._init_validate = Initializer(kwds.pop('validate', None))
        self._init_filter = Initializer(kwds.pop('filter', None))

        if 'virtual' in kwds:
            deprecation_warning(
                "Pyomo Sets ignore the 'virtual' keyword argument",
                logger='pyomo.core.base')
            kwds.pop('virtual')

        IndexedComponent.__init__(self, *args, **kwds)

        # HACK to make the "counted call" syntax work.  We wait until
        # after the base class is set up so that is_indexed() is
        # reliable.
        if self._init_values.__class__ is IndexedCallInitializer:
            self._init_values = CountedCallInitializer(self, self._init_values)

    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing Set, name=%s, from data=%r"
                             % (self.name, data))
        self._constructed = True
        if data is not None:
            # Data supplied to construct() should override data provided
            # to the constructor
            tmp_init, self._init_values = self._init_values, Initializer(
                    data, treat_sequences_as_mappings=False)
        try:
            if type(self._init_values) is ItemInitializer:
                for index in iterkeys(self._init_values._dict):
                    # The index is coming in externally; we need to
                    # validate it
                    IndexedComponent.__getitem__(self, index)
            else:
                for index in self.index_set():
                    self._getitem_when_not_present(index)
        finally:
            # Restore the original initializer (if overridden by data argument)
            if data is not None:
                self._init_values = tmp_init
        timer.report()

    #
    # This method must be defined on subclasses of
    # IndexedComponent that support implicit definition
    #
    def _getitem_when_not_present(self, index):
        """Returns the default component data value."""
        if self._init_values is not None:
            _values = self._init_values(self, index)
            if _values is Set.Skip:
                return
            elif _values is None:
                raise ValueError(
                    "Set rule or initializer returned None instead of Set.Skip")

        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        if self._init_dimen is not None:
            _d = self._init_dimen(self, index)
            if _d is not UnknownSetDimen and (not normalize_index.flatten) \
               and _d is not None:
                logger.warning(
                    "Ignoring non-None dimen (%s) for set %s "
                    "(normalize_index.flatten is False, so dimen "
                    "verification is not available)." % (_d, obj.name))
                _d = None
            obj._dimen = _d
        if self._init_domain is not None:
            obj._domain = self._init_domain(self, index)
            if isinstance(obj._domain, _SetOperator):
                obj._domain.construct()
        if self._init_validate is not None:
            try:
                obj._validate = Initializer(self._init_validate(self, index))
                if obj._validate.constant():
                    # _init_validate was the actual validate function; use it.
                    obj._validate = self._init_validate
            except:
                # We will assume any exceptions raised when getting the
                # validator for this index indicate that the function
                # should have been passed directly to the underlying sets.
                obj._validate = self._init_validate
        if self._init_filter is not None:
            try:
                _filter = Initializer(self._init_filter(self, index))
                if _filter.constant():
                    # _init_filter was the actual filter function; use it.
                    _filter = self._init_filter
            except:
                # We will assume any exceptions raised when getting the
                # filter for this index indicate that the function
                # should have been passed directly to the underlying sets.
                _filter = self._init_filter
        else:
            _filter = None
        if self._init_values is not None:
            # _values was initialized above...
            if obj.isordered() \
                   and type(_values) in self._UnorderedInitializers:
                logger.warning(
                    "Initializing an ordered Set with a fundamentally "
                    "unordered data source (type: %s).  This WILL potentially "
                    "lead to nondeterministic behavior in Pyomo"
                    % (type(_values).__name__,))
            # Special case: set operations that are not first attached
            # to the model must be constructed.
            if isinstance(_values, _SetOperator):
                _values.construct()
            for val in _values:
                if val is Set.End:
                    break
                if _filter is None or _filter(self, val):
                    obj.add(val)
        # We defer adding the filter until now so that add() doesn't
        # call it a second time.
        obj._filter = _filter
        return obj

    @staticmethod
    def _pprint_members(x):
        if x.isfinite():
            return '{' + str(x.ordered_data())[1:-1] + "}"
        else:
            ans = ' | '.join(str(_) for _ in x.ranges())
            if ' | ' in ans:
                return "(" + ans + ")"
            if ans:
                return ans
            else:
                return "[]"

    @staticmethod
    def _pprint_dimen(x):
        d = x.dimen
        if d is UnknownSetDimen:
            return "--"
        return d

    @staticmethod
    def _pprint_domain(x):
        if x._domain is x:
            return x._expression_str()
        else:
            return x._domain

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        #
        # Eventually, we might want to support a 'verbose' flag to
        # pprint() that will suppress som of the very long (less
        # informative) output
        #
        # if verbose:
        #     def members(x):
        #         return '{' + str(x.ordered_data())[1:-1] + "}"
        # else:
        #     MAX_MEMBERES=10
        #     def members(x):
        #         ans = x.ordered_data()
        #         if len(x) > MAX_MEMBERES:
        #             return '{' + str(ans[:MAX_MEMBERES])[1:-1] + ', ...}'
        #         else:
        #             return '{' + str(ans)[1:-1] + "}"

        # TBD: In the current design, we force all _SetData within an
        # indexed Set to have the same isordered value, so we will only
        # print it once in the header.  Is this a good design?
        try:
            _ordered = self.isordered()
            _refClass = type(self)
        except:
            _ordered = issubclass(self._ComponentDataClass, _OrderedSetMixin)
            _refClass = self._ComponentDataClass
        if _ordered:
            # This is a bit of an anachronism.  Historically Pyomo
            # reported "Insertion" for Set.InsertionOrder, "Sorted" for
            # Set.SortedOrder, and "{user}" for everything else.
            # However, we do not preserve that flag any more, so we
            # will infer it from the class hierarchy
            if issubclass(_refClass, _SortedSetMixin):
                if self.parent_component()._sort_fcn is sorted_robust:
                    _ordered =  "Sorted"
                else:
                    _ordered =  "{user}"
            elif issubclass(_refClass, _InsertionOrderSetData):
                _ordered = "Insertion"
        return (
            [("Size", len(self._data)),
             ("Index", self._index if self.is_indexed() else None),
             ("Ordered", _ordered),],
            iteritems(self._data),
            ("Dimen","Domain","Size","Members",),
            lambda k, v: [
                Set._pprint_dimen(v),
                Set._pprint_domain(v),
                len(v) if v.isfinite() else 'Inf',
                Set._pprint_members(v),
            ])


class IndexedSet(Set):
    pass

class FiniteSimpleSet(_FiniteSetData, Set):
    def __init__(self, **kwds):
        _FiniteSetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

class OrderedSimpleSet(_InsertionOrderSetData, Set):
    def __init__(self, **kwds):
        _InsertionOrderSetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

class SortedSimpleSet(_SortedSetData, Set):
    def __init__(self, **kwds):
        _SortedSetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

@disable_methods(_FINITESET_API + _SETDATA_API)
class AbstractFiniteSimpleSet(FiniteSimpleSet):
    pass

@disable_methods(_ORDEREDSET_API + _SETDATA_API)
class AbstractOrderedSimpleSet(OrderedSimpleSet):
    pass

@disable_methods(_ORDEREDSET_API + _SETDATA_API)
class AbstractSortedSimpleSet(SortedSimpleSet):
    pass


############################################################################

class SetOf(_FiniteSetMixin, _SetData, Component):
    """"""
    def __new__(cls, *args, **kwds):
        if cls is not SetOf:
            return super(SetOf, cls).__new__(cls)
        reference, = args
        if isinstance(reference, (tuple, list)):
            return super(SetOf, cls).__new__(OrderedSetOf)
        else:
            return super(SetOf, cls).__new__(UnorderedSetOf)

    def __init__(self, reference, **kwds):
        _SetData.__init__(self, component=self)
        kwds.setdefault('ctype', SetOf)
        Component.__init__(self, **kwds)
        self._ref = reference

    def __contains__(self, value):
        # Note that the efficiency of this depends on the reference object
        #
        # The bulk of single-value set members were stored as scalars.
        # Check that first.
        if value.__class__ is tuple and len(value) == 1:
            if value[0] in self._ref:
                return True
        return value in self._ref

    def __len__(self):
        return len(self._ref)

    def __iter__(self):
        return iter(self._ref)

    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        return str(self._ref)

    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing SetOf, name=%s, from data=%r"
                             % (self.name, data))
        self._constructed = True
        timer.report()

    @property
    def dimen(self):
        _iter = iter(self)
        try:
            x = next(_iter)
            if type(x) is tuple:
                ans = len(x)
            else:
                ans = 1
        except:
            return 0
        for x in _iter:
            _this = len(x) if type(x) is tuple else 1
            if _this != ans:
                return None
        return ans

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Dimen", self.dimen),
             ("Size", len(self)),
             ("Bounds", self.bounds())],
            iteritems( {None: self} ),
            ("Ordered", "Members",),
            lambda k, v: [
                v.isordered(),
                str(v._ref),
            ])

class UnorderedSetOf(SetOf):
    pass

class OrderedSetOf(_OrderedSetMixin, SetOf):
    def __getitem__(self, index):
        i = self._to_0_based_index(index)
        try:
            return self._ref[i]
        except IndexError:
            raise IndexError("%s index out of range" % (self.name))

    def ord(self, item):
        # The bulk of single-value set members are stored as scalars.
        # However, we are now being more careful about matching tuples
        # when they are actually put as Set members.  So, we will look
        # for the exact thing that the user sent us and then fall back
        # on the scalar.
        try:
            return self._ref.index(item) + 1
        except ValueError:
            if item.__class__ is not tuple or len(item) > 1:
                raise
        return self._ref.index(item[0]) + 1


############################################################################

class _InfiniteRangeSetData(_SetData):
    """Data class for a infinite set.

    This Set implements an interface to an *infinite set* defined by one
    or more NumericRange objects.  As there are an infinite
    number of members, Infinite Range Sets are not iterable.

    """

    __slots__ = ('_ranges',)

    def __init__(self, component):
        _SetData.__init__(self, component=component)
        self._ranges = None

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_InfiniteRangeSetData, self).__getstate__()
        for i in _InfiniteRangeSetData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because none of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __contains__(self, value):
        # The bulk of single-value set members were stored as scalars.
        # Check that first.
        if value.__class__ is tuple and len(value) == 1:
            v = value[0]
            if any(v in r for r in self._ranges):
                return True
        return any(value in r for r in self._ranges)

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return all(r.isdiscrete() for r in self.ranges())

    @property
    def dimen(self):
        return 1

    def ranges(self):
        return iter(self._ranges)


class _FiniteRangeSetData( _SortedSetMixin,
                           _OrderedSetMixin,
                           _FiniteSetMixin,
                           _InfiniteRangeSetData ):
    __slots__ = ()

    @staticmethod
    def _range_gen(r):
        start, end = (r.start, r.end) if r.step > 0 else (r.end, r.start)
        step = abs(r.step)
        n = start
        i = 0
        if start == end:
            yield start
        else:
            while n <= end:
                yield n
                i += 1
                n = start + i*step

    def __iter__(self):
        # If there is only a single underlying range, then we will
        # iterate over it
        nIters = len(self._ranges) - 1
        if not nIters:
            for x in _FiniteRangeSetData._range_gen(self._ranges[0]):
                yield x
            return

        # The trick here is that we need to remove any duplicates from
        # the multiple ranges.  We will set up iterators for each range,
        # pull the first element from each iterator, sort and yield the
        # lowest value.
        iters = []
        for r in self._ranges:
            # Note: there should always be at least 1 member in each
            # NumericRange
            i = _FiniteRangeSetData._range_gen(r)
            iters.append([next(i), i])

        iters.sort(reverse=True, key=lambda x: x[0])
        n = None
        while iters:
            if n != iters[-1][0]:
                n = iters[-1][0]
                yield n
            try:
                iters[-1][0] = next(iters[-1][1])
                if nIters and iters[-2][0] < iters[-1][0]:
                    iters.sort(reverse=True)
            except StopIteration:
                iters.pop()
                nIters -= 1

    def __len__(self):
        if len(self._ranges) == 1:
            # If there is only one range, then this set's range is equal
            # to the range's length
            r = self._ranges[0]
            if r.start == r.end:
                return 1
            else:
                return (r.end - r.start) // r.step + 1
        else:
            return sum(1 for _ in self)

    def __getitem__(self, index):
        assert int(index) == index
        idx = self._to_0_based_index(index)
        if len(self._ranges) == 1:
            r = self._ranges[0]
            ans = r.start + (idx)*r.step
            if ans <= r.end:
                return ans
        else:
            for ans in self:
                if not idx:
                    return ans
                idx -= 1
        raise IndexError("%s index out of range" % (self.name,))

    def ord(self, item):
        if len(self._ranges) == 1:
            r = self._ranges[0]
            i = float(item - r.start) / r.step
            if item >= r.start and item <= r.end and \
                    abs(i - math.floor(i+0.5)) < r._EPS:
                return int(math.floor(i+0.5)) + 1
        else:
            ans = 1
            for val in self:
                if val == item:
                    return ans
                ans += 1
        raise ValueError(
            "Cannot identify position of %s in Set %s: item not in Set"
            % (item, self.name))

    # We must redefine ranges() and bounds() so that we get the
    # _InfiniteRangeSetData version and not the one from
    # _FiniteSetMixin.
    bounds = _InfiniteRangeSetData.bounds
    ranges = _InfiniteRangeSetData.ranges


class RangeSet(Component):
    """
    A set object that represents a set of numeric values

    """

    def __new__(cls, *args, **kwds):
        if cls is not RangeSet:
            return super(RangeSet, cls).__new__(cls)

        finite = kwds.pop('finite', None)
        if finite is None:
            if 'ranges' in kwds:
                if any(not r.isfinite() for r in kwds['ranges']):
                    finite = False
            if all(type(_) in native_types for _ in args):
                if None in args or (len(args) > 2 and args[2] == 0):
                    finite = False
            if finite is None:
                finite = True

        if finite:
            return super(RangeSet, cls).__new__(AbstractFiniteSimpleRangeSet)
        else:
            return super(RangeSet, cls).__new__(AbstractInfiniteSimpleRangeSet)


    def __init__(self, *args, **kwds):
        # Finite was processed by __new__
        kwds.setdefault('ctype', RangeSet)
        if len(args) > 3:
            raise ValueError("RangeSet expects 3 or fewer positional "
                             "arguments (received %s)" % (len(args),))
        kwds.pop('finite', None)
        self._init_data = (
            args,
            kwds.pop('ranges', ()),
        )
        Component.__init__(self, **kwds)
        # Shortcut: if all the relevant construction information is
        # simple (hard-coded) values, then it is safe to go ahead and
        # construct the set.
        #
        # NOTE: We will need to revisit this if we ever allow passing
        # data into the construct method (which would override the
        # hard-coded values here).
        if all(type(_) in native_types for _ in args):
            self.construct()


    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        if not self._constructed:
            return type(self).__name__
        ans = ' | '.join(str(_) for _ in self.ranges())
        if ' | ' in ans:
            return "(" + ans + ")"
        if ans:
            return ans
        else:
            return "[]"


    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing RangeSet, name=%s, from data=%r"
                             % (self.name, data))
        if data is not None:
            raise ValueError(
                "RangeSet.construct() does not support the data= argument.")
        self._constructed = True

        args, ranges = self._init_data
        args = tuple(value(_) for _ in args)
        if type(ranges) is not tuple:
            ranges = tuple(ranges)
        if len(args) == 1:
            # This is a bit of a hack for backwards compatability with
            # the old RangeSet implementation, where we did less
            # validation of the RangeSet arguments, and allowed the
            # creation of 0-length RangeSets
            if args[0] != 0:
                # No need to check for floating point - it will
                # automatically be truncated
                ranges = ranges + (NumericRange(1,args[0],1),)
        elif len(args) == 2:
            # This is a bit of a hack for backwards compatability with
            # the old RangeSet implementation, where we did less
            # validation of the RangeSet arguments, and allowed the
            # creation of 0-length RangeSets
            if args[1] - args[0] != -1:
                args = (args[0],args[1],1)

        if len(args) == 3:
            # Discrete ranges anchored by a floating point value or
            # incremented by a floating point value cannot be handled by
            # the NumericRange object.  We will just discretize this
            # range (mostly for backwards compatability)
            start, end, step = args
            if step and int(step) != step:
                if (end >= start) ^ (step > 0):
                    raise ValueError(
                        "RangeSet: start, end ordering incompatible with "
                        "step direction (got [%s:%s:%s])" % (start,end,step))
                n = start
                i = 0
                while (step > 0 and n <= end) or (step < 0 and n >= end):
                    ranges = ranges + (NumericRange(n,n,0),)
                    i += 1
                    n = start + step*i
            else:
                ranges = ranges + (NumericRange(*args),)

        for r in ranges:
            if not isinstance(r, NumericRange):
                raise TypeError(
                    "RangeSet 'ranges' argument must be an "
                    "iterable of NumericRange objects")
            if not r.isfinite() and self.isfinite():
                raise ValueError(
                    "Constructing a finite RangeSet over a non-finite "
                    "range (%s).  Either correct the range data or "
                    "specify 'finite=False' when declaring the RangeSet"
                    % (r,))

        self._ranges = ranges

        timer.report()


    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Dimen", self.dimen),
             ("Size", len(self) if self.isfinite() else 'Inf'),
             ("Bounds", self.bounds())],
            iteritems( {None: self} ),
            ("Finite","Members",),
            lambda k, v: [
                v.isfinite(),#isinstance(v, _FiniteSetMixin),
                ', '.join(str(r) for r in self.ranges()) or '[]',
            ])


class InfiniteSimpleRangeSet(_InfiniteRangeSetData, RangeSet):
    def __init__(self, *args, **kwds):
        _InfiniteRangeSetData.__init__(self, component=self)
        RangeSet.__init__(self, *args, **kwds)

    # We want the RangeSet.__str__ to override the one in _FiniteSetMixin
    __str__ = RangeSet.__str__

class FiniteSimpleRangeSet(_FiniteRangeSetData, RangeSet):
    def __init__(self, *args, **kwds):
        _FiniteRangeSetData.__init__(self, component=self)
        RangeSet.__init__(self, *args, **kwds)

    # We want the RangeSet.__str__ to override the one in _FiniteSetMixin
    __str__ = RangeSet.__str__


@disable_methods(_SET_API)
class AbstractInfiniteSimpleRangeSet(InfiniteSimpleRangeSet):
    pass

@disable_methods(_ORDEREDSET_API)
class AbstractFiniteSimpleRangeSet(FiniteSimpleRangeSet):
    pass


############################################################################
# Set Operators
############################################################################

class _SetOperator(_SetData, Set):
    __slots__ = ('_sets','_implicit_subsets')

    def __init__(self, *args, **kwds):
        _SetData.__init__(self, component=self)
        Set.__init__(self, **kwds)
        implicit = []
        sets = []
        for _set in args:
            _new_set = process_setarg(_set)
            sets.append(_new_set)
            if _new_set is not _set or _new_set.parent_block() is None:
                implicit.append(_new_set)
        self._sets = tuple(sets)
        self._implicit_subsets = tuple(implicit)

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_SetOperator, self).__getstate__()
        for i in _SetOperator.__slots__:
            state[i] = getattr(self, i)
        return state

    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing SetOperator, name=%s, from data=%r"
                             % (self.name, data))
        for s in self._sets:
            s.construct()
        super(_SetOperator, self).construct(data)
        timer.report()

    # Note: because none of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __len__(self):
        """Return the length of this Set

        Because Set objects (and therefore SetOperator objects) are a
        subclass of IndexedComponent, we need to override the definition
        of len() to return the length of the Set and not the Component.
        Failing to do so would result in scalar infinite set operators
        to return a length of "1".

        Python requires len() to return a nonnegatie integer.  Instead
        of returning `float('inf')` here and allowing Python to raise
        the OverflowError, we will raise it directly here where we can
        provide a more informative error message.

        """
        raise OverflowError(
            "The length of a non-finite Set is Inf; however, Python "
            "requires len() to return a non-negative integer value. Check "
            "isfinite() before calling len() for possibly infinite Sets")

    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        return self._expression_str()

    def _expression_str(self):
        _args = []
        for arg in self._sets:
            arg_str = str(arg)
            if ' ' in arg_str and isinstance(arg, _SetOperator):
                arg_str = "(" + arg_str + ")"
            _args.append(arg_str)
        return self._operator.join(_args)

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return all(r.isdiscrete() for r in self.ranges())

    @property
    def _domain(self):
        # We hijack the _domain attribute of _SetOperator so that pprint
        # prints out the expression as the Set's "domain".  Doing this
        # as a property prevents the circular reference
        return self

    @_domain.setter
    def _domain(self, val):
        if val is not Any:
            raise ValueError(
                "Setting the domain of a Set Operator is not allowed: %s" % val)

    @property
    @deprecated("The 'virtual' flag is no longer supported", version='TBD')
    def virtual(self):
        return True

    @staticmethod
    def _checkArgs(*sets):
        ans = []
        for s in sets:
            if isinstance(s, _SetDataBase):
                ans.append((s.isordered(), s.isfinite()))
            elif type(s) in {tuple, list}:
                ans.append((True, True))
            else:
                ans.append((False, True))
        return ans

############################################################################

class SetUnion(_SetOperator):
    __slots__ = tuple()

    _operator = " | "

    def __new__(cls, *args):
        if cls != SetUnion:
            return super(SetUnion, cls).__new__(cls)

        set0, set1 = _SetOperator._checkArgs(*args)
        if set0[0] and set1[0]:
            cls = SetUnion_OrderedSet
        elif set0[1] and set1[1]:
            cls = SetUnion_FiniteSet
        else:
            cls = SetUnion_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        return itertools.chain(*tuple(s.ranges() for s in self._sets))

    @property
    def dimen(self):
        d0 = self._sets[0].dimen
        d1 = self._sets[1].dimen
        if d0 is None or d1 is None:
            return None
        if d0 is UnknownSetDimen or d1 is UnknownSetDimen:
            return UnknownSetDimen
        if d0 == d1:
            return d0
        else:
            return None


class SetUnion_InfiniteSet(SetUnion):
    __slots__ = tuple()

    def __contains__(self, val):
        return any(val in s for s in self._sets)


class SetUnion_FiniteSet(_FiniteSetMixin, SetUnion_InfiniteSet):
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


class SetUnion_OrderedSet(_OrderedSetMixin, SetUnion_FiniteSet):
    __slots__ = tuple()

    def __getitem__(self, index):
        idx = self._to_0_based_index(index)
        set0_len = len(self._sets[0])
        if idx < set0_len:
            return self._sets[0][idx+1]
        else:
            idx -= set0_len - 1
            set1_iter = iter(self._sets[1])
            try:
                while idx:
                    val = next(set1_iter)
                    if val not in self._sets[0]:
                        idx -= 1
            except StopIteration:
                raise IndexError("%s index out of range" % (self.name,))
            return val

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
        idx = len(self._sets[0])
        _iter = iter(self._sets[1])
        while True:
            val = next(_iter)
            if val == item:
                break
            elif val not in self._sets[0]:
                idx += 1
        return idx + 1


############################################################################

class SetIntersection(_SetOperator):
    __slots__ = tuple()

    _operator = " & "

    def __new__(cls, *args):
        if cls != SetIntersection:
            return super(SetIntersection, cls).__new__(cls)

        set0, set1 = _SetOperator._checkArgs(*args)
        if set0[0] or set1[0]:
            cls = SetIntersection_OrderedSet
        elif set0[1] or set1[1]:
            cls = SetIntersection_FiniteSet
        else:
            cls = SetIntersection_InfiniteSet
        return cls.__new__(cls)

    def construct(self, data=None):
        super(SetIntersection, self).construct(data)
        if not self.isfinite():
            _finite = True
            for r in self.ranges():
                if not r.isfinite():
                    _finite = False
                    break
            if _finite:
                self.__class__ = SetIntersection_OrderedSet

    def ranges(self):
        for a in self._sets[0].ranges():
            for r in a.range_intersection(self._sets[1].ranges()):
                yield r

    @property
    def dimen(self):
        d1 = self._sets[0].dimen
        d2 = self._sets[1].dimen
        if d1 is None:
            return d2
        elif d2 is None:
            return d1
        elif d1 == d2:
            return d1
        elif d1 is UnknownSetDimen or d2 is UnknownSetDimen:
            return UnknownSetDimen
        else:
            return 0


class SetIntersection_InfiniteSet(SetIntersection):
    __slots__ = tuple()

    def __contains__(self, val):
        return all(val in s for s in self._sets)


class SetIntersection_FiniteSet(_FiniteSetMixin, SetIntersection_InfiniteSet):
    __slots__ = tuple()

    def __iter__(self):
        set0, set1 = self._sets
        if not set0.isordered():
            if set1.isordered():
                set0, set1 = set1, set0
            elif not set0.isfinite():
                if set1.isfinite():
                    set0, set1 = set1, set0
                else:
                    # The odd case of a finite continuous range
                    # intersected with an infinite discrete range...
                    ranges = []
                    for r0 in set0.ranges():
                        ranges.extend(r0.range_intersection(set1.ranges()))
                    # Note that the RangeSet is automatically
                    # constucted, as it has no non-native positional
                    # parameters.
                    return iter(RangeSet(ranges=ranges))
        return (s for s in set0 if s in set1)

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return sum(1 for _ in self)


class SetIntersection_OrderedSet(_OrderedSetMixin, SetIntersection_FiniteSet):
    __slots__ = tuple()

    def __getitem__(self, index):
        idx = self._to_0_based_index(index)
        _iter = iter(self)
        try:
            while idx:
                next(_iter)
                idx -= 1
            return next(_iter)
        except StopIteration:
            raise IndexError("%s index out of range" % (self.name,))

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

class SetDifference(_SetOperator):
    __slots__ = tuple()

    _operator = " - "

    def __new__(cls, *args):
        if cls != SetDifference:
            return super(SetDifference, cls).__new__(cls)

        set0, set1 = _SetOperator._checkArgs(*args)
        if set0[0]:
            cls = SetDifference_OrderedSet
        elif set0[1]:
            cls = SetDifference_FiniteSet
        else:
            cls = SetDifference_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        for a in self._sets[0].ranges():
            for r in a.range_difference(self._sets[1].ranges()):
                yield r

    @property
    def dimen(self):
        return self._sets[0].dimen

class SetDifference_InfiniteSet(SetDifference):
    __slots__ = tuple()

    def __contains__(self, val):
        return val in self._sets[0] and not val in self._sets[1]


class SetDifference_FiniteSet(_FiniteSetMixin, SetDifference_InfiniteSet):
    __slots__ = tuple()

    def __iter__(self):
        set0, set1 = self._sets
        return (_ for _ in set0 if _ not in set1)

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return sum(1 for _ in self)


class SetDifference_OrderedSet(_OrderedSetMixin, SetDifference_FiniteSet):
    __slots__ = tuple()

    def __getitem__(self, index):
        idx = self._to_0_based_index(index)
        _iter = iter(self)
        try:
            while idx:
                next(_iter)
                idx -= 1
            return next(_iter)
        except StopIteration:
            raise IndexError("%s index out of range" % (self.name,))

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

class SetSymmetricDifference(_SetOperator):
    __slots__ = tuple()

    _operator = " ^ "

    def __new__(cls, *args):
        if cls != SetSymmetricDifference:
            return super(SetSymmetricDifference, cls).__new__(cls)

        set0, set1 = _SetOperator._checkArgs(*args)
        if set0[0] and set1[0]:
            cls = SetSymmetricDifference_OrderedSet
        elif set0[1] and set1[1]:
            cls = SetSymmetricDifference_FiniteSet
        else:
            cls = SetSymmetricDifference_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        # Note: the following loop implements for (a,b), (b,a)
        assert len(self._sets) == 2
        for set_a, set_b in (self._sets, reversed(self._sets)):
            for a_r in set_a.ranges():
                for r in a_r.range_difference(set_b.ranges()):
                    yield r

    @property
    def dimen(self):
        d0 = self._sets[0].dimen
        d1 = self._sets[1].dimen
        if d0 is None or d1 is None:
            return None
        if d0 is UnknownSetDimen or d1 is UnknownSetDimen:
            return UnknownSetDimen
        if d0 == d1:
            return d0
        else:
            return None


class SetSymmetricDifference_InfiniteSet(SetSymmetricDifference):
    __slots__ = tuple()

    def __contains__(self, val):
        return (val in self._sets[0]) ^ (val in self._sets[1])


class SetSymmetricDifference_FiniteSet(_FiniteSetMixin,
                                        SetSymmetricDifference_InfiniteSet):
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


class SetSymmetricDifference_OrderedSet(_OrderedSetMixin,
                                         SetSymmetricDifference_FiniteSet):
    __slots__ = tuple()

    def __getitem__(self, index):
        idx = self._to_0_based_index(index)
        _iter = iter(self)
        try:
            while idx:
                next(_iter)
                idx -= 1
            return next(_iter)
        except StopIteration:
            raise IndexError("%s index out of range" % (self.name,))

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

class SetProduct(_SetOperator):
    __slots__ = tuple()

    _operator = "*"

    def __new__(cls, *args):
        if cls != SetProduct:
            return super(SetProduct, cls).__new__(cls)

        _sets = _SetOperator._checkArgs(*args)
        if all(_[0] for _ in _sets):
            cls = SetProduct_OrderedSet
        elif all(_[1] for _ in _sets):
            cls = SetProduct_FiniteSet
        else:
            cls = SetProduct_InfiniteSet
        return cls.__new__(cls)

    def flatten_cross_product(self):
        # This is recursive, but the chances of a deeply nested product
        # of Sets is exceptionally low.
        for s in self._sets:
            if isinstance(s, SetProduct):
                for ss in s.flatten_cross_product():
                    yield ss
            else:
                yield s

    def ranges(self):
        yield RangeProduct(list(
            list(_.ranges()) for _ in self.flatten_cross_product()
        ))

    def bounds(self):
        return ( tuple(_.bounds()[0] for _ in self.flatten_cross_product()),
                 tuple(_.bounds()[1] for _ in self.flatten_cross_product()) )

    @property
    def dimen(self):
        if not (FLATTEN_CROSS_PRODUCT and normalize_index.flatten):
            return None
        # By convention, "None" trumps UnknownSetDimen.  That is, a set
        # product is "non-dimentioned" if any term is non-dimentioned,
        # even if we do not yet know the dimentionality of another term.
        ans = 0
        _unknown = False
        for s in self._sets:
            s_dim = s.dimen
            if s_dim is None:
                return None
            elif s_dim is UnknownSetDimen:
                _unknown = True
            else:
                ans += s_dim
        return UnknownSetDimen if _unknown else ans


class SetProduct_InfiniteSet(SetProduct):
    __slots__ = tuple()

    def __contains__(self, val):
        return self._find_val(val) is not None

    def _find_val(self, val):
        """Locate a value in this SetProduct

        Locate a value in this SetProduct.  Returns None if the value is
        not found, otherwise returns a (value, cutpoints) tuple.  Value
        is the value that was searched for, possibly normalized.
        Cutpoints is the set of indices that specify how to split the
        value into the corresponding subsets such that subset[i] =
        cutpoints[i:i+1].  Cutpoints is None if the value is trivially
        split with a single index for each subset.

        Returns
        -------
        val: tuple
        cutpoints: list
        """
        # Support for ambiguous cross products: if val matches the
        # number of subsets, we will start by checking each value
        # against the corresponding subset.  Failure is not sufficient
        # to determine the val is not in this set.
        if hasattr(val, '__len__') and len(val) == len(self._sets):
            if all(v in self._sets[i] for i,v in enumerate(val)):
                return val, None

        # If we are not normalizing indices, then if the above did not
        # match, we will NOT attempt to guess how to split the indices
        if not normalize_index.flatten:
            return None

        val = normalize_index(val)
        if val.__class__ is tuple:
            v_len = len(val)
        else:
            val = (val,)
            v_len = 1

        # Get the dimentionality of all the component sets
        setDims = list(s.dimen for s in self._sets)
        # Find the starting index for each subset (based on dimentionality)
        index = [None]*len(setDims)
        lastIndex = 0
        for i,dim in enumerate(setDims):
            index[i] = lastIndex
            if dim is None:
                firstNonDimSet = i
                break
            lastIndex += dim
            # We can also check for this subset member immediately.
            # Non-membership is sufficient to return "not found"
            if lastIndex > v_len:
                return None
            elif val[index[i]:lastIndex] not in self._sets[i]:
                return None
        # The end of the last subset is always the length of the val
        index.append(v_len)

        # If there were no non-dimentioned sets, then we have checked
        # each subset, found a match, and can reach a verdict:
        if None not in setDims:
            return val, index

        # If a subset is non-dimentioned, then we will have broken out
        # of the forward loop early.  Start at the end and work
        # backwards.
        lastIndex = index[-1]
        for iEnd,dim in enumerate(reversed(setDims)):
            i = len(setDims)-(iEnd+1)
            if dim is None:
                lastNonDimSet = i
                break
            lastIndex -= dim
            index[i] = lastIndex
            # We can also check for this subset member immediately.
            # Non-membership is sufficient to return "not found"
            if val[index[i]:index[i+1]] not in self._sets[i]:
                return None

        if firstNonDimSet == lastNonDimSet:
            # We have inferred the subpart of val that must be in the
            # (single) non-dimentioned subset.  Check membership and
            # return the final verdict.
            if ( val[index[firstNonDimSet]:index[firstNonDimSet+1]]
                 in self._sets[firstNonDimSet] ):
                return val, index
            else:
                return None

        # There were multiple subsets with dimen==None.  The only thing
        # we can do at this point is to search for any possible
        # combination that works

        subsets = self._sets[firstNonDimSet:lastNonDimSet+1]
        _val = val[index[firstNonDimSet]:index[lastNonDimSet+1]]
        for cuts in self._cutPointGenerator(subsets, len(_val)):
            if all(_val[cuts[i]:cuts[i+1]] in s for i,s in enumerate(subsets)):
                offset = index[firstNonDimSet]
                for i in xrange(1,len(subsets)):
                    index[firstNonDimSet+i] = offset + cuts[i]
                return val, index
        return None


    @staticmethod
    def _cutPointGenerator(subsets, val_len):
        """Generate the sequence of cut points for a series of subsets.

        This generator produces the valid set of cut points for
        separating a list of length val_len into chunks that are valid
        for the specified subsets.  In this method, the first and last
        subsets must have dimen==None.  The return value is a list with
        length one greater that then number of subsets.  Value slices
        (for membership tests) are determined by

            cuts[i]:cuts[i+1] in subsets[i]

        """
        setDims = list(_.dimen for _ in subsets)
        cutIters = [None] * (len(subsets)+1)
        cutPoints = [0] * (len(subsets)+1)
        i = 1
        cutIters[i] = iter(xrange(val_len+1))
        cutPoints[-1] = val_len
        while i > 0:
            try:
                cutPoints[i] = next(cutIters[i])
                if i < len(subsets)-1:
                    if setDims[i] is not None:
                        cutIters[i+1] = iter((cutPoints[i]+setDims[i],))
                    else:
                        cutIters[i+1] = iter(xrange(cutPoints[i], val_len+1))
                    i += 1
                elif cutPoints[i] > val_len:
                    i -= 1
                else:
                    yield cutPoints
            except StopIteration:
                i -= 1



class SetProduct_FiniteSet(_FiniteSetMixin, SetProduct_InfiniteSet):
    __slots__ = tuple()

    def __iter__(self):
        _iter = itertools.product(*self._sets)
        # Note: if all the member sets are simple 1-d sets, then there
        # is no need to call flatten_tuple.
        if FLATTEN_CROSS_PRODUCT and normalize_index.flatten \
           and self.dimen != len(self._sets):
            return (flatten_tuple(_) for _ in _iter)
        return _iter

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        ans = 1
        for s in self._sets:
            ans *= max(1, len(s))
        return ans


class SetProduct_OrderedSet(_OrderedSetMixin, SetProduct_FiniteSet):
    __slots__ = tuple()

    def __getitem__(self, index):
        _idx = self._to_0_based_index(index)
        _ord = list(len(_) for _ in self._sets)
        i = len(_ord)
        while i:
            i -= 1
            _ord[i], _idx = _idx % _ord[i], _idx // _ord[i]
        if _idx:
            raise IndexError("%s index out of range" % (self.name,))
        ans = tuple(s[i+1] for s,i in zip(self._sets, _ord))
        if FLATTEN_CROSS_PRODUCT and normalize_index.flatten \
           and self.dimen != len(ans):
            return flatten_tuple(ans)
        return ans

    def ord(self, item):
        """
        Return the position index of the input value.

        Note that Pyomo Set objects have positions starting at 1 (not 0).

        If the search item is not in the Set, then an IndexError is raised.
        """
        found = self._find_val(item)
        if found is None:
            raise IndexError(
                "Cannot identify position of %s in Set %s: item not in Set"
                % (item, self.name))
        val, cutPoints = found
        if cutPoints is not None:
            val = tuple( val[cutPoints[i]:cutPoints[i+1]]
                          for i in xrange(len(self._sets)) )
        _idx = tuple(s.ord(val[i])-1 for i,s in enumerate(self._sets))
        _len = list(len(_) for _ in self._sets)
        _len.append(1)
        ans = 0
        for pos, n in zip(_idx, _len[1:]):
            ans += pos
            ans *= n
        return ans+1

############################################################################

class _AnySet(_SetData, Set):
    def __init__(self, **kwds):
        _SetData.__init__(self, component=self)
        kwds.setdefault('domain', self)
        Set.__init__(self, **kwds)

    def __contains__(self, val):
        return True

    def ranges(self):
        yield AnyRange()

    def bounds(self):
        return (None, None)

    @property
    def dimen(self):
        return None


def DeclareGlobalSet(obj):
    """Declare a copy of a set as a global set in the calling module

    This takes a Set object and declares a duplicate of it as a
    GlobalSet object in the global namespace of the caller's module
    using the local name of the passed set.  GlobalSet objects are
    pseudo-singletons, in that copy.deepcopy (and Model.clone()) will
    not duplcicate them, and when you pickle and restore objects
    containing GlobalSets will still refer to the same object.  The
    declaed GlobalSet object will be an instance of the original Set
    type.

    """
    obj.construct()
    class GlobalSet(obj.__class__):
        __doc__ = """%s

        References to this object will not be duplicated by deepcopy
        and be maintained/restored by pickle.

        """ % (obj.doc,)
        # Note: a simple docstring does not appear to be picked up (at
        # least in Python 2.7, so we will explicitly set the __doc__
        # attribute.

        __slots__ = ()

        def __init__(self, _obj):
            _obj.__class__.__setstate__(self, _obj.__getstate__())
            self._component = weakref.ref(self)
            self.construct()
            assert _obj.parent_component() is _obj
            assert _obj.parent_block() is None
            caller_globals = inspect.stack()[1][0].f_globals
            assert self.local_name not in caller_globals
            caller_globals[self.local_name] = self

        def __reduce__(self):
            # Cause pickle to preserve references to this object
            return self.name

        def __deepcopy__(self, memo):
            # Prevent deepcopy from duplicating this object
            return self

        def __str__(self):
            # Override str() to always print out the global set name
            return self.name

    return GlobalSet(obj)


DeclareGlobalSet(_AnySet(
    name='Any',
    doc="A global Pyomo Set that admits any value",
))

DeclareGlobalSet(RangeSet(
    name='Reals',
    doc='A global Pyomo Set that admits any real (floating point) value',
    ranges=(NumericRange(None,None,0),),
))
DeclareGlobalSet(RangeSet(
    name='NonNegativeReals',
    doc='A global Pyomo Set admitting any real value in [0, +inf]',
    ranges=(NumericRange(0,None,0),),
))
DeclareGlobalSet(RangeSet(
    name='NonPositiveReals',
    doc='A global Pyomo Set admitting any real value in [-inf, 0]',
    ranges=(NumericRange(None,0,0),),
))
DeclareGlobalSet(RangeSet(
    name='NegativeReals',
    doc='A global Pyomo Set admitting any real value in [-inf, 0)',
    ranges=(NumericRange(None,0,0,(True,False)),),
))
DeclareGlobalSet(RangeSet(
    name='PositiveReals',
    doc='A global Pyomo Set admitting any real value in (0, +inf]',
    ranges=(NumericRange(0,None,0,(False,True)),),
))

DeclareGlobalSet(RangeSet(
    name='Integers',
    doc='A global Pyomo Set admitting any integer value',
    ranges=(NumericRange(0,None,1), NumericRange(0,None,-1)),
))
DeclareGlobalSet(RangeSet(
    name='NonNegativeIntegers',
    doc='A global Pyomo Set admitting any integer value in [0, +inf]',
    ranges=(NumericRange(0,None,1),),
))
DeclareGlobalSet(RangeSet(
    name='NonPositiveIntegers',
    doc='A global Pyomo Set admitting any integer value in [-inf, 0]',
    ranges=(NumericRange(0,None,-1),),
))
DeclareGlobalSet(RangeSet(
    name='NegativeIntegers',
    doc='A global Pyomo Set admitting any integer value in [-inf, -1]',
    ranges=(NumericRange(-1,None,-1),),
))
DeclareGlobalSet(RangeSet(
    name='PositiveIntegers',
    doc='A global Pyomo Set admitting any integer value in [1, +inf]',
    ranges=(NumericRange(1,None,1),),
))

DeclareGlobalSet(RangeSet(
    name='Binary',
    doc='A global Pyomo Set admitting the integers {0, 1}',
    ranges=(NumericRange(0,1,1),),
))

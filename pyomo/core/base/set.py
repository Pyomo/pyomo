#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import inspect
import itertools
import logging
import math
import six
import sys
import weakref

from six import iteritems
from six.moves import xrange

from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
    native_types, native_numeric_types, as_numeric, value,
)
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.util import (
    disable_methods, InitializerBase, Initializer, 
    CountedCallInitializer, IndexedCallInitializer,
)
from pyomo.core.base.range import (
    NumericRange, NonNumericRange, AnyRange, RangeProduct,
    RangeDifferenceError,
)
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set, normalize_index,
)
from pyomo.core.base.global_set import (
    GlobalSets, GlobalSetBase,
)
from pyomo.core.base.misc import sorted_robust

if six.PY3:
    from collections.abc import Sequence as collections_Sequence
    def formatargspec(fn):
        return str(inspect.signature(fn))
else:
    from collections import Sequence as collections_Sequence
    def formatargspec(fn):
        return str(inspect.formatargspec(*inspect.getargspec(fn)))


logger = logging.getLogger('pyomo.core')

_prePython37 = sys.version_info[:2] < (3,7)

_inf = float('inf')

FLATTEN_CROSS_PRODUCT = True

"""Set objects

Pyomo `Set` objects are designed to be "API-compatible" with Python
`set` objects.  However, not all Set objects implement the full `set`
API (e.g., only finite discrete Sets support `add()`).

All Sets implement one of the following APIs:

0. `class _SetDataBase(ComponentData)`
   *(pure virtual interface)*

1. `class _SetData(_SetDataBase)`
   *(base class for all AML Sets)*

2. `class _FiniteSetMixin(object)`
   *(pure virtual interface, adds support for discrete/iterable sets)*

4. `class _OrderedSetMixin(object)`
   *(pure virtual interface, adds support for ordered Sets)*

This is a bit of a change from python set objects.  First, the
lowest-level (non-abstract) Data object supports infinite sets; that is,
sets that contain an infinite number of values (this includes both
bounded continuous ranges as well as unbounded discrete ranges).  As
there are an infinite number of values, iteration is *not*
supported. The base class also implements all Python set operations.
Note that `_SetData` does *not* implement `len()`, as Python requires
`len()` to return a positive integer.

Finite sets add iteration and support for `len()`.  In addition, they
support access to members through three methods: `data()` returns the
members as a tuple (in the internal storage order), and may not be
deterministic.  `ordered_data()` returns the members, and is guaranteed
to be in a deterministic order (in the case of insertion order sets, up
to the determinism of the script that populated the set).  Finally,
`sorted_data()` returns the members in a sorted order (guaranteed
deterministic, up to the implementation of < and ==).

..TODO: should these three members all return generators?  This would
further change the implementation of `data()`, but would allow consumers
to potentially access the members in a more efficient manner.

Ordered sets add support for `ord()` and `__getitem__`, as well as the
`first`, `last`, `next` and `prev` methods for stepping over set
members.

Note that the base APIs are all declared (and to the extent possible,
implemented) through Mixin classes.
"""

def process_setarg(arg):
    if isinstance(arg, _SetDataBase):
        return arg
    elif isinstance(arg, IndexedComponent):
        raise TypeError("Cannot apply a Set operator to an "
                        "indexed %s component (%s)"
                        % (arg.ctype.__name__, arg.name,))
    elif isinstance(arg, Component):
        raise TypeError("Cannot apply a Set operator to a non-Set "
                        "%s component (%s)"
                        % (arg.__class__.__name__, arg.name,))
    elif isinstance(arg, ComponentData):
        raise TypeError("Cannot apply a Set operator to a non-Set "
                        "component data (%s)" % (arg.name,))

    # DEPRECATED: This functionality has never been documented,
    # and I don't know of a use of it in the wild.
    if hasattr(arg, 'set_options'):
        deprecation_warning("The set_options set attribute is deprecated.  "
                            "Please explicitly construct complex sets",
                            version='5.7.3')
        # If the argument has a set_options attribute, then use
        # it to initialize a set
        args = arg.set_options
        args.setdefault('initialize', arg)
        args.setdefault('ordered', type(arg) not in Set._UnorderedInitializers)
        ans = Set(**args)

        _init = args['initialize']
        if not ( inspect.isgenerator(_init)
                 or inspect.isfunction(_init)
                 or ( isinstance(_init, ComponentData)
                      and not _init.parent_component().is_constructed() )):
            ans.construct()
        return ans

    # TBD: should lists/tuples be copied into Sets, or
    # should we preserve the reference using SetOf?
    # Historical behavior is to *copy* into a Set.
    #
    # ans.append(Set(initialize=arg,
    #               ordered=type(arg) in {tuple, list}))
    # ans.construct()
    #
    # But this causes problems, especially because Set()'s constructor
    # needs to know if the object is ordered (Set defaults to ordered,
    # and will toss a warning if the underlying data source is not
    # ordered)).  While we could add checks where we create the Set
    # (like here and in the __r*__ operators) and pass in a reasonable
    # value for ordered, it is starting to make more sense to use SetOf
    # (which has that logic).  Alternatively, we could use SetOf to
    # create the Set:
    #
    _defer_construct = False
    if inspect.isgenerator(arg):
        _ordered = True
        _defer_construct = True
    elif inspect.isfunction(arg):
        _ordered = True
        _defer_construct = True
    else:
        arg = SetOf(arg)
        _ordered = arg.isordered()

    ans = Set(initialize=arg, ordered=_ordered)
    #
    # Because the resulting set will be attached to the model (at least
    # for the time being), we will NOT construct it here unless the data
    # is already determined (either statically provided, or through an
    # already-constructed component).
    #
    if not _defer_construct:
        ans.construct()
    #
    # Or we can do the simple thing and just use SetOf:
    #
    # ans = SetOf(arg)
    return ans


@deprecated('The set_options decorator is deprecated; create Sets from '
            'functions explicitly by passing the function to the Set '
            'constructor using the "initialize=" keyword argument.',
            version='5.7')
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

    # Because some of our processing of initializer functions relies on
    # knowing the number of positional arguments, we will go to extra
    # effort here to preserve the original function signature.
    _funcdef = """def wrapper_function%s:
        args, varargs, kwds, local_env = inspect.getargvalues(
            inspect.currentframe())
        args = tuple(local_env[_] for _ in args) + (varargs or ())
        value = fn(*args, **(kwds or {}))
        # Map None -> Set.End
        if value is None:
            return Set.End
        return value
""" % (formatargspec(fn),)
    # Create the wrapper in a temporary environment that mimics this
    # function's environment.
    _env = dict(globals())
    _env.update(locals())
    exec(_funcdef, _env)
    return _env['wrapper_function']


class UnknownSetDimen(object): pass

class SetInitializer(InitializerBase):
    """An Initializer wrapper for returning Set objects

    This initializer wraps another Initializer and converts the return
    value to a proper Pyomo Set.  If the initializer is None, then Any
    is returned.  This initializer can be 'intersected' with another
    initializer to return the SetIntersect of the Sets returned by the
    initializers.

    """
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
            return process_setarg(self._set(parent, idx))

    def constant(self):
        return self._set is None or self._set.constant()

    def contains_indices(self):
        return self._set is not None and self._set.contains_indices()

    def indices(self):
        if self._set is not None:
            return self._set.indices()
        else:
            super(SetInitializer, self).indices()

    def setdefault(self, val):
        if self._set is None:
            self._set = Initializer(val)

class SetIntersectInitializer(InitializerBase):
    """An Initializer that returns the intersection of two SetInitializers

    Users will typically not create a SetIntersectInitializer directly.
    Instead, SetInitializer.intersect() may return a SetInitializer that
    contains a SetIntersectInitializer instance.

    """
    __slots__ = ('_A','_B',)
    def __init__(self, setA, setB):
        self._A = setA
        self._B = setB

    def __call__(self, parent, idx):
        return SetIntersection(self._A(parent, idx), self._B(parent, idx))

    def constant(self):
        return self._A.constant() and self._B.constant()

    def contains_indices(self):
        return self._A.contains_indices() or self._B.contains_indices()

    def indices(self):
        if self._A.contains_indices():
            if self._B.contains_indices():
                if set(self._A.indices()) != set (self._B.indices()):
                    raise ValueError(
                        "SetIntersectInitializer contains two "
                        "sub-initializers with inconsistent external indices")
            return self._A.indices()
        else:
            # It is OK (and desirable) for this to raise the exception
            # if B does not contain external indices
            return self._B.indices()

class BoundsInitializer(InitializerBase):
    """An Initializer wrapper that converts bounds information to a RangeSet

    The BoundsInitializer wraps another initializer that is expected to
    return valid arguments to the RangeSet constructor.  Nominally, this
    would be bounds information in the form of (lower bound, upper
    bound), but could also be a single scalar or a 3-tuple.  Calling
    this initializer will return a RangeSet object.

    BoundsInitializer objects can be intersected with other
    SetInitializer objects using the SetInitializer.intersect() method.

    """
    __slots__ = ('_init', 'default_step',)
    def __init__(self, init, default_step=0):
        self._init = Initializer(init, treat_sequences_as_mappings=False)
        self.default_step = default_step

    def __call__(self, parent, idx):
        val = self._init(parent, idx)
        if not isinstance(val, collections_Sequence):
            val = (1, val, self.default_step)
        else:
            val = tuple(val)
            if len(val) == 2:
                val += (self.default_step,)
            elif len(val) == 1:
                val = (1, val[0], self.default_step)
            elif len(val) == 0:
                val = (None, None, self.default_step)
        ans = RangeSet(*tuple(val))
        # We don't need to construct here, as the RangeSet will
        # automatically construct itself if it can
        #ans.construct()
        return ans

    def constant(self):
        return self._init.constant()

    def setdefault(self, val):
        # This is a real range set... there is no default to set
        pass

class TuplizeError(PyomoException):
    pass

class TuplizeValuesInitializer(InitializerBase):
    """An initializer wrapper that will "tuplize" a sequence

    This initializer takes the result of another initializer, and if it
    is a sequence that does not already contain tuples, wil convert it
    to a sequence of tuples, each of length 'dimen' before returning it.

    """
    __slots__ = ('_init', '_dimen')

    def __new__(cls, *args):
        if args == (None,):
            return None
        else:
            return super(TuplizeValuesInitializer, cls).__new__(cls)

    def __init__(self, _init):
        self._init = _init
        self._dimen = UnknownSetDimen

    def __call__(self, parent, index):
        _val = self._init(parent, index)
        if self._dimen in {1, None, UnknownSetDimen}:
            return _val
        elif _val is Set.Skip:
            return _val
        elif _val is None:
            return _val

        if not isinstance(_val, collections_Sequence):
            _val = tuple(_val)
        if len(_val) == 0:
            return _val
        if isinstance(_val[0], tuple):
            return _val
        return self._tuplize(_val, parent, index)

    def constant(self):
        return self._init.constant()

    def contains_indices(self):
        return self._init.contains_indices()

    def indices(self):
        return self._init.indices()

    def _tuplize(self, _val, parent, index):
        d = self._dimen
        if len(_val) % d:
            raise TuplizeError(
                "Cannot tuplize list data for set %%s%%s because its "
                "length %s is not a multiple of dimen=%s" % (len(_val), d))

        return list(tuple(_val[d*i:d*(i+1)]) for i in xrange(len(_val)//d))


class _NotFound(object):
    "Internal type flag used to indicate if an object is not found in a set"
    pass


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
        try:
            ans = self.get(value, _NotFound)
        except TypeError:
            # In Python 3.x, Sets are unhashable
            if isinstance(value, _SetData):
                ans = _NotFound
            else:
                raise

        if ans is _NotFound:
            if isinstance(value, _SetData):
                deprecation_warning(
                    "Testing for set subsets with 'a in b' is deprecated.  "
                    "Use 'a.issubset(b)'.", version='5.7')
                return value.issubset(self)
            else:
                return False
        return True

    def get(self, value, default=None):
        raise DeveloperError("Derived set class (%s) failed to "
                             "implement get()" % (type(self).__name__,))

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return False

    def isfinite(self):
        """Returns True if this is a finite discrete (iterable) Set"""
        return False

    def isordered(self):
        """Returns True if this is an ordered finite discrete (iterable) Set"""
        return False

    def subsets(self, expand_all_set_operators=None):
        return iter((self,))

    def __iter__(self):
        """Iterate over the set members

        Raises AttributeError for non-finite sets.  This must be
        declared for non-finite sets because scalar sets inherit from
        IndexedComponent, which provides an iterator (over the
        underlying indexing set).
        """
        raise TypeError(
            "'%s' object is not iterable (non-finite Set '%s' "
            "is not iterable)" % (self.__class__.__name__, self.name))

    def __eq__(self, other):
        if self is other:
            return True
        # Special case: non-finite range sets that only contain finite
        # ranges (or no ranges).  We will re-generate non-finite sets to
        # make sure we get an accurate "finiteness" flag.
        if hasattr(other, 'isfinite'):
            other_isfinite = other.isfinite()
            if not other_isfinite:
                try:
                    other = RangeSet(ranges=list(other.ranges()))
                    other_isfinite = other.isfinite()
                except TypeError:
                    pass
        elif hasattr(other, '__contains__'):
            # we assume that everything that does not implement
            # isfinite() is a discrete set.
            other_isfinite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
            except:
                pass
        else:
            return False
        if not self.isfinite():
            try:
                self = RangeSet(ranges=list(self.ranges()))
            except TypeError:
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

    @property
    def domain(self):
        raise DeveloperError("Derived set class (%s) failed to "
                             "implement domain" % (type(self).__name__,))

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
        if lb is not None:
            if int(lb) == lb:
                lb = int(lb)
        if ub is not None:
            if int(ub) == ub:
                ub = int(ub)
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
        # routine nondeterministic.  Not a huge issue for the result,
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
    @deprecated("The 'virtual' attribute is no longer supported", version='5.7')
    def virtual(self):
        return isinstance(self, (_AnySet, SetOperator, _InfiniteRangeSetData))

    @virtual.setter
    def virtual(self, value):
        if value != self.virtual:
            raise ValueError(
                "Attempting to set the (deprecated) 'virtual' attribute on %s "
                "to an invalid value (%s)" % (self.name, value))

    @property
    @deprecated("The 'concrete' attribute is no longer supported.  "
                "Use isdiscrete() or isfinite()", version='5.7')
    def concrete(self):
        return self.isfinite()

    @concrete.setter
    def concrete(self, value):
        if value != self.concrete:
            raise ValueError(
                "Attempting to set the (deprecated) 'concrete' attribute on %s "
                "to an invalid value (%s)" % (self.name, value))

    @property
    @deprecated("The 'ordered' attribute is no longer supported.  "
                "Use isordered()", version='5.7')
    def ordered(self):
        return self.isordered()

    @property
    @deprecated("'filter' is no longer a public attribute.",
                version='5.7')
    def filter(self):
        return None

    @deprecated("check_values() is deprecated: Sets only contain valid members",
                version='5.7')
    def check_values(self):
        """
        Verify that the values in this set are valid.
        """
        return True

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
        if hasattr(other, 'isfinite'):
            other_isfinite = other.isfinite()
        elif hasattr(other, '__contains__'):
            # we assume that everything that does not implement
            # isfinite() is a discrete set.
            other_isfinite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
            except:
                pass
        else:
            # Raise an exception consistent with Python's set.isdisjoint()
            raise TypeError(
                "'%s' object is not iterable" % (type(other).__name__,))
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
        # Special case: non-finite range sets that only contain finite
        # ranges (or no ranges).  We will re-generate non-finite sets to
        # make sure we get an accurate "finiteness" flag.
        if hasattr(other, 'isfinite'):
            other_isfinite = other.isfinite()
            if not other_isfinite:
                try:
                    other = RangeSet(ranges=list(other.ranges()))
                    other_isfinite = other.isfinite()
                except TypeError:
                    pass
        elif hasattr(other, '__contains__'):
            # we assume that everything that does not implement
            # isfinite() is a discrete set.
            other_isfinite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
            except:
                pass
        else:
            # Raise an exception consistent with Python's set.issubset()
            raise TypeError(
                "'%s' object is not iterable" % (type(other).__name__,))
        if not self.isfinite():
            try:
                self = RangeSet(ranges=list(self.ranges()))
            except TypeError:
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
        """Test if this Set is a superset of `other`

        Parameters
        ----------
            other : ``Set`` or ``iterable``
                The Set or iterable object to compare this Set against

        Returns
        -------
        bool : True if this set is a superset of `other`
        """
        # Special case: non-finite range sets that only contain finite
        # ranges (or no ranges).  We will re-generate non-finite sets to
        # make sure we get an accurate "finiteness" flag.
        if hasattr(other, 'isfinite'):
            other_isfinite = other.isfinite()
            if not other_isfinite:
                try:
                    other = RangeSet(ranges=list(other.ranges()))
                    other_isfinite = other.isfinite()
                except TypeError:
                    pass
        elif hasattr(other, '__contains__'):
            # we assume that everything that does not implement
            # isfinite() is a discrete set.
            other_isfinite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
            except:
                pass
        else:
            # Raise an exception consistent with Python's set.issuperset()
            raise TypeError(
                "'%s' object is not iterable" % (type(other).__name__,))
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
        if not self.isfinite():
            try:
                self = RangeSet(ranges=list(self.ranges()))
            except TypeError:
                pass
        if self.isfinite():
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

    def _iter_impl(self):
        raise DeveloperError("Derived finite set class (%s) failed to "
                             "implement _iter_impl" % (type(self).__name__,))

    def __iter__(self):
        """Iterate over the finite set

        Note: derived classes should NOT reimplement this method, and
        should instead overload _iter_impl.  The expression template
        system relies on being able to replace this method for all Sets
        during template generation.

        """
        return self._iter_impl()

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

    @property
    @deprecated("The 'value' attribute is deprecated.  Use .data() to "
                "retrieve the values in a finite set.", version='5.7')
    def value(self):
        return set(self)

    @property
    @deprecated("The 'value_list' attribute is deprecated.  Use "
                ".ordered_data() to retrieve the values from a finite set "
                "in a deterministic order.", version='5.7')
    def value_list(self):
        return list(self.ordered_data())

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
        self._domain = Any
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

    def get(self, value, default=None):
        """
        Return True if the set contains a given value.

        This method will raise TypeError for unhashable types.
        """
        if normalize_index.flatten:
            value = normalize_index(value)

        if value in self._values:
            return value
        return default

    def _iter_impl(self):
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
        if self._dimen is UnknownSetDimen:
            # Special case: abstract Sets with constant dimen
            # initializers have a known dimen before construction
            _comp = self.parent_component()
            if not _comp._constructed and _comp._init_dimen.constant():
                return _comp._init_dimen.val
        return self._dimen

    @property
    def domain(self):
        return self._domain

    @property
    @deprecated("'filter' is no longer a public attribute.",
                version='5.7')
    def filter(self):
        return self._filter

    def add(self, *values):
        count = 0
        _block = self.parent_block()
        for value in values:
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

            # We wrap this check in a try-except because some values
            #  (like lists) are not hashable and can raise exceptions.
            try:
                if _value in self:
                    logger.warning(
                        "Element %s already exists in Set %s; no action taken"
                        % (value, self.name))
                    continue
            except:
                exc = sys.exc_info()
                raise TypeError("Unable to insert '%s' into Set %s:\n\t%s: %s"
                                % (value, self.name, exc[0].__name__, exc[1]))

            if self._filter is not None:
                if not self._filter(_block, _value):
                    continue

            if self._validate is not None:
                try:
                    flag = self._validate(_block, _value)
                except:
                    logger.error(
                        "Exception raised while validating element '%s' "
                        "for Set %s" % (value, self.name))
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
                            "The value=%s has dimension %s and is not "
                            "valid for Set %s which has dimen=%s"
                            % (value, _d, self.name, self._dimen))

            # Add the value to this object (this last redirection allows
            # derived classes to implement a different storage mechanism)
            self._add_impl(_value)
            count += 1
        return count

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

    def _iter_impl(self):
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
        if type(val) in Set._UnorderedInitializers:
            logger.warning(
                "Calling set_value() on an insertion order Set with "
                "a fundamentally unordered data source (type: %s).  "
                "This WILL potentially lead to nondeterministic behavior "
                "in Pyomo" % (type(val).__name__,))
        super(_InsertionOrderSetData, self).set_value(val)

    def update(self, values):
        if type(values) in Set._UnorderedInitializers:
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

    def _iter_impl(self):
        """
        Return an iterator for the set.
        """
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self)._iter_impl()

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
        self._values = {j:i for i, j in enumerate(self._ordered_values)}
        self._is_sorted = True


############################################################################

_SET_API = (
    ('__contains__', 'test membership in'),
    'get', 'ranges', 'bounds',
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


@ModelComponentFactory.register(
    "Set data that is used to define a model instance.")
class Set(IndexedComponent):
    """A component used to index other Pyomo components.

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
        constructed.  Values passed to ``initialize`` may be
        overridden by ``data`` passed to the :py:meth:`construct`
        method.

    dimen : initializer(int), optional
        Specify the Set's arity (the required tuple length for all
        members of the Set), or None if no arity is enforced

    ordered : bool or Set.InsertionOrder or Set.SortedOrder or function
        Specifies whether the set is ordered.
        Possible values are:

          ======================  =====================================
          ``False``               Unordered
          ``True``                Ordered by insertion order
          ``Set.InsertionOrder``  Ordered by insertion order [default]
          ``Set.SortedOrder``     Ordered by sort order
          ``<function>``          Ordered with this comparison function
          ======================  =====================================

    within : initialiser(set), optional
        A set that defines the valid values that can be contained
        in this set
    domain : initializer(set), optional
        A set that defines the valid values that can be contained
        in this set
    bounds : initializer(tuple), optional
        A tuple that specifies the bounds for valid Set values
        (accepts 1-, 2-, or 3-tuple RangeSet arguments)
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
      .. note::

        ``domain=``, ``within=``, and ``bounds=`` all provide
        restrictions on the valid set values.  If more than one is
        specified, Set values will be restricted to the intersection of
        ``domain``, ``within``, and ``bounds``.

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
            self._init_domain.intersect(BoundsInitializer(_bounds))

        self._init_dimen = Initializer(
            kwds.pop('dimen', UnknownSetDimen),
            arg_not_specified=UnknownSetDimen)
        self._init_values = TuplizeValuesInitializer(Initializer(
            kwds.pop('initialize', None),
            treat_sequences_as_mappings=False, allow_generators=True))
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
        if self._init_values is not None \
           and self._init_values._init.__class__ is IndexedCallInitializer:
            self._init_values._init = CountedCallInitializer(
                self, self._init_values._init)
        # HACK: the DAT parser needs to know the domain of a set in
        # order to correctly parse the data stream.
        if not self.is_indexed():
            if self._init_domain.constant():
                self._domain = self._init_domain(self.parent_block(), None)
            if self._init_dimen.constant():
                self._dimen = self._init_dimen(self.parent_block(), None)


    @deprecated("check_values() is deprecated: Sets only contain valid members",
                version='5.7')
    def check_values(self):
        """
        Verify that the values in this set are valid.
        """
        return True


    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if is_debug_set(logger):
                logger.debug("Constructing Set, name=%s, from data=%r"
                             % (self.name, data))
        self._constructed = True
        if data is not None:
            # Data supplied to construct() should override data provided
            # to the constructor
            tmp_init, self._init_values \
                = self._init_values, TuplizeValuesInitializer(
                    Initializer(data, treat_sequences_as_mappings=False))
        try:
            if self._init_values is None:
                if not self.is_indexed():
                    # This ensures backwards compatibility by causing all
                    # scalar sets (including set operators) to be
                    # initialized (and potentially empty) after construct().
                    self._getitem_when_not_present(None)
            elif self._init_values.contains_indices():
                # The index is coming in externally; we need to validate it
                for index in self._init_values.indices():
                    IndexedComponent.__getitem__(self, index)
            else:
                # Bypass the index validation and create the member directly
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
        # Because we allow sets within an IndexedSet to have different
        # dimen, we have moved the tuplization logic from PyomoModel
        # into Set (because we cannot know the dimen of a _SetData until
        # we are actually constructing that index).  This also means
        # that we need to potentially communicate the dimen to the
        # (wrapped) value initializer.  So, we will get the dimen first,
        # then get the values.  Only then will we know that this index
        # will actually be constructed (and not Skipped).
        _block = self.parent_block()

        #Note: _init_dimen and _init_domain are guaranteed to be non-None
        _d = self._init_dimen(_block, index)
        if ( not normalize_index.flatten and _d is not UnknownSetDimen
             and _d is not None ):
            logger.warning(
                "Ignoring non-None dimen (%s) for set %s%s "
                "(normalize_index.flatten is False, so dimen "
                "verification is not available)." % (
                    _d, self.name,
                    ("[%s]" % (index,) if self.is_indexed() else "") ))
            _d = None

        domain = self._init_domain(_block, index)
        if _d is UnknownSetDimen and domain is not None \
           and domain.dimen is not None:
            _d = domain.dimen

        if self._init_values is not None:
            self._init_values._dimen = _d
            try:
                _values = self._init_values(_block, index)
            except TuplizeError as e:
                raise ValueError( str(e) % (
                    self._name, "[%s]" % index if self.is_indexed() else ""))

            if _values is Set.Skip:
                return
            elif _values is None:
                raise ValueError(
                    "Set rule or initializer returned None instead of Set.Skip")
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        if _d is not UnknownSetDimen:
            obj._dimen = _d
        if domain is not None:
            obj._domain = domain
            domain.parent_component().construct()
        if self._init_validate is not None:
            try:
                obj._validate = Initializer(self._init_validate(_block, index))
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
                _filter = Initializer(self._init_filter(_block, index))
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
                   and type(_values) in Set._UnorderedInitializers:
                logger.warning(
                    "Initializing ordered Set %s with a fundamentally "
                    "unordered data source (type: %s).  This WILL potentially "
                    "lead to nondeterministic behavior in Pyomo"
                    % (self.name, type(_values).__name__,))
            # Special case: set operations that are not first attached
            # to the model must be constructed.
            if isinstance(_values, SetOperator):
                _values.construct()
            try:
                val_iter = iter(_values)
            except TypeError:
                logger.error(
                    "Initializer for Set %s%s returned non-iterable object "
                    "of type %s." % (
                        self.name,
                        ("[%s]" % (index,) if self.is_indexed() else ""),
                        _values if _values.__class__ is type
                        else type(_values).__name__ ))
                raise
            for val in val_iter:
                if val is Set.End:
                    break
                if _filter is None or _filter(_block, val):
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
        if x._domain is x and isinstance(x, SetOperator):
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
    def data(self):
        "Return a dict containing the data() of each Set in this IndexedSet"
        return {k: v.data() for k,v in iteritems(self)}


class FiniteSimpleSet(_FiniteSetData, Set):
    def __init__(self, **kwds):
        _FiniteSetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

class OrderedSimpleSet(_InsertionOrderSetData, Set):
    def __init__(self, **kwds):
        # In case someone inherits from us, we will provide a rational
        # default for the "ordered" flag
        kwds.setdefault('ordered', Set.InsertionOrder)

        _InsertionOrderSetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

class SortedSimpleSet(_SortedSetData, Set):
    def __init__(self, **kwds):
        # In case someone inherits from us, we will provide a rational
        # default for the "ordered" flag
        kwds.setdefault('ordered', Set.SortedOrder)

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

    def get(self, value, default=None):
        # Note that the efficiency of this depends on the reference object
        #
        # The bulk of single-value set members were stored as scalars.
        # Check that first.
        if value.__class__ is tuple and len(value) == 1:
            if value[0] in self._ref:
                return value[0]
        if value in self._ref:
            return value
        return default

    def __len__(self):
        return len(self._ref)

    def _iter_impl(self):
        return iter(self._ref)

    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        return str(self._ref)

    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if is_debug_set(logger):
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

    @property
    def domain(self):
        return self

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

    def get(self, value, default=None):
        # The bulk of single-value set members were stored as scalars.
        # Check that first.
        if value.__class__ is tuple and len(value) == 1:
            v = value[0]
            if any(v in r for r in self._ranges):
                return v
        if any(value in r for r in self._ranges):
            return value
        return default

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return all(r.isdiscrete() for r in self.ranges())

    @property
    def dimen(self):
        return 1

    @property
    def domain(self):
        return Reals

    def clear(self):
        self._ranges = ()

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

    def _iter_impl(self):
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

    # We must redefine ranges(), bounds(), and domain so that we get the
    # _InfiniteRangeSetData version and not the one from
    # _FiniteSetMixin.
    bounds = _InfiniteRangeSetData.bounds
    ranges = _InfiniteRangeSetData.ranges
    domain = _InfiniteRangeSetData.domain


@ModelComponentFactory.register(
    "A sequence of numeric values.  RangeSet(start,end,step) is a sequence "
    "starting a value 'start', and increasing in values by 'step' until a "
    "value greater than or equal to 'end' is reached.")
class RangeSet(Component):
    """A set object that represents a set of numeric values

    `RangeSet` objects are based around `NumericRange` objects, which
    include support for non-finite ranges (both continuous and
    unbounded). Similarly, boutique ranges (like semi-continuous
    domains) can be represented, e.g.:

    .. doctest::

       >>> from pyomo.core.base.range import NumericRange
       >>> from pyomo.environ import RangeSet
       >>> print(RangeSet(ranges=(NumericRange(0,0,0), NumericRange(1,100,0))))
       ([0] | [1..100])

    The `RangeSet` object continues to support the notation for
    specifying discrete ranges using "[first=1], last, [step=1]" values:

    .. doctest::

        >>> r = RangeSet(3)
        >>> print(r)
        [1:3]
        >>> print(list(r))
        [1, 2, 3]

        >>> r = RangeSet(2, 5)
        >>> print(r)
        [2:5]
        >>> print(list(r))
        [2, 3, 4, 5]

        >>> r = RangeSet(2, 5, 2)
        >>> print(r)
        [2:4:2]
        >>> print(list(r))
        [2, 4]

        >>> r = RangeSet(2.5, 4, 0.5)
        >>> print(r)
        ([2.5] | [3.0] | [3.5] | [4.0])
        >>> print(list(r))
        [2.5, 3.0, 3.5, 4.0]

    By implementing RangeSet using NumericRanges, the global Sets (like
    `Reals`, `Integers`, `PositiveReals`, etc.) are trivial
    instances of a RangeSet and support all Set operations.

    Parameters
    ----------
    *args: int | float | None
        The range defined by ([start=1], end, [step=1]).  If only a
        single positional parameter, `end` is supplied, then the
        RangeSet will be the integers starting at 1 up through and
        including end.  Providing two positional arguments, `x` and `y`,
        will result in a range starting at x up to and including y,
        incrementing by 1.  Providing a 3-tuple enables the
        specification of a step other than 1.

    finite: bool, optional
        This sets if this range is finite (discrete and bounded) or infinite

    ranges: iterable, optional
        The list of range objects that compose this RangeSet

    bounds: tuple, optional
        The lower and upper bounds of values that are admissible in this
        RangeSet

    filter: function, optional
        Function (rule) that returns True if the specified value is in
        the RangeSet or False if it is not.

    validate: function, optional
        Data validation function (rule).  The function will be called
        for every data member of the set, and if it returns False, a
        ValueError will be raised.

    """

    def __new__(cls, *args, **kwds):
        if cls is not RangeSet:
            return super(RangeSet, cls).__new__(cls)

        finite = kwds.pop('finite', None)
        if finite is None:
            if 'ranges' in kwds:
                if any(not r.isfinite() for r in kwds['ranges']):
                    finite = False
            for i,_ in enumerate(args):
                if type(_) not in native_types:
                    # Strange nosetest coverage issue: if the logic is
                    # negated and the continue is in the "else", that
                    # line is not caught as being covered.
                    if not isinstance(_, ComponentData) \
                       or not _.parent_component().is_constructed():
                        continue
                    else:
                        # "Peek" at constructed components to try and
                        # infer if this component will be Infinite
                        _ = value(_)
                if i < 2:
                    if _ in {None, _inf, -_inf}:
                        finite = False
                        break
                elif _ == 0 and args[0] is not args[1]:
                    finite = False
            if finite is None:
                # Assume "undetermined" RangeSets will be finite.  If a
                # user wants them to be infinite, they can always
                # specify finite=False
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
        self._init_validate = Initializer(kwds.pop('validate', None))
        self._init_filter = Initializer(kwds.pop('filter', None))
        self._init_bounds = kwds.pop('bounds', None)
        if self._init_bounds is not None:
            self._init_bounds = BoundsInitializer(self._init_bounds)

        Component.__init__(self, **kwds)
        # Shortcut: if all the relevant construction information is
        # simple (hard-coded) values, then it is safe to go ahead and
        # construct the set.
        #
        # NOTE: We will need to revisit this if we ever allow passing
        # data into the construct method (which would override the
        # hard-coded values here).
        try:
            if all( type(_) in native_types
                    or _.parent_component().is_constructed()
                    for _ in args ):
                self.construct()
        except AttributeError:
            pass


    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        # Unconstructed floating components return their type
        if not self._constructed:
            return type(self).__name__
        # Named, constructed components should return their name e.g., Reals
        if type(self).__name__ != self._name:
            return self.name
        # Floating, unnamed constructed components return their ranges()
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
        if is_debug_set(logger):
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
            if None in args or args[1] - args[0] != -1:
                args = (args[0],args[1],1)

        if len(args) == 3:
            # Discrete ranges anchored by a floating point value or
            # incremented by a floating point value cannot be handled by
            # the NumericRange object.  We will just discretize this
            # range (mostly for backwards compatability)
            start, end, step = args
            if step:
                if start is None:
                    start, end = end, start
                    step *= -1

                if start is None:
                    # Backwards compatability: assume unbounded RangeSet
                    # is grounded at 0
                    ranges += ( NumericRange(0, None, step),
                                NumericRange(0, None, -step) )
                elif int(step) != step:
                    if end is None:
                        raise ValueError(
                            "RangeSet does not support unbounded ranges "
                            "with a non-integer step (got [%s:%s:%s])"
                            % (start, end, step))
                    if (end >= start) ^ (step > 0):
                        raise ValueError(
                            "RangeSet: start, end ordering incompatible with "
                            "step direction (got [%s:%s:%s])"
                            % (start, end, step))
                    n = start
                    i = 0
                    while (step > 0 and n <= end) or (step < 0 and n >= end):
                        ranges += (NumericRange(n,n,0),)
                        i += 1
                        n = start + step*i
                else:
                    ranges += (NumericRange(start, end, step),)
            else:
                ranges += (NumericRange(*args),)

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

        _block = self.parent_block()
        if self._init_bounds is not None:
            bnds = self._init_bounds(_block, None)
            tmp = []
            for r in ranges:
                tmp.extend(r.range_intersection(bnds.ranges()))
            ranges = tuple(tmp)

        self._ranges = ranges

        if self._init_filter is not None:
            if not self.isfinite():
                raise ValueError(
                    "The 'filter' keyword argument is not valid for "
                    "non-finite RangeSet component (%s)" % (self.name,))

            try:
                _filter = Initializer(self._init_filter(_block, None))
                if _filter.constant():
                    # _init_filter was the actual filter function; use it.
                    _filter = self._init_filter
            except:
                # We will assume any exceptions raised when getting the
                # filter for this index indicate that the function
                # should have been passed directly to the underlying sets.
                _filter = self._init_filter

            # If this is a finite set, then we can go ahead and filter
            # all the ranges.  This allows pprint and len to be correct,
            # without special handling
            new_ranges = []
            old_ranges = list(self.ranges())
            old_ranges.reverse()
            while old_ranges:
                r = old_ranges.pop()
                for i,val in enumerate(_FiniteRangeSetData._range_gen(r)):
                    if not _filter(_block, val):
                        split_r = r.range_difference((NumericRange(val,val,0),))
                        if len(split_r) == 2:
                            new_ranges.append(split_r[0])
                            old_ranges.append(split_r[1])
                        elif len(split_r) == 1:
                            if i == 0:
                                old_ranges.append(split_r[0])
                            else:
                                new_ranges.append(split_r[0])
                        i = None
                        break
                if i is not None:
                    new_ranges.append(r)
            self._ranges = new_ranges

        if self._init_validate is not None:
            if not self.isfinite():
                raise ValueError(
                    "The 'validate' keyword argument is not valid for "
                    "non-finite RangeSet component (%s)" % (self.name,))

            try:
                _validate = Initializer(self._init_validate(_block, None))
                if _validate.constant():
                    # _init_validate was the actual validate function; use it.
                    _validate = self._init_validate
            except:
                # We will assume any exceptions raised when getting the
                # validator for this index indicate that the function
                # should have been passed directly to the underlying set.
                _validate = self._init_validate

            for val in self:
                try:
                    flag = _validate(_block, val)
                except:
                    logger.error(
                        "Exception raised while validating element '%s' "
                        "for Set %s" % (val, self.name))
                    raise
                if not flag:
                    raise ValueError(
                        "The value=%s violates the validation rule of "
                        "Set %s" % (val, self.name))

        timer.report()

    #
    # Until the time that we support indexed RangeSet objects, we will
    # mock up some of the IndexedComponent API for consistency with the
    # previous (<=5.6.7) implementation.
    #
    def dim(self):
        return 0
    def index_set(self):
        return UnindexedComponent_set


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

class SetOperator(_SetData, Set):
    __slots__ = ('_sets',)

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
        # We will implicitly construct all set operators if the operands
        # are all constructed.
        if all(_.parent_component()._constructed for _ in self._sets):
            self.construct()

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(SetOperator, self).__getstate__()
        for i in SetOperator.__slots__:
            state[i] = getattr(self, i)
        return state

    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if is_debug_set(logger):
                logger.debug("Constructing SetOperator, name=%s, from data=%r"
                             % (self.name, data))
        for s in self._sets:
            s.parent_component().construct()
        super(SetOperator, self).construct()
        if data:
            deprecation_warning(
                "Providing construction data to SetOperator objects is "
                "deprecated.  This data is ignored and in a future version "
                "will not be allowed", version='5.7')
            fail = len(data) > 1 or None not in data
            if not fail:
                _data = data[None]
                if len(_data) != len(self):
                    fail = True
                else:
                    for v in _data:
                        if v not in self:
                            fail = True
                            break
            if fail:
                raise ValueError(
                    "Constructing SetOperator %s with incompatible data "
                    "(data=%s}" % (self.name, data))
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

    def __deepcopy__(self, memo):
        # SetOperators form an expression system.  As we allow operators
        # on abstract Set objects, it is important to *always* deepcopy
        # SetOperators that have not been assigned to a Block.  For
        # example, consider an abstract indexed model component whose
        # domain is specified by a Set expression:
        #
        #   def x_init(m,i):
        #       if i == 2:
        #           return Set.Skip
        #       else:
        #           return []
        #   m.x = Set( [1,2],
        #              domain={1: m.A*m.B, 2: m.A*m.A},
        #              initialize=x_init )
        #
        # We do not want to automatically add all the Set operators to
        # the model at declaration time, as m.x[2] is never actually
        # created.  Plus, doing so would require complex parsing of the
        # initializers.  BUT, we need to ensure that the operators are
        # deepcopied, otherwise when the model is cloned before
        # construction the operators will still refer to the sets on the
        # original abstract model (in particular, the Set x will have an
        # unknown dimen).
        #
        # Our solution is to cause SetOperators to be automatically
        # cloned if they haven't been assigned to a block.
        if '__block_scope__' in memo:
            if self.parent_block() is None:
                # Hijack the block scope rules to cause this object to
                # be deepcopied.
                memo['__block_scope__'][id(self)] = True
        return super(SetOperator, self).__deepcopy__(memo)

    def _expression_str(self):
        _args = []
        for arg in self._sets:
            arg_str = str(arg)
            if ' ' in arg_str and isinstance(arg, SetOperator):
                arg_str = "(" + arg_str + ")"
            _args.append(arg_str)
        return self._operator.join(_args)

    def isdiscrete(self):
        """Returns True if this set admits only discrete members"""
        return all(r.isdiscrete() for r in self.ranges())

    def subsets(self, expand_all_set_operators=None):
        if not isinstance(self, SetProduct):
            if expand_all_set_operators is None:
                logger.warning("""
                Extracting subsets for Set %s, which is a SetOperator
                other than a SetProduct.  Returning this set and not
                descending into the set operands.  To descend into this
                operator, specify
                'subsets(expand_all_set_operators=True)' or to suppress
                this warning, specify
                'subsets(expand_all_set_operators=False)'""" % ( self.name, ))
                yield self
                return
            elif not expand_all_set_operators:
                yield self
                return
        for s in self._sets:
            for ss in s.subsets(
                    expand_all_set_operators=expand_all_set_operators):
                yield ss

    @property
    @deprecated("SetProduct.set_tuple is deprecated.  "
                "Use SetProduct.subsets() to get the operator arguments.",
                version='5.7')
    def set_tuple(self):
        # Despite its name, in the old SetProduct, set_tuple held a list
        return list(self.subsets())

    @property
    def domain(self):
        return self._domain

    @property
    def _domain(self):
        # We hijack the _domain attribute of SetOperator so that pprint
        # prints out the expression as the Set's "domain".  Doing this
        # as a property prevents the circular reference
        return self

    @_domain.setter
    def _domain(self, val):
        if val is not Any:
            raise ValueError(
                "Setting the domain of a Set Operator is not allowed: %s" % val)


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

class SetUnion(SetOperator):
    __slots__ = tuple()

    _operator = " | "

    def __new__(cls, *args):
        if cls != SetUnion:
            return super(SetUnion, cls).__new__(cls)

        set0, set1 = SetOperator._checkArgs(*args)
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

    def get(self, val, default=None):
        #return any(val in s for s in self._sets)
        for s in self._sets:
            v = s.get(val, default)
            if v is not default:
                return v
        return default


class SetUnion_FiniteSet(_FiniteSetMixin, SetUnion_InfiniteSet):
    __slots__ = tuple()

    def _iter_impl(self):
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

class SetIntersection(SetOperator):
    __slots__ = tuple()

    _operator = " & "

    def __new__(cls, *args):
        if cls != SetIntersection:
            return super(SetIntersection, cls).__new__(cls)

        set0, set1 = SetOperator._checkArgs(*args)
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

    def get(self, val, default=None):
        #return all(val in s for s in self._sets)
        for s in self._sets:
            v = s.get(val, default)
            if v is default:
                return default
        return v


class SetIntersection_FiniteSet(_FiniteSetMixin, SetIntersection_InfiniteSet):
    __slots__ = tuple()

    def _iter_impl(self):
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

class SetDifference(SetOperator):
    __slots__ = tuple()

    _operator = " - "

    def __new__(cls, *args):
        if cls != SetDifference:
            return super(SetDifference, cls).__new__(cls)

        set0, set1 = SetOperator._checkArgs(*args)
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

    def get(self, val, default=None):
        #return val in self._sets[0] and not val in self._sets[1]
        v_l = self._sets[0].get(val, default)
        if v_l is default:
            return default
        v_r = self._sets[1].get(val, default)
        if v_r is default:
            return v_l
        return default


class SetDifference_FiniteSet(_FiniteSetMixin, SetDifference_InfiniteSet):
    __slots__ = tuple()

    def _iter_impl(self):
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

class SetSymmetricDifference(SetOperator):
    __slots__ = tuple()

    _operator = " ^ "

    def __new__(cls, *args):
        if cls != SetSymmetricDifference:
            return super(SetSymmetricDifference, cls).__new__(cls)

        set0, set1 = SetOperator._checkArgs(*args)
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

    def get(self, val, default=None):
        #return (val in self._sets[0]) ^ (val in self._sets[1])
        v_l = self._sets[0].get(val, default)
        v_r = self._sets[1].get(val, default)
        if v_l is default:
            return v_r
        if v_r is default:
            return v_l
        return default


class SetSymmetricDifference_FiniteSet(_FiniteSetMixin,
                                        SetSymmetricDifference_InfiniteSet):
    __slots__ = tuple()

    def _iter_impl(self):
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

class SetProduct(SetOperator):
    __slots__ = tuple()

    _operator = "*"

    def __new__(cls, *args):
        if cls != SetProduct:
            return super(SetProduct, cls).__new__(cls)

        _sets = SetOperator._checkArgs(*args)
        if all(_[0] for _ in _sets):
            cls = SetProduct_OrderedSet
        elif all(_[1] for _ in _sets):
            cls = SetProduct_FiniteSet
        else:
            cls = SetProduct_InfiniteSet
        return cls.__new__(cls)

    def ranges(self):
        yield RangeProduct(list(
            list(_.ranges()) for _ in self.subsets(False)
        ))

    def bounds(self):
        return ( tuple(_.bounds()[0] for _ in self.subsets(False)),
                 tuple(_.bounds()[1] for _ in self.subsets(False)) )

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

    def _flatten_product(self, val):
        """Flatten any nested set product terms (due to nested products)

        Note that because this is called in a recursive context, this
        method is assured that there is no more than a single level of
        nested tuples (so this only needs to check the top-level terms)

        """
        for i in xrange(len(val)-1, -1, -1):
            if val[i].__class__ is tuple:
                val = val[:i] + val[i] + val[i+1:]
        return val

class SetProduct_InfiniteSet(SetProduct):
    __slots__ = tuple()

    def get(self, val, default=None):
        #return self._find_val(val) is not None
        v = self._find_val(val)
        if v is None:
            return default
        if normalize_index.flatten:
            return self._flatten_product(v[0])
        return v[0]

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

        # For this search, if a subset has an unknown dimension, assume
        # it is "None".
        for i,d in enumerate(setDims):
            if d is UnknownSetDimen:
                setDims[i] = None
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
            if lastIndex == v_len:
                return val, index
            else:
                return None

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

    def _iter_impl(self):
        _iter = itertools.product(*self._sets)
        # Note: if all the member sets are simple 1-d sets, then there
        # is no need to call flatten_product.
        if FLATTEN_CROSS_PRODUCT and normalize_index.flatten \
           and self.dimen != len(self._sets):
            return (self._flatten_product(_) for _ in _iter)
        return _iter

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        ans = 1
        for s in self._sets:
            ans *= max(0, len(s))
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
            return self._flatten_product(ans)
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
        # There is a chicken-and-egg game here: the SetInitializer uses
        # Any as part of the processing of the domain/within/bounds
        # domain restrictions.  However, Any has not been declared when
        # constructing Any, so we need to bypass that logic.  This
        # works, but requires us to declare a special domain setter to
        # accept (and ignore) this value.
        kwds.setdefault('domain', self)
        Set.__init__(self, **kwds)

    def get(self, val, default=None):
        return val

    def ranges(self):
        yield AnyRange()

    def bounds(self):
        return (None, None)

    # We need to implement this to override the clear() from IndexedComponent
    def clear(self):
        return

    # We need to implement this to override __len__ from IndexedComponent
    def __len__(self):
        raise TypeError("object of type 'Any' has no len()")

    @property
    def dimen(self):
        return None

    @property
    def domain(self):
        return Any

    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        return type(self).__name__


class _AnyWithNoneSet(_AnySet):
    # Note that we put the deprecation warning on contains() and not on
    # the class because we will always create a global instance for
    # backwards compatability with the Book.
    @deprecated("The AnyWithNone set is deprecated.  "
                "Use Any, which includes None", version='5.7')
    def get(self, val, default=None):
        return super(_AnyWithNoneSet, self).get(val, default)


class _EmptySet(_FiniteSetMixin, _SetData, Set):
    def __init__(self, **kwds):
        _SetData.__init__(self, component=self)
        Set.__init__(self, **kwds)

    def get(self, val, default=None):
        return default

    # We need to implement this to override clear from IndexedComponent
    def clear(self):
        pass

    # We need to implement this to override __len__ from IndexedComponent
    def __len__(self):
        return 0

    def _iter_impl(self):
        return iter(tuple())

    @property
    def dimen(self):
        return 0

    @property
    def domain(self):
        return EmptySet

    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        return type(self).__name__


############################################################################

def DeclareGlobalSet(obj, caller_globals=None):
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
    assert obj.parent_component() is obj
    assert obj.parent_block() is None

    # Build the global set before registering its name so that we don't
    # run afoul of the logic in GlobalSet.__new__
    _name = obj.local_name
    if _name in GlobalSets and obj is not GlobalSets[_name]:
        raise RuntimeError("Duplicate Global Set declaration, %s"
                           % (_name,))

    # Push this object into the caller's module namespace
    # Stack: 0: DeclareGlobalSet()
    #        1: the caller
    if caller_globals is None:
        caller_globals = inspect.currentframe().f_back.f_globals
    if _name in caller_globals and obj is not caller_globals[_name]:
        raise RuntimeError("Refusing to overwrite global object, %s"
                           % (_name,))

    if _name in GlobalSets:
        _set = caller_globals[_name] = GlobalSets[_name]
        return _set

    # Handle duplicate registrations before defining the GlobalSet
    # object to avoid inconsistent MRO order.

    class GlobalSet(GlobalSetBase, obj.__class__):
        __doc__ = """%s

        References to this object will not be duplicated by deepcopy
        and be maintained/restored by pickle.

        """ % (obj.doc,)
        # Note: a simple docstring does not appear to be picked up (at
        # least in Python 2.7), so we will explicitly set the __doc__
        # attribute.

        __slots__ = ()

        global_name = None

        def __new__(cls, *args, **kwds):
            """Hijack __new__ to mock up old RealSet el al. interface

            In the original Set implementation (Pyomo<=5.6.7), the
            global sets were instances of their own virtual set classes
            (RealSet, IntegerSet, BooleanSet), and one could create new
            instances of those sets with modified bounds.  Since the
            GlobalSet mechanism also declares new classes for every
            GlobalSet, we can mock up the old behavior through how we
            handle __new__().
            """
            if cls is GlobalSet and GlobalSet.global_name \
               and issubclass(GlobalSet, RangeSet):
                deprecation_warning(
                    "The use of RealSet, IntegerSet, BinarySet and "
                    "BooleanSet as Pyomo Set class generators is "
                    "deprecated.  Please either use one of the pre-declared "
                    "global Sets (e.g., Reals, NonNegativeReals, Integers, "
                    "PositiveIntegers, Binary), or create a custom RangeSet.",
                    version='5.7.1')
                # Note: we will completely ignore any positional
                # arguments.  In this situation, these could be the
                # parent_block and any indices; e.g.,
                #    Var(m.I, within=RealSet)
                base_set = GlobalSets[GlobalSet.global_name]
                bounds = kwds.pop('bounds', None)
                range_init = SetInitializer(base_set)
                if bounds is not None:
                    range_init.intersect(BoundsInitializer(bounds))
                name = name_kwd = kwds.pop('name', None)
                cls_name = kwds.pop('class_name', None)
                if name is None:
                    if cls_name is None:
                        name = base_set.name
                    else:
                        name = cls_name
                ans = RangeSet( ranges=list(range_init(None, None).ranges()),
                                name=name )
                if name_kwd is None and (
                        cls_name is not None or bounds is not None):
                    ans._name += str(ans.bounds())
            else:
                ans = super(GlobalSet, cls).__new__(cls, *args, **kwds)
            if kwds:
                raise RuntimeError("Unexpected keyword arguments: %s" % (kwds,))
            return ans

    _set = GlobalSet()
    # TODO: Can GlobalSets be a proper Block?
    GlobalSets[_name] = caller_globals[_name] = _set
    GlobalSet.global_name = _name

    _set.__class__.__setstate__(_set, obj.__getstate__())
    _set._component = weakref.ref(_set)
    _set.construct()
    return _set


DeclareGlobalSet(_AnySet(
    name='Any',
    doc="A global Pyomo Set that admits any value",
), globals())
DeclareGlobalSet(_AnyWithNoneSet(
    name='AnyWithNone',
    doc="A global Pyomo Set that admits any value",
), globals())
DeclareGlobalSet(_EmptySet(
    name='EmptySet',
    doc="A global Pyomo Set that contains no members",
), globals())

DeclareGlobalSet(RangeSet(
    name='Reals',
    doc='A global Pyomo Set that admits any real (floating point) value',
    ranges=(NumericRange(None,None,0),),
), globals())
DeclareGlobalSet(RangeSet(
    name='NonNegativeReals',
    doc='A global Pyomo Set admitting any real value in [0, +inf]',
    ranges=(NumericRange(0,None,0),),
), globals())
DeclareGlobalSet(RangeSet(
    name='NonPositiveReals',
    doc='A global Pyomo Set admitting any real value in [-inf, 0]',
    ranges=(NumericRange(None,0,0),),
), globals())
DeclareGlobalSet(RangeSet(
    name='NegativeReals',
    doc='A global Pyomo Set admitting any real value in [-inf, 0)',
    ranges=(NumericRange(None,0,0,(True,False)),),
), globals())
DeclareGlobalSet(RangeSet(
    name='PositiveReals',
    doc='A global Pyomo Set admitting any real value in (0, +inf]',
    ranges=(NumericRange(0,None,0,(False,True)),),
), globals())

DeclareGlobalSet(RangeSet(
    name='Integers',
    doc='A global Pyomo Set admitting any integer value',
    ranges=(NumericRange(0,None,1), NumericRange(0,None,-1)),
), globals())
DeclareGlobalSet(RangeSet(
    name='NonNegativeIntegers',
    doc='A global Pyomo Set admitting any integer value in [0, +inf]',
    ranges=(NumericRange(0,None,1),),
), globals())
DeclareGlobalSet(RangeSet(
    name='NonPositiveIntegers',
    doc='A global Pyomo Set admitting any integer value in [-inf, 0]',
    ranges=(NumericRange(0,None,-1),),
), globals())
DeclareGlobalSet(RangeSet(
    name='NegativeIntegers',
    doc='A global Pyomo Set admitting any integer value in [-inf, -1]',
    ranges=(NumericRange(-1,None,-1),),
), globals())
DeclareGlobalSet(RangeSet(
    name='PositiveIntegers',
    doc='A global Pyomo Set admitting any integer value in [1, +inf]',
    ranges=(NumericRange(1,None,1),),
), globals())

DeclareGlobalSet(RangeSet(
    name='Binary',
    doc='A global Pyomo Set admitting the integers {0, 1}',
    ranges=(NumericRange(0,1,1),),
), globals())

#TODO: Convert Boolean from an alias for Binary to a proper Boolean Set
#      admitting {True, False})
DeclareGlobalSet(RangeSet(
    name='Boolean',
    doc='A global Pyomo Set admitting the integers {0, 1}',
    ranges=(NumericRange(0,1,1),),
), globals())

DeclareGlobalSet(RangeSet(
    name='PercentFraction',
    doc='A global Pyomo Set admitting any real value in [0, 1]',
    ranges=(NumericRange(0,1,0),),
), globals())
DeclareGlobalSet(RangeSet(
    name='UnitInterval',
    doc='A global Pyomo Set admitting any real value in [0, 1]',
    ranges=(NumericRange(0,1,0),),
), globals())

# DeclareGlobalSet(Set(
#     initialize=[None],
#     name='UnindexedComponent_set',
#     doc='A global Pyomo Set for unindexed (scalar) IndexedComponent objects',
# ), globals())


RealSet = Reals.__class__
IntegerSet = Integers.__class__
BinarySet = Binary.__class__
BooleanSet = Boolean.__class__


#
# Backwards compatibility: declare the RealInterval and IntegerInterval
# classes (leveraging the new global RangeSet objects)
#

class RealInterval(RealSet):
    @deprecated("RealInterval has been deprecated.  Please use "
                "RangeSet(lower, upper, 0)", version='5.7')
    def __new__(cls, **kwds):
        kwds.setdefault('class_name', 'RealInterval')
        return super(RealInterval, cls).__new__(RealSet, **kwds)

class IntegerInterval(IntegerSet):
    @deprecated("IntegerInterval has been deprecated.  Please use "
                "RangeSet(lower, upper, 1)", version='5.7')
    def __new__(cls, **kwds):
        kwds.setdefault('class_name', 'IntegerInterval')
        return super(IntegerInterval, cls).__new__(IntegerSet, **kwds)

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

try:
    from math import remainder
except ImportError:
    def remainder(a,b):
        ans = a % b
        if ans > abs(b/2.):
            ans -= b
        return ans

if six.PY2:
    getargspec = inspect.getargspec
else:
    # For our needs, getfullargspec is a drop-in replacement for
    # getargspec (which was removed in Python 3.x)
    getargspec = inspect.getfullargspec

from six import iteritems, iterkeys
from six.moves import xrange

from pyutilib.misc.misc import flatten_tuple

from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
    native_types, native_numeric_types, as_numeric
)
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set
)
from pyomo.core.base.misc import sorted_robust

logger = logging.getLogger('pyomo.core')

_prePython37 = sys.version_info[:2] < (3,7)

FLATTEN_CROSS_PRODUCT = True


def Initializer(init, allow_generators=False,
                treat_sequences_as_mappings=True):
    if init.__class__ in native_types:
        if init is None:
            return None
        return _ConstantInitializer(init)
    elif inspect.isfunction(init):
        if not allow_generators and inspect.isgeneratorfunction(init):
            raise ValueError("Generator functions are not allowed")
        # Historically pyomo.core.base.misc.apply_indexed_rule
        # accepted rules that took only the parent block (even for
        # indexed components).  We will preserve that functionality
        # here.
        _args = getargspec(init)
        if len(_args.args) == 1 and _args.varargs is None:
            return _ScalarCallInitializer(init)
        else:
            return _IndexedCallInitializer(init)
    elif isinstance(init, collections.Mapping):
        return _ItemInitializer(init)
    elif isinstance(init, collections.Sequence) \
            and not isinstance(init, six.string_types):
        if treat_sequences_as_mappings:
            return _ItemInitializer(init)
        else:
            return _ConstantInitializer(init)
    elif inspect.isgenerator(init) or hasattr(init, 'next') \
         or hasattr(init, '__next__'):
        if not allow_generators:
            raise ValueError("Generators are not allowed")
        return _ConstantInitializer(init)
    else:
        return _ConstantInitializer(init)

class _InitializerBase(object):
    __slots__ = ()

    verified = False

    def __getstate__(self):
        return dict((k, getattr(self,k)) for k in self.__slots__)

    def __setstate__(self, state):
        for key, val in iteritems(state):
            object.__setattr__(self, key, val)

class _ConstantInitializer(_InitializerBase):
    __slots__ = ('val','verified')

    def __init__(self, val):
        self.val = val
        self.verified = False

    def __call__(self, parent, idx):
        return self.val

    def constant(self):
        return True

class _ItemInitializer(_InitializerBase):
    __slots__ = ('_dict',)

    def __init__(self, _dict):
        self._dict = _dict

    def __call__(self, parent, idx):
        return self._dict[idx]

    def constant(self):
        return False

class _IndexedCallInitializer(_InitializerBase):
    __slots__ = ('_fcn',)

    def __init__(self, _fcn):
        self._fcn = _fcn

    def __call__(self, parent, idx):
        if idx.__class__ is tuple:
            return self._fcn(parent, *idx)
        else:
            return self._fcn(parent, idx)

    def constant(self):
        return False


class _CountedCallGenerator(object):
    def __init__(self, fcn, scalar, parent, idx):
        self._count = 0
        if scalar:
            self._fcn = lambda c: fcn(parent, c)
        elif idx.__class__ is tuple:
            self._fcn = lambda c: fcn(parent, c, *idx)
        else:
            self._fcn = lambda c: fcn(parent, c, idx)

    def __iter__(self):
        return self

    def __next__(self):
        self._count += 1
        return self._fcn(self._count)

    next = __next__


class _CountedCallInitializer(_InitializerBase):
    # Pyomo has a historical feature for some rules, where the number of
    # times(*) the rule was called could be passed as an additional
    # argument between the block and the index.  This was primarily
    # supported by Set and ConstraintList.  There were many issues with
    # the syntax, including inconsistent support for jagged (dimen=None)
    # indexing sets, inconsistent support for *args rules, and a likely
    # infinite loop if the rule returned Constraint.Skip.
    #
    # As a slight departure from previous implementations, we will ONLY
    # allow the counted rule syntax when the rule does NOT use *args
    #
    # [JDS 6/2019] We will support a slightly restricted but more
    # consistent form of the original implementation for backwards
    # compatability, but I belee that we should deprecate teh syntax
    # entirely.
    __slots__ = ('_fcn','_is_counted_rule', '_scalar',)

    def __init__(self, obj, _indexed_init):
        self._fcn = _indexed_init._fcn
        self._is_counted_rule = None
        self._scalar = not obj.is_indexed()
        if self._scalar:
            self._is_counted_rule = True

    def __call__(self, parent, idx):
        if self._is_counted_rule == False:
            if idx.__class__ is tuple:
                return self._fcn(parent, *idx)
            else:
                return self._fcn(parent, idx)
        if self._is_counted_rule == True:
            return _CountedCallGenerator(self._fcn, self._scalar, parent, idx)

        # Note that this code will only be called once, and only if
        # the object is not a scalar.
        _args = getargspec(self._fcn)
        _len = len(idx) if idx.__class__ is tuple else 1
        if _len + 2 == len(_args.args):
            self._is_counted_rule = True
        else:
            self._is_counted_rule = False
        return self.__call__(parent, idx)

    def constant(self):
        return False

class _ScalarCallInitializer(_InitializerBase):
    __slots__ = ('_fcn',)

    def __init__(self, _fcn):
        self._fcn = _fcn

    def __call__(self, parent, idx):
        return self._fcn(parent)

    def constant(self):
        return False

class SetInitializer(_InitializerBase):
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
                self._set = _SetIntersectInitializer(self._set, other._set)
        else:
            self._set = _SetIntersectInitializer(self._set, other)

    def __call__(self, parent, idx):
        if self._set is None:
            return Any
        else:
            return self._set(parent, idx)

    def constant(self):
        return self._set is None or self._set.constant()

    def setdefault(self, val):
        if self._set is None:
            self._set = _ConstantInitializer(val)

class _SetIntersectInitializer(_InitializerBase):
    __slots__ = ('_A','_B',)
    def __init__(self, setA, setB):
        self._A = setA
        self._B = setB

    def __call__(self, parent, idx):
        return SetIntersection(self._A(parent, idx), self._B(parent, idx))

    def constant(self):
        return self._A.constant() and self._B.constant()

class RangeSetInitializer(_InitializerBase):
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
        return RangeSet(*tuple(val))

    def constant(self):
        return self._init.constant()

    def setdefault(self, val):
        # This is a real range set... there is no default to set
        pass

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

    # TODO: DEPRECETE this functionality? It has never been documented,
    # and I don't know of a use of it in the wild.
    try:
        # If the argument has a set_options attribute, then use
        # it to initialize a set
        options = getattr(arg,'set_options')
        options['initialize'] = arg
    except:
        options = {}

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
    ans = Set(initialize=tmp, ordered=tmp.is_ordered(), **options)
    ans.construct()
    #
    # Or we can do the simple thing and just use SetOf:
    #
    # ans = SetOf(arg)
    return ans


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

class RangeDifferenceError(ValueError): pass

class _UnknownSetDimen(object): pass

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
# Note that is_finite and is_ordered must be resolvable when the class
# is instantiated (*before* construction).  We will key off these fields
# when performing set operations to know what type of operation to
# create, and we will allow set operations in Abstract before
# construction.
#   - TODO: verify that RangeSet checks its params match is_*


#
# Set rewrite TODOs:
#
#   - Right now, many things implicitly only support concrete models.
#     We need to go back and make sure that things all support Abstract
#     models, and in particular that all Simple (scalar) API methods
#     correctly check the _constructed flag.
#
#   - Set() constructs implicitly on declaration with initialize=
#
#   - Test index/ord for equivalence of 1 and (1,)
#
#   - SortedSet should take a custom sorting function
#
#   - Make sure that all classes implement the appropriate methods
#     (e.g., bounds)
#
#   - Sets created with Set.Skip should produce intelligible errors
#
class NumericRange(object):
    """A representation of a numeric range.

    This class represents a contiguous range of numbers.  The class
    mimics the Pyomo (*not* Python) `range` API, with a Start, End, and
    Step.  The Step is a signed int.  If the Step is 0, the range is
    continuous.  The End *is* included in the range.  Ranges are closed,
    unless a closed is spacified as a 2-tuple of bool values.  Only
    continuous ranges may be open (or partially open)

    Closed ranges are not necessarily strictly finite, as None is
    allowed for then End value (as well as the Start value, for
    continuous ranges only).

    """
    __slots__ = ('start','end','step','closed')
    _EPS = 1e-15
    _types_comparable_to_int = {int,}
    _closedMap = {True:True, False:False,
                  '[':True, ']':True, '(':False, ')':False}

    def __init__(self, start, end, step, closed=(True,True)):
        if int(step) != step:
            raise ValueError(
                "NumericRange step must be int (got %s)" % (step,))
        step = int(step)
        if start is None:
            if step:
                raise ValueError("NumericRange: start must not be None "
                                 "for non-continuous steps")
        elif end is not None:
            if step == 0 and end < start:
                raise ValueError(
                    "NumericRange: start must be <= end for "
                    "continuous ranges (got %s..%s)" % (start,end)
                )
            elif (end-start)*step < 0:
                raise ValueError(
                    "NumericRange: start, end ordering incompatible "
                    "with step direction (got [%s:%s:%s])" % (start,end,step)
                )
            if step:
                n = int( (end - start) // step )
                new_end = start + n*step
                assert abs(end - new_end) < abs(step)
                end = new_end
                # It is important (for iterating) that all finite
                # discrete ranges have positive steps
                if step < 0:
                    start, end = end, start
                    step *= -1
        if start == end:
            # If this is a scalar, we will force the step to be 0 (so that
            # things like [1:5:10] == [1:50:100] are easier to validate)
            step = 0

        self.start = start
        self.end = end
        self.step = step

        self.closed = (self._closedMap[closed[0]], self._closedMap[closed[1]])
        if self.is_discrete() and self.closed != (True,True):
            raise ValueError(
                "NumericRange %s is discrete, but passed closed=%s."
                "  Discrete ranges must be closed." % (self, self.closed,))

    def __getstate__(self):
        """
        Retrieve the state of this object as a dictionary.

        This method must be defined because this class uses slots.
        """
        state = {} #super(NumericRange, self).__getstate__()
        for i in NumericRange.__slots__:
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
        if not self.is_discrete():
            return "%s%s..%s%s" % (
                "[" if self.closed[0] else "(",
                self.start, self.end,
                "]" if self.closed[1] else ")",
            )
        if self.start == self.end:
            return "[%s]" % (self.start, )
        elif self.step == 1:
            return "[%s:%s]" % (self.start, self.end)
        else:
            return "[%s:%s:%s]" % (self.start, self.end, self.step)

    __repr__ = __str__

    def __eq__(self, other):
        if type(other) is not NumericRange:
            return False
        return self.start == other.start \
            and self.end == other.end \
            and self.step == other.step \
            and self.closed == other.closed

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        # NumericRanges must hold items that are comparable to ints
        if value.__class__ not in self._types_comparable_to_int:
            try:
                if value.__class__(0) != 0:
                    return False
                else:
                    self._types_comparable_to_int.add(value.__class__)
            except:
                return False

        if self.step:
            _dir = math.copysign(1, self.step)
            return (
                (value - self.start) * math.copysign(1, self.step) >= 0
                and (self.end is None or
                     _dir*(self.end - self.start) >= _dir*(value - self.start))
                and abs(remainder(value - self.start, self.step)) <= self._EPS
            )
        else:
            return (
                self.start is None
                or ( value >= self.start if self.closed[0] else
                     value > self.start )
            ) and (
                self.end is None
                or ( value <= self.end if self.closed[1] else
                     value < self.end )
            )

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

        EPS = NumericRange._EPS
        if cont.end - cont.start - EPS > abs(disc.step):
            return False
        # At this point, the continuous set is shorter than the discrete
        # step.  We need to see if the continuous set overlaps one of the
        # points, or lies completely between two.
        #
        # Note, taking the absolute value of the step is safe because we are
        # seeing if the continuous range overlaps a discrete point in the
        # underlying (unbounded) discrete sequence grounded by disc.start
        rStart = remainder(cont.start - disc.start, abs(disc.step))
        rEnd = remainder(cont.end - disc.start, abs(disc.step))
        return (
            (abs(rStart) > EPS or not cont.closed[0])
            and (abs(rEnd) > EPS or not cont.closed[1])
            and (rStart - rEnd > 0 or not any(cont.closed))
        )


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

    def is_discrete(self):
        return self.step != 0 or \
            (self.start == self.end and self.start is not None)

    def is_finite(self):
        return self.start is not None and self.end is not None \
            and self.is_discrete()

    def isdisjoint(self, other):
        if not isinstance(other, NumericRange):
            # It is easier to just use NonNumericRange/AnyRange's
            # implementation
            return other.isdisjoint(self)

        # First, do a simple sanity check on the endpoints
        if self._nooverlap(other):
            return True
        # Now, check continuous sets
        if not self.step or not other.step:
            # Check the case of scalar values
            if self.start == self.end:
                return self.start not in other
            elif other.start == other.end:
                return other.start not in self

            # We now need to check a continuous set is a subset of a discrete
            # set and the continuous set sits between discrete points
            if self.step:
                return NumericRange._continuous_discrete_disjoint(
                    other, self)
            elif other.step:
                return NumericRange._continuous_discrete_disjoint(
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
        end = NumericRange._firstNonNull(
            self.step > 0,
            self.end,
            NumericRange._firstNonNull(
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
        if not isinstance(other, NumericRange):
            if type(other) is AnyRange:
                return True
            elif type(other) is NonNumericRange:
                return False
            # Other non NumericRange objects wil generate
            # AttributeError exceptions below

        # First, do a simple sanity check on the endpoints
        s1, e1, c1 = self._normalize_bounds()
        s2, e2, c2 = other._normalize_bounds()
        # Checks for unbounded ranges and to make sure self's endpoints are
        # within other's endpoints.
        if s1 is None:
            if s2 is not None:
                return False
        elif s2 is not None:
            if s1 < s2 or ( s1 == s2 and c1[0] and not c2[0] ):
                return False
        if e1 is None:
            if e2 is not None:
                return False
        elif e2 is not None:
            if e1 > e2 or ( e1 == e2 and c1[1] and not c2[1] ):
                return False
        # If other is continuous (even a single point), then by
        # definition, self is a subset (regardless of step)
        if other.step == 0:
            return True
        # If other is discrete and self is continuous, then self can't be a
        # subset unless self is a scalar and is in other
        elif self.step == 0:
            if self.start == self.end:
                return self.start in other
            else:
                return False
        # At this point, both sets are discrete.  Self's period must be a
        # positive integer multiple of other's ...
        EPS = NumericRange._EPS
        if abs(remainder(self.step, other.step)) > EPS:
            return False
        # ...and they must shart a point in common
        return abs(remainder(other.start-self.start, other.step)) <= EPS

    def _normalize_bounds(self):
        """Normalizes this NumericRange.

        This returns a normalized range by reversing lb and ub if the
        NumericRange step is less than zero.  If lb and ub are
        reversed, then closed is updated to reflect that change.

        Returns
        -------
        lb, ub, closed

        """
        if self.step >= 0:
            return self.start, self.end, self.closed
        else:
            return self.end, self.start, (self.closed[1], self.closed[0])

    def _nooverlap(self, other):
        """Return True if the ranges for self and other are strictly separate

        Note: a(None) == +inf and b(None) == -inf

        """
        s1, e1, c1 = self._normalize_bounds()
        s2, e2, c2 = other._normalize_bounds()
        if e1 is not None and s2 is not None:
            if e1 < s2 or ( e1 == s2 and not ( c1[1] and c2[0] )):
                return True
        if e2 is not None and s1 is not None:
            if e2 < s1 or ( e2 == s1 and not ( c2[1] and c1[0] )):
                return True

    @staticmethod
    def _lt(a,b):
        "Return True if a is strictly less than b, with None == -inf"
        if a is None:
            return b is not None
        if b is None:
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
    def _min(*args):
        """Modified implementation of min() with special None handling

        In NumericRange objects, None can represent {positive,
        negative} infintiy.  In the context that this method is used,
        None will always be positive infinity, so None is greater than any
        non-None value.

        """
        a = args[0]
        for b in args[1:]:
            if a is None:
                a = b
            elif b is None:
                pass
            else:
                a = min(a, b)
        return a

    @staticmethod
    def _max(*args):
        """Modified implementation of max() with special None handling

        In NumericRange objects, None can represent {positive,
        negative} infintiy.  In the context that this method is used,
        None will always be negative infinity, so None is less than
        any non-None value.

        """
        a = args[0]
        for b in args[1:]:
            if a is None:
                a = b
            elif b is None:
                pass
            else:
                a = max(a, b)
        return a

    @staticmethod
    def _split_ranges(cnr, new_step):
        """Split a discrete range into a list of ranges using a new step.

        This takes a single NumericRange and splits it into a set
        of new ranges, all of which use a new step.  The new_step must
        be a multiple of the current step.  CNR objects with a step of 0
        are returned unchanged.

        Parameters
        ----------
            cnr: `NumericRange`
                The range to split
            new_step: `int`
                The new step to use for returned ranges

        """
        if cnr.step == 0 or new_step == 0:
            return [cnr]

        assert new_step >= abs(cnr.step)
        assert new_step % cnr.step == 0
        _dir = math.copysign(1, cnr.step)
        _subranges = []
        for i in xrange(abs(new_step // cnr.step)):
            if ( cnr.end is not None
                 and _dir*(cnr.start + i*cnr.step) > _dir*cnr.end ):
                # Once we walk past the end of the range, we are done
                # (all remaining offsets will be farther past the end)
                break

            _subranges.append(NumericRange(
                cnr.start + i*cnr.step, cnr.end, _dir*new_step
            ))
        return _subranges

    def _lcm(self,other_ranges):
        """This computes an approximate Least Common Multiple step"""
        steps = set()
        if self.is_discrete():
            steps.add(abs(self.step) or 1)
        for s in other_ranges:
            if s.is_discrete():
                steps.add(abs(s.step) or 1)
        for step1 in sorted(steps):
            for step2 in sorted(steps):
                if step1 % step2 == 0 and step1 > step2:
                    steps.remove(step2)
        if not steps:
            return 0
        lcm = steps.pop()
        for step in steps:
            lcm *= step
        return lcm

    def _push_to_discrete_boundary(self, val, other, push_toward_end):
        if self.step or val is None or not other.step:
            # If S is discrete, then the code above guarantees
            # it is aligned with T
            return val
        else:
            # S is continuous and T is diecrete.  Move s_min to
            # the first aligned point
            #
            # Note that we need to push the value INTO the range defined
            # by other, so floor/ceil depends on the sign of other.step
            if push_toward_end:
                _rndFcn = math.ceil if other.step > 0 else math.floor
            else:
                _rndFcn = math.floor if other.step > 0 else math.ceil
            return other.step*_rndFcn((val - other.start) / float(other.step))

    def range_difference(self, other_ranges):
        """Return the difference between this range and a list of other ranges.

        Parameters
        ----------
            other_ranges: `iterable`
                An iterable of other range objects to subtract from this range

        """
        _cnr_other_ranges = []
        for r in other_ranges:
            if isinstance(r, NumericRange):
                _cnr_other_ranges.append(r)
            elif type(r) is AnyRange:
                return []
            elif type(r) is NonNumericRange:
                continue
            else:
                # Note: important to check and raise an exception here;
                # otherwise, unrecognized range types would be silently
                # ignored.
                raise ValueError("Unknown range type, %s" % (type(r).__name__,))

        other_ranges = _cnr_other_ranges

        # Find the Least Common Multiple of all the range steps.  We
        # will split discrete ranges into separate ranges with this step
        # so that we can more easily compare them.
        lcm = self._lcm(other_ranges)

        # Split this range into subranges
        _this = NumericRange._split_ranges(self, lcm)
        # Split the other range(s) into subranges
        _other = []
        for s in other_ranges:
            _other.extend(NumericRange._split_ranges(s, lcm))
        # For each rhs subrange, s
        for s in _other:
            _new_subranges = []
            for t in _this:
                if t._nooverlap(s):
                    # If s and t have no overlap, then s cannot remove
                    # any elements from t
                    _new_subranges.append(t)
                    continue

                if t.is_discrete():
                    # s and t are discrete ranges.  Note if there is a
                    # discrete range in the list of ranges, then lcm > 0
                    if s.is_discrete() and (s.start-t.start) % lcm != 0:
                        # s is offset from t and cannot remove any
                        # elements
                        _new_subranges.append(t)
                        continue

                t_min, t_max, t_c = t._normalize_bounds()
                s_min, s_max, s_c = s._normalize_bounds()

                if s.is_discrete() and not t.is_discrete():
                    #
                    # This handles the special case of continuous-discrete
                    if ((s_min is None and t.start is None) or
                        (s_max is None and t.end is None)):
                        raise RangeDifferenceError(
                            "We do not support subtracting an infinite "
                            "discrete range %s from an infinite continuous "
                            "range %s" % (s,t))

                    # At least one of s_min amd t.start must be non-None
                    start = NumericRange._max(
                        s_min, t._push_to_discrete_boundary(t.start, s, True))
                    # At least one of s_max amd t.end must be non-None
                    end = NumericRange._min(
                        s_max, t._push_to_discrete_boundary(t.end, s, False))

                    if NumericRange._lt(t.start, start):
                        _new_subranges.append(NumericRange(
                            t.start, start, 0, (t.closed[0], False)
                        ))
                    if s.step: # i.e., not a single point
                        for i in xrange(int(start//s.step), int(end//s.step)):
                            _new_subranges.append(NumericRange(
                                i*s.step, (i+1)*s.step, 0, '()'
                            ))
                    if NumericRange._gt(t.end, end):
                        _new_subranges.append(NumericRange(
                            end, t.end, 0, (False,t.closed[1])
                        ))
                else:
                    #
                    # This handles discrete-discrete,
                    # continuous-continuous, and discrete-continuous
                    #
                    if NumericRange._lt(t_min, s_min):
                        # Note that s_min will never be None due to the
                        # _lt test
                        if t.step:
                            s_min -= lcm
                            closed1 = True
                        _min = NumericRange._min(t_max, s_min)
                        if not t.step:
                            closed1 = not s_c[0] if _min is s_min else t_c[1]
                        _closed = ( t_c[0], closed1 )
                        _step = abs(t.step)
                        _rng = t_min, _min
                        if t_min is None and t.step:
                            _step = -_step
                            _rng = _rng[1], _rng[0]
                            _closed = _closed[1], _closed[0]

                        _new_subranges.append(NumericRange(
                            _rng[0], _rng[1], _step, _closed))

                    if NumericRange._gt(t_max, s_max):
                        # Note that s_max will never be None due to the _gt test
                        if t.step:
                            s_max += lcm
                            closed0 = True
                        _max = NumericRange._max(t_min, s_max)
                        if not t.step:
                            closed0 = not s_c[1] if _max is s_max else t_c[0]
                        _new_subranges.append(NumericRange(
                            _max, t_max, abs(t.step), (closed0, t_c[1])
                        ))
                _this = _new_subranges
        return _this

    def range_intersection(self, other_ranges):
        """Return the intersection between this range and a set of other ranges.

        Paramters
        ---------
            other_ranges: `iterable`
                An iterable of other range objects to intersect with this range

        """
        _cnr_other_ranges = []
        for r in other_ranges:
            if isinstance(r, NumericRange):
                _cnr_other_ranges.append(r)
            elif type(r) is AnyRange:
                return [self]
            elif type(r) is NonNumericRange:
                continue
            else:
                # Note: important to check and raise an exception here;
                # otherwise, unrecognized range types would be silently
                # ignored.
                raise ValueError("Unknown range type, %s" % (type(r).__name__,))
        other_ranges = _cnr_other_ranges

        # Find the Least Common Multiple of all the range steps.  We
        # will split discrete ranges into separate ranges with this step
        # so that we can more easily compare them.
        lcm = self._lcm(other_ranges)

        ans = []
        # Split this range into subranges
        _this = NumericRange._split_ranges(self, lcm)
        # Split the other range(s) into subranges
        _other = []
        for s in other_ranges:
            _other.extend(NumericRange._split_ranges(s, lcm))
        # For each lhs subrange, t
        for t in _this:
            # Compare it against each rhs range and only keep the
            # subranges of this range that are inside the lhs range
            for s in _other:
                if s.is_discrete() and t.is_discrete():
                    # s and t are discrete ranges.  Note if there is a
                    # finite range in the list of ranges, then lcm > 0
                    if (s.start-t.start) % lcm != 0:
                        # s is offset from t and cannot have any
                        # elements in common
                        continue
                if t._nooverlap(s):
                    continue

                t_min, t_max, t_c = t._normalize_bounds()
                s_min, s_max, s_c = s._normalize_bounds()
                step = abs(t.step if t.step else s.step)

                intersect_start = NumericRange._max(
                    s._push_to_discrete_boundary(s_min, t, True),
                    t._push_to_discrete_boundary(t_min, s, True),
                )

                intersect_end = NumericRange._min(
                    s._push_to_discrete_boundary(s_max, t, False),
                    t._push_to_discrete_boundary(t_max, s, False),
                )
                c = [True,True]
                if intersect_start == t_min:
                    c[0] &= t_c[0]
                if intersect_start == s_min:
                    c[0] &= s_c[0]
                if intersect_end == t_max:
                    c[1] &= t_c[1]
                if intersect_end == s_max:
                    c[1] &= s_c[1]
                if step and intersect_start is None:
                    ans.append(NumericRange(
                        intersect_end, intersect_start, -step, (c[1], c[0])
                    ))
                else:
                    ans.append(NumericRange(
                        intersect_start, intersect_end, step, c
                    ))
        return ans


class NonNumericRange(object):
    """A range-like object for representing a single non-numeric value

    The class name is a bit of a misnomer, as this object does not
    represent a range but rather a single value.  However, as it
    duplicates the Range API (as used by :py:class:`NumericRange`), it
    is called a "Range".

    """

    __slots__ = ('value',)

    def __init__(self, val):
        self.value = val

    def __str__(self):
        return "{%s}" % self.value

    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, NonNumericRange) and other.value == self.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        return value == self.value

    def __getstate__(self):
        """
        Retrieve the state of this object as a dictionary.

        This method must be defined because this class uses slots.
        """
        state = {} #super(NonNumericRange, self).__getstate__()
        for i in NonNumericRange.__slots__:
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

    def is_discrete(self):
        return True

    def is_finite(self):
        return True

    def isdisjoint(self, other):
        return self.value not in other

    def issubset(self, other):
        return self.value in other

    def range_difference(self, other_ranges):
        for r in other_ranges:
            if self.value in r:
                return []
        return [self]

    def range_intersection(self, other_ranges):
        for r in other_ranges:
            if self.value in r:
                return [self]
        return []


class AnyRange(object):
    """A range object for representing Any sets"""

    __slots__ = ()

    def __init__(self):
        pass

    def __str__(self):
        return "[*]"

    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, AnyRange)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        return True

    def is_discrete(self):
        return False

    def is_finite(self):
        return False

    def isdisjoint(self, other):
        return False

    def issubset(self, other):
        return isinstance(other, AnyRange)

    def range_difference(self, other_ranges):
        for r in other_ranges:
            if isinstance(r, AnyRange):
                return []
        else:
            return [self]

    def range_intersection(self, other_ranges):
        return list(other_ranges)


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

    def is_finite(self):
        """Returns True if this is a finite discrete (iterable) Set"""
        return False

    def is_ordered(self):
        """Returns True if this is an ordered finite discrete (iterable) Set"""
        return False

    def __eq__(self, other):
        if self is other:
            return True
        try:
            other_is_finite = other.is_finite()
        except:
            # we assume that everything that does not implement
            # is_finite() is a discrete set.
            other_is_finite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
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

    @property
    def dimen(self):
        raise DeveloperError("Derived finite set class (%s) failed to "
                             "implement dimen" % (type(self).__name__,))

    def isdisjoint(self, other):
        try:
            other_is_finite = other.is_finite()
        except:
            # we assume that everything that does not implement
            # is_finite() is a discrete set.
            other_is_finite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
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
        try:
            other_is_finite = other.is_finite()
        except:
            # we assume that everything that does not implement
            # is_finite() is a discrete set.
            other_is_finite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
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
            other_is_finite = other.is_finite()
        except:
            # we assume that everything that does not implement
            # is_finite() is a discrete set.
            other_is_finite = True
            try:
                # For efficiency, if the other is not a Set, we will try
                # converting it to a Python set() for efficient lookup.
                other = set(other)
            except:
                pass
        if other_is_finite:
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
        ans = Set(initialize=tmp, ordered=tmp.is_ordered())
        ans.construct()
        return ans | self

    def __rand__(self, other):
        # See the discussion of Set vs SetOf in _processArgs below
        #
        # return SetOf(other) & self
        tmp = SetOf(other)
        ans = Set(initialize=tmp, ordered=tmp.is_ordered())
        ans.construct()
        return ans & self

    def __rsub__(self, other):
        # See the discussion of Set vs SetOf in _processArgs below
        #
        # return SetOf(other) - self
        tmp = SetOf(other)
        ans = Set(initialize=tmp, ordered=tmp.is_ordered())
        ans.construct()
        return ans - self

    def __rxor__(self, other):
        # See the discussion of Set vs SetOf in _processArgs below
        #
        # return SetOf(other) ^ self
        tmp = SetOf(other)
        ans = Set(initialize=tmp, ordered=tmp.is_ordered())
        ans.construct()
        return ans ^ self

    def __rmul__(self, other):
        # See the discussion of Set vs SetOf in _processArgs below
        #
        # return SetOf(other) * self
        tmp = SetOf(other)
        ans = Set(initialize=tmp, ordered=tmp.is_ordered())
        ans.construct()
        return ans * self

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
        return reversed(self.data())

    def __len__(self):
        raise DeveloperError("Derived finite set class (%s) failed to "
                             "implement __len__" % (type(self).__name__,))

    def is_finite(self):
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
        self._dimen = None

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
        """
        # The bulk of single-value set members were stored as scalars.
        # Check that first.
        if value.__class__ is tuple and len(value) == 1:
            if value[0] in self._values:
                return True
        return value in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return len(self._values)

    def __reversed__(self):
        # Python will not reverse() sets, so convert to a tuple and reverse
        return reversed(tuple(self._values))

    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        if not self._constructed:
            return type(self).__name__
        return "{" + (', '.join(str(_) for _ in self)) + "}"

    def data(self):
        return tuple(self._values)

    @property
    def dimen(self):
        return self._dimen

    def add(self, value):
        if type(value) is tuple:
            _value = flatten_tuple(value)
            if len(_value) == 1:
                _value = _value[0]
                _d = 1
            else:
                _d = len(_value)
        else:
            _value = value
            _d = 1

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
                if self._dimen is _UnknownSetDimen:
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

class _OrderedSetMixin(object):
    __slots__ = ()

    def is_ordered(self):
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

    def data(self):
        return tuple(self._ordered_values)

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

    def data(self):
        if not self._is_sorted:
            self._sort()
        return super(_SortedSetData, self).data()

    def _add_impl(self, value):
        # Note that the sorted status has no bearing on insertion,
        # so there is no reason to check if the data is correctly sorted
        self._values[value] = len(self._values)
        self._ordered_values.append(value)
        self._is_sorted = False

    # Note: removing data does not affect the sorted flag
    #def remove(self, val):

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

def _pprint_members(x):
    return '{' + str(x.ordered_data())[1:-1] + "}"
def _pprint_dimen(x):
    d = x.dimen
    if d is _UnknownSetDimen:
        return "--"
    return d

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

    class End(object): pass
    class Skip(object): pass
    class InsertionOrder(object): pass
    class SortedOrder(object): pass
    _ValidOrderedAuguments = {True, False, InsertionOrder, SortedOrder}
    _UnorderedInitializers = {set}
    if _prePython37:
        _UnorderedInitializers.add(dict)

    def __new__(cls, *args, **kwds):
        if cls != Set:
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
                return OrderedSimpleSet.__new__(OrderedSimpleSet)
            elif ordered is Set.SortedOrder:
                return SortedSimpleSet.__new__(SortedSimpleSet)
            else:
                return FiniteSimpleSet.__new__(FiniteSimpleSet)
        else:
            newObj = IndexedSet.__new__(IndexedSet)
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

        self._init_dimen = Initializer(kwds.pop('dimen', _UnknownSetDimen))
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
        if self._init_values.__class__ is _IndexedCallInitializer:
            self._init_values = _CountedCallInitializer(self, self._init_values)

    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing Set, name=%s, from data=%r"
                             % (self.name, data))
        self._constructed=True
        if data is not None:
            # Data supplied to construct() should override data provided
            # to the constructor
            tmp_init, self._init_values = self._init_values, Initializer(
                    data, treat_sequences_as_mappings=False)
        try:
            if type(self._init_values) is _ItemInitializer:
                for index in iterkeys(self._init_values._dict):
                    # The index is coming in externally; we need to
                    # validate it
                    IndexedComponent.__getitem__(self, index)
            else:
                for index in self.index_set():
                    self._getitem_when_not_present(index)
        finally:
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
                raise ValueError("Set rule returned None instead of Set.Skip")

        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        if self._init_dimen is not None:
            obj._dimen = self._init_dimen(self, index)
        if self._init_domain is not None:
            obj._domain = self._init_domain(self, index)
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
            if obj.is_ordered() \
                   and type(_values) in self._UnorderedInitializers:
                logger.warning(
                    "Initializing an ordered Set with a fundamentally "
                    "unordered data source (type: %s).  This WILL potentially "
                    "lead to nondeterministic behavior in Pyomo"
                    % (type(_values).__name__,))
            for val in _values:
                if val is Set.End:
                    break
                if _filter is None or _filter(self, val):
                    obj.add(val)
        # We defer adding the filter until now so that add() doesn't
        # call it a second time.
        obj._filter = _filter
        return obj

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

        # TODO: In the current design, we force all _SetData witin an
        # indexed Set to have the same is_ordered value, so we will only
        # print it once in the header.  Is this a good design?
        try:
            _ordered = self.is_ordered()
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
                _pprint_dimen(v),
                v._domain,
                len(v),
                _pprint_members(v),
            ])


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

class IndexedSet(Set):
    pass

############################################################################

class SetOf(_FiniteSetMixin, _SetData, Component):
    """"""
    def __new__(cls, *args, **kwds):
        if cls is not SetOf:
            return super(SetOf, cls).__new__(cls)
        reference, = args
        if isinstance(reference, (tuple, list)):
            return OrderedSetOf.__new__(OrderedSetOf, reference)
        else:
            return UnorderedSetOf.__new__(UnorderedSetOf, reference)

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
        self._constructed=True
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
                v.is_ordered(),
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

    def __init__(self, component, ranges=()):
        _SetData.__init__(self, component=component)
        for r in ranges:
            if not isinstance(r, NumericRange):
                raise TypeError(
                    "_InfiniteRangeSetData ranges argument must be an "
                    "iterable of NumericRange objects")
        self._ranges = ranges

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

    @property
    def dimen(self):
        return 1

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
            r = self._ranges[0]
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
                if any(not r.is_finite() for r in kwds['ranges']):
                    finite = False
            if None in args or (len(args) > 2 and args[2] == 0):
                finite = False
            if finite is None:
                finite = True

        if finite:
            return FiniteSimpleRangeSet.__new__(FiniteSimpleRangeSet)
        else:
            return InfiniteSimpleRangeSet.__new__(InfiniteSimpleRangeSet)


    def __init__(self, *args, **kwds):
        kwds.setdefault('ctype', RangeSet)
        Component.__init__(self, **kwds)


    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        ans = ' | '.join(str(_) for _ in self.ranges())
        if ' | ' in ans:
            return "(" + ans + ")"
        return ans


    def _process_args(self, args, ranges):
        if type(ranges) is not tuple:
            ranges = tuple(ranges)
        if len(args) == 1:
            ranges = ranges + (NumericRange(1,args[0],1),)
        elif len(args) == 2:
            ranges = ranges + (NumericRange(args[0],args[1],1),)
        elif len(args) == 3:
            ranges = ranges + (NumericRange(*args),)
        elif args:
            raise ValueError("RangeSet expects 3 or fewer positional "
                             "arguments (received %s)" % (len(args),))
        return ranges


    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
                logger.debug("Constructing RangeSet, name=%s, from data=%r"
                             % (self.name, data))
        # TODO: verify that the constructed ranges match finite
        self._constructed=True
        timer.report()


    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Dimen", self.dimen),
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
        ranges = self._process_args(args, kwds.pop('ranges', []))
        _InfiniteRangeSetData.__init__(self, component=self, ranges=ranges)
        RangeSet.__init__(self, **kwds)

    # We want the RangeSet.__str__ to override the one in _FiniteSetMixin
    __str__ = RangeSet.__str__

class FiniteSimpleRangeSet(_FiniteRangeSetData, RangeSet):
    def __init__(self, *args, **kwds):
        ranges = self._process_args(args, kwds.pop('ranges', []))
        _FiniteRangeSetData.__init__(self, component=self, ranges=ranges)
        RangeSet.__init__(self, **kwds)

    # We want the RangeSet.__str__ to override the one in _FiniteSetMixin
    __str__ = RangeSet.__str__


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

    # Note: because none of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __str__(self):
        if self.parent_block() is not None:
            return self.name
        return self._operator.join(
            '(%s)' % arg if isinstance(arg, _SetOperator)
            else str(arg) for arg in self._sets)

    @staticmethod
    def _checkArgs(*sets):
        ans = []
        for s in sets:
            if isinstance(s, _SetDataBase):
                ans.append((s.is_ordered(), s.is_finite()))
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
        d1 = self._sets[0].dimen
        if d1 == self._sets[1].dimen:
            return d1
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
            cls = SetIntersection_OrderedSet
            for r0 in args[0].ranges():
                for r01 in r0.range_intersection(args[1].ranges()):
                    if not r01.is_finite():
                        cls = SetIntersection_InfiniteSet
                        return cls.__new__(cls)
        return cls.__new__(cls)

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
        if not set0.is_ordered():
            if set1.is_ordered():
                set0, set1 = set1, set0
            elif not set0.is_finite():
                if set1.is_finite():
                    set0, set1 = set1, set0
                else:
                    # THe odd case of a finite continuous range
                    # intersected with an infinite discrete range...
                    ranges = []
                    for r0 in set0.ranges():
                        ranges.extend(r0.range_intersection(set1.ranges()))
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
        d1 = self._sets[0].dimen
        if d1 == self._sets[1].dimen:
            return d1
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

class SetProduct_InfiniteSet(SetProduct):
    __slots__ = tuple()

    def __contains__(self, val):
        return self._find_val(val) is not None

    def _find_val(self, val):
        if type(val) is not tuple:
            val = (val,)

        # Support for ambiguous cross products: if val matches the
        # number of subsets, we will start by checking each value
        # against the corresponding subset.  Failure is not sufficient
        # to determine the val is not in this set.
        if len(val) == len(self._sets):
            if all(v in self._sets[i] for i,v in enumerate(val)):
                return val, None

        val = flatten_tuple(val)
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
            if lastIndex > len(val):
                return None
            elif val[index[i]:lastIndex] not in self._sets[i]:
                return None
        # The end of the last subset is always the length of the val
        index.append(len(val))

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
        if FLATTEN_CROSS_PRODUCT and self.dimen != len(self._sets):
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

    @property
    def dimen(self):
        ans = 0
        for s in self._sets:
            s_dim = s.dimen
            if s_dim is None:
                return None
            ans += s_dim
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
        if FLATTEN_CROSS_PRODUCT and self.dimen != len(ans):
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
    class GlobalSet(obj.__class__):
        __doc__ = """A "global" instance of a %s object.

        References to this object will not be duplicated by deepcopy
        and be maintained/restored by pickle.

        """ % (obj.__class__.__name__,)
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
            "Cause pickle to preserve references to this object"
            return self.name

        def __deepcopy__(self, memo):
            "Prevent deepcopy from duplicating this object"
            return self

        def __str__(self):
            "Override str() to always print out the global set name"
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

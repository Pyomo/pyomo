#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math
from collections.abc import Sequence

from pyomo.common.autoslots import AutoSlots
from pyomo.common.numeric_types import check_if_numeric_type

try:
    from math import remainder
except ImportError:

    def remainder(a, b):
        ans = a % b
        if ans > abs(b / 2.0):
            ans -= b
        return ans


_inf = float('inf')
_infinite = {_inf, -_inf}


class RangeDifferenceError(ValueError):
    pass


class NumericRange(AutoSlots.Mixin):
    """A representation of a numeric range.

    This class represents a contiguous range of numbers.  The class
    mimics the Pyomo (*not* Python) `range` API, with a Start, End, and
    Step.  The Step is a signed int.  If the Step is 0, the range is
    continuous.  The End *is* included in the range.  Ranges are closed,
    unless `closed` is specified as a 2-tuple of bool values.  Only
    continuous ranges may be open (or partially open)

    Closed ranges are not necessarily strictly finite, as None is
    allowed for the End value (as well as the Start value, for
    continuous ranges only).

    Parameters
    ----------
        start : float
            The starting value for this NumericRange
        end : float
            The last value for this NumericRange
        step : int
            The interval between values in the range.  0 indicates a
            continuous range.  Negative values indicate discrete ranges
            walking backwards.
        closed : tuple of bool, optional
            A 2-tuple of bool values indicating if the beginning and end
            of the range is closed.  Open ranges are only allowed for
            continuous NumericRange objects.
    """

    __slots__ = ('start', 'end', 'step', 'closed')
    _EPS = 1e-15
    _types_comparable_to_int = {int}
    _closedMap = {
        True: True,
        False: False,
        '[': True,
        ']': True,
        '(': False,
        ')': False,
    }

    def __init__(self, start, end, step, closed=(True, True)):
        if int(step) != step:
            raise ValueError("NumericRange step must be int (got %s)" % (step,))
        step = int(step)
        if start is None:
            start = -_inf
        if end is None:
            end = math.copysign(_inf, step)

        if step:
            if start == -_inf:
                raise ValueError(
                    "NumericRange: start must not be None/-inf "
                    "for non-continuous steps"
                )
            if (end - start) * step < 0:
                raise ValueError(
                    "NumericRange: start, end ordering incompatible "
                    "with step direction (got [%s:%s:%s])" % (start, end, step)
                )
            if end not in _infinite:
                n = int((end - start) // step)
                new_end = start + n * step
                assert abs(end - new_end) < abs(step)
                end = new_end
                # It is important (for iterating) that all finite
                # discrete ranges have positive steps
                if step < 0:
                    start, end = end, start
                    step *= -1
        elif end < start:  # and step == 0
            raise ValueError(
                "NumericRange: start must be <= end for "
                "continuous ranges (got %s..%s)" % (start, end)
            )
        if start == end:
            # If this is a scalar, we will force the step to be 0 (so that
            # things like [1:5:10] == [1:50:100] are easier to validate)
            step = 0

        self.start = start
        self.end = end
        self.step = step

        self.closed = (self._closedMap[closed[0]], self._closedMap[closed[1]])
        if self.isdiscrete() and self.closed != (True, True):
            raise ValueError(
                "NumericRange %s is discrete, but passed closed=%s."
                "  Discrete ranges must be closed." % (self, self.closed)
            )

    def __str__(self):
        if not self.isdiscrete():
            return "%s%s..%s%s" % (
                "[" if self.closed[0] else "(",
                self.start,
                self.end,
                "]" if self.closed[1] else ")",
            )
        if self.start == self.end:
            return "[%s]" % (self.start,)
        elif self.step == 1:
            return "[%s:%s]" % (self.start, self.end)
        else:
            return "[%s:%s:%s]" % (self.start, self.end, self.step)

    __repr__ = __str__

    def __eq__(self, other):
        if type(other) is not NumericRange:
            return False
        return (
            self.start == other.start
            and self.end == other.end
            and self.step == other.step
            and self.closed == other.closed
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        # NumericRanges must hold items that are comparable to ints
        if value.__class__ not in self._types_comparable_to_int:
            # Build on numeric_type.check_if_numeric_type to cleanly
            # handle numpy registrations
            if check_if_numeric_type(value):
                self._types_comparable_to_int.add(value.__class__)
            else:
                # Special case: because numpy is fond of returning scalars
                # as length-1 ndarrays, we will include a special case that
                # will unpack things that look like single element arrays.
                try:
                    # Note: trap "value[0] is not value" to catch things like
                    # single-character strings
                    if (
                        hasattr(value, '__len__')
                        and hasattr(value, '__getitem__')
                        and len(value) == 1
                        and value[0] is not value
                    ):
                        return value[0] in self
                except:
                    pass
                # See if this class behaves like a "normal" number: both
                # comparable and creatable
                try:
                    if not (bool(value - 0 > 0) ^ bool(value - 0 <= 0)):
                        return False
                    elif value.__class__(0) != 0 or not value.__class__(0) == 0:
                        return False
                    else:
                        self._types_comparable_to_int.add(value.__class__)
                except:
                    return False

        if self.step:
            _dir = int(math.copysign(1, self.step))
            _from_start = value - self.start
            return (
                0 <= _dir * _from_start <= _dir * (self.end - self.start)
                and abs(remainder(_from_start, self.step)) <= self._EPS
            )
        else:
            return (value >= self.start if self.closed[0] else value > self.start) and (
                value <= self.end if self.closed[1] else value < self.end
            )

    @staticmethod
    def _continuous_discrete_disjoint(cont, disc):
        # At this point, we know the ranges overlap (tested for at the
        # beginning of isdisjoint()
        d_lb = disc.start if disc.step > 0 else disc.end
        d_ub = disc.end if disc.step > 0 else disc.start
        if cont.start <= d_lb:
            return False
        if cont.end >= d_ub:
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

    def isdiscrete(self):
        return self.step or self.start == self.end

    def isfinite(self):
        return (self.step and self.end not in _infinite) or self.end == self.start

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
                return NumericRange._continuous_discrete_disjoint(other, self)
            elif other.step:
                return NumericRange._continuous_discrete_disjoint(self, other)
            else:
                # 2 continuous sets, with overlapping end points: not disjoint
                return False
        # both sets are discrete
        if self.step == other.step:
            return abs(remainder(other.start - self.start, self.step)) > self._EPS
        # Two infinite discrete sets will *eventually* have a common
        # point.  This is trivial for coprime integer steps.  For steps
        # with gcd > 1, we need to ensure that the two ranges are
        # aligned to the gcd period.
        #
        # Note that this all breaks down for for float steps with
        # infinite precision (think a step of PI).  However, for finite
        # precision maths, the "float" times a sufficient power of two
        # is an integer.  Is this a distinction we want to make?
        # Personally, anyone making a discrete set with a non-integer
        # step is asking for trouble.  Maybe the better solution is to
        # require that the step be integer (which is what we do).
        elif (
            self.end in _infinite
            and other.end in _infinite
            and self.step * other.step > 0
        ):
            gcd = NumericRange._gcd(self.step, other.step)
            return abs(remainder(other.start - self.start, gcd)) > self._EPS
        # OK - at this point, there are a finite number of set members
        # that can overlap.  Just check all the members of one set
        # against the other
        if self.step > 0:
            end = min(self.end, max(other.start, other.end))
        else:
            end = max(self.end, min(other.start, other.end))
        i = 0
        item = self.start
        while (self.step > 0 and item <= end) or (self.step < 0 and item >= end):
            if item in other:
                return False
            i += 1
            item = self.start + self.step * i
        return True

    def issubset(self, other):
        if not isinstance(other, NumericRange):
            if type(other) is AnyRange:
                return True
            elif type(other) is NonNumericRange:
                return False
            # Other non NumericRange objects will generate
            # AttributeError exceptions below

        # First, do a simple sanity check on the endpoints
        s1, e1, c1 = self.normalize_bounds()
        s2, e2, c2 = other.normalize_bounds()
        # Checks for unbounded ranges and to make sure self's endpoints are
        # within other's endpoints.
        if s1 < s2:
            return False
        if e1 > e2:
            return False
        if s1 == s2 and c1[0] and not c2[0]:
            return False
        if e1 == e2 and c1[1] and not c2[1]:
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
        return abs(remainder(other.start - self.start, other.step)) <= EPS

    def normalize_bounds(self):
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
        """Return True if the ranges for self and other are strictly separate"""
        s1, e1, c1 = self.normalize_bounds()
        s2, e2, c2 = other.normalize_bounds()
        if (
            e1 < s2
            or e2 < s1
            or (e1 == s2 and not (c1[1] and c2[0]))
            or (e2 == s1 and not (c2[1] and c1[0]))
        ):
            return True
        return False

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
        _dir = int(math.copysign(1, cnr.step))
        _subranges = []
        for i in range(int(abs(new_step // cnr.step))):
            if _dir * (cnr.start + i * cnr.step) > _dir * cnr.end:
                # Once we walk past the end of the range, we are done
                # (all remaining offsets will be farther past the end)
                break
            _subranges.append(
                NumericRange(cnr.start + i * cnr.step, cnr.end, _dir * new_step)
            )
        return _subranges

    @staticmethod
    def _gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    @staticmethod
    def _lcm(a, b):
        gcd = NumericRange._gcd(a, b)
        if not gcd:
            return 0
        return a * b / gcd

    def _step_lcm(self, other_ranges):
        """This computes an approximate Least Common Multiple step"""
        # Note: scalars are discrete, but have a step of 0.  Pretend the
        # step is 1 so that we can compute a realistic "step lcm"
        if self.isdiscrete():
            a = self.step or 1
        else:
            a = 0
        for o in other_ranges:
            if o.isdiscrete():
                b = o.step or 1
            else:
                b = 0
            lcm = NumericRange._lcm(a, b)
            # This is a modified LCM.  LCM(n,0) == 0, but for step
            # calculations, we want it to be n
            if lcm:
                a = lcm
            else:
                # one of the steps was 0: add to preserve the non-zero step
                a += b
        return int(abs(a))

    def _push_to_discrete_element(self, val, push_to_next_larger_value):
        if not self.step or val in _infinite:
            return val
        else:
            # self is discrete and val is a numeric value.  Move val to
            # the first discrete point aligned with self's range
            #
            # Note that we need to push the value INTO the range defined
            # by this set, so floor/ceil depends on the sign of self.step
            if push_to_next_larger_value:
                _rndFcn = math.ceil if self.step > 0 else math.floor
            else:
                _rndFcn = math.floor if self.step > 0 else math.ceil
            return self.start + self.step * _rndFcn(
                (val - self.start) / float(self.step)
            )

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
        lcm = self._step_lcm(other_ranges)

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

                if t.isdiscrete():
                    # s and t are discrete ranges.  Note if there is a
                    # discrete range in the list of ranges, then lcm > 0
                    if s.isdiscrete() and (s.start - t.start) % lcm != 0:
                        # s is offset from t and cannot remove any
                        # elements
                        _new_subranges.append(t)
                        continue

                t_min, t_max, t_c = t.normalize_bounds()
                s_min, s_max, s_c = s.normalize_bounds()

                if s.isdiscrete() and not t.isdiscrete():
                    #
                    # This handles the special case of continuous-discrete
                    if (s_min == -_inf and t.start == -_inf) or (
                        s_max == _inf and t.end == _inf
                    ):
                        raise RangeDifferenceError(
                            "We do not support subtracting an infinite "
                            "discrete range %s from an infinite continuous "
                            "range %s" % (s, t)
                        )

                    # At least one of s_min amd t.start must be non-inf
                    start = max(s_min, s._push_to_discrete_element(t.start, True))
                    # At least one of s_max amd t.end must be non-inf
                    end = min(s_max, s._push_to_discrete_element(t.end, False))

                    if t.start < start:
                        _new_subranges.append(
                            NumericRange(t.start, start, 0, (t.closed[0], False))
                        )
                    if s.step:  # i.e., not a single point
                        for i in range(int((end - start) // s.step)):
                            _new_subranges.append(
                                NumericRange(
                                    start + i * s.step,
                                    start + (i + 1) * s.step,
                                    0,
                                    '()',
                                )
                            )
                    if t.end > end:
                        _new_subranges.append(
                            NumericRange(end, t.end, 0, (False, t.closed[1]))
                        )
                else:
                    #
                    # This handles discrete-discrete,
                    # continuous-continuous, and discrete-continuous
                    #
                    if t_min < s_min:
                        # Note s_min will never be -inf due to the < test
                        if t.step:
                            s_min -= lcm
                            closed1 = True
                        _min = min(t_max, s_min)
                        if not t.step:
                            closed1 = not s_c[0] if _min is s_min else t_c[1]
                        _closed = (t_c[0], closed1)
                        _step = abs(t.step)
                        _rng = t_min, _min
                        if t_min == -_inf and t.step:
                            _step = -_step
                            _rng = _rng[1], _rng[0]
                            _closed = _closed[1], _closed[0]

                        _new_subranges.append(
                            NumericRange(_rng[0], _rng[1], _step, _closed)
                        )
                    elif t_min == s_min and t_c[0] and not s_c[0]:
                        _new_subranges.append(NumericRange(t_min, t_min, 0))

                    if t_max > s_max:
                        # Note s_max will never be inf due to the > test
                        if t.step:
                            s_max += lcm
                            closed0 = True
                        _max = max(t_min, s_max)
                        if not t.step:
                            closed0 = not s_c[1] if _max is s_max else t_c[0]
                        _new_subranges.append(
                            NumericRange(_max, t_max, abs(t.step), (closed0, t_c[1]))
                        )
                    elif t_max == s_max and t_c[1] and not s_c[1]:
                        _new_subranges.append(NumericRange(t_max, t_max, 0))
            _this = _new_subranges
        return _this

    def range_intersection(self, other_ranges):
        """Return the intersection between this range and a set of other ranges.

        Parameters
        ----------
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
        lcm = self._step_lcm(other_ranges)

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
                if s.isdiscrete() and t.isdiscrete():
                    # s and t are discrete ranges.  Note if there is a
                    # finite range in the list of ranges, then lcm > 0
                    if (s.start - t.start) % lcm != 0:
                        # s is offset from t and cannot have any
                        # elements in common
                        continue
                if t._nooverlap(s):
                    continue

                t_min, t_max, t_c = t.normalize_bounds()
                s_min, s_max, s_c = s.normalize_bounds()
                step = abs(t.step if t.step else s.step)

                intersect_start = max(
                    t._push_to_discrete_element(s_min, True),
                    s._push_to_discrete_element(t_min, True),
                )

                intersect_end = min(
                    t._push_to_discrete_element(s_max, False),
                    s._push_to_discrete_element(t_max, False),
                )
                c = [True, True]
                if intersect_start == t_min:
                    c[0] &= t_c[0]
                if intersect_start == s_min:
                    c[0] &= s_c[0]
                if intersect_end == t_max:
                    c[1] &= t_c[1]
                if intersect_end == s_max:
                    c[1] &= s_c[1]
                if step and intersect_start == -_inf:
                    ans.append(
                        NumericRange(
                            intersect_end, intersect_start, -step, (c[1], c[0])
                        )
                    )
                else:
                    ans.append(NumericRange(intersect_start, intersect_end, step, c))
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
        return "{%s}" % (self.value,)

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
        state = {}  # super(NonNumericRange, self).__getstate__()
        for i in NonNumericRange.__slots__:
            state[i] = getattr(self, i)
        return state

    def __setstate__(self, state):
        """
        Set the state of this object using values from a state dictionary.

        This method must be defined because this class uses slots.
        """
        for key, val in state.items():
            # Note: per the Python data model docs, we explicitly
            # set the attribute using object.__setattr__() instead
            # of setting self.__dict__[key] = val.
            object.__setattr__(self, key, val)

    def isdiscrete(self):
        return True

    def isfinite(self):
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

    def isdiscrete(self):
        return False

    def isfinite(self):
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


class RangeProduct(object):
    """A range-like object for representing the cross product of ranges"""

    __slots__ = ('range_lists',)

    def __init__(self, range_lists):
        self.range_lists = range_lists
        # Type checking.  Users should never create a RangeProduct, but
        # just in case...
        assert range_lists.__class__ is list
        for subrange in range_lists:
            assert subrange.__class__ is list

    def __str__(self):
        return (
            "<"
            + ', '.join(
                str(tuple(_)) if len(_) > 1 else str(_[0]) for _ in self.range_lists
            )
            + ">"
        )

    __repr__ = __str__

    def __eq__(self, other):
        return (
            isinstance(other, RangeProduct)
            and self.range_difference([other]) == []
            and other.range_difference([self]) == []
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, value):
        if not isinstance(value, Sequence):
            return False
        if len(value) != len(self.range_lists):
            return False
        return all(
            any(val in rng for rng in rng_list)
            for val, rng_list in zip(value, self.range_lists)
        )

    def __getstate__(self):
        """
        Retrieve the state of this object as a dictionary.

        This method must be defined because this class uses slots.
        """
        state = {}  # super(RangeProduct, self).__getstate__()
        for i in RangeProduct.__slots__:
            state[i] = getattr(self, i)
        return state

    def __setstate__(self, state):
        """
        Set the state of this object using values from a state dictionary.

        This method must be defined because this class uses slots.
        """
        for key, val in state.items():
            # Note: per the Python data model docs, we explicitly
            # set the attribute using object.__setattr__() instead
            # of setting self.__dict__[key] = val.
            object.__setattr__(self, key, val)

    def isdiscrete(self):
        return all(
            all(rng.isdiscrete() for rng in rng_list) for rng_list in self.range_lists
        )

    def isfinite(self):
        return all(
            all(rng.isfinite() for rng in rng_list) for rng_list in self.range_lists
        )

    def isdisjoint(self, other):
        if type(other) is AnyRange:
            return False
        if type(other) is not RangeProduct:
            return True
        if len(other.range_lists) != len(self.range_lists):
            return True
        # Remember, range_lists is a list of lists of range objects.  As
        # isdisjoint only accepts range objects, we need to unpack
        # everything.  Non-disjoint range products require overlaps in
        # all dimensions.
        for s, o in zip(self.range_lists, other.range_lists):
            if all(s_rng.isdisjoint(o_rng) for s_rng in s for o_rng in o):
                return True
        return False

    def issubset(self, other):
        if type(other) is AnyRange:
            return True
        return not any(_ for _ in self.range_difference([other]))

    def range_difference(self, other_ranges):
        # The goal is to start with a single range product and create a
        # set of range products that, when combined, model the
        # range_difference.  This will potentially create (redundant)
        # overlapping regions, but that is OK.
        ans = [self]
        N = len(self.range_lists)
        for other in other_ranges:
            if type(other) is AnyRange:
                return []
            if type(other) is not RangeProduct or len(other.range_lists) != N:
                continue

            tmp = []
            for rp in ans:
                if rp.isdisjoint(other):
                    tmp.append(rp)
                    continue

                for dim in range(N):
                    remainder = []
                    for r in rp.range_lists[dim]:
                        remainder.extend(r.range_difference(other.range_lists[dim]))
                    if remainder:
                        tmp.append(RangeProduct(list(rp.range_lists)))
                        tmp[-1].range_lists[dim] = remainder
            ans = tmp
        return ans

    def range_intersection(self, other_ranges):
        # The goal is to start with a single range product and create a
        # set of range products that, when combined, model the
        # range_difference.  This will potentially create (redundant)
        # overlapping regions, but that is OK.
        ans = list(self.range_lists)
        N = len(self.range_lists)
        for other in other_ranges:
            if type(other) is AnyRange:
                continue
            if type(other) is not RangeProduct or len(other.range_lists) != N:
                return []

            for dim in range(N):
                tmp = []
                for r in ans[dim]:
                    tmp.extend(r.range_intersection(other.range_lists[dim]))
                if not tmp:
                    return []
                ans[dim] = tmp
        return [RangeProduct(ans)]

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math
import six

# Import base classes privately (so that we have handles on them)
import pyutilib.th.pyunit as _pyunit
from pyutilib.th.pyunit import unittest as _unittest

# Now, import the base unittest environment.  We will override things
# specifically later
from unittest import *
from pyutilib.th import *

from pyomo.common.collections import Mapping, Sequence

# This augments the unittest exports with two additional decorators
__all__ = _unittest.__all__ + ['category', 'nottest']

class TestCase(_pyunit.TestCase):
    """A Pyomo-specific class whose instances are single test cases.

    This class derives from unittest.TestCase and provides the following
    additional functionality:
      - extended suport for test categories
      - additional assertions:
        * :py:meth:`assertStructuredAlmostEqual`

    unittest.TestCase documentation
    -------------------------------
    """
    __doc__ += _unittest.TestCase.__doc__

    def assertStructuredAlmostEqual(self, first, second,
                                    places=None, msg=None, delta=None,
                                    reltol=None, abstol=None,
                                    allow_second_superset=False):
        """Test that first and second are equal up to a tolerance

        This compares first and second using both an absolute (`abstol`) and
        relative (`reltol`) tolerance.  It will recursively descend into
        Sequence and Mapping containers (allowing for the relative
        comparison of structured data including lists and dicts).

        `places` and `delta` is supported for compatibility with
        assertAlmostEqual.  If `places` is supplied, `abstol` is
        computed as `10**-places`.  `delta` is an alias for `abstol`.

        If none of {`abstol`, `reltol`, `places`, `delta`} are specified,
        `reltol` defaults to 1e-7.

        If `allow_second_superset` is True, then:

          - only key/value pairs found in mappings in `first` are
            compared to `second` (allowing mappings in `second` to
            contain extra keys)

          - only values found in sequences in `first` are compared to
            `second` (allowing sequences in `second` to contain extra
            values)

        The relative error is computed for numerical values as
            `abs(first - second) / max(abs(first), abs(second))`,
        only when first != second (thereby avoiding divide-by-zero errors).

        Parameters
        ----------
        first:
            the first value to compare
        second:
            the second value to compare
        places: int
            `first` and `second` are considered equivalent if their
            difference is between `places` decimal places; equivalent to
            `abstol = 10**-places` (included for compatibility with
            assertAlmostEqual)
        msg: str
            the message to raise on failure
        delta: float
            alias for `abstol`
        abstol: float
            the absolute tolerance.  `first` and `second` are considered
            equivalent if their absolute difference is less than `abstol`
        reltol: float
            the relative tolerance.  `first` and `second` are considered
            equivalent if their absolute difference divided by the
            larget of `first` and `second` is less than `reltol`
        allow_second_superset: bool
            If True, then extra entries in containers found on second
            will not trigger a failure.

        """
        if sum(1 for _ in (places, delta, abstol) if _ is not None) > 1:
            raise ValueError("Cannot specify more than one of "
                             "{places, delta, abstol}")

        if places is not None:
            abstol = 10**(-places)
        if delta is not None:
            abstol = delta
        if abstol is None and reltol is None:
            reltol = 10**-7

        fail = None
        try:
            self._assertStructuredAlmostEqual(
                first, second, abstol, reltol, not allow_second_superset)
        except self.failureException as e:
            fail = self._formatMessage(
                msg,
                "%s\n    Found when comparing with tolerance "
                "(abs=%s, rel=%s):\n"
                "        first=%s\n        second=%s" % (
                    str(e),
                    abstol,
                    reltol,
                    _unittest.case.safe_repr(first),
                    _unittest.case.safe_repr(second),
                ))

        if fail:
            raise self.failureException(fail)


    def _assertStructuredAlmostEqual(self, first, second,
                                     abstol, reltol, exact):
        """Recursive implementation of assertStructuredAlmostEqual"""

        args = (first, second)
        if all(isinstance(_, Mapping) for _ in args):
            if exact and len(first) != len(second):
                raise self.failureException(
                    "mappings are different sizes (%s != %s)" % (
                        len(first),
                        len(second),
                    ))
            for key in first:
                if key not in second:
                    raise self.failureException(
                        "key (%s) from first not found in second" % (
                            _unittest.case.safe_repr(key),
                        ))
                try:
                    self._assertStructuredAlmostEqual(
                        first[key], second[key], abstol, reltol, exact)
                except self.failureException as e:
                    raise self.failureException(
                        "%s\n    Found when comparing key %s" % (
                            str(e), _unittest.case.safe_repr(key)))
            return # PASS!

        elif any(isinstance(_, six.string_types) for _ in args):
            if first == second:
                return # PASS!

        elif all(isinstance(_, Sequence) for _ in args):
            # Note that Sequence includes strings
            if exact and len(first) != len(second):
                raise self.failureException(
                    "sequences are different sizes (%s != %s)" % (
                        len(first),
                        len(second),
                    ))
            for i, (f, s) in enumerate(zip(first, second)):
                try:
                    self._assertStructuredAlmostEqual(
                        f, s, abstol, reltol, exact)
                except self.failureException as e:
                    raise self.failureException(
                        "%s\n    Found at position %s" % (str(e), i))
            return # PASS!

        else:
            if first == second:
                return
            try:
                f = float(first)
                s = float(second)
                diff = abs(f - s)
                if abstol is not None and diff <= abstol:
                    return # PASS!
                if reltol is not None and diff / max(abs(f), abs(s)) <= reltol:
                    return # PASS!
            except:
                pass

        raise self.failureException(
            "%s !~= %s" % (
                _unittest.case.safe_repr(first),
                _unittest.case.safe_repr(second),
            ))

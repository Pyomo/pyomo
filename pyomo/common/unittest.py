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
        * :py:meth:`assertRelativeEqual`

    unittest.TestCase documentation
    -------------------------------
    """
    __doc__ += _unittest.TestCase.__doc__

    def assertRelativeEqual(self, first, second,
                            places=None, msg=None, delta=None,
                            allow_second_superset=False):
        """Test that first and second are equal up to a relative tolerance

        This compares first and second using a relative tolerance
        (`delta`).  It will recursively descend into Sequence and Mapping
        containers (allowing for the relative comparison of structured
        data including lists and dicts).

        If `places` is supplied, `delta` is computed as `10**-places`.

        If neither `places` nor `delta` is provided, delta defaults to 1e-7.

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

        """
        if places is not None:
            if delta is not None:
                raise ValueError("Cannot specify both places and delta")
            delta = 10**(-places)
        if delta is None:
            delta = 10**-7

        try:
            self._assertRelativeEqual(
                first, second, delta, not allow_second_superset)
        except self.failureException as e:
            raise self.failureException(self._formatMessage(
                msg,
                "%s\n    Found when comparing with relative tolerance %s:"
                "\n        first=%s\n        second=%s" % (
                    str(e),
                    delta,
                    _unittest.case.safe_repr(first),
                    _unittest.case.safe_repr(second),
                )))
            

    def _assertRelativeEqual(self, first, second, delta, exact):
        """Recursive implementation of assertRelativeEqual"""

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
                    self._assertRelativeEqual(
                        first[key], second[key], delta, exact)
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
                    self._assertRelativeEqual(f, s, delta, exact)
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
                if abs(f - s) / max(abs(f), abs(s)) <= delta:
                    return # PASS!
            except:
                pass

        raise self.failureException(
            "%s !~= %s (to relative tolerance %s)" % (
                _unittest.case.safe_repr(first),
                _unittest.case.safe_repr(second),
                delta,
            ))

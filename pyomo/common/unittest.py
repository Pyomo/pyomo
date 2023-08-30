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
#
#  Part of this module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

import enum
import logging
import math
import re
import sys
from io import StringIO


# Now, import the base unittest environment.  We will override things
# specifically later
from unittest import *
import unittest as _unittest
import pytest as pytest

from pyomo.common.collections import Mapping, Sequence
from pyomo.common.errors import InvalidValueError
from pyomo.common.tee import capture_output

from unittest import mock


def _defaultFormatter(msg, default):
    return msg or default


def _floatOrCall(val):
    """Cast the value to float, if that fails call it and then cast.

    This is an "augmented" version of float() to better support
    integration with Pyomo NumericValue objects: if the initial cast to
    float fails by throwing a TypeError (as non-constant NumericValue
    objects will), then it falls back on calling the object and
    returning that value cast to float.

    """
    try:
        return float(val)
    except (TypeError, InvalidValueError):
        pass
    try:
        return float(val())
    except (TypeError, InvalidValueError):
        pass
    try:
        return val.value
    except AttributeError:
        # likely a complex
        return val


def assertStructuredAlmostEqual(
    first,
    second,
    places=None,
    msg=None,
    delta=None,
    reltol=None,
    abstol=None,
    allow_second_superset=False,
    item_callback=_floatOrCall,
    exception=ValueError,
    formatter=_defaultFormatter,
):
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

    Items (entries other than Sequence / Mapping containers, matching
    strings, and items that satisfy `first is second`) are passed to the
    `item_callback` before testing equality and relative tolerances.

    Raises `exception` if `first` and `second` are not equal within
    tolerance.

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
        largest of `first` and `second` is less than `reltol`
    allow_second_superset: bool
        If True, then extra entries in containers found on second
        will not trigger a failure.
    item_callback: function
        items (other than Sequence / Mapping containers, matching
        strings, and items satisfying `is`) are passed to this callback
        to generate the (nominally floating point) value to use for
        comparison.
    exception: Exception
        exception to raise when `first` is not 'almost equal' to `second`.
    formatter: function
        callback for generating the final failure message (for
        compatibility with unittest)

    """
    if sum(1 for _ in (places, delta, abstol) if _ is not None) > 1:
        raise ValueError("Cannot specify more than one of {places, delta, abstol}")

    if places is not None:
        abstol = 10 ** (-places)
    if delta is not None:
        abstol = delta
    if abstol is None and reltol is None:
        reltol = 10**-7

    fail = None
    try:
        _assertStructuredAlmostEqual(
            first,
            second,
            abstol,
            reltol,
            not allow_second_superset,
            item_callback,
            exception,
        )
    except exception as e:
        fail = formatter(
            msg,
            "%s\n    Found when comparing with tolerance "
            "(abs=%s, rel=%s):\n"
            "        first=%s\n        second=%s"
            % (
                str(e),
                abstol,
                reltol,
                _unittest.case.safe_repr(first),
                _unittest.case.safe_repr(second),
            ),
        )

    if fail:
        raise exception(fail)


def _assertStructuredAlmostEqual(
    first, second, abstol, reltol, exact, item_callback, exception
):
    """Recursive implementation of assertStructuredAlmostEqual"""

    args = (first, second)
    f, s = args
    if all(isinstance(_, Mapping) for _ in args):
        if exact and len(first) != len(second):
            raise exception(
                "mappings are different sizes (%s != %s)" % (len(first), len(second))
            )
        for key in first:
            if key not in second:
                raise exception(
                    "key (%s) from first not found in second"
                    % (_unittest.case.safe_repr(key),)
                )
            try:
                _assertStructuredAlmostEqual(
                    first[key],
                    second[key],
                    abstol,
                    reltol,
                    exact,
                    item_callback,
                    exception,
                )
            except exception as e:
                raise exception(
                    "%s\n    Found when comparing key %s"
                    % (str(e), _unittest.case.safe_repr(key))
                )
        return  # PASS!

    elif any(isinstance(_, str) for _ in args):
        if first == second:
            return  # PASS!

    elif all(isinstance(_, Sequence) for _ in args):
        # Note that Sequence includes strings
        if exact and len(first) != len(second):
            raise exception(
                "sequences are different sizes (%s != %s)" % (len(first), len(second))
            )
        for i, (f, s) in enumerate(zip(first, second)):
            try:
                _assertStructuredAlmostEqual(
                    f, s, abstol, reltol, exact, item_callback, exception
                )
            except exception as e:
                raise exception("%s\n    Found at position %s" % (str(e), i))
        return  # PASS!

    else:
        # Catch things like None, which may cause problems for the
        # item_callback [like float(None)])
        #
        # Test `is` and `==`, but this is not necessarily fatal: we will
        # continue and allow the item_callback to potentially convert
        # the values to be comparable.
        try:
            if first is second or first == second:
                return  # PASS!
        except:
            pass
        try:
            f = item_callback(first)
            s = item_callback(second)
            if f == s:
                return
            diff = abs(f - s)
            if abstol is not None and diff <= abstol:
                return  # PASS!
            if reltol is not None and diff / max(abs(f), abs(s)) <= reltol:
                return  # PASS!
            if math.isnan(f) and math.isnan(s):
                return  # PASS! (we will treat NaN as equal)
        except:
            pass

    msg = "%s !~= %s" % (
        _unittest.case.safe_repr(first),
        _unittest.case.safe_repr(second),
    )
    if f is not first or s is not second:
        msg = "%s !~= %s (%s)" % (
            _unittest.case.safe_repr(f),
            _unittest.case.safe_repr(s),
            msg,
        )
    raise exception(msg)


def _runner(q, qualname):
    "Utility wrapper for running functions, used by timeout()"
    resultType = _RunnerResult.call
    if q in _runner.data:
        fcn, args, kwargs = _runner.data[q]
    elif isinstance(qualname, str):
        # Use unittest to instantiate the TestCase and run it
        resultType = _RunnerResult.unittest

        def fcn():
            s = _unittest.TestLoader().loadTestsFromName(qualname)
            r = _unittest.TestResult()
            s.run(r)
            return r.errors + r.failures, r.skipped

        args = ()
        kwargs = {}
    else:
        qualname, fcn, args, kwargs = qualname
    _runner.data[qualname] = None
    OUT = StringIO()
    try:
        with capture_output(OUT):
            result = fcn(*args, **kwargs)
        q.put((resultType, result, OUT.getvalue()))
    except:
        import traceback

        etype, e, tb = sys.exc_info()
        if not isinstance(e, AssertionError):
            e = etype(
                "%s\nOriginal traceback:\n%s" % (e, ''.join(traceback.format_tb(tb)))
            )
        q.put((_RunnerResult.exception, e, OUT.getvalue()))
    finally:
        _runner.data.pop(qualname)


# Data structure for passing functions/arguments to the _runner
# without forcing them to be pickled / unpickled
_runner.data = {}


class _RunnerResult(enum.Enum):
    exception = 0
    call = 1
    unittest = 2


def timeout(seconds, require_fork=False, timeout_raises=TimeoutError):
    """Function decorator to timeout the decorated function.

    This decorator will wrap a function call with a timeout, returning
    the result of the wrapped function.  The timeout is implemented
    using multiprocessing to execute the function in a forked process.
    If the wrapped function raises an exception, then the exception will
    be re-raised in this process.  If the function times out, a
    :python:`TimeoutError` will be raised.

    Note that as this method uses multiprocessing, the wrapped function
    should NOT spawn any subprocesses.  The timeout is implemented using
    `multiprocessing.Process.terminate()`, which sends a SIGTERM signal
    to the subprocess.  Any spawned subprocesses are not collected and
    will be orphaned and left running.

    Parameters
    ----------
    seconds: float
        Number of seconds to wait before timing out the function

    require_fork: bool
        Require support of the 'fork' interface.  If not present,
        immediately raises unittest.SkipTest

    timeout_raises: Exception
        Exception class to raise in the event of a timeout

    Examples
    --------
    >>> import pyomo.common.unittest as unittest
    >>> @unittest.timeout(1)
    ... def test_function():
    ...     return 42
    >>> test_function()
    42

    >>> @unittest.timeout(0.01)
    ... def test_function():
    ...     while 1:
    ...         pass
    >>> test_function()
    Traceback (most recent call last):
        ...
    TimeoutError: test timed out after 0.01 seconds

    """
    import functools
    import multiprocessing
    import queue

    def timeout_decorator(fcn):
        @functools.wraps(fcn)
        def test_timer(*args, **kwargs):
            qualname = '%s.%s' % (fcn.__module__, fcn.__qualname__)
            if qualname in _runner.data:
                return fcn(*args, **kwargs)
            if require_fork and multiprocessing.get_start_method() != 'fork':
                raise _unittest.SkipTest("timeout requires unavailable fork interface")

            q = multiprocessing.Queue()
            if multiprocessing.get_start_method() == 'fork':
                # Option 1: leverage fork if possible.  This minimizes
                # the reliance on serialization and ensures that the
                # wrapped function operates in the same environment.
                _runner.data[q] = (fcn, args, kwargs)
                runner_args = (q, qualname)
            elif (
                args
                and fcn.__name__.startswith('test')
                and _unittest.case.TestCase in args[0].__class__.__mro__
            ):
                # Option 2: this is wrapping a unittest.  Re-run
                # unittest in the child process with this function as
                # the sole target.  This ensures that things like setUp
                # and tearDown are correctly called.
                runner_args = (q, qualname)
            else:
                # Option 3: attempt to serialize the function and all
                # arguments and send them to the (spawned) child
                # process.  The wrapped function cannot count on any
                # environment configuration that it does not set up
                # itself.
                runner_args = (q, (qualname, test_timer, args, kwargs))
            test_proc = multiprocessing.Process(target=_runner, args=runner_args)
            test_proc.daemon = True
            try:
                test_proc.start()
            except:
                if type(runner_args[1]) is tuple:
                    logging.getLogger(__name__).error(
                        "Exception raised spawning timeout subprocess "
                        "on a platform that does not support 'fork'.  "
                        "It is likely that either the wrapped function or "
                        "one of its arguments is not serializable"
                    )
                raise
            try:
                resultType, result, stdout = q.get(True, seconds)
            except queue.Empty:
                test_proc.terminate()
                raise timeout_raises(
                    "test timed out after %s seconds" % (seconds,)
                ) from None
            finally:
                _runner.data.pop(q, None)
            sys.stdout.write(stdout)
            test_proc.join()
            if resultType == _RunnerResult.call:
                return result
            elif resultType == _RunnerResult.unittest:
                for name, msg in result[0]:
                    with args[0].subTest(name):
                        raise args[0].failureException(msg)
                for name, msg in result[1]:
                    with args[0].subTest(name):
                        args[0].skipTest(msg)
            else:
                raise result

        return test_timer

    return timeout_decorator


class _AssertRaisesContext_NormalizeWhitespace(_unittest.case._AssertRaisesContext):
    def __exit__(self, exc_type, exc_value, tb):
        try:
            _save_re = self.expected_regex
            self.expected_regex = None
            if not super().__exit__(exc_type, exc_value, tb):
                return False
        finally:
            self.expected_regex = _save_re

        exc_value = re.sub(r'(?s)\s+', ' ', str(exc_value))
        if not _save_re.search(exc_value):
            self._raiseFailure(
                '"{}" does not match "{}"'.format(_save_re.pattern, exc_value)
            )
        return True


class TestCase(_unittest.TestCase):
    """A Pyomo-specific class whose instances are single test cases.

    This class derives from unittest.TestCase and provides the following
    additional functionality:
      - additional assertions:
        * :py:meth:`assertStructuredAlmostEqual`

    unittest.TestCase documentation
    -------------------------------
    """

    __doc__ += _unittest.TestCase.__doc__

    def assertStructuredAlmostEqual(
        self,
        first,
        second,
        places=None,
        msg=None,
        delta=None,
        reltol=None,
        abstol=None,
        allow_second_superset=False,
        item_callback=_floatOrCall,
    ):
        assertStructuredAlmostEqual(
            first=first,
            second=second,
            places=places,
            msg=msg,
            delta=delta,
            reltol=reltol,
            abstol=abstol,
            allow_second_superset=allow_second_superset,
            item_callback=item_callback,
            exception=self.failureException,
            formatter=self._formatMessage,
        )

    def assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs):
        """Asserts that the message in a raised exception matches a regex.

        This is a light weight wrapper around
        :py:meth:`unittest.TestCase.assertRaisesRegex` that adds
        handling of a `normalize_whitespace` keyword argument that
        normalizes all consecutive whitespace in the exception message
        to a single space before checking the regular expression.

        Args:
            expected_exception: Exception class expected to be raised.
            expected_regex: Regex (re.Pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertRaisesRegex is used as a context manager.
            normalize_whitespace: Optional bool that, if True, collapses
                    consecutive whitespace (including newlines) into a
                    single space before checking against the regular
                    expression

        """
        normalize_whitespace = kwargs.pop('normalize_whitespace', False)
        if normalize_whitespace:
            contextClass = _AssertRaisesContext_NormalizeWhitespace
        else:
            contextClass = _unittest.case._AssertRaisesContext
        context = contextClass(expected_exception, self, expected_regex)
        return context.handle('assertRaisesRegex', args, kwargs)

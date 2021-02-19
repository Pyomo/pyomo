#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
import logging
import six
import sys

# Import base classes privately (so that we have handles on them)
import pyutilib.th.pyunit as _pyunit
from pyutilib.th.pyunit import unittest as _unittest

# Now, import the base unittest environment.  We will override things
# specifically later
from unittest import *
from pyutilib.th import *

from pyomo.common.collections import Mapping, Sequence
from pyomo.common.tee import capture_output

# This augments the unittest exports with two additional decorators
__all__ = _unittest.__all__ + ['category', 'nottest']

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
    OUT = six.StringIO()
    try:
        with capture_output(OUT):
            result = fcn(*args, **kwargs)
        q.put((resultType, result, OUT.getvalue()))
    except:
        import traceback
        etype, e, tb = sys.exc_info()
        if not isinstance(e, AssertionError):
            e = etype("%s\nOriginal traceback:\n%s" % (
                e, ''.join(traceback.format_tb(tb))))
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


def timeout(seconds, require_fork=False):
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
                raise _unittest.SkipTest(
                    "timeout requires unavailable fork interface")

            q = multiprocessing.Queue()
            if multiprocessing.get_start_method() == 'fork':
                # Option 1: leverage fork if possible.  This minimizes
                # the reliance on serialization and ensures that the
                # wrapped function operates in the same environment.
                _runner.data[q] = (fcn, args, kwargs)
                runner_args = (q, qualname)
            elif (args and fcn.__name__.startswith('test')
                  and _unittest.case.TestCase in args[0].__class__.__mro__):
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
            test_proc = multiprocessing.Process(
                target=_runner, args=runner_args)
            test_proc.daemon = True
            try:
                test_proc.start()
            except:
                if type(runner_args[1]) is tuple:
                    logging.getLogger(__name__).error(
                        "Exception raised spawning timeout subprocess "
                        "on a platform that does not support 'fork'.  "
                        "It is likely that either the wrapped function or "
                        "one of its arguments is not serializable")
                raise
            try:
                resultType, result, stdout = q.get(True, seconds)
            except queue.Empty:
                test_proc.terminate()
                raise TimeoutError(
                    "test timed out after %s seconds" % (seconds,)) from None
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

        elif any(isinstance(_, str) for _ in args):
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

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
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
import sys
import os
import argparse
import subprocess
from io import StringIO


# Now, import the base unittest environment.  We will override things
# specifically later
from unittest import *
import unittest as _unittest

from pyomo.common.collections import Mapping, Sequence
from pyomo.common.tee import capture_output

# This augments the unittest exports with two additional decorators
__all__ = _unittest.__all__ + ['category', 'nottest']

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
    except TypeError:
        return float(val())

def assertStructuredAlmostEqual(first, second,
                                places=None, msg=None, delta=None,
                                reltol=None, abstol=None,
                                allow_second_superset=False,
                                item_callback=_floatOrCall,
                                exception=ValueError,
                                formatter=_defaultFormatter):
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

    Items (entries other than Sequence / Mapping containters, matching
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
        _assertStructuredAlmostEqual(
            first, second, abstol, reltol, not allow_second_superset,
            item_callback, exception)
    except exception as e:
        fail = formatter(
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
        raise exception(fail)


def _assertStructuredAlmostEqual(first, second,
                                 abstol, reltol, exact,
                                 item_callback, exception):
    """Recursive implementation of assertStructuredAlmostEqual"""

    args = (first, second)
    f, s = args
    if all(isinstance(_, Mapping) for _ in args):
        if exact and len(first) != len(second):
            raise exception(
                "mappings are different sizes (%s != %s)" % (
                    len(first),
                    len(second),
                ))
        for key in first:
            if key not in second:
                raise exception(
                    "key (%s) from first not found in second" % (
                        _unittest.case.safe_repr(key),
                    ))
            try:
                _assertStructuredAlmostEqual(
                    first[key], second[key], abstol, reltol, exact,
                    item_callback, exception)
            except exception as e:
                raise exception(
                    "%s\n    Found when comparing key %s" % (
                        str(e), _unittest.case.safe_repr(key)))
        return # PASS!

    elif any(isinstance(_, str) for _ in args):
        if first == second:
            return # PASS!

    elif all(isinstance(_, Sequence) for _ in args):
        # Note that Sequence includes strings
        if exact and len(first) != len(second):
            raise exception(
                "sequences are different sizes (%s != %s)" % (
                    len(first),
                    len(second),
                ))
        for i, (f, s) in enumerate(zip(first, second)):
            try:
                _assertStructuredAlmostEqual(
                    f, s, abstol, reltol, exact, item_callback, exception)
            except exception as e:
                raise exception(
                    "%s\n    Found at position %s" % (str(e), i))
        return # PASS!

    else:
        # Catch things like None, which may cause problems for the
        # item_callback [like float(None)])
        #
        # Test `is` and `==`, but this is not necessarily fatal: we will
        # continue and allow the item_callback to potentially convert
        # the values to be comparable.
        try:
            if first is second or first == second:
                return # PASS!
        except:
            pass
        try:
            f = item_callback(first)
            s = item_callback(second)
            if f == s:
                return
            diff = abs(f - s)
            if abstol is not None and diff <= abstol:
                return # PASS!
            if reltol is not None and diff / max(abs(f), abs(s)) <= reltol:
                return # PASS!
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


def _category_to_tuple(_cat):
    _cat = str(_cat).lower().strip()
    if _cat.endswith('=0') or _cat.endswith('=1'):
        _val = int(_cat[-1])
        _cat = _cat[:-2]
    else:
        _val = 1
    if _cat and _cat[0] in '!~-':
        _val = 1 - _val
        _cat = _cat[1:]
    return _cat, _val

def category(*args, **kwargs):
    # Get the set of categories for this test
    _categories = {}
    for cat in args:
        _cat, _val = _category_to_tuple(cat)
        if not _cat:
            continue
        _categories[_cat] = _val

    # Note: we used to try and short-circuit the nosetest test selection
    # and return test skips for tests that couldn't/wouldn't be run.
    # However, this code was unreliable, as categories could be set by
    # both decorating the TestCase (class) and the function.  As a
    # result, we will just rely on nosetest to do the right thing.

    def _id(func):
        if hasattr(func, '__mro__') and TestCase in func.__mro__:
            # @category() called on a TestCase class
            if len(_categories) > (1 if 'fragile' in _categories else 0):
                for c,v in func.unspecified_categories.items():
                    setattr(func, c, v)
                    _categories.setdefault(c, v)
            default_updates = {}
            for c,v in _categories.items():
                if c in func.unspecified_categories:
                    default_updates[c] = v
                setattr(func, c, v)
            if default_updates:
                for fcn in func.__dict__.values():
                    if hasattr(fcn, '_categories'):
                        for c,v in default_updates.items():
                            if c not in fcn._categories:
                                setattr(fcn, c, v)
        else:
            # This is a (currently unbound) method definition
            if len(_categories) > (1 if 'fragile' in _categories else 0):
                for c,v in TestCase.unspecified_categories.items():
                    setattr(func, c, v)
            for c,v in _categories.items():
                setattr(func, c, v)
            setattr(func, '_categories', _categories)
        return func
    return _id


try:
    from nose.tools import nottest
except ImportError:

    def nottest(func):
        """Decorator to mark a function or method as *not* a test"""
        func.__test__ = False
        return func


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
                raise timeout_raises(
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


class TestCase(_unittest.TestCase):
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
    smoke = 1
    nightly = 1
    expensive = 0
    fragile = 0
    pyomo_unittest = 1
    _default_categories = True
    unspecified_categories = {
        'smoke':0, 'nightly':0, 'expensive':0, 'fragile':0 }


    @staticmethod
    def parse_categories(category_string):
        return tuple(
            tuple(_category_to_tuple(_cat) for _cat in _set.split(','))
            for _set in category_string.split()
        )

    @staticmethod
    def categories_to_string(categories):
        return ' '.join(','.join("%s=%s" % y for y in x) for x in categories)

    def shortDescription(self):
        # Disable nose's use of test docstrings for the test description.
        return None

    def currentTestPassed(self):
        # Note: this only works for Python 3.4+
        return not (self._outcome and any(
            test is self and err for test, err in self._outcome.errors))

    def assertStructuredAlmostEqual(self, first, second,
                                    places=None, msg=None, delta=None,
                                    reltol=None, abstol=None,
                                    allow_second_superset=False,
                                    item_callback=_floatOrCall):
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

def buildParser():
    parser = argparse.ArgumentParser(usage='python -m pyomo.common.unittest [TARGETS] [OPTIONS]')

    parser.add_argument(
        'targets',
        action='store',
        nargs='*',
        default=['pyomo'],
        help='Packages to test')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        dest='verbose',
        help='Verbose output')
    parser.add_argument(
        '--cat',
        '--category',
        action='append',
        dest='cat',
        default=[],
        help='Specify the test category. \
            Can be used several times for multiple categories (e.g., \
            --cat=nightly --cat=smoke).')
    parser.add_argument('--xunit',
        action='store_true',
        dest='xunit',
        help='Enable the nose XUnit plugin')
    parser.add_argument('-x',
        '--stop',
        action='store_true',
        dest='stop',
        help='Stop running tests after the first error or failure.')
    parser.add_argument('--dry-run',
        action='store_true',
        dest='dryrun',
        help='Dry run: collect but do not execute the tests')
    return parser


def runtests(options):

    from pyomo.common.fileutils import PYOMO_ROOT_DIR as basedir, Executable
    env = os.environ.copy()
    os.chdir(basedir)

    print("Running tests in directory %s" % (basedir,))

    if sys.platform.startswith('win'):
        binDir = os.path.join(sys.exec_prefix, 'Scripts')
        nosetests = os.path.join(binDir, 'nosetests.exe')
    else:
        binDir = os.path.join(sys.exec_prefix, 'bin')
        nosetests = os.path.join(binDir, 'nosetests')

    if os.path.exists(nosetests):
        cmd = [nosetests]
    else:
        nose = Executable('nosetests')
        cmd = [sys.executable, nose.path()]

    if (sys.platform.startswith('win') and sys.version_info[0:2] >= (3, 8)):
        #######################################################
        # This option is required due to a (likely) bug within nosetests.
        # Nose is no longer maintained, but this workaround is based on a public forum suggestion:
        #   https://stackoverflow.com/questions/58556183/nose-unittest-discovery-broken-on-python-3-8
        #######################################################
        cmd.append('--traverse-namespace')

    if binDir not in env['PATH']:
        env['PATH'] = os.pathsep.join([binDir, env.get('PATH','')])

    if options.verbose:
        cmd.append('-v')
    if options.stop:
        cmd.append('-x')
    if options.dryrun:
        cmd.append('--collect-only')

    if options.xunit:
        cmd.append('--with-xunit')
        cmd.append('--xunit-file=TEST-pyomo.xml')

    attr = []
    _with_performance = False
    _categories = []
    for x in options.cat:
        _categories.extend( TestCase.parse_categories(x) )

    # If no one specified a category, default to "smoke" (and anything
    # not built on pyomo.common.unittest.TestCase)
    if not _categories:
        _categories = [ (('smoke',1),), (('pyomo_unittest',0),) ]
    # process each category set (that is, each conjunction of categories)
    for _category_set in _categories:
        _attrs = []
        # "ALL" deletes the categories, and just runs everything.  Note
        # that "ALL" disables performance testing
        if ('all', 1) in _category_set:
            _categories = []
            _with_performance = False
            attr = []
            break
        # For each category set, unless the user explicitly says
        # something about fragile, assume that fragile should be
        # EXCLUDED.
        if ('fragile',1) not in _category_set \
           and ('fragile',0) not in _category_set:
            _category_set = _category_set + (('fragile',0),)
        # Process each category in the conjection and add to the nose
        # "attrib" plugin arguments
        for _category, _value in _category_set:
            if not _category:
                continue
            if _value:
                _attrs.append(_category)
            else:
                _attrs.append("(not %s)" % (_category,))
            if _category == 'performance' and _value == 1:
                _with_performance = True
        if _attrs:
            attr.append("--eval-attr=%s" % (' and '.join(_attrs),))
    cmd.extend(attr)
    if attr:
        print(" ... for test categor%s: %s" %
              ('y' if len(attr)<=2 else 'ies',
               ' '.join(attr[1::2])))

    if _with_performance:
        cmd.append('--with-testdata')
        env['NOSE_WITH_TESTDATA'] = '1'
        env['NOSE_WITH_FORCED_GC'] = '1'

    cmd.extend(options.targets)
    print(cmd)
    print("Running...\n    %s\n" % (
            ' '.join( (x if ' ' not in x else '"'+x+'"') for x in cmd ), ))
    sys.stdout.flush()
    result = subprocess.run(cmd, env=env)
    rc = result.returncode
    return rc


if __name__ == '__main__':
    parser = buildParser()
    options = parser.parse_args()
    sys.exit(runtests(options))

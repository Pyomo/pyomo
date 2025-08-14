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
#
#  Part of this module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO


# Now, import the base unittest environment.  We will override things
# specifically later
from unittest import *
import unittest as _unittest

from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version, multiprocessing
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output

from unittest import mock

# We defer this import so that we don't add a hard dependence on pytest.
# Note that importing test modules may cause this import to be resolved
# (and then enforce a strict dependence on pytest)
pytest, pytest_available = attempt_import('pytest')


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

        `abs(first - second) / max(abs(first), abs(second))`

    only when `first != second` (thereby avoiding divide-by-zero errors).

    Items (entries other than Sequence / Mapping containers, matching
    strings, and items that satisfy `first is second`) are passed to the
    `item_callback` before testing equality and relative tolerances.

    Raises `exception` if `first` and `second` are not equal within
    tolerance.

    Parameters
    ----------
    first :
        the first value to compare

    second :
        the second value to compare

    places : int
        `first` and `second` are considered equivalent if their
        difference is between `places` decimal places; equivalent to
        `abstol = 10**-places` (included for compatibility with
        assertAlmostEqual)

    msg : str
        the message to raise on failure

    delta : float
        alias for `abstol`

    abstol : float
        the absolute tolerance.  `first` and `second` are considered
        equivalent if their absolute difference is less than `abstol`

    reltol : float
        the relative tolerance.  `first` and `second` are considered
        equivalent if their absolute difference divided by the
        largest of `first` and `second` is less than `reltol`

    allow_second_superset : bool
        If True, then extra entries in containers found on second
        will not trigger a failure.

    item_callback : function
        items (other than Sequence / Mapping containers, matching
        strings, and items satisfying `is`) are passed to this callback
        to generate the (nominally floating point) value to use for
        comparison.

    exception : Exception
        exception to raise when `first` is not 'almost equal' to `second`.

    formatter : function
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


def _runner(pipe, qualname):
    "Utility wrapper for running functions, used by timeout()"
    resultType = _RunnerResult.call
    if pipe in _runner.data:
        fcn, args, kwargs = _runner.data[pipe]
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
    try:
        with capture_output() as OUT:
            result = fcn(*args, **kwargs)
        pipe.send((resultType, result, OUT.getvalue()))
    except:
        import traceback

        etype, e, tb = sys.exc_info()
        if not isinstance(e, AssertionError):
            e = etype(
                "%s\nOriginal traceback:\n%s" % (e, ''.join(traceback.format_tb(tb)))
            )
        pipe.send((_RunnerResult.exception, e, OUT.getvalue()))
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
    :class:`TimeoutError` will be raised.

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
    .. doctest::
       :skipif: multiprocessing.get_start_method() != 'fork'

       >>> import pyomo.common.unittest as unittest
       >>> @unittest.timeout(1)
       ... def test_function():
       ...     return 42
       >>> test_function()
       42

    .. doctest::
       :skipif: multiprocessing.get_start_method() != 'fork'

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
    import queue

    def timeout_decorator(fcn):
        @functools.wraps(fcn)
        def test_timer(*args, **kwargs):
            qualname = '%s.%s' % (fcn.__module__, fcn.__qualname__)
            # If qualname is in the data dict, then we are in the child
            # process and are being asked to run the wrapped function.
            if qualname in _runner.data:
                return fcn(*args, **kwargs)
            # Parent process: spawn a subprocess to execute the wrapped
            # function and monitor for timeout
            if require_fork and multiprocessing.get_start_method() != 'fork':
                raise _unittest.SkipTest(
                    "timeout() requires unavailable fork interface"
                )

            pipe_recv, pipe_send = multiprocessing.Pipe(False)
            if multiprocessing.get_start_method() == 'fork':
                # Option 1: leverage fork if possible.  This minimizes
                # the reliance on serialization and ensures that the
                # wrapped function operates in the same environment.
                _runner.data[pipe_send] = (fcn, args, kwargs)
                runner_arg = qualname
            elif (
                args
                and fcn.__name__.startswith('test')
                and _unittest.case.TestCase in args[0].__class__.__mro__
            ):
                # Option 2: this is wrapping a unittest.  Re-run
                # unittest in the child process with this function as
                # the sole target.  This ensures that things like setUp
                # and tearDown are correctly called.
                runner_arg = qualname
            else:
                # Option 3: attempt to serialize the function and all
                # arguments and send them to the (spawned) child
                # process.  The wrapped function cannot count on any
                # environment configuration that it does not set up
                # itself.
                runner_arg = (qualname, test_timer, args, kwargs)
            test_proc = multiprocessing.Process(
                target=_runner, args=(pipe_send, runner_arg)
            )
            # Set daemon: if the parent process is killed, the child
            # process should be killed and collected.
            test_proc.daemon = True
            try:
                test_proc.start()
            except:
                if type(runner_arg) is tuple:
                    logging.getLogger(__name__).error(
                        "Exception raised spawning timeout() subprocess "
                        "on a platform that does not support 'fork'.  "
                        "It is likely that either the wrapped function or "
                        "one of its arguments is not serializable"
                    )
                raise
            try:
                if pipe_recv.poll(seconds):
                    resultType, result, stdout = pipe_recv.recv()
                else:
                    test_proc.terminate()
                    raise timeout_raises(
                        "test timed out after %s seconds" % (seconds,)
                    ) from None
            finally:
                _runner.data.pop(pipe_send, None)
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

    * additional assertions:
       - :py:meth:`~TestCase.assertStructuredAlmostEqual`
       - :py:meth:`assertExpressionsEqual`
       - :py:meth:`assertExpressionsStructurallyEqual`

    * updated assertions:
       - :py:meth:`assertRaisesRegex`

    :py:class:`unittest.TestCase` documentation
    -------------------------------------------
    """

    # Note that the current unittest.TestCase documentation generates
    # sphinx warnings.  We will clean up that documentation to suppress
    # the warnings.
    __doc__ += (
        re.sub(
            r'^( +)(\* +[^:]+:) *',
            r'\n\1\2\n\1    ',
            _unittest.TestCase.__doc__.rstrip(),
            flags=re.M,
        )
        + "\n\n"
    )

    # By default, we always want to spend the time to create the full
    # diff of the test reault and the baseline
    maxDiff = None

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
        # Note: __doc__ copied from assertStructuredAlmostEqual below
        #
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

        Parameters
        ----------
        expected_exception : Exception
            Exception class expected to be raised.

        expected_regex : `re.Pattern` or str
            Regular expression expected to be found in error message.

        *args :
            Function to be called and extra positional args.

        **kwargs :
            Extra keyword args.

        msg : str
            Optional message used in case of failure. Can only be used
            when assertRaisesRegex is used as a context manager.

        normalize_whitespace : bool, default=False
            If True, collapses consecutive whitespace (including
            newlines) into a single space before checking against the
            regular expression

        """
        normalize_whitespace = kwargs.pop('normalize_whitespace', False)
        if normalize_whitespace:
            contextClass = _AssertRaisesContext_NormalizeWhitespace
        else:
            contextClass = _unittest.case._AssertRaisesContext
        context = contextClass(expected_exception, self, expected_regex)
        return context.handle('assertRaisesRegex', args, kwargs)

    def assertExpressionsEqual(self, a, b, include_named_exprs=True, places=None):
        """Assert that two Pyomo expressions are equal.

        This converts the expressions `a` and `b` into prefix notation
        and then compares the resulting lists.  All nodes in the tree
        are compared using py:meth:`assertEqual` (or
        py:meth:`assertAlmostEqual`)

        Parameters
        ----------
        a: ExpressionBase or native type

        b: ExpressionBase or native type

        include_named_exprs : bool
            If True (the default), the comparison expands all named
            expressions when generating the prefix notation

        places : float
            Number of decimal places required for equality of floating
            point numbers in the expression. If None (the default), the
            expressions must be exactly equal.

        """
        from pyomo.core.expr.compare import assertExpressionsEqual

        return assertExpressionsEqual(self, a, b, include_named_exprs, places)

    def assertExpressionsStructurallyEqual(
        self, a, b, include_named_exprs=True, places=None
    ):
        """Assert that two Pyomo expressions are structurally equal.

        This converts the expressions `a` and `b` into prefix notation
        and then compares the resulting lists.  Operators and
        (non-native type) leaf nodes in the prefix representation are
        converted to strings before comparing (so that things like
        variables can be compared across clones or pickles)

        Parameters
        ----------
        a: ExpressionBase or native type

        b: ExpressionBase or native type

        include_named_exprs: bool
            If True (the default), the comparison expands all named
            expressions when generating the prefix notation

        places: float
            Number of decimal places required for equality of floating
            point numbers in the expression. If None (the default), the
            expressions must be exactly equal.

        """
        from pyomo.core.expr.compare import assertExpressionsStructurallyEqual

        return assertExpressionsStructurallyEqual(
            self, a, b, include_named_exprs, places
        )


TestCase.assertStructuredAlmostEqual.__doc__ = re.sub(
    'exception :.*', '', assertStructuredAlmostEqual.__doc__, flags=re.S
)


class BaselineTestDriver(object):
    """Generic driver for performing baseline tests in bulk

    This test driver was originally crafted for testing the examples in
    the Pyomo Book, and has since been generalized to reuse in testing
    ".. literalinclude:" examples from the Online Docs.

    We expect that consumers of this class will derive from both this
    class and `pyomo.common.unittest.TestCase`, and then use
    `parameterized` to declare tests that call either the
    :py:meth:`python_test_driver` or :py:meth:`shell_test_driver`
    methods.

    Note that derived classes must declare two class attributes:

    Class Attributes
    ----------------
    solver_dependencies: Dict[str, List[str]]

        maps the test name to a list of required solvers.  If any solver
        is not available, then the test will be skipped.

    package_dependencies: Dict[str, List[str]]

        maps the test name to a list of required modules.  If any module
        is not available, then the test will be skipped.

    """

    @staticmethod
    def custom_name_func(test_func, test_num, test_params):
        func_name = test_func.__name__
        return "test_%s_%s" % (test_params.args[0], func_name[-2:])

    def __init__(self, test):
        # Finalize the class, if necessary...
        if getattr(self.__class__, 'solver_available', None) is None:
            self.initialize_dependencies()
        super().__init__(test)

    def initialize_dependencies(self):
        # Note: as a rule, pyomo.common is not allowed to import from
        # the rest of Pyomo.  we permit it here because a) this is not
        # at module scope, and b) there is really no better / more
        # logical place in pyomo to put this code.
        from pyomo.opt import check_available_solvers

        cls = self.__class__
        #
        # Initialize the availability data
        #
        solvers_used = set(sum(list(cls.solver_dependencies.values()), []))
        available_solvers = check_available_solvers(*solvers_used)
        cls.solver_available = {
            solver_: (solver_ in available_solvers) for solver_ in solvers_used
        }

        cls.package_available = {}
        cls.package_modules = {}
        packages_used = set(sum(list(cls.package_dependencies.values()), []))
        for package_ in packages_used:
            pack, pack_avail = attempt_import(package_, defer_import=False)
            cls.package_available[package_] = pack_avail
            cls.package_modules[package_] = pack

    @classmethod
    def _find_tests(cls, test_dirs, pattern):
        test_tuples = []
        for testdir in test_dirs:
            # Find all pattern files in the test directory and any immediate
            # sub-directories
            for fname in list(glob.glob(os.path.join(testdir, pattern))) + list(
                glob.glob(os.path.join(testdir, '*', pattern))
            ):
                test_file = os.path.abspath(fname)
                bname = os.path.basename(test_file)
                dir_ = os.path.dirname(test_file)
                name = os.path.splitext(bname)[0]
                tname = os.path.basename(dir_) + '_' + name

                suffix = None
                # Look for txt and yml file names matching py file names. Add
                # a test for any found
                for suffix_ in ['.txt', '.yml']:
                    if os.path.exists(os.path.join(dir_, name + suffix_)):
                        suffix = suffix_
                        break
                if suffix is not None:
                    tname = tname.replace('-', '_')
                    tname = tname.replace('.', '_')

                    # Create list of tuples with (test_name, test_file, baseline_file)
                    test_tuples.append(
                        (tname, test_file, os.path.join(dir_, name + suffix))
                    )

        # Ensure a deterministic test ordering
        test_tuples.sort()
        return test_tuples

    @classmethod
    def gather_tests(cls, test_dirs):
        # Find all .sh files in the test directories
        sh_test_tuples = cls._find_tests(test_dirs, '*.sh')

        # Find all .py files in the test directories
        py_test_tuples = cls._find_tests(test_dirs, '*.py')

        # If there is both a .py and a .sh, defer to the sh
        sh_files = set(map(operator.itemgetter(1), sh_test_tuples))
        py_test_tuples = list(
            filter(lambda t: t[1][:-3] + '.sh' not in sh_files, py_test_tuples)
        )

        return py_test_tuples, sh_test_tuples

    def check_skip(self, name):
        """
        Return a boolean if the test should be skipped
        """

        if name in self.solver_dependencies:
            solvers_ = self.solver_dependencies[name]
            if not all([self.solver_available[i] for i in solvers_]):
                # Skip the test because a solver is not available
                _missing = []
                for i in solvers_:
                    if not self.solver_available[i]:
                        _missing.append(i)
                return "Solver%s %s %s not available" % (
                    's' if len(_missing) > 1 else '',
                    ", ".join(_missing),
                    'are' if len(_missing) > 1 else 'is',
                )

        if name in self.package_dependencies:
            packages_ = self.package_dependencies[name]
            if not all([self.package_available[i] for i in packages_]):
                # Skip the test because a package is not available
                _missing = []
                for i in packages_:
                    if not self.package_available[i]:
                        _missing.append(i)
                return "Package%s %s %s not available" % (
                    's' if len(_missing) > 1 else '',
                    ", ".join(_missing),
                    'are' if len(_missing) > 1 else 'is',
                )

            # This is a hack, xlrd dropped support for .xlsx files in 2.0.1 which
            # causes problems with older versions of Pandas<=1.1.5 so skipping
            # tests requiring both these packages when incompatible versions are found
            if (
                'pandas' in self.package_dependencies[name]
                and 'xlrd' in self.package_dependencies[name]
            ):
                if check_min_version(
                    self.package_modules['xlrd'], '2.0.1'
                ) and not check_min_version(self.package_modules['pandas'], '1.1.6'):
                    return "Incompatible versions of xlrd and pandas"

        return False

    def filter_fcn(self, line):
        """
        Ignore certain text when comparing output with baseline
        """
        for field in (
            '[',
            'password:',
            'http:',
            'Job ',
            'Importing module',
            'Function',
            'File',
            'Matplotlib',
            'Memory:',
            '-------',
            '=======',
            '    ^',
        ):
            if line.startswith(field):
                return True
        for field in (
            'Total CPU',
            'Ipopt',
            'license',
            #'Status: optimal',
            #'Status: feasible',
            'time:',
            'Time:',
            'with format cpxlp',
            'usermodel = <module',
            'execution time=',
            'Solver results file:',
            'TokenServer',
            # next 6 patterns ignore entries in pstats reports:
            'function calls',
            'List reduced',
            '.py:',  # timing/profiling output
            ' {built-in method',
            ' {method',
            ' {pyomo.core.expr.numvalue.as_numeric}',
            ' {gurobipy.',
        ):
            if field in line:
                return True
        return False

    def filter_file_contents(self, lines, abstol=None):
        _numpy_scalar_re = re.compile(r'np.(int|float)\d+\(([^\)]+)\)')
        filtered = []
        deprecated = None
        for line in lines:
            # Ignore all deprecation warnings
            if line.startswith('WARNING: DEPRECATED:'):
                deprecated = ''
            if deprecated is not None:
                deprecated += line
                if re.search(r'\(called\s+from[^)]+\)', deprecated):
                    deprecated = None
                continue

            if not line or self.filter_fcn(line):
                continue

            # Strip off beginning of lines giving time in seconds
            # Needed for the performance chapter tests
            if "seconds" in line:
                s = line.find("seconds") + 7
                line = line[s:]

            item_list = []
            items = line.strip().split()
            for i in items:
                # Split up lists, dicts, and sets
                while i and i[0] in '[{':
                    item_list.append(i[0])
                    i = i[1:]
                tail = []
                while i and i[-1] in ',:]}':
                    tail.append(i[-1])
                    i = i[:-1]

                # A few substitutions to get tests passing on pypy3
                if ".inf" in i:
                    i = i.replace(".inf", "inf")
                if "null" in i:
                    i = i.replace("null", "None")

                try:
                    # Numpy 2.x changed the repr for scalars.  Convert
                    # the new scalar reprs back to the original (which
                    # were indistinguishable from python floats/ints)
                    np_match = _numpy_scalar_re.match(i)
                    if np_match:
                        item_list.append(float(np_match.group(2)))
                    else:
                        item_list.append(float(i))
                except:
                    item_list.append(i)
                if tail:
                    tail.reverse()
                    item_list.extend(tail)

            # We can get printed results objects where the baseline is
            # exactly 0 (and omitted) and the test is slightly non-zero.
            # We will look for the pattern of values printed from
            # results objects and remote them if they are within
            # tolerance of 0
            if (
                len(item_list) == 3
                and item_list[0] == 'Value'
                and item_list[1] == ':'
                and type(item_list[2]) is float
                and abs(item_list[2]) < (abstol or 0)
                and len(filtered[-1]) == 2
                and filtered[-1][1] == ':'
            ):
                filtered.pop()
            else:
                filtered.append(item_list)

        return filtered

    def compare_baseline(self, test_output, baseline, abstol=1e-6, reltol=1e-8):
        # Filter files independently and then compare filtered contents
        out_filtered = self.filter_file_contents(
            test_output.strip().split('\n'), abstol
        )
        base_filtered = self.filter_file_contents(baseline.strip().split('\n'), abstol)

        try:
            self.assertStructuredAlmostEqual(
                out_filtered,
                base_filtered,
                abstol=abstol,
                reltol=reltol,
                allow_second_superset=False,
            )
            return True
        except self.failureException:
            # Print helpful information when file comparison fails
            print('---------------------------------')
            print('BASELINE FILE')
            print('---------------------------------')
            print(baseline)
            print('=================================')
            print('---------------------------------')
            print('TEST OUTPUT FILE')
            print('---------------------------------')
            print(test_output)
            raise

    def python_test_driver(self, tname, test_file, base_file):
        bname = os.path.basename(test_file)
        _dir = os.path.dirname(test_file)

        skip_msg = self.check_skip('test_' + tname)
        if skip_msg:
            raise _unittest.SkipTest(skip_msg)

        with open(base_file, 'r') as FILE:
            baseline = FILE.read()

        cwd = os.getcwd()
        try:
            os.chdir(_dir)
            # This is roughly equivalent to:
            #    subprocess.run([sys.executable, bname],
            #                   stdout=f, stderr=f, cwd=_dir)
            with capture_output(None, True) as OUT:
                # Note: we want LoggingIntercept to log to the
                # *current* stdout (which is the TeeStream from
                # capture_output).  This ensures that log messages and
                # normal output appear in the correct order.
                with LoggingIntercept(sys.stdout, formatter=pyomo_formatter):
                    import_file(bname, infer_package=False, module_name='__main__')
        finally:
            os.chdir(cwd)

        try:
            self.compare_baseline(OUT.getvalue(), baseline)
        except:
            if os.environ.get('PYOMO_TEST_UPDATE_BASELINES', None):
                with open(base_file, 'w') as FILE:
                    FILE.write(OUT.getvalue())
            raise

    def shell_test_driver(self, tname, test_file, base_file):
        bname = os.path.basename(test_file)
        _dir = os.path.dirname(test_file)

        skip_msg = self.check_skip('test_' + tname)
        if skip_msg:
            raise _unittest.SkipTest(skip_msg)

        # Skip all shell tests on Windows.
        if os.name == 'nt':
            raise _unittest.SkipTest("Shell tests are not runnable on Windows")

        with open(base_file, 'r') as FILE:
            baseline = FILE.read()

        cwd = os.getcwd()
        try:
            os.chdir(_dir)
            _env = os.environ.copy()
            _env['PATH'] = os.pathsep.join(
                [os.path.dirname(sys.executable), _env['PATH']]
            )
            rc = subprocess.run(
                ['bash', bname],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=_dir,
                env=_env,
            )
        finally:
            os.chdir(cwd)

        try:
            self.compare_baseline(rc.stdout.decode(), baseline)
        except:
            if os.environ.get('PYOMO_TEST_UPDATE_BASELINES', None):
                with open(base_file, 'w') as FILE:
                    FILE.write(rc.stdout.decode())
            raise

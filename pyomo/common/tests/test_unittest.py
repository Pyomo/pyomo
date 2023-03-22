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

import datetime
import multiprocessing
from io import StringIO
import time

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Var, Param


@unittest.timeout(10)
def short_sleep():
    return 42


@unittest.timeout(0.01)
def long_sleep():
    time.sleep(1)
    return 42


@unittest.timeout(10)
def raise_exception():
    foo.bar


@unittest.timeout(10)
def fail():
    raise AssertionError("0 != 1")


class TestPyomoUnittest(unittest.TestCase):
    def test_assertStructuredAlmostEqual_comparison(self):
        a = 1
        b = 1
        self.assertStructuredAlmostEqual(a, b)
        # default relative tolerance is 1e-7.  This should have a reltol
        # of "exactly" 1e-7, but due to roundoff error, it is not.  The
        # choice of 9.999e-8 and 9.999-e7 is specifically so that
        # roundoff error doesn't cause tests to fail
        b -= 9.999e-8
        self.assertStructuredAlmostEqual(a, b)
        b -= 9.999e-8
        with self.assertRaisesRegex(self.failureException, '1 !~= 0.9999'):
            self.assertStructuredAlmostEqual(a, b)

        b = 1
        self.assertStructuredAlmostEqual(a, b, reltol=1e-6)
        b -= 9.999e-7
        self.assertStructuredAlmostEqual(a, b, reltol=1e-6)
        b -= 9.999e-7
        with self.assertRaisesRegex(self.failureException, '1 !~= 0.999'):
            self.assertStructuredAlmostEqual(a, b, reltol=1e-6)

        b = 1
        self.assertStructuredAlmostEqual(a, b, places=6)
        b -= 9.999e-7
        self.assertStructuredAlmostEqual(a, b, places=6)
        b -= 9.999e-7
        with self.assertRaisesRegex(self.failureException, '1 !~= 0.999'):
            self.assertStructuredAlmostEqual(a, b, places=6)

        with self.assertRaisesRegex(self.failureException, '10 !~= 10.01'):
            self.assertStructuredAlmostEqual(10, 10.01, abstol=1e-3)
        self.assertStructuredAlmostEqual(10, 10.01, reltol=1e-3)
        with self.assertRaisesRegex(self.failureException, '10 !~= 10.01'):
            self.assertStructuredAlmostEqual(10, 10.01, delta=1e-3)

    def test_assertStructuredAlmostEqual_nan(self):
        a = float('nan')
        b = float('nan')
        self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_errorChecking(self):
        with self.assertRaisesRegex(
            ValueError, "Cannot specify more than one of {places, delta, abstol}"
        ):
            self.assertStructuredAlmostEqual(1, 1, places=7, delta=1)

    def test_assertStructuredAlmostEqual_str(self):
        self.assertStructuredAlmostEqual("hi", "hi")
        with self.assertRaisesRegex(self.failureException, "'hi' !~= 'hello'"):
            self.assertStructuredAlmostEqual("hi", "hello")
        with self.assertRaisesRegex(self.failureException, r"'hi' !~= \['h',"):
            self.assertStructuredAlmostEqual("hi", ['h', 'i'])

    def test_assertStructuredAlmostEqual_othertype(self):
        a = datetime.datetime(1, 1, 1)
        b = datetime.datetime(1, 1, 1)
        self.assertStructuredAlmostEqual(a, b)
        b = datetime.datetime(1, 1, 2)
        with self.assertRaisesRegex(self.failureException, "datetime.* !~= datetime"):
            self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_list(self):
        a = [1, 2]
        b = [1, 2, 3]
        with self.assertRaisesRegex(
            self.failureException, r'sequences are different sizes \(2 != 3\)'
        ):
            self.assertStructuredAlmostEqual(a, b)
        self.assertStructuredAlmostEqual(a, b, allow_second_superset=True)
        a.append(3)

        self.assertStructuredAlmostEqual(a, b)
        b[1] -= 1.999e-7
        self.assertStructuredAlmostEqual(a, b)
        b[1] -= 1.999e-7
        with self.assertRaisesRegex(self.failureException, '2 !~= 1.999'):
            self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_dict(self):
        a = {1: 2, 3: 4}
        b = {1: 2, 3: 4, 5: 6}
        with self.assertRaisesRegex(
            self.failureException, r'mappings are different sizes \(2 != 3\)'
        ):
            self.assertStructuredAlmostEqual(a, b)
        self.assertStructuredAlmostEqual(a, b, allow_second_superset=True)
        a[5] = 6

        self.assertStructuredAlmostEqual(a, b)
        b[1] -= 1.999e-7
        self.assertStructuredAlmostEqual(a, b)
        b[1] -= 1.999e-7
        with self.assertRaisesRegex(self.failureException, '2 !~= 1.999'):
            self.assertStructuredAlmostEqual(a, b)

        del b[1]
        b[6] = 6
        with self.assertRaisesRegex(
            self.failureException, r'key \(1\) from first not found in second'
        ):
            self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_nested(self):
        a = {1.1: [1, 2, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
        b = {1.1: [1, 2, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
        self.assertStructuredAlmostEqual(a, b)
        b[1.1][2] -= 1.999e-7
        b[3][1] -= 9.999e-8
        self.assertStructuredAlmostEqual(a, b)
        b[1.1][2] -= 1.999e-7
        with self.assertRaisesRegex(self.failureException, '3 !~= 2.999'):
            self.assertStructuredAlmostEqual(a, b)

    def test_assertStructuredAlmostEqual_numericvalue(self):
        m = ConcreteModel()
        m.v = Var(initialize=2.0)
        m.p = Param(initialize=2.0)
        a = {1.1: [1, m.p, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
        b = {1.1: [1, m.v, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
        self.assertStructuredAlmostEqual(a, b)
        m.v.set_value(m.v.value - 1.999e-7)
        self.assertStructuredAlmostEqual(a, b)
        m.v.set_value(m.v.value - 1.999e-7)
        with self.assertRaisesRegex(self.failureException, '2.0 !~= 1.999'):
            self.assertStructuredAlmostEqual(a, b)

    def test_timeout_fcn_call(self):
        self.assertEqual(short_sleep(), 42)
        with self.assertRaisesRegex(TimeoutError, 'test timed out after 0.01 seconds'):
            long_sleep()
        with self.assertRaisesRegex(
            NameError, r"name 'foo' is not defined\s+Original traceback:"
        ):
            raise_exception()
        with self.assertRaisesRegex(AssertionError, r"^0 != 1$"):
            fail()

    @unittest.timeout(10)
    def test_timeout(self):
        self.assertEqual(0, 0)

    @unittest.expectedFailure
    @unittest.timeout(0.01)
    def test_timeout_timeout(self):
        time.sleep(1)
        self.assertEqual(0, 1)

    @unittest.timeout(10)
    def test_timeout_skip(self):
        if TestPyomoUnittest.test_timeout_skip.skip:
            self.skipTest("Skipping this test")
        self.assertEqual(0, 1)

    test_timeout_skip.skip = True

    def test_timeout_skip_fails(self):
        try:
            with self.assertRaisesRegex(unittest.SkipTest, r"Skipping this test"):
                self.test_timeout_skip()
            TestPyomoUnittest.test_timeout_skip.skip = False
            with self.assertRaisesRegex(AssertionError, r"0 != 1"):
                self.test_timeout_skip()
        finally:
            TestPyomoUnittest.test_timeout_skip.skip = True

    @unittest.timeout(10)
    def bound_function(self):
        self.assertEqual(0, 0)

    def test_bound_function(self):
        if multiprocessing.get_start_method() == 'fork':
            self.bound_function()
            return
        LOG = StringIO()
        with LoggingIntercept(LOG):
            with self.assertRaises((TypeError, EOFError, AttributeError)):
                self.bound_function()
        self.assertIn("platform that does not support 'fork'", LOG.getvalue())
        self.assertIn("one of its arguments is not serializable", LOG.getvalue())

    @unittest.timeout(10, require_fork=True)
    def bound_function_require_fork(self):
        self.assertEqual(0, 0)

    def test_bound_function_require_fork(self):
        if multiprocessing.get_start_method() == 'fork':
            self.bound_function_require_fork()
            return
        with self.assertRaisesRegex(
            unittest.SkipTest, "timeout requires unavailable fork interface"
        ):
            self.bound_function_require_fork()


if __name__ == '__main__':
    unittest.main()

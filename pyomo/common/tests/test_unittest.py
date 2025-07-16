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

import datetime
import multiprocessing
import os
import time

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
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
        self.assertEqual(0, 0)

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
        with LoggingIntercept() as LOG:
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
            unittest.SkipTest, r"timeout\(\) requires unavailable fork interface"
        ):
            self.bound_function_require_fork()


baseline = """
[    0.00] Setting up Pyomo environment
[    0.00] Applying Pyomo preprocessing actions
[    0.00] Creating model
[    0.00] Applying solver
[    0.05] Processing results
    Number of solutions: 1
    Solution Information
      Gap: None
      Status: optimal
      Function Value: -9.99943939749e-05
    Solver results file: results.yml
[    0.05] Applying Pyomo postprocessing actions
[    0.05] Pyomo Finished
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Problem Information
# ----------------------------------------------------------
Problem:
- Lower bound: -inf
  Upper bound: inf
  Number of objectives: 1
  Number of constraints: 3
  Number of variables: 3
  Sense: unknown
# ----------------------------------------------------------
#   Solver Information
# ----------------------------------------------------------
Solver:
- Status: ok
  Message: Ipopt 3.12.3\x3a Optimal Solution Found
  Termination condition: optimal
  Id: 0
  Error rc: 0
  Time: 0.0408430099487
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution:
- number of solutions: 1
  number of solutions displayed: 1
- Gap: None
  Status: optimal
  Message: Ipopt 3.12.3\x3a Optimal Solution Found
  Objective:
    f1:
      Value: -9.99943939749e-05
  Variable:
    compl.v:
      Value: 9.99943939749e-05
    y:
      Value: 9.99943939749e-05
  Constraint: No values
"""

pass_ref = """
[    0.00] Setting up Pyomo environment
[    0.00] Applying Pyomo preprocessing actions
WARNING: DEPRECATED: The Model.preprocess() method is deprecated and no longer
    performs any actions  (deprecated in 6.0) (called from <stdin>:1)
[    0.00] Creating model
[    0.01] Applying solver
[    0.06] Processing results
    Number of solutions: 1
    Solution Information
      Gap: None
      Status: optimal
      Function Value: -0.00010001318188373491
    Solver results file: results.yml
[    0.06] Applying Pyomo postprocessing actions
[    0.06] Pyomo Finished
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Problem Information
# ----------------------------------------------------------
Problem:
- Lower bound: -inf
  Upper bound: inf
  Number of objectives: 1
  Number of constraints: 3
  Number of variables: 3
  Sense: unknown
# ----------------------------------------------------------
#   Solver Information
# ----------------------------------------------------------
Solver:
- Status: ok
  Message: Ipopt 3.14.13\x3a Optimal Solution Found
  Termination condition: optimal
  Id: 0
  Error rc: 0
  Time: 0.04224729537963867
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution:
- number of solutions: 1
  number of solutions displayed: 1
- Gap: None
  Status: optimal
  Message: Ipopt 3.14.13\x3a Optimal Solution Found
  Objective:
    f1:
      Value: -0.00010001318188373491
  Variable:
    compl.v:
      Value: 9.99943939749205e-05
    x:
      Value: -9.39395440720558e-09
    y:
      Value: 9.99943939749205e-05
  Constraint: No values

"""

fail_ref = """
[    0.00] Setting up Pyomo environment
[    0.00] Applying Pyomo preprocessing actions
[    0.00] Creating model
[    0.01] Applying solver
[    0.06] Processing results
    Number of solutions: 1
    Solution Information
      Gap: None
      Status: optimal
      Function Value: -0.00010001318188373491
    Solver results file: results.yml
[    0.06] Applying Pyomo postprocessing actions
[    0.06] Pyomo Finished
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Problem Information
# ----------------------------------------------------------
Problem:
- Lower bound: -inf
  Upper bound: inf
  Number of objectives: 1
  Number of constraints: 3
  Number of variables: 3
  Sense: unknown
# ----------------------------------------------------------
#   Solver Information
# ----------------------------------------------------------
Solver:
- Status: ok
  Message: Ipopt 3.14.13\x3a Optimal Solution Found
  Termination condition: optimal
  Id: 0
  Error rc: 0
  Time: 0.04224729537963867
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution:
- number of solutions: 1
  number of solutions displayed: 1
- Gap: None
  Status: optimal
  Message: Ipopt 3.14.13\x3a Optimal Solution Found
  Objective:
    f1:
      Value: -0.00010001318188373491
  Variable:
    compl.v:
      Value: 9.79943939749205e-05
    x:
      Value: -9.39395440720558e-09
    y:
      Value: 9.99943939749205e-05
  Constraint: No values

"""


class TestBaselineTestDriver(unittest.BaselineTestDriver, unittest.TestCase):
    solver_dependencies = {}
    package_dependencies = {}

    def test_baseline_pass(self):
        self.compare_baseline(pass_ref, baseline, abstol=1e-6)

        with self.assertRaises(self.failureException):
            with capture_output() as OUT:
                self.compare_baseline(pass_ref, baseline, None)
        self.assertEqual(
            OUT.getvalue(),
            f"""---------------------------------
BASELINE FILE
---------------------------------
{baseline}
=================================
---------------------------------
TEST OUTPUT FILE
---------------------------------
{pass_ref}
""",
        )

    def test_baseline_fail(self):
        with self.assertRaises(self.failureException):
            with capture_output() as OUT:
                self.compare_baseline(fail_ref, baseline)
        self.assertEqual(
            OUT.getvalue(),
            f"""---------------------------------
BASELINE FILE
---------------------------------
{baseline}
=================================
---------------------------------
TEST OUTPUT FILE
---------------------------------
{fail_ref}
""",
        )

    def test_testcase_collection(self):
        with TempfileManager.new_context() as TMP:
            tmpdir = TMP.create_tempdir()
            for fname in (
                'a.py',
                'b.py',
                'b.txt',
                'c.py',
                'c.sh',
                'c.yml',
                'd.sh',
                'd.txt',
                'e.sh',
            ):
                with open(os.path.join(tmpdir, fname), 'w'):
                    pass

            py_tests, sh_tests = unittest.BaselineTestDriver.gather_tests([tmpdir])
            self.assertEqual(
                py_tests,
                [
                    (
                        os.path.basename(tmpdir) + '_b',
                        os.path.join(tmpdir, 'b.py'),
                        os.path.join(tmpdir, 'b.txt'),
                    )
                ],
            )
            self.assertEqual(
                sh_tests,
                [
                    (
                        os.path.basename(tmpdir) + '_c',
                        os.path.join(tmpdir, 'c.sh'),
                        os.path.join(tmpdir, 'c.yml'),
                    ),
                    (
                        os.path.basename(tmpdir) + '_d',
                        os.path.join(tmpdir, 'd.sh'),
                        os.path.join(tmpdir, 'd.txt'),
                    ),
                ],
            )

            self.python_test_driver(*py_tests[0])

            _update_baselines = os.environ.pop('PYOMO_TEST_UPDATE_BASELINES', None)
            try:
                with open(os.path.join(tmpdir, 'b.py'), 'w') as FILE:
                    FILE.write('print("Hello, World")\n')

                with self.assertRaises(self.failureException):
                    self.python_test_driver(*py_tests[0])
                with open(os.path.join(tmpdir, 'b.txt'), 'r') as FILE:
                    self.assertEqual(FILE.read(), "")

                os.environ['PYOMO_TEST_UPDATE_BASELINES'] = '1'

                with self.assertRaises(self.failureException):
                    self.python_test_driver(*py_tests[0])
                with open(os.path.join(tmpdir, 'b.txt'), 'r') as FILE:
                    self.assertEqual(FILE.read(), "Hello, World\n")

            finally:
                os.environ.pop('PYOMO_TEST_UPDATE_BASELINES', None)
                if _update_baselines is not None:
                    os.environ['PYOMO_TEST_UPDATE_BASELINES'] = _update_baselines

            self.shell_test_driver(*sh_tests[1])
            _update_baselines = os.environ.pop('PYOMO_TEST_UPDATE_BASELINES', None)
            try:
                with open(os.path.join(tmpdir, 'd.sh'), 'w') as FILE:
                    FILE.write('echo "Hello, World"\n')

                with self.assertRaises(self.failureException):
                    self.shell_test_driver(*sh_tests[1])
                with open(os.path.join(tmpdir, 'd.txt'), 'r') as FILE:
                    self.assertEqual(FILE.read(), "")

                os.environ['PYOMO_TEST_UPDATE_BASELINES'] = '1'

                with self.assertRaises(self.failureException):
                    self.shell_test_driver(*sh_tests[1])
                with open(os.path.join(tmpdir, 'd.txt'), 'r') as FILE:
                    self.assertEqual(FILE.read(), "Hello, World\n")

            finally:
                os.environ.pop('PYOMO_TEST_UPDATE_BASELINES', None)
                if _update_baselines is not None:
                    os.environ['PYOMO_TEST_UPDATE_BASELINES'] = _update_baselines


if __name__ == '__main__':
    unittest.main()

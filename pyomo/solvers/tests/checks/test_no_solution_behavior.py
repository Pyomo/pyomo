#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import types
try:
    import new
    new_available=True
except:
    new_available=False

import pyutilib.th as unittest

from pyomo.solvers.tests.models.base import test_models
from pyomo.solvers.tests.testcases import test_scenarios
from pyomo.common.log import LoggingIntercept

from six import StringIO

# The test directory
thisDir = os.path.dirname(os.path.abspath( __file__ ))

# Cleanup Expected Failure Results Files
_cleanup_expected_failures = True

#
# A function that returns a function that gets
# added to a test class.
#
@unittest.nottest
def create_test_method(model,
                       solver,
                       io,
                       test_case):

    is_expected_failure = test_case.status == 'expected failure'

    #
    # Create a function that executes the test
    #
    def failed_solve_test(self):
        # Create the model test class
        model_class = test_case.model()

        # Create the model instance
        model_class.generate_model(test_case.testcase.import_suffixes)
        model_class.warmstart_model()

        # solve
        load_solutions = True
        symbolic_labels = False
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.solvers'):
            with LoggingIntercept(out, 'pyomo.opt'):
                opt, results = model_class.solve(
                    solver,
                    io,
                    test_case.testcase.io_options,
                    test_case.testcase.options,
                    symbolic_labels,
                    load_solutions)
        model_class.post_solve_test_validation(self, results)
        if len(results.solution) == 0:
            self.assertIn("No solution is available",
                          out.getvalue())
        else:
            # Note ASL solvers might still return a solution
            # file with garbage values in it for a failed solve
            self.assertEqual(len(results.solution), 1)

    # Skip this test if the status is 'skip'
    if test_case.status == 'skip':
        def skipping_test(self):
            return self.skipTest(test_case.msg)
        return skipping_test

    if is_expected_failure:
        @unittest.expectedFailure
        def failing_failed_solve_test(self):
            return failed_solve_test(self)
        # Return a test that is expected to fail
        return failing_failed_solve_test

    # Return a normal test
    return failed_solve_test

cls = None

#
# Create test driver classes for each test model
#
driver = {}
for model in test_models():
    # Get the test case for the model
    case = test_models(model)
    if case().solve_should_fail:
        # Create the test class
        name = "Test_%s" % model
        if new_available:
            cls = new.classobj(name, (unittest.TestCase,), {})
        else:
            cls = types.new_class(name, (unittest.TestCase,))
            cls.__module__ = __name__
        cls = unittest.category(*case.level)(cls)
        driver[model] = cls
        globals()[name] = cls

#
# Iterate through all test scenarios and add test methods
#
for key, value in test_scenarios():
    model, solver, io = key
    if model in driver:
        cls = driver[model]
        # TODO: expand these tests to cover ASL models once
        #       a change in load_solutions behavior is
        #       propagated into that framework.
        if "_kernel" in cls.__name__:
            test_name = "test_"+solver+"_"+io
            test_method = create_test_method(model, solver, io, value)
            if test_method is not None:
                test_method = unittest.category('smoke','nightly',solver)(
                    test_method)
                setattr(cls, test_name, test_method)
                test_method = None

# Reset the cls variable, since it contains a unittest.TestCase subclass.
# This prevents this class from being processed twice!
cls = None

if __name__ == "__main__":
    unittest.main()

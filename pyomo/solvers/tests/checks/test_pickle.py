#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pickle
import os
from os.path import join, dirname, abspath
import warnings
import types
try:
    import new
    new_available=True
except:
    new_available=False

import pyutilib.th as unittest
from pyomo.solvers.tests.models.base import test_models
from pyomo.solvers.tests.testcases import test_scenarios


#
# A function that function that returns a function that gets
# added to a test class.
#
def create_test_method(model, solver, io,
                     test_case,
                     symbolic_labels):

    # Ignore expected failures?
    is_expected_failure = False
    #is_expected_failure = test_case.status == 'expected failure'

    def pickle_test(self):

        # Instantiate the model class
        model_class = test_case.model()

        # Create the model instance
        model_class.generate_model(test_case.testcase.import_suffixes)
        model_class.warmstart_model()

        # try to pickle then unpickle instance
        instance1 = model_class.model.clone()
        instance2 = pickle.loads(pickle.dumps(instance1))
        self.assertNotEqual(id(instance1),id(instance2))

        # try to solve the original instance
        model_class.model = instance1
        load_solutions = True
        opt, results1 = model_class.solve(solver,
                                          io,
                                          test_case.testcase.io_options,
                                          test_case.testcase.options,
                                          symbolic_labels,
                                          load_solutions)

        # try to solve the unpickled instance
        model_class.model = instance2
        opt, results2 = model_class.solve(solver,
                                          io,
                                          test_case.testcase.io_options,
                                          test_case.testcase.options,
                                          symbolic_labels,
                                          load_solutions)

        # try to pickle the instance and results,
        # then unpickle and load results
        inst, res = pickle.loads(pickle.dumps([instance1,results1]))
        #inst.solutions.load(res)

        # try to pickle the instance and results,
        # then unpickle and load results
        inst, res = pickle.loads(pickle.dumps([instance2,results2]))

    # Skip this test if the status is 'skip'
    if test_case.status == 'skip':
        def skipping_test(self):
            return self.skipTest(test_case.msg)
        return skipping_test

    if is_expected_failure:
        @unittest.expectedFailure
        def failing_writer_test(self):
            return pickle_test(self)
        # Return a test that is expected to fail
        return failing_writer_test

    return pickle_test


#
# Create test driver classes for each test model
#
driver = {}
for model in test_models():
    # Get the test case for the model
    case = test_models(model)

    # Create the test class
    name = "Test_%s" % model
    if new_available:
        cls = new.classobj(name, (unittest.TestCase,), {})
    else:
        cls = types.new_class(name, (unittest.TestCase,))
    cls = unittest.category(*case.level)(cls)
    driver[model] = cls
    globals()[name] = cls
#
# Iterate through all test scenarios and add test methods
#
for key, value in test_scenarios(lambda c: c.test_pickling):
    model, solver, io = key
    cls = driver[model]
    # Symbolic labels
    test_name = "test_"+solver+"_"+io +"_symbolic_labels"
    test_method = create_test_method(model, solver, io, value, True)
    if test_method is not None:
        setattr(cls, test_name, test_method)
    # Non-symbolic labels
    test_name = "test_"+solver+"_"+io +"_nonsymbolic_labels"
    test_method = create_test_method(model, solver, io, value, False)
    if test_method is not None:
        setattr(cls, test_name, test_method)

# Reset the cls variable, since it contains a unittest.TestCase subclass.
# This prevents this class from being processed twice!
cls = None


if __name__ == "__main__":
    unittest.main()

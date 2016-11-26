#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pickle
from os.path import dirname, abspath
thisDir = dirname(abspath( __file__ ))

import pyutilib.th as unittest

from pyomo.core import Suffix
from pyomo.opt import ProblemFormat, PersistentSolver
from pyomo.solvers.tests.io import model_types
from pyomo.solvers.tests.io.writer_test_cases import testCases

def CreateTestMethod(test_case,
                     modelClass,
                     test_name,
                     symbolic_labels=False):

    # We do not want to test the plugin case on a model
    # class it is not capable of handling
    if not modelClass().validateCapabilities(test_case):
        return None

    # Skip this test if the solver is not available on the system
    if not test_case.available:
        def skipping_test(self):
            return self.skipTest('Solver unavailable: '
                                 +test_case.name+' ('+test_case.io+')')
        return skipping_test

    def pickle_test(self):

        #self.skipTest("Ignoring pickling tests for now")

        # Instantiate the model class
        model_class = modelClass()

        # Make sure we start from a new solver plugin
        # each time. We don't want them to maintain
        # some state that carries over between tests
        opt, io_options = test_case.initialize()

        try:
            if test_case.io == 'nl':
                self.assertEqual(opt.problem_format(), ProblemFormat.nl)
            elif test_case.io == 'lp':
                self.assertEqual(opt.problem_format(), ProblemFormat.cpxlp)
            elif test_case.io == 'python':
                self.assertEqual(opt.problem_format(), None)

            # check that the solver plugin is at least as capable as the
            # test_case advertises, otherwise the plugin capabilities need
            # to be change or the test case should be removed
            if not all(opt.has_capability(tag)
                       for tag in test_case.capabilities):
                self.fail("Actual plugin capabilities are less than "
                          "that of the of test case for the plugin: "
                          +test_case.name+' ('+test_case.io+')')
        finally:
            opt.deactivate()

        # Create the model instance and send to the solver
        model_class.generateModel()
        model_class.warmstartModel()

        model = model_class.model
        self.assertTrue(model is not None)

        test_suffixes = [] if model_class.disableSuffixTests() else \
                        test_case.import_suffixes

        for suffix in test_suffixes:
            setattr(model,suffix,Suffix(direction=Suffix.IMPORT))

        def _solve(_model):
            _opt, io_options = test_case.initialize()
            try:
                if isinstance(_opt, PersistentSolver):
                    _opt.compile_instance(_model,
                                          symbolic_solver_labels=symbolic_labels)
                if _opt.warm_start_capable():
                    return _opt.solve(_model,
                                      symbolic_solver_labels=symbolic_labels,
                                      warmstart=True,
                                      **io_options)
                else:
                    return _opt.solve(_model,
                                      symbolic_solver_labels=symbolic_labels,
                                      **io_options)
            finally:
                _opt.deactivate()
            del _opt

        results = _solve(model)

        instance1 = model.clone()
        # try to pickle then unpickle instance
        instance2 = pickle.loads(pickle.dumps(instance1))
        self.assertNotEqual(id(instance1),id(instance2))

        # try to solve the original instance
        results1 = _solve(instance1)
        #instance1.solutions.load(results1)

        # try to solve the unpickled instance
        results2 = _solve(instance2)
        #instance2.solutions.load(results2)

        # try to pickle the instance and results,
        # then unpickle and load results
        inst, res = pickle.loads(pickle.dumps([instance1,results1]))
        #inst.solutions.load(res)

        # try to pickle the instance and results,
        # then unpickle and load results
        inst, res = pickle.loads(pickle.dumps([instance2,results2]))
        #inst.solutions.load(res)

    return pickle_test

def addfntests(cls, tests, modelClass, symbolic_labels=False):
    for case in tests:
        test_name = ("test_"+case.name+"_"+case.io
                     +("" if symbolic_labels else "_non")
                     +"_symbolic_labels")
        test_method = CreateTestMethod(case,
                                       modelClass,
                                       test_name,
                                       symbolic_labels=symbolic_labels)
        if test_method is not None:
            setattr(cls, test_name, test_method)

class PickleTests_simple_LP(unittest.TestCase): pass
PickleTests_simple_LP = unittest.category('smoke','nightly','expensive')(PickleTests_simple_LP)
addfntests(PickleTests_simple_LP,testCases, model_types.simple_LP, symbolic_labels=False)
addfntests(PickleTests_simple_LP,testCases, model_types.simple_LP, symbolic_labels=True)

class PickleTests_piecewise_LP(unittest.TestCase): pass
PickleTests_piecewise_LP = unittest.category('smoke','nightly','expensive')(PickleTests_piecewise_LP)
addfntests(PickleTests_piecewise_LP,testCases, model_types.piecewise_LP, symbolic_labels=False)
addfntests(PickleTests_piecewise_LP,testCases, model_types.piecewise_LP, symbolic_labels=True)

class PickleTests_unused_vars_LP(unittest.TestCase): pass
PickleTests_unused_vars_LP = unittest.category('smoke','nightly','expensive')(PickleTests_unused_vars_LP)
addfntests(PickleTests_unused_vars_LP,testCases, model_types.unused_vars_LP, symbolic_labels=False)
addfntests(PickleTests_unused_vars_LP,testCases, model_types.unused_vars_LP, symbolic_labels=True)

class PickleTests_unused_vars_MILP(unittest.TestCase): pass
PickleTests_unused_vars_MILP = unittest.category('smoke','nightly','expensive')(PickleTests_unused_vars_MILP)
addfntests(PickleTests_unused_vars_MILP,testCases, model_types.unused_vars_MILP, symbolic_labels=False)
addfntests(PickleTests_unused_vars_MILP,testCases, model_types.unused_vars_MILP, symbolic_labels=True)

class PickleTests_discrete_var_bounds_MILP(unittest.TestCase): pass
PickleTests_discrete_var_bounds_MILP = unittest.category('smoke','nightly','expensive')(PickleTests_discrete_var_bounds_MILP)
addfntests(PickleTests_discrete_var_bounds_MILP,testCases, model_types.discrete_var_bounds_MILP, symbolic_labels=False)
addfntests(PickleTests_discrete_var_bounds_MILP,testCases, model_types.discrete_var_bounds_MILP, symbolic_labels=True)

class PickleTests_constant_objective_LP1(unittest.TestCase): pass
PickleTests_constant_objective_LP1 = unittest.category('smoke','nightly','expensive')(PickleTests_constant_objective_LP1)
addfntests(PickleTests_constant_objective_LP1,testCases, model_types.constant_objective_LP1, symbolic_labels=False)
addfntests(PickleTests_constant_objective_LP1,testCases, model_types.constant_objective_LP1, symbolic_labels=True)

class PickleTests_constant_objective_LP2(unittest.TestCase): pass
PickleTests_constant_objective_LP2 = unittest.category('smoke','nightly','expensive')(PickleTests_constant_objective_LP2)
addfntests(PickleTests_constant_objective_LP2,testCases, model_types.constant_objective_LP2, symbolic_labels=False)
addfntests(PickleTests_constant_objective_LP2,testCases, model_types.constant_objective_LP2, symbolic_labels=True)

class PickleTests_constant_objective_QP(unittest.TestCase): pass
PickleTests_constant_objective_QP = unittest.category('smoke','nightly','expensive')(PickleTests_constant_objective_QP)
addfntests(PickleTests_constant_objective_QP,testCases, model_types.constant_objective_QP, symbolic_labels=False)
addfntests(PickleTests_constant_objective_QP,testCases, model_types.constant_objective_QP, symbolic_labels=True)

class PickleTests_block_LP(unittest.TestCase): pass
PickleTests_block_LP = unittest.category('smoke','nightly','expensive')(PickleTests_block_LP)
addfntests(PickleTests_block_LP,testCases, model_types.block_LP, symbolic_labels=False)
addfntests(PickleTests_block_LP,testCases, model_types.block_LP, symbolic_labels=True)

class PickleTests_inactive_index_LP(unittest.TestCase): pass
PickleTests_inactive_index_LP = unittest.category('smoke','nightly','expensive')(PickleTests_inactive_index_LP)
addfntests(PickleTests_inactive_index_LP,testCases, model_types.inactive_index_LP, symbolic_labels=False)
addfntests(PickleTests_inactive_index_LP,testCases, model_types.inactive_index_LP, symbolic_labels=True)

class PickleTests_simple_MILP(unittest.TestCase): pass
PickleTests_simple_MILP = unittest.category('nightly','expensive')(PickleTests_simple_MILP)
addfntests(PickleTests_simple_MILP,testCases, model_types.simple_MILP, symbolic_labels=False)
addfntests(PickleTests_simple_MILP,testCases, model_types.simple_MILP, symbolic_labels=True)

class PickleTests_simple_QP(unittest.TestCase): pass
PickleTests_simple_QP = unittest.category('nightly','expensive')(PickleTests_simple_QP)
addfntests(PickleTests_simple_QP,testCases, model_types.simple_QP, symbolic_labels=False)
addfntests(PickleTests_simple_QP,testCases, model_types.simple_QP, symbolic_labels=True)

class PickleTests_simple_MIQP(unittest.TestCase): pass
PickleTests_simple_MIQP = unittest.category('nightly','expensive')(PickleTests_simple_MIQP)
addfntests(PickleTests_simple_MIQP,testCases, model_types.simple_MIQP, symbolic_labels=False)
addfntests(PickleTests_simple_MIQP,testCases, model_types.simple_MIQP, symbolic_labels=True)

class PickleTests_simple_QCP(unittest.TestCase): pass
PickleTests_simple_QCP = unittest.category('nightly','expensive')(PickleTests_simple_QCP)
addfntests(PickleTests_simple_QCP,testCases, model_types.simple_QCP, symbolic_labels=False)
addfntests(PickleTests_simple_QCP,testCases, model_types.simple_QCP, symbolic_labels=True)

class PickleTests_simple_MIQCP(unittest.TestCase): pass
PickleTests_simple_MIQCP = unittest.category('nightly','expensive')(PickleTests_simple_MIQCP)
addfntests(PickleTests_simple_MIQCP,testCases, model_types.simple_MIQCP, symbolic_labels=False)
addfntests(PickleTests_simple_MIQCP,testCases, model_types.simple_MIQCP, symbolic_labels=True)

class PickleTests_simple_SOS1(unittest.TestCase): pass
PickleTests_simple_SOS1 = unittest.category('nightly','expensive')(PickleTests_simple_SOS1)
addfntests(PickleTests_simple_SOS1,testCases, model_types.simple_SOS1, symbolic_labels=False)
addfntests(PickleTests_simple_SOS1,testCases, model_types.simple_SOS1, symbolic_labels=True)

class PickleTests_simple_SOS2(unittest.TestCase): pass
PickleTests_simple_SOS2 = unittest.category('nightly','expensive')(PickleTests_simple_SOS2)
addfntests(PickleTests_simple_SOS2,testCases, model_types.simple_SOS2, symbolic_labels=False)
addfntests(PickleTests_simple_SOS2,testCases, model_types.simple_SOS2, symbolic_labels=True)

class PickleTests_duals_maximize(unittest.TestCase): pass
PickleTests_duals_maximize = unittest.category('nightly','expensive')(PickleTests_duals_maximize)
addfntests(PickleTests_duals_maximize,testCases, model_types.duals_maximize, symbolic_labels=False)
addfntests(PickleTests_duals_maximize,testCases, model_types.duals_maximize, symbolic_labels=True)

class PickleTests_duals_minimize(unittest.TestCase): pass
PickleTests_duals_minimize = unittest.category('nightly','expensive')(PickleTests_duals_minimize)
addfntests(PickleTests_duals_minimize,testCases, model_types.duals_minimize, symbolic_labels=False)
addfntests(PickleTests_duals_minimize,testCases, model_types.duals_minimize, symbolic_labels=True)

if __name__ == "__main__":
    unittest.main()

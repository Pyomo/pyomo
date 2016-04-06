#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
from os.path import join, dirname, abspath
import warnings

from pyomo.core import Suffix
from pyomo.opt import ProblemFormat, PersistentSolver
import pyutilib.th as unittest
from pyomo.solvers.tests.io import model_types
from pyomo.solvers.tests.io.writer_test_cases import testCases

# TODO: Test the output_fixed_variable_bounds option

thisDir = dirname(abspath( __file__ ))

# Cleanup Expected Failure Results Files
_cleanup_expected_failures = True

# For expected failures that appear in all known version
_trunk_version =  (float('inf'), float('inf'), float('inf'), float('inf'))
# These are usually due to a bug in the latest version of the
# thirdparty solver Tests will be expected to fail. If they do not,
# that means the solver has been fixed and that particular case should
# no longer exist in the list of expected failures
ExpectedFailures = []

#
# GLPK
#

ExpectedFailures.append(
	('glpk', 'lp', (4,52,0,0),
	 model_types.discrete_var_bounds_MILP,
	 "Glpk ignores bounds on Binary variables through the "
	 "LP file interface. A ticket has been filed."))
ExpectedFailures.append(
	('glpk', 'mps', _trunk_version,
	 model_types.duals_maximize,
	 "Glpk does not accept the OBJSENSE section of the Free MPS format. "
         "Therefore maximization models are not explicitly handled."))

#
# CBC
#

ExpectedFailures.append(
	('cbc', 'lp', _trunk_version,
	 model_types.duals_maximize,
	 "For a maximization problem where a variable is pushed to its "
	 "lower bound, Cbc reports the reduced cost as a positive number. In "
	 "practice this should be reported as a negative number. A ticket has "
	 "been filed at:\nhttps://projects.coin-or.org/Cbc/ticket/125"))

#
# PICO
#

ExpectedFailures.append(
	('pico', 'lp', _trunk_version,
	 model_types.discrete_var_bounds_MILP,
	 "Pico ignores bounds on Binary variables through the "
	 "LP file interface. A ticket has been filed."))

ExpectedFailures.append(
	('pico', 'nl', _trunk_version,
	 model_types.piecewise_LP,
	 "Pico reports an incorrect dual solution for this "
	 "problem when using the NL file interface."))

ExpectedFailures.append(
	('pico', 'nl', _trunk_version,
	 model_types.duals_maximize,
	 "Pico classifies certain models with equality "
	 "constraints as infeasible when using the NL "
	 "file interface. A ticket has been filed."))

ExpectedFailures.append(
	('pico', 'nl', _trunk_version,
	 model_types.duals_minimize,
	 "Pico classifies certain models with equality "
	 "constraints as infeasible when using the NL "
	 "file interface. A ticket has been filed."))

ExpectedFailures.append(
	('pico', 'nl', _trunk_version,
	 model_types.inactive_index_LP,
	 "Pico reports the wrong objective function value."))

ExpectedFailures.append(
	('pico', 'nl', _trunk_version,
	 model_types.simple_LP,
	 "Pico just gets the wrong answer."))

ExpectedFailures.append(
	('pico', 'nl', _trunk_version,
	 model_types.compiled_LP,
	 "Pico just gets the wrong answer."))

ExpectedFailures.append(
	('pico', 'nl', _trunk_version,
	 model_types.trivial_constraints_LP,
	 "Pico just gets the wrong answer."))

#
# SCIP
#

ExpectedFailures.append(
	('scip', 'nl', (3, 1, 0, 9),
	 model_types.simple_SOS2,
	 "SCIP (scipampl) does not recognize sos2 constraints "
	 "inside NL files. A ticket has been filed."))

ExpectedFailures.append(
	('scip', 'nl', (3, 1, 0, 9),
	 model_types.simple_SOS1,
	 "SCIP (scipampl) does not recognize sos2 constraints "
	 "inside NL files. A ticket has been filed."))

#
# CPLEX
#

ExpectedFailures.append(
	('cplex', 'lp', _trunk_version,
	 model_types.simple_QCP,
	 "Cplex does not report duals of quadratic constraints."))

ExpectedFailures.append(
	('cplex', 'mps', _trunk_version,
	 model_types.simple_QCP,
	 "Cplex does not report duals of quadratic constraints."))

ExpectedFailures.append(
	('cplex', 'python', _trunk_version,
	 model_types.simple_QCP,
	 "Cplex does not report duals of quadratic constraints."))

ExpectedFailures.append(
	('_cplex_persistent', 'python', _trunk_version,
	 model_types.simple_QCP,
	 "Cplex does not report duals of quadratic constraints."))

ExpectedFailures.append(
	('cplex', 'nl', (12,5,9,9),
	 model_types.simple_QCP,
	 "Cplex does not report duals of quadratic constraints."))

#
# BARON
#

"""
ExpectedFailures.append(
    ('baron', 'bar', _trunk_version,
     model_types.duals_minimize,
     "The model is too large for a Baron trial license."))

ExpectedFailures.append(
    ('baron', 'bar', _trunk_version,
     model_types.duals_maximize,
     "The model is too large for a Baron trial license."))
"""

ExpectedFailures.append(
    ('baron', 'bar', (15,0,0,0),
     model_types.piecewise_LP,
     "Baron will not return dual solution when a solution is "
     "found during preprocessing."))

ExpectedFailures.append(
    ('baron', 'bar', (15,2,0,0),
     model_types.simple_QP,
     "Baron will not return dual solution when a solution is "
     "found during preprocessing."))

ExpectedFailures.append(
    ('baron', 'bar', _trunk_version,
     model_types.simple_QCP,
     "Baron will not return dual solution when a solution is "
     "found during preprocessing."))

def check_expected_failures(test_case, model_class):

    # If this situation is an expected failure then return the message why
    for case in ExpectedFailures:
        if (case[0] == test_case.name) and \
           (case[1] == test_case.io) and \
           (test_case.solver.version() is not None) and \
           (test_case.solver.version() <= case[2]) and \
           (case[3] is model_class) :
            return True, case[4]
    return False, ""

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

    def writer_test(self):

        # Instantiate the model class
        model_class = modelClass()
        save_filename = join(thisDir,
                             (model_class.descrStr()+"."
                              +test_name+".results"))

        # cleanup possibly existing old test files
        try:
            os.remove(save_filename)
        except OSError:
            pass

        # Make sure we start from a new solver plugin
        # each time. We don't want them to maintain
        # some state that carries over between tests
        opt, io_options = test_case.initialize()

        if test_case.io == 'nl':
            self.assertEqual(opt.problem_format(), ProblemFormat.nl)
        elif test_case.io == 'lp':
            self.assertEqual(opt.problem_format(), ProblemFormat.cpxlp)
        elif test_case.io == 'mps':
            self.assertEqual(opt.problem_format(), ProblemFormat.mps)
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

        # Create the model instance and send to the solver
        model_class.generateModel()
        model_class.warmstartModel()

        model = model_class.model
        self.assertTrue(model is not None)

        test_suffixes = [] if model_class.disableSuffixTests() else \
                        test_case.import_suffixes

        for suffix in test_suffixes:
            setattr(model,suffix,Suffix(direction=Suffix.IMPORT))

        if isinstance(opt, PersistentSolver):
            opt.compile_instance(model,
                                 symbolic_solver_labels=symbolic_labels)

        # solve
        if opt.warm_start_capable():
            results = opt.solve(
                model,
                symbolic_solver_labels=symbolic_labels,
                warmstart=True,
                load_solutions=False,
                **io_options)
        else:
            results = opt.solve(
                model,
                symbolic_solver_labels=symbolic_labels,
                load_solutions=False,
                **io_options)

        model_class.postSolveTestValidation(self, results)

        model.solutions.load_from(results,
				  default_variable_value=opt.default_variable_value())

        model_class.saveCurrentSolution(save_filename,
                                        suffixes=test_suffixes)

        # There are certain cases where the latest solver version has
        # a bug so this should not cause a pyomo test to fail
        is_expected_failure, failure_msg = \
            check_expected_failures(test_case, modelClass)

        # validate the solution returned by the solver
        rc = model_class.validateCurrentSolution(suffixes=test_suffixes)

        if is_expected_failure:
            if rc[0] is True:
                warnings.warn("\nTest model '%s' was marked as an expected "
                              "failure but no failure occured. The "
                              "reason given for the expected failure "
                              "is:\n\n****\n%s\n****\n\n"
                              "Please remove this case as an expected "
                              "failure if the above issue has been "
                              "corrected in the latest version of the "
                              "solver." % (model_class.descrStr(), failure_msg))
            if _cleanup_expected_failures:
                os.remove(save_filename)

        if not rc[0]:
            try:
                model.solutions.store_to(results)
            except ValueError:
                pass
            self.fail("Solution mismatch for plugin "+test_case.name
                      +' '+str(opt.version())+', '+test_case.io+
                      " interface and problem type "
                      +model_class.descrStr()+"\n"+rc[1]+"\n"
                      +(str(results.Solution(0)) if len(results.solution) else "No Solution"))

        # cleanup if the test passed
        try:
            os.remove(save_filename)
        except OSError:
            pass

    @unittest.expectedFailure
    def failing_writer_test(self):
        return writer_test(self)

    is_expected_failure, failure_msg = \
        check_expected_failures(test_case, modelClass)

    if is_expected_failure is True:
        return failing_writer_test
    else:
        return writer_test

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

class WriterTests_simple_LP(unittest.TestCase): pass
WriterTests_simple_LP = unittest.category('smoke','nightly','expensive')(WriterTests_simple_LP)
addfntests(WriterTests_simple_LP,testCases, model_types.simple_LP, symbolic_labels=False)
addfntests(WriterTests_simple_LP,testCases, model_types.simple_LP, symbolic_labels=True)

class WriterTests_compiled_LP(unittest.TestCase): pass
WriterTests_compiled_LP = unittest.category('smoke','nightly','expensive')(WriterTests_compiled_LP)
addfntests(WriterTests_compiled_LP,testCases, model_types.compiled_LP, symbolic_labels=False)
addfntests(WriterTests_compiled_LP,testCases, model_types.compiled_LP, symbolic_labels=True)

class WriterTests_trivial_constraints_LP(unittest.TestCase): pass
WriterTests_trivial_constraints_LP = unittest.category('smoke','nightly','expensive')(WriterTests_trivial_constraints_LP)
addfntests(WriterTests_trivial_constraints_LP,testCases, model_types.trivial_constraints_LP, symbolic_labels=False)
addfntests(WriterTests_trivial_constraints_LP,testCases, model_types.trivial_constraints_LP, symbolic_labels=True)

class WriterTests_piecewise_LP(unittest.TestCase): pass
WriterTests_piecewise_LP = unittest.category('smoke','nightly','expensive')(WriterTests_piecewise_LP)
addfntests(WriterTests_piecewise_LP,testCases, model_types.piecewise_LP, symbolic_labels=False)
addfntests(WriterTests_piecewise_LP,testCases, model_types.piecewise_LP, symbolic_labels=True)

class WriterTests_piecewise_LP_nosuffixes(unittest.TestCase): pass
WriterTests_piecewise_LP_nosuffixes = unittest.category('smoke','nightly','expensive')(WriterTests_piecewise_LP_nosuffixes)
addfntests(WriterTests_piecewise_LP_nosuffixes,testCases, model_types.piecewise_LP_nosuffixes, symbolic_labels=False)
addfntests(WriterTests_piecewise_LP_nosuffixes,testCases, model_types.piecewise_LP_nosuffixes, symbolic_labels=True)

class WriterTests_unused_vars_LP(unittest.TestCase): pass
WriterTests_unused_vars_LP = unittest.category('smoke','nightly','expensive')(WriterTests_unused_vars_LP)
addfntests(WriterTests_unused_vars_LP,testCases, model_types.unused_vars_LP, symbolic_labels=False)
addfntests(WriterTests_unused_vars_LP,testCases, model_types.unused_vars_LP, symbolic_labels=True)

class WriterTests_unused_vars_MILP(unittest.TestCase): pass
WriterTests_unused_vars_MILP = unittest.category('smoke','nightly','expensive')(WriterTests_unused_vars_MILP)
addfntests(WriterTests_unused_vars_MILP,testCases, model_types.unused_vars_MILP, symbolic_labels=False)
addfntests(WriterTests_unused_vars_MILP,testCases, model_types.unused_vars_MILP, symbolic_labels=True)

class WriterTests_discrete_var_bounds_MILP(unittest.TestCase): pass
WriterTests_discrete_var_bounds_MILP = unittest.category('smoke','nightly','expensive')(WriterTests_discrete_var_bounds_MILP)
addfntests(WriterTests_discrete_var_bounds_MILP,testCases, model_types.discrete_var_bounds_MILP, symbolic_labels=False)
addfntests(WriterTests_discrete_var_bounds_MILP,testCases, model_types.discrete_var_bounds_MILP, symbolic_labels=True)

class WriterTests_constant_objective_LP1(unittest.TestCase): pass
WriterTests_constant_objective_LP1 = unittest.category('smoke','nightly','expensive')(WriterTests_constant_objective_LP1)
addfntests(WriterTests_constant_objective_LP1,testCases, model_types.constant_objective_LP1, symbolic_labels=False)
addfntests(WriterTests_constant_objective_LP1,testCases, model_types.constant_objective_LP1, symbolic_labels=True)

class WriterTests_constant_objective_LP2(unittest.TestCase): pass
WriterTests_constant_objective_LP2 = unittest.category('smoke','nightly','expensive')(WriterTests_constant_objective_LP2)
addfntests(WriterTests_constant_objective_LP2,testCases, model_types.constant_objective_LP2, symbolic_labels=False)
addfntests(WriterTests_constant_objective_LP2,testCases, model_types.constant_objective_LP2, symbolic_labels=True)

class WriterTests_constant_objective_QP(unittest.TestCase): pass
WriterTests_constant_objective_QP = unittest.category('smoke','nightly','expensive')(WriterTests_constant_objective_QP)
addfntests(WriterTests_constant_objective_QP,testCases, model_types.constant_objective_QP, symbolic_labels=False)
addfntests(WriterTests_constant_objective_QP,testCases, model_types.constant_objective_QP, symbolic_labels=True)

class WriterTests_block_LP(unittest.TestCase): pass
WriterTests_block_LP = unittest.category('smoke','nightly','expensive')(WriterTests_block_LP)
addfntests(WriterTests_block_LP,testCases, model_types.block_LP, symbolic_labels=False)
addfntests(WriterTests_block_LP,testCases, model_types.block_LP, symbolic_labels=True)

class WriterTests_inactive_index_LP(unittest.TestCase): pass
WriterTests_inactive_index_LP = unittest.category('smoke','nightly','expensive')(WriterTests_inactive_index_LP)
addfntests(WriterTests_inactive_index_LP,testCases, model_types.inactive_index_LP, symbolic_labels=False)
addfntests(WriterTests_inactive_index_LP,testCases, model_types.inactive_index_LP, symbolic_labels=True)

class WriterTests_simple_MILP(unittest.TestCase): pass
WriterTests_simple_MILP = unittest.category('nightly','expensive')(WriterTests_simple_MILP)
addfntests(WriterTests_simple_MILP,testCases, model_types.simple_MILP, symbolic_labels=False)
addfntests(WriterTests_simple_MILP,testCases, model_types.simple_MILP, symbolic_labels=True)

class WriterTests_simple_QP(unittest.TestCase): pass
WriterTests_simple_QP = unittest.category('nightly','expensive')(WriterTests_simple_QP)
addfntests(WriterTests_simple_QP,testCases, model_types.simple_QP, symbolic_labels=False)
addfntests(WriterTests_simple_QP,testCases, model_types.simple_QP, symbolic_labels=True)

class WriterTests_simple_QP_nosuffixes(unittest.TestCase): pass
WriterTests_simple_QP_nosuffixes = unittest.category('nightly','expensive')(WriterTests_simple_QP_nosuffixes)
addfntests(WriterTests_simple_QP_nosuffixes,testCases, model_types.simple_QP_nosuffixes, symbolic_labels=False)
addfntests(WriterTests_simple_QP_nosuffixes,testCases, model_types.simple_QP_nosuffixes, symbolic_labels=True)

class WriterTests_simple_MIQP(unittest.TestCase): pass
WriterTests_simple_MIQP = unittest.category('nightly','expensive')(WriterTests_simple_MIQP)
addfntests(WriterTests_simple_MIQP,testCases, model_types.simple_MIQP, symbolic_labels=False)
addfntests(WriterTests_simple_MIQP,testCases, model_types.simple_MIQP, symbolic_labels=True)

class WriterTests_simple_QCP(unittest.TestCase): pass
WriterTests_simple_QCP = unittest.category('nightly','expensive')(WriterTests_simple_QCP)
addfntests(WriterTests_simple_QCP,testCases, model_types.simple_QCP, symbolic_labels=False)
addfntests(WriterTests_simple_QCP,testCases, model_types.simple_QCP, symbolic_labels=True)

class WriterTests_simple_QCP_nosuffixes(unittest.TestCase): pass
WriterTests_simple_QCP_nosuffixes = unittest.category('nightly','expensive')(WriterTests_simple_QCP_nosuffixes)
addfntests(WriterTests_simple_QCP_nosuffixes,testCases, model_types.simple_QCP_nosuffixes, symbolic_labels=False)
addfntests(WriterTests_simple_QCP_nosuffixes,testCases, model_types.simple_QCP_nosuffixes, symbolic_labels=True)


class WriterTests_simple_MIQCP(unittest.TestCase): pass
WriterTests_simple_MIQCP = unittest.category('nightly','expensive')(WriterTests_simple_MIQCP)
addfntests(WriterTests_simple_MIQCP,testCases, model_types.simple_MIQCP, symbolic_labels=False)
addfntests(WriterTests_simple_MIQCP,testCases, model_types.simple_MIQCP, symbolic_labels=True)

class WriterTests_simple_SOS1(unittest.TestCase): pass
WriterTests_simple_SOS1 = unittest.category('nightly','expensive')(WriterTests_simple_SOS1)
addfntests(WriterTests_simple_SOS1,testCases, model_types.simple_SOS1, symbolic_labels=False)
addfntests(WriterTests_simple_SOS1,testCases, model_types.simple_SOS1, symbolic_labels=True)

class WriterTests_simple_SOS2(unittest.TestCase): pass
WriterTests_simple_SOS2 = unittest.category('nightly','expensive')(WriterTests_simple_SOS2)
addfntests(WriterTests_simple_SOS2,testCases, model_types.simple_SOS2, symbolic_labels=False)
addfntests(WriterTests_simple_SOS2,testCases, model_types.simple_SOS2, symbolic_labels=True)

class WriterTests_duals_maximize(unittest.TestCase): pass
WriterTests_duals_maximize = unittest.category('nightly','expensive')(WriterTests_duals_maximize)
addfntests(WriterTests_duals_maximize,testCases, model_types.duals_maximize, symbolic_labels=False)
addfntests(WriterTests_duals_maximize,testCases, model_types.duals_maximize, symbolic_labels=True)

class WriterTests_duals_minimize(unittest.TestCase): pass
WriterTests_duals_minimize = unittest.category('nightly','expensive')(WriterTests_duals_minimize)
addfntests(WriterTests_duals_minimize,testCases, model_types.duals_minimize, symbolic_labels=False)
addfntests(WriterTests_duals_minimize,testCases, model_types.duals_minimize, symbolic_labels=True)

# A "solver should fail" test, which I am archiving for now
"""
            # If the solver is not capable of handling this
            # model class then we better get a failure here
            try:
                model.load(opt.solve(model))
                model_class.saveCurrentSolution(save_filename,
                                        suffixes=test_case.import_suffixes)
            except:
                pass
            else:
                # Okay so we may get to this point if we are using a
                # plugin like ASL which must advertise having all capabilities
                # since it supports many solvers. And its possible that
                # sending something like a discrete model to ipopt can slip
                # through the cracks without error or warning. Hopefully the test
                # case was set up so that the solution check will turn up bad.
                if model_class.validateCurrentSolution() is True:
                    warnings.warn("Plugin "+test_case.name+' ('+test_case.io+") is not capable of handling model class "+test_model_name+" "\
                                  "but no exception was thrown and solution matched baseline.")
"""

if __name__ == "__main__":
    unittest.main()

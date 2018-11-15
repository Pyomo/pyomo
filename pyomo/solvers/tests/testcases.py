#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import logging

import pyutilib.th as unittest
from pyutilib.misc import Options

from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import test_models
from pyomo.solvers.tests.solvers import test_solver_cases
import pyomo.kernel
from pyomo.core.kernel.block import IBlock

# For expected failures that appear in all known version
_trunk_version =  (float('inf'), float('inf'), float('inf'), float('inf'))

# These are usually due to a bug in the latest version of the
# thirdparty solver Tests will be expected to fail. If they do not,
# that means the solver has been fixed and that particular case should
# no longer exist in the list of expected failures
ExpectedFailures = {}

#
# CPLEX
#

ExpectedFailures['cplex', 'lp', 'QCP_simple'] = \
    (lambda v: v <= _trunk_version,
    "Cplex does not report duals of quadratic constraints.")

ExpectedFailures['cplex', 'mps', 'QCP_simple'] =\
    (lambda v: v <= _trunk_version,
    "Cplex does not report duals of quadratic constraints.")

ExpectedFailures['cplex', 'python', 'QCP_simple'] =\
    (lambda v: v <= _trunk_version,
    "Cplex does not report duals of quadratic constraints.")

ExpectedFailures['cplex_persistent', 'python', 'QCP_simple'] =\
    (lambda v: v <= _trunk_version,
    "Cplex does not report duals of quadratic constraints.")

ExpectedFailures['cplex', 'nl', 'QCP_simple'] = \
    (lambda v: v <= (12,5,9,9),
    "Cplex does not report duals of quadratic constraints.")

#
# GUROBI
#
# NO EXPECTED FAILURES
#

#
# GLPK
#

ExpectedFailures['glpk', 'lp', 'MILP_discrete_var_bounds'] = \
    (lambda v: v <= (4,52,0,0),
    "Glpk ignores bounds on Binary variables through the "
    "LP file interface. A ticket has been filed.")

ExpectedFailures['glpk', 'mps', 'LP_duals_maximize'] = \
    (lambda v: v <= _trunk_version,
    "Glpk does not accept the OBJSENSE section of the Free MPS format. "
    "Therefore maximization models are not explicitly handled.")

#
# CBC
#

ExpectedFailures['cbc', 'lp', 'LP_duals_maximize'] = \
    (lambda v: v <= _trunk_version,
    "For a maximization problem where a variable is pushed to its "
    "lower bound, Cbc reports the reduced cost as a positive number. In "
    "practice this should be reported as a negative number. A ticket has "
    "been filed at:\nhttps://projects.coin-or.org/Cbc/ticket/125")

ExpectedFailures['cbc', 'nl', 'MILP_unbounded'] = \
    (lambda v: v <= _trunk_version,
     "Cbc fails to report a MILP model as unbounded when it"
     "is defined as an NL file.")

#
# PICO
#

ExpectedFailures['pico', 'lp', 'MILP_discrete_var_bounds'] = \
    (lambda v: v <= _trunk_version,
    "Pico ignores bounds on Binary variables through the "
    "LP file interface. A ticket has been filed.")

ExpectedFailures['pico', 'nl', 'LP_piecewise'] = \
    (lambda v: v <= _trunk_version,
    "Pico reports an incorrect dual solution for this "
    "problem when using the NL file interface.")

ExpectedFailures['pico', 'nl', 'LP_duals_maximize'] = \
    (lambda v: v <= _trunk_version,
    "Pico classifies certain models with equality "
    "constraints as infeasible when using the NL "
    "file interface. A ticket has been filed.")

ExpectedFailures['pico', 'nl', 'LP_duals_minimize'] = \
    (lambda v: v <= _trunk_version,
    "Pico classifies certain models with equality "
    "constraints as infeasible when using the NL "
    "file interface. A ticket has been filed.")

ExpectedFailures['pico', 'nl', 'LP_inactive_index'] = \
    (lambda v: v <= _trunk_version,
    "Pico reports the wrong objective function value.")

ExpectedFailures['pico', 'nl', 'LP_simple'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

ExpectedFailures['pico', 'nl', 'LP_compiled'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

ExpectedFailures['pico', 'nl', 'LP_trivial_constraints'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

ExpectedFailures['pico', 'nl', 'MILP_unbounded'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

ExpectedFailures['pico', 'lp', 'MILP_unbounded'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

ExpectedFailures['pico', 'nl', 'LP_unbounded'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

ExpectedFailures['pico', 'lp', 'LP_unbounded'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")


ExpectedFailures['pico', 'lp', 'MILP_infeasible1'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

ExpectedFailures['pico', 'nl', 'MILP_infeasible1'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

ExpectedFailures['pico', 'nl', 'LP_infeasible1'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

ExpectedFailures['pico', 'nl', 'LP_infeasible2'] = \
    (lambda v: v <= _trunk_version,
    "Pico just gets the wrong answer.")

#
# XPRESS
#
# NO EXPECTED FAILURES
#

#
# IPOPT
#

ExpectedFailures['ipopt', 'nl', 'LP_duals_maximize'] = \
    (lambda v: v == (3,10,3,0),
    "Ipopt returns duals with a different sign convention. "
    "Fixed in Ipopt 3.10.4")

ExpectedFailures['ipopt', 'nl', 'QCP_simple'] = \
    (lambda v: v == (3,10,3,0),
    "Ipopt returns duals with a different sign convention. "
    "Fixed in Ipopt 3.10.4")

ExpectedFailures['ipopt', 'nl', 'LP_block'] = \
    (lambda v: v <= (3,10,2,0),
    "Ipopt returns duals with a different sign convention. "
    "Fixed in Ipopt 3.10.3")

ExpectedFailures['ipopt', 'nl', 'LP_duals_minimize'] = \
    (lambda v: v <= (3,10,2,0),
    "Ipopt returns duals with a different sign convention. "
    "Fixed in Ipopt 3.10.3")

ExpectedFailures['ipopt', 'nl', 'LP_inactive_index'] = \
    (lambda v: v <= (3,10,2,0),
    "Ipopt returns duals with a different sign convention. "
    "Fixed in Ipopt 3.10.3")

ExpectedFailures['ipopt', 'nl', 'LP_piecewise'] = \
    (lambda v: v <= (3,10,2,0),
    "Ipopt returns duals with a different sign convention. "
    "Fixed in Ipopt 3.10.3")

ExpectedFailures['ipopt', 'nl', 'LP_simple'] = \
    (lambda v: v <= (3,10,2,0),
    "Ipopt returns duals with a different sign convention. "
    "Fixed in Ipopt 3.10.3")

ExpectedFailures['ipopt', 'nl', 'QP_simple'] = \
    (lambda v: v <= (3,10,2,0),
    "Ipopt returns duals with a different sign convention. "
    "Fixed in Ipopt 3.10.3")

ExpectedFailures['ipopt', 'nl', 'LP_trivial_constraints'] = \
    (lambda v: v <= (3,10,2,0),
    "Ipopt returns duals with a different sign convention. "
    "Fixed in Ipopt 3.10.3")

#
# SCIP
#

ExpectedFailures['scip', 'nl', 'SOS2_simple'] = \
    (lambda v: v <= (3, 1, 0, 9),
    "SCIP (scipampl) does not recognize sos2 constraints "
    "inside NL files. A ticket has been filed.")

ExpectedFailures['scip', 'nl', 'SOS1_simple'] = \
    (lambda v: v <= (3, 1, 0, 9),
    "SCIP (scipampl) does not recognize sos1 constraints "
    "inside NL files. A ticket has been filed.")

#
# BARON
#

ExpectedFailures['baron', 'bar', 'LP_piecewise'] = \
    (lambda v: v <= (15,0,0,0),
    "Baron will not return dual solution when a solution is "
    "found during preprocessing.")

ExpectedFailures['baron', 'bar', 'QP_simple'] = \
    (lambda v: v <= (15,2,0,0),
    "Baron will not return dual solution when a solution is "
    "found during preprocessing.")

ExpectedFailures['baron', 'bar', 'QCP_simple'] = \
    (lambda v: v <= _trunk_version,
    "Baron will not return dual solution when a solution is "
    "found during preprocessing.")

ExpectedFailures['baron', 'bar', 'MILP_unbounded'] = \
    (lambda v: v < (17,4,1,0),
     "Baron fails to report a MILP model as unbounded")

#
# KNITROAMPL
#
# NO EXPECTED FAILURES
#


@unittest.nottest
def test_scenarios(arg=None):
    """
    Generate scenarios
    """
    for model in sorted(test_models()):
        _model = test_models(model)
        if not arg is None and not arg(_model):
            continue
        for solver, io in sorted(test_solver_cases()):
            _solver_case = test_solver_cases(solver, io)

            # Skip this test case if the solver doesn't support the
            # capabilities required by the model
            if not _model.capabilities.issubset( _solver_case.capabilities ):
                continue

            # Set status values for expected failures
            status='ok'
            msg=""
            if not _solver_case.available:
                status='skip'
                msg="Skipping test because solver %s (%s) is unavailable" % (solver,io)
            if (solver,io,_model.description) in ExpectedFailures:
                case = ExpectedFailures[solver,io,_model.description]
                if _solver_case.version is not None and\
                   case[0](_solver_case.version):
                    status='expected failure'
                    msg=case[1]

            # Return scenario dimensions and scenario information
            yield (model, solver, io), Options(status=status, msg=msg, model=_model, solver=None, testcase=_solver_case)


@unittest.nottest
def run_test_scenarios(options):
    logging.disable(logging.WARNING)

    solvers = set(options.solver)
    stat = {}

    for key, test_case in test_scenarios():
        model, solver, io = key
        if len(solvers) > 0 and not solver in solvers:
            continue
        if test_case.status == 'skip':
            continue

        # Create the model test class
        model_class = test_case.model()
        # Create the model instance
        model_class.generate_model()
        model_class.warmstart_model()
        # Solve
        symbolic_labels = False
        load_solutions = False
        opt, results = model_class.solve(
            solver,
            io,
            test_case.testcase.io_options,
            {},
            symbolic_labels,
            load_solutions)

        termination_condition = results['Solver'][0]['termination condition']
        # Validate solution status
        try:
            model_class.post_solve_test_validation(None, results)
        except:
            if test_case.status == 'expected failure':
                stat[key] = (True, "Expected failure")
            else:
                stat[key] = (False, "Unexpected termination condition: %s" % str(termination_condition))
            continue
        if termination_condition == TerminationCondition.unbounded or \
           termination_condition == TerminationCondition.infeasible:
            # Unbounded or Infeasible
            stat[key] = (True, "")
        else:
            # Validate the solution returned by the solver
            if isinstance(model_class.model, IBlock):
                model_class.model.load_solution(results.solution)
            else:
                model_class.model.solutions.load_from(
                    results,
                    default_variable_value=opt.default_variable_value())
            rc = model_class.validate_current_solution(suffixes=model_class.test_suffixes)

            if test_case.status == 'expected failure':
                if rc[0] is True:
                    stat[key] = (False, "Unexpected success")
                else:
                    stat[key] = (True, "Expected failure")
            else:
                if rc[0] is True:
                    stat[key] = (True, "")
                else:
                    stat[key] = (False, "Unexpected failure")

    if options.verbose:
        print("---------------")
        print(" Test Failures")
        print("---------------")
    nfail = 0
    #
    # Summarize the runtime statistics, by solver
    #
    summary = {}
    total = Options(NumEPass=0, NumEFail=0, NumUPass=0, NumUFail=0)
    for key in stat:
        model, solver, io = key
        if not solver in summary:
            summary[solver] = Options(NumEPass=0, NumEFail=0, NumUPass=0, NumUFail=0)
        _pass, _str = stat[key]
        if _pass:
            if _str == "Expected failure":
                summary[solver].NumEFail += 1
            else:
                summary[solver].NumEPass += 1
        else:
            nfail += 1
            if _str == "Unexpected failure":
                summary[solver].NumUFail += 1
                if options.verbose:
                    print("- Unexpected Test Failure: "+", ".join((model, solver, io)))
            else:
                summary[solver].NumUPass += 1
                if options.verbose:
                    print("- Unexpected Test Success: "+", ".join((model, solver, io)))
    if options.verbose:
        if nfail == 0:
            print("- NONE")
        print("")

    stream = sys.stdout
    maxSolverNameLen = max([max(len(name) for name in summary), len("Solver")])
    fmtStr = "{{0:<{0}}}| {{1:>8}} | {{2:>8}} | {{3:>10}} | {{4:>10}} | {{5:>13}}\n".format(maxSolverNameLen + 2)
    #
    stream.write("\n")
    stream.write("Solver Test Summary\n")
    stream.write("=" * (maxSolverNameLen + 66) + "\n")
    stream.write(fmtStr.format("Solver", "# Pass", "# Fail", "# OK Fail", "# Bad Pass", "% OK"))
    stream.write("=" * (maxSolverNameLen + 66) + "\n")
    #
    for _solver in sorted(summary):
        ans = summary[_solver]
        total.NumEPass += ans.NumEPass
        total.NumEFail += ans.NumEFail
        total.NumUPass += ans.NumUPass
        total.NumUFail += ans.NumUFail
        stream.write(fmtStr.format(_solver, str(ans.NumEPass), str(ans.NumUFail), str(ans.NumEFail), str(ans.NumUPass), str(int(100.0*(ans.NumEPass+ans.NumEFail)/(ans.NumEPass+ans.NumEFail+ans.NumUFail+ans.NumUPass)))))
    #
    stream.write("=" * (maxSolverNameLen + 66) + "\n")
    stream.write(fmtStr.format("TOTALS", str(total.NumEPass), str(total.NumUFail), str(total.NumEFail), str(total.NumUPass), str(int(100.0*(total.NumEPass+total.NumEFail)/(total.NumEPass+total.NumEFail+total.NumUFail+total.NumUPass)))))
    stream.write("=" * (maxSolverNameLen + 66) + "\n")

    logging.disable(logging.NOTSET)


if __name__ == "__main__":
    from pyomo.solvers.tests.models.base import test_models

    print("")
    print("Testing model generation")
    print("-"*30)
    for key in sorted(test_models()):
        print(key)
        obj = test_models(key)()
        obj.generate_model()
        obj.warmstart_model()

    print("")
    print("Testing scenario generation")
    print("-"*30)
    for key, value in test_scenarios():
        print(", ".join(key))
        print("   %s: %s" % (value.status, value.msg))


#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyutilib.misc import Options
from pyomo.solvers.tests.models.base import test_models
from pyomo.solvers.tests.solvers import test_solvers, test_solver_cases


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

ExpectedFailures['cplex', '_cplex_persistent', 'QCP_simple'] =\
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

#
# KNITROAMPL
#
# NO EXPECTED FAILURES
#


def test_scenarios(arg=None):
    """
    Generate scenarios
    """
    for model in sorted(test_models()):
        _model = test_models(model)
        if not arg is None and not arg(_model):
            continue
        for solver, io in sorted(test_solver_cases()):
            #_solver      = test_solvers(solver)
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
            if (solver, io, model) in ExpectedFailures:
                case = ExpectedFailures[solver, io, model]
                if _solver_case.version is not None and\
                   case[0](_solver_case.version):
                    status='expected failure'
                    msg=case[1]

            # Return scenario dimensions and scenario information
            yield (model, solver, io), Options(status=status, msg=msg, model=_model, solver=None, testcase=_solver_case)



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


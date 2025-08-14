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

import sys
import logging

from pyomo.common.collections import Bunch
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import all_models
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases
from pyomo.core.kernel.block import IBlock

# For expected failures that appear in all known version
_trunk_version = (float('inf'), float('inf'), float('inf'), float('inf'))

# These are usually due to a bug in the latest version of the
# thirdparty solver. Tests will be expected to fail. If they do not,
# that means the solver has been fixed and that particular case should
# no longer exist in the list of expected failures
ExpectedFailures = {}

# These are usually due to a bug in the latest version of the thirdparty
# solver. The solver is expected to run successfully, but will not
# return suffix information. If they return suffix information, that
# means the solver has been fixed and that particular case should no
# longer exist in the list of expected failures.  This dict has (solver,
# io, test) tuples as keys and values that are either a dict mapping
# suffix to "(bool(enforce), set(object_names))" or a list of suffix
# names (in which case enforcing is set to True and the set is empty,
# indicating ALL objects).  If enforcing is True the test will fail if
# the missing suffix was found.  Set enforcing to false for tests where
# the solver is inconsistent in returning duals.
MissingSuffixFailures = {}

# These are tests that must be skipped for certain solvers / versions
# because attempting the solve will break the test suite (usually due to
# infinite loops / timeouts)
SkipTests = {}

#
# MOSEK
#

for _io in ('python', 'persistent'):
    for _test in ('QCP_simple', 'QCP_simple_nosuffixes', 'MIQCP_simple'):
        ExpectedFailures['mosek', _io, _test] = (
            lambda v: True,
            "Mosek does not handle nonconvex quadratic constraints",
        )

    for _test in ('MIQP_simple',):
        SkipTests['mosek', _io, _test] = (
            lambda v: v[0] == 10 and v < (10, 0, 30),
            "Mosek 10 fails on assertion warmstarting MIQP models; see #2613",
        )

#
# CPLEX
#

MissingSuffixFailures['cplex', 'lp', 'QCP_simple'] = (
    lambda v: v <= _trunk_version,
    {'dual': (True, {'qc0', 'qc1'})},
    "Cplex does not report duals of quadratic constraints.",
)

MissingSuffixFailures['cplex', 'mps', 'QCP_simple'] = (
    lambda v: v <= _trunk_version,
    {'dual': (True, {'qc0', 'qc1'})},
    "Cplex does not report duals of quadratic constraints.",
)

MissingSuffixFailures['cplex', 'python', 'QCP_simple'] = (
    lambda v: v <= _trunk_version,
    {'dual': (True, {'qc0', 'qc1'})},
    "Cplex does not report duals of quadratic constraints.",
)

MissingSuffixFailures['cplex_persistent', 'python', 'QCP_simple'] = (
    lambda v: v <= _trunk_version,
    {'dual': (True, {'qc0', 'qc1'})},
    "Cplex does not report duals of quadratic constraints.",
)

MissingSuffixFailures['cplex', 'nl', 'QCP_simple'] = (
    lambda v: v < (12, 6, 0, 0),
    {'dual': (True, {'qc0', 'qc1'})},
    "Cplex does not report duals of quadratic constraints.",
)

SkipTests['cplex', 'nl', 'QCP_simple'] = (
    lambda v: v == (12, 6, 0, 0),
    "Cplex 12.6.0.0 produces inconsistent dual values based on "
    "NL variable ordering (which changes between the NLv1 and NLv2 writers",
)

#
# GUROBI
#
# NO EXPECTED FAILURES
#

#
# GAMS
#

ExpectedFailures['gams', 'gms', 'MILP_unbounded'] = (
    lambda v: v <= _trunk_version,
    "GAMS requires finite bounds for integer variables. 1.0E100 is as extreme"
    "as GAMS will define, and should be enough to appear unbounded. If the"
    "solver cannot handle this bound, explicitly set a smaller bound on"
    "the pyomo model, or try a different GAMS solver.",
)

ExpectedFailures['gams', 'python', 'MILP_unbounded'] = (
    lambda v: v <= _trunk_version,
    "GAMS requires finite bounds for integer variables. 1.0E100 is as extreme"
    "as GAMS will define, and should be enough to appear unbounded. If the"
    "solver cannot handle this bound, explicitly set a smaller bound on"
    "the pyomo model, or try a different GAMS solver.",
)

#
# GLPK
#

ExpectedFailures['glpk', 'lp', 'MILP_discrete_var_bounds'] = (
    lambda v: v <= (4, 52, 0, 0),
    "Glpk ignores bounds on Binary variables through the "
    "LP file interface. A ticket has been filed.",
)

ExpectedFailures['glpk', 'mps', 'LP_duals_maximize'] = (
    lambda v: v <= _trunk_version,
    "Glpk does not accept the OBJSENSE section of the Free MPS format. "
    "Therefore maximization models are not explicitly handled.",
)

#
# CBC
#

ExpectedFailures['cbc', 'nl', 'MILP_unbounded'] = (
    lambda v: v <= _trunk_version,
    "Cbc fails to report an unbounded MILP model as unbounded through "
    "the NL interface (through 2.9.x), and fails with invalid free() "
    "(in 2.10.x).",
)

# The following is due to a bug introduced into Clp as part of CBC 2.10
# and was resolved by Clp commit 130dd199 (13 Feb 2021), and included
# in the CBC 2.10.6 release
ExpectedFailures['cbc', 'nl', 'LP_unbounded'] = (
    lambda v: v > (2, 10) and v < (2, 10, 6),
    "Cbc fails (invalid free()) for unbounded LP models through "
    "the NL interface in 2.10.x versions through 2.10.5 "
    "(reported upstream as coin-or/Cbc#389)",
)

ExpectedFailures['cbc', 'nl', 'SOS1_simple'] = (
    lambda v: v[:2] == (2, 10),
    "Cbc segfaults for SOS constraints in the NL interface "
    "(reported upstream as coin-or/Cbc#388)",
)

ExpectedFailures['cbc', 'nl', 'SOS2_simple'] = (
    lambda v: v[:2] == (2, 10),
    "Cbc segfaults for SOS constraints in the NL interface "
    "(reported upstream as coin-or/Cbc#388)",
)

#
# XPRESS
#
# NO EXPECTED FAILURES
#

#
# IPOPT
#

ExpectedFailures['ipopt', 'nl', 'LP_duals_maximize'] = (
    lambda v: v == (3, 10, 3, 0),
    "Ipopt returns duals with a different sign convention. Fixed in Ipopt 3.10.4",
)

ExpectedFailures['ipopt', 'nl', 'QCP_simple'] = (
    lambda v: v == (3, 10, 3, 0),
    "Ipopt returns duals with a different sign convention. Fixed in Ipopt 3.10.4",
)

ExpectedFailures['ipopt', 'nl', 'LP_block'] = (
    lambda v: v <= (3, 10, 2, 0),
    "Ipopt returns duals with a different sign convention. Fixed in Ipopt 3.10.3",
)

ExpectedFailures['ipopt', 'nl', 'LP_duals_minimize'] = (
    lambda v: v <= (3, 10, 2, 0),
    "Ipopt returns duals with a different sign convention. Fixed in Ipopt 3.10.3",
)

ExpectedFailures['ipopt', 'nl', 'LP_inactive_index'] = (
    lambda v: v <= (3, 10, 2, 0),
    "Ipopt returns duals with a different sign convention. Fixed in Ipopt 3.10.3",
)

ExpectedFailures['ipopt', 'nl', 'LP_piecewise'] = (
    lambda v: v <= (3, 10, 2, 0),
    "Ipopt returns duals with a different sign convention. Fixed in Ipopt 3.10.3",
)

ExpectedFailures['ipopt', 'nl', 'LP_simple'] = (
    lambda v: v <= (3, 10, 2, 0),
    "Ipopt returns duals with a different sign convention. Fixed in Ipopt 3.10.3",
)

ExpectedFailures['ipopt', 'nl', 'QP_simple'] = (
    lambda v: v <= (3, 10, 2, 0),
    "Ipopt returns duals with a different sign convention. Fixed in Ipopt 3.10.3",
)

ExpectedFailures['ipopt', 'nl', 'LP_trivial_constraints'] = (
    lambda v: v <= (3, 10, 2, 0),
    "Ipopt returns duals with a different sign convention. Fixed in Ipopt 3.10.3",
)

#
# SCIP
#

ExpectedFailures['scip', 'nl', 'SOS2_simple'] = (
    lambda v: v <= (3, 1, 0, 9),
    "SCIP (scip) does not recognize sos2 constraints "
    "inside NL files. A ticket has been filed.",
)

ExpectedFailures['scip', 'nl', 'SOS1_simple'] = (
    lambda v: v <= (3, 1, 0, 9),
    "SCIP (scip) does not recognize sos1 constraints "
    "inside NL files. A ticket has been filed.",
)

#
# BARON
#
SkipTests['baron', 'bar', 'LP_trivial_constraints'] = (
    lambda v: v[:3] == (22, 1, 19),
    'BARON 22.1.19 hits an infinite loop for this test case',
)

for prob in (
    'QP_simple_nosuffixes',
    'QP_simple_nosuffixes_kernel',
    'QP_simple',
    'QP_simple_kernel',
    'MIQP_simple',
    'MIQP_simple_kernel',
    'MILP_simple',
    'MILP_simple_kernel',
    'LP_simple',
    'LP_simple_kernel',
    'LP_block',
    'LP_block_kernel',
):
    ExpectedFailures['baron', 'bar', prob] = (
        lambda v: v[:3] == (22, 1, 19),
        'BARON 22.1.19 reports model as infeasible',
    )

for prob in ('LP_unbounded', 'LP_unbounded_kernel'):
    ExpectedFailures['baron', 'bar', prob] = (
        lambda v: v[:3] == (22, 1, 19),
        'BARON 22.1.19 reports model as optimal',
    )
for prob in (
    'LP_block',
    'LP_duals_maximize',
    'LP_duals_minimize',
    'LP_inactive_index',
    'LP_simple',
    'LP_trivial_constraints',
    'QCP_simple',
    'QP_simple',
):
    ExpectedFailures['baron', 'bar', prob] = (
        lambda v: (25, 7, 10) <= v[:3] <= (25, 7, 16),
        "BARON 25.7.16 returns 0 for duals/rc for models solved in preprocessing",
    )


#
# KNITROAMPL
#
for prob in ('LP_trivial_constraints', 'LP_trivial_constraints_kernel'):
    ExpectedFailures['knitroampl', 'nl', prob] = (
        lambda v: True,
        'Knitro does not consider tight trivial constraints to have zero dual value',
    )

for prob in ('MILP_unbounded', 'MILP_unbounded_kernel'):
    ExpectedFailures['knitroampl', 'nl', prob] = (
        lambda v: v[:2] <= (14, 2),
        'Unbounded MILP detection not operational in Knitro, fixed in 15.0',
    )


def generate_scenarios(arg=None):
    """
    Generate scenarios
    """
    for model in sorted(all_models()):
        _model = all_models(model)
        if not arg is None and not arg(_model):
            continue
        for solver, io in sorted(_test_solver_cases()):
            _solver_case = _test_solver_cases(solver, io)
            _ver = _solver_case.version

            # Skip this test case if the solver doesn't support the
            # capabilities required by the model
            if not _model.capabilities.issubset(_solver_case.capabilities):
                continue

            # Set status values for expected failures
            exclude_suffixes = {}
            status = 'ok'
            msg = ""
            case_skip = SkipTests.get((solver, io, _model.description), None)
            case_suffix = MissingSuffixFailures.get(
                (solver, io, _model.description), None
            )
            case_fail = ExpectedFailures.get((solver, io, _model.description), None)
            if not _solver_case.available:
                status = 'skip'
                msg = "Skipping test because solver %s (%s) is unavailable" % (
                    solver,
                    io,
                )
            elif case_skip is not None and _ver is not None and case_skip[0](_ver):
                status = 'skip'
                msg = case_skip[1]
            elif case_fail is not None and _ver is not None and case_fail[0](_ver):
                status = 'expected failure'
                msg = case_fail[1]
            elif case_suffix is not None and _ver is not None and case_suffix[0](_ver):
                if type(case_suffix[1]) is dict:
                    exclude_suffixes.update(case_suffix[1])
                else:
                    for x in case_suffix[1]:
                        exclude_suffixes[x] = (True, {})
                msg = case_suffix[2]

            # Return scenario dimensions and scenario information
            yield (model, solver, io), Bunch(
                status=status,
                msg=msg,
                model=_model,
                solver=None,
                testcase=_solver_case,
                demo_limits=_solver_case.demo_limits,
                exclude_suffixes=exclude_suffixes,
            )


def run_scenarios(options):
    logging.disable(logging.WARNING)

    solvers = set(options.solver)
    stat = {}

    for key, test_case in generate_scenarios():
        model, solver, io = key
        if len(solvers) > 0 and solver not in solvers:
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
            load_solutions,
        )

        termination_condition = results['Solver'][0]['termination condition']
        # Validate solution status
        try:
            model_class.post_solve_test_validation(None, results)
        except Exception:
            if test_case.status == 'expected failure':
                stat[key] = (True, "Expected failure")
            else:
                stat[key] = (
                    False,
                    "Unexpected termination condition: %s" % str(termination_condition),
                )
            continue
        if (
            termination_condition == TerminationCondition.unbounded
            or termination_condition == TerminationCondition.infeasible
        ):
            # Unbounded or Infeasible
            stat[key] = (True, "")
        else:
            # Validate the solution returned by the solver
            if isinstance(model_class.model, IBlock):
                model_class.model.load_solution(results.solution)
            else:
                model_class.model.solutions.load_from(
                    results, default_variable_value=opt.default_variable_value()
                )
            rc = model_class.validate_current_solution(
                suffixes=model_class.test_suffixes
            )

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
    total = Bunch(NumEPass=0, NumEFail=0, NumUPass=0, NumUFail=0)
    for key in stat:
        model, solver, io = key
        if solver not in summary:
            summary[solver] = Bunch(NumEPass=0, NumEFail=0, NumUPass=0, NumUFail=0)
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
                    print(
                        "- Unexpected Test Failure: " + ", ".join((model, solver, io))
                    )
            else:
                summary[solver].NumUPass += 1
                if options.verbose:
                    print(
                        "- Unexpected Test Success: " + ", ".join((model, solver, io))
                    )
    if options.verbose:
        if nfail == 0:
            print("- NONE")
        print("")

    stream = sys.stdout
    maxSolverNameLen = max([max(len(name) for name in summary), len("Solver")])
    fmtStr = (
        "{{0:<{0}}}| {{1:>8}} | {{2:>8}} | {{3:>10}} | {{4:>10}} | {{5:>13}}\n".format(
            maxSolverNameLen + 2
        )
    )
    #
    stream.write("\n")
    stream.write("Solver Test Summary\n")
    stream.write("=" * (maxSolverNameLen + 66) + "\n")
    stream.write(
        fmtStr.format("Solver", "# Pass", "# Fail", "# OK Fail", "# Bad Pass", "% OK")
    )
    stream.write("=" * (maxSolverNameLen + 66) + "\n")
    #
    for _solver in sorted(summary):
        ans = summary[_solver]
        total.NumEPass += ans.NumEPass
        total.NumEFail += ans.NumEFail
        total.NumUPass += ans.NumUPass
        total.NumUFail += ans.NumUFail
        stream.write(
            fmtStr.format(
                _solver,
                str(ans.NumEPass),
                str(ans.NumUFail),
                str(ans.NumEFail),
                str(ans.NumUPass),
                str(
                    int(
                        100.0
                        * (ans.NumEPass + ans.NumEFail)
                        / (ans.NumEPass + ans.NumEFail + ans.NumUFail + ans.NumUPass)
                    )
                ),
            )
        )
    #
    stream.write("=" * (maxSolverNameLen + 66) + "\n")
    stream.write(
        fmtStr.format(
            "TOTALS",
            str(total.NumEPass),
            str(total.NumUFail),
            str(total.NumEFail),
            str(total.NumUPass),
            str(
                int(
                    100.0
                    * (total.NumEPass + total.NumEFail)
                    / (
                        total.NumEPass
                        + total.NumEFail
                        + total.NumUFail
                        + total.NumUPass
                    )
                )
            ),
        )
    )
    stream.write("=" * (maxSolverNameLen + 66) + "\n")

    logging.disable(logging.NOTSET)


if __name__ == "__main__":
    print("")
    print("Testing model generation")
    print("-" * 30)
    for key in sorted(all_models()):
        print(key)
        obj = all_models(key)()
        obj.generate_model()
        obj.warmstart_model()

    print("")
    print("Testing scenario generation")
    print("-" * 30)
    for key, value in generate_scenarios():
        print(", ".join(key))
        print("   %s: %s" % (value.status, value.msg))

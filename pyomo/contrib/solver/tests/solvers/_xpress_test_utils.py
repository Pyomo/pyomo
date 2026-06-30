# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Shared test utilities for the Xpress connector test suite.

Imported by test_xpress_direct.py and test_xpress_persistent.py.
Not a test module itself (underscore prefix prevents pytest collection).
"""

import pyomo.environ as pyo

from typing import TypedDict

from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition


def _simple_lp():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.NonNegativeReals)
    m.y = pyo.Var(domain=pyo.NonNegativeReals)
    m.c1 = pyo.Constraint(expr=m.x + m.y <= 4)
    m.c2 = pyo.Constraint(expr=2 * m.x + m.y <= 6)
    m.obj = pyo.Objective(expr=-m.x - 2 * m.y)
    return m


def _simple_mip():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.NonNegativeIntegers)
    m.y = pyo.Var(domain=pyo.NonNegativeIntegers)
    m.c1 = pyo.Constraint(expr=m.x + m.y <= 4)
    m.obj = pyo.Objective(expr=-m.x - 2 * m.y)
    return m


class _SolveExpected(TypedDict, total=False):
    termination: TerminationCondition  # default: convergenceCriteriaSatisfied
    status: SolutionStatus             # default: optimal
    objective: float                   # required when status is optimal
    vars: list                         # required when status is optimal; [(pyo_var, float), ...]
    obj_places: int                    # default: 6
    var_places: int                    # default: 6


def _solve_and_check(test_case, opt, model, expected: _SolveExpected, **solve_kwargs):
    """Solve model, enforce baseline health checks, return the result.

    Always asserts termination_condition and solution_status.
    When status resolves to optimal, 'objective' and 'vars' are mandatory
    in expected -- a test that solves optimally without checking the solution
    is not a useful test.

    **solve_kwargs: forwarded verbatim to opt.solve().
    """
    tc_default = TerminationCondition.convergenceCriteriaSatisfied
    st_default = SolutionStatus.optimal

    termination = expected.get('termination', tc_default)
    status = expected.get('status', st_default)
    obj_places = expected.get('obj_places', 6)
    var_places = expected.get('var_places', 6)

    if status == st_default:
        assert (
            'objective' in expected
        ), "_solve_and_check: 'objective' is required when status is optimal"
        assert (
            'vars' in expected
        ), "_solve_and_check: 'vars' is required when status is optimal"
        num_model_vars = model.nvariables()
        assert len(expected['vars']) == num_model_vars, (
            f"_solve_and_check: 'vars' must cover all {num_model_vars} active variables "
            f"in the model, got {len(expected['vars'])}"
        )

    res = opt.solve(model, **solve_kwargs)
    tc = test_case
    tc.assertEqual(res.termination_condition, termination)
    tc.assertEqual(res.solution_status, status)
    if 'objective' in expected:
        tc.assertAlmostEqual(
            res.incumbent_objective, expected['objective'], places=obj_places
        )
    if 'vars' in expected:
        for var, expected_val in expected['vars']:
            tc.assertAlmostEqual(pyo.value(var), expected_val, places=var_places)
    return res


def _trivial_model():
    """Single bounded variable, no constraints, minimise x.

    Used by persistent API surface tests that need a live xp.problem but do not
    care about the specific solution (handles, controls, state checks, etc.).
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.obj = pyo.Objective(expr=m.x)
    return m


def _solve_lp_no_load(opt):
    """Return (model, result) for _simple_lp() solved with load_solutions=False.

    Avoids the two-line preamble that is repeated in every test that exercises
    the solution-loader interface (get_vars, get_duals, get_reduced_costs, etc.).
    """
    m = _simple_lp()
    res = opt.solve(m, load_solutions=False)
    return m, res


def _solve_check_mutate_check(
    test_case,
    opt,
    model,
    expected_before: _SolveExpected,
    param,
    new_value,
    expected_after: _SolveExpected,
    **solve_kwargs,
):
    """Two-step mutable param test: solve + check, mutate param, solve + check again.

    Covers the pervasive pattern in persistent tests where one param change is
    applied between two consecutive solves and both solutions are verified.
    Returns (res_before, res_after).

    Only use when there are no assertions that compare values across the two
    solves (e.g. assertLess(x2, x1)). Those tests must keep explicit solve calls.
    """
    res_before = _solve_and_check(test_case, opt, model, expected_before, **solve_kwargs)
    param.set_value(new_value)
    res_after = _solve_and_check(test_case, opt, model, expected_after, **solve_kwargs)
    return res_before, res_after

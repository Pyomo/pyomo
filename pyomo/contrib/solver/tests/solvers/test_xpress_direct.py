# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import math
import os
import tempfile
from unittest.mock import MagicMock

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.timing import HierarchicalTimer

from pyomo.contrib.solver.common.results import TerminationCondition
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoDualsError,
    NoFeasibleSolutionError,
    NoReducedCostsError,
    NoSolutionError,
)
from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.contrib.solver.solvers.xpress import XpressDirect
from pyomo.contrib.solver.solvers.xpress.xpress_base import _exit_external_function
from pyomo.contrib.solver.tests.solvers._xpress_test_utils import (
    _simple_lp,
    _simple_mip,
    _solve_and_check,
    _solve_lp_no_load,
)

if not XpressDirect().available():
    raise unittest.SkipTest('Xpress not available')


def _infeasible():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.x >= 10)
    m.c2 = pyo.Constraint(expr=m.x <= 1)
    m.obj = pyo.Objective(expr=m.x)
    return m


@unittest.pytest.mark.solver("xpress_direct")
class TestXpressDirect(unittest.TestCase):
    def setUp(self):
        self.opt = XpressDirect()

    def test_symbolic_solver_labels_lp(self):
        m = pyo.ConcreteModel()
        m.distinctive_x = pyo.Var(domain=pyo.NonNegativeReals)
        m.distinctive_y = pyo.Var(domain=pyo.NonNegativeReals)
        m.distinctive_c1 = pyo.Constraint(expr=m.distinctive_x + m.distinctive_y <= 4)
        m.distinctive_c2 = pyo.Constraint(
            expr=2 * m.distinctive_x + m.distinctive_y <= 6
        )
        m.obj = pyo.Objective(expr=-m.distinctive_x - 2 * m.distinctive_y)
        res = _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -8.0,
                'vars': [(m.distinctive_x, 0.0), (m.distinctive_y, 4.0)],
            },
            symbolic_solver_labels=True,
        )
        # Verify that Pyomo names appear in the LP file -- symbolic_solver_labels
        # must not be silently dropped.
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, 'm')
            res.solution_loader._xp_prob.writeProb(base + '.lp', flags='l')
            with open(base + '.lp', 'r') as f:
                content = f.read()
        self.assertIn('distinctive_x', content)
        self.assertIn('distinctive_c1', content)

    def test_symbolic_solver_labels_mip(self):
        m = pyo.ConcreteModel()
        m.distinctive_x = pyo.Var(domain=pyo.NonNegativeIntegers)
        m.distinctive_y = pyo.Var(domain=pyo.NonNegativeIntegers)
        m.distinctive_c1 = pyo.Constraint(expr=m.distinctive_x + m.distinctive_y <= 4)
        m.obj = pyo.Objective(expr=-m.distinctive_x - 2 * m.distinctive_y)
        res = _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -8.0,
                'vars': [(m.distinctive_x, 0.0), (m.distinctive_y, 4.0)],
            },
            symbolic_solver_labels=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, 'm')
            res.solution_loader._xp_prob.writeProb(base + '.lp', flags='l')
            with open(base + '.lp', 'r') as f:
                content = f.read()
        self.assertIn('distinctive_x', content)
        self.assertIn('distinctive_c1', content)

    def test_positive_time_limit(self):
        m = _simple_lp()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            time_limit=60,
        )

    def test_solver_options_passthrough(self):
        m = _simple_lp()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            solver_options={'outputlog': 0},
        )
        # Invalid control names are forwarded to Xpress and raise -- proves
        # solver_options are not silently ignored.
        with self.assertRaises(Exception):
            self.opt.solve(m, solver_options={'_invalid_control_xyz': 1})

    def test_rel_gap(self):
        # Verify rel_gap is accepted and the solve completes without error.
        # A trivial MIP solves to optimality regardless of gap tolerance.
        m = _simple_mip()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            rel_gap=0.01,
        )

    def test_threads(self):
        m = _simple_lp()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            threads=1,
        )

    def test_mip_no_duals_no_reduced_costs(self):
        m = _simple_mip()
        res = self.opt.solve(m, load_solutions=False)
        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        with self.assertRaises(NoDualsError):
            res.solution_loader.get_duals()
        with self.assertRaises(NoReducedCostsError):
            res.solution_loader.get_reduced_costs()

    def test_get_vars_no_solution(self):
        m = _infeasible()
        res = _solve_and_check(
            self,
            self.opt,
            m,
            {
                'termination': TerminationCondition.provenInfeasible,
                'status': SolutionStatus.infeasible,
            },
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
        )
        with self.assertRaises(NoSolutionError):
            res.solution_loader.get_vars()

    def test_warmstart(self):
        m = _simple_mip()
        # Provide a valid feasible hint: x=0, y=4 satisfies all constraints
        # and is the optimal solution. Verify that the hint is consumed (at least
        # one MIP integer solution found, which includes the warm-start point).
        m.x.set_value(0)
        m.y.set_value(4)
        res = _solve_and_check(
            self, self.opt, m, {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]}
        )
        self.assertGreaterEqual(res.extra_info.mip_solutions_found, 1)

    def test_extra_info_and_timing(self):
        m = _simple_lp()
        res = _solve_and_check(
            self, self.opt, m, {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]}
        )
        self.assertGreaterEqual(res.timing_info.xpress_time, 0)
        # LP always runs at least one iteration
        self.assertGreaterEqual(res.extra_info.simplex_iterations, 1)
        self.assertGreaterEqual(res.extra_info.barrier_iterations, 0)
        # LP has no B&B nodes
        self.assertEqual(res.extra_info.node_count, 0)
        # LP has no MIP solutions
        self.assertEqual(res.extra_info.mip_solutions_found, 0)

    def test_load_solutions_infeasible(self):
        m = _infeasible()
        with self.assertRaises(NoFeasibleSolutionError):
            self.opt.solve(
                m, raise_exception_on_nonoptimal_result=False, load_solutions=True
            )

    def test_load_vars_subset(self):
        m, res = _solve_lp_no_load(self.opt)
        m.x.set_value(99.0)
        m.y.set_value(99.0)
        res.solution_loader.load_vars([m.y])
        self.assertAlmostEqual(m.y.value, 4.0)
        self.assertAlmostEqual(m.x.value, 99.0)  # untouched
        res.solution_loader.load_vars([m.x, m.y])
        self.assertAlmostEqual(m.x.value, 0.0)
        self.assertAlmostEqual(m.y.value, 4.0)

    def test_get_vars_subset(self):
        m, res = _solve_lp_no_load(self.opt)
        result = res.solution_loader.get_vars([m.x])
        self.assertIn(m.x, result)
        self.assertNotIn(m.y, result)
        self.assertAlmostEqual(result[m.x], 0.0)
        # two variables -- both orders must produce correct mapping
        result = res.solution_loader.get_vars([m.x, m.y])
        self.assertAlmostEqual(result[m.x], 0.0)
        self.assertAlmostEqual(result[m.y], 4.0)
        result = res.solution_loader.get_vars([m.y, m.x])
        self.assertAlmostEqual(result[m.x], 0.0)
        self.assertAlmostEqual(result[m.y], 4.0)

    def test_get_reduced_costs_subset(self):
        # y is in-basis at the LP optimum, so its reduced cost is 0.
        m, res = _solve_lp_no_load(self.opt)
        result = res.solution_loader.get_reduced_costs([m.y])
        self.assertIn(m.y, result)
        self.assertNotIn(m.x, result)
        self.assertAlmostEqual(result[m.y], 0.0)
        result = res.solution_loader.get_reduced_costs([m.x, m.y])
        self.assertIn(m.x, result)
        self.assertIn(m.y, result)
        self.assertAlmostEqual(result[m.y], 0.0)

    def test_get_vars_all(self):
        # get_vars() with no argument exercises the full-variable path in
        # _get_solution_vals (vars_to_load=None -> return all self._vars).
        m, res = _solve_lp_no_load(self.opt)
        result = res.solution_loader.get_vars()
        self.assertIn(m.x, result)
        self.assertIn(m.y, result)
        self.assertAlmostEqual(result[m.x], 0.0)
        self.assertAlmostEqual(result[m.y], 4.0)

    def test_multiple_objectives_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.obj1 = pyo.Objective(expr=m.x)
        m.obj2 = pyo.Objective(expr=-m.x)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

    def test_sos1_direct(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.sos1 = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        m.obj = pyo.Objective(expr=m.x[1] + 2 * m.x[2] + 3 * m.x[3], sense=pyo.maximize)
        # SOS1 optimal: only x[3] can be nonzero (highest weight/coefficient), x[3]=1 -> obj=3.0
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 3.0, 'vars': [(m.x[1], 0.0), (m.x[2], 0.0), (m.x[3], 1.0)]},
        )

    def test_sos1_vars_not_in_objective(self):
        # SOS vars that do NOT appear in any constraint or objective.
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.y = pyo.Var(bounds=(0, 10))
        m.sos1 = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        m.obj = pyo.Objective(expr=m.y)  # x vars intentionally absent
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': 0.0,
                'vars': [(m.x[1], 0.0), (m.x[2], 0.0), (m.x[3], 0.0), (m.y, 0.0)],
            },
        )

    def test_sos1_no_duplicate_columns(self):
        # SOS vars that also appear in the objective must NOT get duplicate
        # Xpress columns -- the problem should have exactly 3 columns (x[1..3]).
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.sos1 = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        m.obj = pyo.Objective(expr=m.x[1] + 2 * m.x[2] + 3 * m.x[3], sense=pyo.maximize)
        xp_prob = self.opt._create_xpress_model(
            m,
            self.opt.config,
            __import__(
                'pyomo.common.timing', fromlist=['HierarchicalTimer']
            ).HierarchicalTimer(),
        )[0]
        self.assertEqual(xp_prob.attributes.cols, 3)

    def test_get_duals_single_constraint(self):
        # Xpress may return a scalar (not a list) when queried for a single
        # constraint dual -- the scalar-to-list normalization must be exercised.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x >= 2)
        m.obj = pyo.Objective(expr=m.x)
        res = _solve_and_check(
            self, self.opt, m, {'objective': 2.0, 'vars': [(m.x, 2.0)]}
        )
        duals = res.solution_loader.get_duals([m.c])
        self.assertIn(m.c, duals)
        self.assertIsInstance(duals[m.c], float)

    def test_reduced_costs_value_correctness(self):
        # _simple_lp() optimal: x=0 (non-basic at lb), y=4 (in-basis).
        # Analytically: RC[x] = c_x - u1 * a_{x,c1} = -1 - (-2)*1 = 1.0 exactly.
        # RC[y] = 0 (y is basic in the LP).
        m, res = _solve_lp_no_load(self.opt)
        rcs = res.solution_loader.get_reduced_costs()
        self.assertAlmostEqual(rcs[m.x], 1.0, places=6)
        self.assertAlmostEqual(rcs[m.y], 0.0, places=6)

    def test_duals_value_correctness(self):
        # c1 (x+y<=4) is binding at the optimum.
        # Shadow price: relaxing c1 by 1 allows y to increase by 1, improving obj by -2.
        # Xpress convention: dual[c1] = -2.0 for this minimization problem.
        # c2 (2x+y<=6) has slack 2 at the optimum -> dual = 0.
        m, res = _solve_lp_no_load(self.opt)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c1], -2.0, places=6)
        self.assertAlmostEqual(duals[m.c2], 0.0, places=6)


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressDirectQuadratic(unittest.TestCase):
    def setUp(self):
        self.opt = XpressDirect()

    def test_qp_objective_direct(self):
        # min x^2 + y^2  s.t. x + y >= 1  ->  optimal x=y=0.5, obj=0.5
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1)
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        _solve_and_check(
            self, self.opt, m, {'objective': 0.5, 'vars': [(m.x, 0.5), (m.y, 0.5)]}
        )

    def test_qcp_constraint_direct(self):
        # max x + y  s.t. x^2 + y^2 <= 1, x >= 0, y >= 0  ->  optimal at x=y=1/sqrt(2)
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, None))
        m.y = pyo.Var(bounds=(0, None))
        m.qc = pyo.Constraint(expr=m.x**2 + m.y**2 <= 1)
        m.obj = pyo.Objective(expr=-(m.x + m.y))  # maximize x+y

        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -math.sqrt(2),
                'vars': [(m.x, math.sqrt(2) / 2), (m.y, math.sqrt(2) / 2)],
                'obj_places': 5,
                'var_places': 5,
            },
        )

    def test_nl_cubic_constraint_direct(self):
        # x**3 >= 1, min x: optimal solution is x=1 via Xpress NLP solver.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x**3 >= 1)
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, 1.0)]})


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressDirectMisc(unittest.TestCase):
    """Edge cases and rarely-hit branches for the direct connector."""

    def setUp(self):
        self.opt = XpressDirect()

    def test_nl_cubic_objective_direct(self):
        # min x**3 for x in [0, 1]: optimal is x=0 (boundary minimum).
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x**3)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})

    def test_abs_gap_passthrough(self):
        # Verifies abs_gap config is forwarded to mipabsstop. Trivial MIP solves
        # to optimality regardless of gap tolerance.
        m = _simple_mip()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            abs_gap=0.5,
        )

    def test_working_dir_chdir_and_restore(self):
        # working_dir must chdir into the directory before optimize and restore
        # the original cwd after, even if optimize raises.
        m = _simple_lp()
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            self.opt.solve(m, working_dir=tmp)
            self.assertEqual(os.getcwd(), original_cwd)

    def test_empty_constraint_model(self):
        # No constraints at all: only bounds and objective. Optimal at lower bound.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(2, 10))
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(self, self.opt, m, {'objective': 2.0, 'vars': [(m.x, 2.0)]})

    def test_no_objective_feasibility(self):
        # No objective: Xpress treats it as a feasibility problem. The reported
        # incumbent_objective should be None (has_obj=False path).
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x >= 3)
        res = self.opt.solve(m)
        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertIsNone(res.incumbent_objective)

    def test_constant_objective(self):
        # Objective with no variable terms (constant only). Exercises the
        # len(xp_vars) == 0 branch in _set_objective_impl.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.obj = pyo.Objective(expr=5.0)
        # x value is solver-determined (feasibility only); Xpress returns x=1.0 (lb of constraint)
        _solve_and_check(self, self.opt, m, {'objective': 5.0, 'vars': [(m.x, 1.0)]})

    def test_range_constraint_lp(self):
        # Range constraint: 1 <= x + y <= 3. Exercises the 'R' rowtype path
        # in get_rhs_and_sense (rng_arr branch).
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x + m.y, 3))
        # obj=-2x-y: unique optimal at upper bound is x=3, y=0 (x has larger coefficient)
        m.obj = pyo.Objective(expr=-2 * m.x - m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': -6.0, 'vars': [(m.x, 3.0), (m.y, 0.0)]}
        )

    def test_get_duals_no_args(self):
        # Default cons_to_load=None path exercises maps.cons.values() ordering.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c1 = pyo.Constraint(expr=m.x >= 1)
        m.c2 = pyo.Constraint(expr=m.x >= 2)
        m.obj = pyo.Objective(expr=m.x)
        res = _solve_and_check(
            self, self.opt, m, {'objective': 2.0, 'vars': [(m.x, 2.0)]}
        )
        duals = res.solution_loader.get_duals()
        self.assertIn(m.c1, duals)
        self.assertIn(m.c2, duals)

    def test_fixed_var_without_value_raises(self):
        # var.fix() with no argument sets fixed=True but leaves value=None.
        # _var_bounds must raise a descriptive error rather than the opaque
        # TypeError: float() argument must be a real number, not NoneType.
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.x.fix()  # fixed=True, value stays None
        m.obj = pyo.Objective(expr=m.x)
        self.assertIsNone(m.x.value)
        with self.assertRaises(ValueError):
            self.opt.solve(m)

    def test_controls_unit(self):
        # Unit test for _apply_solver_controls: verify controls are actually written
        # to xp.problem. A bug that silently drops the call would leave all four
        # solver-config tests green while this one fails.

        m = _simple_lp()
        opt = XpressDirect()
        xp_prob, _, _ = opt._create_xpress_model(m, opt.config, HierarchicalTimer())
        config = opt.config({'time_limit': 42, 'threads': 2})
        opt._apply_solver_controls(xp_prob, config)
        self.assertEqual(xp_prob.controls.timelimit, 42.0)
        self.assertEqual(xp_prob.controls.threads, 2)

    def test_infeasible_model_returns_infeasible_result(self):
        # Model with lb > ub (inverted bounds) is detected as infeasible by Xpress.
        # The connector must return provenInfeasible without raising.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(5, 3))  # inverted: lb=5 > ub=3
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'termination': TerminationCondition.provenInfeasible,
                'status': SolutionStatus.infeasible,
            },
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
        )

    def test_time_limit_zero(self):
        # time_limit=0 passes 0.0 directly to timelimit (Xpress interprets 0 as no limit).
        m = _simple_lp()
        opt = XpressDirect()
        xp_prob, _, _ = opt._create_xpress_model(m, opt.config, HierarchicalTimer())
        config = opt.config({'time_limit': 0})
        opt._apply_solver_controls(xp_prob, config)
        self.assertEqual(xp_prob.controls.timelimit, 0.0)

    def test_working_dir_restored_on_exception(self):
        # The finally block in solve() restores cwd even when an exception propagates
        # (e.g., from an invalid solver option -- distinct from the InfeasibleConstraintException
        # catch which is handled separately).
        m = _simple_lp()
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                self.opt.solve(
                    m, working_dir=tmp, solver_options={'_invalid_control_xyz': 1}
                )
            except Exception:
                pass  # expected: invalid control raises inside the try block
            self.assertEqual(os.getcwd(), original_cwd)


@unittest.pytest.mark.solver('xpress_persistent')
class TestXpressDirectNLP(unittest.TestCase):
    """NLP integration tests for the direct connector."""

    def setUp(self):
        self.opt = XpressDirect()

    def _check_optimal(self, res):
        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )

    def test_nl_exp_objective_linear_constraints(self):
        # min exp(x) s.t. x >= 1, x in [0,3]
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 3))
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.obj = pyo.Objective(expr=pyo.exp(m.x))
        _solve_and_check(self, self.opt, m, {'objective': math.e, 'vars': [(m.x, 1.0)]})

    def test_nl_sin_constraints_linear_objective(self):
        # min x+y s.t. sin(x) + y <= 1, x in [0, pi/2], y in [0,1]
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, math.pi / 2))
        m.y = pyo.Var(bounds=(0, 1))
        m.c = pyo.Constraint(expr=pyo.sin(m.x) + m.y <= 1)
        m.obj = pyo.Objective(expr=m.x + m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0), (m.y, 0.0)]}
        )

    def test_nl_objective_nl_constraint(self):
        # min sin(x) s.t. exp(x) <= 2, x in [0, 2]
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 2))
        m.c = pyo.Constraint(expr=pyo.exp(m.x) <= 2)
        m.obj = pyo.Objective(expr=pyo.sin(m.x))
        # exp(x) <= 2 -> x <= ln(2); minimizing sin(x) on [0, ln(2)] -> x=0
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})

    def test_nl_range_constraint(self):
        # 0.5 <= sin(x) <= 1, min x for x in [0, pi]
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, math.pi))
        m.c = pyo.Constraint(expr=pyo.inequality(0.5, pyo.sin(m.x), 1.0))
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(
            self, self.opt, m, {'objective': math.pi / 6, 'vars': [(m.x, math.pi / 6)]}
        )
        self.assertAlmostEqual(pyo.sin(pyo.value(m.x)), 0.5, places=6)

    def test_fixed_variable_in_nl_constraint(self):
        # sin(x_fixed) + y <= 5, min y; with x fixed at pi/2, sin(x)=1.
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.x.fix(math.pi / 2)
        m.c = pyo.Constraint(expr=pyo.sin(m.x) + m.y <= 5)
        m.obj = pyo.Objective(expr=m.y)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 0.0, 'vars': [(m.x, math.pi / 2), (m.y, 0.0)]},
        )

    def test_nl_abs_objective(self):
        # min abs(x - 2), x in [0, 5]: optimal x=2, obj=0
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj = pyo.Objective(expr=abs(m.x - 2))
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 2.0)]})


@unittest.pytest.mark.solver('xpress_persistent')
@unittest.pytest.mark.solver('xpress_direct')
class TestXpressExternalFunction(unittest.TestCase):
    """Integration tests for ExternalFunction support via xp.user (SLP)."""

    def setUp(self):
        self.opt = XpressDirect()

    def _check_solved(self, res):
        # xp.user (SLP) reports SolutionStatus.feasible on local-optimum convergence,
        # not SS.optimal (which requires global-optimality proof).
        self.assertIn(
            res.solution_status, (SolutionStatus.optimal, SolutionStatus.feasible)
        )
        self.assertIsNotNone(res.incumbent_objective)

    def test_external_function_no_gradient(self):
        """ExternalFunction without gradient: model builds and solves correctly.

        min f(x,y) = x^2 + y  s.t.  x in [0,5], y in [1,5].
        Optimum: x=0, y=1, obj=1.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(1, 5))
        m.f = pyo.ExternalFunction(function=lambda x, y: x**2 + y)
        m.obj = pyo.Objective(expr=m.f(m.x, m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'status': SolutionStatus.feasible,
                'objective': 1.0,
                'vars': [(m.x, 0.0), (m.y, 1.0)],
            },
        )

    def test_external_function_with_gradient(self):
        """ExternalFunction with gradient: derivatives='always' path; solves correctly.

        min f(x,y) = x^2 + y  s.t.  x in [0,5], y in [1,5].
        Optimum: x=0, y=1, obj=1.
        """

        def f(x, y):
            return x**2 + y

        def grad(args, fixed):
            x, _ = args
            return [2 * x, 1.0]

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(1, 5))
        m.f = pyo.ExternalFunction(function=f, gradient=grad)
        m.obj = pyo.Objective(expr=m.f(m.x, m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'status': SolutionStatus.feasible,
                'objective': 1.0,
                'vars': [(m.x, 0.0), (m.y, 1.0)],
            },
        )

    def test_external_function_ampl_raises(self):
        """Non-PythonCallbackFunction must raise IncompatibleModelError.

        Calls _exit_external_function directly with a mock to avoid needing a
        working AMPLExternalFunction (which requires a shared library on disk).
        """
        node = MagicMock()
        node._fcn = MagicMock()  # not a PythonCallbackFunction instance
        with self.assertRaises(IncompatibleModelError):
            _exit_external_function(None, node)

    def test_external_function_in_constraint(self):
        """External function in a constraint: constraint is respected at optimum.

        g(y) = y  (identity),  constraint g(y) >= 1,  objective min y, y in [0,5].
        Optimum: y=1, objective=1.  The external function identity forces y>=1.
        """

        def g(y):
            return y

        def grad(args, fixed):
            return [1.0]

        m = pyo.ConcreteModel()
        m.y = pyo.Var(bounds=(0, 5))
        m.g = pyo.ExternalFunction(function=g, gradient=grad)
        m.c = pyo.Constraint(expr=m.g(m.y) >= 1)
        m.obj = pyo.Objective(expr=m.y)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'status': SolutionStatus.feasible, 'objective': 1.0, 'vars': [(m.y, 1.0)]},
        )

    def test_external_function_multiple(self):
        """Two external functions in the same model.

        Objective: f1(x) + f2(y) = x^2 + (y+1), x in [0,5], y in [0,5].
        Optimum: x=0, y=0, obj=1.
        """

        def f1(x):
            return x**2

        def f2(y):
            return y + 1.0

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(0, 5))
        m.f1 = pyo.ExternalFunction(function=f1)
        m.f2 = pyo.ExternalFunction(function=f2)
        m.obj = pyo.Objective(expr=m.f1(m.x) + m.f2(m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'status': SolutionStatus.feasible,
                'objective': 1.0,
                'vars': [(m.x, 0.0), (m.y, 0.0)],
            },
        )

    def test_external_function_fgh_callback(self):
        """ExternalFunction with fgh= callback: exercises the _fgh code path in
        _exit_external_function (xpress_base.py lines 176-180).

        fgh(args, fgh_flag, fixed) returns (f, g, None):
          f  = x^2 + y        (function value)
          g  = [2x, 1.0]     (gradient, computed when fgh_flag != 0)

        min x^2 + y  s.t.  x in [0,5], y in [1,5] -> optimal x=0, y=1, obj=1.
        """

        def fgh_func(args, fgh_flag, fixed):
            x, y = args
            f = x**2 + y
            g = [2.0 * x, 1.0] if fgh_flag else None
            return f, g, None

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(1, 5))
        m.f = pyo.ExternalFunction(fgh=fgh_func)
        m.obj = pyo.Objective(expr=m.f(m.x, m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'status': SolutionStatus.feasible,
                'objective': 1.0,
                'vars': [(m.x, 0.0), (m.y, 1.0)],
            },
        )


if __name__ == '__main__':
    unittest.main()

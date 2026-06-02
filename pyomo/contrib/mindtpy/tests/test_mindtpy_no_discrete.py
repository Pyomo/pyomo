# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


from unittest.mock import MagicMock, patch

from pyomo.opt import TerminationCondition as tc, SolverStatus
import pyomo.common.unittest as unittest

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    SolverFactory,
    Var,
    minimize,
    maximize,
    value,
)

required_nlp_solvers = 'ipopt'
# Open-source (or generally available) solver pair used by MindtPy tests that
# don't require persistent commercial solvers.
if SolverFactory('appsi_highs').available(exception_flag=False) and SolverFactory(
    'appsi_highs'
).version() >= (1, 7, 0):
    short_circuit_required_solvers = ('ipopt', 'appsi_highs')
else:
    short_circuit_required_solvers = ('ipopt', 'glpk')

short_circuit_subsolvers_available = all(
    SolverFactory(s).available(exception_flag=False)
    for s in short_circuit_required_solvers
)


@unittest.skipIf(
    not short_circuit_subsolvers_available,
    'Required subsolvers %s are not available' % (short_circuit_required_solvers,),
)
class TestMindtPyShortCircuitNoDiscrete(unittest.TestCase):
    def test_short_circuit_model_with_no_objective_uses_temporary_dummy_objective(self):
        """No-objective models should short-circuit and leave the user model unchanged."""
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        m.c1 = Constraint(expr=m.x >= 1)
        m.c2 = Constraint(expr=m.x <= 1)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertEqual(results.solver.termination_condition, tc.optimal)
        self.assertEqual(results.problem.number_of_objectives, 0)
        self.assertAlmostEqual(m.x.value, 1.0, places=4)
        self.assertEqual(
            len(list(m.component_data_objects(ctype=Objective, active=True))), 0
        )

    def test_no_discrete_decisions_short_circuit_loads_values(self):
        """Regression test for MindtPy short-circuit with no discrete decisions.

        If all discrete variables are fixed, MindtPy should directly solve the
        original model (LP/NLP) and still return a valid SolverResults and load
        primal values onto the provided model.
        """
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        # Nonlinear constraint forces the NLP short-circuit branch
        m.c = Constraint(expr=m.x**2 >= 1 + m.y)
        m.objective = Objective(expr=m.x, sense=minimize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertIn(
            results.solver.termination_condition,
            [tc.optimal, tc.locallyOptimal, tc.feasible],
        )
        # Core regression: primal values must be loaded onto the model.
        self.assertIsNotNone(
            m.x.value,
            "x.value is None; MindtPy did not populate primal values in the short-circuit path",
        )
        obj_val = value(m.objective.expr, exception=False)
        self.assertIsNotNone(
            obj_val, "Objective evaluates to None; model variables were not populated"
        )
        # Sanity check on the solution (y is fixed to 0, so x >= 1)
        self.assertGreaterEqual(m.x.value, 1.0 - 1e-6)
        self.assertAlmostEqual(m.x.value, 1.0, places=4)
        self.assertAlmostEqual(obj_val, 1.0, places=4)

    def test_short_circuit_infeasible_nlp_returns_valid_results(self):
        """Infeasible NLP short-circuit should return results, not load bad data."""
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        # Infeasible: x >= 0 AND x^2 <= -1
        m.c1 = Constraint(expr=m.x >= 0)
        m.c2 = Constraint(expr=m.x**2 <= -1)
        m.objective = Objective(expr=m.x, sense=minimize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertEqual(results.solver.termination_condition, tc.infeasible)

    def test_short_circuit_linear_model_uses_lp_path(self):
        """Linear model with fixed discrete should use LP short-circuit."""
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        # Pure LP (polynomial degree 1)
        m.c = Constraint(expr=m.x >= 1)
        m.objective = Objective(expr=m.x, sense=minimize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertEqual(results.solver.termination_condition, tc.optimal)
        self.assertAlmostEqual(m.x.value, 1.0, places=4)

    def test_short_circuit_minimize_nlp_correct_solution(self):
        """Minimization NLP short-circuit should solve correctly."""
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        m.c = Constraint(expr=m.x**2 >= 1)
        m.objective = Objective(expr=m.x, sense=minimize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertAlmostEqual(m.x.value, 1.0, places=4)

    def test_short_circuit_maximize_nlp(self):
        """Maximization NLP short-circuit should set lower bound only."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(domain=Binary)
        m.y.fix(0)

        # Nonlinear constraint: x^2 <= 4 => x <= 2
        m.c = Constraint(expr=m.x**2 <= 4)
        m.objective = Objective(expr=m.x, sense=maximize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertIn(
            results.solver.termination_condition,
            [tc.optimal, tc.locallyOptimal, tc.feasible],
        )
        self.assertAlmostEqual(m.x.value, 2.0, places=4)
        # Lower bound should be populated (primal bound for maximization)
        self.assertIsNotNone(results.problem.lower_bound)

    def test_short_circuit_maximize_lp(self):
        """Maximization LP short-circuit should set lower bound only."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.y = Var(domain=Binary)
        m.y.fix(0)

        m.c = Constraint(expr=m.x <= 5)
        m.objective = Objective(expr=m.x, sense=maximize)

        with SolverFactory('mindtpy') as opt:
            results = opt.solve(
                m,
                strategy='OA',
                mip_solver=short_circuit_required_solvers[1],
                nlp_solver=short_circuit_required_solvers[0],
            )

        self.assertIsNotNone(results)
        self.assertEqual(results.solver.termination_condition, tc.optimal)
        self.assertAlmostEqual(m.x.value, 5.0, places=4)

    def test_short_circuit_multiobjective_model_raises(self):
        """Multiple active objectives should be rejected before short-circuit solve."""
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=Binary)
        m.y.fix(0)

        m.c = Constraint(expr=m.x >= 1)
        m.obj1 = Objective(expr=m.x, sense=minimize)
        m.obj2 = Objective(expr=2 * m.x, sense=minimize)

        with SolverFactory('mindtpy') as opt:
            with self.assertRaisesRegex(
                ValueError, 'Model has multiple active objectives.'
            ):
                opt.solve(
                    m,
                    strategy='OA',
                    mip_solver=short_circuit_required_solvers[1],
                    nlp_solver=short_circuit_required_solvers[0],
                )


class _SimpleNamespace:
    """A plain object for tracking attribute assignments in tests."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)


class TestMirrorDirectSolveResults(unittest.TestCase):
    """Unit tests for _mirror_direct_solve_results covering all branches."""

    _SENTINEL = object()  # marker for "never assigned"

    def _make_algorithm_stub(self):
        """Create a minimal stub of _MindtPyAlgorithm with only the fields
        needed by _mirror_direct_solve_results."""
        from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm

        stub = MagicMock(spec=_MindtPyAlgorithm)
        stub.results = MagicMock()
        stub.results.solver = MagicMock()
        stub.results.problem = MagicMock()
        # Bind the real method to the stub
        stub._mirror_direct_solve_results = (
            _MindtPyAlgorithm._mirror_direct_solve_results.__get__(stub)
        )
        return stub

    def _make_solver_results(self, term_cond, lb=None, ub=None, status=None, msg=None):
        """Build a mock SolverResults."""
        res = MagicMock()
        res.solver.termination_condition = term_cond
        res.solver.status = status if status is not None else SolverStatus.ok
        res.solver.message = msg

        prob = MagicMock()
        prob.lower_bound = lb
        prob.upper_bound = ub
        res.problem = prob
        return res

    def _make_obj(self, sense, expr_value):
        """Build a mock objective with a given sense and expression value."""
        obj = MagicMock()
        obj.sense = sense
        obj.expr = expr_value  # will be passed to value()
        return obj

    def _make_prob(self):
        """Create a plain namespace for the prob argument so we can track
        exactly which attributes were set."""
        return _SimpleNamespace(
            sense=None, lower_bound=self._SENTINEL, upper_bound=self._SENTINEL
        )

    def test_minimize_no_explicit_bounds_sets_upper_only(self):
        """Minimization + no solver bounds → upper_bound = obj_val, no lower_bound."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(tc.optimal, lb=None, ub=None)
        obj = self._make_obj(minimize, 42.0)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        self.assertEqual(prob.upper_bound, 42.0)
        self.assertEqual(prob.sense, minimize)
        # lower_bound must NOT have been set to obj_val
        self.assertIs(prob.lower_bound, self._SENTINEL)

    def test_maximize_no_explicit_bounds_sets_lower_only(self):
        """Maximization + no solver bounds → lower_bound = obj_val, no upper_bound."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(tc.optimal, lb=None, ub=None)
        obj = self._make_obj(maximize, 99.0)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        self.assertEqual(prob.lower_bound, 99.0)
        self.assertEqual(prob.sense, maximize)
        # upper_bound must NOT have been set to obj_val
        self.assertIs(prob.upper_bound, self._SENTINEL)

    def test_explicit_bounds_from_solver_are_used(self):
        """When solver provides both bounds, they should be used directly."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(tc.optimal, lb=10.0, ub=20.0)
        obj = self._make_obj(minimize, 15.0)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        self.assertEqual(prob.lower_bound, 10.0)
        self.assertEqual(prob.upper_bound, 20.0)

    def test_explicit_lb_from_solver_minimize_fallback_ub(self):
        """Solver provides lb only; minimize fallback should set ub from obj_val."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(tc.optimal, lb=5.0, ub=None)
        obj = self._make_obj(minimize, 7.0)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        self.assertEqual(prob.lower_bound, 5.0)
        self.assertEqual(prob.upper_bound, 7.0)

    def test_explicit_ub_from_solver_maximize_fallback_lb(self):
        """Solver provides ub only; maximize fallback should set lb from obj_val."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(tc.optimal, lb=None, ub=100.0)
        obj = self._make_obj(maximize, 80.0)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        self.assertEqual(prob.lower_bound, 80.0)
        self.assertEqual(prob.upper_bound, 100.0)

    def test_minimize_solver_provides_ub_no_fallback_on_ub(self):
        """Solver provides ub, lb missing; minimize fallback enters but ub is
        already set, so ub should keep the solver value (not be overwritten)."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(tc.optimal, lb=None, ub=30.0)
        obj = self._make_obj(minimize, 25.0)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        # ub was set from solver; fallback should NOT overwrite it
        self.assertEqual(prob.upper_bound, 30.0)

    def test_maximize_solver_provides_lb_no_fallback_on_lb(self):
        """Solver provides lb, ub missing; maximize fallback enters but lb is
        already set, so lb should keep the solver value."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(tc.optimal, lb=50.0, ub=None)
        obj = self._make_obj(maximize, 60.0)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        # lb was set from solver; fallback should NOT overwrite it
        self.assertEqual(prob.lower_bound, 50.0)

    def test_non_optimal_termination_skips_fallback(self):
        """Non-optimal termination should not infer any bounds from obj_val."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(tc.infeasible, lb=None, ub=None)
        obj = self._make_obj(minimize, 42.0)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        self.assertEqual(prob.sense, minimize)
        # Neither bound should be set
        self.assertIs(prob.lower_bound, self._SENTINEL)
        self.assertIs(prob.upper_bound, self._SENTINEL)

    def test_none_obj_val_skips_fallback(self):
        """If value(obj.expr) returns None, no bounds should be inferred."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(tc.optimal, lb=None, ub=None)
        obj = self._make_obj(minimize, None)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        self.assertEqual(prob.sense, minimize)
        # The fallback is entered but obj_val is None → no bound set
        self.assertIs(prob.lower_bound, self._SENTINEL)
        self.assertIs(prob.upper_bound, self._SENTINEL)

    def test_solver_status_and_message_mirrored(self):
        """Solver status, termination_condition, and message are mirrored."""
        algo = self._make_algorithm_stub()
        prob = self._make_prob()

        solver_res = self._make_solver_results(
            tc.optimal, status=SolverStatus.ok, msg="All good"
        )
        obj = self._make_obj(minimize, 1.0)

        algo._mirror_direct_solve_results(solver_res, obj, prob)

        self.assertEqual(algo.results.solver.status, SolverStatus.ok)
        self.assertEqual(algo.results.solver.termination_condition, tc.optimal)
        self.assertEqual(algo.results.solver.message, "All good")


class _FakeLegacyMIPSolver:
    def __init__(
        self,
        *,
        quadratic_objective=False,
        quadratic_constraint=False,
        termination_condition=tc.optimal,
    ):
        self.options = {}
        self._capabilities = {
            'quadratic_objective': quadratic_objective,
            'quadratic_constraint': quadratic_constraint,
        }
        self.solve = MagicMock(
            return_value=_make_direct_solve_results(termination_condition)
        )

    def has_capability(self, capability):
        return self._capabilities[capability]


class _FakeSolver:
    def __init__(self, termination_condition=tc.optimal):
        self.options = {}
        self.solve = MagicMock(
            return_value=_make_direct_solve_results(termination_condition)
        )


def _make_direct_solve_results(termination_condition):
    return _SimpleNamespace(
        solution=[],
        solver=_SimpleNamespace(
            termination_condition=termination_condition,
            status=SolverStatus.ok,
            message=None,
        ),
        problem=_SimpleNamespace(lower_bound=None, upper_bound=None),
    )


class TestMindtPyShortCircuitRouting(unittest.TestCase):
    def _make_algorithm(
        self,
        model,
        mip_solver_name,
        mip_opt,
        nlp_opt=None,
        mip_constraint_polynomial_degree=None,
        mip_objective_polynomial_degree=None,
    ):
        from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm

        algo = _MindtPyAlgorithm()
        algo.config = _SimpleNamespace(
            logger=MagicMock(),
            mip_solver=mip_solver_name,
            nlp_solver='ipopt',
            mip_solver_tee=False,
            nlp_solver_tee=False,
            mip_solver_args={},
            nlp_solver_args={},
            feasibility_norm='L1',
            add_slack=False,
            max_slack=0,
        )
        algo.mip_constraint_polynomial_degree = (
            {0, 1}
            if mip_constraint_polynomial_degree is None
            else mip_constraint_polynomial_degree
        )
        algo.mip_objective_polynomial_degree = (
            {0, 1}
            if mip_objective_polynomial_degree is None
            else mip_objective_polynomial_degree
        )
        algo.original_model = model
        algo.working_model = model.clone()
        algo.create_utility_block(algo.working_model, 'MindtPy_utils')
        algo.mip_opt = mip_opt
        algo.nlp_opt = nlp_opt if nlp_opt is not None else _FakeSolver()
        algo.mip_load_solutions = False
        algo.nlp_load_solutions = False
        algo._mirror_direct_solve_results = MagicMock()
        algo.timing = _SimpleNamespace(main_timer_start_time=0.0)
        return algo

    def _make_lp_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, None))
        m.c = Constraint(expr=m.x >= 1)
        m.obj = Objective(expr=m.x, sense=minimize)
        return m

    def _make_qp_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, None))
        m.c = Constraint(expr=m.x >= 1)
        m.obj = Objective(expr=(m.x - 2) ** 2, sense=minimize)
        return m

    def _make_qcp_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.c = Constraint(expr=m.x**2 <= 4)
        m.obj = Objective(expr=m.x, sense=maximize)
        return m

    def _make_qcp_with_quadratic_objective_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.c = Constraint(expr=m.x**2 <= 9)
        m.obj = Objective(expr=(m.x - 1) ** 2, sense=minimize)
        return m

    def _make_nlp_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.c = Constraint(expr=m.x >= 1)
        m.obj = Objective(expr=m.x**3, sense=minimize)
        return m

    def test_short_circuit_lp_routes_to_mip(self):
        algo = self._make_algorithm(
            self._make_lp_model(),
            mip_solver_name='glpk',
            mip_opt=_FakeLegacyMIPSolver(),
        )

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_called_once()
        algo.nlp_opt.solve.assert_not_called()

    def test_short_circuit_qp_uses_legacy_mip_solver_when_supported(self):
        algo = self._make_algorithm(
            self._make_qp_model(),
            mip_solver_name='gurobi',
            mip_opt=_FakeLegacyMIPSolver(quadratic_objective=True),
        )

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_called_once()
        algo.nlp_opt.solve.assert_not_called()
        self.assertIs(
            algo._mirror_direct_solve_results.call_args.kwargs['obj'],
            algo.original_model.obj,
        )

    def test_short_circuit_qp_uses_nlp_when_appsi_highs_lacks_quadratic_support(self):
        algo = self._make_algorithm(
            self._make_qp_model(), mip_solver_name='appsi_highs', mip_opt=_FakeSolver()
        )

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_not_called()
        algo.nlp_opt.solve.assert_called_once()

    def test_short_circuit_qcp_uses_appsi_cplex_when_supported(self):
        algo = self._make_algorithm(
            self._make_qcp_model(), mip_solver_name='appsi_cplex', mip_opt=_FakeSolver()
        )

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_called_once()
        algo.nlp_opt.solve.assert_not_called()

    def test_short_circuit_qcp_uses_nlp_when_legacy_mip_solver_lacks_qcp_support(self):
        algo = self._make_algorithm(
            self._make_qcp_model(),
            mip_solver_name='cbc',
            mip_opt=_FakeLegacyMIPSolver(
                quadratic_objective=False, quadratic_constraint=False
            ),
        )

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_not_called()
        algo.nlp_opt.solve.assert_called_once()

    def test_short_circuit_qcp_uses_nlp_when_solver_lacks_quadratic_objective(self):
        algo = self._make_algorithm(
            self._make_qcp_with_quadratic_objective_model(),
            mip_solver_name='cbc',
            mip_opt=_FakeLegacyMIPSolver(
                quadratic_objective=False, quadratic_constraint=True
            ),
        )

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_not_called()
        algo.nlp_opt.solve.assert_called_once()

    def test_short_circuit_nlp_uses_nlp_even_with_quadratic_capable_mip_solver(self):
        algo = self._make_algorithm(
            self._make_nlp_model(),
            mip_solver_name='gurobi',
            mip_opt=_FakeLegacyMIPSolver(
                quadratic_objective=True, quadratic_constraint=True
            ),
        )

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_not_called()
        algo.nlp_opt.solve.assert_called_once()

    def test_short_circuit_mip_failure_does_not_fallback_to_nlp(self):
        algo = self._make_algorithm(
            self._make_qp_model(),
            mip_solver_name='gurobi',
            mip_opt=_FakeLegacyMIPSolver(
                quadratic_objective=True, termination_condition=tc.error
            ),
        )

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_called_once()
        algo.nlp_opt.solve.assert_not_called()

    def test_short_circuit_qcp_detection_ignores_quadratic_strategy(self):
        algo = self._make_algorithm(
            self._make_qcp_model(),
            mip_solver_name='cbc',
            mip_opt=_FakeLegacyMIPSolver(
                quadratic_objective=False, quadratic_constraint=False
            ),
            mip_constraint_polynomial_degree={0, 1, 2},
            mip_objective_polynomial_degree={0, 1, 2},
        )

        util = algo.working_model.MindtPy_utils
        self.assertEqual(len(util.linear_constraint_list), 1)
        self.assertEqual(len(util.nonlinear_constraint_list), 0)
        self.assertTrue(util.has_quadratic_constraints)
        self.assertFalse(util.has_nonquadratic_constraints)

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_not_called()
        algo.nlp_opt.solve.assert_called_once()

    def _make_mixed_degree_model(self):
        """Model with one quadratic and one cubic constraint (mixed degree)."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.c_quad = Constraint(expr=m.x**2 <= 9)
        m.c_cubic = Constraint(expr=m.x**3 <= 27)
        m.obj = Objective(expr=m.x, sense=minimize)
        return m

    def test_short_circuit_mixed_degree_model_routes_to_nlp(self):
        """A model with both quadratic and cubic constraints must route to NLP.

        has_quadratic_constraints and has_nonquadratic_constraints are both
        True.  The NLP short-circuit path should be taken regardless of whether
        the configured MIP solver claims quadratic support, because the cubic
        constraint makes this an NLP.
        """
        algo = self._make_algorithm(
            self._make_mixed_degree_model(),
            mip_solver_name='gurobi',
            mip_opt=_FakeLegacyMIPSolver(
                quadratic_objective=True, quadratic_constraint=True
            ),
        )

        util = algo.working_model.MindtPy_utils
        self.assertTrue(util.has_quadratic_constraints)
        self.assertTrue(util.has_nonquadratic_constraints)

        with patch(
            'pyomo.contrib.mindtpy.algorithm_base_class.update_solver_timelimit'
        ):
            self.assertFalse(algo.model_is_valid())

        algo.mip_opt.solve.assert_not_called()
        algo.nlp_opt.solve.assert_called_once()

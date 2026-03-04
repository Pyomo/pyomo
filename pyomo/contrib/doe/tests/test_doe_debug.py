# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.dependencies import numpy_available, scipy_available
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.config import ConfigBlock, ConfigValue

from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.doe.doe import _SMALL_TOLERANCE_DEFINITENESS

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")


class _FakeResult:
    """Minimal fake solver result object matching attributes used by DoE."""

    class _SolverData:
        """Container for solver status fields expected by DoE result parsing."""

        status = "ok"
        termination_condition = "optimal"
        message = "fake solve"

    solver = _SolverData()


class _RecordingSolver:
    """Fake solver that records options seen at each solve call."""

    def __init__(self):
        """Initialize mutable options and per-call option snapshots."""
        self.options = {}
        self.solve_call_options = []

    def solve(self, model, tee=False):
        """Record current options and return a fake optimal solve result."""
        self.solve_call_options.append(dict(self.options))
        return _FakeResult()


class _MutatingRecordingSolver(_RecordingSolver):
    """
    Fake solver that perturbs FIM during dummy-objective initialization solve.

    This emulates a realistic DoE square-initialization behavior where the
    solve updates FIM values. It is used to verify that post-solve
    re-synchronization of ``L``, ``L_inv``, ``fim_inv``, and ``cov_trace``
    is performed correctly.
    """

    def solve(self, model, tee=False):
        """Record options, then mutate 2x2 FIM values during init-stage solve."""
        super().solve(model, tee=tee)
        if (
            hasattr(model, "dummy_obj")
            and model.dummy_obj.active
            and hasattr(model, "fim")
            and len(list(model.parameter_names)) == 2
        ):
            p1, p2 = list(model.parameter_names)
            model.fim[p1, p1].set_value(16.0)
            model.fim[p2, p1].set_value(4.0)
            if (p1, p2) in model.fim:
                model.fim[p1, p2].set_value(0.0)
            model.fim[p2, p2].set_value(9.0)
        return _FakeResult()


class _SimpleExperiment:
    """
    Minimal one-parameter experiment fixture for DoE unit tests.

    Mathematical model
    ------------------
    The fixture builds a tiny algebraic model with one input, one unknown
    parameter, and one measured output:

        y = theta + x

    where:
    - ``x`` is labeled as an experimental input,
    - ``theta`` is labeled as an unknown parameter,
    - ``y`` is labeled as an experimental output with unit measurement error.

    Why this model is used
    ----------------------
    This model is intentionally simple so tests can isolate DoE framework
    behavior (solver option routing, inspection output schema, and result
    bookkeeping) without confounding numerical complexity from the underlying
    process model.

    What this fixture tests
    -----------------------
    This fixture is used by tests that check:
    - default `run_doe()` behavior remains stable,
    - split scenario/final solver option handling, and
    - inspection report schema generation.

    Why this does not inherit from Parmest ``Experiment``
    ------------------------------------------------------
    ``DesignOfExperiments`` only requires an object exposing
    ``get_labeled_model()`` that returns a model with expected suffix labels.
    For unit testing, a lightweight class keeps dependencies minimal and avoids
    unrelated Parmest behaviors.
    """

    def get_labeled_model(self):
        """Return a labeled Pyomo model for the test fixture."""
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.theta = pyo.Var(initialize=2.0)
        m.y = pyo.Var(initialize=3.0)
        m.eq = pyo.Constraint(expr=m.y == m.theta + m.x)

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.x] = 1.0
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters[m.theta] = 2.0
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[m.y] = 3.0
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[m.y] = 1.0
        return m


class _TwoParamExperiment:
    """
    Minimal two-parameter experiment fixture for trace/Cholesky tests.

    Mathematical model
    ------------------
    The fixture builds a separable two-output model:

        y1 = theta1 * x1
        y2 = theta2 * x2

    with two experimental inputs ``(x1, x2)``, two unknown parameters
    ``(theta1, theta2)``, and two outputs ``(y1, y2)`` (unit measurement
    errors).

    Why this model is used
    ----------------------
    Two parameters are the smallest case that exercises matrix-valued
    quantities central to trace/A-opt with Cholesky structures
    (``fim``, ``L``, ``L_inv``, ``fim_inv``, ``cov_trace``). This fixture is
    used to test post-initialization consistency constraints for those
    variables.

    What this fixture tests
    -----------------------
    This fixture supports trace/A-opt initialization tests, specifically that
    Cholesky- and inverse-related consistency constraints are satisfied after
    initialization when FIM values have changed.

    Why this does not inherit from Parmest ``Experiment``
    ------------------------------------------------------
    As with ``_SimpleExperiment``, inheriting Parmest ``Experiment`` is not
    needed for these unit tests. The DoE interface contract is satisfied by
    providing ``get_labeled_model()`` plus the required suffix labels.
    """

    def get_labeled_model(self):
        """Return a labeled two-parameter Pyomo model for testing."""
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var(initialize=1.0)
        m.x2 = pyo.Var(initialize=1.0)
        m.theta1 = pyo.Var(initialize=2.0)
        m.theta2 = pyo.Var(initialize=3.0)
        m.y1 = pyo.Var(initialize=2.0)
        m.y2 = pyo.Var(initialize=3.0)
        m.eq1 = pyo.Constraint(expr=m.y1 == m.theta1 * m.x1)
        m.eq2 = pyo.Constraint(expr=m.y2 == m.theta2 * m.x2)

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.x1] = 1.0
        m.experiment_inputs[m.x2] = 1.0
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters[m.theta1] = 2.0
        m.unknown_parameters[m.theta2] = 3.0
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[m.y1] = 2.0
        m.experiment_outputs[m.y2] = 3.0
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[m.y1] = 1.0
        m.measurement_error[m.y2] = 1.0
        return m


def _make_doe_object():
    """Build a DoE object with the one-parameter fixture and recording solver."""
    solver = _RecordingSolver()
    solver.options["max_iter"] = 3000
    return DesignOfExperiments(
        experiment=_SimpleExperiment(),
        fd_formula="central",
        step=1e-3,
        objective_option="zero",
        solver=solver,
        tee=False,
    )


def _make_trace_doe_object():
    """Build a trace-objective DoE object with FIM-mutating fake solver."""
    solver = _MutatingRecordingSolver()
    solver.options["max_iter"] = 3000
    return DesignOfExperiments(
        experiment=_TwoParamExperiment(),
        fd_formula="central",
        step=1e-3,
        objective_option="trace",
        solver=solver,
        tee=False,
    )


class TestDoeDebugOptions(unittest.TestCase):
    """Tests for advanced run_doe configuration and debug-facing behavior."""

    def test_run_doe_default_behavior_unchanged(self):
        """Verify default run_doe path still performs normal solve workflow."""
        doe_obj = _make_doe_object()
        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")
        self.assertEqual(doe_obj.results["Termination Condition"], "optimal")
        self.assertNotIn("Constraint Residuals", doe_obj.results)
        self.assertEqual(
            [c["max_iter"] for c in doe_obj.solver.solve_call_options],
            [3000, 3000, 3000, 3000, 3000],
        )

    def test_split_scenario_and_final_solver_options(self):
        """Verify scenario and final solves receive distinct solver options."""
        doe_obj = _make_doe_object()
        doe_obj.run_doe(
            run_config={
                "scenario_solver_options": {"max_iter": 25},
                "final_solver_options": {"max_iter": 0},
            },
        )

        self.assertEqual(
            [c["max_iter"] for c in doe_obj.solver.solve_call_options[:4]],
            [25, 25, 25, 25],
        )
        self.assertEqual(doe_obj.solver.solve_call_options[4]["max_iter"], 0)
        self.assertEqual(doe_obj.solver.options["max_iter"], 3000)

    def test_inspection_mode_generates_expected_schema(self):
        """Verify inspection mode returns residual entries with expected keys."""
        doe_obj = _make_doe_object()
        doe_obj.run_doe(
            run_config={
                "final_solve": False,
                "inspection": {"enabled": True, "top_constraints": 3},
            },
        )

        self.assertEqual(doe_obj.results["Solver Status"], "not_run")
        self.assertIn("Constraint Residuals", doe_obj.results)
        self.assertIn("post_initialization", doe_obj.results["Constraint Residuals"])
        self.assertIn("post_final_stage", doe_obj.results["Constraint Residuals"])
        residuals = doe_obj.results["Constraint Residuals"]["post_initialization"]
        self.assertLessEqual(len(residuals), 3)
        self.assertGreater(len(residuals), 0)
        self.assertEqual(
            set(residuals[0].keys()),
            {
                "constraint_name",
                "body",
                "lower_bound",
                "upper_bound",
                "violation",
                "constraint_type",
            },
        )

    def test_max_iter_zero_final_does_not_block_scenario_solves(self):
        """Verify final max_iter=0 does not affect prerequisite scenario solves."""
        doe_obj = _make_doe_object()
        doe_obj.run_doe(
            run_config={
                "scenario_solver_options": {"max_iter": 100},
                "final_solver_options": {"max_iter": 0},
            },
        )

        self.assertEqual(doe_obj.solver.solve_call_options[0]["max_iter"], 100)
        self.assertEqual(doe_obj.solver.solve_call_options[-1]["max_iter"], 0)


class TestConstraintResidualUtility(unittest.TestCase):
    """Unit tests for structured constraint residual reporting helper."""

    def test_residual_report_orders_by_violation(self):
        """Residual report should be sorted descending by violation magnitude."""
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0.0)
        m.c1 = pyo.Constraint(expr=m.x >= 1.0)
        m.c2 = pyo.Constraint(expr=m.x <= 10.0)
        m.c3 = pyo.Constraint(expr=m.x == 0.5)

        doe_obj = _make_doe_object()
        residuals = doe_obj.get_constraint_residuals(model=m, top_n=2)

        self.assertEqual(len(residuals), 2)
        self.assertEqual(residuals[0]["constraint_name"], "c1")
        self.assertEqual(residuals[0]["constraint_type"], "inequality")
        self.assertEqual(residuals[1]["constraint_type"], "equation")


class TestCholeskyInitialization(unittest.TestCase):
    """Tests for Cholesky/FIM initialization helper behavior."""

    def test_compute_cholesky_jitter_raises_negative_eigenvalue(self):
        """Negative minimum eigenvalue should produce positive corrective jitter."""
        doe_obj = _make_doe_object()
        min_eig = -1.0e-3
        jitter = doe_obj._compute_cholesky_jitter(min_eig)
        self.assertAlmostEqual(
            jitter, _SMALL_TOLERANCE_DEFINITENESS - min_eig, places=14
        )

    def test_compute_cholesky_jitter_zero_when_not_needed(self):
        """Positive minimum eigenvalue above tolerance should yield zero jitter."""
        doe_obj = _make_doe_object()
        min_eig = 1.0e-2
        jitter = doe_obj._compute_cholesky_jitter(min_eig)
        self.assertEqual(jitter, 0.0)

    def test_trace_initialization_resynchronizes_fim_inverse_variables(self):
        """
        Verify trace-mode initialization re-synchronizes inverse-related variables.

        The fake solver mutates FIM during initialization solve. The test then
        checks that post-initialization residuals for:
        - ``obj_cons.cholesky_inv_cons``,
        - ``obj_cons.cholesky_LLinv_cons``, and
        - ``obj_cons.cov_trace_rule``
        are all near zero.
        """
        doe_obj = _make_trace_doe_object()
        run_config = ConfigBlock()
        run_config.declare("final_solve", ConfigValue(default=False))
        run_config.declare("inspection", ConfigBlock())
        run_config.inspection.declare("enabled", ConfigValue(default=True))
        run_config.inspection.declare("top_constraints", ConfigValue(default=200))
        doe_obj.run_doe(
            run_config=run_config,
        )
        residuals = doe_obj.results["Constraint Residuals"]["post_initialization"]
        by_name = {entry["constraint_name"]: entry["violation"] for entry in residuals}

        cholesky_inv = [
            violation
            for name, violation in by_name.items()
            if name.startswith("obj_cons.cholesky_inv_cons[")
        ]
        llinv = [
            violation
            for name, violation in by_name.items()
            if name.startswith("obj_cons.cholesky_LLinv_cons[")
        ]
        self.assertGreater(len(cholesky_inv), 0)
        self.assertGreater(len(llinv), 0)
        self.assertLess(max(cholesky_inv), 1e-8)
        self.assertLess(max(llinv), 1e-8)
        self.assertLess(by_name["obj_cons.cov_trace_rule"], 1e-8)

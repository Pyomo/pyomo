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
    class _SolverData:
        status = "ok"
        termination_condition = "optimal"
        message = "fake solve"

    solver = _SolverData()


class _RecordingSolver:
    def __init__(self):
        self.options = {}
        self.solve_call_options = []

    def solve(self, model, tee=False):
        self.solve_call_options.append(dict(self.options))
        return _FakeResult()


class _MutatingRecordingSolver(_RecordingSolver):
    def solve(self, model, tee=False):
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
    def get_labeled_model(self):
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
    def get_labeled_model(self):
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
    def test_run_doe_default_behavior_unchanged(self):
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
    def test_residual_report_orders_by_violation(self):
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
    def test_compute_cholesky_jitter_raises_negative_eigenvalue(self):
        doe_obj = _make_doe_object()
        min_eig = -1.0e-3
        jitter = doe_obj._compute_cholesky_jitter(min_eig)
        self.assertAlmostEqual(
            jitter, _SMALL_TOLERANCE_DEFINITENESS - min_eig, places=14
        )

    def test_compute_cholesky_jitter_zero_when_not_needed(self):
        doe_obj = _make_doe_object()
        min_eig = 1.0e-2
        jitter = doe_obj._compute_cholesky_jitter(min_eig)
        self.assertEqual(jitter, 0.0)

    def test_trace_initialization_resynchronizes_fim_inverse_variables(self):
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

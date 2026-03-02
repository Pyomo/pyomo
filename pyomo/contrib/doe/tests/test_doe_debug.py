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

from pyomo.contrib.doe import DesignOfExperiments

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
            scenario_solver_options={"max_iter": 25},
            final_solver_options={"max_iter": 0},
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
            solve_final_model=False,
            inspect_constraints=True,
            inspect_top_constraints=3,
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
            scenario_solver_options={"max_iter": 100},
            final_solver_options={"max_iter": 0},
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

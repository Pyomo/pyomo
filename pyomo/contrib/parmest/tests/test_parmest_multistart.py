#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2026 National Technology and Engineering Solutions of
#  Sandia, LLC Under the terms of Contract DE-NA0003525 with National
#  Technology and Engineering Solutions of Sandia, LLC, the U.S. Government
#  retains certain rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import numpy as np, pandas as pd
from unittest.mock import patch

import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment

ipopt_available = pyo.SolverFactory("ipopt").available()


class LinearThetaExperiment(Experiment):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()
        m.theta = pyo.Var(initialize=0.0, bounds=(-10.0, 10.0))
        m.x = pyo.Param(initialize=float(self.x_data), mutable=False)
        m.y = pyo.Var(initialize=float(self.y_data))
        m.eq = pyo.Constraint(expr=m.y == m.theta + m.x)
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, float(self.y_data))])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


class IndexedThetaExperiment(Experiment):
    def __init__(self):
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=["a", "b"])
        m.theta = pyo.Var(m.I, initialize={"a": 1.0, "b": 2.0})
        m.theta["a"].setlb(0.0)
        m.theta["a"].setub(5.0)
        m.theta["b"].setlb(0.0)
        m.theta["b"].setub(5.0)
        m.theta["a"].fix()
        m.theta["b"].fix()
        m.y = pyo.Var(initialize=3.0)
        m.eq = pyo.Constraint(expr=m.y == m.theta["a"] + m.theta["b"])
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, 3.0)])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


class NoBoundsExperiment(Experiment):
    def __init__(self):
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()
        m.theta = pyo.Var(initialize=1.0)
        m.y = pyo.Var(initialize=2.0)
        m.eq = pyo.Constraint(expr=m.y == m.theta + 1.0)
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, 2.0)])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


def _build_linear_estimator():
    exp_list = [LinearThetaExperiment(1.0, 2.0), LinearThetaExperiment(2.0, 3.0)]
    return parmest.Estimator(exp_list, obj_function="SSE")


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
class TestParmestMultistart(unittest.TestCase):
    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_multistart_baseline_equivalence_n1(self):
        pest = _build_linear_estimator()
        obj1, theta1 = pest.theta_est()
        _, best_theta, best_obj = pest.theta_est_multistart(
            n_restarts=1, multistart_sampling_method="uniform_random", seed=7
        )
        self.assertAlmostEqual(obj1, best_obj, places=7)
        self.assertAlmostEqual(theta1["theta"], best_theta["theta"], places=7)

    def test_uniform_sampling_is_deterministic_with_seed(self):
        pest = _build_linear_estimator()
        model = pest._create_parmest_model(0)
        df1 = pest._generate_initial_theta(
            parmest_model=model,
            seed=4,
            n_restarts=5,
            multistart_sampling_method="uniform_random",
        )
        df2 = pest._generate_initial_theta(
            parmest_model=model,
            seed=4,
            n_restarts=5,
            multistart_sampling_method="uniform_random",
        )
        self.assertTrue(df1[["theta"]].equals(df2[["theta"]]))

    def test_uniform_sampling_changes_with_different_seed(self):
        pest = _build_linear_estimator()
        model = pest._create_parmest_model(0)
        df1 = pest._generate_initial_theta(
            parmest_model=model,
            seed=4,
            n_restarts=5,
            multistart_sampling_method="uniform_random",
        )
        df2 = pest._generate_initial_theta(
            parmest_model=model,
            seed=5,
            n_restarts=5,
            multistart_sampling_method="uniform_random",
        )
        self.assertFalse(df1[["theta"]].equals(df2[["theta"]]))

    def test_latin_hypercube_sampling_is_deterministic(self):
        pest = _build_linear_estimator()
        model = pest._create_parmest_model(0)
        df1 = pest._generate_initial_theta(
            parmest_model=model,
            seed=11,
            n_restarts=4,
            multistart_sampling_method="latin_hypercube",
        )
        df2 = pest._generate_initial_theta(
            parmest_model=model,
            seed=11,
            n_restarts=4,
            multistart_sampling_method="latin_hypercube",
        )
        self.assertTrue(df1[["theta"]].equals(df2[["theta"]]))

    def test_sobol_sampling_is_deterministic(self):
        pest = _build_linear_estimator()
        model = pest._create_parmest_model(0)
        df1 = pest._generate_initial_theta(
            parmest_model=model,
            seed=12,
            n_restarts=4,
            multistart_sampling_method="sobol_sampling",
        )
        df2 = pest._generate_initial_theta(
            parmest_model=model,
            seed=12,
            n_restarts=4,
            multistart_sampling_method="sobol_sampling",
        )
        self.assertTrue(df1[["theta"]].equals(df2[["theta"]]))

    def test_generated_starts_are_within_bounds(self):
        pest = _build_linear_estimator()
        model = pest._create_parmest_model(0)
        for method in ("uniform_random", "latin_hypercube", "sobol_sampling"):
            df = pest._generate_initial_theta(
                parmest_model=model,
                seed=1,
                n_restarts=8,
                multistart_sampling_method=method,
            )
            self.assertTrue(((df["theta"] >= -10.0) & (df["theta"] <= 10.0)).all())

    def test_missing_bounds_raise_error(self):
        pest = parmest.Estimator([NoBoundsExperiment()], obj_function="SSE")
        model = pest._create_parmest_model(0)
        with self.assertRaisesRegex(
            ValueError, "lower and upper bounds for the theta values must be defined"
        ):
            pest._generate_initial_theta(
                parmest_model=model,
                seed=1,
                n_restarts=2,
                multistart_sampling_method="uniform_random",
            )

    def test_invalid_bounds_raise_error(self):
        pest = _build_linear_estimator()
        model = pest._create_parmest_model(0)
        model.theta.setlb(2.0)
        model.theta.setub(1.0)
        with self.assertRaisesRegex(ValueError, "lower bound must be less than"):
            pest._generate_initial_theta(
                parmest_model=model,
                seed=1,
                n_restarts=2,
                multistart_sampling_method="uniform_random",
            )

    def test_user_provided_values_dimension_mismatch_raises(self):
        pest = _build_linear_estimator()
        user_df = pd.DataFrame([[1.0, 2.0]], columns=["theta", "extra"])
        with self.assertRaisesRegex(
            ValueError, "same number of columns as the number of theta names"
        ):
            pest.theta_est_multistart(
                n_restarts=1,
                multistart_sampling_method="user_provided_values",
                user_provided_df=user_df,
            )

    def test_user_provided_values_column_order_maps_by_name(self):
        pest = parmest.Estimator([IndexedThetaExperiment()], obj_function="SSE")
        user_df = pd.DataFrame(
            [[0.3, 4.2], [0.4, 4.1]], columns=["theta[b]", "theta[a]"]
        )
        results_df, _, _ = pest.theta_est_multistart(
            n_restarts=2,
            multistart_sampling_method="user_provided_values",
            user_provided_df=user_df,
        )
        self.assertAlmostEqual(results_df.loc[0, "theta[a]"], 4.2, places=12)
        self.assertAlmostEqual(results_df.loc[0, "theta[b]"], 0.3, places=12)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_state_isolation_between_starts(self):
        pest = _build_linear_estimator()
        init = pd.DataFrame([[-9.0], [9.0]], columns=["theta"])
        results_df, _, _ = pest.theta_est_multistart(
            theta_values=init, save_results=False
        )
        # Initial starts should remain exactly as supplied.
        self.assertAlmostEqual(results_df.loc[0, "theta"], -9.0, places=12)
        self.assertAlmostEqual(results_df.loc[1, "theta"], 9.0, places=12)
        # Both runs converge to the same optimum, showing no cross-start contamination.
        self.assertAlmostEqual(
            results_df.loc[0, "converged_theta"],
            results_df.loc[1, "converged_theta"],
            places=8,
        )

    def test_one_start_failure_returns_best_feasible(self):
        pest = _build_linear_estimator()
        theta_values = pd.DataFrame([[-1.0], [2.0]], columns=["theta"])

        def fake_q_opt(*args, **kwargs):
            theta = kwargs["theta_vals"]["theta"]
            if theta < 0:
                raise RuntimeError("boom")
            return 1.25, {"theta": 1.0}, pyo.TerminationCondition.optimal

        with patch.object(pest, "_Q_opt", side_effect=fake_q_opt):
            results_df, best_theta, best_obj = pest.theta_est_multistart(
                theta_values=theta_values, save_results=False
            )

        self.assertTrue(
            str(results_df.loc[0, "solver termination"]).startswith("exception(start=0")
        )
        self.assertAlmostEqual(best_obj, 1.25, places=12)
        self.assertAlmostEqual(best_theta["theta"], 1.0, places=12)

    def test_all_starts_fail_returns_diagnostics(self):
        pest = _build_linear_estimator()
        theta_values = pd.DataFrame([[1.0], [2.0]], columns=["theta"])

        def fake_q_opt(*args, **kwargs):
            raise RuntimeError("all failed")

        with patch.object(pest, "_Q_opt", side_effect=fake_q_opt):
            results_df, best_theta, best_obj = pest.theta_est_multistart(
                theta_values=theta_values, save_results=False
            )

        self.assertIsNone(best_theta)
        self.assertTrue(math.isnan(best_obj))
        self.assertTrue(
            results_df["solver termination"]
            .astype(str)
            .str.contains("exception\\(start=", regex=True)
            .all()
        )

    def test_best_selection_filters_nonoptimal_status(self):
        pest = _build_linear_estimator()
        theta_values = pd.DataFrame([[1.0], [2.0]], columns=["theta"])

        def fake_q_opt(*args, **kwargs):
            theta = kwargs["theta_vals"]["theta"]
            if theta < 1.5:
                return 0.1, {"theta": 0.1}, pyo.TerminationCondition.maxIterations
            return 0.2, {"theta": 0.2}, pyo.TerminationCondition.optimal

        with patch.object(pest, "_Q_opt", side_effect=fake_q_opt):
            _, best_theta, best_obj = pest.theta_est_multistart(
                theta_values=theta_values, save_results=False
            )

        self.assertAlmostEqual(best_obj, 0.2, places=12)
        self.assertAlmostEqual(best_theta["theta"], 0.2, places=12)

    def test_tie_breaking_is_deterministic_first_index(self):
        pest = _build_linear_estimator()
        theta_values = pd.DataFrame([[5.0], [6.0], [7.0]], columns=["theta"])

        def fake_q_opt(*args, **kwargs):
            theta = kwargs["theta_vals"]["theta"]
            return 1.0, {"theta": theta}, pyo.TerminationCondition.optimal

        with patch.object(pest, "_Q_opt", side_effect=fake_q_opt):
            _, best_theta, best_obj = pest.theta_est_multistart(
                theta_values=theta_values, save_results=False
            )

        self.assertAlmostEqual(best_obj, 1.0, places=12)
        self.assertAlmostEqual(best_theta["theta"], 5.0, places=12)

    def test_indexed_unknown_parameters_supported_in_sampling(self):
        pest = parmest.Estimator([IndexedThetaExperiment()], obj_function="SSE")
        model = pest._create_parmest_model(0)
        df = pest._generate_initial_theta(
            parmest_model=model,
            seed=10,
            n_restarts=3,
            multistart_sampling_method="uniform_random",
        )
        self.assertTrue({"theta[a]", "theta[b]"}.issubset(set(df.columns)))

    def test_count_total_experiments_uses_one_output_family(self):
        class MultiOutputExperiment(Experiment):
            def create_model(self):
                m = pyo.ConcreteModel()
                m.theta = pyo.Var(initialize=0.0, bounds=(-10, 10))
                m.y = pyo.Var(initialize=1.0)
                m.z = pyo.Var(initialize=2.0)
                m.c1 = pyo.Constraint(expr=m.y == m.theta + 1.0)
                m.c2 = pyo.Constraint(expr=m.z == 2.0 * m.theta + 2.0)
                self.model = m

            def label_model(self):
                m = self.model
                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update([(m.y, 1.0), (m.z, 2.0)])
                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update([(m.y, None), (m.z, None)])

            def get_labeled_model(self):
                self.create_model()
                self.label_model()
                return self.model

        total_points = parmest._count_total_experiments(
            [MultiOutputExperiment(), MultiOutputExperiment()]
        )
        self.assertEqual(total_points, 2)

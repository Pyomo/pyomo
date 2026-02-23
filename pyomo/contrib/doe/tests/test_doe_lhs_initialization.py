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
"""
Tests for the LHS and user-provided initialization features of
``optimize_experiments`` in ``DesignOfExperiments``.

The Rooney-Biegler model (one experiment input: ``hour`` in [0, 10]) is used
as the test vehicle because it is small and already used by other DoE tests.
"""

import warnings

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy_available,
)
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.opt import SolverFactory

# Skip entire module if required packages are missing
if not (numpy_available and scipy_available and pandas_available):
    raise unittest.SkipTest("Pyomo.DoE LHS tests require numpy, scipy, and pandas.")

from pyomo.contrib.doe import DesignOfExperiments

ipopt_available = SolverFactory("ipopt").available()


# ---------------------------------------------------------------------------
# Helper: Rooney-Biegler experiment prepared for multi-experiment DoE
# ---------------------------------------------------------------------------


def _rooney_biegler_model(data, theta=None):
    """Create a concrete Rooney-Biegler model."""
    model = pyo.ConcreteModel()

    if theta is None:
        theta = {'asymptote': 15, 'rate_constant': 0.5}

    model.asymptote = pyo.Var(initialize=theta['asymptote'])
    model.rate_constant = pyo.Var(initialize=theta['rate_constant'])
    model.asymptote.fix()
    model.rate_constant.fix()

    # Experiment input: hour in [1, 10] — lower bound > 0 avoids the
    # singularity where y = asymptote*(1-exp(0)) = 0, which conflicts
    # with the PositiveReals domain constraint on y.
    model.hour = pyo.Var(initialize=max(data['hour'], 1.0), bounds=(1.0, 10.0))
    model.hour.fix()

    model.y = pyo.Var(within=pyo.PositiveReals, initialize=data['y'])

    def response_rule(m):
        return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.hour))

    model.response_function = pyo.Constraint(rule=response_rule)
    return model


class RooneyBieglerMultiExp(Experiment):
    """Rooney-Biegler experiment with sym_break_cons for multi-experiment DoE."""

    def __init__(self, hour=2.0, y=10.0, theta=None, measure_error=0.1):
        self.hour = hour
        self.y = y
        self.theta = theta if theta is not None else {'asymptote': 15, 'rate_constant': 0.5}
        self.measure_error = measure_error

    def get_labeled_model(self):
        """Always return a fresh model so the DoE object can clone it."""
        data = {'hour': self.hour, 'y': self.y}
        m = _rooney_biegler_model(data, theta=self.theta)

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[m.y] = self.y

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
        )

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[m.y] = self.measure_error

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.hour] = self.hour

        # Symmetry-breaking variable for multi-experiment (ordering constraint)
        m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.sym_break_cons[m.hour] = None

        return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment_list(n_exp=2, hours=None):
    """Create *n_exp* identical Rooney-Biegler experiments."""
    if hours is None:
        hours = [2.0] * n_exp
    return [RooneyBieglerMultiExp(hour=h) for h in hours]


def _make_doe(experiment_list, objective_option='determinant', **kwargs):
    """Build a DoE object without calling any solver."""
    return DesignOfExperiments(
        experiment_list=experiment_list,
        objective_option=objective_option,
        step=0.01,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests: parameter validation
# ---------------------------------------------------------------------------


class TestInitializationParameterValidation(unittest.TestCase):
    """Test that invalid arguments raise appropriate errors."""

    def test_invalid_initialization_method_string(self):
        """Unknown initialization_method string must raise ValueError."""
        doe = _make_doe(_make_experiment_list(n_exp=1))
        with self.assertRaises(ValueError, msg="Should raise for unknown method string"):
            doe.optimize_experiments(initialization_method="bad_method")

    def test_lhs_invalid_n_samples_zero(self):
        """lhs_n_samples=0 must raise ValueError."""
        doe = _make_doe(_make_experiment_list(n_exp=1))
        with self.assertRaises(ValueError):
            doe.optimize_experiments(
                initialization_method="lhs",
                lhs_n_samples=0,
            )

    def test_lhs_invalid_n_samples_negative(self):
        """lhs_n_samples=-1 must raise ValueError."""
        doe = _make_doe(_make_experiment_list(n_exp=1))
        with self.assertRaises(ValueError):
            doe.optimize_experiments(
                initialization_method="lhs",
                lhs_n_samples=-1,
            )

    def test_lhs_invalid_n_samples_float(self):
        """lhs_n_samples=2.5 (float) must raise ValueError."""
        doe = _make_doe(_make_experiment_list(n_exp=1))
        with self.assertRaises(ValueError):
            doe.optimize_experiments(
                initialization_method="lhs",
                lhs_n_samples=2.5,
            )

    def test_n_exp_with_multi_experiment_list_raises(self):
        """Passing n_exp when experiment_list has >1 items must raise ValueError."""
        doe = _make_doe(_make_experiment_list(n_exp=2))
        with self.assertRaises(ValueError, msg="Should reject n_exp for multi-item list"):
            doe.optimize_experiments(n_exp=2)

    def test_n_exp_zero_raises(self):
        """n_exp=0 must raise ValueError."""
        doe = _make_doe(_make_experiment_list(n_exp=1))
        with self.assertRaises(ValueError):
            doe.optimize_experiments(n_exp=0)

    def test_n_exp_negative_raises(self):
        """n_exp=-1 must raise ValueError."""
        doe = _make_doe(_make_experiment_list(n_exp=1))
        with self.assertRaises(ValueError):
            doe.optimize_experiments(n_exp=-1)

    def test_n_exp_float_raises(self):
        """n_exp=1.5 (float) must raise ValueError."""
        doe = _make_doe(_make_experiment_list(n_exp=1))
        with self.assertRaises(ValueError):
            doe.optimize_experiments(n_exp=1.5)


# ---------------------------------------------------------------------------
# Tests: helper methods (unit tests, no solver needed)
# ---------------------------------------------------------------------------


class TestEvaluateObjectiveFromFIM(unittest.TestCase):
    """Unit tests for _evaluate_objective_from_fim."""

    def _make_doe_with_prior(self, objective_option, prior=None):
        doe = _make_doe(
            _make_experiment_list(n_exp=1), objective_option=objective_option
        )
        doe.prior_FIM = prior if prior is not None else np.zeros((2, 2))
        return doe

    def test_determinant_positive_definite(self):
        doe = self._make_doe_with_prior('determinant')
        fim = np.array([[4.0, 1.0], [1.0, 3.0]])
        val = doe._evaluate_objective_from_fim(fim)
        self.assertAlmostEqual(val, np.linalg.det(fim), places=10)

    def test_pseudo_trace(self):
        doe = self._make_doe_with_prior('pseudo_trace')
        fim = np.array([[4.0, 1.0], [1.0, 3.0]])
        val = doe._evaluate_objective_from_fim(fim)
        self.assertAlmostEqual(val, np.trace(fim), places=10)

    def test_trace_aopt(self):
        doe = self._make_doe_with_prior('trace')
        fim = np.array([[4.0, 1.0], [1.0, 3.0]])
        val = doe._evaluate_objective_from_fim(fim)
        self.assertAlmostEqual(val, np.trace(np.linalg.inv(fim)), places=10)

    def test_singular_fim_returns_fallback_for_trace(self):
        """Singular FIM (not invertible) should return +inf for trace (minimize)."""
        doe = self._make_doe_with_prior('trace')
        singular_fim = np.zeros((2, 2))
        val = doe._evaluate_objective_from_fim(singular_fim)
        self.assertEqual(val, np.inf)

    def test_singular_fim_returns_fallback_for_determinant(self):
        """Singular FIM gives det=0; that value is returned (not -inf)."""
        doe = self._make_doe_with_prior('determinant')
        singular_fim = np.zeros((2, 2))
        val = doe._evaluate_objective_from_fim(singular_fim)
        self.assertAlmostEqual(val, 0.0, places=10)


# ---------------------------------------------------------------------------
# Tests: full optimize_experiments integrations  (require IPOPT)
# ---------------------------------------------------------------------------


@unittest.skipUnless(ipopt_available, "IPOPT not available")
class TestLHSInitialization(unittest.TestCase):
    """Test optimization with LHS-based initialization."""

    def test_lhs_single_experiment_determinant(self):
        """LHS init, 1 experiment, D-optimality: should run without errors."""
        exp_list = _make_experiment_list(n_exp=1, hours=[2.0])
        doe = _make_doe(exp_list, objective_option='determinant')
        doe.optimize_experiments(
            initialization_method="lhs",
            lhs_n_samples=3,
            lhs_seed=42,
        )
        self.assertEqual(doe.results["Initialization Method"], "lhs")
        self.assertEqual(doe.results["LHS Samples Per Dimension"], 3)
        self.assertEqual(doe.results["LHS Seed"], 42)
        self.assertIn("LHS Best Initial Points", doe.results)
        # One initial point per experiment
        self.assertEqual(len(doe.results["LHS Best Initial Points"]), 1)

    def test_lhs_two_experiments_determinant(self):
        """LHS init, 2 experiments, D-optimality: results structure is correct."""
        exp_list = _make_experiment_list(n_exp=2, hours=[1.0, 3.0])
        doe = _make_doe(exp_list, objective_option='determinant')
        doe.optimize_experiments(
            initialization_method="lhs",
            lhs_n_samples=3,
            lhs_seed=0,
        )
        self.assertEqual(doe.results["Number of Experiments per Scenario"], 2)
        self.assertEqual(len(doe.results["LHS Best Initial Points"]), 2)
        # Each initial point has one value (hour)
        for pt in doe.results["LHS Best Initial Points"]:
            self.assertEqual(len(pt), 1)

    def test_lhs_two_experiments_pseudo_trace(self):
        """LHS init, 2 experiments, pseudo-trace objective."""
        exp_list = _make_experiment_list(n_exp=2, hours=[2.0, 4.0])
        doe = _make_doe(exp_list, objective_option='pseudo_trace')
        doe.optimize_experiments(
            initialization_method="lhs",
            lhs_n_samples=3,
            lhs_seed=123,
        )
        self.assertEqual(doe.results["Initialization Method"], "lhs")
        fim = np.array(doe.results['Scenarios'][0]['Total FIM'])
        self.assertEqual(fim.shape, (2, 2))

    def test_lhs_two_experiments_trace(self):
        """LHS init, 2 experiments, A-optimality (minimize trace of inv FIM)."""
        exp_list = _make_experiment_list(n_exp=2, hours=[2.0, 4.0])
        doe = _make_doe(exp_list, objective_option='trace')
        doe.optimize_experiments(
            initialization_method="lhs",
            lhs_n_samples=3,
            lhs_seed=7,
        )
        self.assertEqual(doe.results["Initialization Method"], "lhs")

    def test_lhs_seed_reproducibility(self):
        """Same seed must produce the same LHS best initial points."""
        def run(seed):
            exp_list = _make_experiment_list(n_exp=1, hours=[2.0])
            doe = _make_doe(exp_list, objective_option='determinant')
            doe.optimize_experiments(
                initialization_method="lhs",
                lhs_n_samples=3,
                lhs_seed=seed,
            )
            return doe.results["LHS Best Initial Points"]

        pts_a = run(seed=42)
        pts_b = run(seed=42)
        self.assertEqual(pts_a, pts_b)

    def test_lhs_best_points_within_bounds(self):
        """LHS best initial points must lie within the variable bounds [0, 10]."""
        exp_list = _make_experiment_list(n_exp=2, hours=[1.0, 5.0])
        doe = _make_doe(exp_list, objective_option='determinant')
        doe.optimize_experiments(
            initialization_method="lhs",
            lhs_n_samples=5,
            lhs_seed=0,
        )
        for pt in doe.results["LHS Best Initial Points"]:
            for val in pt:
                self.assertGreaterEqual(val, 1.0)
                self.assertLessEqual(val, 10.0)

    def test_lhs_none_initialization_still_works(self):
        """initialization_method=None (default) must behave like the original."""
        exp_list = _make_experiment_list(n_exp=1, hours=[5.0])
        doe = _make_doe(exp_list, objective_option='determinant')
        doe.optimize_experiments(initialization_method=None)
        self.assertEqual(doe.results["Initialization Method"], "none")
        self.assertNotIn("LHS Best Initial Points", doe.results)

    def test_template_mode_n_exp(self):
        """Single template experiment + n_exp=2 should clone the template and optimize."""
        exp_list = _make_experiment_list(n_exp=1, hours=[2.0])
        doe = _make_doe(exp_list, objective_option='determinant')
        doe.optimize_experiments(
            n_exp=2,
            initialization_method="lhs",
            lhs_n_samples=3,
            lhs_seed=42,
        )
        self.assertEqual(doe.results["Number of Experiments per Scenario"], 2)
        self.assertEqual(len(doe.results["LHS Best Initial Points"]), 2)
        fim = np.array(doe.results['Scenarios'][0]['Total FIM'])
        self.assertEqual(fim.shape, (2, 2))


@unittest.skipUnless(ipopt_available, "IPOPT not available")
class TestComputeFIMAtPointNoPrior(unittest.TestCase):
    """Unit tests for _compute_fim_at_point_no_prior (with a running kernel)."""

    def test_returns_correct_shape(self):
        """FIM matrix must have shape (n_params, n_params)."""
        exp_list = _make_experiment_list(n_exp=1, hours=[5.0])
        doe = _make_doe(exp_list, objective_option='determinant')
        # We need prior_FIM to be set (happens inside create_doe_model)
        # Set it manually here since we're calling the helper directly
        doe.prior_FIM = np.zeros((2, 2))
        fim = doe._compute_fim_at_point_no_prior(
            experiment_index=0, input_values=[5.0]
        )
        self.assertEqual(fim.shape, (2, 2))

    def test_prior_not_included(self):
        """Result must NOT include the prior FIM.

        We compare the FIM returned by the helper against the FIM computed by
        _sequential_FIM with a zero prior.  The helper must strip the prior
        before calling _sequential_FIM and restore it afterwards.
        """
        exp_list = _make_experiment_list(n_exp=1, hours=[5.0])
        doe = _make_doe(exp_list, objective_option='determinant')

        # Compute baseline FIM with zero prior
        doe.prior_FIM = np.zeros((2, 2))
        fim_baseline = doe._compute_fim_at_point_no_prior(
            experiment_index=0, input_values=[5.0]
        )

        # Now set a large prior and call the helper again -- result must match
        # the baseline (i.e. prior was NOT included)
        large_prior = np.array([[100.0, 0.0], [0.0, 100.0]])
        doe.prior_FIM = large_prior.copy()
        fim_no_prior = doe._compute_fim_at_point_no_prior(
            experiment_index=0, input_values=[5.0]
        )
        np.testing.assert_allclose(
            fim_no_prior, fim_baseline, rtol=1e-6,
            err_msg="Prior should not be included in the returned FIM.",
        )

    def test_prior_restored_after_call(self):
        """self.prior_FIM must be restored to original after the call."""
        exp_list = _make_experiment_list(n_exp=1, hours=[5.0])
        doe = _make_doe(exp_list, objective_option='determinant')
        original_prior = np.array([[3.0, 1.0], [1.0, 2.0]])
        doe.prior_FIM = original_prior.copy()
        doe._compute_fim_at_point_no_prior(
            experiment_index=0, input_values=[5.0]
        )
        np.testing.assert_array_equal(doe.prior_FIM, original_prior)


if __name__ == "__main__":
    unittest.main()

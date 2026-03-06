# __________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# __________________________________________________________________________________

from pyomo.common.unittest import pytest
import pyomo.environ as pyo
from pyomo.common.dependencies import pandas as pd, numpy as np

import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment

pytestmark = [
    pytest.mark.default,
    pytest.mark.skipif(
        not parmest.parmest_available,
        reason="Cannot test parmest regularization: required dependencies are missing",
    ),
]


class LinearExperiment(Experiment):
    """Minimal labeled experiment with two unknown parameters and one data point."""

    def __init__(self, x, y, theta0_init=0.0, theta1_init=0.0):
        self.x = float(x)
        self.y = float(y)
        self.theta0_init = float(theta0_init)
        self.theta1_init = float(theta1_init)
        super().__init__(model=None)
        self.create_model()
        self.label_model()

    def create_model(self):
        m = pyo.ConcreteModel()
        m.theta0 = pyo.Param(initialize=self.theta0_init, mutable=True)
        m.theta1 = pyo.Param(initialize=self.theta1_init, mutable=True)
        m.x = pyo.Param(initialize=self.x, mutable=True)
        m.pred = pyo.Expression(expr=m.theta0 + m.theta1 * m.x)
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.pred, self.y)])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.ComponentUID(k)) for k in [m.theta0, m.theta1]
        )


class DummyExperiment(Experiment):
    """Tiny experiment used for covariance helper tests."""

    def __init__(self):
        m = pyo.ConcreteModel()
        m.theta0 = pyo.Param(initialize=0.0, mutable=True)
        m.theta1 = pyo.Param(initialize=0.0, mutable=True)
        m.pred = pyo.Expression(expr=m.theta0 + m.theta1)
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.pred, 0.0)])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.ComponentUID(k)) for k in [m.theta0, m.theta1]
        )
        super().__init__(model=m)


def _make_var_labeled_model(y_obs=5.0):
    m = pyo.ConcreteModel()
    m.theta0 = pyo.Var(initialize=0.0)
    m.theta1 = pyo.Var(initialize=0.0)
    m.pred = pyo.Expression(expr=m.theta0 + 2.0 * m.theta1)
    m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.experiment_outputs.update([(m.pred, float(y_obs))])
    m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.unknown_parameters.update((k, pyo.ComponentUID(k)) for k in [m.theta0, m.theta1])
    return m


def _obj_at_theta(pest, theta0, theta1):
    theta = pd.DataFrame([[theta0, theta1]], columns=["theta0", "theta1"])
    return pest.objective_at_theta(theta_values=theta).iloc[0]["obj"]


def _argmin_on_grid(pest, theta1_grid, theta0=0.0):
    thetas = pd.DataFrame(
        {"theta0": [theta0] * len(theta1_grid), "theta1": theta1_grid}
    )
    obj_df = pest.objective_at_theta(theta_values=thetas)
    best_idx = obj_df["obj"].idxmin()
    return float(obj_df.loc[best_idx, "theta1"])


def test_l2_objective_value_matches_manual_quadratic():
    m = _make_var_labeled_model(y_obs=5.0)
    m.theta0.set_value(4.0)
    m.theta1.set_value(-1.0)

    prior_fim = pd.DataFrame(
        [[2.0, 0.0], [0.0, 4.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )
    theta_ref = pd.Series({"theta0": 1.0, "theta1": 2.0})
    weight = 3.0

    expr = parmest.L2_regularized_objective(
        m,
        prior_FIM=prior_fim,
        theta_ref=theta_ref,
        regularization_weight=weight,
        obj_function=parmest.SSE,
    )

    # SSE = (5 - (4 + 2*(-1)))^2 = 9
    sse_expected = 9.0
    # delta = [3, -3], 0.5 * delta^T*diag(2,4)*delta = 27
    l2_expected = 27.0
    expected = sse_expected + weight * l2_expected

    assert pyo.value(expr) == pytest.approx(expected)


def test_l2_objective_difference_equals_weighted_penalty_once():
    m = _make_var_labeled_model(y_obs=4.0)
    m.theta0.set_value(2.0)
    m.theta1.set_value(1.5)

    prior_fim = pd.DataFrame(
        [[1.0, 0.2], [0.2, 3.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )
    theta_ref = pd.Series({"theta0": 0.0, "theta1": 0.0})
    weight = 2.5

    base = parmest.SSE(m)
    penalty = parmest._calculate_L2_penalty(m, prior_fim, theta_ref)
    reg = parmest.L2_regularized_objective(
        m,
        prior_FIM=prior_fim,
        theta_ref=theta_ref,
        regularization_weight=weight,
        obj_function=parmest.SSE,
    )

    assert pyo.value(reg) - pyo.value(base) == pytest.approx(
        weight * pyo.value(penalty)
    )


def test_l2_penalty_not_double_counted_across_scenarios():
    exp_list = [LinearExperiment(1.0, 1.0), LinearExperiment(2.0, 2.0)]
    prior_fim = pd.DataFrame(
        [[0.0, 0.0], [0.0, 2.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )
    theta_ref = pd.Series({"theta0": 0.0, "theta1": 0.0})

    pest = parmest.Estimator(
        exp_list,
        obj_function="SSE",
        regularization="L2",
        prior_FIM=prior_fim,
        theta_ref=theta_ref,
        regularization_weight=1.0,
    )

    theta0 = 0.0
    theta1 = 1.0
    obj_val = _obj_at_theta(pest, theta0, theta1)

    # Average SSE: ((1 - 1)^2 + (2 - 2)^2) / 2 = 0
    sse_avg = 0.0
    # penalty: 0.5 * 2 * (1 - 0)^2 = 1
    penalty = 1.0
    assert obj_val == pytest.approx(sse_avg + penalty)


def test_regularization_requires_explicit_option_when_prior_supplied():
    exp_list = [LinearExperiment(1.0, 1.0)]
    prior_fim = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )

    with pytest.raises(
        ValueError, match="regularization must be set when supplying prior_FIM"
    ):
        parmest.Estimator(exp_list, obj_function="SSE", prior_FIM=prior_fim)


def test_l2_regularization_requires_prior_fim():
    exp_list = [LinearExperiment(1.0, 1.0)]

    with pytest.raises(ValueError, match="prior_FIM must be provided"):
        parmest.Estimator(exp_list, obj_function="SSE", regularization="L2")


def test_user_specified_unsupported_regularization_raises():
    exp_list = [LinearExperiment(1.0, 1.0)]

    with pytest.raises(
        TypeError, match="regularization must be None or one of \\['L2'\\]"
    ):
        parmest.Estimator(
            exp_list, obj_function="SSE", regularization=lambda m: m.theta0**2
        )


def test_lambda_zero_matches_unregularized_objective():
    exp_list = [LinearExperiment(1.0, 1.0), LinearExperiment(2.0, 2.0)]
    prior_fim = pd.DataFrame(
        [[2.0, 0.0], [0.0, 1.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )
    theta_ref = pd.Series({"theta0": 0.0, "theta1": 0.0})

    pest_base = parmest.Estimator(exp_list, obj_function="SSE")
    pest_l2_zero = parmest.Estimator(
        exp_list,
        obj_function="SSE",
        regularization="L2",
        prior_FIM=prior_fim,
        theta_ref=theta_ref,
        regularization_weight=0.0,
    )

    for theta0, theta1 in [(0.0, 0.0), (0.5, 1.5), (-1.0, 2.0)]:
        obj_base = _obj_at_theta(pest_base, theta0, theta1)
        obj_l2_zero = _obj_at_theta(pest_l2_zero, theta0, theta1)
        assert obj_l2_zero == pytest.approx(obj_base)


def test_lambda_increase_shrinks_theta_toward_zero_grid_search():
    exp_list = [
        LinearExperiment(1.0, 2.0),
        LinearExperiment(2.0, 4.0),
        LinearExperiment(3.0, 6.0),
    ]
    prior_fim = pd.DataFrame(
        [[0.0, 0.0], [0.0, 1.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )
    theta_ref = pd.Series({"theta0": 0.0, "theta1": 0.0})
    theta1_grid = np.linspace(0.0, 2.5, 101)

    lambdas = [0.0, 0.2, 1.0, 5.0]
    best_abs = []
    for lam in lambdas:
        pest = parmest.Estimator(
            exp_list,
            obj_function="SSE",
            regularization="L2",
            prior_FIM=prior_fim,
            theta_ref=theta_ref,
            regularization_weight=lam,
        )
        theta1_hat = _argmin_on_grid(pest, theta1_grid, theta0=0.0)
        best_abs.append(abs(theta1_hat))

    assert best_abs[0] >= best_abs[1] >= best_abs[2] >= best_abs[3]


def test_lambda_increase_shrinks_theta_toward_target_grid_search():
    exp_list = [
        LinearExperiment(1.0, 2.0),
        LinearExperiment(2.0, 4.0),
        LinearExperiment(3.0, 6.0),
    ]
    prior_fim = pd.DataFrame(
        [[0.0, 0.0], [0.0, 1.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )
    theta_ref = pd.Series({"theta0": 0.0, "theta1": 1.5})
    theta1_grid = np.linspace(1.0, 2.5, 151)

    lambdas = [0.0, 0.2, 1.0, 5.0]
    best_dist = []
    for lam in lambdas:
        pest = parmest.Estimator(
            exp_list,
            obj_function="SSE",
            regularization="L2",
            prior_FIM=prior_fim,
            theta_ref=theta_ref,
            regularization_weight=lam,
        )
        theta1_hat = _argmin_on_grid(pest, theta1_grid, theta0=0.0)
        best_dist.append(abs(theta1_hat - 1.5))

    assert best_dist[0] >= best_dist[1] >= best_dist[2] >= best_dist[3]


def test_prior_subset_penalizes_only_selected_parameter():
    exp_list = [LinearExperiment(1.0, 1.0)]
    prior_fim = pd.DataFrame([[4.0]], index=["theta1"], columns=["theta1"])

    pest_base = parmest.Estimator(exp_list, obj_function="SSE")
    pest_l2 = parmest.Estimator(
        exp_list,
        obj_function="SSE",
        regularization="L2",
        prior_FIM=prior_fim,
        theta_ref=pd.Series({"theta1": 0.0}),
        regularization_weight=1.0,
    )

    # Change theta0 only: no extra penalty should appear.
    obj_base_theta0 = _obj_at_theta(pest_base, theta0=1.0, theta1=0.0)
    obj_l2_theta0 = _obj_at_theta(pest_l2, theta0=1.0, theta1=0.0)
    assert obj_l2_theta0 == pytest.approx(obj_base_theta0)

    # Change theta1: penalty should be added.
    obj_base_theta1 = _obj_at_theta(pest_base, theta0=0.0, theta1=1.0)
    obj_l2_theta1 = _obj_at_theta(pest_l2, theta0=0.0, theta1=1.0)
    expected_penalty = 0.5 * 4.0 * (1.0**2)
    assert obj_l2_theta1 - obj_base_theta1 == pytest.approx(expected_penalty)


def test_negative_regularization_weight_raises():
    exp_list = [LinearExperiment(1.0, 1.0)]
    prior_fim = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )

    with pytest.raises(ValueError, match="regularization_weight must be nonnegative"):
        parmest.Estimator(
            exp_list,
            obj_function="SSE",
            regularization="L2",
            prior_FIM=prior_fim,
            regularization_weight=-1.0,
        )


def test_missing_theta_ref_entries_raise_clear_error():
    exp_list = [LinearExperiment(1.0, 1.0)]
    prior_fim = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )

    pest = parmest.Estimator(
        exp_list,
        obj_function="SSE",
        regularization="L2",
        prior_FIM=prior_fim,
        theta_ref=pd.Series({"theta0": 0.0}),
        regularization_weight=1.0,
    )

    with pytest.raises(
        ValueError, match=r"theta_ref is missing values for parameter\(s\): theta1"
    ):
        _ = _obj_at_theta(pest, theta0=0.0, theta1=0.0)


def test_non_psd_prior_fim_rejected():
    exp_list = [LinearExperiment(1.0, 1.0)]
    non_psd = pd.DataFrame(
        [[1.0, 2.0], [2.0, -1.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )

    with pytest.raises(ValueError, match="positive semi-definite"):
        parmest.Estimator(
            exp_list, obj_function="SSE", regularization="L2", prior_FIM=non_psd
        )


def test_compute_covariance_matrix_adds_prior_fim_weighted(monkeypatch):
    # Build two dummy experiments so FIM sum is deterministic with monkeypatch.
    exp_list = [DummyExperiment(), DummyExperiment()]

    def fake_finite_difference_FIM(*args, **kwargs):
        return np.eye(2)

    monkeypatch.setattr(parmest, "_finite_difference_FIM", fake_finite_difference_FIM)

    prior_fim = pd.DataFrame(
        [[1.0, 0.0], [0.0, 3.0]],
        index=["theta0", "theta1"],
        columns=["theta0", "theta1"],
    )
    theta_vals = {"theta0": 0.0, "theta1": 0.0}

    cov = parmest.compute_covariance_matrix(
        experiment_list=exp_list,
        method=parmest.CovarianceMethod.finite_difference.value,
        obj_function=parmest.SSE,
        theta_vals=theta_vals,
        step=1e-3,
        solver="ipopt",
        tee=False,
        prior_FIM=prior_fim,
        regularization_weight=2.0,
    )

    # FIM_total = 2*I + 2*diag(1,3) = diag(4,8) => cov = diag(1/4, 1/8)
    assert cov.loc["theta0", "theta0"] == pytest.approx(0.25)
    assert cov.loc["theta1", "theta1"] == pytest.approx(0.125)
    assert cov.loc["theta0", "theta1"] == pytest.approx(0.0)
    assert cov.loc["theta1", "theta0"] == pytest.approx(0.0)

# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.doe.doe import _SMALL_TOLERANCE_DEFINITENESS


class _FakeResult:
    """Minimal fake solver result object matching attributes used by DoE."""

    class _SolverData:
        status = "ok"
        termination_condition = "optimal"
        message = "fake solve"

    solver = _SolverData()


class _MutatingRecordingSolver:
    """
    Fake solver that perturbs FIM during dummy-objective initialization solve.

    The key behavior we want to exercise is not optimization success, but the
    fact that the square initialization solve can change the FIM after the
    object has already created Cholesky-related variables. That is exactly the
    bug this regression test protects.
    """

    def __init__(self):
        self.options = {}

    def solve(self, model, tee=False):
        """Mutate 2x2 FIM values during init-stage solve."""
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


class _TwoParamExperiment:
    """
    Minimal two-parameter experiment fixture for trace/Cholesky tests.

    Two parameters is the smallest useful case here because it creates a real
    matrix-valued FIM while still keeping the expected values easy to reason
    about by hand. That makes it a good target for the initialization
    synchronization regression.
    """

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


def _make_trace_doe_object():
    """
    Build a trace-objective DoE object with a FIM-mutating fake solver.

    The trace objective is the path that introduces ``L``, ``L_inv``,
    ``fim_inv``, and ``cov_trace``, so it is the right place to check that the
    helper re-synchronizes all of those values after the init solve.
    """
    return DesignOfExperiments(
        experiment=_TwoParamExperiment(),
        fd_formula="central",
        step=1e-3,
        objective_option="trace",
        solver=_MutatingRecordingSolver(),
        tee=False,
    )


@unittest.skipIf(not numpy_available, "Pyomo.DoE needs numpy to run tests")
class TestCholeskyInitialization(unittest.TestCase):
    """Regression tests for DoE initialization and Cholesky sync."""

    def test_compute_cholesky_jitter_raises_negative_eigenvalue(self):
        """
        Negative minimum eigenvalue should produce positive corrective jitter.

        This is a direct unit test for the helper used by the initialization
        sync path, so we can catch arithmetic regressions independently of the
        solver/modeled example below.
        """
        doe_obj = _make_trace_doe_object()
        min_eig = -1.0e-3
        jitter = doe_obj._compute_cholesky_jitter(min_eig)
        self.assertAlmostEqual(
            jitter, _SMALL_TOLERANCE_DEFINITENESS - min_eig, places=14
        )

    def test_compute_cholesky_jitter_zero_when_not_needed(self):
        """
        Positive minimum eigenvalue above tolerance should yield zero jitter.

        We keep this separate from the negative-eigenvalue case because it
        verifies the helper does not add unnecessary regularization when the
        FIM is already well-conditioned.
        """
        doe_obj = _make_trace_doe_object()
        min_eig = 1.0e-2
        jitter = doe_obj._compute_cholesky_jitter(min_eig)
        self.assertEqual(jitter, 0.0)

    def test_trace_initialization_resynchronizes_fim_inverse_variables(self):
        """
        Verify trace-mode initialization re-synchronizes inverse-related vars.

        The fake solver mutates the FIM during the square solve. After that
        solve, the DoE code should rebuild the dependent quantities from the
        updated FIM so that:
        - ``L`` matches the Cholesky factor of the current FIM,
        - ``L_inv`` is the inverse of that Cholesky factor,
        - ``fim_inv`` matches the inverse FIM, and
        - ``cov_trace`` matches the trace of the inverse FIM.
        """
        doe_obj = _make_trace_doe_object()
        doe_obj.run_doe()

        model = doe_obj.model
        params = list(model.parameter_names)
        fim = np.array(
            [[pyo.value(model.fim[i, j]) for j in params] for i in params], dtype=float
        )
        L = np.array([[pyo.value(model.L[i, j]) for j in params] for i in params])
        L_inv = np.array(
            [[pyo.value(model.L_inv[i, j]) for j in params] for i in params]
        )
        fim_inv = np.array(
            [[pyo.value(model.fim_inv[i, j]) for j in params] for i in params]
        )

        expected_fim = np.array([[16.0, 4.0], [4.0, 9.0]])
        expected_fim_inv = np.linalg.inv(expected_fim)
        expected_L = np.linalg.cholesky(expected_fim)
        expected_L_inv = np.linalg.inv(expected_L)
        fim_sym = fim + fim.T - np.diag(np.diag(fim))
        fim_inv_sym = fim_inv + fim_inv.T - np.diag(np.diag(fim_inv))

        # The model stores only the lower triangle when only_compute_fim_lower=True.
        # Reconstruct the symmetric FIM before checking the initialization math,
        # because Cholesky and inverse quantities are defined on the full matrix.
        self.assertTrue(np.allclose(fim_sym, expected_fim))
        self.assertTrue(np.allclose(L, expected_L))
        self.assertTrue(np.allclose(L @ L_inv, np.eye(len(params)), atol=1e-8))
        self.assertTrue(np.allclose(fim_inv_sym, expected_fim_inv))
        self.assertTrue(np.allclose(L_inv, expected_L_inv))
        self.assertAlmostEqual(
            pyo.value(model.cov_trace), float(np.trace(expected_fim_inv)), places=10
        )


if __name__ == "__main__":
    unittest.main()

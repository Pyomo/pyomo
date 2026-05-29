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
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe.utils import (
    _SMALL_TOLERANCE_DEFINITENESS,
    regularize_fim_for_cholesky,
)
from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()


class _StructurallyUnidentifiableExperiment(Experiment):
    """
    Minimal real experiment with a rank-deficient FIM.

    The single measured response depends only on ``theta1 + theta2``:
    ``y = (theta1 + theta2) * x``. With one output and two parameters, both
    sensitivities are identical, so the analytical FIM is
    ``[[1, 1], [1, 1]]`` when ``x = 1`` and the measurement error is 1.
    """

    def get_labeled_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0, bounds=(1.0, 1.0))
        m.theta1 = pyo.Var(initialize=2.0)
        m.theta2 = pyo.Var(initialize=3.0)
        m.theta1.fix()
        m.theta2.fix()
        m.y = pyo.Var(initialize=5.0)
        m.response = pyo.Constraint(expr=m.y == (m.theta1 + m.theta2) * m.x)

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.x] = 1.0
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters[m.theta1] = 2.0
        m.unknown_parameters[m.theta2] = 3.0
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[m.y] = 5.0
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[m.y] = 1.0
        return m


def _make_unidentifiable_doe_object(objective_option="trace"):
    """
    Build a DoE object for the structurally unidentifiable two-parameter model.
    """
    return DesignOfExperiments(
        experiment=_StructurallyUnidentifiableExperiment(),
        fd_formula="central",
        step=1e-3,
        objective_option=objective_option,
        solver=SolverFactory("ipopt"),
        tee=False,
    )


@unittest.skipIf(not numpy_available, "Pyomo.DoE needs numpy to run tests")
class TestCholeskyInitialization(unittest.TestCase):
    """Regression tests for DoE initialization and Cholesky sync."""

    def test_regularize_fim_for_cholesky_raises_negative_eigenvalue(self):
        """
        Negative minimum eigenvalue should produce positive corrective jitter.

        This is a direct unit test for the helper used by the initialization
        sync path, so we can catch arithmetic regressions independently of the
        solver/modeled example below.
        """
        fim = np.array([[2.0, -1.0], [-1.0, 0.0]])
        fim_pd, jitter = regularize_fim_for_cholesky(fim)
        self.assertGreater(jitter, 0.0)
        min_eig = np.min(np.linalg.eigvalsh(fim_pd))
        self.assertAlmostEqual(
            min_eig,
            _SMALL_TOLERANCE_DEFINITENESS,
            delta=_SMALL_TOLERANCE_DEFINITENESS / 10,
        )

    def test_regularize_fim_for_cholesky_zero_when_not_needed(self):
        """
        Positive minimum eigenvalue above tolerance should yield zero jitter.

        We keep this separate from the negative-eigenvalue case because it
        verifies the helper does not add unnecessary regularization when the
        FIM is already well-conditioned.
        """
        fim = np.array([[1.0e-2]])
        fim_pd, jitter = regularize_fim_for_cholesky(fim)
        self.assertEqual(jitter, 0.0)
        self.assertAlmostEqual(fim_pd[0, 0], 1.0e-2)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_trace_initialization_regularizes_structurally_unidentifiable_fim(self):
        """
        Verify trace-mode initialization regularizes a singular real-model FIM.

        This experiment is structurally unidentifiable because the single
        response depends only on ``theta1 + theta2``. Its analytical FIM is
        rank-deficient, so Cholesky-based initialization must add diagonal
        regularization before computing the Cholesky factors.

        We intentionally do not call ``run_doe()`` here. The behavior under
        test is the Cholesky/FIM synchronization step after a singular FIM is
        available. ``run_doe()`` continues on to additional solver-driven
        optimization steps and does not expose a clean pause point between
        "the FIM values are now known" and "initialize the Cholesky-related
        variables from those values." Instead, this test uses the real DoE
        model-building machinery, writes the analytically known singular FIM
        for this experiment onto ``model.fim``, and then directly exercises
        ``_initialize_cholesky_from_fim()``.
        """
        doe_obj = _make_unidentifiable_doe_object(objective_option="trace")
        doe_obj.create_doe_model()
        doe_obj.create_objective_function()

        model = doe_obj.model
        params = list(model.parameter_names)
        expected_fim = np.array([[1.0, 1.0], [1.0, 1.0]])
        # Pyomo.DoE stores only the lower triangle when
        # ``only_compute_fim_lower`` is enabled. Populate ``model.fim`` using
        # the analytical singular FIM for this experiment so the initialization
        # helper sees the exact pathological case we want to regularize.
        for i, p in enumerate(params):
            for j, q in enumerate(params):
                if doe_obj.only_compute_fim_lower and i < j:
                    model.fim[p, q].set_value(0.0)
                else:
                    model.fim[p, q].set_value(expected_fim[i, j])

        doe_obj._initialize_cholesky_from_fim()

        L = np.array([[pyo.value(model.L[i, j]) for j in params] for i in params])

        expected_fim_pd, jitter = regularize_fim_for_cholesky(expected_fim)
        expected_L = np.linalg.cholesky(expected_fim_pd)
        reconstructed_fim = L @ L.T

        self.assertGreater(jitter, 0.0)
        self.assertTrue(np.allclose(L, expected_L))
        self.assertTrue(np.allclose(reconstructed_fim, expected_fim_pd))
        self.assertAlmostEqual(
            np.min(np.linalg.eigvalsh(reconstructed_fim)),
            _SMALL_TOLERANCE_DEFINITENESS,
            delta=_SMALL_TOLERANCE_DEFINITENESS / 10,
        )


if __name__ == "__main__":
    unittest.main()

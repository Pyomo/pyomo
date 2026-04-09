# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.environ as pyo

from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.parmest.experiment import Experiment


class PolynomialExperiment(Experiment):
    """A small algebraic experiment used for symbolic-gradient testing."""

    def __init__(self):
        self.model = None

    def get_labeled_model(self):
        """Build and label the experiment model on first access."""
        if self.model is None:
            self.create_model()
            self.label_experiment()
        return self.model

    def create_model(self):
        """Define a polynomial model for testing symbolic sensitivities.

        y = a*x1 + b*x2 + c*x1*x2 + d
        """

        m = self.model = pyo.ConcreteModel()

        m.x1 = pyo.Var(bounds=(-5, 5), initialize=2.0)
        m.x2 = pyo.Var(bounds=(-5, 5), initialize=3.0)

        m.a = pyo.Var(bounds=(-5, 5), initialize=2)
        m.b = pyo.Var(bounds=(-5, 5), initialize=-1)
        m.c = pyo.Var(bounds=(-5, 5), initialize=0.5)
        m.d = pyo.Var(bounds=(-5, 5), initialize=-1)

        m.y = pyo.Var(initialize=0)

        @m.Constraint()
        def output_equation(m):
            return m.y == m.a * m.x1 + m.b * m.x2 + m.c * m.x1 * m.x2 + m.d

    def label_experiment(self):
        """Attach the standard DoE suffixes to the polynomial model."""
        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[m.y] = None

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[m.y] = 1

        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.x1] = None
        m.experiment_inputs[m.x2] = None

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.a, m.b, m.c, m.d])


def run_polynomial_doe():
    """Run a small symbolic DoE FIM calculation for the polynomial model."""
    experiment = PolynomialExperiment()

    doe_obj = DesignOfExperiments(
        experiment,
        gradient_method="pynumero",
        step=1e-3,
        objective_option="determinant",
        scale_constant_value=1,
        scale_nominal_param_value=False,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=pyo.SolverFactory("ipopt"),
        tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    return doe_obj.compute_FIM()


if __name__ == "__main__":
    run_polynomial_doe()

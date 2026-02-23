# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
import pyomo.environ as pyo

from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment,
)


class BadExperiment:
    def __init__(self):
        self.model = None


class RooneyBieglerExperimentFlag(RooneyBieglerExperiment):
    """
    A version of RooneyBieglerExperiment that supports flag parameter
    to create incomplete models for testing error handling.

    Flag values:
    - 0: Full model (all suffixes present)
    - 1: No experiment_outputs suffix
    - 2: No measurement_error suffix
    - 3: No experiment_inputs suffix
    - 4: No unknown_parameters suffix
    - 5: Incorrect measurement error structure (only one output gets error)
    """

    def get_labeled_model(self, flag=0):
        """Get the labeled model with optional flag to create incomplete models."""
        m = super().get_labeled_model()

        # If flag is non-zero, remove the corresponding suffix to create an incomplete model
        if flag == 1:
            m.del_component(m.experiment_outputs)
        elif flag == 2:
            m.del_component(m.measurement_error)
        elif flag == 3:
            m.del_component(m.experiment_inputs)
        elif flag == 4:
            m.del_component(m.unknown_parameters)
        elif flag == 5:
            # Create mismatch: 1 experiment output but multiple measurement errors
            # This tests the validation that checks output/error length match
            # Add a fake component to measurement_error to create length mismatch
            m.fake_output = pyo.Var()
            m.measurement_error[m.fake_output] = 0.1

        return m


class RooneyBieglerExperimentBad(RooneyBieglerExperiment):
    """
    A bad version of RooneyBieglerExperiment with conflicting constraints
    for testing error handling.
    """

    def get_labeled_model(self):
        m = super().get_labeled_model()

        # Add conflicting constraints that make the model infeasible
        # The hour variable should be >= 10 and <= 0 (impossible)
        m.bad_con_1 = pyo.Constraint(expr=m.hour >= 10.0)
        m.bad_con_2 = pyo.Constraint(expr=m.hour <= 0.0)

        return m


def rooney_biegler_multiexperiment_model(data, theta=None, hour_bounds=(1.0, 10.0)):
    """Small Rooney-Biegler model for fast multi-experiment DoE tests."""
    model = pyo.ConcreteModel()

    if theta is None:
        theta = {'asymptote': 15, 'rate_constant': 0.5}

    model.asymptote = pyo.Var(initialize=theta['asymptote'])
    model.rate_constant = pyo.Var(initialize=theta['rate_constant'])
    model.asymptote.fix()
    model.rate_constant.fix()

    model.hour = pyo.Var(initialize=data['hour'], bounds=hour_bounds)
    model.hour.fix()

    model.y = pyo.Var(within=pyo.PositiveReals, initialize=data['y'])
    model.response_function = pyo.Constraint(
        expr=model.y
        == model.asymptote * (1 - pyo.exp(-model.rate_constant * model.hour))
    )
    return model


class RooneyBieglerMultiExperiment(Experiment):
    """
    Experiment class based on the multi-experiment Rooney-Biegler prototype.

    This mirrors the implementation in
    ``examples/multiexperiment-prototype/rooney_biegler_multiexperiment.py``
    while allowing test-time control over initial hour and bounds.
    """

    def __init__(
        self,
        hour=2.0,
        y=10.0,
        theta=None,
        measure_error=0.1,
        hour_bounds=(1.0, 10.0),
    ):
        self.hour = hour
        self.y = y
        self.theta = theta if theta is not None else {'asymptote': 15, 'rate_constant': 0.5}
        self.measure_error = measure_error
        self.hour_bounds = hour_bounds

    def get_labeled_model(self):
        m = rooney_biegler_multiexperiment_model(
            {'hour': self.hour, 'y': self.y},
            theta=self.theta,
            hour_bounds=self.hour_bounds,
        )

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

        m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.sym_break_cons[m.hour] = None
        return m

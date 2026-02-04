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
import pyomo.environ as pyo

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
        if flag == 1 and hasattr(m, 'experiment_outputs'):
            delattr(m, 'experiment_outputs')
        elif flag == 2 and hasattr(m, 'measurement_error'):
            delattr(m, 'measurement_error')
        elif flag == 3 and hasattr(m, 'experiment_inputs'):
            delattr(m, 'experiment_inputs')
        elif flag == 4 and hasattr(m, 'unknown_parameters'):
            delattr(m, 'unknown_parameters')
        elif flag == 5:
            # Create mismatch: 1 experiment output but multiple measurement errors
            # This tests the validation that checks output/error length match
            if hasattr(m, 'measurement_error'):
                import pyomo.environ as pyo

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

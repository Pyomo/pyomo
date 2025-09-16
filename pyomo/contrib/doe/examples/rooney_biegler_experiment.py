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
Rooney Biegler model, based on Rooney, W. C. and Biegler, L. T. (2001). Design for
model parameter uncertainty using nonlinear confidence regions. AIChE Journal,
47(8), 1794-1804.
"""

from pyomo.common.dependencies import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment


class RooneyBieglerExperimentDoE(Experiment):
    def __init__(self, data=None, theta=None):
        if data is None:
            self.data = {}
            self.data['hour'] = 1
            self.data['y'] = 8.3
        else:
            self.data = data
        if theta is None:
            self.theta = {}
            self.theta['asymptote'] = 19.143
            self.theta['rate constant'] = 0.5311
        else:
            self.theta = theta
        self.model = None

    def create_model(self):
        # Creates Roony-Biegler model for
        # individual data points as
        # an experimental decision.
        m = self.model = pyo.ConcreteModel()

        # Specify the unknown parameters
        m.asymptote = pyo.Var(initialize=self.theta['asymptote'])
        m.rate_constant = pyo.Var(initialize=self.theta['rate constant'])

        # Fix the unknown parameters
        m.asymptote.fix()
        m.rate_constant.fix()

        # Add the experiment inputs
        m.hour = pyo.Var(initialize=self.data['hour'], bounds=(0, 10))

        # Fix the experimental design variable
        m.hour.fix()

        # Add the experiment outputs
        m.y = pyo.Var(initialize=self.data['y'])

        # Add governing equation
        m.response_function = pyo.Constraint(
            expr=m.y - m.asymptote * (1 - pyo.exp(-m.rate_constant * m.hour)) == 0
        )

    def finalize_model(self):
        m = self.model
        pass

    def label_model(self):
        m = self.model

        # Add y value as experiment output
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs[m.y] = m.y()

        # Add measurement error associated with y
        # We are assuming a flat error of 0.3
        # or about 1-3 percent
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error[m.y] = 1

        # Add hour as experiment input
        # We are deciding when to sample
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs[m.hour] = m.hour()

        # Adding the unknown parameters
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, k.value) for k in [m.asymptote, m.rate_constant]
        )

    def get_labeled_model(self):
        self.create_model()
        self.finalize_model()
        self.label_model()

        return self.model

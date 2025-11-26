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


def rooney_biegler_model(data, theta=None):
    # if theta is  None:
    #     theta = {}
    #     theta['asymptote'] = 15
    #     theta['rate constant'] = 0.5
    model = pyo.ConcreteModel()

    model.asymptote = pyo.Var(initialize=theta['asymptote'])
    model.rate_constant = pyo.Var(initialize=theta['rate constant'])

    model.y = pyo.Var(within=pyo.PositiveReals, initialize=5)

    def response_rule(m, h):
        return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * h))

    model.response_function = pyo.Constraint(data.hour, rule=response_rule)

    return model


class RooneyBieglerExperiment(Experiment):

    def __init__(self, data=None, measure_error=None, theta=None):
        self.data = data

        if measure_error is None:
            self.measure_error = 1
        else:
            self.measure_error = measure_error

        if theta is None:
            self.theta = {}
            self.theta['asymptote'] = 15.0
            self.theta['rate constant'] = 0.5
        else:
            self.theta = theta

        self.model = None

    def create_model(self):
        # rooney_biegler_model expects a dataframe
        data_df = self.data.to_frame().transpose()
        self.model = rooney_biegler_model(data_df, theta=self.theta)

    def label_model(self):

        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, self.data.loc['y'])])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.ComponentUID(k)) for k in [m.asymptote, m.rate_constant]
        )

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, self.measure_error)])

    def finalize_model(self):

        m = self.model

        # Experiment input values
        m.hour = self.data['hour']

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        self.finalize_model()

        return self.model

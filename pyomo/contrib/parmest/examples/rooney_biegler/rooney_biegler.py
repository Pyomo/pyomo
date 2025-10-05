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


def rooney_biegler_model(data):
    model = pyo.ConcreteModel()

    model.asymptote = pyo.Var(initialize=15)
    model.rate_constant = pyo.Var(initialize=0.5)

    model.y = pyo.Var(data.hour, within=pyo.PositiveReals, initialize=5)

    def response_rule(m, h):
        return m.y[h] == m.asymptote * (1 - pyo.exp(-m.rate_constant * h))

    model.response_function = pyo.Constraint(data.hour, rule=response_rule)

    def SSE_rule(m):
        return sum((data.y[i] - m.y[data.hour[i]]) ** 2 for i in data.index)

    model.SSE = pyo.Objective(rule=SSE_rule, sense=pyo.minimize)

    return model


class RooneyBieglerExperiment(Experiment):

    def __init__(self, data, measure_error=None):
        self.data = data
        self.model = None
        self.measure_error = measure_error

    def create_model(self):
        # rooney_biegler_model expects a dataframe
        data_df = self.data.to_frame().transpose()
        self.model = rooney_biegler_model(data_df)

    def label_model(self):

        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y[self.data['hour']], self.data['y'])])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.ComponentUID(k)) for k in [m.asymptote, m.rate_constant]
        )

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y[self.data['hour']], self.measure_error)])

    def finalize_model(self):

        m = self.model

        # Experiment input values
        m.hour = self.data['hour']

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        self.finalize_model()

        return self.model


def main():
    # These were taken from Table A1.4 in Bates and Watts (1988).
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )

    model = rooney_biegler_model(data)
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model)

    print('asymptote = ', model.asymptote())
    print('rate constant = ', model.rate_constant())


if __name__ == '__main__':
    main()

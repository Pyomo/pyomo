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
    """Create a Pyomo model for the Rooney-Biegler parameter estimation problem.

    This model implements an exponential response function based on Rooney & Biegler (2001).
    The response is: y = asymptote * (1 - exp(-rate_constant * hour))

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing 'hour' and 'y' columns for the experiment data point.
    theta : dict, optional
        Dictionary with 'asymptote' and 'rate_constant' keys for parameter initialization.
        Default is {'asymptote': 15, 'rate_constant': 0.5}.

    Returns
    -------
    pyo.ConcreteModel
        A Pyomo ConcreteModel with fixed parameters, experiment inputs (hour),
        and response variable (y) with the exponential constraint.
    """
    model = pyo.ConcreteModel()

    if theta is None:
        theta = {'asymptote': 15, 'rate_constant': 0.5}

    model.asymptote = pyo.Var(initialize=theta['asymptote'])
    model.rate_constant = pyo.Var(initialize=theta['rate_constant'])

    # Fix the unknown parameters
    model.asymptote.fix()
    model.rate_constant.fix()

    # Add the experiment inputs
    model.hour = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))

    # Fix the experiment inputs
    model.hour.fix()

    # Add the response variable
    model.y = pyo.Var(within=pyo.PositiveReals, initialize=data["y"].iloc[0])

    def response_rule(m):
        return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.hour))

    model.response_function = pyo.Constraint(rule=response_rule)

    return model


class RooneyBieglerExperiment(Experiment):
    """Experiment class for Rooney-Biegler parameter estimation and design of experiments.

    This class wraps the Rooney-Biegler exponential model for use with Pyomo's
    parmest (parameter estimation) and DoE (Design of Experiments) tools.
    """

    def __init__(self, data, measure_error=None, theta=None):
        """Initialize a Rooney-Biegler experiment instance.

        Parameters
        ----------
        data : pandas.Series or dict
            Experiment data containing 'hour' (time input) and 'y' (response) values.
        measure_error : float, optional
            Standard deviation of measurement error for the response variable.
            Required for DoE and covariance estimation. Default is None.
        theta : dict, optional
            Initial parameter values with 'asymptote' and 'rate_constant' keys.
            Default is None, which uses {'asymptote': 15, 'rate_constant': 0.5}.
        """
        self.data = data
        self.model = None
        self.measure_error = measure_error
        self.theta = theta

    def create_model(self):
        # rooney_biegler_model expects a dataframe
        if hasattr(self.data, 'to_frame'):
            # self.data is a pandas Series
            data_df = self.data.to_frame().transpose()
        else:
            # self.data is a dict
            data_df = pd.DataFrame([self.data])
        self.model = rooney_biegler_model(data_df, theta=self.theta)

    def label_model(self):

        m = self.model

        # Add experiment outputs as a suffix
        # Experiment outputs suffix is required for parmest
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, self.data['y'])])

        # Add unknown parameters as a suffix
        # Unknown parameters suffix is required for both Pyomo.DoE and parmest
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
        )

        # Add measurement error as a suffix
        # Measurement error suffix is required for Pyomo.DoE and
        #  `cov` estimation in parmest
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, self.measure_error)])

        # Add hour as an experiment input
        # Experiment inputs suffix is required for Pyomo.DoE
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.hour, self.data['hour'])])

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.label_model()
        return self.model

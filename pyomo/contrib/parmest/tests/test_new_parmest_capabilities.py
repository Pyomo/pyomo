#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ________________________________________________________________________
#  ___

import platform
import sys
import os
import subprocess
from itertools import product

import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae

from pyomo.common.dependencies import numpy as np, pandas as pd, scipy, matplotlib
from pyomo.common.fileutils import this_file_dir
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.opt import SolverFactory

is_osx = platform.mac_ver()[0] != ""
ipopt_available = SolverFactory("ipopt").available()
pynumero_ASL_available = AmplInterface.available()
testdir = this_file_dir()


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)

# Test class for the built-in Parmest `SSE_weighted` objective function
@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestModelVariants(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        # create the Rooney-Biegler model
        def rooney_biegler_model():
            """
            Formulates the Pyomo model of the Rooney-Biegler example

            Returns:
                m: Pyomo model
            """
            m = pyo.ConcreteModel()

            m.asymptote = pyo.Var(within=pyo.NonNegativeReals, initialize=10)
            m.rate_constant = pyo.Var(within=pyo.NonNegativeReals, initialize=0.2)

            m.hour = pyo.Var(within=pyo.PositiveReals, initialize=0.1)
            m.y = pyo.Var(within=pyo.NonNegativeReals)

            @m.Constraint()
            def response_rule(m):
                return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.hour))

            return m

        # create the Experiment class
        class RooneyBieglerExperiment(Experiment):
            def __init__(self, experiment_number, hour, y):
                self.y = y
                self.hour = hour
                self.experiment_number = experiment_number
                self.model = None

            def get_labeled_model(self):
                if self.model is None:
                    self.create_model()
                    self.finalize_model()
                    self.label_model()
                return self.model

            def create_model(self):
                m = self.model = rooney_biegler_model()

                return m

            def finalize_model(self):
                m = self.model

                # fix the input variable
                m.hour.fix(self.hour)

                return m

            def label_model(self):
                m = self.model

                # add experiment inputs
                m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_inputs.update([(m.hour, self.hour)])

                # add experiment outputs
                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update([(m.y, self.y)])

                # add unknown parameters
                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
                )

                # add measurement error
                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update([(m.y, None)])

                return m

        # creat the experiments list
        rooney_biegler_exp_list = []
        hour_data = self.data["hour"]
        y_data = self.data["y"]
        for i in range(self.data.shape[0]):
            rooney_biegler_exp_list.append(
                RooneyBieglerExperiment(i, hour_data[i], y_data[i])
            )

        self.exp_list = rooney_biegler_exp_list

        self.objective_function = (
            "SSE"  # testing the new covariance calculations for the `SSE` objective
        )

    def check_rooney_biegler_results(self, objval, cov):
        """
        Checks if the results are equal to the expected values and agree with the results of Rooney-Biegler

        Argument:
            objval: the objective value of the annotated Pyomo model
            cov: covariance matrix of the estimated parameters
        """

        # get indices in covariance matrix
        cov_cols = cov.columns.to_list()
        asymptote_index = [idx for idx, s in enumerate(cov_cols) if "asymptote" in s][0]
        rate_constant_index = [
            idx for idx, s in enumerate(cov_cols) if "rate_constant" in s
        ][0]

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            cov.iloc[asymptote_index, asymptote_index], 6.229612, places=2
        )  # 6.22864 from paper
        self.assertAlmostEqual(
            cov.iloc[asymptote_index, rate_constant_index], -0.432265, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(
            cov.iloc[rate_constant_index, asymptote_index], -0.432265, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(
            cov.iloc[rate_constant_index, rate_constant_index], 0.041242, places=2
        )  # 0.04124 from paper

    def test_parmest_basics(self):
        """
        Calculates the parameter estimates and covariance matrix, and compares them with the results of Rooney-Biegler
        """
        pest = parmest.Estimator(self.exp_list, obj_function=self.objective_function)

        objval, thetavals = pest.theta_est()
        cov = pest.cov_est(cov_n=6, method="finite_difference")

        self.check_rooney_biegler_results(objval, cov)

    def test_cov_scipy_least_squares_comparison(self):
        """
        Scipy results differ in the 3rd decimal place from the paper. It is possible
        the paper used an alternative finite difference approximation for the Jacobian.
        """

        def model(theta, t):
            """
            Model to be fitted y = model(theta, t)
            Arguments:
                theta: vector of fitted parameters
                t: independent variable [hours]

            Returns:
                y: model predictions [need to check paper for units]
            """
            asymptote = theta[0]
            rate_constant = theta[1]

            return asymptote * (1 - np.exp(-rate_constant * t))

        def residual(theta, t, y):
            """
            Calculate residuals
            Arguments:
                theta: vector of fitted parameters
                t: independent variable [hours]
                y: dependent variable [?]
            """
            return y - model(theta, t)

        # define data
        t = self.data["hour"].to_numpy()
        y = self.data["y"].to_numpy()

        # define initial guess
        theta_guess = np.array([15, 0.5])

        ## solve with optimize.least_squares
        sol = scipy.optimize.least_squares(
            residual, theta_guess, method="trf", args=(t, y), verbose=2
        )
        theta_hat = sol.x

        self.assertAlmostEqual(
            theta_hat[0], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(theta_hat[1], 0.5311, places=2)  # 0.5311 from the paper

        # calculate residuals
        r = residual(theta_hat, t, y)

        # calculate variance of the residuals
        # -2 because there are 2 fitted parameters
        sigre = np.matmul(r.T, r / (len(y) - 2))

        # approximate covariance
        # Need to divide by 2 because optimize.least_squares scaled the objective by 1/2
        cov = sigre * np.linalg.inv(np.matmul(sol.jac.T, sol.jac))

        self.assertAlmostEqual(cov[0, 0], 6.22864, places=2)  # 6.22864 from paper
        self.assertAlmostEqual(cov[0, 1], -0.4322, places=2)  # -0.4322 from paper
        self.assertAlmostEqual(cov[1, 0], -0.4322, places=2)  # -0.4322 from paper
        self.assertAlmostEqual(cov[1, 1], 0.04124, places=2)  # 0.04124 from paper

    def test_cov_scipy_curve_fit_comparison(self):
        """
        Scipy results differ in the 3rd decimal place from the paper. It is possible
        the paper used an alternative finite difference approximation for the Jacobian.
        """

        ## solve with optimize.curve_fit
        def model(t, asymptote, rate_constant):
            return asymptote * (1 - np.exp(-rate_constant * t))

        # define data
        t = self.data["hour"].to_numpy()
        y = self.data["y"].to_numpy()

        # define initial guess
        theta_guess = np.array([15, 0.5])

        theta_hat, cov = scipy.optimize.curve_fit(model, t, y, p0=theta_guess)

        self.assertAlmostEqual(
            theta_hat[0], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(theta_hat[1], 0.5311, places=2)  # 0.5311 from the paper

        self.assertAlmostEqual(cov[0, 0], 6.22864, places=2)  # 6.22864 from paper
        self.assertAlmostEqual(cov[0, 1], -0.4322, places=2)  # -0.4322 from paper
        self.assertAlmostEqual(cov[1, 0], -0.4322, places=2)  # -0.4322 from paper
        self.assertAlmostEqual(cov[1, 1], 0.04124, places=2)  # 0.04124 from paper


if __name__ == "__main__":
    unittest.main()

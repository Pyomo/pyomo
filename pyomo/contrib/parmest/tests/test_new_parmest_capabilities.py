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

import pytest
from parameterized import parameterized, parameterized_class

import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
import pyomo.environ as pyo

from pyomo.common.dependencies import numpy as np, pandas as pd, scipy, matplotlib
from pyomo.contrib.parmest.experiment import Experiment

ipopt_available = pyo.SolverFactory("ipopt").available()

# Test class for the built-in "SSE" and "SSE_weighted" objective functions
# validated the results using the Rooney-Biegler example
# Rooney-Biegler example is the case when the measurement error is None
# we considered another case when the user supplies the value of the measurement error
@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")

# we use parameterized_class to test the two objective functions over the two cases of measurement error
# included a third objective function to test the error message when an incorrect objective function is supplied
@parameterized_class(("measurement_std", "objective_function"), [(None, "SSE"), (None, "SSE_weighted"),
                            (None, "incorrect_obj"), (0.1, "SSE"), (0.1, "SSE_weighted"), (0.1, "incorrect_obj")])
class TestRooneyBiegler(unittest.TestCase):

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
            def __init__(self, experiment_number, hour, y, measurement_error_std):
                self.y = y
                self.hour = hour
                self.experiment_number = experiment_number
                self.model = None
                self.measurement_error_std = measurement_error_std

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
                m.measurement_error.update([(m.y, self.measurement_error_std)])

                return m

        # extract the input and output variables
        hour_data = self.data["hour"]
        y_data = self.data["y"]

        # create the experiments list
        rooney_biegler_exp_list = []
        for i in range(self.data.shape[0]):
            rooney_biegler_exp_list.append(
                RooneyBieglerExperiment(i, hour_data[i], y_data[i], self.measurement_std)
            )

        self.exp_list = rooney_biegler_exp_list


    def check_rooney_biegler_parameters(self, obj_val, theta_vals, obj_function, measurement_error):
        """
        Checks if the objective value and parameter estimates are equal to the expected values
        and agree with the results of the Rooney-Biegler paper

        Argument:
            obj_val: the objective value of the annotated Pyomo model
            theta_vals: dictionary of the estimated parameters
            obj_function: a string of the objective function supplied by the user, e.g., 'SSE'
            measurement_error: float or integer value of the measurement error standard deviation
        """
        if obj_function == "SSE":
            self.assertAlmostEqual(obj_val, 4.33171, places=2)
        elif obj_function == "SSE_weighted" and measurement_error is not None:
            self.assertAlmostEqual(obj_val, 216.58556, places=2)

        self.assertAlmostEqual(
            theta_vals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            theta_vals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper


    def check_rooney_biegler_covariance(self, cov, cov_method, obj_function, measurement_error):
        """
        Checks if the covariance matrix elements are equal to the expected values
        and agree with the results of the Rooney-Biegler paper

        Argument:
            cov: pd.DataFrame, covariance matrix of the estimated parameters
            cov_method: string ``method`` object specified by the user
                        options - 'finite_difference', 'reduced_hessian', and 'automatic_differentiation_kaug'
            obj_function: a string of the objective function supplied by the user, e.g., 'SSE'
            measurement_error: float or integer value of the measurement error standard deviation
        """

        # get indices in covariance matrix
        cov_cols = cov.columns.to_list()
        asymptote_index = [idx for idx, s in enumerate(cov_cols) if "asymptote" in s][0]
        rate_constant_index = [
            idx for idx, s in enumerate(cov_cols) if "rate_constant" in s
        ][0]

        if measurement_error is None:
            if obj_function == "SSE":
                if (
                    cov_method == "finite_difference"
                    or cov_method == "automatic_differentiation_kaug"
                ):
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
                else:
                    self.assertAlmostEqual(
                        cov.iloc[asymptote_index, asymptote_index], 36.935351, places=2
                    )  # 6.22864 from paper
                    self.assertAlmostEqual(
                        cov.iloc[asymptote_index, rate_constant_index], -2.551392, places=2
                    )  # -0.4322 from paper
                    self.assertAlmostEqual(
                        cov.iloc[rate_constant_index, asymptote_index], -2.551392, places=2
                    )  # -0.4322 from paper
                    self.assertAlmostEqual(
                        cov.iloc[rate_constant_index, rate_constant_index], 0.243428, places=2
                    )  # 0.04124 from paper
        else:
            if obj_function == "SSE" or obj_function == "SSE_weighted":
                if (
                    cov_method == "finite_difference"
                    or cov_method == "automatic_differentiation_kaug"
                ):
                    self.assertAlmostEqual(
                        cov.iloc[asymptote_index, asymptote_index], 0.009588, places=4
                    )
                    self.assertAlmostEqual(
                        cov.iloc[asymptote_index, rate_constant_index], -0.000665, places=4
                    )
                    self.assertAlmostEqual(
                        cov.iloc[rate_constant_index, asymptote_index], -0.000665, places=4
                    )
                    self.assertAlmostEqual(
                        cov.iloc[rate_constant_index, rate_constant_index], 0.000063, places=4
                    )
                else:
                    self.assertAlmostEqual(
                        cov.iloc[asymptote_index, asymptote_index], 0.056845, places=4
                    )
                    self.assertAlmostEqual(
                        cov.iloc[asymptote_index, rate_constant_index], -0.003927, places=4
                    )
                    self.assertAlmostEqual(
                        cov.iloc[rate_constant_index, asymptote_index], -0.003927, places=4
                    )
                    self.assertAlmostEqual(
                        cov.iloc[rate_constant_index, rate_constant_index], 0.000375, places=4
                    )


    # test and check the covariance calculation for all the three supported methods
    # added an 'unsupported_method' to test the error message when the method supplied is not supported
    @parameterized.expand([("finite_difference"), ("automatic_differentiation_kaug"), ("reduced_hessian"),
                           ("unsupported_method")])
    def test_parmest_cov(self, cov_method):
        """
        Calculates the parameter estimates and covariance matrix and compares them with the results of Rooney-Biegler

        Argument:
            cov_method: string ``method`` object specified by the user
                        options - 'finite_difference', 'reduced_hessian', and 'automatic_differentiation_kaug'
        """
        if self.measurement_std is None:
            if self.objective_function == "SSE":
                pest = parmest.Estimator(self.exp_list, obj_function=self.objective_function)

                # estimate the parameters
                obj_val, theta_vals = pest.theta_est()

                # check the parameter estimation result
                self.check_rooney_biegler_parameters(obj_val, theta_vals, obj_function=self.objective_function,
                                                     measurement_error=self.measurement_std)

                # calculate the covariance matrix
                if cov_method in ("finite_difference", "automatic_differentiation_kaug", "reduced_hessian"):
                    cov = pest.cov_est(cov_n=6, method=cov_method)

                    # check the covariance calculation results
                    self.check_rooney_biegler_covariance(cov, cov_method, obj_function=self.objective_function,
                                                         measurement_error=self.measurement_std)
                else:
                    with pytest.raises(ValueError,
                                       match=r"Invalid method: 'unsupported_method'\. Choose from \['finite_difference', "
                                             "'automatic_differentiation_kaug', 'reduced_hessian'\]\."):
                        cov = pest.cov_est(cov_n=6, method=cov_method)
            elif self.objective_function == "SSE_weighted":
                with pytest.raises(ValueError, match='One or more values are missing from '
                                                     '"measurement_error". All values of the measurement errors are '
                                                     'required for the "SSE_weighted" objective.'):
                    pest = parmest.Estimator(self.exp_list, obj_function=self.objective_function)

                    # we expect this error when estimating the parameters
                    obj_val, theta_vals = pest.theta_est()
            else:
                with pytest.raises(ValueError, match=r"Invalid objective function: 'incorrect_obj'\. "
                                                     r"Choose from \['SSE', 'SSE_weighted'\]\."):
                    pest = parmest.Estimator(self.exp_list, obj_function=self.objective_function)
        else:
            if self.objective_function == "SSE" or self.objective_function == "SSE_weighted":
                pest = parmest.Estimator(self.exp_list, obj_function=self.objective_function)

                # estimate the parameters
                obj_val, theta_vals = pest.theta_est()

                # check the parameter estimation results
                self.check_rooney_biegler_parameters(obj_val, theta_vals, obj_function=self.objective_function,
                                                     measurement_error=self.measurement_std)

                # calculate the covariance matrix
                if cov_method in ("finite_difference", "automatic_differentiation_kaug", "reduced_hessian"):
                    cov = pest.cov_est(cov_n=6, method=cov_method)

                    # check the covariance calculation results
                    self.check_rooney_biegler_covariance(cov, cov_method, obj_function=self.objective_function,
                                                         measurement_error=self.measurement_std)
                else:
                    with pytest.raises(ValueError,
                                       match=r"Invalid method: 'unsupported_method'\. Choose from \['finite_difference', "
                                             "'automatic_differentiation_kaug', 'reduced_hessian'\]\."):
                        cov = pest.cov_est(cov_n=6, method=cov_method)
            else:
                with pytest.raises(ValueError, match="Invalid objective function: 'incorrect_obj'\. "
                                                     "Choose from \['SSE', 'SSE_weighted'\]\."):
                    pest = parmest.Estimator(self.exp_list, obj_function=self.objective_function)

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

        if self.measurement_std is None:
            # calculate variance of the residuals
            # -2 because there are 2 fitted parameters
            sigre = np.matmul(r.T, r / (len(y) - 2))

            # approximate covariance
            cov = sigre * np.linalg.inv(np.matmul(sol.jac.T, sol.jac))

            self.assertAlmostEqual(cov[0, 0], 6.22864, places=2)  # 6.22864 from paper
            self.assertAlmostEqual(cov[0, 1], -0.4322, places=2)  # -0.4322 from paper
            self.assertAlmostEqual(cov[1, 0], -0.4322, places=2)  # -0.4322 from paper
            self.assertAlmostEqual(cov[1, 1], 0.04124, places=2)  # 0.04124 from paper
        else:
            # use the user-supplied measurement error standard deviation
            sigre = self.measurement_std ** 2

            cov = sigre * np.linalg.inv(np.matmul(sol.jac.T, sol.jac))

            self.assertAlmostEqual(cov[0, 0], 0.009588, places=4)
            self.assertAlmostEqual(cov[0, 1], -0.000665, places=4)
            self.assertAlmostEqual(cov[1, 0], -0.000665, places=4)
            self.assertAlmostEqual(cov[1, 1], 0.000063, places=4)

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

        # estimate the parameters and covariance matrix
        if self.measurement_std is None:
            theta_hat, cov = scipy.optimize.curve_fit(model, t, y, p0=theta_guess)

            self.assertAlmostEqual(
                theta_hat[0], 19.1426, places=2
            )  # 19.1426 from the paper
            self.assertAlmostEqual(theta_hat[1], 0.5311, places=2)  # 0.5311 from the paper

            self.assertAlmostEqual(cov[0, 0], 6.22864, places=2)  # 6.22864 from paper
            self.assertAlmostEqual(cov[0, 1], -0.4322, places=2)  # -0.4322 from paper
            self.assertAlmostEqual(cov[1, 0], -0.4322, places=2)  # -0.4322 from paper
            self.assertAlmostEqual(cov[1, 1], 0.04124, places=2)  # 0.04124 from paper
        else:
            theta_hat, cov = scipy.optimize.curve_fit(model, t, y, p0=theta_guess, sigma=self.measurement_std,
                                                      absolute_sigma=True)

            self.assertAlmostEqual(
                theta_hat[0], 19.1426, places=2
            )  # 19.1426 from the paper
            self.assertAlmostEqual(theta_hat[1], 0.5311, places=2)  # 0.5311 from the paper

            self.assertAlmostEqual(cov[0, 0], 0.0095875, places=4)
            self.assertAlmostEqual(cov[0, 1], -0.0006653, places=4)
            self.assertAlmostEqual(cov[1, 0], -0.0006653, places=4)
            self.assertAlmostEqual(cov[1, 1], 0.00006347, places=4)


if __name__ == "__main__":
    unittest.main()

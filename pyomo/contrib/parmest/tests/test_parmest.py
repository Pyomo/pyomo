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

import sys
import os
import subprocess
from itertools import product

from pyomo.common.unittest import pytest
from parameterized import parameterized, parameterized_class
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

ipopt_available = pyo.SolverFactory("ipopt").available()
pynumero_ASL_available = AmplInterface.available()
testdir = this_file_dir()

# Set the global seed for random number generation in tests
_RANDOM_SEED_FOR_TESTING = 524


# Test class for the built-in "SSE" and "SSE_weighted" objective functions
# validated the results using the Rooney-Biegler paper example linked below
# https://doi.org/10.1002/aic.690470811
# The Rooney-Biegler paper example is the case when the measurement error is None
# we considered another case when the user supplies the value of the measurement error
@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")

# we use parameterized_class to test the two objective functions
# over the two cases of the measurement error. Included a third objective function
# to test the error message when an incorrect objective function is supplied
@parameterized_class(
    ("measurement_std", "objective_function"),
    [
        (None, "SSE"),
        (None, "SSE_weighted"),
        (None, "incorrect_obj"),
        (0.1, "SSE"),
        (0.1, "SSE_weighted"),
        (0.1, "incorrect_obj"),
    ],
)
class TestParmestCovEst(unittest.TestCase):

    def setUp(self):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
            RooneyBieglerExperiment,
        )

        self.data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        # Create an experiment list
        exp_list = []
        for i in range(self.data.shape[0]):
            exp_list.append(
                RooneyBieglerExperiment(self.data.loc[i, :], self.measurement_std)
            )

        self.exp_list = exp_list

        if self.objective_function == "incorrect_obj":
            with pytest.raises(
                ValueError,
                match=r"Invalid objective function: 'incorrect_obj'\. "
                r"Choose from: \['SSE', 'SSE_weighted'\]\.",
            ):
                self.pest = parmest.Estimator(
                    self.exp_list, obj_function=self.objective_function, tee=True
                )
        else:
            self.pest = parmest.Estimator(
                self.exp_list, obj_function=self.objective_function, tee=True
            )

    def check_rooney_biegler_parameters(
        self, obj_val, theta_vals, obj_function, measurement_error
    ):
        """
        Checks if the objective value and parameter estimates are equal to the
        expected values and agree with the results of the Rooney-Biegler paper

        Argument:
            obj_val: float or integer value of the objective function
            theta_vals: dictionary of the estimated parameters
            obj_function: string objective function supplied by the user,
                e.g., 'SSE'
            measurement_error: float or integer value of the measurement error
                standard deviation
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

    def check_rooney_biegler_covariance(
        self, cov, cov_method, obj_function, measurement_error
    ):
        """
        Checks if the covariance matrix elements are equal to the expected
        values and agree with the results of the Rooney-Biegler paper

        Argument:
            cov: pd.DataFrame, covariance matrix of the estimated parameters
            cov_method: string ``method`` object specified by the user
                Options - 'finite_difference', 'reduced_hessian',
                        and 'automatic_differentiation_kaug'
            obj_function: string objective function supplied by the user,
                e.g., 'SSE'
            measurement_error: float or integer value of the measurement error
                standard deviation
        """

        # get indices in covariance matrix
        cov_cols = cov.columns.to_list()
        asymptote_index = [idx for idx, s in enumerate(cov_cols) if "asymptote" in s][0]
        rate_constant_index = [
            idx for idx, s in enumerate(cov_cols) if "rate_constant" in s
        ][0]

        if measurement_error is None and obj_function == "SSE":
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
                    cov.iloc[rate_constant_index, rate_constant_index],
                    0.041242,
                    places=2,
                )  # 0.04124 from paper
            else:
                self.assertAlmostEqual(
                    cov.iloc[asymptote_index, asymptote_index], 6.155892, places=2
                )  # 6.22864 from paper
                self.assertAlmostEqual(
                    cov.iloc[asymptote_index, rate_constant_index], -0.425232, places=2
                )  # -0.4322 from paper
                self.assertAlmostEqual(
                    cov.iloc[rate_constant_index, asymptote_index], -0.425232, places=2
                )  # -0.4322 from paper
                self.assertAlmostEqual(
                    cov.iloc[rate_constant_index, rate_constant_index],
                    0.040571,
                    places=2,
                )  # 0.04124 from paper
        elif measurement_error is not None and obj_function in ("SSE", "SSE_weighted"):
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
                    cov.iloc[rate_constant_index, rate_constant_index],
                    0.000063,
                    places=4,
                )
            else:
                self.assertAlmostEqual(
                    cov.iloc[asymptote_index, asymptote_index], 0.009474, places=4
                )
                self.assertAlmostEqual(
                    cov.iloc[asymptote_index, rate_constant_index], -0.000654, places=4
                )
                self.assertAlmostEqual(
                    cov.iloc[rate_constant_index, asymptote_index], -0.000654, places=4
                )
                self.assertAlmostEqual(
                    cov.iloc[rate_constant_index, rate_constant_index],
                    0.000062,
                    places=4,
                )

    # test the covariance calculation of the three supported methods
    # added a 'unsupported_method' to test the error message when the method supplied
    # is not supported
    @parameterized.expand(
        [
            ("finite_difference"),
            ("automatic_differentiation_kaug"),
            ("reduced_hessian"),
            ("unsupported_method"),
        ]
    )
    def test_parmest_covariance(self, cov_method):
        """
        Estimates the parameters and covariance matrix and compares them
        with the results of the Rooney-Biegler paper

        Argument:
            cov_method: string ``method`` specified by the user
                Options - 'finite_difference', 'reduced_hessian',
                and 'automatic_differentiation_kaug'
        """
        valid_cov_methods = (
            "finite_difference",
            "automatic_differentiation_kaug",
            "reduced_hessian",
        )

        if self.measurement_std is None and self.objective_function == "SSE_weighted":
            with pytest.raises(
                ValueError,
                match='One or more values are missing from '
                '"measurement_error". All values of the measurement errors are '
                'required for the "SSE_weighted" objective.',
            ):
                # we expect this error when estimating the parameters
                obj_val, theta_vals = self.pest.theta_est()
        elif self.objective_function != "incorrect_obj":

            # estimate the parameters
            obj_val, theta_vals = self.pest.theta_est()

            # check the parameter estimation result
            self.check_rooney_biegler_parameters(
                obj_val,
                theta_vals,
                obj_function=self.objective_function,
                measurement_error=self.measurement_std,
            )

            # calculate the covariance matrix
            if cov_method in valid_cov_methods:
                cov = self.pest.cov_est(method=cov_method)

                # check the covariance calculation results
                self.check_rooney_biegler_covariance(
                    cov,
                    cov_method,
                    obj_function=self.objective_function,
                    measurement_error=self.measurement_std,
                )
            else:
                with pytest.raises(
                    ValueError,
                    match=r"Invalid method: 'unsupported_method'\. Choose from: "
                    r"\['finite_difference', "
                    r"'automatic_differentiation_kaug', "
                    r"'reduced_hessian'\]\.",
                ):
                    cov = self.pest.cov_est(method=cov_method)


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestRooneyBiegler(unittest.TestCase):
    def setUp(self):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
            RooneyBieglerExperiment,
        )

        np.random.seed(_RANDOM_SEED_FOR_TESTING)  # Set seed for reproducibility

        # Note, the data used in this test has been corrected to use
        # data.loc[5,'hour'] = 7 (instead of 6)
        data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        # Sum of squared error function
        def SSE(model):
            expr = (
                model.experiment_outputs[model.y[model.hour]] - model.y[model.hour]
            ) ** 2
            return expr

        # Create an experiment list
        exp_list = []
        for i in range(data.shape[0]):
            exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

        # Create an instance of the parmest estimator
        pest = parmest.Estimator(exp_list, obj_function=SSE)

        solver_options = {"tol": 1e-8}

        self.data = data
        self.pest = parmest.Estimator(
            exp_list, obj_function=SSE, solver_options=solver_options, tee=True
        )

    def test_custom_covariance_exception(self):
        """
        Tests the error raised when a user attempts to calculate
        the covariance matrix using a custom objective function
        """

        # estimate the parameters
        obj_val, theta_vals = self.pest.theta_est()

        # check the error raised when the user tries to calculate the
        # covariance matrix using the custom objective function
        with pytest.raises(
            ValueError,
            match=r"Invalid objective function for covariance calculation\. The "
            r"covariance matrix can only be calculated using the built-in "
            r"objective functions: \['SSE', 'SSE_weighted'\]\. Supply "
            r"the Estimator object one of these built-in objectives and "
            r"re-run the code\.",
        ):
            cov = self.pest.cov_est()

    def test_parmest_exception(self):
        """
        Test the exception raised by parmest when the "experiment_outputs"
        attribute is not defined in the model
        """
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
            RooneyBieglerExperiment,
        )

        # create an instance of the RooneyBieglerExperiment class
        # without the "experiment_outputs" attribute
        class RooneyBieglerExperimentException(RooneyBieglerExperiment):
            def label_model(self):
                m = self.model

                # add the unknown parameters
                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    (k, pyo.ComponentUID(k)) for k in [m.asymptote, m.rate_constant]
                )

        # create an experiment list
        exp_list = []
        for i in range(self.data.shape[0]):
            exp_list.append(RooneyBieglerExperimentException(self.data.loc[i, :]))

        # check the exception raised by parmest due to not defining
        # the "experiment_outputs"
        with self.assertRaises(AttributeError) as context:
            parmest.Estimator(exp_list, obj_function="SSE", tee=True)

        self.assertIn("experiment_outputs", str(context.exception))

    def test_theta_est(self):
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

    @unittest.skipIf(
        not graphics.imports_available, "parmest.graphics imports are unavailable"
    )
    def test_bootstrap(self):
        objval, thetavals = self.pest.theta_est()

        num_bootstraps = 10
        theta_est = self.pest.theta_est_bootstrap(
            num_bootstraps, return_samples=True, seed=_RANDOM_SEED_FOR_TESTING
        )

        num_samples = theta_est["samples"].apply(len)
        self.assertEqual(len(theta_est.index), 10)
        self.assertTrue(num_samples.equals(pd.Series([6] * 10)))

        del theta_est["samples"]

        # apply confidence region test
        CR = self.pest.confidence_region_test(theta_est, "MVN", [0.5, 0.75, 1.0])

        self.assertTrue(set(CR.columns) >= set([0.5, 0.75, 1.0]))
        self.assertEqual(CR[0.5].sum(), 5)
        self.assertEqual(CR[0.75].sum(), 7)
        self.assertEqual(CR[1.0].sum(), 10)  # all true

        graphics.pairwise_plot(theta_est, seed=_RANDOM_SEED_FOR_TESTING)
        graphics.pairwise_plot(theta_est, thetavals, seed=_RANDOM_SEED_FOR_TESTING)
        graphics.pairwise_plot(
            theta_est,
            thetavals,
            0.8,
            ["MVN", "KDE", "Rect"],
            seed=_RANDOM_SEED_FOR_TESTING,
        )

    @unittest.skipIf(
        not graphics.imports_available, "parmest.graphics imports are unavailable"
    )
    def test_likelihood_ratio(self):
        objval, thetavals = self.pest.theta_est()

        asym = np.arange(10, 30, 2)
        rate = np.arange(0, 1.5, 0.25)
        theta_vals = pd.DataFrame(
            list(product(asym, rate)), columns=['asymptote', 'rate_constant']
        )
        obj_at_theta = self.pest.objective_at_theta(theta_vals)

        LR = self.pest.likelihood_ratio_test(obj_at_theta, objval, [0.8, 0.9, 1.0])

        self.assertTrue(set(LR.columns) >= set([0.8, 0.9, 1.0]))
        self.assertEqual(LR[0.8].sum(), 6)
        self.assertEqual(LR[0.9].sum(), 10)
        self.assertEqual(LR[1.0].sum(), 60)  # all true

        graphics.pairwise_plot(LR, thetavals, 0.8)

    def test_leaveNout(self):
        lNo_theta = self.pest.theta_est_leaveNout(1)
        self.assertTrue(lNo_theta.shape == (6, 2))

        results = self.pest.leaveNout_bootstrap_test(
            1, None, 3, "Rect", [0.5, 1.0], seed=_RANDOM_SEED_FOR_TESTING
        )
        self.assertEqual(len(results), 6)  # 6 lNo samples
        i = 1
        samples = results[i][0]  # list of N samples that are left out
        lno_theta = results[i][1]
        bootstrap_theta = results[i][2]
        self.assertTrue(samples == [1])  # sample 1 was left out
        self.assertEqual(lno_theta.shape[0], 1)  # lno estimate for sample 1
        self.assertTrue(set(lno_theta.columns) >= set([0.5, 1.0]))
        self.assertEqual(lno_theta[1.0].sum(), 1)  # all true
        self.assertEqual(bootstrap_theta.shape[0], 3)  # bootstrap for sample 1
        self.assertEqual(bootstrap_theta[1.0].sum(), 3)  # all true

    @pytest.mark.expensive
    def test_diagnostic_mode(self):
        self.pest.diagnostic_mode = True

        objval, thetavals = self.pest.theta_est()

        asym = np.arange(10, 30, 2)
        rate = np.arange(0, 1.5, 0.25)
        theta_vals = pd.DataFrame(
            list(product(asym, rate)), columns=['asymptote', 'rate_constant']
        )

        obj_at_theta = self.pest.objective_at_theta(theta_vals)

        self.pest.diagnostic_mode = False

    @unittest.pytest.mark.mpi
    def test_parallel_parmest(self):
        """use mpiexec and mpi4py"""
        p = str(parmestbase.__path__)
        l = p.find("'")
        r = p.find("'", l + 1)
        parmestpath = p[l + 1 : r]
        rbpath = (
            parmestpath
            + os.sep
            + "examples"
            + os.sep
            + "rooney_biegler"
            + os.sep
            + "rooney_biegler.py"
        )
        rbpath = os.path.abspath(rbpath)  # paranoia strikes deep...
        rlist = ["mpiexec", "--allow-run-as-root", "-n", "2", sys.executable, rbpath]
        if sys.version_info >= (3, 5):
            ret = subprocess.run(rlist)
            retcode = ret.returncode
        else:
            retcode = subprocess.call(rlist)
        self.assertEqual(retcode, 0)

    @unittest.skipIf(not pynumero_ASL_available, "pynumero_ASL is not available")
    def test_theta_est_cov(self):
        objval, thetavals, cov = self.pest.theta_est(calc_cov=True, cov_n=6)

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

        # Covariance matrix
        self.assertAlmostEqual(
            cov["asymptote"]["asymptote"], 6.155892, places=2
        )  # 6.22864 from paper
        self.assertAlmostEqual(
            cov["asymptote"]["rate_constant"], -0.425232, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(
            cov["rate_constant"]["asymptote"], -0.425232, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(
            cov["rate_constant"]["rate_constant"], 0.040571, places=2
        )  # 0.04124 from paper

        """ Why does the covariance matrix from parmest not match the paper? Parmest is
        calculating the exact reduced Hessian. The paper (Rooney and Bielger, 2001) likely
        employed the first order approximation common for nonlinear regression. The paper
        values were verified with Scipy, which uses the same first order approximation.
        The formula used in parmest was verified against equations (7-5-15) and (7-5-16) in
        "Nonlinear Parameter Estimation", Y. Bard, 1974.
        """

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


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestModelVariants(unittest.TestCase):

    def setUp(self):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
            RooneyBieglerExperiment,
        )

        np.random.seed(_RANDOM_SEED_FOR_TESTING)  # Set seed for reproducibility
        self.data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        def rooney_biegler_params(data):
            model = pyo.ConcreteModel()

            model.asymptote = pyo.Param(initialize=15, mutable=True)
            model.rate_constant = pyo.Param(initialize=0.5, mutable=True)

            model.hour = pyo.Param(within=pyo.PositiveReals, mutable=True)
            model.y = pyo.Param(within=pyo.PositiveReals, mutable=True)

            def response_rule(m, h):
                expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            return model

        class RooneyBieglerExperimentParams(RooneyBieglerExperiment):

            def create_model(self):
                data_df = self.data.to_frame().transpose()
                self.model = rooney_biegler_params(data_df)

            def label_model(self):

                m = self.model

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update(
                    [(m.hour, self.data["hour"]), (m.y, self.data["y"])]
                )

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    (k, pyo.ComponentUID(k)) for k in [m.asymptote, m.rate_constant]
                )

        rooney_biegler_params_exp_list = []
        for i in range(self.data.shape[0]):
            rooney_biegler_params_exp_list.append(
                RooneyBieglerExperimentParams(self.data.loc[i, :])
            )

        def rooney_biegler_indexed_params(data):
            model = pyo.ConcreteModel()

            model.param_names = pyo.Set(initialize=["asymptote", "rate_constant"])
            model.theta = pyo.Param(
                model.param_names,
                initialize={"asymptote": 15, "rate_constant": 0.5},
                mutable=True,
            )

            model.hour = pyo.Param(within=pyo.PositiveReals, mutable=True)
            model.y = pyo.Param(within=pyo.PositiveReals, mutable=True)

            def response_rule(m, h):
                expr = m.theta["asymptote"] * (
                    1 - pyo.exp(-m.theta["rate_constant"] * h)
                )
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            return model

        class RooneyBieglerExperimentIndexedParams(RooneyBieglerExperiment):

            def create_model(self):
                data_df = self.data.to_frame().transpose()
                self.model = rooney_biegler_indexed_params(data_df)

            def label_model(self):

                m = self.model

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update(
                    [(m.hour, self.data["hour"]), (m.y, self.data["y"])]
                )

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update((k, pyo.ComponentUID(k)) for k in [m.theta])

        rooney_biegler_indexed_params_exp_list = []
        for i in range(self.data.shape[0]):
            rooney_biegler_indexed_params_exp_list.append(
                RooneyBieglerExperimentIndexedParams(self.data.loc[i, :])
            )

        def rooney_biegler_vars(data):
            model = pyo.ConcreteModel()

            model.asymptote = pyo.Var(initialize=15)
            model.rate_constant = pyo.Var(initialize=0.5)
            model.asymptote.fixed = True  # parmest will unfix theta variables
            model.rate_constant.fixed = True

            model.hour = pyo.Param(within=pyo.PositiveReals, mutable=True)
            model.y = pyo.Param(within=pyo.PositiveReals, mutable=True)

            def response_rule(m, h):
                expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            return model

        class RooneyBieglerExperimentVars(RooneyBieglerExperiment):

            def create_model(self):
                data_df = self.data.to_frame().transpose()
                self.model = rooney_biegler_vars(data_df)

            def label_model(self):

                m = self.model

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update(
                    [(m.hour, self.data["hour"]), (m.y, self.data["y"])]
                )

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    (k, pyo.ComponentUID(k)) for k in [m.asymptote, m.rate_constant]
                )

        rooney_biegler_vars_exp_list = []
        for i in range(self.data.shape[0]):
            rooney_biegler_vars_exp_list.append(
                RooneyBieglerExperimentVars(self.data.loc[i, :])
            )

        def rooney_biegler_indexed_vars(data):
            model = pyo.ConcreteModel()

            model.var_names = pyo.Set(initialize=["asymptote", "rate_constant"])
            model.theta = pyo.Var(
                model.var_names, initialize={"asymptote": 15, "rate_constant": 0.5}
            )
            model.theta["asymptote"].fixed = (
                True  # parmest will unfix theta variables, even when they are indexed
            )
            model.theta["rate_constant"].fixed = True

            model.hour = pyo.Param(within=pyo.PositiveReals, mutable=True)
            model.y = pyo.Param(within=pyo.PositiveReals, mutable=True)

            def response_rule(m, h):
                expr = m.theta["asymptote"] * (
                    1 - pyo.exp(-m.theta["rate_constant"] * h)
                )
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            return model

        class RooneyBieglerExperimentIndexedVars(RooneyBieglerExperiment):

            def create_model(self):
                data_df = self.data.to_frame().transpose()
                self.model = rooney_biegler_indexed_vars(data_df)

            def label_model(self):

                m = self.model

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update(
                    [(m.hour, self.data["hour"]), (m.y, self.data["y"])]
                )

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update((k, pyo.ComponentUID(k)) for k in [m.theta])

        rooney_biegler_indexed_vars_exp_list = []
        for i in range(self.data.shape[0]):
            rooney_biegler_indexed_vars_exp_list.append(
                RooneyBieglerExperimentIndexedVars(self.data.loc[i, :])
            )

        # Sum of squared error function
        def SSE(model):
            expr = (
                model.experiment_outputs[model.y]
                - model.response_function[model.experiment_outputs[model.hour]]
            ) ** 2
            return expr

        self.objective_function = SSE

        theta_vals = pd.DataFrame([20, 1], index=["asymptote", "rate_constant"]).T
        theta_vals_index = pd.DataFrame(
            [20, 1], index=["theta['asymptote']", "theta['rate_constant']"]
        ).T

        self.input = {
            "param": {
                "exp_list": rooney_biegler_params_exp_list,
                "theta_names": ["asymptote", "rate_constant"],
                "theta_vals": theta_vals,
            },
            "param_index": {
                "exp_list": rooney_biegler_indexed_params_exp_list,
                "theta_names": ["theta"],
                "theta_vals": theta_vals_index,
            },
            "vars": {
                "exp_list": rooney_biegler_vars_exp_list,
                "theta_names": ["asymptote", "rate_constant"],
                "theta_vals": theta_vals,
            },
            "vars_index": {
                "exp_list": rooney_biegler_indexed_vars_exp_list,
                "theta_names": ["theta"],
                "theta_vals": theta_vals_index,
            },
            "vars_quoted_index": {
                "exp_list": rooney_biegler_indexed_vars_exp_list,
                "theta_names": ["theta['asymptote']", "theta['rate_constant']"],
                "theta_vals": theta_vals_index,
            },
            "vars_str_index": {
                "exp_list": rooney_biegler_indexed_vars_exp_list,
                "theta_names": ["theta[asymptote]", "theta[rate_constant]"],
                "theta_vals": theta_vals_index,
            },
        }

    @unittest.skipIf(not pynumero_ASL_available, "pynumero_ASL is not available")
    def check_rooney_biegler_results(self, objval, cov):

        # get indices in covariance matrix
        cov_cols = cov.columns.to_list()
        asymptote_index = [idx for idx, s in enumerate(cov_cols) if "asymptote" in s][0]
        rate_constant_index = [
            idx for idx, s in enumerate(cov_cols) if "rate_constant" in s
        ][0]

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            cov.iloc[asymptote_index, asymptote_index], 6.30579403, places=2
        )  # 6.22864 from paper
        self.assertAlmostEqual(
            cov.iloc[asymptote_index, rate_constant_index], -0.4395341, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(
            cov.iloc[rate_constant_index, asymptote_index], -0.4395341, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(
            cov.iloc[rate_constant_index, rate_constant_index], 0.04193591, places=2
        )  # 0.04124 from paper

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics(self):

        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(
                parmest_input["exp_list"], obj_function=self.objective_function
            )

            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)
            self.check_rooney_biegler_results(objval, cov)

            obj_at_theta = pest.objective_at_theta(parmest_input["theta_vals"])
            self.assertAlmostEqual(obj_at_theta["obj"][0], 16.531953, places=2)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics_with_initialize_parmest_model_option(self):

        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(
                parmest_input["exp_list"], obj_function=self.objective_function
            )

            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)
            self.check_rooney_biegler_results(objval, cov)

            obj_at_theta = pest.objective_at_theta(
                parmest_input["theta_vals"], initialize_parmest_model=True
            )

            self.assertAlmostEqual(obj_at_theta["obj"][0], 16.531953, places=2)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics_with_square_problem_solve(self):

        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(
                parmest_input["exp_list"], obj_function=self.objective_function
            )

            obj_at_theta = pest.objective_at_theta(
                parmest_input["theta_vals"], initialize_parmest_model=True
            )

            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)
            self.check_rooney_biegler_results(objval, cov)

            self.assertAlmostEqual(obj_at_theta["obj"][0], 16.531953, places=2)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics_with_square_problem_solve_no_theta_vals(self):

        for model_type, parmest_input in self.input.items():

            pest = parmest.Estimator(
                parmest_input["exp_list"], obj_function=self.objective_function
            )

            obj_at_theta = pest.objective_at_theta(initialize_parmest_model=True)

            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)
            self.check_rooney_biegler_results(objval, cov)


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestReactorDesign(unittest.TestCase):
    def setUp(self):
        from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
            ReactorDesignExperiment,
        )

        # Data from the design
        data = pd.DataFrame(
            data=[
                [1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5],
                [1.10, 10000, 3535.1, 1064.8, 1613.3, 1893.4],
                [1.15, 10000, 3609.1, 1067.8, 1547.5, 1887.8],
                [1.20, 10000, 3680.7, 1070.0, 1486.1, 1881.6],
                [1.25, 10000, 3750.0, 1071.4, 1428.6, 1875.0],
                [1.30, 10000, 3817.1, 1072.2, 1374.6, 1868.0],
                [1.35, 10000, 3882.2, 1072.4, 1324.0, 1860.7],
                [1.40, 10000, 3945.4, 1072.1, 1276.3, 1853.1],
                [1.45, 10000, 4006.7, 1071.3, 1231.4, 1845.3],
                [1.50, 10000, 4066.4, 1070.1, 1189.0, 1837.3],
                [1.55, 10000, 4124.4, 1068.5, 1148.9, 1829.1],
                [1.60, 10000, 4180.9, 1066.5, 1111.0, 1820.8],
                [1.65, 10000, 4235.9, 1064.3, 1075.0, 1812.4],
                [1.70, 10000, 4289.5, 1061.8, 1040.9, 1803.9],
                [1.75, 10000, 4341.8, 1059.0, 1008.5, 1795.3],
                [1.80, 10000, 4392.8, 1056.0, 977.7, 1786.7],
                [1.85, 10000, 4442.6, 1052.8, 948.4, 1778.1],
                [1.90, 10000, 4491.3, 1049.4, 920.5, 1769.4],
                [1.95, 10000, 4538.8, 1045.8, 893.9, 1760.8],
            ],
            columns=["sv", "caf", "ca", "cb", "cc", "cd"],
        )

        # Create an experiment list
        exp_list = []
        for i in range(data.shape[0]):
            exp_list.append(ReactorDesignExperiment(data, i))

        solver_options = {"max_iter": 6000}

        self.pest = parmest.Estimator(
            exp_list, obj_function="SSE", solver_options=solver_options
        )

    def test_theta_est(self):
        # used in data reconciliation
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(thetavals["k1"], 5.0 / 6.0, places=4)
        self.assertAlmostEqual(thetavals["k2"], 5.0 / 3.0, places=4)
        self.assertAlmostEqual(thetavals["k3"], 1.0 / 6000.0, places=7)

    def test_return_values(self):
        objval, thetavals, data_rec = self.pest.theta_est(
            return_values=["ca", "cb", "cc", "cd", "caf"]
        )
        self.assertAlmostEqual(data_rec["cc"].loc[18], 893.84924, places=3)


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestReactorDesign_DAE(unittest.TestCase):
    # Based on a reactor example in `Chemical Reactor Analysis and Design Fundamentals`,
    # https://sites.engineering.ucsb.edu/~jbraw/chemreacfun/
    # https://sites.engineering.ucsb.edu/~jbraw/chemreacfun/fig-html/appendix/fig-A-10.html

    def setUp(self):
        def ABC_model(data):
            ca_meas = data["ca"]
            cb_meas = data["cb"]
            cc_meas = data["cc"]

            np.random.seed(_RANDOM_SEED_FOR_TESTING)  # Set seed for reproducibility

            if isinstance(data, pd.DataFrame):
                meas_t = data.index  # time index
            else:  # dictionary
                meas_t = list(ca_meas.keys())  # nested dictionary

            ca0 = 1.0
            cb0 = 0.0
            cc0 = 0.0

            m = pyo.ConcreteModel()

            m.k1 = pyo.Var(initialize=0.5, bounds=(1e-4, 10))
            m.k2 = pyo.Var(initialize=3.0, bounds=(1e-4, 10))

            m.time = dae.ContinuousSet(bounds=(0.0, 5.0), initialize=meas_t)

            # initialization and bounds
            m.ca = pyo.Var(m.time, initialize=ca0, bounds=(-1e-3, ca0 + 1e-3))
            m.cb = pyo.Var(m.time, initialize=cb0, bounds=(-1e-3, ca0 + 1e-3))
            m.cc = pyo.Var(m.time, initialize=cc0, bounds=(-1e-3, ca0 + 1e-3))

            m.dca = dae.DerivativeVar(m.ca, wrt=m.time)
            m.dcb = dae.DerivativeVar(m.cb, wrt=m.time)
            m.dcc = dae.DerivativeVar(m.cc, wrt=m.time)

            def _dcarate(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                else:
                    return m.dca[t] == -m.k1 * m.ca[t]

            m.dcarate = pyo.Constraint(m.time, rule=_dcarate)

            def _dcbrate(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                else:
                    return m.dcb[t] == m.k1 * m.ca[t] - m.k2 * m.cb[t]

            m.dcbrate = pyo.Constraint(m.time, rule=_dcbrate)

            def _dccrate(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                else:
                    return m.dcc[t] == m.k2 * m.cb[t]

            m.dccrate = pyo.Constraint(m.time, rule=_dccrate)

            def ComputeFirstStageCost_rule(m):
                return 0

            m.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

            def ComputeSecondStageCost_rule(m):
                return sum(
                    (m.ca[t] - ca_meas[t]) ** 2
                    + (m.cb[t] - cb_meas[t]) ** 2
                    + (m.cc[t] - cc_meas[t]) ** 2
                    for t in meas_t
                )

            m.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

            def total_cost_rule(model):
                return model.FirstStageCost + model.SecondStageCost

            m.Total_Cost_Objective = pyo.Objective(
                rule=total_cost_rule, sense=pyo.minimize
            )

            disc = pyo.TransformationFactory("dae.collocation")
            disc.apply_to(m, nfe=20, ncp=2)

            return m

        class ReactorDesignExperimentDAE(Experiment):

            def __init__(self, data):

                self.data = data
                self.model = None

            def create_model(self):
                self.model = ABC_model(self.data)

            def label_model(self):

                m = self.model

                if isinstance(self.data, pd.DataFrame):
                    meas_time_points = self.data.index
                else:
                    meas_time_points = list(self.data["ca"].keys())

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update(
                    (m.ca[t], self.data["ca"][t]) for t in meas_time_points
                )
                m.experiment_outputs.update(
                    (m.cb[t], self.data["cb"][t]) for t in meas_time_points
                )
                m.experiment_outputs.update(
                    (m.cc[t], self.data["cc"][t]) for t in meas_time_points
                )

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    (k, pyo.ComponentUID(k)) for k in [m.k1, m.k2]
                )

            def get_labeled_model(self):
                self.create_model()
                self.label_model()

                return self.model

        # This example tests data formatted in 3 ways
        # Each format holds 1 scenario
        # 1. dataframe with time index
        # 2. nested dictionary {ca: {t, val pairs}, ... }
        data = [
            [0.000, 0.957, -0.031, -0.015],
            [0.263, 0.557, 0.330, 0.044],
            [0.526, 0.342, 0.512, 0.156],
            [0.789, 0.224, 0.499, 0.310],
            [1.053, 0.123, 0.428, 0.454],
            [1.316, 0.079, 0.396, 0.556],
            [1.579, 0.035, 0.303, 0.651],
            [1.842, 0.029, 0.287, 0.658],
            [2.105, 0.025, 0.221, 0.750],
            [2.368, 0.017, 0.148, 0.854],
            [2.632, -0.002, 0.182, 0.845],
            [2.895, 0.009, 0.116, 0.893],
            [3.158, -0.023, 0.079, 0.942],
            [3.421, 0.006, 0.078, 0.899],
            [3.684, 0.016, 0.059, 0.942],
            [3.947, 0.014, 0.036, 0.991],
            [4.211, -0.009, 0.014, 0.988],
            [4.474, -0.030, 0.036, 0.941],
            [4.737, 0.004, 0.036, 0.971],
            [5.000, -0.024, 0.028, 0.985],
        ]
        data = pd.DataFrame(data, columns=["t", "ca", "cb", "cc"])
        data_df = data.set_index("t")
        data_dict = {
            "ca": {k: v for (k, v) in zip(data.t, data.ca)},
            "cb": {k: v for (k, v) in zip(data.t, data.cb)},
            "cc": {k: v for (k, v) in zip(data.t, data.cc)},
        }

        # Create an experiment list
        exp_list_df = [ReactorDesignExperimentDAE(data_df)]
        exp_list_dict = [ReactorDesignExperimentDAE(data_dict)]

        self.pest_df = parmest.Estimator(exp_list_df)
        self.pest_dict = parmest.Estimator(exp_list_dict)

        # Estimator object with multiple scenarios
        exp_list_df_multiple = [
            ReactorDesignExperimentDAE(data_df),
            ReactorDesignExperimentDAE(data_df),
        ]
        exp_list_dict_multiple = [
            ReactorDesignExperimentDAE(data_dict),
            ReactorDesignExperimentDAE(data_dict),
        ]

        self.pest_df_multiple = parmest.Estimator(exp_list_df_multiple)
        self.pest_dict_multiple = parmest.Estimator(exp_list_dict_multiple)

        # Create an instance of the model
        self.m_df = ABC_model(data_df)
        self.m_dict = ABC_model(data_dict)

        # create an instance of the ReactorDesignExperimentDAE class
        # without the "unknown_parameters" attribute
        class ReactorDesignExperimentException(ReactorDesignExperimentDAE):
            def label_model(self):

                m = self.model

                if isinstance(self.data, pd.DataFrame):
                    meas_time_points = self.data.index
                else:
                    meas_time_points = list(self.data["ca"].keys())

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update(
                    (m.ca[t], self.data["ca"][t]) for t in meas_time_points
                )
                m.experiment_outputs.update(
                    (m.cb[t], self.data["cb"][t]) for t in meas_time_points
                )
                m.experiment_outputs.update(
                    (m.cc[t], self.data["cc"][t]) for t in meas_time_points
                )

        # create an experiment list without the "unknown_parameters" attribute
        exp_list_df_no_params = [ReactorDesignExperimentException(data_df)]
        exp_list_dict_no_params = [ReactorDesignExperimentException(data_dict)]

        self.exp_list_df_no_params = exp_list_df_no_params
        self.exp_list_dict_no_params = exp_list_dict_no_params

    def test_parmest_exception(self):
        """
        Test the exception raised by parmest when the "unknown_parameters"
        attribute is not defined in the model
        """
        with self.assertRaises(AttributeError) as context:
            parmest.Estimator(self.exp_list_df_no_params, obj_function="SSE")

        self.assertIn("unknown_parameters", str(context.exception))

        with self.assertRaises(AttributeError) as context:
            parmest.Estimator(self.exp_list_dict_no_params, obj_function="SSE")

        self.assertIn("unknown_parameters", str(context.exception))

    def test_dataformats(self):
        obj1, theta1 = self.pest_df.theta_est()
        obj2, theta2 = self.pest_dict.theta_est()

        self.assertAlmostEqual(obj1, obj2, places=6)
        self.assertAlmostEqual(theta1["k1"], theta2["k1"], places=6)
        self.assertAlmostEqual(theta1["k2"], theta2["k2"], places=6)

    def test_return_continuous_set(self):
        """
        test if ContinuousSet elements are returned correctly from theta_est()
        """
        obj1, theta1, return_vals1 = self.pest_df.theta_est(return_values=["time"])
        obj2, theta2, return_vals2 = self.pest_dict.theta_est(return_values=["time"])
        self.assertAlmostEqual(return_vals1["time"].loc[0][18], 2.368, places=3)
        self.assertAlmostEqual(return_vals2["time"].loc[0][18], 2.368, places=3)

    def test_return_continuous_set_multiple_datasets(self):
        """
        test if ContinuousSet elements are returned correctly from theta_est()
        """
        obj1, theta1, return_vals1 = self.pest_df_multiple.theta_est(
            return_values=["time"]
        )
        obj2, theta2, return_vals2 = self.pest_dict_multiple.theta_est(
            return_values=["time"]
        )
        self.assertAlmostEqual(return_vals1["time"].loc[1][18], 2.368, places=3)
        self.assertAlmostEqual(return_vals2["time"].loc[1][18], 2.368, places=3)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_covariance(self):
        from pyomo.contrib.interior_point.inverse_reduced_hessian import (
            inv_reduced_hessian_barrier,
        )

        # Number of datapoints.
        # 3 data components (ca, cb, cc), 20 timesteps, 1 scenario = 60
        # In this example, this is the number of data points in data_df, but that's
        # only because the data is indexed by time and contains no additional information.
        n = 60

        # Compute covariance using parmest
        obj, theta, cov = self.pest_df.theta_est(calc_cov=True, cov_n=n)

        # Compute covariance using interior_point
        vars_list = [self.m_df.k1, self.m_df.k2]
        solve_result, inv_red_hes = inv_reduced_hessian_barrier(
            self.m_df, independent_variables=vars_list, tee=True
        )
        l = len(vars_list)
        cov_interior_point = 2 * obj / (n - l) * inv_red_hes
        cov_interior_point = pd.DataFrame(
            cov_interior_point, ["k1", "k2"], ["k1", "k2"]
        )

        cov_diff = (cov - cov_interior_point).abs().sum().sum()

        self.assertTrue(cov.loc["k1", "k1"] > 0)
        self.assertTrue(cov.loc["k2", "k2"] > 0)
        self.assertAlmostEqual(cov_diff, 0, places=6)


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestSquareInitialization_RooneyBiegler(unittest.TestCase):
    def setUp(self):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler_with_constraint import (
            RooneyBieglerExperiment,
        )

        # Note, the data used in this test has been corrected to use data.loc[5,'hour'] = 7 (instead of 6)
        data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        # Sum of squared error function
        def SSE(model):
            expr = (
                model.experiment_outputs[model.y]
                - model.response_function[model.experiment_outputs[model.hour]]
            ) ** 2
            return expr

        exp_list = []
        for i in range(data.shape[0]):
            exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

        solver_options = {"tol": 1e-8}

        self.data = data
        self.pest = parmest.Estimator(
            exp_list, obj_function=SSE, solver_options=solver_options, tee=True
        )

    def test_theta_est_with_square_initialization(self):
        obj_init = self.pest.objective_at_theta(initialize_parmest_model=True)
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

    def test_theta_est_with_square_initialization_and_custom_init_theta(self):
        theta_vals_init = pd.DataFrame(
            data=[[19.0, 0.5]], columns=["asymptote", "rate_constant"]
        )
        obj_init = self.pest.objective_at_theta(
            theta_values=theta_vals_init, initialize_parmest_model=True
        )
        objval, thetavals = self.pest.theta_est()
        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

    def test_theta_est_with_square_initialization_diagnostic_mode_true(self):
        self.pest.diagnostic_mode = True
        obj_init = self.pest.objective_at_theta(initialize_parmest_model=True)
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

        self.pest.diagnostic_mode = False


###########################
# tests for deprecated UI #
###########################


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestRooneyBieglerDeprecated(unittest.TestCase):
    def setUp(self):

        def rooney_biegler_model(data):
            model = pyo.ConcreteModel()

            model.asymptote = pyo.Var(initialize=15)
            model.rate_constant = pyo.Var(initialize=0.5)

            def response_rule(m, h):
                expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            def SSE_rule(m):
                return sum(
                    (data.y[i] - m.response_function[data.hour[i]]) ** 2
                    for i in data.index
                )

            model.SSE = pyo.Objective(rule=SSE_rule, sense=pyo.minimize)

            return model

        # Note, the data used in this test has been corrected to use data.loc[5,'hour'] = 7 (instead of 6)
        data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        theta_names = ["asymptote", "rate_constant"]

        def SSE(model, data):
            expr = sum(
                (data.y[i] - model.response_function[data.hour[i]]) ** 2
                for i in data.index
            )
            return expr

        solver_options = {"tol": 1e-8}

        self.data = data
        self.pest = parmest.Estimator(
            rooney_biegler_model,
            data,
            theta_names,
            SSE,
            solver_options=solver_options,
            tee=True,
        )

    def test_theta_est(self):
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

    @unittest.skipIf(
        not graphics.imports_available, "parmest.graphics imports are unavailable"
    )
    def test_bootstrap(self):
        objval, thetavals = self.pest.theta_est()

        num_bootstraps = 10
        theta_est = self.pest.theta_est_bootstrap(num_bootstraps, return_samples=True)

        num_samples = theta_est["samples"].apply(len)
        self.assertTrue(len(theta_est.index), 10)
        self.assertTrue(num_samples.equals(pd.Series([6] * 10)))

        del theta_est["samples"]

        # apply confidence region test
        CR = self.pest.confidence_region_test(theta_est, "MVN", [0.5, 0.75, 1.0])

        self.assertTrue(set(CR.columns) >= set([0.5, 0.75, 1.0]))
        self.assertTrue(CR[0.5].sum() == 5)
        self.assertTrue(CR[0.75].sum() == 7)
        self.assertTrue(CR[1.0].sum() == 10)  # all true

        graphics.pairwise_plot(theta_est)
        graphics.pairwise_plot(theta_est, thetavals)
        graphics.pairwise_plot(theta_est, thetavals, 0.8, ["MVN", "KDE", "Rect"])

    @unittest.skipIf(
        not graphics.imports_available, "parmest.graphics imports are unavailable"
    )
    def test_likelihood_ratio(self):
        objval, thetavals = self.pest.theta_est()

        asym = np.arange(10, 30, 2)
        rate = np.arange(0, 1.5, 0.25)
        theta_vals = pd.DataFrame(
            list(product(asym, rate)), columns=self.pest._return_theta_names()
        )

        obj_at_theta = self.pest.objective_at_theta(theta_vals)

        LR = self.pest.likelihood_ratio_test(obj_at_theta, objval, [0.8, 0.9, 1.0])

        self.assertTrue(set(LR.columns) >= set([0.8, 0.9, 1.0]))
        self.assertTrue(LR[0.8].sum() == 6)
        self.assertTrue(LR[0.9].sum() == 10)
        self.assertTrue(LR[1.0].sum() == 60)  # all true

        graphics.pairwise_plot(LR, thetavals, 0.8)

    def test_leaveNout(self):
        lNo_theta = self.pest.theta_est_leaveNout(1)
        self.assertTrue(lNo_theta.shape == (6, 2))

        results = self.pest.leaveNout_bootstrap_test(
            1, None, 3, "Rect", [0.5, 1.0], seed=_RANDOM_SEED_FOR_TESTING
        )
        self.assertTrue(len(results) == 6)  # 6 lNo samples
        i = 1
        samples = results[i][0]  # list of N samples that are left out
        lno_theta = results[i][1]
        bootstrap_theta = results[i][2]
        self.assertTrue(samples == [1])  # sample 1 was left out
        self.assertTrue(lno_theta.shape[0] == 1)  # lno estimate for sample 1
        self.assertTrue(set(lno_theta.columns) >= set([0.5, 1.0]))
        self.assertTrue(lno_theta[1.0].sum() == 1)  # all true
        self.assertTrue(bootstrap_theta.shape[0] == 3)  # bootstrap for sample 1
        self.assertTrue(bootstrap_theta[1.0].sum() == 3)  # all true

    def test_diagnostic_mode(self):
        self.pest.diagnostic_mode = True

        objval, thetavals = self.pest.theta_est()

        asym = np.arange(10, 30, 2)
        rate = np.arange(0, 1.5, 0.25)
        theta_vals = pd.DataFrame(
            list(product(asym, rate)), columns=self.pest._return_theta_names()
        )

        obj_at_theta = self.pest.objective_at_theta(theta_vals)

        self.pest.diagnostic_mode = False

    @unittest.pytest.mark.mpi
    def test_parallel_parmest(self):
        """use mpiexec and mpi4py"""
        p = str(parmestbase.__path__)
        l = p.find("'")
        r = p.find("'", l + 1)
        parmestpath = p[l + 1 : r]
        rbpath = (
            parmestpath
            + os.sep
            + "examples"
            + os.sep
            + "rooney_biegler"
            + os.sep
            + "rooney_biegler.py"
        )
        rbpath = os.path.abspath(rbpath)  # paranoia strikes deep...
        rlist = ["mpiexec", "--allow-run-as-root", "-n", "2", sys.executable, rbpath]
        if sys.version_info >= (3, 5):
            ret = subprocess.run(rlist)
            retcode = ret.returncode
        else:
            retcode = subprocess.call(rlist)
        assert retcode == 0

    @unittest.skipIf(not pynumero_ASL_available, "pynumero_ASL is not available")
    def test_theta_est_cov(self):
        objval, thetavals, cov = self.pest.theta_est(calc_cov=True, cov_n=6)

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

        # Covariance matrix
        self.assertAlmostEqual(
            cov.iloc[0, 0], 6.30579403, places=2
        )  # 6.22864 from paper
        self.assertAlmostEqual(
            cov.iloc[0, 1], -0.4395341, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(
            cov.iloc[1, 0], -0.4395341, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(cov.iloc[1, 1], 0.04124, places=2)  # 0.04124 from paper

        """ Why does the covariance matrix from parmest not match the paper? Parmest is
        calculating the exact reduced Hessian. The paper (Rooney and Bielger, 2001) likely
        employed the first order approximation common for nonlinear regression. The paper
        values were verified with Scipy, which uses the same first order approximation.
        The formula used in parmest was verified against equations (7-5-15) and (7-5-16) in
        "Nonlinear Parameter Estimation", Y. Bard, 1974.
        """

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


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestModelVariantsDeprecated(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        def rooney_biegler_params(data):
            model = pyo.ConcreteModel()

            model.asymptote = pyo.Param(initialize=15, mutable=True)
            model.rate_constant = pyo.Param(initialize=0.5, mutable=True)

            def response_rule(m, h):
                expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            return model

        def rooney_biegler_indexed_params(data):
            model = pyo.ConcreteModel()

            model.param_names = pyo.Set(initialize=["asymptote", "rate_constant"])
            model.theta = pyo.Param(
                model.param_names,
                initialize={"asymptote": 15, "rate_constant": 0.5},
                mutable=True,
            )

            def response_rule(m, h):
                expr = m.theta["asymptote"] * (
                    1 - pyo.exp(-m.theta["rate_constant"] * h)
                )
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            return model

        def rooney_biegler_vars(data):
            model = pyo.ConcreteModel()

            model.asymptote = pyo.Var(initialize=15)
            model.rate_constant = pyo.Var(initialize=0.5)
            model.asymptote.fixed = True  # parmest will unfix theta variables
            model.rate_constant.fixed = True

            def response_rule(m, h):
                expr = m.asymptote * (1 - pyo.exp(-m.rate_constant * h))
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            return model

        def rooney_biegler_indexed_vars(data):
            model = pyo.ConcreteModel()

            model.var_names = pyo.Set(initialize=["asymptote", "rate_constant"])
            model.theta = pyo.Var(
                model.var_names, initialize={"asymptote": 15, "rate_constant": 0.5}
            )
            model.theta["asymptote"].fixed = (
                True  # parmest will unfix theta variables, even when they are indexed
            )
            model.theta["rate_constant"].fixed = True

            def response_rule(m, h):
                expr = m.theta["asymptote"] * (
                    1 - pyo.exp(-m.theta["rate_constant"] * h)
                )
                return expr

            model.response_function = pyo.Expression(data.hour, rule=response_rule)

            return model

        def SSE(model, data):
            expr = sum(
                (data.y[i] - model.response_function[data.hour[i]]) ** 2
                for i in data.index
            )
            return expr

        self.objective_function = SSE

        theta_vals = pd.DataFrame([20, 1], index=["asymptote", "rate_constant"]).T
        theta_vals_index = pd.DataFrame(
            [20, 1], index=["theta['asymptote']", "theta['rate_constant']"]
        ).T

        self.input = {
            "param": {
                "model": rooney_biegler_params,
                "theta_names": ["asymptote", "rate_constant"],
                "theta_vals": theta_vals,
            },
            "param_index": {
                "model": rooney_biegler_indexed_params,
                "theta_names": ["theta"],
                "theta_vals": theta_vals_index,
            },
            "vars": {
                "model": rooney_biegler_vars,
                "theta_names": ["asymptote", "rate_constant"],
                "theta_vals": theta_vals,
            },
            "vars_index": {
                "model": rooney_biegler_indexed_vars,
                "theta_names": ["theta"],
                "theta_vals": theta_vals_index,
            },
            "vars_quoted_index": {
                "model": rooney_biegler_indexed_vars,
                "theta_names": ["theta['asymptote']", "theta['rate_constant']"],
                "theta_vals": theta_vals_index,
            },
            "vars_str_index": {
                "model": rooney_biegler_indexed_vars,
                "theta_names": ["theta[asymptote]", "theta[rate_constant]"],
                "theta_vals": theta_vals_index,
            },
        }

    @unittest.skipIf(not pynumero_ASL_available, "pynumero_ASL is not available")
    def test_parmest_basics(self):
        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(
                parmest_input["model"],
                self.data,
                parmest_input["theta_names"],
                self.objective_function,
            )

            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)

            self.assertAlmostEqual(objval, 4.3317112, places=2)
            self.assertAlmostEqual(
                cov.iloc[0, 0], 6.30579403, places=2
            )  # 6.22864 from paper
            self.assertAlmostEqual(
                cov.iloc[0, 1], -0.4395341, places=2
            )  # -0.4322 from paper
            self.assertAlmostEqual(
                cov.iloc[1, 0], -0.4395341, places=2
            )  # -0.4322 from paper
            self.assertAlmostEqual(
                cov.iloc[1, 1], 0.04193591, places=2
            )  # 0.04124 from paper

            obj_at_theta = pest.objective_at_theta(parmest_input["theta_vals"])
            self.assertAlmostEqual(obj_at_theta["obj"][0], 16.531953, places=2)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics_with_initialize_parmest_model_option(self):
        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(
                parmest_input["model"],
                self.data,
                parmest_input["theta_names"],
                self.objective_function,
            )

            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)

            self.assertAlmostEqual(objval, 4.3317112, places=2)
            self.assertAlmostEqual(
                cov.iloc[0, 0], 6.30579403, places=2
            )  # 6.22864 from paper
            self.assertAlmostEqual(
                cov.iloc[0, 1], -0.4395341, places=2
            )  # -0.4322 from paper
            self.assertAlmostEqual(
                cov.iloc[1, 0], -0.4395341, places=2
            )  # -0.4322 from paper
            self.assertAlmostEqual(
                cov.iloc[1, 1], 0.04193591, places=2
            )  # 0.04124 from paper

            obj_at_theta = pest.objective_at_theta(
                parmest_input["theta_vals"], initialize_parmest_model=True
            )

            self.assertAlmostEqual(obj_at_theta["obj"][0], 16.531953, places=2)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics_with_square_problem_solve(self):
        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(
                parmest_input["model"],
                self.data,
                parmest_input["theta_names"],
                self.objective_function,
            )

            obj_at_theta = pest.objective_at_theta(
                parmest_input["theta_vals"], initialize_parmest_model=True
            )

            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)

            self.assertAlmostEqual(objval, 4.3317112, places=2)
            self.assertAlmostEqual(
                cov.iloc[0, 0], 6.30579403, places=2
            )  # 6.22864 from paper
            self.assertAlmostEqual(
                cov.iloc[0, 1], -0.4395341, places=2
            )  # -0.4322 from paper
            self.assertAlmostEqual(
                cov.iloc[1, 0], -0.4395341, places=2
            )  # -0.4322 from paper
            self.assertAlmostEqual(
                cov.iloc[1, 1], 0.04193591, places=2
            )  # 0.04124 from paper

            self.assertAlmostEqual(obj_at_theta["obj"][0], 16.531953, places=2)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics_with_square_problem_solve_no_theta_vals(self):
        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(
                parmest_input["model"],
                self.data,
                parmest_input["theta_names"],
                self.objective_function,
            )

            obj_at_theta = pest.objective_at_theta(initialize_parmest_model=True)

            objval, thetavals, cov = pest.theta_est(calc_cov=True, cov_n=6)

            self.assertAlmostEqual(objval, 4.3317112, places=2)
            self.assertAlmostEqual(
                cov.iloc[0, 0], 6.30579403, places=2
            )  # 6.22864 from paper
            self.assertAlmostEqual(
                cov.iloc[0, 1], -0.4395341, places=2
            )  # -0.4322 from paper
            self.assertAlmostEqual(
                cov.iloc[1, 0], -0.4395341, places=2
            )  # -0.4322 from paper
            self.assertAlmostEqual(
                cov.iloc[1, 1], 0.04193591, places=2
            )  # 0.04124 from paper


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestReactorDesignDeprecated(unittest.TestCase):
    def setUp(self):

        def reactor_design_model(data):
            # Create the concrete model
            model = pyo.ConcreteModel()

            # Rate constants
            model.k1 = pyo.Param(
                initialize=5.0 / 6.0, within=pyo.PositiveReals, mutable=True
            )  # min^-1
            model.k2 = pyo.Param(
                initialize=5.0 / 3.0, within=pyo.PositiveReals, mutable=True
            )  # min^-1
            model.k3 = pyo.Param(
                initialize=1.0 / 6000.0, within=pyo.PositiveReals, mutable=True
            )  # m^3/(gmol min)

            # Inlet concentration of A, gmol/m^3
            if isinstance(data, dict) or isinstance(data, pd.Series):
                model.caf = pyo.Param(
                    initialize=float(data["caf"]), within=pyo.PositiveReals
                )
            elif isinstance(data, pd.DataFrame):
                model.caf = pyo.Param(
                    initialize=float(data.iloc[0]["caf"]), within=pyo.PositiveReals
                )
            else:
                raise ValueError("Unrecognized data type.")

            # Space velocity (flowrate/volume)
            if isinstance(data, dict) or isinstance(data, pd.Series):
                model.sv = pyo.Param(
                    initialize=float(data["sv"]), within=pyo.PositiveReals
                )
            elif isinstance(data, pd.DataFrame):
                model.sv = pyo.Param(
                    initialize=float(data.iloc[0]["sv"]), within=pyo.PositiveReals
                )
            else:
                raise ValueError("Unrecognized data type.")

            # Outlet concentration of each component
            model.ca = pyo.Var(initialize=5000.0, within=pyo.PositiveReals)
            model.cb = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
            model.cc = pyo.Var(initialize=2000.0, within=pyo.PositiveReals)
            model.cd = pyo.Var(initialize=1000.0, within=pyo.PositiveReals)

            # Objective
            model.obj = pyo.Objective(expr=model.cb, sense=pyo.maximize)

            # Constraints
            model.ca_bal = pyo.Constraint(
                expr=(
                    0
                    == model.sv * model.caf
                    - model.sv * model.ca
                    - model.k1 * model.ca
                    - 2.0 * model.k3 * model.ca**2.0
                )
            )

            model.cb_bal = pyo.Constraint(
                expr=(
                    0
                    == -model.sv * model.cb + model.k1 * model.ca - model.k2 * model.cb
                )
            )

            model.cc_bal = pyo.Constraint(
                expr=(0 == -model.sv * model.cc + model.k2 * model.cb)
            )

            model.cd_bal = pyo.Constraint(
                expr=(0 == -model.sv * model.cd + model.k3 * model.ca**2.0)
            )

            return model

        # Data from the design
        data = pd.DataFrame(
            data=[
                [1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5],
                [1.10, 10000, 3535.1, 1064.8, 1613.3, 1893.4],
                [1.15, 10000, 3609.1, 1067.8, 1547.5, 1887.8],
                [1.20, 10000, 3680.7, 1070.0, 1486.1, 1881.6],
                [1.25, 10000, 3750.0, 1071.4, 1428.6, 1875.0],
                [1.30, 10000, 3817.1, 1072.2, 1374.6, 1868.0],
                [1.35, 10000, 3882.2, 1072.4, 1324.0, 1860.7],
                [1.40, 10000, 3945.4, 1072.1, 1276.3, 1853.1],
                [1.45, 10000, 4006.7, 1071.3, 1231.4, 1845.3],
                [1.50, 10000, 4066.4, 1070.1, 1189.0, 1837.3],
                [1.55, 10000, 4124.4, 1068.5, 1148.9, 1829.1],
                [1.60, 10000, 4180.9, 1066.5, 1111.0, 1820.8],
                [1.65, 10000, 4235.9, 1064.3, 1075.0, 1812.4],
                [1.70, 10000, 4289.5, 1061.8, 1040.9, 1803.9],
                [1.75, 10000, 4341.8, 1059.0, 1008.5, 1795.3],
                [1.80, 10000, 4392.8, 1056.0, 977.7, 1786.7],
                [1.85, 10000, 4442.6, 1052.8, 948.4, 1778.1],
                [1.90, 10000, 4491.3, 1049.4, 920.5, 1769.4],
                [1.95, 10000, 4538.8, 1045.8, 893.9, 1760.8],
            ],
            columns=["sv", "caf", "ca", "cb", "cc", "cd"],
        )

        theta_names = ["k1", "k2", "k3"]

        def SSE(model, data):
            expr = (
                (float(data.iloc[0]["ca"]) - model.ca) ** 2
                + (float(data.iloc[0]["cb"]) - model.cb) ** 2
                + (float(data.iloc[0]["cc"]) - model.cc) ** 2
                + (float(data.iloc[0]["cd"]) - model.cd) ** 2
            )
            return expr

        solver_options = {"max_iter": 6000}

        self.pest = parmest.Estimator(
            reactor_design_model, data, theta_names, SSE, solver_options=solver_options
        )

    def test_theta_est(self):
        # used in data reconciliation
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(thetavals["k1"], 5.0 / 6.0, places=4)
        self.assertAlmostEqual(thetavals["k2"], 5.0 / 3.0, places=4)
        self.assertAlmostEqual(thetavals["k3"], 1.0 / 6000.0, places=7)

    def test_return_values(self):
        objval, thetavals, data_rec = self.pest.theta_est(
            return_values=["ca", "cb", "cc", "cd", "caf"]
        )
        self.assertAlmostEqual(data_rec["cc"].loc[18], 893.84924, places=3)


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestReactorDesign_DAE_Deprecated(unittest.TestCase):
    # Based on a reactor example in `Chemical Reactor Analysis and Design Fundamentals`,
    # https://sites.engineering.ucsb.edu/~jbraw/chemreacfun/
    # https://sites.engineering.ucsb.edu/~jbraw/chemreacfun/fig-html/appendix/fig-A-10.html

    def setUp(self):
        def ABC_model(data):
            ca_meas = data["ca"]
            cb_meas = data["cb"]
            cc_meas = data["cc"]

            if isinstance(data, pd.DataFrame):
                meas_t = data.index  # time index
            else:  # dictionary
                meas_t = list(ca_meas.keys())  # nested dictionary

            ca0 = 1.0
            cb0 = 0.0
            cc0 = 0.0

            m = pyo.ConcreteModel()

            m.k1 = pyo.Var(initialize=0.5, bounds=(1e-4, 10))
            m.k2 = pyo.Var(initialize=3.0, bounds=(1e-4, 10))

            m.time = dae.ContinuousSet(bounds=(0.0, 5.0), initialize=meas_t)

            # initialization and bounds
            m.ca = pyo.Var(m.time, initialize=ca0, bounds=(-1e-3, ca0 + 1e-3))
            m.cb = pyo.Var(m.time, initialize=cb0, bounds=(-1e-3, ca0 + 1e-3))
            m.cc = pyo.Var(m.time, initialize=cc0, bounds=(-1e-3, ca0 + 1e-3))

            m.dca = dae.DerivativeVar(m.ca, wrt=m.time)
            m.dcb = dae.DerivativeVar(m.cb, wrt=m.time)
            m.dcc = dae.DerivativeVar(m.cc, wrt=m.time)

            def _dcarate(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                else:
                    return m.dca[t] == -m.k1 * m.ca[t]

            m.dcarate = pyo.Constraint(m.time, rule=_dcarate)

            def _dcbrate(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                else:
                    return m.dcb[t] == m.k1 * m.ca[t] - m.k2 * m.cb[t]

            m.dcbrate = pyo.Constraint(m.time, rule=_dcbrate)

            def _dccrate(m, t):
                if t == 0:
                    return pyo.Constraint.Skip
                else:
                    return m.dcc[t] == m.k2 * m.cb[t]

            m.dccrate = pyo.Constraint(m.time, rule=_dccrate)

            def ComputeFirstStageCost_rule(m):
                return 0

            m.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

            def ComputeSecondStageCost_rule(m):
                return sum(
                    (m.ca[t] - ca_meas[t]) ** 2
                    + (m.cb[t] - cb_meas[t]) ** 2
                    + (m.cc[t] - cc_meas[t]) ** 2
                    for t in meas_t
                )

            m.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

            def total_cost_rule(model):
                return model.FirstStageCost + model.SecondStageCost

            m.Total_Cost_Objective = pyo.Objective(
                rule=total_cost_rule, sense=pyo.minimize
            )

            disc = pyo.TransformationFactory("dae.collocation")
            disc.apply_to(m, nfe=20, ncp=2)

            return m

        # This example tests data formatted in 3 ways
        # Each format holds 1 scenario
        # 1. dataframe with time index
        # 2. nested dictionary {ca: {t, val pairs}, ... }
        data = [
            [0.000, 0.957, -0.031, -0.015],
            [0.263, 0.557, 0.330, 0.044],
            [0.526, 0.342, 0.512, 0.156],
            [0.789, 0.224, 0.499, 0.310],
            [1.053, 0.123, 0.428, 0.454],
            [1.316, 0.079, 0.396, 0.556],
            [1.579, 0.035, 0.303, 0.651],
            [1.842, 0.029, 0.287, 0.658],
            [2.105, 0.025, 0.221, 0.750],
            [2.368, 0.017, 0.148, 0.854],
            [2.632, -0.002, 0.182, 0.845],
            [2.895, 0.009, 0.116, 0.893],
            [3.158, -0.023, 0.079, 0.942],
            [3.421, 0.006, 0.078, 0.899],
            [3.684, 0.016, 0.059, 0.942],
            [3.947, 0.014, 0.036, 0.991],
            [4.211, -0.009, 0.014, 0.988],
            [4.474, -0.030, 0.036, 0.941],
            [4.737, 0.004, 0.036, 0.971],
            [5.000, -0.024, 0.028, 0.985],
        ]
        data = pd.DataFrame(data, columns=["t", "ca", "cb", "cc"])
        data_df = data.set_index("t")
        data_dict = {
            "ca": {k: v for (k, v) in zip(data.t, data.ca)},
            "cb": {k: v for (k, v) in zip(data.t, data.cb)},
            "cc": {k: v for (k, v) in zip(data.t, data.cc)},
        }

        theta_names = ["k1", "k2"]

        self.pest_df = parmest.Estimator(ABC_model, [data_df], theta_names)
        self.pest_dict = parmest.Estimator(ABC_model, [data_dict], theta_names)

        # Estimator object with multiple scenarios
        self.pest_df_multiple = parmest.Estimator(
            ABC_model, [data_df, data_df], theta_names
        )
        self.pest_dict_multiple = parmest.Estimator(
            ABC_model, [data_dict, data_dict], theta_names
        )

        # Create an instance of the model
        self.m_df = ABC_model(data_df)
        self.m_dict = ABC_model(data_dict)

    def test_dataformats(self):
        obj1, theta1 = self.pest_df.theta_est()
        obj2, theta2 = self.pest_dict.theta_est()

        self.assertAlmostEqual(obj1, obj2, places=6)
        self.assertAlmostEqual(theta1["k1"], theta2["k1"], places=6)
        self.assertAlmostEqual(theta1["k2"], theta2["k2"], places=6)

    def test_return_continuous_set(self):
        """
        test if ContinuousSet elements are returned correctly from theta_est()
        """
        obj1, theta1, return_vals1 = self.pest_df.theta_est(return_values=["time"])
        obj2, theta2, return_vals2 = self.pest_dict.theta_est(return_values=["time"])
        self.assertAlmostEqual(return_vals1["time"].loc[0][18], 2.368, places=3)
        self.assertAlmostEqual(return_vals2["time"].loc[0][18], 2.368, places=3)

    def test_return_continuous_set_multiple_datasets(self):
        """
        test if ContinuousSet elements are returned correctly from theta_est()
        """
        obj1, theta1, return_vals1 = self.pest_df_multiple.theta_est(
            return_values=["time"]
        )
        obj2, theta2, return_vals2 = self.pest_dict_multiple.theta_est(
            return_values=["time"]
        )
        self.assertAlmostEqual(return_vals1["time"].loc[1][18], 2.368, places=3)
        self.assertAlmostEqual(return_vals2["time"].loc[1][18], 2.368, places=3)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_covariance(self):
        from pyomo.contrib.interior_point.inverse_reduced_hessian import (
            inv_reduced_hessian_barrier,
        )

        # Number of datapoints.
        # 3 data components (ca, cb, cc), 20 timesteps, 1 scenario = 60
        # In this example, this is the number of data points in data_df, but that's
        # only because the data is indexed by time and contains no additional information.
        n = 60

        # Compute covariance using parmest
        obj, theta, cov = self.pest_df.theta_est(calc_cov=True, cov_n=n)

        # Compute covariance using interior_point
        vars_list = [self.m_df.k1, self.m_df.k2]
        solve_result, inv_red_hes = inv_reduced_hessian_barrier(
            self.m_df, independent_variables=vars_list, tee=True
        )
        l = len(vars_list)
        cov_interior_point = 2 * obj / (n - l) * inv_red_hes
        cov_interior_point = pd.DataFrame(
            cov_interior_point, ["k1", "k2"], ["k1", "k2"]
        )

        cov_diff = (cov - cov_interior_point).abs().sum().sum()

        self.assertTrue(cov.loc["k1", "k1"] > 0)
        self.assertTrue(cov.loc["k2", "k2"] > 0)
        self.assertAlmostEqual(cov_diff, 0, places=6)


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestSquareInitialization_RooneyBiegler_Deprecated(unittest.TestCase):
    def setUp(self):

        def rooney_biegler_model_with_constraint(data):
            model = pyo.ConcreteModel()

            model.asymptote = pyo.Var(initialize=15)
            model.rate_constant = pyo.Var(initialize=0.5)
            model.response_function = pyo.Var(data.hour, initialize=0.0)

            # changed from expression to constraint
            def response_rule(m, h):
                return m.response_function[h] == m.asymptote * (
                    1 - pyo.exp(-m.rate_constant * h)
                )

            model.response_function_constraint = pyo.Constraint(
                data.hour, rule=response_rule
            )

            def SSE_rule(m):
                return sum(
                    (data.y[i] - m.response_function[data.hour[i]]) ** 2
                    for i in data.index
                )

            model.SSE = pyo.Objective(rule=SSE_rule, sense=pyo.minimize)

            return model

        # Note, the data used in this test has been corrected to use data.loc[5,'hour'] = 7 (instead of 6)
        data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        theta_names = ["asymptote", "rate_constant"]

        def SSE(model, data):
            expr = sum(
                (data.y[i] - model.response_function[data.hour[i]]) ** 2
                for i in data.index
            )
            return expr

        solver_options = {"tol": 1e-8}

        self.data = data
        self.pest = parmest.Estimator(
            rooney_biegler_model_with_constraint,
            data,
            theta_names,
            SSE,
            solver_options=solver_options,
            tee=True,
        )

    def test_theta_est_with_square_initialization(self):
        obj_init = self.pest.objective_at_theta(initialize_parmest_model=True)
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

    def test_theta_est_with_square_initialization_and_custom_init_theta(self):
        theta_vals_init = pd.DataFrame(
            data=[[19.0, 0.5]], columns=["asymptote", "rate_constant"]
        )
        obj_init = self.pest.objective_at_theta(
            theta_values=theta_vals_init, initialize_parmest_model=True
        )
        objval, thetavals = self.pest.theta_est()
        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

    def test_theta_est_with_square_initialization_diagnostic_mode_true(self):
        self.pest.diagnostic_mode = True
        obj_init = self.pest.objective_at_theta(initialize_parmest_model=True)
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(
            thetavals["asymptote"], 19.1426, places=2
        )  # 19.1426 from the paper
        self.assertAlmostEqual(
            thetavals["rate_constant"], 0.5311, places=2
        )  # 0.5311 from the paper

        self.pest.diagnostic_mode = False


if __name__ == "__main__":
    unittest.main()

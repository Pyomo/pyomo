# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

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
import io
import logging

from pyomo.common.dependencies import numpy as np, pandas as pd, scipy, matplotlib
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
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
            expr = (model.experiment_outputs[model.y] - model.y) ** 2
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

    def test_k_aug_solver_exception(self):
        """
        Tests the error message raised when a user passes
        the solver option as "k_aug"
        """

        # estimate the parameters
        with pytest.raises(RuntimeError, match=r"k_aug no longer supported."):
            obj_val, theta_vals = self.pest.theta_est(solver="k_aug")

    def test_unknown_solver_exception(self):
        """
        Tests the error message raised when a user passes an
        unsupported solver option
        """

        # estimate the parameters
        with pytest.raises(RuntimeError, match=r"Unknown solver in Q_Opt=random"):
            obj_val, theta_vals = self.pest.theta_est(solver="random")

    def test_exp_outputs_exception(self):
        """
        Tests the exception raised by parmest when the "experiment_outputs"
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

        # Updated models to use Vars for experiment output, and Constraints
        def rooney_biegler_params(data):
            model = pyo.ConcreteModel()

            model.asymptote = pyo.Param(initialize=15, mutable=True)
            model.rate_constant = pyo.Param(initialize=0.5, mutable=True)

            # Add the experiment inputs
            model.h = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))

            # Fix the experiment inputs
            model.h.fix()

            # Add experiment outputs
            model.y = pyo.Var(initialize=data['y'].iloc[0], within=pyo.PositiveReals)

            # Define the model equations
            def response_rule(m):
                return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.h))

            model.response_con = pyo.Constraint(rule=response_rule)

            return model

        class RooneyBieglerExperimentParams(RooneyBieglerExperiment):

            def create_model(self):
                data_df = self.data.to_frame().transpose()
                self.model = rooney_biegler_params(data_df)

            def label_model(self):

                m = self.model

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update([(m.y, self.data["y"])])

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    (k, pyo.ComponentUID(k)) for k in [m.asymptote, m.rate_constant]
                )
                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update([(m.y, None)])

        rooney_biegler_params_exp_list = []
        for i in range(self.data.shape[0]):
            rooney_biegler_params_exp_list.append(
                RooneyBieglerExperimentParams(self.data.loc[i, :])
            )

        def rooney_biegler_indexed_params(data):
            model = pyo.ConcreteModel()

            # Define the indexed parameters
            model.param_names = pyo.Set(initialize=["asymptote", "rate_constant"])
            model.theta = pyo.Param(
                model.param_names,
                initialize={"asymptote": 15, "rate_constant": 0.5},
                mutable=True,
            )
            # Add the experiment inputs
            model.h = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))

            # Fix the experiment inputs
            model.h.fix()

            # Add experiment outputs
            model.y = pyo.Var(initialize=data['y'].iloc[0], within=pyo.PositiveReals)

            # Define the model equations
            def response_rule(m):
                return m.y == m.theta["asymptote"] * (
                    1 - pyo.exp(-m.theta["rate_constant"] * m.h)
                )

            # Add the model equations to the model
            model.response_con = pyo.Constraint(rule=response_rule)
            return model

        class RooneyBieglerExperimentIndexedParams(RooneyBieglerExperiment):

            def create_model(self):
                data_df = self.data.to_frame().transpose()
                self.model = rooney_biegler_indexed_params(data_df)

            def label_model(self):

                m = self.model

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update([(m.y, self.data["y"])])

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update((k, pyo.ComponentUID(k)) for k in [m.theta])

                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update([(m.y, None)])

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

            # Add the experiment inputs
            model.h = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))

            # Fix the experiment inputs
            model.h.fix()

            # Add experiment outputs
            model.y = pyo.Var(initialize=data['y'].iloc[0], within=pyo.PositiveReals)

            # Define the model equations
            def response_rule(m):
                return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.h))

            model.response_con = pyo.Constraint(rule=response_rule)

            return model

        class RooneyBieglerExperimentVars(RooneyBieglerExperiment):

            def create_model(self):
                data_df = self.data.to_frame().transpose()
                self.model = rooney_biegler_vars(data_df)

            def label_model(self):

                m = self.model

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update([(m.y, self.data["y"])])

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    (k, pyo.ComponentUID(k)) for k in [m.asymptote, m.rate_constant]
                )
                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update([(m.y, None)])

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

            # Add the experiment inputs
            model.h = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))

            # Fix the experiment inputs
            model.h.fix()

            # Add experiment outputs
            model.y = pyo.Var(initialize=data['y'].iloc[0], within=pyo.PositiveReals)

            # Define the model equations
            def response_rule(m):
                return m.y == m.theta["asymptote"] * (
                    1 - pyo.exp(-m.theta["rate_constant"] * m.h)
                )

            model.response_con = pyo.Constraint(rule=response_rule)

            return model

        class RooneyBieglerExperimentIndexedVars(RooneyBieglerExperiment):

            def create_model(self):
                data_df = self.data.to_frame().transpose()
                self.model = rooney_biegler_indexed_vars(data_df)

            def label_model(self):

                m = self.model

                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update([(m.y, self.data["y"])])

                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update((k, pyo.ComponentUID(k)) for k in [m.theta])

                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update([(m.y, None)])

        rooney_biegler_indexed_vars_exp_list = []
        for i in range(self.data.shape[0]):
            rooney_biegler_indexed_vars_exp_list.append(
                RooneyBieglerExperimentIndexedVars(self.data.loc[i, :])
            )

        self.objective_function = 'SSE'

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
            cov.iloc[asymptote_index, asymptote_index], 6.155892, places=2
        )  # 6.22864 from paper
        self.assertAlmostEqual(
            cov.iloc[asymptote_index, rate_constant_index], -0.425232, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(
            cov.iloc[rate_constant_index, asymptote_index], -0.425232, places=2
        )  # -0.4322 from paper
        self.assertAlmostEqual(
            cov.iloc[rate_constant_index, rate_constant_index], 0.040571, places=2
        )  # 0.04124 from paper

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics(self):

        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(
                parmest_input["exp_list"], obj_function=self.objective_function
            )
            # estimate the parameters and covariance matrix
            objval, thetavals = pest.theta_est()
            # For covariance, using reduced_hessian method since finite difference
            # and automatic differentiation may differ from paper results in the
            # 3rd decimal place, likely due to differences in finite difference
            # approximation of the Jacobian
            cov = pest.cov_est(method="reduced_hessian")
            self.check_rooney_biegler_results(objval, cov)

            obj_at_theta = pest.objective_at_theta(parmest_input["theta_vals"])
            self.assertAlmostEqual(obj_at_theta["obj"][0], 16.531953, places=2)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics_with_initialize_parmest_model_option(self):

        for model_type, parmest_input in self.input.items():
            pest = parmest.Estimator(
                parmest_input["exp_list"], obj_function=self.objective_function
            )

            objval, thetavals = pest.theta_est()
            cov = pest.cov_est(method="reduced_hessian")
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

            objval, thetavals = pest.theta_est()
            cov = pest.cov_est(method="reduced_hessian")
            self.check_rooney_biegler_results(objval, cov)

            self.assertAlmostEqual(obj_at_theta["obj"][0], 16.531953, places=2)

    @unittest.skipUnless(pynumero_ASL_available, 'pynumero_ASL is not available')
    def test_parmest_basics_with_square_problem_solve_no_theta_vals(self):

        for model_type, parmest_input in self.input.items():

            pest = parmest.Estimator(
                parmest_input["exp_list"], obj_function=self.objective_function
            )

            obj_at_theta = pest.objective_at_theta(initialize_parmest_model=True)

            objval, thetavals = pest.theta_est()
            cov = pest.cov_est(method="reduced_hessian")
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

            # Model objective component names adjusted to prevent reserved name error.
            m.FirstStage = pyo.Expression(rule=ComputeFirstStageCost_rule)

            def ComputeSecondStageCost_rule(m):
                return sum(
                    (m.ca[t] - ca_meas[t]) ** 2
                    + (m.cb[t] - cb_meas[t]) ** 2
                    + (m.cc[t] - cc_meas[t]) ** 2
                    for t in meas_t
                )

            m.SecondStage = pyo.Expression(rule=ComputeSecondStageCost_rule)

            def total_cost_rule(model):
                return model.FirstStage + model.SecondStage

            m.Total_Cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

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
                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update((m.ca[t], None) for t in meas_time_points)
                m.measurement_error.update((m.cb[t], None) for t in meas_time_points)
                m.measurement_error.update((m.cc[t], None) for t in meas_time_points)

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

        self.pest_df = parmest.Estimator(exp_list_df, obj_function="SSE")
        self.pest_dict = parmest.Estimator(exp_list_dict, obj_function="SSE")

        # Estimator object with multiple scenarios
        exp_list_df_multiple = [
            ReactorDesignExperimentDAE(data_df),
            ReactorDesignExperimentDAE(data_df),
        ]
        exp_list_dict_multiple = [
            ReactorDesignExperimentDAE(data_dict),
            ReactorDesignExperimentDAE(data_dict),
        ]

        self.pest_df_multiple = parmest.Estimator(
            exp_list_df_multiple, obj_function="SSE"
        )
        self.pest_dict_multiple = parmest.Estimator(
            exp_list_dict_multiple, obj_function="SSE"
        )

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

    def test_unknown_parameters_exception(self):
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
        # In this example, there are 20 time points and 1 experiment = 20 data points
        # The data is indexed by time, so we do not consider the number of experimental
        # outputs.
        n = 20

        # Compute covariance using parmest
        obj, theta = self.pest_df.theta_est()
        cov = self.pest_df.cov_est(method="reduced_hessian")

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
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
            RooneyBieglerExperiment,
        )

        # Note, the data used in this test has been corrected to use
        # data.loc[5,'hour'] = 7 (instead of 6)
        data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

        # Sum of squared error function
        def SSE(model):
            expr = (model.experiment_outputs[model.y] - model.y) ** 2

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


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest regularization: required dependencies are missing",
)
class TestRegularizationCore(unittest.TestCase):
    # These tests intentionally use a tiny linear model so each expected
    # regularization term can be computed analytically and reviewed quickly.
    class LinearExperiment(Experiment):
        def __init__(self, x, y, theta0_init=0.0, theta1_init=0.0):
            self.x = float(x)
            self.y = float(y)
            self.theta0_init = float(theta0_init)
            self.theta1_init = float(theta1_init)
            super().__init__(model=None)
            self.create_model()
            self.label_model()

        def create_model(self):
            m = pyo.ConcreteModel()
            m.theta0 = pyo.Var(initialize=self.theta0_init)
            m.theta1 = pyo.Var(initialize=self.theta1_init)
            m.x = pyo.Param(initialize=self.x, mutable=True)
            m.pred = pyo.Var(initialize=self.y)
            m.pred_link = pyo.Constraint(expr=m.pred == m.theta0 + m.theta1 * m.x)
            self.model = m

        def label_model(self):
            m = self.model
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_outputs.update([(m.pred, self.y)])
            m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.unknown_parameters.update(
                (k, pyo.ComponentUID(k)) for k in [m.theta0, m.theta1]
            )
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error.update([(m.pred, None)])

    class DummyExperiment(Experiment):
        def __init__(self):
            m = pyo.ConcreteModel()
            m.theta0 = pyo.Var(initialize=0.0)
            m.theta1 = pyo.Var(initialize=0.0)
            m.pred = pyo.Var(initialize=0.0)
            m.pred_link = pyo.Constraint(expr=m.pred == m.theta0 + m.theta1)
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_outputs.update([(m.pred, 0.0)])
            m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.unknown_parameters.update(
                (k, pyo.ComponentUID(k)) for k in [m.theta0, m.theta1]
            )
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error.update([(m.pred, None)])
            super().__init__(model=m)

    @staticmethod
    def _make_var_labeled_model(y_obs=5.0):
        # Var-based helper used for direct objective-expression checks.
        m = pyo.ConcreteModel()
        m.theta0 = pyo.Var(initialize=0.0)
        m.theta1 = pyo.Var(initialize=0.0)
        m.pred = pyo.Expression(expr=m.theta0 + 2.0 * m.theta1)
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.pred, float(y_obs))])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.ComponentUID(k)) for k in [m.theta0, m.theta1]
        )
        return m

    @staticmethod
    def _obj_at_theta(pest, theta0, theta1):
        # Evaluate objective for a single theta row to keep assertions explicit.
        theta = pd.DataFrame([[theta0, theta1]], columns=["theta0", "theta1"])
        return pest.objective_at_theta(theta_values=theta).iloc[0]["obj"]

    def test_l2_objective_value_matches_manual_quadratic(self):
        m = self._make_var_labeled_model(y_obs=5.0)
        m.theta0.set_value(4.0)
        m.theta1.set_value(-1.0)

        prior_fim = pd.DataFrame(
            [[2.0, 0.0], [0.0, 4.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )
        theta_ref = {"theta0": 1.0, "theta1": 2.0}
        weight = 3.0

        expr = parmest.L2_regularized_objective(
            m,
            prior_FIM=prior_fim,
            theta_ref=theta_ref,
            regularization_weight=weight,
            obj_function=parmest.SSE,
        )

        sse_expected = 9.0
        l2_expected = 54.0
        expected = sse_expected + weight * l2_expected

        self.assertAlmostEqual(pyo.value(expr), expected)

    @unittest.skipUnless(ipopt_available, "Test requires ipopt")
    def test_l2_penalty_not_double_counted_across_scenarios(self):
        # Confirms regularization is applied once at the estimator level,
        # not once per scenario.
        exp_list = [self.LinearExperiment(1.0, 1.0), self.LinearExperiment(2.0, 2.0)]
        prior_fim = pd.DataFrame(
            [[0.0, 0.0], [0.0, 2.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )
        theta_ref = {"theta0": 0.0, "theta1": 0.0}

        pest = parmest.Estimator(
            exp_list,
            obj_function="SSE",
            regularization="L2",
            prior_FIM=prior_fim,
            theta_ref=theta_ref,
            regularization_weight=1.0,
        )

        obj_val = self._obj_at_theta(pest, 0.0, 1.0)
        self.assertAlmostEqual(obj_val, 2.0)

    def test_regularization_requires_explicit_option_when_prior_supplied(self):
        # Guardrail: passing prior/FIM arguments without selecting a
        # regularization mode should fail fast.
        exp_list = [self.LinearExperiment(1.0, 1.0)]
        prior_fim = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        with pytest.raises(
            ValueError, match="regularization must be set when supplying prior_FIM"
        ):
            parmest.Estimator(exp_list, obj_function="SSE", prior_FIM=prior_fim)

    def test_l2_regularization_requires_prior_fim(self):
        exp_list = [self.LinearExperiment(1.0, 1.0)]

        with pytest.raises(ValueError, match="prior_FIM must be provided"):
            parmest.Estimator(exp_list, obj_function="SSE", regularization="L2")

    def test_user_specified_unsupported_regularization_raises(self):
        exp_list = [self.LinearExperiment(1.0, 1.0)]

        with pytest.raises(TypeError, match="regularization must be None or one of"):
            parmest.Estimator(
                exp_list, obj_function="SSE", regularization=lambda m: m.theta0**2
            )

    @unittest.skipUnless(ipopt_available, "Test requires ipopt")
    def test_l2_lambda_zero_matches_unregularized_objective(self):
        exp_list = [self.LinearExperiment(1.0, 1.0), self.LinearExperiment(2.0, 2.0)]
        prior_fim = pd.DataFrame(
            [[2.0, 0.0], [0.0, 1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )
        theta_ref = {"theta0": 0.0, "theta1": 0.0}

        pest_base = parmest.Estimator(exp_list, obj_function="SSE")
        pest_l2_zero = parmest.Estimator(
            exp_list,
            obj_function="SSE",
            regularization="L2",
            prior_FIM=prior_fim,
            theta_ref=theta_ref,
            regularization_weight=0.0,
        )

        for theta0, theta1 in [(0.0, 0.0), (0.5, 1.5), (-1.0, 2.0)]:
            obj_base = self._obj_at_theta(pest_base, theta0, theta1)
            obj_l2_zero = self._obj_at_theta(pest_l2_zero, theta0, theta1)
            self.assertAlmostEqual(obj_l2_zero, obj_base)

    @unittest.skipUnless(ipopt_available, "Test requires ipopt")
    def test_prior_subset_penalizes_only_selected_parameter(self):
        # Prior indexed only by theta1 should leave theta0 unpenalized.
        exp_list = [self.LinearExperiment(1.0, 1.0)]
        prior_fim = pd.DataFrame([[4.0]], index=["theta1"], columns=["theta1"])

        pest_base = parmest.Estimator(exp_list, obj_function="SSE")
        pest_l2 = parmest.Estimator(
            exp_list,
            obj_function="SSE",
            regularization="L2",
            prior_FIM=prior_fim,
            theta_ref={"theta1": 0.0},
            regularization_weight=1.0,
        )

        obj_base_theta0 = self._obj_at_theta(pest_base, theta0=1.0, theta1=0.0)
        obj_l2_theta0 = self._obj_at_theta(pest_l2, theta0=1.0, theta1=0.0)
        self.assertAlmostEqual(obj_l2_theta0, obj_base_theta0)

        obj_base_theta1 = self._obj_at_theta(pest_base, theta0=0.0, theta1=1.0)
        obj_l2_theta1 = self._obj_at_theta(pest_l2, theta0=0.0, theta1=1.0)
        expected_penalty = 4.0 * (1.0**2)
        self.assertAlmostEqual(obj_l2_theta1 - obj_base_theta1, expected_penalty)

    def test_negative_regularization_weight_raises(self):
        exp_list = [self.LinearExperiment(1.0, 1.0)]
        prior_fim = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        with pytest.raises(
            ValueError, match="regularization_weight must be nonnegative"
        ):
            parmest.Estimator(
                exp_list,
                obj_function="SSE",
                regularization="L2",
                prior_FIM=prior_fim,
                regularization_weight=-1.0,
            )

    def test_non_dict_theta_ref_raises_type_error(self):
        exp_list = [self.LinearExperiment(1.0, 1.0)]
        prior_fim = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        with pytest.raises(
            TypeError,
            match="theta_ref must be a dict mapping parameter names to reference values.",
        ):
            parmest.Estimator(
                exp_list,
                obj_function="SSE",
                regularization="L2",
                prior_FIM=prior_fim,
                theta_ref=pd.Series({"theta0": 0.0, "theta1": 0.0}),
                regularization_weight=1.0,
            )

    def test_missing_theta_ref_entries_raise_clear_error(self):
        exp_list = [self.LinearExperiment(1.0, 1.0)]
        prior_fim = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        pest = parmest.Estimator(
            exp_list,
            obj_function="SSE",
            regularization="L2",
            prior_FIM=prior_fim,
            theta_ref={"theta0": 0.0},
            regularization_weight=1.0,
        )

        with pytest.raises(
            ValueError, match=r"theta_ref is missing values for parameter\(s\): theta1"
        ):
            _ = self._obj_at_theta(pest, theta0=0.0, theta1=0.0)

    def test_non_psd_prior_fim_rejected(self):
        exp_list = [self.LinearExperiment(1.0, 1.0)]
        non_psd = pd.DataFrame(
            [[1.0, 2.0], [2.0, -1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        with pytest.raises(ValueError, match="positive semi-definite"):
            parmest.Estimator(
                exp_list, obj_function="SSE", regularization="L2", prior_FIM=non_psd
            )

    def test_prior_fim_must_be_dataframe(self):
        with pytest.raises(TypeError, match="prior_FIM must be a pandas DataFrame."):
            parmest._validate_prior_FIM([[1.0, 0.0], [0.0, 1.0]])

    def test_prior_fim_row_and_column_labels_must_match(self):
        prior_fim = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta2"],
        )

        with pytest.raises(
            ValueError,
            match="prior_FIM row and column labels must match the same parameter names.",
        ):
            parmest._validate_prior_FIM(prior_fim)

    def test_prior_fim_entries_must_be_numeric(self):
        prior_fim = pd.DataFrame(
            [["a", 0.0], [0.0, "b"]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        with pytest.raises(TypeError, match="prior_FIM entries must be numeric."):
            parmest._validate_prior_FIM(prior_fim)

    def test_prior_fim_entries_must_be_finite(self):
        prior_fim = pd.DataFrame(
            [[1.0, np.nan], [np.nan, 1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        with pytest.raises(ValueError, match="prior_FIM entries must be finite."):
            parmest._validate_prior_FIM(prior_fim)

    def test_prior_fim_must_be_symmetric(self):
        prior_fim = pd.DataFrame(
            [[1.0, 2.0], [0.0, 1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        with pytest.raises(ValueError, match="prior_FIM must be symmetric."):
            parmest._validate_prior_FIM(prior_fim)

    def test_prior_fim_can_skip_psd_check(self):
        prior_fim = pd.DataFrame(
            [[1.0, 2.0], [2.0, -1.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        parmest._validate_prior_FIM(prior_fim, require_psd=False)

    def test_l2_penalty_with_missing_theta_ref_uses_model_reference(self):
        m = self._make_var_labeled_model(y_obs=5.0)
        m.theta0.set_value(4.0)
        m.theta1.set_value(-1.0)

        prior_fim = pd.DataFrame(
            [[2.0, 0.0], [0.0, 4.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )

        with self.assertLogs(parmest.logger.name, level="INFO") as logs:
            penalty = parmest._calculate_L2_penalty(
                m, prior_FIM=prior_fim, theta_ref=None
            )

        self.assertAlmostEqual(pyo.value(penalty), 0.0)
        self.assertTrue(
            any(
                "theta_ref is None. Using initialized parameter values as reference."
                in msg
                for msg in logs.output
            )
        )

    def test_l2_penalty_returns_zero_when_prior_has_no_matching_parameters(self):
        m = self._make_var_labeled_model(y_obs=5.0)
        prior_fim = pd.DataFrame([[1.0]], index=["alpha"], columns=["alpha"])

        with self.assertLogs(parmest.logger.name, level="WARNING") as logs:
            penalty = parmest._calculate_L2_penalty(
                m, prior_FIM=prior_fim, theta_ref=None
            )

        self.assertEqual(penalty, 0.0)
        self.assertTrue(
            any(
                "No matching parameters found between Model and Prior FIM" in msg
                for msg in logs.output
            )
        )

    def test_compute_covariance_matrix_adds_twice_weighted_prior_fim_for_sse(self):
        exp_list = [self.DummyExperiment()]

        data_fim = np.array([[5.0, 1.0], [1.0, 4.0]])

        def fake_finite_difference_FIM(*args, **kwargs):
            return data_fim

        prior_fim = pd.DataFrame(
            [[2.0, 0.5], [0.5, 3.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )
        theta_vals = {"theta0": 0.0, "theta1": 0.0}
        regularization_weight = 0.25

        original_finite_difference_FIM = parmest._finite_difference_FIM
        try:
            parmest._finite_difference_FIM = fake_finite_difference_FIM
            cov = parmest.compute_covariance_matrix(
                experiment_list=exp_list,
                method=parmest.CovarianceMethod.finite_difference.value,
                obj_function=parmest.SSE,
                theta_vals=theta_vals,
                step=1e-3,
                solver="ipopt",
                tee=False,
                prior_FIM=prior_fim,
                regularization_weight=regularization_weight,
            )
        finally:
            parmest._finite_difference_FIM = original_finite_difference_FIM

        expected_fim = data_fim + 2.0 * regularization_weight * prior_fim.values
        expected_cov = np.linalg.inv(expected_fim)

        np.testing.assert_allclose(
            cov.loc[["theta0", "theta1"], ["theta0", "theta1"]].values,
            expected_cov,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_compute_covariance_matrix_reorders_prior_fim_by_parameter_name(self):
        exp_list = [self.DummyExperiment()]

        data_fim = np.array([[5.0, 1.0], [1.0, 4.0]])

        def fake_finite_difference_FIM(*args, **kwargs):
            return data_fim

        prior_fim = pd.DataFrame(
            [[3.0, 0.5], [0.5, 2.0]],
            index=["theta1", "theta0"],
            columns=["theta1", "theta0"],
        )
        theta_vals = {"theta0": 0.0, "theta1": 0.0}
        regularization_weight = 0.25

        original_finite_difference_FIM = parmest._finite_difference_FIM
        try:
            parmest._finite_difference_FIM = fake_finite_difference_FIM
            cov = parmest.compute_covariance_matrix(
                experiment_list=exp_list,
                method=parmest.CovarianceMethod.finite_difference.value,
                obj_function=parmest.SSE_weighted,
                theta_vals=theta_vals,
                step=1e-3,
                solver="ipopt",
                tee=False,
                prior_FIM=prior_fim,
                regularization_weight=regularization_weight,
            )
        finally:
            parmest._finite_difference_FIM = original_finite_difference_FIM

        expected_prior_fim = np.array([[2.0, 0.5], [0.5, 3.0]])
        expected_fim = data_fim + regularization_weight * expected_prior_fim
        expected_cov = np.linalg.inv(expected_fim)

        np.testing.assert_allclose(
            cov.loc[["theta0", "theta1"], ["theta0", "theta1"]].values,
            expected_cov,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_compute_covariance_matrix_adds_once_weighted_prior_fim_for_sse_weighted(
        self,
    ):
        exp_list = [self.DummyExperiment()]

        data_fim = np.array([[5.0, 1.0], [1.0, 4.0]])

        def fake_finite_difference_FIM(*args, **kwargs):
            return data_fim

        prior_fim = pd.DataFrame(
            [[2.0, 0.5], [0.5, 3.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )
        theta_vals = {"theta0": 0.0, "theta1": 0.0}
        regularization_weight = 0.25

        original_finite_difference_FIM = parmest._finite_difference_FIM
        try:
            parmest._finite_difference_FIM = fake_finite_difference_FIM
            cov = parmest.compute_covariance_matrix(
                experiment_list=exp_list,
                method=parmest.CovarianceMethod.finite_difference.value,
                obj_function=parmest.SSE_weighted,
                theta_vals=theta_vals,
                step=1e-3,
                solver="ipopt",
                tee=False,
                prior_FIM=prior_fim,
                regularization_weight=regularization_weight,
            )
        finally:
            parmest._finite_difference_FIM = original_finite_difference_FIM

        expected_fim = data_fim + regularization_weight * prior_fim.values
        expected_cov = np.linalg.inv(expected_fim)

        np.testing.assert_allclose(
            cov.loc[["theta0", "theta1"], ["theta0", "theta1"]].values,
            expected_cov,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_l2_weighted_objective_applies_half_regularization_factor(self):
        m = self._make_var_labeled_model(y_obs=5.0)
        m.theta0.set_value(4.0)
        m.theta1.set_value(-1.0)

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.pred, 1.0)])

        prior_fim = pd.DataFrame(
            [[2.0, 0.0], [0.0, 4.0]],
            index=["theta0", "theta1"],
            columns=["theta0", "theta1"],
        )
        theta_ref = {"theta0": 1.0, "theta1": 2.0}
        weight = 3.0

        expr = parmest.L2_regularized_objective(
            m,
            prior_FIM=prior_fim,
            theta_ref=theta_ref,
            regularization_weight=weight,
            obj_function=parmest.SSE_weighted,
        )

        # pred = 4 + 2*(-1) = 2, residual = 5 - 2 = 3
        # WSSE = 0.5 * 3**2 = 4.5
        # raw L2 = [3, -3]^T diag(2, 4) [3, -3] = 54
        # weighted objective should use 0.5 * raw L2
        expected = 4.5 + weight * 0.5 * 54.0

        self.assertAlmostEqual(pyo.value(expr), expected)

    @unittest.skipUnless(ipopt_available, "Test requires ipopt")
    def test_indexed_unknown_parameters_regularization_uses_scalar_theta_names(self):
        exp_list = [IndexedThetaExperiment(2.0, 5.0)]
        prior_fim = pd.DataFrame(
            [[3.0, 0.0], [0.0, 4.0]],
            index=["theta[a]", "theta[b]"],
            columns=["theta[a]", "theta[b]"],
        )
        theta_ref = {"theta[a]": 0.0, "theta[b]": 1.0}

        pest = parmest.Estimator(
            exp_list,
            obj_function="SSE",
            regularization="L2",
            prior_FIM=prior_fim,
            theta_ref=theta_ref,
        )

        theta_values = pd.DataFrame([[1.0, 2.0]], columns=["theta[a]", "theta[b]"])
        obj_val = pest.objective_at_theta(theta_values=theta_values).iloc[0]["obj"]

        self.assertAlmostEqual(obj_val, 7.0)


class LinearThetaExperiment(Experiment):
    def __init__(self, x, y, include_second_output=False):
        self.x_data = x
        self.y_data = y
        self.include_second_output = include_second_output
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()

        m.theta = pyo.Var(initialize=0.0, bounds=(-10.0, 10.0))
        m.x = pyo.Param(initialize=float(self.x_data), mutable=False)
        m.y = pyo.Var(initialize=float(self.y_data))

        m.y_link = pyo.Constraint(expr=m.y == m.theta + m.x)

        if self.include_second_output:
            m.z = pyo.Var(initialize=2.0 * float(self.y_data))
            m.z_link = pyo.Constraint(expr=m.z == 2.0 * m.theta + m.x)

        self.model = m

    def label_model(self):
        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, float(self.y_data))])

        if self.include_second_output:
            m.experiment_outputs.update([(m.z, float(2.0 * self.y_data))])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

        if self.include_second_output:
            m.measurement_error.update([(m.z, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


class IndexedThetaExperiment(Experiment):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()

        m.theta_index = pyo.Set(initialize=["a", "b"])
        m.theta = pyo.Var(
            m.theta_index, initialize={"a": 0.0, "b": 1.0}, bounds=(-10.0, 10.0)
        )

        m.x = pyo.Param(initialize=float(self.x_data), mutable=False)
        m.y = pyo.Var(initialize=float(self.y_data))

        m.y_link = pyo.Constraint(expr=m.y == m.theta["a"] + m.theta["b"] * m.x)

        self.model = m

    def label_model(self):
        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, float(self.y_data))])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


class BoundedLinearThetaExperiment(Experiment):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()

        m.theta = pyo.Var(initialize=0.0, bounds=(-10.0, 10.0))
        m.x = pyo.Param(initialize=float(self.x_data), mutable=False)
        m.y = pyo.Var(initialize=float(self.y_data))

        m.y_link = pyo.Constraint(expr=m.y == m.theta + m.x)

        # This allows fixed-theta tests to create a real infeasible model.
        # For example, theta=2 and x=1 implies y=3, violating y <= 2.
        m.upper_limit = pyo.Constraint(expr=m.y <= 2.0)

        self.model = m

    def label_model(self):
        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, float(self.y_data))])

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


class IndexedOutputExperiment(Experiment):
    def __init__(self, y_points, z_points):
        self.y_points = list(y_points)
        self.z_points = list(z_points)
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()

        m.theta = pyo.Var(initialize=0.0, bounds=(-10.0, 10.0))

        m.y_index = pyo.Set(dimen=2, ordered=True, initialize=self.y_points)
        m.z_index = pyo.Set(dimen=2, ordered=True, initialize=self.z_points)

        m.y = pyo.Var(m.y_index, initialize=0.0)
        m.z = pyo.Var(m.z_index, initialize=0.0)

        self.model = m

    def label_model(self):
        m = self.model

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        m.experiment_outputs.update(
            (m.y[idx], float(i)) for i, idx in enumerate(self.y_points, start=1)
        )

        m.experiment_outputs.update(
            (m.z[idx], float(i)) for i, idx in enumerate(self.z_points, start=1)
        )

        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])

        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        m.measurement_error.update((m.y[idx], None) for idx in self.y_points)
        m.measurement_error.update((m.z[idx], None) for idx in self.z_points)

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


def _build_estimator(data, include_second_output=False):
    exp_list = [
        LinearThetaExperiment(x=x, y=y, include_second_output=include_second_output)
        for x, y in data
    ]

    return parmest.Estimator(exp_list, obj_function="SSE")


def _build_indexed_theta_estimator(data):
    exp_list = [IndexedThetaExperiment(x=x, y=y) for x, y in data]

    return parmest.Estimator(exp_list, obj_function="SSE")


def _build_bounded_estimator(data):
    exp_list = [BoundedLinearThetaExperiment(x=x, y=y) for x, y in data]

    return parmest.Estimator(exp_list, obj_function="SSE")


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
class TestParmestBlockEF(unittest.TestCase):

    def test_block_ef_structure_counts(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])
        model = pest._create_scenario_blocks()

        theta_names = model._parmest_theta_names
        self.assertEqual(list(model.scenario_indices), [0, 1])
        self.assertEqual(
            [pyo.value(model.scenario_number[i]) for i in model.scenario_indices],
            [0, 1],
        )
        self.assertEqual(list(model.exp_scenarios.keys()), list(model.scenario_indices))
        self.assertEqual(len(list(model.exp_scenarios.keys())), 2)
        self.assertEqual(len(model.theta_link_constraints), 2 * len(theta_names))
        self.assertTrue(hasattr(model, "Obj"))

        for block in model.exp_scenarios.values():
            self.assertFalse(block.Total_Cost_Objective.active)
            self.assertFalse(block.theta.fixed)
            self.assertAlmostEqual(
                pyo.value(block.theta),
                pyo.value(model.parmest_theta["theta"]),
                places=10,
            )

    def test_fix_theta_sets_all_scenario_theta_values(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])
        model = pest._create_scenario_blocks(theta_vals={"theta": 1.0}, fix_theta=True)

        self.assertTrue(model.parmest_theta["theta"].fixed)
        self.assertAlmostEqual(pyo.value(model.parmest_theta["theta"]), 1.0, places=10)
        self.assertEqual(len(model.theta_link_constraints), 0)

        for block in model.exp_scenarios.values():
            self.assertTrue(block.theta.fixed)
            self.assertAlmostEqual(pyo.value(block.theta), 1.0, places=10)

    def test_duplicate_bootlist_preserves_scenario_mapping(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])
        model = pest._create_scenario_blocks(bootlist=[0, 1, 1])

        self.assertEqual(pest.obj_probability_constant, 3)
        self.assertEqual(list(model.scenario_indices), [0, 1, 2])
        self.assertEqual(list(model.exp_scenarios.keys()), [0, 1, 2])
        self.assertEqual(
            [pyo.value(model.scenario_number[i]) for i in model.scenario_indices],
            [0, 1, 1],
        )
        self.assertIsNot(model.exp_scenarios[1], model.exp_scenarios[2])
        self.assertAlmostEqual(pyo.value(model.exp_scenarios[1].x), 2.0, places=10)
        self.assertAlmostEqual(pyo.value(model.exp_scenarios[2].x), 2.0, places=10)

    def test_indexed_unknown_parameters_are_expanded_and_fixed(self):
        pest = _build_indexed_theta_estimator([(1.0, 2.0), (2.0, 4.0)])

        model = pest._create_scenario_blocks(
            theta_vals={"theta[a]": 1.0, "theta[b]": 2.0}, fix_theta=True
        )

        self.assertEqual(list(model._parmest_theta_names), ["theta[a]", "theta[b]"])
        self.assertEqual(len(model.theta_link_constraints), 0)

        for block in model.exp_scenarios.values():
            self.assertTrue(block.theta["a"].fixed)
            self.assertTrue(block.theta["b"].fixed)
            self.assertAlmostEqual(pyo.value(block.theta["a"]), 1.0, places=10)
            self.assertAlmostEqual(pyo.value(block.theta["b"]), 2.0, places=10)

    def test_indexed_unknown_parameter_names_are_expanded_consistently(self):
        pest = _build_indexed_theta_estimator([(1.0, 2.0), (2.0, 4.0)])

        self.assertEqual(pest._return_theta_names(), ["theta[a]", "theta[b]"])

    @unittest.skipUnless(ipopt_available, "Test requires ipopt")
    def test_cov_est_counts_expanded_indexed_unknown_parameters(self):
        pest = _build_indexed_theta_estimator([(1.0, 2.0), (2.0, 4.0)])

        obj, theta = pest.theta_est()

        with self.assertRaisesRegex(
            AssertionError,
            "The number of datapoints must be greater than the number of parameters to estimate.",
        ):
            pest.cov_est()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_q_opt_solves_block_ef_and_returns_theta(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])

        obj, theta = pest._Q_opt()

        self.assertAlmostEqual(theta["theta"], 1.5, places=7)
        self.assertAlmostEqual(obj, 0.25, places=7)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_q_opt_returns_requested_values(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])

        obj, theta, var_values = pest._Q_opt(return_values=["y"])

        self.assertAlmostEqual(theta["theta"], 1.5, places=7)
        self.assertIsInstance(var_values, pd.DataFrame)
        self.assertEqual(list(var_values.columns), ["y"])
        self.assertEqual(len(var_values), 2)
        self.assertAlmostEqual(var_values.loc[0, "y"], 2.5, places=7)
        self.assertAlmostEqual(var_values.loc[1, "y"], 3.5, places=7)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_q_opt_fixed_theta_returns_objective_theta_and_status(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])

        obj, theta, status = pest._Q_opt(theta_vals={"theta": 1.0}, fix_theta=True)

        self.assertEqual(status, pyo.TerminationCondition.optimal)
        self.assertEqual(theta, {"theta": 1.0})
        self.assertAlmostEqual(obj, 0.5, places=8)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_q_opt_fixed_theta_infeasible_returns_none(self):
        pest = _build_bounded_estimator([(1.0, 2.0), (2.0, 3.0)])

        obj, theta, status = pest._Q_opt(theta_vals={"theta": 2.0}, fix_theta=True)

        self.assertIsNone(obj)
        self.assertEqual(theta, {"theta": 2.0})
        self.assertEqual(status, pyo.TerminationCondition.infeasible)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_objective_at_theta_fixed_value(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])

        theta_values = pd.DataFrame([[1.0]], columns=["theta"])
        obj_at_theta = pest.objective_at_theta(theta_values=theta_values)

        self.assertAlmostEqual(obj_at_theta.loc[0, "obj"], 0.5, places=8)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_objective_at_theta_none_uses_initial_theta(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 3.0)])

        obj_at_theta = pest.objective_at_theta()

        self.assertAlmostEqual(obj_at_theta.loc[0, "obj"], 1.0, places=8)
        self.assertAlmostEqual(obj_at_theta.loc[0, "theta"], 0.0, places=8)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_objective_at_theta_omits_infeasible_rows(self):
        pest = _build_bounded_estimator([(1.0, 2.0), (2.0, 3.0)])

        theta_values = pd.DataFrame([[0.0], [2.0]], columns=["theta"])

        obj_at_theta = pest.objective_at_theta(theta_values=theta_values)

        self.assertEqual(len(obj_at_theta), 1)
        self.assertEqual(list(obj_at_theta.columns), ["theta", "obj"])
        self.assertAlmostEqual(obj_at_theta.loc[0, "theta"], 0.0, places=8)
        self.assertAlmostEqual(obj_at_theta.loc[0, "obj"], 1.0, places=8)

    def test_invalid_solver_name_raises_runtimeerror(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])

        with self.assertRaisesRegex(
            RuntimeError, "Unknown solver in Q_Opt=not_a_solver"
        ):
            pest.theta_est(solver="not_a_solver")

    def test_theta_values_duplicate_columns_rejected(self):
        pest = _build_estimator([(1.0, 2.0), (2.0, 4.0)])

        duplicate_cols = pd.DataFrame([[1.0, 2.0]], columns=["theta", "theta"])

        with self.assertRaisesRegex(
            ValueError, "Duplicate theta names are not allowed"
        ):
            pest.objective_at_theta(theta_values=duplicate_cols)


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
class TestCountTotalExperiments(unittest.TestCase):
    def test_count_total_experiments_multi_output(self):
        exp_list = [
            LinearThetaExperiment(1.0, 2.0, include_second_output=True),
            LinearThetaExperiment(2.0, 4.0, include_second_output=True),
        ]

        total_points = parmest._count_total_experiments(exp_list)

        # The current parmest convention counts datapoints for one output family.
        self.assertEqual(total_points, 2)

    def test_count_total_experiments_tuple_index_multi_output(self):
        exp_list = [
            IndexedOutputExperiment(
                y_points=[(0.0, "A"), (1.0, "A")], z_points=[(0.0, "A"), (1.0, "A")]
            ),
            IndexedOutputExperiment(
                y_points=[(0.5, "A"), (1.5, "A")], z_points=[(0.5, "A"), (1.5, "A")]
            ),
        ]

        total_points = parmest._count_total_experiments(exp_list)

        self.assertEqual(total_points, 4)

    def test_count_total_experiments_rejects_mismatched_output_lengths(self):
        exp_list = [
            IndexedOutputExperiment(
                y_points=[(0.0, "A"), (1.0, "A")], z_points=[(0.0, "A")]
            )
        ]

        with self.assertRaisesRegex(
            AssertionError,
            "Experiment outputs must have the same number of indices or data points",
        ):
            parmest._count_total_experiments(exp_list)

    def test_count_total_experiments_rejects_mismatched_time_points(self):
        exp_list = [
            IndexedOutputExperiment(
                y_points=[(0.0, "A"), (1.0, "A")], z_points=[(0.0, "A"), (2.0, "A")]
            )
        ]

        with self.assertRaisesRegex(
            AssertionError,
            "Experiment outputs must share the same indices or data points",
        ):
            parmest._count_total_experiments(exp_list)

    def test_count_total_experiments_rejects_time_not_in_first_index(self):
        exp_list = [
            IndexedOutputExperiment(
                y_points=[(0.0, "A"), (1.0, "A")], z_points=[("A", 0.0), ("A", 1.0)]
            )
        ]

        with self.assertRaisesRegex(
            AssertionError,
            "The first index of experiment outputs must be the data point",
        ):
            parmest._count_total_experiments(exp_list)


class IndexedThetaMultistartExperiment(Experiment):
    def __init__(self):
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=["a", "b"])
        m.theta = pyo.Var(m.I, initialize={"a": 1.0, "b": 2.0})
        m.theta["a"].setlb(0.0)
        m.theta["a"].setub(5.0)
        m.theta["b"].setlb(0.0)
        m.theta["b"].setub(5.0)
        m.theta["a"].fix()
        m.theta["b"].fix()
        m.y = pyo.Var(initialize=3.0)
        m.eq = pyo.Constraint(expr=m.y == m.theta["a"] + m.theta["b"])
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, 3.0)])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


class NoBoundsExperiment(Experiment):
    def __init__(self):
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()
        m.theta = pyo.Var(initialize=1.0)
        m.y = pyo.Var(initialize=2.0)
        m.eq = pyo.Constraint(expr=m.y == m.theta + 1.0)
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, 2.0)])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


class StartCoupledExperiment(Experiment):
    """
    Model intentionally couples a fixed term ("bias") to theta_initial at
    build time. This exposes stale-model bugs in multistart paths.
    """

    def __init__(self, theta_initial=None):
        self.theta_initial = (
            theta_initial if theta_initial is not None else {"theta": 0.0}
        )
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()
        m.theta = pyo.Var(
            initialize=float(self.theta_initial["theta"]), bounds=(-10.0, 10.0)
        )
        m.bias = pyo.Param(initialize=float(self.theta_initial["theta"]), mutable=False)
        m.y = pyo.Var(initialize=0.0)
        m.eq = pyo.Constraint(expr=m.y == m.theta + m.bias)
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, 0.0)])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.label_model()
        return self.model


class QuotedIndexExperiment(Experiment):
    """
    Indexed theta over string indices that Pyomo quotes in component names
    (theta['1'], theta['2']), with two outputs so both entries are
    identifiable. The optimum is theta['1'] = 2.5, theta['2'] = 1.5.
    """

    def get_labeled_model(self):
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=["1", "2"])
        m.theta = pyo.Var(m.I, initialize=1.0, bounds=(0.0, 5.0))
        m.theta.fix()
        m.y = pyo.Var(initialize=0.0)
        m.z = pyo.Var(initialize=0.0)
        m.y_link = pyo.Constraint(expr=m.y == m.theta["1"] + m.theta["2"])
        m.z_link = pyo.Constraint(expr=m.z == m.theta["1"] - m.theta["2"])
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, 4.0), (m.z, 1.0)])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None), (m.z, None)])
        return m


class SineExperiment(Experiment):
    """
    One point of y = sin(k x). Over k in [0.1, 10] the SSE has several local
    minima, so different starts converge to different solutions.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_labeled_model(self):
        m = pyo.ConcreteModel()
        m.k = pyo.Var(initialize=1.0, bounds=(0.1, 10.0))
        m.k.fix()
        m.y_hat = pyo.Var(initialize=0.0)
        m.y_hat_link = pyo.Constraint(expr=m.y_hat == pyo.sin(m.k * self.x))
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y_hat, self.y)])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update([(m.k, pyo.ComponentUID(m.k))])
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y_hat, None)])
        return m


def _build_sine_estimator(solver_options=None, noise=0.0):
    # Data from k = 2: the global minimum is near k = 2 (SSE = 0 without
    # noise), and a start at k = 6 converges to a worse local minimum near
    # k = 6.2. Noise (seeded) gives a nonzero covariance at the best start.
    xs = np.linspace(0.2, 3.0, 8)
    ys = np.sin(2.0 * xs) + noise * np.random.default_rng(0).standard_normal(8)
    exp_list = [SineExperiment(x, float(y)) for x, y in zip(xs, ys)]
    return parmest.Estimator(
        exp_list, obj_function="SSE", solver_options=solver_options
    )


def _build_linear_estimator():
    exp_list = [LinearThetaExperiment(1.0, 2.0), LinearThetaExperiment(2.0, 3.0)]
    return parmest.Estimator(exp_list, obj_function="SSE")


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
class TestParmestMultistart(unittest.TestCase):
    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_multistart_baseline_equivalence_n1(self):
        pest = _build_linear_estimator()
        obj1, theta1 = pest.theta_est()
        _, best_theta, best_obj = pest.theta_est_multistart(
            n_restarts=1, multistart_sampling_method="uniform_random", seed=7
        )
        print(f"obj1: {obj1}, best_obj: {best_obj}")
        print(f"theta1: {theta1}, best_theta: {best_theta}")
        self.assertAlmostEqual(obj1, best_obj, places=7)
        self.assertAlmostEqual(theta1["theta"], best_theta["theta"], places=7)

    def test_uniform_sampling_is_deterministic_with_seed(self):
        pest = _build_linear_estimator()
        df1 = pest._generate_initial_theta(
            seed=4, n_restarts=5, multistart_sampling_method="uniform_random"
        )
        df2 = pest._generate_initial_theta(
            seed=4, n_restarts=5, multistart_sampling_method="uniform_random"
        )
        self.assertTrue(df1[["theta"]].equals(df2[["theta"]]))

    def test_uniform_sampling_changes_with_different_seed(self):
        pest = _build_linear_estimator()
        df1 = pest._generate_initial_theta(
            seed=4, n_restarts=5, multistart_sampling_method="uniform_random"
        )
        df2 = pest._generate_initial_theta(
            seed=5, n_restarts=5, multistart_sampling_method="uniform_random"
        )
        self.assertFalse(df1[["theta"]].equals(df2[["theta"]]))

    def test_latin_hypercube_sampling_is_deterministic(self):
        pest = _build_linear_estimator()
        df1 = pest._generate_initial_theta(
            seed=11, n_restarts=4, multistart_sampling_method="latin_hypercube"
        )
        df2 = pest._generate_initial_theta(
            seed=11, n_restarts=4, multistart_sampling_method="latin_hypercube"
        )
        self.assertTrue(df1[["theta"]].equals(df2[["theta"]]))

    def test_sobol_sampling_is_deterministic(self):
        pest = _build_linear_estimator()
        df1 = pest._generate_initial_theta(
            seed=12, n_restarts=4, multistart_sampling_method="sobol_sampling"
        )
        df2 = pest._generate_initial_theta(
            seed=12, n_restarts=4, multistart_sampling_method="sobol_sampling"
        )
        self.assertTrue(df1[["theta"]].equals(df2[["theta"]]))

    def test_generated_starts_are_within_bounds(self):
        pest = _build_linear_estimator()
        for method in ("uniform_random", "latin_hypercube", "sobol_sampling"):
            df = pest._generate_initial_theta(
                seed=1, n_restarts=8, multistart_sampling_method=method
            )
            self.assertTrue(((df["theta"] >= -10.0) & (df["theta"] <= 10.0)).all())

    def test_missing_bounds_raise_error(self):
        pest = parmest.Estimator([NoBoundsExperiment()], obj_function="SSE")
        with self.assertRaisesRegex(
            ValueError, "lower and upper bounds for the theta values must be defined"
        ):
            pest._generate_initial_theta(
                seed=1, n_restarts=2, multistart_sampling_method="uniform_random"
            )

    def test_invalid_bounds_raise_error(self):
        class InvalidBoundsExperiment(Experiment):
            def __init__(self):
                self.model = None

            def create_model(self):
                m = pyo.ConcreteModel()
                m.theta = pyo.Var(initialize=1.0)
                m.theta.setlb(2.0)
                m.theta.setub(1.0)
                m.y = pyo.Var(initialize=2.0)
                m.eq = pyo.Constraint(expr=m.y == m.theta + 1.0)
                self.model = m

            def label_model(self):
                m = self.model
                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update([(m.y, 2.0)])
                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update([(m.y, None)])

            def get_labeled_model(self):
                self.create_model()
                self.label_model()
                return self.model

        pest = parmest.Estimator([InvalidBoundsExperiment()], obj_function="SSE")
        with self.assertRaisesRegex(ValueError, "lower bound must be less than"):
            pest._generate_initial_theta(
                seed=1, n_restarts=2, multistart_sampling_method="uniform_random"
            )

    def test_user_provided_values_dimension_mismatch_raises(self):
        pest = _build_linear_estimator()
        user_df = pd.DataFrame([[1.0, 2.0]], columns=["theta", "extra"])
        with self.assertRaisesRegex(ValueError, "exactly one column per theta name"):
            pest.theta_est_multistart(
                n_restarts=1,
                multistart_sampling_method="user_provided_values",
                user_provided_df=user_df,
            )

    def test_user_provided_values_column_order_maps_by_name(self):
        pest = parmest.Estimator(
            [IndexedThetaMultistartExperiment()], obj_function="SSE"
        )
        user_df = pd.DataFrame(
            [[0.3, 4.2], [0.4, 4.1]], columns=["theta[b]", "theta[a]"]
        )
        results_df, _, _ = pest.theta_est_multistart(
            n_restarts=2,
            multistart_sampling_method="user_provided_values",
            user_provided_df=user_df,
        )
        self.assertAlmostEqual(results_df.loc[0, "theta[a]"], 4.2, places=12)
        self.assertAlmostEqual(results_df.loc[0, "theta[b]"], 0.3, places=12)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_state_isolation_between_starts(self):
        pest = _build_linear_estimator()
        init = pd.DataFrame([[-9.0], [9.0]], columns=["theta"])
        results_df, _, _ = pest.theta_est_multistart(
            user_provided_df=init, save_results=False
        )
        # Initial starts should remain exactly as supplied.
        self.assertAlmostEqual(results_df.loc[0, "theta"], -9.0, places=12)
        self.assertAlmostEqual(results_df.loc[1, "theta"], 9.0, places=12)
        # Both runs converge to the same optimum, showing no cross-start contamination.
        self.assertAlmostEqual(
            results_df.loc[0, "converged_theta"],
            results_df.loc[1, "converged_theta"],
            places=8,
        )

    def test_indexed_unknown_parameters_supported_in_sampling(self):
        pest = parmest.Estimator(
            [IndexedThetaMultistartExperiment()], obj_function="SSE"
        )
        df = pest._generate_initial_theta(
            seed=10, n_restarts=3, multistart_sampling_method="uniform_random"
        )
        self.assertTrue({"theta[a]", "theta[b]"}.issubset(set(df.columns)))

    def test_count_total_experiments_uses_one_output_family(self):
        class MultiOutputExperiment(Experiment):
            def create_model(self):
                m = pyo.ConcreteModel()
                m.theta = pyo.Var(initialize=0.0, bounds=(-10, 10))
                m.y = pyo.Var(initialize=1.0)
                m.z = pyo.Var(initialize=2.0)
                m.c1 = pyo.Constraint(expr=m.y == m.theta + 1.0)
                m.c2 = pyo.Constraint(expr=m.z == 2.0 * m.theta + 2.0)
                self.model = m

            def label_model(self):
                m = self.model
                m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.experiment_outputs.update([(m.y, 1.0), (m.z, 2.0)])
                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update([(m.theta, pyo.ComponentUID(m.theta))])
                m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.measurement_error.update([(m.y, None), (m.z, None)])

            def get_labeled_model(self):
                self.create_model()
                self.label_model()
                return self.model

        total_points = parmest._count_total_experiments(
            [MultiOutputExperiment(), MultiOutputExperiment()]
        )
        self.assertEqual(total_points, 2)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_quoted_index_names_map_starts_and_results(self):
        pest = parmest.Estimator([QuotedIndexExperiment()], obj_function="SSE")
        _, theta_single = pest.theta_est()
        # Columns use the theta_est names, given in a different order.
        starts = pd.DataFrame(
            [[4.2, 0.3], [0.1, 4.5]], columns=["theta['2']", "theta['1']"]
        )
        results_df, best_theta, best_obj = pest.theta_est_multistart(
            user_provided_df=starts
        )
        # Columns and best_theta use the same names as theta_est.
        self.assertEqual(set(best_theta), set(theta_single))
        self.assertAlmostEqual(results_df.loc[0, "theta['1']"], 0.3, places=12)
        self.assertAlmostEqual(results_df.loc[1, "theta['2']"], 0.1, places=12)
        for i in range(2):
            self.assertAlmostEqual(
                results_df.loc[i, "converged_theta['1']"], 2.5, places=6
            )
            self.assertAlmostEqual(
                results_df.loc[i, "converged_theta['2']"], 1.5, places=6
            )
        self.assertAlmostEqual(best_theta["theta['1']"], 2.5, places=6)
        self.assertAlmostEqual(best_theta["theta['2']"], 1.5, places=6)
        self.assertAlmostEqual(best_obj, 0.0, places=8)

    def test_quoted_index_starts_initialize_ef_model(self):
        pest = parmest.Estimator([QuotedIndexExperiment()], obj_function="SSE")
        starts = pd.DataFrame([[0.3, 4.2]], columns=["theta['1']", "theta['2']"])
        df = pest._generate_initial_theta(user_provided_df=starts)
        # The row, keyed by the table's column names, is what
        # theta_est_multistart passes as theta_vals for each start.
        theta_vals = {name: float(df.loc[0, name]) for name in starts.columns}
        model = pest._create_scenario_blocks(theta_vals=theta_vals)
        self.assertEqual(pyo.value(model.parmest_theta["theta['1']"]), 0.3)
        self.assertEqual(pyo.value(model.parmest_theta["theta['2']"]), 4.2)
        child = model.exp_scenarios[0]
        self.assertEqual(pyo.value(child.theta["1"]), 0.3)
        self.assertEqual(pyo.value(child.theta["2"]), 4.2)

    def test_user_provided_columns_must_match_theta_names(self):
        pest = parmest.Estimator([QuotedIndexExperiment()], obj_function="SSE")
        for cols in (["theta[1]", "theta[2]"], ["theta['1']", "theta['1']"]):
            starts = pd.DataFrame([[0.3, 4.2]], columns=cols)
            with self.assertRaisesRegex(ValueError, "must match the theta names"):
                pest._generate_initial_theta(user_provided_df=starts)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_estimator_state_after_multistart_is_best_start(self):
        pest = _build_sine_estimator(noise=0.02)
        # Best start first and a worse local minimum last, so leftover state
        # from the last solve would be detected.
        starts = pd.DataFrame({"k": [2.3, 6.0]})
        results_df, best_theta, best_obj = pest.theta_est_multistart(
            user_provided_df=starts
        )
        self.assertAlmostEqual(results_df.loc[1, "converged_k"], 6.18, places=1)
        self.assertAlmostEqual(best_theta["k"], 2.0, delta=0.01)
        self.assertLess(best_obj, results_df.loc[1, "final objective"])
        # The best start's solution is restored as-is, not solved again.
        self.assertEqual(pest.estimated_theta, best_theta)
        self.assertEqual(pest.obj_value, best_obj)
        self.assertEqual(
            pyo.value(pest.ef_instance.parmest_theta["k"]), best_theta["k"]
        )
        # cov_est must be evaluated at the best start: compare with a run
        # whose only start is the best one. reduced_hessian uses the solved
        # model (ef_instance), finite_difference only the estimated theta.
        ref = _build_sine_estimator(noise=0.02)
        ref.theta_est_multistart(user_provided_df=pd.DataFrame({"k": [2.3]}))
        methods = ["finite_difference"]
        if parmest.inverse_reduced_hessian_available:
            methods.append("reduced_hessian")
        for method in methods:
            np.testing.assert_allclose(
                pest.cov_est(method=method).to_numpy(),
                ref.cov_est(method=method).to_numpy(),
                rtol=1e-6,
                err_msg=method,
            )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_nonoptimal_starts_record_termination_and_keep_state(self):
        pest = _build_sine_estimator(solver_options={"max_iter": 1})
        # State from an earlier estimate must survive a multistart run in
        # which no start converges.
        pest.estimated_theta = {"k": 1.23}
        pest.obj_value = 4.56
        log = io.StringIO()
        with LoggingIntercept(log, "pyomo.contrib.parmest", logging.WARNING):
            results_df, best_theta, best_obj = pest.theta_est_multistart(
                n_restarts=3, seed=3
            )
        self.assertTrue(
            (
                results_df["solver termination"]
                == str(pyo.TerminationCondition.maxIterations)
            ).all()
        )
        self.assertTrue(results_df["final objective"].isna().all())
        self.assertIsNone(best_theta)
        self.assertTrue(np.isnan(best_obj))
        self.assertIn("none of the 3 starts terminated optimally", log.getvalue())
        self.assertEqual(pest.estimated_theta, {"k": 1.23})
        self.assertEqual(pest.obj_value, 4.56)

    @unittest.pytest.mark.mpi
    def test_multistart_parallel_ranks_share_starts(self):
        """use mpiexec and mpi4py"""
        # With seed=None, each rank would sample different starts if the root
        # rank's table were not shared. The driver checks that every rank
        # returns the same starts, holds the best start's solution (and gets
        # the same cov_est), and that the saved CSV matches the starts.
        driver = """
import sys
from mpi4py import MPI
from pyomo.common.dependencies import numpy as np, pandas as pd
import pyomo.environ as pyo
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment,
)

comm = MPI.COMM_WORLD
data = pd.DataFrame(
    data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
    columns=["hour", "y"],
)
exp_list = [RooneyBieglerExperiment(data.loc[i, :]) for i in range(data.shape[0])]
pest = parmest.Estimator(exp_list, obj_function="SSE")
results_df, best_theta, best_obj = pest.theta_est_multistart(
    n_restarts=4, seed=None, save_results=True, file_name=sys.argv[1]
)
cols = ["asymptote", "rate_constant"]
# Every rank holds the best start's solution, including the rank(s) that did
# not solve it and loaded its variable values.
assert best_theta is not None
assert pest.estimated_theta == best_theta and pest.obj_value == best_obj
for name in cols:
    assert pyo.value(pest.ef_instance.parmest_theta[name]) == best_theta[name]
cov = pest.cov_est().to_numpy()
all_dfs = comm.gather(results_df, root=0)
all_covs = comm.gather(cov, root=0)
if comm.rank == 0:
    for df in all_dfs[1:]:
        assert df[cols].equals(all_dfs[0][cols]), "ranks used different starts"
    for c in all_covs[1:]:
        assert np.allclose(c, all_covs[0]), "ranks computed different covariances"
    saved = pd.read_csv(sys.argv[1])
    assert np.allclose(saved[cols].to_numpy(), results_df[cols].to_numpy())
"""
        with TempfileManager.new_context() as tempfile:
            tmpdir = tempfile.mkdtemp()
            driver_path = os.path.join(tmpdir, "multistart_mpi_driver.py")
            with open(driver_path, "w") as f:
                f.write(driver)
            csv_path = os.path.join(tmpdir, "results.csv")
            rlist = [
                "mpiexec",
                "--allow-run-as-root",
                "-n",
                "2",
                sys.executable,
                driver_path,
                csv_path,
            ]
            ret = subprocess.run(rlist)
            self.assertEqual(ret.returncode, 0)

    # Not sure if this test is needed, but leaving here until I decide.
    # @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    # def test_multistart_results_reproducible_when_rerun_from_recorded_init(self):
    #     pest = parmest.Estimator([StartCoupledExperiment()], obj_function="SSE")
    #     init_df = pd.DataFrame([[2.0], [1.5], [3.0]], columns=["theta"])
    #     print(f"init_df:\n{init_df}")
    #     results_df, _, _ = pest.theta_est_multistart(
    #         user_provided_df=init_df, save_results=False
    #     )

    #     for _, row in results_df.iterrows():
    #         theta_init = {"theta": float(row["theta"])}
    #         exp = StartCoupledExperiment(theta_initial=theta_init)
    #         rerun = parmest.Estimator([exp], obj_function="SSE")
    #         obj, theta = rerun.theta_est()

    #         print(f"obj: {obj}, row['final objective']: {row['final objective']}")
    #         self.assertTrue(
    #             np.isclose(obj, row["final objective"], rtol=1e-6, atol=1e-8)
    #         )
    #         print(f"theta: {theta['theta']}, row['converged_theta']: {row['converged_theta']}")
    #         self.assertTrue(
    #             np.isclose(theta["theta"], row["converged_theta"], rtol=1e-6, atol=1e-8)
    #         )


class AffineTwoThetaExperiment(Experiment):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.model = None

    def create_model(self):
        m = pyo.ConcreteModel()
        m.theta_a = pyo.Var(initialize=1.0, bounds=(0.0, 4.0))
        m.theta_b = pyo.Var(initialize=0.0, bounds=(-3.0, 3.0))
        m.x = pyo.Param(initialize=float(self.x_data), mutable=False)
        m.y = pyo.Var(initialize=float(self.y_data))
        m.eq = pyo.Constraint(expr=m.y == m.theta_a * m.x + m.theta_b)
        self.model = m

    def label_model(self):
        m = self.model
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, float(self.y_data))])
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            [
                (m.theta_a, pyo.ComponentUID(m.theta_a)),
                (m.theta_b, pyo.ComponentUID(m.theta_b)),
            ]
        )
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, None)])

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


class _InitRecordingEstimator(parmest.Estimator):
    """Estimator that records the initial theta values of each profile solve."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_theta_inits = []

    def _Q_opt(self, *args, **kwargs):
        if kwargs.get("fixed_theta_values"):
            self.profile_theta_inits.append(dict(kwargs["theta_vals"]))
        return super()._Q_opt(*args, **kwargs)


def _build_two_theta_estimator(estimator_class=parmest.Estimator):
    # Data lie exactly on y = 2x + 1, so theta_hat = (2, 1) and, with
    # theta_a fixed at a, the profiled optimum is theta_b = 5 - 2a.
    exp_list = [
        AffineTwoThetaExperiment(1.0, 3.0),
        AffineTwoThetaExperiment(2.0, 5.0),
        AffineTwoThetaExperiment(3.0, 7.0),
    ]
    return estimator_class(exp_list, obj_function="SSE")


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
class TestParmestProfileLikelihood(unittest.TestCase):
    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_contains_baseline_minimum_point(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt"
        )
        prof = res["profiles"]
        self.assertAlmostEqual(res["baseline"]["obj_hat"], prof["obj"].min(), places=8)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_partial_fix_enforced(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt"
        )
        prof = res["profiles"]
        self.assertTrue(np.allclose(prof["theta_value"], prof["theta__theta_a"]))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_other_thetas_unfixed(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt"
        )
        prof = res["profiles"].sort_values("theta_value")
        self.assertGreater(
            prof["theta__theta_b"].max() - prof["theta__theta_b"].min(), 1e-6
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_repeatability(self):
        pest = _build_two_theta_estimator()
        res1 = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt"
        )
        res2 = pest.profile_likelihood(
            "theta_a", grid=[1.5, 2.0, 2.5], solver="ef_ipopt"
        )
        cols = [
            "theta_value",
            "obj",
            "status",
            "success",
            "theta__theta_a",
            "theta__theta_b",
        ]
        self.assertTrue(res1["profiles"][cols].equals(res2["profiles"][cols]))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_warmstart_neighbor_values_used(self):
        # theta_hat["theta_b"] is deliberately off (0.0 instead of 1.0) so that
        # warm-started initial values are distinguishable from theta_hat.
        theta_hat = {"theta_a": 2.0, "theta_b": 0.0}
        grid = [2.0, 2.1, 2.2]

        pest = _build_two_theta_estimator(_InitRecordingEstimator)
        pest.profile_likelihood(
            "theta_a", grid=grid, theta_hat=theta_hat, obj_hat=0.0, warmstart="neighbor"
        )
        inits = pest.profile_theta_inits
        self.assertEqual([init["theta_a"] for init in inits], grid)
        # First solve starts from theta_hat; each later solve starts from the
        # previous converged point (theta_b = 5 - 2 * theta_a).
        self.assertEqual(inits[0]["theta_b"], 0.0)
        self.assertAlmostEqual(inits[1]["theta_b"], 1.0, places=6)
        self.assertAlmostEqual(inits[2]["theta_b"], 0.8, places=6)

        pest = _build_two_theta_estimator(_InitRecordingEstimator)
        pest.profile_likelihood(
            "theta_a", grid=grid, theta_hat=theta_hat, obj_hat=0.0, warmstart="none"
        )
        self.assertEqual(
            [init["theta_b"] for init in pest.profile_theta_inits], [0.0, 0.0, 0.0]
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_failure_recorded_continue(self):
        pest = _build_two_theta_estimator()
        # theta_a = 5.0 is outside the model bounds (0, 4), so that solve
        # fails while the neighboring grid points still succeed.
        with LoggingIntercept(level=logging.WARNING) as LOG:
            res = pest.profile_likelihood(
                "theta_a",
                grid=[1.9, 5.0, 2.1],
                theta_hat={"theta_a": 2.0, "theta_b": 1.0},
                obj_hat=0.0,
            )
        self.assertIn("outside the bounds", LOG.getvalue())

        prof = res["profiles"].set_index("theta_value")
        self.assertEqual(len(prof), 4)
        self.assertIn("exception", str(prof.loc[5.0, "status"]))
        self.assertFalse(bool(prof.loc[5.0, "success"]))
        self.assertTrue(np.isnan(prof.loc[5.0, "obj"]))
        self.assertTrue(prof.drop(index=5.0)["success"].all())

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_nonoptimal_point_records_termination(self):
        exp_list = [
            AffineTwoThetaExperiment(1.0, 3.0),
            AffineTwoThetaExperiment(2.0, 5.0),
            AffineTwoThetaExperiment(3.0, 7.0),
        ]
        pest = parmest.Estimator(
            exp_list, obj_function="SSE", solver_options={"max_iter": 1}
        )
        # Solves that stop at the iteration limit are recorded with their
        # termination condition, not as exceptions.
        res = pest.profile_likelihood(
            "theta_a",
            grid=[1.9, 2.1],
            theta_hat={"theta_a": 2.0, "theta_b": 1.0},
            obj_hat=0.0,
        )
        prof = res["profiles"]
        self.assertEqual(len(prof), 3)
        self.assertTrue(
            (prof["status"] == str(pyo.TerminationCondition.maxIterations)).all()
        )
        self.assertFalse(prof["success"].any())
        self.assertTrue(prof["obj"].isna().all())

    def test_profile_all_failures_returns_structure(self):
        pest = _build_two_theta_estimator()
        # An unknown solver makes every profile solve raise.
        res = pest.profile_likelihood(
            "theta_a",
            grid=[1.9, 2.0, 2.1],
            theta_hat={"theta_a": 2.0, "theta_b": 0.0},
            obj_hat=1.0,
            solver="not_a_solver",
        )

        prof = res["profiles"]
        self.assertEqual(len(prof), 3)
        self.assertFalse(prof["success"].any())
        self.assertTrue(prof["status"].astype(str).str.contains("exception").all())
        self.assertTrue(prof["obj"].isna().all())

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_user_grid_preserved(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a",
            grid=[1.2, 2.4, 2.0],
            theta_hat={"theta_a": 2.0, "theta_b": 1.0},
            obj_hat=0.0,
        )

        attempted = res["profiles"]["theta_value"].tolist()
        self.assertEqual(attempted, [1.2, 2.0, 2.4])

    def test_profile_auto_grid_includes_theta_hat(self):
        pest = _build_two_theta_estimator()
        grid = pest._build_profile_grid(
            profiled_theta="theta_a",
            grid=[1.0, 1.5, 2.5],
            n_grid=5,
            theta_hat={"theta_a": 2.0, "theta_b": 0.0},
        )
        self.assertIn(2.0, set(np.round(grid, 12)))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_result_columns_schema(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a",
            grid=[1.9, 2.0],
            theta_hat={"theta_a": 2.0, "theta_b": 1.0},
            obj_hat=0.0,
        )
        prof = res["profiles"]
        self.assertTrue(
            set(
                [
                    "profiled_theta",
                    "theta_value",
                    "obj",
                    "delta_obj",
                    "lr_stat",
                    "status",
                    "success",
                    "solve_time",
                    "theta__theta_a",
                    "theta__theta_b",
                ]
            ).issubset(prof.columns)
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_baseline_from_multistart(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            profiled_theta="theta_a",
            n_grid=5,
            use_multistart_for_baseline=True,
            baseline_multistart_kwargs={
                "n_restarts": 3,
                "multistart_sampling_method": "uniform_random",
                "seed": 7,
            },
        )
        self.assertIn("baseline", res)
        self.assertIn("profiles", res)
        self.assertTrue(np.isfinite(res["baseline"]["obj_hat"]))
        self.assertTrue(res["baseline"]["used_multistart"])
        self.assertGreaterEqual(res["profiles"].shape[0], 1)

    @parameterized.expand([("SSE", None), ("SSE_weighted", 0.5)])
    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_lr_stat_is_likelihood_ratio(self, obj_function, sigma):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
            RooneyBieglerExperiment,
        )

        data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )
        pest = parmest.Estimator(
            [
                RooneyBieglerExperiment(data.loc[i, :], measure_error=sigma)
                for i in range(data.shape[0])
            ],
            obj_function=obj_function,
        )
        res = pest.profile_likelihood("asymptote", grid=[16.0, 18.0, 22.0])
        prof = res["profiles"]
        self.assertTrue(prof["success"].all())

        # Independent calculation from the model y = a * (1 - exp(-b * t))
        t, y = data["hour"].to_numpy(float), data["y"].to_numpy(float)

        def ssr(a, b):
            return np.sum((y - a * (1 - np.exp(-b * t))) ** 2)

        theta_hat = res["baseline"]["theta_hat"]
        ssr_hat = ssr(theta_hat["asymptote"], theta_hat["rate_constant"])
        ssr_prof = np.array(
            [
                ssr(a, b)
                for a, b in zip(prof["theta__asymptote"], prof["theta__rate_constant"])
            ]
        )
        if obj_function == "SSE":
            # unknown variance profiled out: N * log(SSR / SSR_hat)
            expected = len(y) * np.log(ssr_prof / ssr_hat)
        else:
            # known variance: (SSR - SSR_hat) / sigma^2
            expected = (ssr_prof - ssr_hat) / sigma**2
        np.testing.assert_allclose(prof["lr_stat"], expected, rtol=1e-6, atol=1e-8)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_lr_stat_undefined_for_custom_objective(self):
        exp_list = [
            AffineTwoThetaExperiment(1.0, 3.0),
            AffineTwoThetaExperiment(2.0, 5.0),
            AffineTwoThetaExperiment(3.0, 7.0),
        ]
        pest = parmest.Estimator(exp_list, obj_function=parmest.SSE)
        with LoggingIntercept(level=logging.WARNING) as LOG:
            res = pest.profile_likelihood(
                "theta_a",
                grid=[1.9, 2.1],
                theta_hat={"theta_a": 2.0, "theta_b": 1.0},
                obj_hat=0.0,
            )
        self.assertIn("lr_stat is only defined", LOG.getvalue())
        self.assertTrue(res["profiles"]["success"].all())
        self.assertTrue(res["profiles"]["lr_stat"].isna().all())

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_lr_stat_perfect_fit(self):
        # The data lie exactly on the model, so SSE_hat = 0 and the likelihood
        # ratio is not defined for the SSE objective.
        pest = _build_two_theta_estimator()
        with LoggingIntercept(level=logging.WARNING) as LOG:
            res = pest.profile_likelihood(
                "theta_a",
                grid=[1.9, 2.1],
                theta_hat={"theta_a": 2.0, "theta_b": 1.0},
                obj_hat=0.0,
            )
        self.assertIn("perfect fit", LOG.getvalue())
        self.assertTrue(res["profiles"]["success"].all())
        self.assertTrue(res["profiles"]["lr_stat"].isna().all())

    def test_split_grid_at_center(self):
        pest = _build_two_theta_estimator()
        branches = pest._split_grid_at_center([1.0, 1.5, 2.0, 2.5, 3.0], center=2.0)
        self.assertEqual(branches, [[2.0, 2.5, 3.0], [1.5, 1.0]])
        # Grid values are rounded, so center may differ slightly from the grid
        # value; the closest grid value still starts the upper side.
        branches = pest._split_grid_at_center([1.0, 2.0, 3.0], center=2.0 + 1e-15)
        self.assertEqual(branches, [[2.0, 3.0], [1.0]])
        # A side with no values is left out.
        branches = pest._split_grid_at_center([2.0, 2.5], center=2.0)
        self.assertEqual(branches, [[2.0, 2.5]])

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_failures_stop_only_one_side(self):
        pest = _build_two_theta_estimator()
        # theta_hat is near the upper bound of theta_a (4.0): the solve at
        # 4.2 fails, which stops the upper side, while every value below
        # theta_hat is still solved.
        with LoggingIntercept(level=logging.WARNING):
            res = pest.profile_likelihood(
                "theta_a",
                grid=[3.0, 3.4, 3.8, 4.2, 4.4],
                theta_hat={"theta_a": 3.9, "theta_b": 5.0 - 2.0 * 3.9},
                obj_hat=0.0,
                max_consecutive_failures=1,
            )
        prof = res["profiles"].set_index("theta_value")
        self.assertEqual(sorted(prof.index), [3.0, 3.4, 3.8, 3.9, 4.2])
        self.assertFalse(bool(prof.loc[4.2, "success"]))
        self.assertTrue(prof.loc[[3.0, 3.4, 3.8, 3.9], "success"].all())
        for a in (3.0, 3.4, 3.8):
            self.assertAlmostEqual(prof.loc[a, "theta__theta_b"], 5.0 - 2.0 * a, 6)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_given_baseline_does_not_use_multistart(self):
        pest = _build_two_theta_estimator()
        res = pest.profile_likelihood(
            "theta_a",
            grid=[1.9, 2.1],
            theta_hat={"theta_a": 2.0, "theta_b": 1.0},
            obj_hat=0.0,
            use_multistart_for_baseline=True,
        )
        self.assertFalse(res["baseline"]["used_multistart"])

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_profile_restores_estimator_state(self):
        pest = _build_two_theta_estimator()
        obj_hat, theta_hat = pest.theta_est()
        state = (pest.ef_instance, dict(pest.estimated_theta), pest.obj_value)
        # The last grid value solved (theta_a = 3.5) is far from theta_hat.
        pest.profile_likelihood(
            "theta_a", grid=[1.0, 3.5], theta_hat=dict(theta_hat), obj_hat=obj_hat
        )
        self.assertIs(pest.ef_instance, state[0])
        self.assertEqual(pest.estimated_theta, state[1])
        # cov_est reads only these, so it is unchanged by profiling.
        self.assertEqual(pest.obj_value, state[2])

    @unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
    def test_fixed_theta_values_fix_at_given_values(self):
        pest = _build_two_theta_estimator()
        # No theta_vals: theta_a must be fixed at 3.0 from fixed_theta_values,
        # and the profiled optimum is theta_b = 5 - 2 * 3.0 = -1.
        obj, theta, term = pest._Q_opt(fixed_theta_values={"theta_a": 3.0})
        self.assertEqual(str(term), str(pyo.TerminationCondition.optimal))
        self.assertEqual(theta["theta_a"], 3.0)
        self.assertAlmostEqual(theta["theta_b"], -1.0, places=6)
        # fixed_theta_values takes precedence over theta_vals.
        model = pest._create_scenario_blocks(
            theta_vals={"theta_a": 1.5, "theta_b": 0.0},
            fixed_theta_values={"theta_a": 3.0},
        )
        self.assertTrue(model.parmest_theta["theta_a"].fixed)
        self.assertEqual(pyo.value(model.parmest_theta["theta_a"]), 3.0)
        self.assertFalse(model.parmest_theta["theta_b"].fixed)
        for block in model.exp_scenarios.values():
            self.assertEqual(pyo.value(block.theta_a), 3.0)
            self.assertTrue(block.theta_a.fixed)


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

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy,
    scipy_available,
    matplotlib,
    matplotlib_available,
)

import platform

is_osx = platform.mac_ver()[0] != ""

import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product

import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()

from pyomo.common.fileutils import find_library

pynumero_ASL_available = False if find_library("pynumero_ASL") is None else True

testdir = os.path.dirname(os.path.abspath(__file__))


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestRooneyBiegler(unittest.TestCase):
    def setUp(self):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
            rooney_biegler_model,
        )

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
            list(product(asym, rate)), columns=self.pest.theta_names
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
            1, None, 3, "Rect", [0.5, 1.0], seed=5436
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
            list(product(asym, rate)), columns=self.pest.theta_names
        )

        obj_at_theta = self.pest.objective_at_theta(theta_vals)

        self.pest.diagnostic_mode = False

    @unittest.skip("Presently having trouble with mpiexec on appveyor")
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
            + "rooney_biegler_parmest.py"
        )
        rbpath = os.path.abspath(rbpath)  # paranoia strikes deep...
        rlist = ["mpiexec", "--allow-run-as-root", "-n", "2", sys.executable, rbpath]
        if sys.version_info >= (3, 5):
            ret = subprocess.run(rlist)
            retcode = ret.returncode
        else:
            retcode = subprocess.call(rlist)
        assert retcode == 0

    @unittest.skip("Most folks don't have k_aug installed")
    def test_theta_k_aug_for_Hessian(self):
        # this will fail if k_aug is not installed
        objval, thetavals, Hessian = self.pest.theta_est(solver="k_aug")
        self.assertAlmostEqual(objval, 4.4675, places=2)

    @unittest.skipIf(not pynumero_ASL_available, "pynumero ASL is not available")
    @unittest.skipIf(
        not parmest.inverse_reduced_hessian_available,
        "Cannot test covariance matrix: required ASL dependency is missing",
    )
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
class TestModelVariants(unittest.TestCase):
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
            model.theta[
                "asymptote"
            ].fixed = (
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

    @unittest.skipIf(not pynumero_ASL_available, "pynumero ASL is not available")
    @unittest.skipIf(
        not parmest.inverse_reduced_hessian_available,
        "Cannot test covariance matrix: required ASL dependency is missing",
    )
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
class TestReactorDesign(unittest.TestCase):
    def setUp(self):
        from pyomo.contrib.parmest.examples.reactor_design.reactor_design import (
            reactor_design_model,
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
class TestReactorDesign_DAE(unittest.TestCase):
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
            rooney_biegler_model_with_constraint,
        )

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

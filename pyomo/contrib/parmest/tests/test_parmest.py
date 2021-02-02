#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import (
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
    scipy, scipy_available,
    matplotlib, matplotlib_available,
)

import platform
is_osx = platform.mac_ver()[0] != ''

import pyutilib.th as unittest
import sys
import os
import subprocess
from itertools import product

import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo

from pyomo.opt import SolverFactory
ipopt_available = SolverFactory('ipopt').available()

testdir = os.path.dirname(os.path.abspath(__file__))

class Object_from_string_Tester(unittest.TestCase):
    def setUp(self):
        self.instance = pyo.ConcreteModel()
        self.instance.IDX = pyo.Set(initialize=['a', 'b', 'c'])
        self.instance.x = pyo.Var(self.instance.IDX, initialize=1134)
        # TBD add a block
        if parmest.parmest_available:
            np.random.seed(1134)

    def tearDown(self):
        pass

    def test_Var(self):
        # just making sure it executes
        pyo_Var_obj = parmest._object_from_string(self.instance, "x[b]")
        fixstatus = pyo_Var_obj.fixed
        self.assertEqual(fixstatus, False)


@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class parmest_object_Tester_RB(unittest.TestCase):

    def setUp(self):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import rooney_biegler_model

        data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                                  [4,16.0],[5,15.6],[6,19.8]], columns=['hour', 'y'])

        theta_names = ['asymptote', 'rate_constant']

        def SSE(model, data):
            expr = sum((data.y[i] - model.response_function[data.hour[i]])**2 for i in data.index)
            return expr

        solver_options = {
                'tol': 1e-8,
                }

        self.pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE,
                solver_options=solver_options)

    def test_theta_est(self):
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(objval, 4.4675, places=2)
        self.assertAlmostEqual(thetavals['asymptote'], 19.2189, places=2) # 19.1426 from the paper
        self.assertAlmostEqual(thetavals['rate_constant'], 0.5312, places=2) # 0.5311 from the paper

    @unittest.skipIf(not graphics.imports_available,
                     "parmest.graphics imports are unavailable")
    def test_bootstrap(self):
        objval, thetavals = self.pest.theta_est()

        num_bootstraps=10
        theta_est = self.pest.theta_est_bootstrap(num_bootstraps, return_samples=True)

        num_samples = theta_est['samples'].apply(len)
        self.assertTrue(len(theta_est.index), 10)
        self.assertTrue(num_samples.equals(pd.Series([6]*10)))

        del theta_est['samples']

        # apply cofidence region test
        CR = self.pest.confidence_region_test(theta_est, 'MVN', [0.5, 0.75, 1.0])

        self.assertTrue(set(CR.columns) >= set([0.5, 0.75, 1.0]))
        self.assertTrue(CR[0.5].sum() == 5)
        self.assertTrue(CR[0.75].sum() == 7)
        self.assertTrue(CR[1.0].sum() == 10) # all true

        graphics.pairwise_plot(theta_est)
        graphics.pairwise_plot(theta_est, thetavals)
        graphics.pairwise_plot(theta_est, thetavals, 0.8, ['MVN', 'KDE', 'Rect'])

    @unittest.skipIf(not graphics.imports_available,
                     "parmest.graphics imports are unavailable")
    def test_likelihood_ratio(self):
        objval, thetavals = self.pest.theta_est()

        asym = np.arange(10, 30, 2)
        rate = np.arange(0, 1.5, 0.25)
        theta_vals = pd.DataFrame(list(product(asym, rate)), columns=self.pest.theta_names)

        obj_at_theta = self.pest.objective_at_theta(theta_vals)

        LR = self.pest.likelihood_ratio_test(obj_at_theta, objval, [0.8, 0.9, 1.0])

        self.assertTrue(set(LR.columns) >= set([0.8, 0.9, 1.0]))
        self.assertTrue(LR[0.8].sum() == 7)
        self.assertTrue(LR[0.9].sum() == 11)
        self.assertTrue(LR[1.0].sum() == 60) # all true

        graphics.pairwise_plot(LR, thetavals, 0.8)

    def test_leaveNout(self):
        lNo_theta = self.pest.theta_est_leaveNout(1)
        self.assertTrue(lNo_theta.shape == (6,2))

        results = self.pest.leaveNout_bootstrap_test(1, None, 3, 'Rect', [0.5, 1.0])
        self.assertTrue(len(results) == 6) # 6 lNo samples
        i = 1
        samples = results[i][0] # list of N samples that are left out
        lno_theta = results[i][1]
        bootstrap_theta = results[i][2]
        self.assertTrue(samples == [1]) # sample 1 was left out
        self.assertTrue(lno_theta.shape[0] == 1) # lno estimate for sample 1
        self.assertTrue(set(lno_theta.columns) >= set([0.5, 1.0]))
        self.assertTrue(lno_theta[1.0].sum() == 1) # all true
        self.assertTrue(bootstrap_theta.shape[0] == 3) # bootstrap for sample 1
        self.assertTrue(bootstrap_theta[1.0].sum() == 3) # all true

    def test_diagnostic_mode(self):
        self.pest.diagnostic_mode = True

        objval, thetavals = self.pest.theta_est()

        asym = np.arange(10, 30, 2)
        rate = np.arange(0, 1.5, 0.25)
        theta_vals = pd.DataFrame(list(product(asym, rate)), columns=self.pest.theta_names)

        obj_at_theta = self.pest.objective_at_theta(theta_vals)

        self.pest.diagnostic_mode = False

    def test_rb_main(self):
        """ test __main__ for rooney biegler """
        p = str(parmestbase.__path__)
        l = p.find("'")
        r = p.find("'", l+1)
        parmestpath = p[l+1:r]
        rbpath = parmestpath + os.sep + "examples" + os.sep + \
                   "rooney_biegler" + os.sep + "rooney_biegler.py"
        rbpath = os.path.abspath(rbpath) # paranoia strikes deep...
        if sys.version_info >= (3,5):
            ret = subprocess.run([sys.executable, rbpath])
            retcode = ret.returncode
        else:
            retcode = subprocess.call([sys.executable, rbpath])
        assert(retcode == 0)

    @unittest.skip("Presently having trouble with mpiexec on appveyor")
    def test_parallel_parmest(self):
        """ use mpiexec and mpi4py """
        p = str(parmestbase.__path__)
        l = p.find("'")
        r = p.find("'", l+1)
        parmestpath = p[l+1:r]
        rbpath = parmestpath + os.sep + "examples" + os.sep + \
                   "rooney_biegler" + os.sep + "rooney_biegler_parmest.py"
        rbpath = os.path.abspath(rbpath) # paranoia strikes deep...
        rlist = ["mpiexec", "--allow-run-as-root", "-n", "2", sys.executable, rbpath]
        if sys.version_info >= (3,5):
            ret = subprocess.run(rlist)
            retcode = ret.returncode
        else:
            retcode = subprocess.call(rlist)
        assert(retcode == 0)

    @unittest.skip("Most folks don't have k_aug installed")
    def test_theta_k_aug_for_Hessian(self):
        # this will fail if k_aug is not installed
        objval, thetavals, Hessian = self.pest.theta_est(solver="k_aug")
        self.assertAlmostEqual(objval, 4.4675, places=2)


'''
The test cases above were developed with a transcription mistake in the dataset.
This test works with the correct dataset.
'''
@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class parmest_object_Tester_RB_match_paper(unittest.TestCase):

    def setUp(self):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import rooney_biegler_model

        data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                                  [4,16.0],[5,15.6],[7,19.8]], columns=['hour', 'y'])

        theta_names = ['asymptote', 'rate_constant']

        def SSE(model, data):
            expr = sum((data.y[i] - model.response_function[data.hour[i]])**2 for i in data.index)
            return expr

        self.pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE)

    def test_theta_est(self):
        objval, thetavals = self.pest.theta_est(calc_cov=False)

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(thetavals['asymptote'], 19.1426, places=2) # 19.1426 from the paper
        self.assertAlmostEqual(thetavals['rate_constant'], 0.5311, places=2) # 0.5311 from the paper

    @unittest.skipIf(not parmest.inverse_reduced_hessian_available,
                     "Cannot test covariance matrix: required ASL dependency is missing")
    def test_theta_est_cov(self):
        objval, thetavals, cov = self.pest.theta_est(calc_cov=True)

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(thetavals['asymptote'], 19.1426, places=2) # 19.1426 from the paper
        self.assertAlmostEqual(thetavals['rate_constant'], 0.5311, places=2) # 0.5311 from the paper

        # Covariance matrix
        self.assertAlmostEqual(cov.iloc[0,0], 6.30579403, places=2) # 6.22864 from paper
        self.assertAlmostEqual(cov.iloc[0,1], -0.4395341, places=2) # -0.4322 from paper
        self.assertAlmostEqual(cov.iloc[1,0], -0.4395341, places=2) # -0.4322 from paper
        self.assertAlmostEqual(cov.iloc[1,1], 0.04193591, places=2) # 0.04124 from paper

        ''' Why does the covariance matrix from parmest not match the paper? Parmest is
        calculating the exact reduced Hessian. The paper (Rooney and Bielger, 2001) likely
        employed the first order approximation common for nonlinear regression. The paper
        values were verified with Scipy, which uses the same first order approximation.
        The formula used in parmest was verified against equations (7-5-15) and (7-5-16) in
        "Nonlinear Parameter Estimation", Y. Bard, 1974.
        '''


@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class Test_parmest_indexed_variables(unittest.TestCase):

    def make_model(self, theta_names):

        data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                                  [4,16.0],[5,15.6],[7,19.8]], columns=['hour', 'y'])

        def rooney_biegler_model_alternate(data):
            ''' Alternate model definition used in a unit test
            Here, the fitted parameters are defined as a single variable over a set
            A bit silly for this specific example
            '''

            model = pyo.ConcreteModel()

            model.var_names = pyo.Set(initialize=['asymptote','rate_constant'])

            model.theta = pyo.Var(model.var_names, initialize={'asymptote':15, 'rate_constant':0.5})

            def response_rule(m, h):
                expr = m.theta['asymptote'] * (1 - pyo.exp(-m.theta['rate_constant'] * h))
                return expr
            model.response_function = pyo.Expression(data.hour, rule = response_rule)

            def SSE_rule(m):
                return sum((data.y[i] - m.response_function[data.hour[i]])**2 for i in data.index)
            model.SSE = pyo.Objective(rule = SSE_rule, sense=pyo.minimize)

            return model

        def SSE(model, data):
            expr = sum((data.y[i] - model.response_function[data.hour[i]])**2 for i in data.index)
            return expr

        return parmest.Estimator(rooney_biegler_model_alternate, data, theta_names, SSE)

    def test_theta_est_quotedIndex(self):

        theta_names = ["theta['asymptote']", "theta['rate_constant']"]

        pest = self.make_model(theta_names)
        objval, thetavals = pest.theta_est(calc_cov=False)

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(thetavals["theta[asymptote]"], 19.1426, places=2) # 19.1426 from the paper
        self.assertAlmostEqual(thetavals['theta[rate_constant]'], 0.5311, places=2) # 0.5311 from the paper

    def test_theta_est_impliedStrIndex(self):

        theta_names = ["theta[asymptote]", "theta[rate_constant]"]

        pest = self.make_model(theta_names)
        objval, thetavals = pest.theta_est(calc_cov=False)

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(thetavals["theta[asymptote]"], 19.1426, places=2) # 19.1426 from the paper
        self.assertAlmostEqual(thetavals['theta[rate_constant]'], 0.5311, places=2) # 0.5311 from the paper


    @unittest.skipIf(
        not parmest.inverse_reduced_hessian_available,
        "Cannot test covariance matrix: required ASL dependency is missing")
    def test_theta_est_cov(self):
        theta_names = ["theta[asymptote]", "theta[rate_constant]"]

        pest = self.make_model(theta_names)
        objval, thetavals, cov = pest.theta_est(calc_cov=True)

        self.assertAlmostEqual(objval, 4.3317112, places=2)
        self.assertAlmostEqual(thetavals["theta[asymptote]"], 19.1426, places=2) # 19.1426 from the paper
        self.assertAlmostEqual(thetavals['theta[rate_constant]'], 0.5311, places=2) # 0.5311 from the paper

        # Covariance matrix
        self.assertAlmostEqual(cov.iloc[0,0], 6.30579403, places=2) # 6.22864 from paper
        self.assertAlmostEqual(cov.iloc[0,1], -0.4395341, places=2) # -0.4322 from paper
        self.assertAlmostEqual(cov.iloc[1,0], -0.4395341, places=2) # -0.4322 from paper
        self.assertAlmostEqual(cov.iloc[1,1], 0.04193591, places=2) # 0.04124 from paper



@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available,
                 "The 'ipopt' solver is not available")
class parmest_object_Tester_reactor_design(unittest.TestCase):

    def setUp(self):
        from pyomo.contrib.parmest.examples.reactor_design.reactor_design import reactor_design_model

        # Data from the design
        data = pd.DataFrame(data=[[1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5],
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
                                  [1.80, 10000, 4392.8, 1056.0,  977.7, 1786.7],
                                  [1.85, 10000, 4442.6, 1052.8,  948.4, 1778.1],
                                  [1.90, 10000, 4491.3, 1049.4,  920.5, 1769.4],
                                  [1.95, 10000, 4538.8, 1045.8,  893.9, 1760.8]],
                          columns=['sv', 'caf', 'ca', 'cb', 'cc', 'cd'])

        theta_names = ['k1', 'k2', 'k3']

        def SSE(model, data):
            expr = (float(data['ca']) - model.ca)**2 + \
                   (float(data['cb']) - model.cb)**2 + \
                   (float(data['cc']) - model.cc)**2 + \
                   (float(data['cd']) - model.cd)**2
            return expr

        solver_options = {"max_iter": 6000}

        self.pest = parmest.Estimator(reactor_design_model, data,
                                      theta_names, SSE, solver_options)

    def test_theta_est(self):
        objval, thetavals = self.pest.theta_est()

        self.assertAlmostEqual(thetavals['k1'], 5.0/6.0, places=4)
        self.assertAlmostEqual(thetavals['k2'], 5.0/3.0, places=4)
        self.assertAlmostEqual(thetavals['k3'], 1.0/6000.0, places=7)

    def test_return_values(self):
        objval, thetavals, data_rec =\
            self.pest.theta_est(return_values=['ca', 'cb', 'cc', 'cd', 'caf'])
        self.assertAlmostEqual(data_rec["cc"].loc[18], 893.84924, places=3)



@unittest.skipIf(not parmest.parmest_available,
                 "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not graphics.imports_available,
                 "parmest.graphics imports are unavailable")
@unittest.skipIf(is_osx, "Disabling graphics tests on OSX due to issue in Matplotlib, see Pyomo PR #1337")
class parmest_graphics(unittest.TestCase):

    def setUp(self):
        self.A = pd.DataFrame(np.random.randint(0,100,size=(100,4)), columns=list('ABCD'))
        self.B = pd.DataFrame(np.random.randint(0,100,size=(100,4)), columns=list('ABCD'))

    def test_pairwise_plot(self):
        graphics.pairwise_plot(self.A, alpha=0.8, distributions=['Rect', 'MVN', 'KDE'])

    def test_grouped_boxplot(self):
        graphics.grouped_boxplot(self.A, self.B, normalize=True,
                                group_names=['A', 'B'])

    def test_grouped_violinplot(self):
        graphics.grouped_violinplot(self.A, self.B)

if __name__ == '__main__':
    unittest.main()

# the matpolotlib stuff is to avoid $DISPLAY errors on Travis (DLW Oct 2018)
try:
    import matplotlib
    matplotlib.use('Agg')
except:
    pass
try:
    import numpy as np
    import pandas as pd
    imports_not_present = False
except:
    imports_not_present = True
import pyutilib.th as unittest
import tempfile
import sys
import os
import shutil
import glob
import subprocess
from itertools import product

import pyomo.contrib.parmest.parmest as parmest
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
        if not imports_not_present:
            np.random.seed(1134)
        
    def tearDown(self):
        pass
    
    def test_Var(self):
        # just making sure it executes
        pyo_Var_obj = parmest._object_from_string(self.instance, "x[b]")
        fixstatus = pyo_Var_obj.fixed
        self.assertEqual(fixstatus, False)

        
@unittest.skipIf(imports_not_present, "Cannot test parmest: required dependencies are missing")
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
        
        self.pest = parmest.Estimator(rooney_biegler_model, data, theta_names, SSE)

    def test_theta_est(self):
        objval, thetavals = self.pest.theta_est()
        
        self.assertAlmostEqual(objval, 4.4675, places=2)
        self.assertAlmostEqual(thetavals['asymptote'], 19.2189, places=2) # 19.1426 from the paper
        self.assertAlmostEqual(thetavals['rate_constant'], 0.5312, places=2) # 0.5311 from the paper
        
    def test_bootstrap(self):
        objval, thetavals = self.pest.theta_est()
        
        num_bootstraps=10
        theta_est = self.pest.theta_est_bootstrap(num_bootstraps, return_samples=True)
        
        num_samples = theta_est['samples'].apply(len)
        self.assertTrue(len(theta_est.index), 10)
        self.assertTrue(num_samples.equals(pd.Series([6]*10)))

        del theta_est['samples']
        
        filename = os.path.abspath(os.path.join(testdir, 'pairwise_bootstrap.png'))
        if os.path.isfile(filename):
            os.remove(filename)
        parmest.pairwise_plot(theta_est, filename=filename)
        #self.assertTrue(os.path.isfile(filename))
        
        filename = os.path.abspath(os.path.join(testdir, 'pairwise_bootstrap_theta.png'))
        if os.path.isfile(filename):
            os.remove(filename)
        parmest.pairwise_plot(theta_est, thetavals, filename=filename)
        #self.assertTrue(os.path.isfile(filename))
        
        filename = os.path.abspath(os.path.join(testdir, 'pairwise_bootstrap_theta_CI.png'))
        if os.path.isfile(filename):
            os.remove(filename)
        parmest.pairwise_plot(theta_est, thetavals, 0.8, ['MVN', 'KDE', 'Rect'],
                                         filename=filename)
        #self.assertTrue(os.path.isfile(filename))
        
    def test_likelihood_ratio(self):
        # tbd: write the plot file(s) to a temp dir and delete in cleanup
        objval, thetavals = self.pest.theta_est()
        
        asym = np.arange(10, 30, 2)
        rate = np.arange(0, 1.5, 0.25)
        theta_vals = pd.DataFrame(list(product(asym, rate)), columns=self.pest.theta_names)
        
        obj_at_theta = self.pest.objective_at_theta(theta_vals)
        
        LR = self.pest.likelihood_ratio_test(obj_at_theta, objval, [0.8, 0.85, 0.9, 0.95])
        
        self.assertTrue(set(LR.columns) >= set([0.8, 0.85, 0.9, 0.95]))
        
        filename = os.path.abspath(os.path.join(testdir, 'pairwise_LR_plot.png'))
        if os.path.isfile(filename):
            os.remove(filename)
        parmest.pairwise_plot(LR, thetavals, 0.8,  filename=filename)
        #self.assertTrue(os.path.isfile(filename))

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
        if sys.version_info >= (3,0):
            ret = subprocess.run(["python", rbpath])
            retcode = ret.returncode
        else:
            retcode = subprocess.call(["python", rbpath])
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
        rlist = ["mpiexec", "--allow-run-as-root", "-n", "2", "python", rbpath]
        if sys.version_info >= (3,0):
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
        

@unittest.skipIf(imports_not_present, "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
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
        
        self.pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)

    def test_theta_est(self):
        objval, thetavals = self.pest.theta_est()
        
        self.assertAlmostEqual(thetavals['k1'], 5.0/6.0, places=4) 
        self.assertAlmostEqual(thetavals['k2'], 5.0/3.0, places=4) 
        self.assertAlmostEqual(thetavals['k3'], 1.0/6000.0, places=7) 
        

if __name__ == '__main__':
    unittest.main()

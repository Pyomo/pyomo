# Provide some test for parmest
# Author: Started by David L. Woodruff (summer 2018)

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
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest as parmestbase
if not imports_not_present:
    import pyomo.contrib.parmest.graphics as graphics
import pyomo.environ as pyo

__author__ = 'David L. Woodruff <DLWoodruff@UCDavis.edu>'
__date__ = 'July 2018'
__version__ = 0.21

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

@unittest.skipIf(imports_not_present, "Cannot test parmest: required dependencies are missing")
class parmest_object_Tester_RB(unittest.TestCase):
    def setUp(self):
        self.thetalist = ['asymptote', 'rate_constant']
        self.num_samples = 6
        self.sample_list = range(self.num_samples)
        self.fsfilename = \
            "pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler"
        self.callback_funcname = "instance_creation_callback"
        experiment_data = pd.DataFrame(data=[[1,8.3],[2,10.3],[3,19.0],
                                      [4,16.0],[5,15.6],[6,19.8]],
                                      columns=['hour', 'y'])
        

        self.pest = parmest.ParmEstimator(self.fsfilename,
                                          self.callback_funcname,
                                          "SecondStageCost",
                                          self.sample_list,
                                          self.thetalist,
                                          cb_data = experiment_data)

    def test_theta_est(self):
        objval, thetavals = self.pest.theta_est()
        
        self.assertAlmostEqual(objval, 4.4675, places=2)
        self.assertAlmostEqual(thetavals['asymptote'], 19.2189, places=2) # 19.1426 from the paper
        self.assertAlmostEqual(thetavals['rate_constant'], 0.5312, places=2) # 0.5311 from the paper

    def test_bootstrap(self):
        objval, thetavals = self.pest.theta_est()
        
        num_bootstraps=10
        theta_est = self.pest.bootstrap(num_bootstraps)
        
        num_samples = theta_est['samples'].apply(len)
        self.assertTrue(len(theta_est.index), 10)
        self.assertTrue(num_samples.equals(pd.Series([6]*10)))

        filename = os.path.abspath(os.path.join(testdir, 'pairwise.png'))
        if os.path.isfile(filename):
            os.remove(filename)
        graphics.pairwise_plot(theta_est, filename=filename)
        #self.assertTrue(os.path.isfile(filename))
        
        filename = os.path.abspath(os.path.join(testdir, 'pairwise_bootstrap.png'))
        if os.path.isfile(filename):
            os.remove(filename)
        graphics.pairwise_bootstrap_plot(theta_est, thetavals, 0.8, 
                                         filename=filename)
        #self.assertTrue(os.path.isfile(filename))
        
    def test_likelihood_ratio(self):
        # tbd: write the plot file(s) to a temp dir and delete in cleanup
        objval, thetavals = self.pest.theta_est()
        search_ranges = {}
        search_ranges["asymptote"] = np.arange(10, 30, 5)
        search_ranges["rate_constant"] = np.arange(0, 1.5, 0.25)
        SSE = self.pest.likelihood_ratio(search_ranges=search_ranges)
            
        filename = os.path.abspath(os.path.join(testdir, 'pairwise_LR_plot.png'))
        if os.path.isfile(filename):
            os.remove(filename)
        graphics.pairwise_likelihood_ratio_plot(SSE, objval, 0.9, self.num_samples, 
                                                filename=filename)
        #self.assertTrue(os.path.isfile(filename))

    def test_diagnostic_mode(self):
        self.pest.diagnostic_mode = True

        objval, thetavals = self.pest.theta_est()

        search_ranges = {}
        search_ranges["asymptote"] = np.arange(10, 30, 10)
        search_ranges["rate_constant"] = np.arange(0, 1.5, 0.5)
        self.pest.likelihood_ratio(search_ranges=search_ranges)

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
                   "rooney_biegler" + os.sep + "rb_drive_parmest.py"
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
        
#=====================================================================
@unittest.skipIf(imports_not_present, "Cannot test parmest: required dependencies are missing")
class parmest_object_Tester_SB(unittest.TestCase):
    def setUp(self):
        self.tempdirpath = tempfile.mkdtemp()
        # assuming we are in the test subdir
        import pyomo.contrib.parmest.examples.semibatch as sbroot
        p = str(sbroot.__path__)
        l = p.find("'")
        r = p.find("'", l+1)
        sbrootpath = p[l+1:r]

        for file in glob.glob(sbrootpath + os.sep + 'exp*.out'):
            shutil.copy(file, self.tempdirpath)
        self.save_cwd = os.getcwd()
        os.chdir(self.tempdirpath)
        num_experiments = 10
        exp_list = range(1,num_experiments+1) # callback uses one-based file names
        thetalist = ['k1', 'k2', 'E1', 'E2']

        np.random.seed(1134)

        from pyomo.contrib.parmest.examples.semibatch.semibatch import pysp_instance_creation_callback
        # non-string callback specification
        self.pest = parmest.ParmEstimator(None, pysp_instance_creation_callback,
                                     "SecondStageCost", exp_list, thetalist) 

    def tearDown(self):
        os.chdir(self.save_cwd)
        shutil.rmtree(self.tempdirpath)

    def quicky(self):
        objval, thetavals = self.pest.theta_est()
        
if __name__ == '__main__':
    unittest.main()

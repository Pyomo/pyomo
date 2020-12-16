#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
from os.path import abspath, dirname
pyomodir = dirname(abspath(__file__))+os.sep+".."+os.sep+".."+os.sep
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyomo.common
from pyomo.common.tempfiles import TempfileManager

from pyomo.core import ConcreteModel
from pyomo.opt import ResultsFormat, SolverResults, SolverFactory

old_ignore_time = None
old_tempdir = None
def setUpModule():
    global old_tempdir
    global old_ignore_time
    old_tempdir = TempfileManager.tempdir
    old_ignore_time = SolverResults.default_print_options.ignore_time
    SolverResults.default_print_options.ignore_time = True
    TempfileManager.tempdir = currdir

def tearDownModule():
    TempfileManager.tempdir = old_tempdir
    SolverResults.default_print_options.ignore_time = old_ignore_time

cplexamp_available = False
class mock_all(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global cplexamp_available
        import pyomo.environ
        from pyomo.solvers.tests.solvers import test_solver_cases
        cplexamp_available = test_solver_cases('cplex', 'nl').available

    def setUp(self):
        self.do_setup(False)

    def do_setup(self,flag):
        global tmpdir
        tmpdir = os.getcwd()
        os.chdir(currdir)
        TempfileManager.sequential_files(0)
        if flag:
            if not cplexamp_available:
                self.skipTest("The 'cplexamp' command is not available")
            self.asl = SolverFactory('asl:cplexamp')
        else:
            self.asl = SolverFactory('_mock_asl:cplexamp')

    def tearDown(self):
        global tmpdir
        TempfileManager.clear_tempfiles()
        TempfileManager.unique_files()
        os.chdir(tmpdir)
        self.asl = None

    def test_path(self):
        """ Verify that the ASL path is what is expected """
        if type(self.asl) == 'ASL':
            self.assertEqual(self.asl.executable.split(os.sep)[-1],
                             "ASL"+pyomo.common.executable_extension)

    def test_solve4(self):
        """ Test ASL - test4.nl """
        results = self.asl.solve(currdir+"test4.nl",
                                 logfile=currdir+"test_solve4.log",
                                 suffixes=['.*'])
        results.write(filename=currdir+"test_solve4.txt",
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(currdir+"test_solve4.txt",
                                       currdir+"test4_asl.txt",
                                       tolerance=1e-4)
        os.remove(currdir+"test_solve4.log")
        if os.path.exists(currdir+"test4.soln"):
            os.remove(currdir+"test4.soln")

    #
    # This test is disabled, but it's useful for interactively exercising
    # the option specifications of a solver
    #
    def Xtest_options(self):
        """ Test ASL options behavior """
        results = self.asl.solve(currdir+"bell3a.mps",
                                 logfile=currdir+"test_options.log",
                                 options="sec=0.1 foo=1 bar='a=b c=d' xx_zz=yy",
                                 suffixes=['.*'])
        results.write(filename=currdir+"test_options.txt",
                      times=False)
        self.assertFileEqualsBaseline(currdir+"test_options.txt",
                                      currdir+  "test4_asl.txt")
        #os.remove(currdir+"test4.sol")
        #os.remove(currdir+"test_solve4.log")

    def test_error1(self):
        """ Bad results format """
        try:
            model = ConcreteModel()
            results = self.asl.solve(model,
                                     format=ResultsFormat.sol,
                                     suffixes=['.*'])
            self.fail("test_error1")
        except ValueError:
            pass

    def test_error2(self):
        """ Bad solve option """
        try:
            model = ConcreteModel()
            results = self.asl.solve(model,
                                     foo="bar")
            self.fail("test_error2")
        except ValueError:
            pass

    def test_error3(self):
        """ Bad solve option """
        try:
            results = self.asl.solve(currdir+"model.py",
                                     foo="bar")
            self.fail("test_error3")
        except ValueError:
            pass

class mip_all(mock_all):

    def setUp(self):
        self.do_setup(True)


if __name__ == "__main__":
    unittest.main()

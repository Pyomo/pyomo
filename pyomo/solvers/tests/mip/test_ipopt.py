#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
from os.path import abspath, dirname
pyomodir = dirname(abspath(__file__))+os.sep+".."+os.sep+".."+os.sep
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

import pyomo.opt
from pyomo.core import *

old_tempdir = None
def setUpModule():
    global old_tempdir
    old_tempdir = pyutilib.services.TempfileManager.tempdir
    pyutilib.services.TempfileManager.tempdir = currdir

def tearDownModule():
    pyutilib.services.TempfileManager.tempdir = old_tempdir

ipopt_available = False
class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global ipopt_available
        import pyomo.environ
        from pyomo.solvers.tests.solvers import test_solver_cases
        ipopt_available = test_solver_cases('ipopt','nl').available

    def setUp(self):
        if not ipopt_available:
            self.skipTest("The 'ipopt' command is not available")
        self.do_setup()

    def do_setup(self):
        global tmpdir
        tmpdir = os.getcwd()
        os.chdir(currdir)
        pyutilib.services.TempfileManager.sequential_files(0)

        self.asl = pyomo.opt.SolverFactory('asl:ipopt', keepfiles=True)

        #ver = subprocess.Popen(['ipopt','-v'],stdout=subprocess.PIPE).communicate()[0].split()[1]
        #if ver != '3.9.3':
        #    for line in fileinput.input(currdir+"test_solve_from_nl.baseline",inplace=1):
        #            print line.replace('3.9.3',ver),
        #    for line in fileinput.input(currdir+"test_solve_from_instance.baseline",inplace=1):
        #            print line.replace('3.9.3',ver),

        # The sisser CUTEr instance
        # Formulated in Pyomo by Carl D. Laird, Daniel P. Word, Brandon C. Barrera and Saumyajyoti Chaudhuri
        # Taken from:

        #   Source:
        #   F.S. Sisser,
        #   "Elimination of bounds in optimization problems by transforming
        #   variables",
        #   Mathematical Programming 20:110-121, 1981.

        #   See also Buckley#216 (p. 91)

        #   SIF input: Ph. Toint, Dec 1989.

        #   classification OUR2-AN-2-0

        sisser_instance = ConcreteModel()

        sisser_instance.N = RangeSet(1,2)
        sisser_instance.xinit = Param(sisser_instance.N, initialize={ 1 : 1.0, 2 : 0.1})

        def fa(model, i):
            return value(model.xinit[i])
        sisser_instance.x = Var(sisser_instance.N,initialize=fa)

        def f(model):
            return 3*model.x[1]**4 - 2*(model.x[1]*model.x[2])**2 + 3*model.x[2]**4
        sisser_instance.f = Objective(rule=f,sense=minimize)

        self.sisser_instance = sisser_instance

    def tearDown(self):
        global tmpdir
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.unique_files()
        os.chdir(tmpdir)

    def test_solve_from_nl(self):
        # Test ipopt solve from nl file
        results = self.asl.solve(currdir+"sisser.pyomo.nl",
                                 logfile=currdir+"test_solve_from_nl.log",
                                 suffixes=['.*'])
        # We don't want the test to care about which Ipopt version we are using
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        results.write(filename=currdir+"test_solve_from_nl.txt",
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(currdir+"test_solve_from_nl.txt",
                                       currdir+"test_solve_from_nl.baseline",
                                       tolerance=1e-7)
        os.remove(currdir+"test_solve_from_nl.log")

    def test_solve_from_instance(self):
        # Test ipopt solve from a pyomo instance and load the solution
        results = self.asl.solve(self.sisser_instance,
                                 suffixes=['.*'])
        # We don't want the test to care about which Ipopt version we are using
        self.sisser_instance.solutions.store_to(results)
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        results.write(filename=currdir+"test_solve_from_instance.txt",
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(currdir+"test_solve_from_instance.txt",
                                       currdir+"test_solve_from_instance.baseline",
                                       tolerance=1e-7)
        #self.sisser_instance.load_solutions(results)

if __name__ == "__main__":
    unittest.main()

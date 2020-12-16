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
from os.path import abspath, dirname, join
currdir = dirname(abspath(__file__))

import pyutilib.th as unittest
from pyomo.common.tempfiles import TempfileManager

import pyomo.opt
from pyomo.core import ConcreteModel, RangeSet, Var, Param, Objective, ConstraintList, value, minimize

old_tempdir = None
def setUpModule():
    global old_tempdir
    old_tempdir = TempfileManager.tempdir
    TempfileManager.tempdir = currdir

def tearDownModule():
    TempfileManager.tempdir = old_tempdir

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
        TempfileManager.sequential_files(0)

        self.asl = pyomo.opt.SolverFactory('asl:ipopt', keepfiles=True)
        self.ipopt = pyomo.opt.SolverFactory('ipopt', keepfiles=True)

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
        TempfileManager.clear_tempfiles()
        TempfileManager.unique_files()
        os.chdir(tmpdir)

    def test_version_asl(self):
        self.assertTrue(self.asl.version() is not None)
        self.assertTrue(type(self.asl.version()) is tuple)
        self.assertEqual(len(self.asl.version()), 4)

    def test_version_ipopt(self):
        self.assertTrue(self.ipopt.version() is not None)
        self.assertTrue(type(self.ipopt.version()) is tuple)
        self.assertEqual(len(self.ipopt.version()), 4)

    def test_asl_solve_from_nl(self):
        # Test ipopt solve from nl file
        results = self.asl.solve(join(currdir, "sisser.pyomo.nl"),
                                 logfile=join(currdir, "test_asl_solve_from_nl.log"),
                                 suffixes=['.*'])
        # We don't want the test to care about which Ipopt version we are using
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        results.write(filename=join(currdir, "test_asl_solve_from_nl.txt"),
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(join(currdir, "test_asl_solve_from_nl.txt"),
                                       join(currdir, "test_solve_from_nl.baseline"),
                                       tolerance=1e-7)
        os.remove(join(currdir, "test_asl_solve_from_nl.log"))

    def test_ipopt_solve_from_nl(self):
        # Test ipopt solve from nl file
        results = self.ipopt.solve(join(currdir, "sisser.pyomo.nl"),
                                   logfile=join(currdir, "test_ipopt_solve_from_nl.log"),
                                   suffixes=['.*'])
        # We don't want the test to care about which Ipopt version we are using
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        results.write(filename=join(currdir, "test_ipopt_solve_from_nl.txt"),
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(join(currdir, "test_ipopt_solve_from_nl.txt"),
                                       join(currdir, "test_solve_from_nl.baseline"),
                                       tolerance=1e-7)
        os.remove(join(currdir, "test_ipopt_solve_from_nl.log"))

    def test_asl_solve_from_instance(self):
        # Test ipopt solve from a pyomo instance and load the solution
        results = self.asl.solve(self.sisser_instance,
                                 suffixes=['.*'])
        # We don't want the test to care about which Ipopt version we are using
        self.sisser_instance.solutions.store_to(results)
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        results.write(filename=join(currdir, "test_asl_solve_from_instance.txt"),
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(join(currdir, "test_asl_solve_from_instance.txt"),
                                       join(currdir, "test_solve_from_instance.baseline"),
                                       tolerance=1e-7)
        #self.sisser_instance.load_solutions(results)

    def test_ipopt_solve_from_instance(self):
        # Test ipopt solve from a pyomo instance and load the solution
        results = self.ipopt.solve(self.sisser_instance,
                                   suffixes=['.*'])
        # We don't want the test to care about which Ipopt version we are using
        self.sisser_instance.solutions.store_to(results)
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        results.write(filename=join(currdir, "test_ipopt_solve_from_instance.txt"),
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(join(currdir, "test_ipopt_solve_from_instance.txt"),
                                       join(currdir, "test_solve_from_instance.baseline"),
                                       tolerance=1e-7)
        #self.sisser_instance.load_solutions(results)

    def test_ipopt_solve_from_instance_OF_options(self):

        with self.assertRaises(ValueError):
            # using OF_ options AND option_file_name
            # is not allowed
            self.ipopt.solve(self.sisser_instance,
                             suffixes=['.*'],
                             options={"OF_mu_init": 0.1,
                                      "option_file_name": "junk.opt"})
        # Creating a dummy ipopt.opt file in the cwd
        # will cover the code that prints a warning
        assert os.getcwd() == currdir, str(os.getcwd())+" "+currdir
        with open(join(currdir, 'ipopt.opt'), "w") as f:
            pass
        # Test ipopt solve from a pyomo instance and load the solution
        results = self.ipopt.solve(self.sisser_instance,
                                   suffixes=['.*'],
                                   options={"OF_mu_init": 0.1})
        os.remove(join(currdir, 'ipopt.opt'))
        # We don't want the test to care about which Ipopt version we are using
        self.sisser_instance.solutions.store_to(results)
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        results.write(filename=join(currdir, "test_ipopt_solve_from_instance.txt"),
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(join(currdir, "test_ipopt_solve_from_instance.txt"),
                                       join(currdir, "test_solve_from_instance.baseline"),
                                       tolerance=1e-7)
        #self.sisser_instance.load_solutions(results)

    def test_bad_dof(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.c = ConstraintList()
        m.c.add(m.x + m.y == 1)
        m.c.add(m.x - m.y == 0)
        m.c.add(2*m.x - 3*m.y == 1)
        m.write('j.nl')
        res = self.ipopt.solve(m)
        self.assertEqual(str(res.solver.status), "warning")
        self.assertEqual(str(res.solver.termination_condition), "other")
        self.assertTrue("Too few degrees of freedom" in res.solver.message)

if __name__ == "__main__":
    unittest.main()

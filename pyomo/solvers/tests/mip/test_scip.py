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
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint

old_tempdir = None
def setUpModule():
    global old_tempdir
    old_tempdir = TempfileManager.tempdir
    TempfileManager.tempdir = currdir

def tearDownModule():
    TempfileManager.tempdir = old_tempdir

scip_available = False
class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global scip_available
        import pyomo.environ
        from pyomo.solvers.tests.solvers import test_solver_cases
        scip_available = test_solver_cases('scip','nl').available

    def setUp(self):
        if not scip_available:
            self.skipTest("The 'scipampl' command is not available")
        self.do_setup()

    def do_setup(self):
        global tmpdir
        tmpdir = os.getcwd()
        os.chdir(currdir)
        TempfileManager.sequential_files(0)

        self.scip = SolverFactory('scip', solver_io='nl')

        m = self.model = ConcreteModel()
        m.v = Var()
        m.o = Objective(expr=m.v)
        m.c = Constraint(expr=m.v >= 1)

    def tearDown(self):
        global tmpdir
        TempfileManager.clear_tempfiles()
        TempfileManager.unique_files()
        os.chdir(tmpdir)

    def test_version_scip(self):
        self.assertTrue(self.scip.version() is not None)
        self.assertTrue(type(self.scip.version()) is tuple)
        self.assertEqual(len(self.scip.version()), 4)

    def test_scip_solve_from_instance(self):
        # Test scip solve from a pyomo instance and load the solution
        results = self.scip.solve(self.model,
                                  suffixes=['.*'])
        # We don't want the test to care about which Scip version we are using
        self.model.solutions.store_to(results)
        results.Solution(0).Message = "Scip"
        results.Solver.Message = "Scip"
        results.Solver.Time = 0
        results.write(filename=join(currdir, "test_scip_solve_from_instance.txt"),
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(join(currdir, "test_scip_solve_from_instance.txt"),
                                       join(currdir, "test_scip_solve_from_instance.baseline"),
                                       tolerance=1e-7)

    def test_scip_solve_from_instance_options(self):

        # Creating a dummy scip.set file in the cwd
        # will cover the code that prints a warning
        assert os.getcwd() == currdir, str(os.getcwd())+" "+currdir
        with open(join(currdir, 'scip.set'), "w") as f:
            pass
        # Test scip solve from a pyomo instance and load the solution
        results = self.scip.solve(self.model,
                                  suffixes=['.*'],
                                  options={"limits/softtime": 100})
        os.remove(join(currdir, 'scip.set'))
        # We don't want the test to care about which Scip version we are using
        self.model.solutions.store_to(results)
        results.Solution(0).Message = "Scip"
        results.Solver.Message = "Scip"
        results.Solver.Time = 0
        results.write(filename=join(currdir, "test_scip_solve_from_instance.txt"),
                      times=False,
                      format='json')
        self.assertMatchesJsonBaseline(join(currdir, "test_scip_solve_from_instance.txt"),
                                       join(currdir, "test_scip_solve_from_instance.baseline"),
                                       tolerance=1e-7)

if __name__ == "__main__":
    unittest.main()

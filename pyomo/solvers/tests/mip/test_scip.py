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

import json
import os
from os.path import join

import pyomo.common.unittest as unittest

from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager

from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint

currdir = this_file_dir()
deleteFiles = True

scip_available = False


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global scip_available
        import pyomo.environ
        from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases

        scip_available = _test_solver_cases('scip', 'nl').available

    def setUp(self):
        if not scip_available:
            self.skipTest("The 'scipampl' command is not available")
        TempfileManager.push()

        self.scip = SolverFactory('scip', solver_io='nl')

        m = self.model = ConcreteModel()
        m.v = Var()
        m.o = Objective(expr=m.v)
        m.c = Constraint(expr=m.v >= 1)

    def tearDown(self):
        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())

    def compare_json(self, file1, file2):
        with open(file1, 'r') as out, open(file2, 'r') as txt:
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-7, allow_second_superset=True
            )

    def test_version_scip(self):
        self.assertTrue(self.scip.version() is not None)
        self.assertTrue(type(self.scip.version()) is tuple)
        self.assertEqual(len(self.scip.version()), 4)

    def test_scip_solve_from_instance(self):
        # Test scip solve from a pyomo instance and load the solution
        results = self.scip.solve(self.model, suffixes=['.*'])
        # We don't want the test to care about which Scip version we are using
        self.model.solutions.store_to(results)
        results.Solution(0).Message = "Scip"
        results.Solver.Message = "Scip"
        results.Solver.Time = 0
        _out = TempfileManager.create_tempfile(".txt")
        results.write(filename=_out, times=False, format='json')
        self.compare_json(_out, join(currdir, "test_scip_solve_from_instance.baseline"))

    def test_scip_solve_from_instance_options(self):
        # Creating a dummy scip.set file in the cwd
        # will cover the code that prints a warning
        _cwd = os.getcwd()
        tmpdir = TempfileManager.create_tempdir()
        try:
            os.chdir(tmpdir)
            open(join(tmpdir, 'scip.set'), "w").close()
            # Test scip solve from a pyomo instance and load the solution
            with LoggingIntercept() as LOG:
                results = self.scip.solve(
                    self.model, suffixes=['.*'], options={"limits/softtime": 100}
                )
            self.assertRegex(
                LOG.getvalue().replace("\n", " "),
                r"A file named (.*) exists in the current working "
                r"directory, but SCIP options are being "
                r"set using a separate options file. The "
                r"options file \1 will be ignored.",
            )
        finally:
            os.chdir(_cwd)
        # We don't want the test to care about which Scip version we are using
        self.model.solutions.store_to(results)
        results.Solution(0).Message = "Scip"
        results.Solver.Message = "Scip"
        results.Solver.Time = 0
        _out = TempfileManager.create_tempfile(".txt")
        results.write(filename=_out, times=False, format='json')
        self.compare_json(_out, join(currdir, "test_scip_solve_from_instance.baseline"))

    def test_scip_solve_from_instance_with_reoptimization(self):
        # Test scip with re-optimization option enabled
        # This case changes the Scip output results which may break the results parser
        self.scip.options['reoptimization/enable'] = True
        self.test_scip_solve_from_instance()


if __name__ == "__main__":
    deleteFiles = False
    unittest.main()

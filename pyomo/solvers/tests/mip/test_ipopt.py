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
from os.path import abspath, dirname, join

import pyomo.common.unittest as unittest

from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager

import pyomo.opt
from pyomo.core import (
    ConcreteModel,
    RangeSet,
    Var,
    Param,
    Objective,
    ConstraintList,
    value,
    minimize,
)

currdir = this_file_dir()
deleteFiles = True

ipopt_available = False


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ipopt_available
        import pyomo.environ
        from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases

        ipopt_available = _test_solver_cases('ipopt', 'nl').available

    def setUp(self):
        if not ipopt_available:
            self.skipTest("The 'ipopt' command is not available")
        TempfileManager.push()

        self.asl = pyomo.opt.SolverFactory('asl:ipopt', keepfiles=True)
        self.ipopt = pyomo.opt.SolverFactory('ipopt', keepfiles=True)

        # The sisser CUTEr instance
        # Formulated in Pyomo by Carl D. Laird, Daniel P. Word, Brandon
        #     C. Barrera and Saumyajyoti Chaudhuri
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

        sisser_instance.N = RangeSet(1, 2)
        sisser_instance.xinit = Param(sisser_instance.N, initialize={1: 1.0, 2: 0.1})

        def fa(model, i):
            return value(model.xinit[i])

        sisser_instance.x = Var(sisser_instance.N, initialize=fa)

        def f(model):
            return (
                3 * model.x[1] ** 4
                - 2 * (model.x[1] * model.x[2]) ** 2
                + 3 * model.x[2] ** 4
            )

        sisser_instance.f = Objective(rule=f, sense=minimize)

        self.sisser_instance = sisser_instance

    def tearDown(self):
        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())

    def compare_json(self, file1, file2):
        with open(file1, 'r') as out, open(file2, 'r') as txt:
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-7, allow_second_superset=True
            )

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
        _log = TempfileManager.create_tempfile(".test_ipopt.log")
        results = self.asl.solve(
            join(currdir, "sisser.pyomo.nl"), logfile=_log, suffixes=['.*']
        )
        # We don't want the test to care about which Ipopt version we are using
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        _out = TempfileManager.create_tempfile(".test_ipopt.txt")
        results.write(filename=_out, times=False, format='json')
        self.compare_json(_out, join(currdir, "test_solve_from_nl.baseline"))

    def test_ipopt_solve_from_nl(self):
        # Test ipopt solve from nl file
        _log = TempfileManager.create_tempfile(".test_ipopt.log")
        results = self.ipopt.solve(
            join(currdir, "sisser.pyomo.nl"), logfile=_log, suffixes=['.*']
        )
        # We don't want the test to care about which Ipopt version we are using
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        _out = TempfileManager.create_tempfile(".test_ipopt.txt")
        results.write(filename=_out, times=False, format='json')
        self.compare_json(_out, join(currdir, "test_solve_from_nl.baseline"))

    def test_asl_solve_from_instance(self):
        # Test ipopt solve from a pyomo instance and load the solution
        results = self.asl.solve(self.sisser_instance, suffixes=['.*'])
        # We don't want the test to care about which Ipopt version we are using
        self.sisser_instance.solutions.store_to(results)
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        _out = TempfileManager.create_tempfile(".test_ipopt.txt")
        results.write(filename=_out, times=False, format='json')
        self.compare_json(_out, join(currdir, "test_solve_from_instance.baseline"))
        # self.sisser_instance.load_solutions(results)

    def test_ipopt_solve_from_instance(self):
        # Test ipopt solve from a pyomo instance and load the solution
        results = self.ipopt.solve(self.sisser_instance, suffixes=['.*'])
        # We don't want the test to care about which Ipopt version we are using
        self.sisser_instance.solutions.store_to(results)
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        _out = TempfileManager.create_tempfile(".test_ipopt.txt")
        results.write(filename=_out, times=False, format='json')
        self.compare_json(_out, join(currdir, "test_solve_from_instance.baseline"))
        # self.sisser_instance.load_solutions(results)

    def test_ipopt_solve_from_instance_OF_options(self):
        with self.assertRaises(ValueError):
            # using OF_ options AND option_file_name
            # is not allowed
            self.ipopt.solve(
                self.sisser_instance,
                suffixes=['.*'],
                options={"OF_mu_init": 0.1, "option_file_name": "junk.opt"},
            )
        # Creating a dummy ipopt.opt file in the cwd
        # will cover the code that prints a warning
        _cwd = os.getcwd()
        tmpdir = TempfileManager.create_tempdir()
        try:
            os.chdir(tmpdir)
            # create an empty ipopt.opt file
            open(join(tmpdir, 'ipopt.opt'), "w").close()
            # Test ipopt solve from a pyomo instance and load the solution
            with LoggingIntercept() as LOG:
                results = self.ipopt.solve(
                    self.sisser_instance, suffixes=['.*'], options={"OF_mu_init": 0.1}
                )
            self.assertRegex(
                LOG.getvalue().replace("\n", " "),
                r"A file named (.*) exists in the current working "
                r"directory, but Ipopt options file options \(i.e., "
                r"options that start with 'OF_'\) were provided. The "
                r"options file \1 will be ignored.",
            )
        finally:
            os.chdir(_cwd)

        # We don't want the test to care about which Ipopt version we are using
        self.sisser_instance.solutions.store_to(results)
        results.Solution(0).Message = "Ipopt"
        results.Solver.Message = "Ipopt"
        _out = TempfileManager.create_tempfile(".test_ipopt.txt")
        results.write(filename=_out, times=False, format='json')
        self.compare_json(_out, join(currdir, "test_solve_from_instance.baseline"))
        # self.sisser_instance.load_solutions(results)

    def test_bad_dof(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.c = ConstraintList()
        m.c.add(m.x + m.y == 1)
        m.c.add(m.x - m.y == 0)
        m.c.add(2 * m.x - 3 * m.y == 1)
        res = self.ipopt.solve(m)
        self.assertEqual(str(res.solver.status), "warning")
        self.assertEqual(str(res.solver.termination_condition), "other")
        self.assertTrue("Too few degrees of freedom" in res.solver.message)


if __name__ == "__main__":
    deleteFiles = False
    unittest.main()

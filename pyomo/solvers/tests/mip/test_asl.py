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

import json
import os
from os.path import join
from filecmp import cmp

import pyomo.common.unittest as unittest

import pyomo.common
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager

from pyomo.core import ConcreteModel
from pyomo.opt import ResultsFormat, SolverResults, SolverFactory

currdir = this_file_dir()
deleteFiles = True

old_ignore_time = None


def setUpModule():
    global old_ignore_time
    old_ignore_time = SolverResults.default_print_options.ignore_time
    SolverResults.default_print_options.ignore_time = True


def tearDownModule():
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

    def do_setup(self, flag):
        TempfileManager.push()
        if flag:
            if not cplexamp_available:
                self.skipTest("The 'cplexamp' command is not available")
            self.asl = SolverFactory('asl:cplexamp')
        else:
            self.asl = SolverFactory('_mock_asl:cplexamp')

    def tearDown(self):
        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())
        self.asl = None

    def test_path(self):
        """Verify that the ASL path is what is expected"""
        if type(self.asl) == 'ASL':
            self.assertEqual(
                self.asl.executable.split(os.sep)[-1],
                "ASL" + pyomo.common.executable_extension,
            )

    def test_solve4(self):
        """Test ASL - test4.nl"""
        _log = TempfileManager.create_tempfile(".test_solve4.log")
        _out = TempfileManager.create_tempfile(".test_solve4.txt")

        results = self.asl.solve(
            join(currdir, "test4.nl"), logfile=_log, suffixes=['.*']
        )
        results.write(filename=_out, times=False, format='json')
        _baseline = join(currdir, "test4_asl.txt")
        with open(_out, 'r') as out, open(_baseline, 'r') as txt:
            self.assertStructuredAlmostEqual(
                json.load(txt), json.load(out), abstol=1e-4, allow_second_superset=True
            )

    #
    # This test is disabled, but it's useful for interactively exercising
    # the option specifications of a solver
    #
    def Xtest_options(self):
        """Test ASL options behavior"""
        results = self.asl.solve(
            currdir + "bell3a.mps",
            logfile=currdir + "test_options.log",
            options="sec=0.1 foo=1 bar='a=b c=d' xx_zz=yy",
            suffixes=['.*'],
        )
        results.write(filename=currdir + "test_options.txt", times=False)
        _out, _log = join(currdir, "test_options.txt"), join(currdir, "test4_asl.txt")
        self.assertTrue(cmp(_out, _log), msg="Files %s and %s differ" % (_out, _log))
        # os.remove(currdir+"test4.sol")
        # os.remove(currdir+"test_solve4.log")

    def test_error1(self):
        """Bad results format"""
        try:
            model = ConcreteModel()
            results = self.asl.solve(model, format=ResultsFormat.sol, suffixes=['.*'])
            self.fail("test_error1")
        except ValueError:
            pass

    def test_error2(self):
        """Bad solve option"""
        try:
            model = ConcreteModel()
            results = self.asl.solve(model, foo="bar")
            self.fail("test_error2")
        except ValueError:
            pass

    def test_error3(self):
        """Bad solve option"""
        try:
            results = self.asl.solve(currdir + "model.py", foo="bar")
            self.fail("test_error3")
        except ValueError:
            pass


class mip_all(mock_all):
    def setUp(self):
        self.do_setup(True)


if __name__ == "__main__":
    deleteFiles = False
    unittest.main()

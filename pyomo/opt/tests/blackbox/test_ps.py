#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for pyomo.opt.blackbox.ps
#

import os
from os.path import abspath, dirname
pyomodir = dirname(dirname(dirname(dirname(abspath(__file__)))))
pyomodir += os.sep
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

import pyomo.opt
import pyomo.opt.blackbox

old_tempdir = pyutilib.services.TempfileManager.tempdir

class TestProblem1(pyomo.opt.blackbox.MixedIntOptProblem):

    def __init__(self):
        pyomo.opt.blackbox.MixedIntOptProblem.__init__(self)
        self.real_lower=[0.0, -1.0, 1.0, None]
        self.real_upper=[None, 0.0, 2.0, -1.0]
        self.nreal=4

    def function_value(self, point):
        self.validate(point)
        return point.reals[0] - point.reals[1] + (point.reals[2]-1.5)**2 + (point.reals[3]+2)**4




class OptPatternSearchDebug(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def setUp(self):
        self.do_setup(False)
        pyutilib.services.TempfileManager.tempdir = currdir

    def do_setup(self,flag):
        pyutilib.services.TempfileManager.tempdir = currdir
        self.ps = pyomo.opt.SolverFactory('ps')

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir

    def test_solve1(self):
        """ Test PatternSearch - test1.mps """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        results = self.ps.solve(logfile=currdir+"test_solve1.log")
        results.write(filename=currdir+"test_solve1.txt", times=False, format='json')
        self.assertMatchesJsonBaseline(currdir+"test_solve1.txt", currdir+"test1_ps.txt")
        if os.path.exists(currdir+"test_solve1.log"):
            os.remove(currdir+"test_solve1.log")

    def test_solve2(self):
        """ Test PatternSearch - test1.mps """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, -0.5, 2.0, -1.0]
        self.ps.reset()
        self.ps.min_function_value = 1e-1
        results = self.ps.solve(logfile=currdir+"test_solve2.log")
        results.write(filename=currdir+"test_solve2.txt", times=False, format='json')
        self.assertMatchesJsonBaseline(currdir+"test_solve2.txt", currdir+"test2_ps.txt")
        if os.path.exists(currdir+"test_solve2.log"):
            os.remove(currdir+"test_solve2.log")

    def test_error1(self):
        """ An error is generated when no initial point is provided """
        problem=TestProblem1()
        self.ps.problem=problem
        try:
            self.ps.reset()
            self.fail("Excepted ValueError")
        except ValueError:
            pass

    def test_error2(self):
        """ An error is generated when the initial point is infeasible """
        problem=TestProblem1()
        self.ps.problem=problem
        self.ps.initial_point = [1.0, 0.5, 2.0, -1.0]
        self.assertEqual(self.ps.feasible(self.ps.initial_point), True)
        self.real_lower=[0.0, -1.0, 1.0, None]
        self.real_upper=[None, 0.0, 2.0, -1.0]
        try:
            self.ps.reset()
            self.fail("Excepted ValueError")
        except ValueError:
            pass
        self.ps.initial_point = [-1.0, -0.5, 2.0, -1.0]
        self.real_lower=[0.0, -1.0, 1.0, None]
        self.real_upper=[None, 0.0, 2.0, -1.0]
        try:
            self.ps.reset()
            self.fail("Excepted ValueError")
        except ValueError:
            pass


if __name__ == "__main__":
    unittest.main()

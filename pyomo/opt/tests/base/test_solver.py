#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for util/misc
#

import os
from os.path import abspath, dirname
pyomodir = dirname(abspath(__file__))+"/../.."
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

import pyomo.util.plugin
import pyomo.opt

old_tempdir = pyutilib.services.TempfileManager.tempdir

class TestSolver1(pyomo.opt.OptSolver):

    pyomo.util.plugin.alias('stest1')

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        kwds['doc'] = 'TestSolver1 Documentation'
        pyomo.opt.OptSolver.__init__(self,**kwds)

    def enabled(self):
        return False


class OptSolverDebug(unittest.TestCase):

    def setUp(self):
        pyomo.opt.SolverFactory.activate('stest1')
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir


    def test_solver_init1(self):
        """
        Verify the processing of 'type', 'name' and 'doc' options
        """
        ans = pyomo.opt.SolverFactory("stest1")
        self.assertEqual(type(ans), TestSolver1)
        self.assertEqual(ans._doc, "TestSolver1 Documentation")

        ans = pyomo.opt.SolverFactory("stest1", doc="My Doc")
        self.assertEqual(type(ans), TestSolver1)
        self.assertEqual(ans._doc, "TestSolver1 Documentation")

        ans = pyomo.opt.SolverFactory("stest1", name="my name")
        self.assertEqual(type(ans), TestSolver1)
        self.assertEqual(ans._doc, "TestSolver1 Documentation")

    def test_solver_init2(self):
        """
        Verify that options can be passed in.
        """
        opt = {}
        opt['a'] = 1
        opt['b'] = "two"
        ans = pyomo.opt.SolverFactory("stest1", name="solver_init2", options=opt)
        self.assertEqual(ans.options['a'], opt['a'])
        self.assertEqual(ans.options['b'], opt['b'])

    def test_avail(self):
        ans = pyomo.opt.SolverFactory("stest1")
        try:
            ans.available()
            self.fail("Expected exception for 'stest1' solver, which is disabled")
        except pyutilib.common.ApplicationError:
            pass

    def test_problem_format(self):
        opt = pyomo.opt.SolverFactory("stest1")
        opt._problem_format = 'a'
        self.assertEqual(opt.problem_format(), 'a')
        opt._problem_format = None
        self.assertEqual(opt.problem_format(), None)

    def test_results_format(self):
        opt = pyomo.opt.SolverFactory("stest1")
        opt._results_format = 'a'
        self.assertEqual(opt.results_format(), 'a')
        opt._results_format = None
        self.assertEqual(opt.results_format(), None)

    def test_set_problem_format(self):
        opt = pyomo.opt.SolverFactory("stest1")
        opt._valid_problem_formats = []
        try:
            opt.set_problem_format('a')
        except ValueError:
            pass
        else:
            self.fail("Should not be able to set the problem format undless it's declared as valid.")
        opt._valid_problem_formats = ['a']
        self.assertEqual(opt.results_format(), None)
        opt.set_problem_format('a')
        self.assertEqual(opt.problem_format(), 'a')
        self.assertEqual(opt.results_format(), opt._default_results_format('a'))

    def test_set_results_format(self):
        opt = pyomo.opt.SolverFactory("stest1")
        opt._valid_problem_formats = ['a']
        opt._valid_results_formats = {'a':'b'}
        self.assertEqual(opt.problem_format(), None)
        try:
            opt.set_results_format('b')
        except ValueError:
            pass
        else:
            self.fail("Should not be able to set the results format unless it's "\
                      "declared as valid for the current problem format.")
        opt.set_problem_format('a')
        self.assertEqual(opt.problem_format(), 'a')
        opt.set_results_format('b')
        self.assertEqual(opt.results_format(), 'b')


if __name__ == "__main__":
    unittest.main()

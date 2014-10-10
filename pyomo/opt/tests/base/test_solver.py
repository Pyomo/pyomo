#
# Unit Tests for util/misc
#
#

import os
from os.path import abspath, dirname
cooprdir = dirname(abspath(__file__))+"/../.."
currdir = dirname(abspath(__file__))+os.sep

import unittest
from nose.tools import nottest
import coopr.opt
import coopr
import pyutilib.services
from coopr.core.plugin import alias

old_tempdir = pyutilib.services.TempfileManager.tempdir

class TestSolver1(coopr.opt.OptSolver):

    alias('stest1')

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        kwds['doc'] = 'TestSolver1 Documentation'
        coopr.opt.OptSolver.__init__(self,**kwds)

    def enabled(self):
        return False


class OptSolverDebug(unittest.TestCase):

    def setUp(self):
        coopr.opt.SolverFactory.activate('stest1')
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir


    def test_solver_init1(self):
        """
        Verify the processing of 'type', 'name' and 'doc' options
        """
        ans = coopr.opt.SolverFactory("stest1")
        self.assertEqual(type(ans), TestSolver1)
        self.assertEqual(ans._doc, "TestSolver1 Documentation")

        ans = coopr.opt.SolverFactory("stest1", doc="My Doc")
        self.assertEqual(type(ans), TestSolver1)
        self.assertEqual(ans._doc, "TestSolver1 Documentation")

        ans = coopr.opt.SolverFactory("stest1", name="my name")
        self.assertEqual(type(ans), TestSolver1)
        self.assertEqual(ans._doc, "TestSolver1 Documentation")

    def test_solver_init2(self):
        """
        Verify that options can be passed in.
        """
        opt = {}
        opt['a'] = 1
        opt['b'] = "two"
        ans = coopr.opt.SolverFactory("stest1", name="solver_init2", options=opt)
        self.assertEqual(ans.options['a'], opt['a'])
        self.assertEqual(ans.options['b'], opt['b'])

    def test_avail(self):
        ans = coopr.opt.SolverFactory("stest1")
        try:
            ans.available()
            self.fail("Expected exception for 'stest1' solver, which is disabled")
        except pyutilib.common.ApplicationError:
            pass

    def test_problem_format(self):
        opt = coopr.opt.SolverFactory("stest1")
        opt._problem_format = 'a'
        self.assertEqual(opt.problem_format(), 'a')
        opt._problem_format = None
        self.assertEqual(opt.problem_format(), None)

    def test_results_format(self):
        opt = coopr.opt.SolverFactory("stest1")
        opt._results_format = 'a'
        self.assertEqual(opt.results_format(), 'a')
        opt._results_format = None
        self.assertEqual(opt.results_format(), None)

    def test_set_problem_format(self):
        opt = coopr.opt.SolverFactory("stest1")
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
        opt = coopr.opt.SolverFactory("stest1")
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

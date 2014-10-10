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
from coopr.opt.base.solvers import UnknownSolver
import coopr.opt.plugins.sol
import coopr
import pyutilib.services
import coopr.core.plugin
import coopr.environ

old_tempdir = pyutilib.services.TempfileManager.tempdir

coopr.core.plugin.push('coopr.opt')


class TestWriter(coopr.opt.AbstractProblemWriter):

    coopr.core.plugin.alias('wtest')

    def __init__(self, name=None):
        coopr.opt.AbstractProblemWriter.__init__(self,name)


class TestReader(coopr.opt.AbstractResultsReader):

    coopr.core.plugin.alias('rtest')

    def __init__(self, name=None):
        coopr.opt.AbstractResultsReader.__init__(self,name)


class TestSolver(coopr.opt.OptSolver):

    coopr.core.plugin.alias('stest')

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        kwds['doc'] = 'TestSolver Documentation'
        coopr.opt.OptSolver.__init__(self,**kwds)

    def enabled(self):
        return False


coopr.core.plugin.pop()


class Test(unittest.TestCase):

    def run(self, result=None):
        unittest.TestCase.run(self,result)

    def setUp(self):
        coopr.opt.SolverFactory.activate('stest')
        coopr.opt.WriterFactory.activate('wtest')
        coopr.opt.ReaderFactory.activate('rtest')
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir
        coopr.opt.SolverFactory.deactivate('stest')
        coopr.opt.WriterFactory.deactivate('wtest')
        coopr.opt.ReaderFactory.deactivate('rtest')

    def test_solver_factory(self):
        #"""
        #Testing the coopr.opt solver factory
        #"""
        ans = sorted(coopr.opt.SolverFactory.services())
        #self.assertEqual(len(ans),8)
        self.assertTrue(set(['stest']) <= set(ans))

    def test_solver_instance(self):
        #"""
        #Testing that we get a specific solver instance
        #"""
        ans = coopr.opt.SolverFactory("none")
        self.assertTrue(isinstance(ans, UnknownSolver))
        ans = coopr.opt.SolverFactory("stest")
        self.assertEqual(type(ans), TestSolver)
        ans = coopr.opt.SolverFactory("stest", name="mymock")
        self.assertEqual(type(ans), TestSolver)
        self.assertEqual(ans.name,  "mymock")

    def test_solver_registration(self):
        #"""
        #Testing methods in the solverwriter factory registration process
        #"""
        coopr.opt.SolverFactory.deactivate("stest")
        self.assertTrue(not 'stest' in coopr.opt.SolverFactory.services())
        coopr.opt.SolverFactory.activate("stest")
        self.assertTrue('stest' in coopr.opt.SolverFactory.services())

    def test_writer_factory(self):
        #"""
        #Testing the coopr.opt writer factory
        #"""
        factory = coopr.opt.WriterFactory.services()
        self.assertTrue(set(['wtest']) <= set(factory))

    def test_writer_instance(self):
        #"""
        #Testing that we get a specific writer instance
        #
        #Note: this simply provides code coverage right now, but
        #later it should be adapted to generate a specific writer.
        #"""
        ans = coopr.opt.WriterFactory("none")
        self.assertEqual(ans, None)
        ans = coopr.opt.WriterFactory("wtest")
        self.assertNotEqual(ans, None)

    def test_writer_registration(self):
        #"""
        #Testing methods in the writer factory registration process
        #"""
        coopr.opt.WriterFactory.deactivate("wtest")
        self.assertTrue(not 'wtest' in coopr.opt.WriterFactory.services())
        coopr.opt.WriterFactory.activate("wtest")
        self.assertTrue('wtest' in coopr.opt.WriterFactory.services())


    def test_reader_factory(self):
        #"""
        #Testing the coopr.opt reader factory
        #"""
        ans = coopr.opt.ReaderFactory.services()
        self.assertTrue(set(ans) >= set(["rtest", "sol", "yaml", "json"]))

    def test_reader_instance(self):
        #"""
        #Testing that we get a specific reader instance
        #"""
        ans = coopr.opt.ReaderFactory("none")
        self.assertEqual(ans, None)
        ans = coopr.opt.ReaderFactory("sol")
        self.assertEqual(type(ans), coopr.opt.plugins.sol.ResultsReader_sol)
        #ans = coopr.opt.ReaderFactory("osrl", "myreader")
        #self.assertEqual(type(ans), coopr.opt.reader.OS.ResultsReader_osrl)
        #self.assertEqual(ans.name, "myreader")

    def test_reader_registration(self):
        #"""
        #Testing methods in the reader factory registration process
        #"""
        coopr.opt.ReaderFactory.deactivate("rtest")
        self.assertTrue(not 'rtest' in coopr.opt.ReaderFactory.services())
        coopr.opt.ReaderFactory.activate("rtest")
        self.assertTrue('rtest' in coopr.opt.ReaderFactory.services())

if __name__ == "__main__":
    unittest.main()

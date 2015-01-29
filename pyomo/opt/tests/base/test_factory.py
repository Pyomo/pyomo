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
import pyomo.opt.plugins.sol
from pyomo.opt.base.solvers import UnknownSolver

old_tempdir = pyutilib.services.TempfileManager.tempdir

pyomo.util.plugin.push('pyomo.opt')


class TestWriter(pyomo.opt.AbstractProblemWriter):

    pyomo.util.plugin.alias('wtest')

    def __init__(self, name=None):
        pyomo.opt.AbstractProblemWriter.__init__(self,name)


class TestReader(pyomo.opt.AbstractResultsReader):

    pyomo.util.plugin.alias('rtest')

    def __init__(self, name=None):
        pyomo.opt.AbstractResultsReader.__init__(self,name)


class TestSolver(pyomo.opt.OptSolver):

    pyomo.util.plugin.alias('stest')

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        kwds['doc'] = 'TestSolver Documentation'
        pyomo.opt.OptSolver.__init__(self,**kwds)

    def enabled(self):
        return False


pyomo.util.plugin.pop()


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def run(self, result=None):
        unittest.TestCase.run(self,result)

    def setUp(self):
        pyomo.opt.SolverFactory.activate('stest')
        pyomo.opt.WriterFactory.activate('wtest')
        pyomo.opt.ReaderFactory.activate('rtest')
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir
        pyomo.opt.SolverFactory.deactivate('stest')
        pyomo.opt.WriterFactory.deactivate('wtest')
        pyomo.opt.ReaderFactory.deactivate('rtest')

    def test_solver_factory(self):
        #"""
        #Testing the pyomo.opt solver factory
        #"""
        ans = sorted(pyomo.opt.SolverFactory.services())
        #self.assertEqual(len(ans),8)
        self.assertTrue(set(['stest']) <= set(ans))

    def test_solver_instance(self):
        #"""
        #Testing that we get a specific solver instance
        #"""
        ans = pyomo.opt.SolverFactory("none")
        self.assertTrue(isinstance(ans, UnknownSolver))
        ans = pyomo.opt.SolverFactory("stest")
        self.assertEqual(type(ans), TestSolver)
        ans = pyomo.opt.SolverFactory("stest", name="mymock")
        self.assertEqual(type(ans), TestSolver)
        self.assertEqual(ans.name,  "mymock")

    def test_solver_registration(self):
        #"""
        #Testing methods in the solverwriter factory registration process
        #"""
        pyomo.opt.SolverFactory.deactivate("stest")
        self.assertTrue(not 'stest' in pyomo.opt.SolverFactory.services())
        pyomo.opt.SolverFactory.activate("stest")
        self.assertTrue('stest' in pyomo.opt.SolverFactory.services())

    def test_writer_factory(self):
        #"""
        #Testing the pyomo.opt writer factory
        #"""
        factory = pyomo.opt.WriterFactory.services()
        self.assertTrue(set(['wtest']) <= set(factory))

    def test_writer_instance(self):
        #"""
        #Testing that we get a specific writer instance
        #
        #Note: this simply provides code coverage right now, but
        #later it should be adapted to generate a specific writer.
        #"""
        ans = pyomo.opt.WriterFactory("none")
        self.assertEqual(ans, None)
        ans = pyomo.opt.WriterFactory("wtest")
        self.assertNotEqual(ans, None)

    def test_writer_registration(self):
        #"""
        #Testing methods in the writer factory registration process
        #"""
        pyomo.opt.WriterFactory.deactivate("wtest")
        self.assertTrue(not 'wtest' in pyomo.opt.WriterFactory.services())
        pyomo.opt.WriterFactory.activate("wtest")
        self.assertTrue('wtest' in pyomo.opt.WriterFactory.services())


    def test_reader_factory(self):
        #"""
        #Testing the pyomo.opt reader factory
        #"""
        ans = pyomo.opt.ReaderFactory.services()
        self.assertTrue(set(ans) >= set(["rtest", "sol", "yaml", "json"]))

    def test_reader_instance(self):
        #"""
        #Testing that we get a specific reader instance
        #"""
        ans = pyomo.opt.ReaderFactory("none")
        self.assertEqual(ans, None)
        ans = pyomo.opt.ReaderFactory("sol")
        self.assertEqual(type(ans), pyomo.opt.plugins.sol.ResultsReader_sol)
        #ans = pyomo.opt.ReaderFactory("osrl", "myreader")
        #self.assertEqual(type(ans), pyomo.opt.reader.OS.ResultsReader_osrl)
        #self.assertEqual(ans.name, "myreader")

    def test_reader_registration(self):
        #"""
        #Testing methods in the reader factory registration process
        #"""
        pyomo.opt.ReaderFactory.deactivate("rtest")
        self.assertTrue(not 'rtest' in pyomo.opt.ReaderFactory.services())
        pyomo.opt.ReaderFactory.activate("rtest")
        self.assertTrue('rtest' in pyomo.opt.ReaderFactory.services())

if __name__ == "__main__":
    unittest.main()

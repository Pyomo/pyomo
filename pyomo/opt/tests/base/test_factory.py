#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for util/misc
#

import os
from os.path import abspath, dirname
pyomodir = dirname(abspath(__file__))+"/../.."
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

import pyomo.opt
import pyomo.opt.plugins.sol
from pyomo.opt.base.solvers import UnknownSolver

old_tempdir = pyutilib.services.TempfileManager.tempdir


class TestWriter(pyomo.opt.AbstractProblemWriter):

    def __init__(self, name=None):
        pyomo.opt.AbstractProblemWriter.__init__(self,name)


class TestReader(pyomo.opt.AbstractResultsReader):

    def __init__(self, name=None):
        pyomo.opt.AbstractResultsReader.__init__(self,name)


class TestSolver(pyomo.opt.OptSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        kwds['doc'] = 'TestSolver Documentation'
        pyomo.opt.OptSolver.__init__(self,**kwds)


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def run(self, result=None):
        unittest.TestCase.run(self,result)

    def setUp(self):
        pyomo.opt.WriterFactory.register('wtest')(TestWriter)
        pyomo.opt.ReaderFactory.register('rtest')(TestReader)
        pyomo.opt.SolverFactory.register('stest')(TestSolver)
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir
        pyomo.opt.WriterFactory.unregister('wtest')
        pyomo.opt.ReaderFactory.unregister('rtest')
        pyomo.opt.SolverFactory.unregister('stest')

    def test_solver_factory(self):
        #"""
        #Testing the pyomo.opt solver factory
        #"""
        ans = sorted(list(pyomo.opt.SolverFactory))
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

    def test_writer_factory(self):
        #"""
        #Testing the pyomo.opt writer factory
        #"""
        factory = pyomo.opt.WriterFactory
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


    def test_reader_factory(self):
        #"""
        #Testing the pyomo.opt reader factory
        #"""
        ans = pyomo.opt.ReaderFactory
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

if __name__ == "__main__":
    unittest.main()

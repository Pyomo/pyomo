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

import pyomo
import pyomo.opt
from pyomo.opt.base.solvers import UnknownSolver

old_tempdir = None
def setUpModule():
    global old_tempdir
    old_tempdir = pyutilib.services.TempfileManager.tempdir
    pyutilib.services.TempfileManager.tempdir = currdir

def tearDownModule():
    pyutilib.services.TempfileManager.tempdir = old_tempdir

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


class OptFactoryDebug(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ
        import pyomo.solvers.plugins

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyomo.opt.ReaderFactory.unregister('rtest3')
        pyomo.opt.ReaderFactory.unregister('stest3')
        pyomo.opt.ReaderFactory.unregister('wtest3')

    def test_solver_factory(self):
        """
        Testing the pyomo.opt solver factory with MIP solvers
        """
        pyomo.opt.SolverFactory.register('stest3')(TestSolver)
        ans = sorted(pyomo.opt.SolverFactory)
        tmp = ['_mock_asl', '_mock_cbc', '_mock_cplex', '_mock_glpk', '_mock_pico', 'cbc', 'cplex', 'glpk', 'pico', 'stest3', 'asl']
        tmp.sort()
        self.assertTrue(set(tmp) <= set(ans), msg="Set %s is not a subset of set %s" %(tmp,ans))

    def test_solver_instance(self):
        """
        Testing that we get a specific solver instance
        """
        ans = pyomo.opt.SolverFactory("none")
        self.assertTrue(isinstance(ans, UnknownSolver))
        ans = pyomo.opt.SolverFactory("_mock_pico")
        self.assertEqual(type(ans), pyomo.solvers.plugins.solvers.PICO.MockPICO)
        ans = pyomo.opt.SolverFactory("_mock_pico", name="mymock")
        self.assertEqual(type(ans), pyomo.solvers.plugins.solvers.PICO.MockPICO)
        self.assertEqual(ans.name,  "mymock")

    def test_solver_registration(self):
        """
        Testing methods in the solverwriter factory registration process
        """
        pyomo.opt.SolverFactory.unregister('stest3')
        self.assertTrue('stest3' not in pyomo.opt.SolverFactory)
        pyomo.opt.SolverFactory.register('stest3')(TestSolver)
        self.assertTrue('stest3' in pyomo.opt.SolverFactory)
        self.assertTrue('_mock_pico' in pyomo.opt.SolverFactory)

    def test_writer_factory(self):
        """
        Testing the pyomo.opt writer factory with MIP writers
        """
        pyomo.opt.WriterFactory.register('wtest3')(TestWriter)
        factory = pyomo.opt.WriterFactory
        self.assertTrue(set(['wtest3']) <= set(factory))

    def test_writer_instance(self):
        """
        Testing that we get a specific writer instance

        Note: this simply provides code coverage right now, but
        later it should be adapted to generate a specific writer.
        """
        ans = pyomo.opt.WriterFactory("none")
        self.assertEqual(ans, None)
        ans = pyomo.opt.WriterFactory("wtest3")
        self.assertNotEqual(ans, None)

    def test_writer_registration(self):
        """
        Testing methods in the writer factory registration process
        """
        pyomo.opt.WriterFactory.unregister('wtest3')
        self.assertTrue(not 'wtest3' in pyomo.opt.WriterFactory)
        pyomo.opt.WriterFactory.register('wtest3')(TestWriter)
        self.assertTrue('wtest3' in pyomo.opt.WriterFactory)


    def test_reader_factory(self):
        """
        Testing the pyomo.opt reader factory
        """
        pyomo.opt.ReaderFactory.register('rtest3')(TestReader)
        ans = pyomo.opt.ReaderFactory
        #self.assertEqual(len(ans),4)
        self.assertTrue(set(ans) >= set(["rtest3", "sol","yaml", "json"]))

    def test_reader_instance(self):
        """
        Testing that we get a specific reader instance
        """
        ans = pyomo.opt.ReaderFactory("none")
        self.assertEqual(ans, None)
        ans = pyomo.opt.ReaderFactory("sol")
        self.assertEqual(type(ans), pyomo.opt.plugins.sol.ResultsReader_sol)
        #ans = pyomo.opt.ReaderFactory("osrl", "myreader")
        #self.assertEqual(type(ans), pyomo.opt.reader.OS.ResultsReader_osrl)
        #self.assertEqual(ans.name, "myreader")

    def test_reader_registration(self):
        """
        Testing methods in the reader factory registration process
        """
        pyomo.opt.ReaderFactory.unregister('rtest3')
        self.assertTrue(not 'rtest3' in pyomo.opt.ReaderFactory)
        pyomo.opt.ReaderFactory.register('rtest3')(TestReader)
        self.assertTrue('rtest3' in pyomo.opt.ReaderFactory)

if __name__ == "__main__":
    unittest.main()

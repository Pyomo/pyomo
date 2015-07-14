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

import pyomo
import pyomo.opt
from pyomo.opt.base.solvers import UnknownSolver

pyomo.util.plugin.push('pyomo.solvers.test')

old_tempdir = None
def setUpModule():
    global old_tempdir
    old_tempdir = pyutilib.services.TempfileManager.tempdir
    pyutilib.services.TempfileManager.tempdir = currdir

def tearDownModule():
    pyutilib.services.TempfileManager.tempdir = old_tempdir

class TestWriter(pyomo.opt.AbstractProblemWriter):

    pyomo.util.plugin.alias('wtest3')

    def __init__(self, name=None):
        pyomo.opt.AbstractProblemWriter.__init__(self,name)


class TestReader(pyomo.opt.AbstractResultsReader):

    pyomo.util.plugin.alias('rtest3')

    def __init__(self, name=None):
        pyomo.opt.AbstractResultsReader.__init__(self,name)


class TestSolver(pyomo.opt.OptSolver):

    pyomo.util.plugin.alias('stest3')

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        kwds['doc'] = 'TestSolver Documentation'
        pyomo.opt.OptSolver.__init__(self,**kwds)

    def enabled(self):
        return False

pyomo.util.plugin.pop()


class OptFactoryDebug(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.util.plugin
        import pyomo.environ
        import pyomo.solvers.plugins

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()

    def test_solver_factory(self):
        """
        Testing the pyomo.opt solver factory with MIP solvers
        """
        ans = sorted(pyomo.opt.SolverFactory.services())
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
        pyomo.opt.SolverFactory.deactivate('stest3')
        self.assertTrue(not 'stest3' in pyomo.opt.SolverFactory.services())
        pyomo.opt.SolverFactory.activate('stest3')
        self.assertTrue('stest3' in pyomo.opt.SolverFactory.services())
        self.assertTrue('_mock_pico' in pyomo.opt.SolverFactory.services())

    def test_writer_factory(self):
        """
        Testing the pyomo.opt writer factory with MIP writers
        """
        factory = pyomo.opt.WriterFactory.services()
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
        pyomo.opt.WriterFactory.deactivate('wtest3')
        self.assertTrue(not 'wtest3' in pyomo.opt.WriterFactory.services())
        pyomo.opt.WriterFactory.activate('wtest3')
        self.assertTrue('wtest3' in pyomo.opt.WriterFactory.services())


    def test_reader_factory(self):
        """
        Testing the pyomo.opt reader factory
        """
        ans = pyomo.opt.ReaderFactory.services()
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
        pyomo.opt.ReaderFactory.deactivate('rtest3')
        self.assertTrue(not 'rtest3' in pyomo.opt.ReaderFactory.services())
        pyomo.opt.ReaderFactory.activate('rtest3')
        self.assertTrue('rtest3' in pyomo.opt.ReaderFactory.services())

if __name__ == "__main__":
    unittest.main()

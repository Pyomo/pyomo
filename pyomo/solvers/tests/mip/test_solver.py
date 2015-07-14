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
import pyutilib.common

import pyomo.opt
import pyomo.solvers.plugins.solvers
from pyomo.util.plugin import alias

old_tempdir = None
def setUpModule():
    global old_tempdir
    old_tempdir = pyutilib.services.TempfileManager.tempdir
    pyutilib.services.TempfileManager.tempdir = currdir

def tearDownModule():
    pyutilib.services.TempfileManager.tempdir = old_tempdir

class TestSolver2(pyomo.opt.OptSolver):

    alias('stest2')

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        pyomo.opt.OptSolver.__init__(self,**kwds)

    def enabled(self):
        return False

class OptSolverDebug(unittest.TestCase):

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()

    def test_solver_init1(self):
        """
        Verify the processing of 'type', 'name' and 'doc' options
        """
        ans = pyomo.opt.SolverFactory("_mock_pico")
        self.assertEqual(type(ans), pyomo.solvers.plugins.solvers.PICO.MockPICO)
        self.assertEqual(ans._doc, "pico OptSolver")

        ans = pyomo.opt.SolverFactory("_mock_pico", doc="My Doc")
        self.assertEqual(type(ans), pyomo.solvers.plugins.solvers.PICO.MockPICO)
        self.assertEqual(ans._doc, "My Doc")

        ans = pyomo.opt.SolverFactory("_mock_pico", name="my name")
        self.assertEqual(type(ans), pyomo.solvers.plugins.solvers.PICO.MockPICO)
        self.assertEqual(ans._doc, "my name OptSolver (type pico)")

    def test_solver_init2(self):
        """
        Verify that options can be passed in.
        """
        opt = {}
        opt['a'] = 1
        opt['b'] = "two"
        ans = pyomo.opt.SolverFactory("_mock_pico", name="solver_init2", options=opt)
        self.assertEqual(ans.options['a'], opt['a'])
        self.assertEqual(ans.options['b'], opt['b'])

    def test_avail(self):
        ans = pyomo.opt.SolverFactory("stest2")
        try:
            ans.available()
            self.fail("Expected exception for 'stest2' solver, which is disabled")
        except pyutilib.common.ApplicationError:
            pass



if __name__ == "__main__":
    unittest.main()

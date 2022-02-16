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

import pyomo.common.unittest as unittest

import pyomo.opt
import pyomo.solvers.plugins.solvers

class MockSolver2(pyomo.opt.OptSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        pyomo.opt.OptSolver.__init__(self,**kwds)

    def enabled(self):
        return False


class OptSolverDebug(unittest.TestCase):

    def setUp(self):
        pyomo.opt.SolverFactory.register('stest2')(MockSolver2)

    def tearDown(self):
        pyomo.opt.SolverFactory.unregister('stest2')

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
        # No exception should be generated
        ans.available()


if __name__ == "__main__":
    unittest.main()

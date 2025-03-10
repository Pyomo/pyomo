#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
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
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC


class MockSolver2(pyomo.opt.OptSolver):
    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        pyomo.opt.OptSolver.__init__(self, **kwds)

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
        ans = pyomo.opt.SolverFactory("_mock_cbc")
        self.assertEqual(type(ans), MockCBC)
        self.assertEqual(ans._doc, "cbc OptSolver")

        ans = pyomo.opt.SolverFactory("_mock_cbc", doc="My Doc")
        self.assertEqual(type(ans), MockCBC)
        self.assertEqual(ans._doc, "My Doc")

        ans = pyomo.opt.SolverFactory("_mock_cbc", name="my name")
        self.assertEqual(type(ans), MockCBC)
        self.assertEqual(ans._doc, "my name OptSolver (type cbc)")

    def test_solver_init2(self):
        """
        Verify that options can be passed in.
        """
        opt = {}
        opt['a'] = 1
        opt['b'] = "two"
        ans = pyomo.opt.SolverFactory("_mock_cbc", name="solver_init2", options=opt)
        self.assertEqual(ans.options['a'], opt['a'])
        self.assertEqual(ans.options['b'], opt['b'])

    def test_avail(self):
        ans = pyomo.opt.SolverFactory("stest2")
        # No exception should be generated
        ans.available()


if __name__ == "__main__":
    unittest.main()

#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for pyomo.opt.base.OS
#

import os
from os.path import abspath, dirname
pyomodir = dirname(abspath(__file__))+os.sep+".."+os.sep+".."+os.sep
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

import pyomo.opt
from pyomo.opt import (TerminationCondition,
                       SolutionStatus,
                       SolverStatus)

old_tempdir = pyutilib.services.TempfileManager.tempdir

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def setUp(self):
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir
        if os.path.exists(currdir+"test_sol.txt"):
            os.remove(currdir+"test_sol.txt")

    def test_factory(self):
        with pyomo.opt.ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            soln = reader(currdir+"test4_sol.sol", suffixes=["dual"])
            soln.write(filename=currdir+"factory.txt", format='json')
            self.assertMatchesJsonBaseline(currdir+"factory.txt", currdir+"test4_sol.jsn")

    def test_infeasible1(self):
        with pyomo.opt.ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            soln = reader(currdir+"infeasible1.sol")
            self.assertEqual(soln.solver.termination_condition,
                             TerminationCondition.infeasible)
            self.assertEqual(soln.solution.status,
                             SolutionStatus.infeasible)
            self.assertEqual(soln.solver.status,
                             SolverStatus.warning)

    def test_infeasible2(self):
        with pyomo.opt.ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            soln = reader(currdir+"infeasible2.sol")
            self.assertEqual(soln.solver.termination_condition,
                             TerminationCondition.infeasible)
            self.assertEqual(soln.solution.status,
                             SolutionStatus.infeasible)
            self.assertEqual(soln.solver.status,
                             SolverStatus.warning)

    def test_conopt_optimal(self):
        with pyomo.opt.ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            soln = reader(currdir+"conopt_optimal.sol")
            self.assertEqual(soln.solver.termination_condition,
                             TerminationCondition.optimal)
            self.assertEqual(soln.solution.status,
                             SolutionStatus.optimal)
            self.assertEqual(soln.solver.status,
                             SolverStatus.ok)

    def test_bad_options(self):
        with pyomo.opt.ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            with self.assertRaises(ValueError):
                soln = reader(currdir+"bad_options.sol")

    def test_bad_objno(self):
        with pyomo.opt.ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            with self.assertRaises(ValueError):
                soln = reader(currdir+"bad_objno.sol")

    def test_bad_objnoline(self):
        with pyomo.opt.ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            with self.assertRaises(ValueError):
                soln = reader(currdir+"bad_objnoline.sol")

if __name__ == "__main__":
    unittest.main()

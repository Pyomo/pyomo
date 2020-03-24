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

    def test_iis_no_variable_values(self):
        with pyomo.opt.ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            result = reader(currdir+"iis_no_variable_values.sol", suffixes=["iis"])
            soln = result.solution(0)
            self.assertEqual(len(list(soln.variable['v0'].keys())), 1)
            self.assertEqual(soln.variable['v0']['iis'], 1)
            self.assertEqual(len(list(soln.variable['v1'].keys())), 1)
            self.assertEqual(soln.variable['v1']['iis'], 1)
            self.assertEqual(len(list(soln.constraint['c0'].keys())), 1)
            self.assertEqual(soln.constraint['c0']['Iis'], 4)
            import pyomo.kernel as pmo
            m = pmo.block()
            m.v0 = pmo.variable()
            m.v1 = pmo.variable()
            m.c0 = pmo.constraint()
            m.iis = pmo.suffix(direction=pmo.suffix.IMPORT)
            from pyomo.core.expr.symbol_map import SymbolMap
            soln.symbol_map = SymbolMap()
            soln.symbol_map.addSymbol(m.v0, 'v0')
            soln.symbol_map.addSymbol(m.v1, 'v1')
            soln.symbol_map.addSymbol(m.c0, 'c0')
            m.load_solution(soln)
            pmo.pprint(m.iis)
            self.assertEqual(m.iis[m.v0], 1)
            self.assertEqual(m.iis[m.v1], 1)
            self.assertEqual(m.iis[m.c0], 4)

if __name__ == "__main__":
    unittest.main()

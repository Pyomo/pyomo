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
# Unit Tests for pyomo.opt.base.OS
#

import json
import os
from os.path import join

import pyomo.common.unittest as unittest

from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager

from pyomo.opt import (
    TerminationCondition,
    ReaderFactory,
    SolutionStatus,
    SolverStatus,
    check_optimal_termination,
    assert_optimal_termination,
)

currdir = this_file_dir()
deleteFiles = True


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    def setUp(self):
        TempfileManager.push()

    def tearDown(self):
        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())

    def test_factory(self):
        with ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            soln = reader(join(currdir, "test4_sol.sol"), suffixes=["dual"])
            _test = TempfileManager.create_tempfile('factory.txt')
            soln.write(filename=_test, format='json')
            with (
                open(_test, 'r') as out,
                open(join(currdir, "test4_sol.jsn"), 'r') as txt,
            ):
                self.assertStructuredAlmostEqual(
                    json.load(txt), json.load(out), allow_second_superset=True
                )

    def test_infeasible1(self):
        with ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            soln = reader(join(currdir, "infeasible1.sol"))
            self.assertEqual(
                soln.solver.termination_condition, TerminationCondition.infeasible
            )
            self.assertEqual(soln.solution.status, SolutionStatus.infeasible)
            self.assertEqual(soln.solver.status, SolverStatus.warning)

            self.assertFalse(check_optimal_termination(soln))

            with self.assertRaises(RuntimeError):
                assert_optimal_termination(soln)

    def test_infeasible2(self):
        with ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            soln = reader(join(currdir, "infeasible2.sol"))
            self.assertEqual(
                soln.solver.termination_condition, TerminationCondition.infeasible
            )
            self.assertEqual(soln.solution.status, SolutionStatus.infeasible)
            self.assertEqual(soln.solver.status, SolverStatus.warning)

    def test_conopt_optimal(self):
        with ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            soln = reader(join(currdir, "conopt_optimal.sol"))
            self.assertEqual(
                soln.solver.termination_condition, TerminationCondition.optimal
            )
            self.assertEqual(soln.solution.status, SolutionStatus.optimal)
            self.assertEqual(soln.solver.status, SolverStatus.ok)
            self.assertTrue(check_optimal_termination(soln))
            assert_optimal_termination(soln)

    def test_bad_options(self):
        with ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            with self.assertRaises(ValueError):
                soln = reader(join(currdir, "bad_options.sol"))

    def test_bad_objno(self):
        with ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            with self.assertRaises(ValueError):
                soln = reader(join(currdir, "bad_objno.sol"))

    def test_bad_objnoline(self):
        with ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            with self.assertRaises(ValueError):
                soln = reader(join(currdir, "bad_objnoline.sol"))

    def test_iis_no_variable_values(self):
        with ReaderFactory("sol") as reader:
            if reader is None:
                raise IOError("Reader 'sol' is not registered")
            result = reader(
                join(currdir, "iis_no_variable_values.sol"), suffixes=["iis"]
            )
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
    deleteFiles = False
    unittest.main()

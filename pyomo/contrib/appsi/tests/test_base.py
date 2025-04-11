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

from pyomo.common import unittest
from pyomo.contrib import appsi
import pyomo.environ as pyo
from pyomo.core.base.var import ScalarVar


class TestResults(unittest.TestCase):
    def test_uninitialized(self):
        res = appsi.base.Results()
        self.assertIsNone(res.best_feasible_objective)
        self.assertIsNone(res.best_objective_bound)
        self.assertEqual(
            res.termination_condition, appsi.base.TerminationCondition.unknown
        )

        with self.assertRaisesRegex(
            RuntimeError, '.*does not currently have a valid solution.*'
        ):
            res.solution_loader.load_vars()
        with self.assertRaisesRegex(
            RuntimeError, '.*does not currently have valid duals.*'
        ):
            res.solution_loader.get_duals()
        with self.assertRaisesRegex(
            RuntimeError, '.*does not currently have valid reduced costs.*'
        ):
            res.solution_loader.get_reduced_costs()
        with self.assertRaisesRegex(
            RuntimeError, '.*does not currently have valid slacks.*'
        ):
            res.solution_loader.get_slacks()

    def test_results(self):
        m = pyo.ConcreteModel()
        m.x = ScalarVar()
        m.y = ScalarVar()
        m.c1 = pyo.Constraint(expr=m.x == 1)
        m.c2 = pyo.Constraint(expr=m.y == 2)

        primals = dict()
        primals[id(m.x)] = (m.x, 1)
        primals[id(m.y)] = (m.y, 2)
        duals = dict()
        duals[m.c1] = 3
        duals[m.c2] = 4
        rc = dict()
        rc[id(m.x)] = (m.x, 5)
        rc[id(m.y)] = (m.y, 6)
        slacks = dict()
        slacks[m.c1] = 7
        slacks[m.c2] = 8

        res = appsi.base.Results()
        res.solution_loader = appsi.base.SolutionLoader(
            primals=primals, duals=duals, slacks=slacks, reduced_costs=rc
        )

        res.solution_loader.load_vars()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 2)

        m.x.value = None
        m.y.value = None

        res.solution_loader.load_vars([m.y])
        self.assertIsNone(m.x.value)
        self.assertAlmostEqual(m.y.value, 2)

        duals2 = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c1], duals2[m.c1])
        self.assertAlmostEqual(duals[m.c2], duals2[m.c2])

        duals2 = res.solution_loader.get_duals([m.c2])
        self.assertNotIn(m.c1, duals2)
        self.assertAlmostEqual(duals[m.c2], duals2[m.c2])

        rc2 = res.solution_loader.get_reduced_costs()
        self.assertAlmostEqual(rc[id(m.x)][1], rc2[m.x])
        self.assertAlmostEqual(rc[id(m.y)][1], rc2[m.y])

        rc2 = res.solution_loader.get_reduced_costs([m.y])
        self.assertNotIn(m.x, rc2)
        self.assertAlmostEqual(rc[id(m.y)][1], rc2[m.y])

        slacks2 = res.solution_loader.get_slacks()
        self.assertAlmostEqual(slacks[m.c1], slacks2[m.c1])
        self.assertAlmostEqual(slacks[m.c2], slacks2[m.c2])

        slacks2 = res.solution_loader.get_slacks([m.c2])
        self.assertNotIn(m.c1, slacks2)
        self.assertAlmostEqual(slacks[m.c2], slacks2[m.c2])

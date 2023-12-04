#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.solver import results
from pyomo.solver import solution
import pyomo.environ as pyo
from pyomo.core.base.var import ScalarVar


class TestTerminationCondition(unittest.TestCase):
    def test_member_list(self):
        member_list = results.TerminationCondition._member_names_
        expected_list = [
            'unknown',
            'convergenceCriteriaSatisfied',
            'maxTimeLimit',
            'iterationLimit',
            'objectiveLimit',
            'minStepLength',
            'unbounded',
            'provenInfeasible',
            'locallyInfeasible',
            'infeasibleOrUnbounded',
            'error',
            'interrupted',
            'licensingProblems',
        ]
        self.assertEqual(member_list.sort(), expected_list.sort())

    def test_codes(self):
        self.assertEqual(results.TerminationCondition.unknown.value, 42)
        self.assertEqual(
            results.TerminationCondition.convergenceCriteriaSatisfied.value, 0
        )
        self.assertEqual(results.TerminationCondition.maxTimeLimit.value, 1)
        self.assertEqual(results.TerminationCondition.iterationLimit.value, 2)
        self.assertEqual(results.TerminationCondition.objectiveLimit.value, 3)
        self.assertEqual(results.TerminationCondition.minStepLength.value, 4)
        self.assertEqual(results.TerminationCondition.unbounded.value, 5)
        self.assertEqual(results.TerminationCondition.provenInfeasible.value, 6)
        self.assertEqual(results.TerminationCondition.locallyInfeasible.value, 7)
        self.assertEqual(results.TerminationCondition.infeasibleOrUnbounded.value, 8)
        self.assertEqual(results.TerminationCondition.error.value, 9)
        self.assertEqual(results.TerminationCondition.interrupted.value, 10)
        self.assertEqual(results.TerminationCondition.licensingProblems.value, 11)


class TestSolutionStatus(unittest.TestCase):
    def test_member_list(self):
        member_list = results.SolutionStatus._member_names_
        expected_list = ['noSolution', 'infeasible', 'feasible', 'optimal']
        self.assertEqual(member_list, expected_list)

    def test_codes(self):
        self.assertEqual(results.SolutionStatus.noSolution.value, 0)
        self.assertEqual(results.SolutionStatus.infeasible.value, 10)
        self.assertEqual(results.SolutionStatus.feasible.value, 20)
        self.assertEqual(results.SolutionStatus.optimal.value, 30)


class TestResults(unittest.TestCase):
    def test_declared_items(self):
        res = results.Results()
        expected_declared = {
            'extra_info',
            'incumbent_objective',
            'iteration_count',
            'objective_bound',
            'solution_loader',
            'solution_status',
            'solver_name',
            'solver_version',
            'termination_condition',
            'timing_info',
        }
        actual_declared = res._declared
        self.assertEqual(expected_declared, actual_declared)

    def test_uninitialized(self):
        res = results.Results()
        self.assertIsNone(res.incumbent_objective)
        self.assertIsNone(res.objective_bound)
        self.assertEqual(
            res.termination_condition, results.TerminationCondition.unknown
        )
        self.assertEqual(res.solution_status, results.SolutionStatus.noSolution)
        self.assertIsNone(res.solver_name)
        self.assertIsNone(res.solver_version)
        self.assertIsNone(res.iteration_count)
        self.assertIsInstance(res.timing_info, ConfigDict)
        self.assertIsInstance(res.extra_info, ConfigDict)
        self.assertIsNone(res.timing_info.start_timestamp)
        self.assertIsNone(res.timing_info.wall_time)
        self.assertIsNone(res.timing_info.solver_wall_time)
        res.solution_loader = solution.SolutionLoader(None, None, None, None)

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

        primals = {}
        primals[id(m.x)] = (m.x, 1)
        primals[id(m.y)] = (m.y, 2)
        duals = {}
        duals[m.c1] = 3
        duals[m.c2] = 4
        rc = {}
        rc[id(m.x)] = (m.x, 5)
        rc[id(m.y)] = (m.y, 6)
        slacks = {}
        slacks[m.c1] = 7
        slacks[m.c2] = 8

        res = results.Results()
        res.solution_loader = solution.SolutionLoader(
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

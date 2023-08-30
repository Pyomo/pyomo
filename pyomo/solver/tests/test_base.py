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
from pyomo.solver import base
import pyomo.environ as pe
from pyomo.core.base.var import ScalarVar


class TestTerminationCondition(unittest.TestCase):
    def test_member_list(self):
        member_list = base.TerminationCondition._member_names_
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
        self.assertEqual(member_list, expected_list)

    def test_codes(self):
        self.assertEqual(base.TerminationCondition.unknown.value, 42)
        self.assertEqual(
            base.TerminationCondition.convergenceCriteriaSatisfied.value, 0
        )
        self.assertEqual(base.TerminationCondition.maxTimeLimit.value, 1)
        self.assertEqual(base.TerminationCondition.iterationLimit.value, 2)
        self.assertEqual(base.TerminationCondition.objectiveLimit.value, 3)
        self.assertEqual(base.TerminationCondition.minStepLength.value, 4)
        self.assertEqual(base.TerminationCondition.unbounded.value, 5)
        self.assertEqual(base.TerminationCondition.provenInfeasible.value, 6)
        self.assertEqual(base.TerminationCondition.locallyInfeasible.value, 7)
        self.assertEqual(base.TerminationCondition.infeasibleOrUnbounded.value, 8)
        self.assertEqual(base.TerminationCondition.error.value, 9)
        self.assertEqual(base.TerminationCondition.interrupted.value, 10)
        self.assertEqual(base.TerminationCondition.licensingProblems.value, 11)


class TestSolutionStatus(unittest.TestCase):
    def test_member_list(self):
        member_list = base.SolutionStatus._member_names_
        expected_list = ['noSolution', 'infeasible', 'feasible', 'optimal']
        self.assertEqual(member_list, expected_list)

    def test_codes(self):
        self.assertEqual(base.SolutionStatus.noSolution.value, 0)
        self.assertEqual(base.SolutionStatus.infeasible.value, 10)
        self.assertEqual(base.SolutionStatus.feasible.value, 20)
        self.assertEqual(base.SolutionStatus.optimal.value, 30)


class TestSolverBase(unittest.TestCase):
    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_solver_base(self):
        self.instance = base.SolverBase()
        self.assertFalse(self.instance.is_persistent())
        self.assertEqual(self.instance.version(), None)
        self.assertEqual(self.instance.config, None)
        self.assertEqual(self.instance.solve(None), None)
        self.assertEqual(self.instance.available(), None)

    @unittest.mock.patch.multiple(base.SolverBase, __abstractmethods__=set())
    def test_solver_availability(self):
        self.instance = base.SolverBase()
        self.instance.Availability._value_ = 1
        self.assertTrue(self.instance.Availability.__bool__(self.instance.Availability))
        self.instance.Availability._value_ = -1
        self.assertFalse(
            self.instance.Availability.__bool__(self.instance.Availability)
        )


class TestPersistentSolverBase(unittest.TestCase):
    def test_abstract_member_list(self):
        expected_list = ['remove_params',
                         'version',
                         'config',
                         'update_variables',
                         'remove_variables',
                         'add_constraints',
                         'get_primals',
                         'set_instance',
                         'set_objective',
                         'update_params',
                         'remove_block',
                         'add_block',
                         'available',
                         'update_config',
                         'add_params',
                         'remove_constraints',
                         'add_variables',
                         'solve']
        member_list = list(base.PersistentSolverBase.__abstractmethods__)
        self.assertEqual(sorted(expected_list), sorted(member_list))

    @unittest.mock.patch.multiple(base.PersistentSolverBase, __abstractmethods__=set())
    def test_persistent_solver_base(self):
        self.instance = base.PersistentSolverBase()
        self.assertTrue(self.instance.is_persistent())
        self.assertEqual(self.instance.get_primals(), None)
        self.assertEqual(self.instance.update_config, None)
        self.assertEqual(self.instance.set_instance(None), None)
        self.assertEqual(self.instance.add_variables(None), None)
        self.assertEqual(self.instance.add_params(None), None)
        self.assertEqual(self.instance.add_constraints(None), None)
        self.assertEqual(self.instance.add_block(None), None)
        self.assertEqual(self.instance.remove_variables(None), None)
        self.assertEqual(self.instance.remove_params(None), None)
        self.assertEqual(self.instance.remove_constraints(None), None)
        self.assertEqual(self.instance.remove_block(None), None)
        self.assertEqual(self.instance.set_objective(None), None)
        self.assertEqual(self.instance.update_variables(None), None)
        self.assertEqual(self.instance.update_params(), None)
        with self.assertRaises(NotImplementedError):
            self.instance.get_duals()
        with self.assertRaises(NotImplementedError):
            self.instance.get_slacks()
        with self.assertRaises(NotImplementedError):
            self.instance.get_reduced_costs()


class TestResults(unittest.TestCase):
    def test_uninitialized(self):
        res = base.Results()
        self.assertIsNone(res.incumbent_objective)
        self.assertIsNone(res.objective_bound)
        self.assertEqual(res.termination_condition, base.TerminationCondition.unknown)

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
        m = pe.ConcreteModel()
        m.x = ScalarVar()
        m.y = ScalarVar()
        m.c1 = pe.Constraint(expr=m.x == 1)
        m.c2 = pe.Constraint(expr=m.y == 2)

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

        res = base.Results()
        res.solution_loader = base.SolutionLoader(
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

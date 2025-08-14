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

from io import StringIO
from typing import Sequence, Dict, Optional, Mapping, MutableMapping

import pyomo.environ as pyo
from pyomo.common.config import ConfigDict
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.common.collections import ComponentMap
from pyomo.contrib.solver.common import results
from pyomo.contrib.solver.common import solution_loader
from pyomo.core.base.var import Var
from pyomo.common import unittest


class SolutionLoaderExample(solution_loader.SolutionLoaderBase):
    """
    This is an example instantiation of a SolutionLoader that is used for
    testing generated results.
    """

    def __init__(
        self,
        primals: Optional[MutableMapping],
        duals: Optional[MutableMapping],
        reduced_costs: Optional[MutableMapping],
    ):
        """
        Parameters
        ----------
        primals: dict
            maps id(Var) to (var, value)
        duals: dict
            maps Constraint to dual value
        reduced_costs: dict
            maps id(Var) to (var, reduced_cost)
        """
        self._primals = primals
        self._duals = duals
        self._reduced_costs = reduced_costs

    def get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if self._primals is None:
            raise RuntimeError(
                'Solution loader does not currently have a valid solution. Please '
                'check the termination condition.'
            )
        if vars_to_load is None:
            return ComponentMap(self._primals.values())
        else:
            primals = ComponentMap()
            for v in vars_to_load:
                primals[v] = self._primals[id(v)][1]
            return primals

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Dict[ConstraintData, float]:
        if self._duals is None:
            raise RuntimeError(
                'Solution loader does not currently have valid duals. Please '
                'check the termination condition and ensure the solver returns duals '
                'for the given problem type.'
            )
        if cons_to_load is None:
            duals = dict(self._duals)
        else:
            duals = {}
            for c in cons_to_load:
                duals[c] = self._duals[c]
        return duals

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if self._reduced_costs is None:
            raise RuntimeError(
                'Solution loader does not currently have valid reduced costs. Please '
                'check the termination condition and ensure the solver returns reduced '
                'costs for the given problem type.'
            )
        if vars_to_load is None:
            rc = ComponentMap(self._reduced_costs.values())
        else:
            rc = ComponentMap()
            for v in vars_to_load:
                rc[v] = self._reduced_costs[id(v)][1]
        return rc


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
    def test_member_list(self):
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
            'solver_log',
            'solver_config',
        }
        actual_declared = res._declared
        self.assertEqual(expected_declared, actual_declared)

    def test_default_initialization(self):
        res = results.Results()
        self.assertIsNone(res.solution_loader)
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

    def test_display(self):
        res = results.Results()
        stream = StringIO()
        res.display(ostream=stream)
        expected_print = """termination_condition: TerminationCondition.unknown
solution_status: SolutionStatus.noSolution
incumbent_objective: None
objective_bound: None
solver_name: None
solver_version: None
iteration_count: None
timing_info:
  start_timestamp: None
  wall_time: None
extra_info:
"""
        out = stream.getvalue()
        if 'null' in out:
            out = out.replace('null', 'None')
        self.assertEqual(expected_print, out)

    def test_generated_results(self):
        m = pyo.ConcreteModel()
        m.x = Var()
        m.y = Var()
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

        res = results.Results()
        res.solution_loader = SolutionLoaderExample(
            primals=primals, duals=duals, reduced_costs=rc
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

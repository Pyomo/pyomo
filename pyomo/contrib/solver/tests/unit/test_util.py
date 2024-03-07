#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib.solver.util import (
    collect_vars_and_named_exprs,
    get_objective,
    check_optimal_termination,
    assert_optimal_termination,
    SolverStatus,
    LegacyTerminationCondition,
)
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
from typing import Callable
from pyomo.common.gsl import find_GSL
from pyomo.opt.results import SolverResults


class TestGenericUtils(unittest.TestCase):
    def basics_helper(self, collector: Callable, *args):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.E = pyo.Expression(expr=2 * m.z + 1)
        m.y.fix(3)
        e = m.x * m.y + m.x * m.E
        named_exprs, var_list, fixed_vars, external_funcs = collector(e, *args)
        self.assertEqual([m.E], named_exprs)
        self.assertEqual([m.x, m.y, m.z], var_list)
        self.assertEqual([m.y], fixed_vars)
        self.assertEqual([], external_funcs)

    def test_collect_vars_basics(self):
        self.basics_helper(collect_vars_and_named_exprs)

    def external_func_helper(self, collector: Callable, *args):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find amplgsl.dll library')

        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.hypot = pyo.ExternalFunction(library=DLL, function='gsl_hypot')
        func = m.hypot(m.x, m.x * m.y)
        m.E = pyo.Expression(expr=2 * func)
        m.y.fix(3)
        e = m.z + m.x * m.E
        named_exprs, var_list, fixed_vars, external_funcs = collector(e, *args)
        self.assertEqual([m.E], named_exprs)
        self.assertEqual([m.z, m.x, m.y], var_list)
        self.assertEqual([m.y], fixed_vars)
        self.assertEqual([func], external_funcs)

    def test_collect_vars_external(self):
        self.external_func_helper(collect_vars_and_named_exprs)

    def simple_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
        model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])
        model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)
        return model

    def test_get_objective_success(self):
        model = self.simple_model()
        self.assertEqual(model.OBJ, get_objective(model))

    def test_get_objective_raise(self):
        model = self.simple_model()
        model.OBJ2 = pyo.Objective(expr=model.x[1] - 4 * model.x[2])
        with self.assertRaises(ValueError):
            get_objective(model)

    def test_check_optimal_termination_new_interface(self):
        results = Results()
        results.solution_status = SolutionStatus.optimal
        results.termination_condition = (
            TerminationCondition.convergenceCriteriaSatisfied
        )
        # Both items satisfied
        self.assertTrue(check_optimal_termination(results))
        # Termination condition not satisfied
        results.termination_condition = TerminationCondition.iterationLimit
        self.assertFalse(check_optimal_termination(results))
        # Both not satisfied
        results.solution_status = SolutionStatus.noSolution
        self.assertFalse(check_optimal_termination(results))

    def test_check_optimal_termination_condition_legacy_interface(self):
        results = SolverResults()
        results.solver.status = SolverStatus.ok
        results.solver.termination_condition = LegacyTerminationCondition.optimal
        # Both items satisfied
        self.assertTrue(check_optimal_termination(results))
        # Termination condition not satisfied
        results.solver.termination_condition = LegacyTerminationCondition.unknown
        self.assertFalse(check_optimal_termination(results))
        # Both not satisfied
        results.solver.termination_condition = SolverStatus.aborted
        self.assertFalse(check_optimal_termination(results))

    def test_assert_optimal_termination_new_interface(self):
        results = Results()
        results.solution_status = SolutionStatus.optimal
        results.termination_condition = (
            TerminationCondition.convergenceCriteriaSatisfied
        )
        assert_optimal_termination(results)
        # Termination condition not satisfied
        results.termination_condition = TerminationCondition.iterationLimit
        with self.assertRaises(RuntimeError):
            assert_optimal_termination(results)
        # Both not satisfied
        results.solution_status = SolutionStatus.noSolution
        with self.assertRaises(RuntimeError):
            assert_optimal_termination(results)

    def test_assert_optimal_termination_legacy_interface(self):
        results = SolverResults()
        results.solver.status = SolverStatus.ok
        results.solver.termination_condition = LegacyTerminationCondition.optimal
        assert_optimal_termination(results)
        # Termination condition not satisfied
        results.solver.termination_condition = LegacyTerminationCondition.unknown
        with self.assertRaises(RuntimeError):
            assert_optimal_termination(results)
        # Both not satisfied
        results.solver.termination_condition = SolverStatus.aborted
        with self.assertRaises(RuntimeError):
            assert_optimal_termination(results)

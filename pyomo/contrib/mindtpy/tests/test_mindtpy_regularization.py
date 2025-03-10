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

# -*- coding: utf-8 -*-
"""Tests for the MindtPy solver."""
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
    ConstraintQualificationExample,
)
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition

required_solvers = ('ipopt', 'cplex')
# required_solvers = ('gams', 'gams')
if all(SolverFactory(s).available(exception_flag=False) for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


model_list = [EightProcessFlowsheet(convex=True), ConstraintQualificationExample()]


@unittest.skipIf(
    not subsolvers_available,
    'Required subsolvers %s are not available' % (required_solvers,),
)
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def check_optimal_solution(self, model, places=1):
        for var in model.optimal_solution:
            self.assertAlmostEqual(
                var.value, model.optimal_solution[var], places=places
            )

    def test_ROA_L1(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='level_L1',
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ROA_L2(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='level_L2',
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ROA_Linf(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='level_L_infinity',
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ROA_grad_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='grad_lag',
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ROA_hess_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='hess_lag',
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ROA_hess_only_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='hess_only_lag',
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ROA_sqp_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='sqp_lag',
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ROA_sqp_lag_equality_relaxation(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='sqp_lag',
                    equality_relaxation=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ROA_sqp_lag_add_no_good_cuts(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='sqp_lag',
                    equality_relaxation=True,
                    add_no_good_cuts=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_ROA_sqp_lag_level_coef(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_regularization='sqp_lag',
                    equality_relaxation=True,
                    level_coef=0.4,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)


if __name__ == '__main__':
    unittest.main()

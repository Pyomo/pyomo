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

"""Tests for the MindtPy solver."""
import sys
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
    ConstraintQualificationExample,
)
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition

required_nlp_solvers = 'ipopt'
required_mip_solvers = ['cplex_persistent', 'gurobi_persistent']
available_mip_solvers = [
    s for s in required_mip_solvers if SolverFactory(s).available(False)
]

if (
    SolverFactory(required_nlp_solvers).available(exception_flag=False)
    and available_mip_solvers
):
    subsolvers_available = True
else:
    subsolvers_available = False


model_list = [
    EightProcessFlowsheet(convex=True),
    ConstraintQualificationExample(),
    SimpleMINLP(),
    SimpleMINLP3(),
]


def known_solver_failure(mip_solver, model):
    if (
        mip_solver == 'gurobi_persistent'
        and model.name in {'DuranEx3', 'SimpleMINLP'}
        and sys.platform.startswith('win')
        and SolverFactory(mip_solver).version()[:3] == (9, 5, 0)
    ):
        sys.stderr.write(
            f"Skipping sub-test {model.name} with {mip_solver} due to known "
            f"failure when running Gurobi 9.5.0 on Windows\n"
        )
        return True
    return False


@unittest.skipIf(
    not subsolvers_available,
    'Required subsolvers %s are not available'
    % ([required_nlp_solvers] + required_mip_solvers),
)
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def check_optimal_solution(self, model, places=1):
        for var in model.optimal_solution:
            self.assertAlmostEqual(
                var.value, model.optimal_solution[var], places=places
            )

    @unittest.skipUnless(
        'cplex_persistent' in available_mip_solvers,
        'cplex_persistent solver is not available',
    )
    def test_LPNLP_CPLEX(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver='cplex_persistent',
                    nlp_solver=required_nlp_solvers,
                    single_tree=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    @unittest.skipUnless(
        'gurobi_persistent' in available_mip_solvers,
        'gurobi_persistent solver is not available',
    )
    def test_LPNLP_Gurobi(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver='gurobi_persistent',
                    nlp_solver=required_nlp_solvers,
                    single_tree=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_RLPNLP_L1(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                for mip_solver in available_mip_solvers:
                    results = opt.solve(
                        model,
                        strategy='OA',
                        mip_solver=mip_solver,
                        nlp_solver=required_nlp_solvers,
                        single_tree=True,
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

    def test_RLPNLP_L2(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                for mip_solver in available_mip_solvers:
                    if known_solver_failure(mip_solver, model):
                        continue
                    results = opt.solve(
                        model,
                        strategy='OA',
                        mip_solver=mip_solver,
                        nlp_solver=required_nlp_solvers,
                        single_tree=True,
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

    def test_RLPNLP_Linf(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                for mip_solver in available_mip_solvers:
                    results = opt.solve(
                        model,
                        strategy='OA',
                        mip_solver=mip_solver,
                        nlp_solver=required_nlp_solvers,
                        single_tree=True,
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

    def test_RLPNLP_grad_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                for mip_solver in available_mip_solvers:
                    results = opt.solve(
                        model,
                        strategy='OA',
                        mip_solver=mip_solver,
                        nlp_solver=required_nlp_solvers,
                        single_tree=True,
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

    def test_RLPNLP_hess_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                for mip_solver in available_mip_solvers:
                    if known_solver_failure(mip_solver, model):
                        continue
                    results = opt.solve(
                        model,
                        strategy='OA',
                        mip_solver=mip_solver,
                        nlp_solver=required_nlp_solvers,
                        single_tree=True,
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

    def test_RLPNLP_hess_only_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                for mip_solver in available_mip_solvers:
                    if known_solver_failure(mip_solver, model):
                        continue
                    results = opt.solve(
                        model,
                        strategy='OA',
                        mip_solver=mip_solver,
                        nlp_solver=required_nlp_solvers,
                        single_tree=True,
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

    def test_RLPNLP_sqp_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                for mip_solver in available_mip_solvers:
                    if known_solver_failure(mip_solver, model):
                        continue
                    results = opt.solve(
                        model,
                        strategy='OA',
                        mip_solver=mip_solver,
                        nlp_solver=required_nlp_solvers,
                        single_tree=True,
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


if __name__ == '__main__':
    unittest.main()

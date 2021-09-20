#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests for the MindtPy solver."""
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import ConstraintQualificationExample
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition

required_nlp_solvers = 'ipopt'
required_mip_solvers = ['cplex_persistent', 'gurobi_persistent']


if SolverFactory(required_nlp_solvers).available(False) and any(SolverFactory(s).available(False) for s in required_mip_solvers):
    subsolvers_available = True
    available_mip_solvers = [s for s in required_mip_solvers
                             if SolverFactory(s).available(False)]
else:
    subsolvers_available = False


model_list = [EightProcessFlowsheet(convex=True),
              ConstraintQualificationExample(),
              SimpleMINLP()
              ]


@unittest.skipIf(not subsolvers_available,
                 'Required subsolvers %s are not available'
                 % ([required_nlp_solvers] + required_mip_solvers))
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def test_LPNLP(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                for mip_solver in available_mip_solvers:
                    results = opt.solve(model, strategy='OA',
                                        mip_solver=mip_solver,
                                        nlp_solver=required_nlp_solvers,
                                        single_tree=True)

                    self.assertIn(results.solver.termination_condition,
                                  [TerminationCondition.optimal, TerminationCondition.feasible])
                    self.assertAlmostEqual(
                        value(model.objective.expr), model.optimal_value, places=1)

    def test_RLPNLP_L1(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                for mip_solver in available_mip_solvers:
                    results = opt.solve(model, strategy='OA',
                                        mip_solver=mip_solver,
                                        nlp_solver=required_nlp_solvers,
                                        single_tree=True,
                                        add_regularization='level_L1')

                    self.assertIn(results.solver.termination_condition,
                                  [TerminationCondition.optimal, TerminationCondition.feasible])
                    self.assertAlmostEqual(
                        value(model.objective.expr), model.optimal_value, places=1)

    def test_RLPNLP_L2(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                for mip_solver in available_mip_solvers:
                    results = opt.solve(model, strategy='OA',
                                        mip_solver=mip_solver,
                                        nlp_solver=required_nlp_solvers,
                                        single_tree=True,
                                        add_regularization='level_L2')

                    self.assertIn(results.solver.termination_condition,
                                  [TerminationCondition.optimal, TerminationCondition.feasible])
                    self.assertAlmostEqual(
                        value(model.objective.expr), model.optimal_value, places=1)

    def test_RLPNLP_Linf(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                for mip_solver in available_mip_solvers:
                    results = opt.solve(model, strategy='OA',
                                        mip_solver=mip_solver,
                                        nlp_solver=required_nlp_solvers,
                                        single_tree=True,
                                        add_regularization='level_L_infinity')

                    self.assertIn(results.solver.termination_condition,
                                  [TerminationCondition.optimal, TerminationCondition.feasible])
                    self.assertAlmostEqual(
                        value(model.objective.expr), model.optimal_value, places=1)

    def test_RLPNLP_grad_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                for mip_solver in available_mip_solvers:
                    results = opt.solve(model, strategy='OA',
                                        mip_solver=mip_solver,
                                        nlp_solver=required_nlp_solvers,
                                        single_tree=True,
                                        add_regularization='grad_lag')

                    self.assertIn(results.solver.termination_condition,
                                  [TerminationCondition.optimal, TerminationCondition.feasible])
                    self.assertAlmostEqual(
                        value(model.objective.expr), model.optimal_value, places=1)

    def test_RLPNLP_hess_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                for mip_solver in available_mip_solvers:
                    results = opt.solve(model, strategy='OA',
                                        mip_solver=mip_solver,
                                        nlp_solver=required_nlp_solvers,
                                        single_tree=True,
                                        add_regularization='hess_lag')

                    self.assertIn(results.solver.termination_condition,
                                  [TerminationCondition.optimal, TerminationCondition.feasible])
                    self.assertAlmostEqual(
                        value(model.objective.expr), model.optimal_value, places=1)

    def test_RLPNLP_hess_only_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                for mip_solver in available_mip_solvers:
                    results = opt.solve(model, strategy='OA',
                                        mip_solver=mip_solver,
                                        nlp_solver=required_nlp_solvers,
                                        single_tree=True,
                                        add_regularization='hess_only_lag')

                    self.assertIn(results.solver.termination_condition,
                                  [TerminationCondition.optimal, TerminationCondition.feasible])
                    self.assertAlmostEqual(
                        value(model.objective.expr), model.optimal_value, places=1)

    def test_RLPNLP_sqp_lag(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                for mip_solver in available_mip_solvers:
                    results = opt.solve(model, strategy='OA',
                                        mip_solver=mip_solver,
                                        nlp_solver=required_nlp_solvers,
                                        single_tree=True,
                                        add_regularization='sqp_lag')

                    self.assertIn(results.solver.termination_condition,
                                  [TerminationCondition.optimal, TerminationCondition.feasible])
                    self.assertAlmostEqual(
                        value(model.objective.expr), model.optimal_value, places=1)


if __name__ == '__main__':
    unittest.main()

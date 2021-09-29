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
import pyomo.core.base.symbolic
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.MINLP5_simple import SimpleMINLP5
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import ConstraintQualificationExample
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition

required_nlp_solvers = 'ipopt'
required_mip_solvers = ['cplex_persistent', 'gurobi_persistent']
available_mip_solvers = [s for s in required_mip_solvers
                         if SolverFactory(s).available(False)]

if SolverFactory(required_nlp_solvers).available(False) and available_mip_solvers:
    subsolvers_available = True
else:
    subsolvers_available = False


model_list = [EightProcessFlowsheet(convex=True),
              ConstraintQualificationExample(),
              SimpleMINLP()
              ]


@unittest.skipIf(not subsolvers_available,
                 'Required subsolvers %s are not available'
                 % ([required_nlp_solvers] + required_mip_solvers))
@unittest.skipIf(not pyomo.core.base.symbolic.differentiate_available,
                 'Symbolic differentiation is not available')
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    @unittest.skipUnless('cplex_persistent' in available_mip_solvers,
                         'cplex_persistent solver is not available')
    def test_LPNLP_CPLEX(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='OA',
                                    mip_solver='cplex_persistent',
                                    nlp_solver=required_nlp_solvers,
                                    single_tree=True)

                self.assertIn(results.solver.termination_condition,
                              [TerminationCondition.optimal, TerminationCondition.feasible])
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1)

    @unittest.skipUnless('gurobi_persistent' in available_mip_solvers,
                         'gurobi_persistent solver is not available')
    def test_LPNLP_GUROBI(self):
        """Test the LP/NLP decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='OA',
                                    mip_solver='gurobi_persistent',
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

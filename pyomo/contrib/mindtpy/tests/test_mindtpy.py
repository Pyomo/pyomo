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
import pyutilib.th as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import ConstraintQualificationExample
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value, maximize
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.opt import TerminationCondition

required_solvers = ('ipopt', 'glpk')
# required_solvers = ('gams', 'gams')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


@unittest.skipIf(not subsolvers_available,
                 "Required subsolvers %s are not available"
                 % (required_solvers,))
@unittest.skipIf(not pyomo.core.base.symbolic.differentiate_available,
                 "Symbolic differentiation is not available")
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def test_OA_8PP(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=False)
            print('\n Solving 8PP problem with Outer Approximation')
            results = opt.solve(model, strategy='OA',
                                init_strategy='rNLP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5)

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_OA_8PP_init_max_binary(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=False)
            print('\n Solving 8PP problem with Outer Approximation(max_binary)')
            results = opt.solve(model, strategy='OA',
                                init_strategy='max_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_OA_8PP_L2_norm(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=False)
            print('\n Solving 8PP problem with Outer Approximation(max_binary)')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                feasibility_norm='L2')

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_OA_8PP_sympy(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=False)
            print('\n Solving 8PP problem with Outer Approximation(max_binary)')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                differentiate_mode='sympy')

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    # def test_PSC(self):
    #     """Test the partial surrogate cuts decomposition algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         model = EightProcessFlowsheet()
    #         print('\n Solving problem with Partial Surrogate Cuts')
    #         opt.solve(model, strategy='PSC',
    #                   init_strategy='rNLP', mip_solver=required_solvers[1],
    #                   nlp_solver=required_solvers[0])
    #
    #         # self.assertIs(results.solver.termination_condition,
    #         #               TerminationCondition.optimal)
    #         self.assertTrue(fabs(value(model.cost.expr) - 68) <= 1E-2)

    # def test_GBD(self):
    #     """Test the generalized Benders Decomposition algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         model = EightProcessFlowsheet()
    #         print('\n Solving problem with Generalized Benders Decomposition')
    #         opt.solve(model, strategy='GBD',
    #                   init_strategy='rNLP', mip_solver=required_solvers[1],
    #                   nlp_solver=required_solvers[0])
    #
    #         # self.assertIs(results.solver.termination_condition,
    #         #               TerminationCondition.optimal)
    #         self.assertTrue(fabs(value(model.cost.expr) - 68) <= 1E-2)
    #
    # def test_ECP(self):
    #     """Test the Extended Cutting Planes algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         model = EightProcessFlowsheet()
    #         print('\n Solving problem with Extended Cutting Planes')
    #         opt.solve(model, strategy='ECP',
    #                   init_strategy='rNLP', mip_solver=required_solvers[1],
    #                   nlp_solver=required_solvers[0])
    #
    #         # self.assertIs(results.solver.termination_condition,
    #         #               TerminationCondition.optimal)
    #         self.assertTrue(fabs(value(model.cost.expr) - 68) <= 1E-2)

    def test_OA_MINLP_simple(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving MINLP_simple problem with Outer Approximation')
            results = opt.solve(model, strategy='OA',
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 3.5, places=2)

    def test_OA_MINLP2_simple(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP2()
            print('\n Solving MINLP2_simple problem with Outer Approximation')
            results = opt.solve(model, strategy='OA',
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 6.00976, places=2)

    def test_OA_MINLP3_simple(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP3()
            print('\n Solving MINLP3_simple problem with Outer Approximation')
            results = opt.solve(model, strategy='OA', init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), -5.512, places=2)

    def test_OA_Proposal(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Outer Approximation')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0])

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_OA_Proposal_with_int_cuts(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Outer Approximation(integer cuts)')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_nogood_cuts=True,
                                integer_to_binary=True  # if we use lazy callback, we cannot set integer_to_binary True
                                )

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_OA_ConstraintQualificationExample(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print('\n Solving Constraint Qualification Example with Outer Approximation')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0]
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_OA_ConstraintQualificationExample_integer_cut(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print(
                '\n Solving Constraint Qualification Example with Outer Approximation(integer cut)')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_nogood_cuts=True
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_OA_OnlineDocExample(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with Outer Approximation')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0]
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    def test_OA_OnlineDocExample_L_infinity_norm(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with Outer Approximation')
            results = opt.solve(model, strategy='OA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                feasibility_norm="L_infinity"
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    # the following tests are used to improve code coverage

    def test_iteration_limit(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print('\n test iteration_limit  to improve code coverage')
            opt.solve(model, strategy='OA',
                      iteration_limit=1,
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0]
                      )
            # self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_time_limit(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print('\n test time_limit to improve code coverage')
            opt.solve(model, strategy='OA',
                      time_limit=1,
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0]
                      )

    def test_LP_case(self):
        with SolverFactory('mindtpy') as opt:
            m_class = LP_unbounded()
            m_class._generate_model()
            model = m_class.model
            print('\n Solving LP case with Outer Approximation')
            opt.solve(model, strategy='OA',
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0],
                      )

    def test_QCP_case(self):
        with SolverFactory('mindtpy') as opt:
            m_class = QCP_simple()
            m_class._generate_model()
            model = m_class.model
            print('\n Solving QCP case with Outer Approximation')
            opt.solve(model, strategy='OA',
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0],
                      )

    def test_maximize_obj(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            model.obj.sense = maximize
            print('\n test maximize case to improve code coverage')
            opt.solve(model, strategy='OA',
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0],
                      #   mip_solver_args={'timelimit': 0.9}
                      )
            self.assertAlmostEqual(value(model.obj.expr), 14.83, places=1)

    def test_rNLP_add_slack(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print(
                '\n Test rNLP initialize strategy and add_slack to improve code coverage')
            opt.solve(model, strategy='OA',
                      init_strategy='rNLP',
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0],
                      bound_tolerance=1E-5,
                      add_slack=True)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_initial_binary_add_slack(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print(
                '\n Test initial_binary initialize strategy and add_slack to improve code coverage')
            results = opt.solve(model, strategy='OA',
                                init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_slack=True)

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 3.5, places=2)

    # def test_OA_OnlineDocExample4(self):
    #     with SolverFactory('mindtpy') as opt:
    #         m = ConcreteModel()
    #         m.x = Var(within=Binary)
    #         m.y = Var(within=Reals)
    #         m.o = Objective(expr=m.x*m.y)
    #         print('\n Solving problem with Outer Approximation')
    #         opt.solve(m, strategy='OA',
    #                   mip_solver=required_solvers[1],
    #                   nlp_solver=required_solvers[0],
    #                   )

    # def test_PSC(self):
    #     """Test the partial surrogate cuts decomposition algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         model = SimpleMINLP()
    #         print('\n Solving problem with Partial Surrogate Cuts')
    #         opt.solve(model, strategy='PSC', init_strategy='initial_binary',
    #                   mip_solver=required_solvers[1],
    #                   nlp_solver=required_solvers[0])
    #
    #         # self.assertIs(results.solver.termination_condition,
    #         #               TerminationCondition.optimal)
    #         self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)
    #
    # def test_GBD(self):
    #     """Test the generalized Benders Decomposition algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         model = SimpleMINLP()
    #         print('\n Solving problem with Generalized Benders Decomposition')
    #         opt.solve(model, strategy='GBD', init_strategy='initial_binary',
    #                   mip_solver=required_solvers[1],
    #                   nlp_solver=required_solvers[0])
    #
    #         # self.assertIs(results.solver.termination_condition,
    #         #               TerminationCondition.optimal)
    #         self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)
    #
    # def test_ECP(self):
    #     """Test the Extended Cutting Planes algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         model = SimpleMINLP()
    #         print('\n Solving problem with Extended Cutting Planes')
    #         opt.solve(model, strategy='ECP', init_strategy='initial_binary',
    #                   ECP_tolerance=1E-4,
    #                   mip_solver=required_solvers[1],
    #                   nlp_solver=required_solvers[0])
    #
    #         # self.assertIs(results.solver.termination_condition,
    #         #               TerminationCondition.optimal)
    #         self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)
    #


if __name__ == "__main__":
    unittest.main()

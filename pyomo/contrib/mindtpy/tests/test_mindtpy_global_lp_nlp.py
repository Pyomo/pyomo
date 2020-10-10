"""Tests for the MindtPy solver."""
from math import fabs
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
from pyomo.contrib.mindtpy.tests.nonconvex1 import Nonconvex1
from pyomo.contrib.mindtpy.tests.nonconvex2 import Nonconvex2
from pyomo.contrib.mindtpy.tests.nonconvex3 import Nonconvex3
from pyomo.contrib.mindtpy.tests.nonconvex4 import Nonconvex4
from pyomo.environ import SolverFactory, value
from pyomo.environ import *
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.solvers.tests.models.MIQCP_simple import MIQCP_simple

from pyomo.opt import TerminationCondition

required_solvers = ('baron', 'cplex_persistent')
if not all(SolverFactory(s).available(False) for s in required_solvers):
    subsolvers_available = False
elif not SolverFactory('baron').license_is_valid():
    subsolvers_available = False
else:
    subsolvers_available = True


@unittest.skipIf(not subsolvers_available,
                 "Required subsolvers %s are not available"
                 % (required_solvers,))
@unittest.skipIf(not pyomo.core.base.symbolic.differentiate_available,
                 "Symbolic differentiation is not available")
@unittest.skipIf(not pyomo.contrib.mcpp.pyomo_mcpp.mcpp_available(),
                 "MC++ is not available")
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def test_GOA_8PP(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving 8PP problem with Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                init_strategy='rNLP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_nogood_cuts=True,
                                bound_tolerance=1E-5,
                                single_tree=True)

            self.assertIn(results.solver.termination_condition,
                          [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_GOA_8PP_init_max_binary(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving 8PP problem with Outer Approximation(max_binary)')
            results = opt.solve(model, strategy='GOA',
                                init_strategy='max_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_nogood_cuts=True,
                                single_tree=True)

            self.assertIn(results.solver.termination_condition,
                          [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_GOA_8PP_L2_norm(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving 8PP problem with Outer Approximation(L2_norm)')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                feasibility_norm='L2',
                                add_nogood_cuts=True,
                                single_tree=True)

            self.assertIn(results.solver.termination_condition,
                          [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_GOA_8PP_sympy(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving 8PP problem with Outer Approximation(sympy)')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                differentiate_mode='sympy',
                                add_nogood_cuts=True,
                                single_tree=True)

            self.assertIn(results.solver.termination_condition,
                          [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_GOA_MINLP_simple(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving MINLP_simple problem with Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                obj_bound=10,
                                add_nogood_cuts=True,
                                single_tree=True)

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 3.5, places=2)

    # if no affine cuts is added in lp/nlp , stop

    def test_GOA_MINLP2_simple(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP2()
            print('\n Solving MINLP2_simple problem with Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver='baron',
                                add_nogood_cuts=True,
                                single_tree=True,
                                bound_tolerance=1E-2
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 6.00976, places=2)

    def test_GOA_MINLP3_simple(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP3()
            print('\n Solving MINLP3_simple problem with Outer Approximation')
            results = opt.solve(model, strategy='GOA', init_strategy='initial_binary',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                obj_bound=10,
                                add_nogood_cuts=True,
                                use_mcpp=True,
                                single_tree=True)
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.feasible)
            self.assertAlmostEqual(value(model.cost.expr), -5.512, places=2)

    def test_GOA_Proposal(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_nogood_cuts=True,
                                integer_to_binary=True,
                                single_tree=True)

            self.assertIn(results.solver.termination_condition,
                          [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_GOA_Proposal_with_int_cuts(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving Proposal problem with Outer Approximation(integer cuts)')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_nogood_cuts=True,
                                integer_to_binary=True,  # if we use lazy callback, we cannot set integer_to_binary True
                                single_tree=True
                                )

            self.assertIn(results.solver.termination_condition,
                          [TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)

    def test_GOA_ConstraintQualificationExample(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print(
                '\n Solving Constraint Qualification Example with global Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_nogood_cuts=True,
                                single_tree=True
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_GOA_ConstraintQualificationExample_integer_cut(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print(
                '\n Solving Constraint Qualification Example with global Outer Approximation(integer cut)')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_nogood_cuts=True,
                                single_tree=True
                                )
            self.assertIn(results.solver.termination_condition, [
                          TerminationCondition.optimal, TerminationCondition.feasible])
            self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_GOA_OnlineDocExample(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with global Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                add_nogood_cuts=True,
                                single_tree=True
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    def test_GOA_OnlineDocExample_L_infinity_norm(self):
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving Online Doc Example with global Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                feasibility_norm="L_infinity",
                                add_nogood_cuts=True,
                                single_tree=True
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 2.438447, places=2)

    def test_GOA_Nonconvex1(self):
        with SolverFactory('mindtpy') as opt:
            model = Nonconvex1()
            print('\n Solving Nonconvex1 with global Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver='baron',
                                single_tree=True
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 7.667, places=2)

    def test_GOA_Nonconvex2(self):
        with SolverFactory('mindtpy') as opt:
            model = Nonconvex2()
            print('\n Solving Nonconvex2 with global Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver='baron',
                                single_tree=True,
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), -0.94347, places=2)

    def test_GOA_Nonconvex3(self):
        with SolverFactory('mindtpy') as opt:
            model = Nonconvex3()
            print('\n Solving Nonconvex3 with global Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver='baron',
                                single_tree=True
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), 31, places=2)

    def test_GOA_Nonconvex4(self):
        with SolverFactory('mindtpy') as opt:
            model = Nonconvex4()
            print('\n Solving Nonconvex4 with global Outer Approximation')
            results = opt.solve(model, strategy='GOA',
                                mip_solver=required_solvers[1],
                                nlp_solver='baron',
                                single_tree=True,
                                )
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(
                value(model.objective.expr), -17, places=2)


if __name__ == "__main__":
    unittest.main()

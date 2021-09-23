"""Tests for the MindtPy solver."""
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import ConstraintQualificationExample
from pyomo.environ import SolverFactory, value, maximize
from pyomo.opt import TerminationCondition


model_list = [EightProcessFlowsheet(convex=True),
              ConstraintQualificationExample(),
              SimpleMINLP2(),
              ]

required_solvers = ('ipopt', 'cplex_persistent')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


@unittest.skipIf(not subsolvers_available,
                 'Required subsolvers %s are not available'
                 % (required_solvers,))
@unittest.skipIf(not differentiate_available,
                 'Symbolic differentiation is not available')
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def test_OA_solution_pool(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            print('\n Solving 8PP problem with Outer Approximation')
            for model in model_list:
                results = opt.solve(model, strategy='OA',
                                    init_strategy='rNLP',
                                    solution_pool=True,
                                    mip_solver=required_solvers[1],
                                    nlp_solver=required_solvers[0],
                                    )
                self.assertIn(results.solver.termination_condition,
                              [TerminationCondition.optimal, TerminationCondition.feasible])
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1)

"""Tests for solution pool in the MindtPy solver."""
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
    ConstraintQualificationExample,
)
from pyomo.environ import SolverFactory, value, maximize
from pyomo.opt import TerminationCondition


model_list = [
    EightProcessFlowsheet(convex=True),
    ConstraintQualificationExample(),
    SimpleMINLP2(),
]


try:
    import cplex

    cplexpy_available = True
except ImportError:
    cplexpy_available = False

required_solvers = ('ipopt', 'cplex_persistent', 'gurobi_persistent')
ipopt_available = SolverFactory('ipopt').available()
cplex_persistent_available = SolverFactory('cplex_persistent').available(
    exception_flag=False
)
gurobi_persistent_available = SolverFactory('gurobi_persistent').available(
    exception_flag=False
)


@unittest.skipIf(
    not differentiate_available, 'Symbolic differentiation is not available'
)
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def check_optimal_solution(self, model, places=1):
        for var in model.optimal_solution:
            self.assertAlmostEqual(
                var.value, model.optimal_solution[var], places=places
            )

    @unittest.skipIf(
        not (ipopt_available and cplex_persistent_available and cplexpy_available),
        'Required subsolvers are not available',
    )
    def test_OA_solution_pool_cplex(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='rNLP',
                    solution_pool=True,
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )
                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2
                )
                self.check_optimal_solution(model)

    @unittest.skipIf(
        not (ipopt_available and gurobi_persistent_available),
        'Required subsolvers are not available',
    )
    def test_OA_solution_pool_gurobi(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='rNLP',
                    solution_pool=True,
                    mip_solver=required_solvers[2],
                    nlp_solver=required_solvers[0],
                )
                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2
                )
                self.check_optimal_solution(model)

    # the following tests are used to increase the code coverage
    @unittest.skipIf(
        not (ipopt_available and cplex_persistent_available),
        'Required subsolvers are not available',
    )
    def test_OA_solution_pool_coverage1(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='rNLP',
                    solution_pool=True,
                    mip_solver='glpk',
                    nlp_solver=required_solvers[0],
                    num_solution_iteration=1,
                )
                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2
                )
                self.check_optimal_solution(model)


if __name__ == '__main__':
    unittest.main()

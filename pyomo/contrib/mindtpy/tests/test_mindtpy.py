#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests for the MindtPy solver."""
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.MINLP4_simple import SimpleMINLP4
from pyomo.contrib.mindtpy.tests.MINLP5_simple import SimpleMINLP5
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
    ConstraintQualificationExample,
)
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value, maximize
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.opt import TerminationCondition


full_model_list = [
    EightProcessFlowsheet(convex=True),
    ConstraintQualificationExample(),
    SimpleMINLP(),
    SimpleMINLP2(),
    SimpleMINLP3(),
    SimpleMINLP4(),
    SimpleMINLP5(),
    ProposalModel(),
    OnlineDocExample(),
]
model_list = [
    EightProcessFlowsheet(convex=True),
    ConstraintQualificationExample(),
    SimpleMINLP2(),
]
nonconvex_model_list = [EightProcessFlowsheet(convex=False)]

obj_nonlinear_sum_model_list = [SimpleMINLP(), SimpleMINLP5()]

LP_model = LP_unbounded()
LP_model._generate_model()

QCP_model = QCP_simple()
QCP_model._generate_model()
extreme_model_list = [LP_model.model, QCP_model.model]

required_solvers = ('ipopt', 'glpk')
if all(SolverFactory(s).available(exception_flag=False) for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


@unittest.skipIf(
    not subsolvers_available,
    'Required subsolvers %s are not available' % (required_solvers,),
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

    def test_OA_rNLP(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='rNLP',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_OA_extreme_model(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in extreme_model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='rNLP',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

    def test_OA_L2_norm(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='rNLP',
                    feasibility_norm='L2',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_OA_L_infinity_norm(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='rNLP',
                    feasibility_norm='L_infinity',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_OA_max_binary(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='max_binary',
                    feasibility_norm='L1',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_OA_sympy(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    differentiate_mode='sympy',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_OA_initial_binary(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='initial_binary',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_OA_no_good_cuts(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
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

    @unittest.skipUnless(
        SolverFactory('cplex').available() or SolverFactory('gurobi').available(),
        "CPLEX or Gurobi not available.",
    )
    def test_OA_quadratic_strategy(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel().clone()
            if SolverFactory('cplex').available():
                mip_solver = 'cplex'
            elif SolverFactory('gurobi').available():
                mip_solver = 'gurobi'
            for quadratic_strategy in (0, 1, 2):
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=mip_solver,
                    nlp_solver=required_solvers[0],
                    quadratic_strategy=quadratic_strategy,
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
        SolverFactory('appsi_cplex').available(exception_flag=False),
        "APPSI_CPLEX not available.",
    )
    def test_OA_APPSI_solver(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver='appsi_cplex',
                    nlp_solver=required_solvers[0],
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )

    @unittest.skipUnless(
        SolverFactory('appsi_ipopt').available(exception_flag=False),
        "APPSI_IPOPT not available.",
    )
    def test_OA_APPSI_ipopt(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver='appsi_ipopt',
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )

    @unittest.skipUnless(
        SolverFactory('cyipopt').available(exception_flag=False),
        "APPSI_IPOPT not available.",
    )
    def test_OA_cyipopt(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in nonconvex_model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver='cyipopt',
                    heuristic_nonconvex=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )

    def test_OA_integer_to_binary(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    integer_to_binary=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_OA_partition_obj_nonlinear_terms(self):
        """Test the outer approximation decomposition algorithm (partition_obj_nonlinear_terms)."""
        with SolverFactory('mindtpy') as opt:
            for model in obj_nonlinear_sum_model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    partition_obj_nonlinear_terms=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_OA_add_slack(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='initial_binary',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_slack=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

                results = opt.solve(
                    model,
                    strategy='OA',
                    init_strategy='rNLP',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    add_slack=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_OA_nonconvex(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in nonconvex_model_list:
                model = model.clone()
                results = opt.solve(
                    model,
                    strategy='OA',
                    mip_solver=required_solvers[1],
                    nlp_solver=required_solvers[0],
                    heuristic_nonconvex=True,
                )

                self.assertIn(
                    results.solver.termination_condition,
                    [TerminationCondition.optimal, TerminationCondition.feasible],
                )
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1
                )
                self.check_optimal_solution(model)

    def test_iteration_limit(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            opt.solve(
                model,
                strategy='OA',
                iteration_limit=1,
                mip_solver=required_solvers[1],
                nlp_solver=required_solvers[0],
            )
            # self.assertAlmostEqual(value(model.objective.expr), 3, places=2)

    def test_time_limit(self):
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            opt.solve(
                model,
                strategy='OA',
                time_limit=1,
                mip_solver=required_solvers[1],
                nlp_solver=required_solvers[0],
            )

    def test_maximize_obj(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel().clone()
            model.objective.sense = maximize
            opt.solve(
                model,
                strategy='OA',
                mip_solver=required_solvers[1],
                nlp_solver=required_solvers[0],
            )
            self.assertAlmostEqual(value(model.objective.expr), 14.83, places=1)

    def test_infeasible_model(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP().clone()
            model.X[1].fix(0)
            model.Y[1].fix(0)
            results = opt.solve(
                model,
                strategy='OA',
                mip_solver=required_solvers[1],
                nlp_solver=required_solvers[0],
            )
            self.assertIs(
                results.solver.termination_condition, TerminationCondition.infeasible
            )


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
"""Tests for the MindtPy solver."""
from math import fabs
import pyomo.core.base.symbolic
from pyomo.core.expr import template_expr
import pyutilib.th as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import ConstraintQualificationExample
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value
from pyomo.environ import *
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.solvers.tests.models.MIQCP_simple import MIQCP_simple
from pyomo.opt import TerminationCondition
from pyomo.contrib.gdpopt.util import is_feasible
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.contrib.mindtpy.tests.feasibility_pump1 import Feasibility_Pump1
from pyomo.contrib.mindtpy.tests.feasibility_pump2 import Feasibility_Pump2

required_solvers = ('ipopt', 'glpk', 'cplex')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


@unittest.skipIf(not subsolvers_available,
                 'Required subsolvers %s are not available'
                 % (required_solvers,))
@unittest.skipIf(not pyomo.core.base.symbolic.differentiate_available,
                 'Symbolic differentiation is not available')
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver."""

    def get_config(self, solver):
        config = solver.CONFIG
        return config

    """Test pure feasibility pump."""

    def test_FP_8PP(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print('\n Solving 8PP problem using feasibility pump')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5)
            log_infeasible_constraints(model)
            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_8PP_Norm2(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print(
                '\n Solving 8PP problem using feasibility pump with squared Norm2 in mip projection problem')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[2],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5,
                                fp_master_norm='L2')
            log_infeasible_constraints(model)
            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_8PP_Norm_infinity(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print(
                '\n Solving 8PP problem using feasibility pump with Norm infinity in mip projection problem')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5,
                                fp_master_norm='L_infinity')
            log_infeasible_constraints(model)
            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_8PP_Norm_infinity_with_norm_constraint(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print(
                '\n Solving 8PP problem using feasibility pump with Norm infinity in mip projection problem')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5,
                                fp_master_norm='L_infinity',
                                fp_norm_constraint=False)
            log_infeasible_constraints(model)
            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_simpleMINLP(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving 8PP problem using feasibility pump')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5)
            log_infeasible_constraints(model)
            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_Feasibility_Pump1(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = Feasibility_Pump1()
            print('\n Solving Feasibility_Pump1 with feasibility pump')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5)
            log_infeasible_constraints(model)
            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_Feasibility_Pump2(self):
        """Test the extended cutting plane decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = Feasibility_Pump2()
            print('\n Solving Feasibility_Pump2 with feasibility pump')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-3,
                                )
            log_infeasible_constraints(model)
            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_fp_MINLP2_simple(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP2()
            print('\n Solving SimpleMINLP2 using feasibility pump')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_fp_MINLP3_simple(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP3()
            print('\n Solving SimpleMINLP3 using feasibility pump')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_fp_Proposal(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving ProposalModel using feasibility pump')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_fp_OnlineDocExample(self):
        """Test the feasibility pump algorithm."""
        """TODO: bug fix"""
        with SolverFactory('mindtpy') as opt:
            model = OnlineDocExample()
            print('\n Solving OnlineDocExample using feasibility pump')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=0)
            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_fp_ConstraintQualificationExample(self):
        """Test the feasibility pump algorithm."""
        # TODO: bug fix
        with SolverFactory('mindtpy') as opt:
            model = ConstraintQualificationExample()
            print('\n Solving ProposalModel using feasibility pump')
            results = opt.solve(model, strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))
    """Test FP-OA"""
    # oa cuts will cut off integer solutions.

    def test_FP_OA_8PP(self):
        """Test the FP-OA algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet(convex=True)
            print('\n Solving 8PP problem using FP-OA')
            results = opt.solve(model, strategy='OA',
                                init_strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5)
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 68, places=1)

    def test_FP_OA_simpleMINLP(self):
        """Test the FP-OA algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving 8PP problem using FP-OA')
            results = opt.solve(model, strategy='OA',
                                init_strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5)
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 3.5, places=2)

    def test_FP_OA_Feasibility_Pump1(self):
        """Test the FP-OA algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = Feasibility_Pump1()
            print('\n Solving Feasibility_Pump1 with FP-OA')
            results = opt.solve(model, strategy='OA',
                                init_strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-5)
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_OA_MINLP2_simple(self):
        """Test the FP-OA algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP2()
            print('\n Solving SimpleMINLP2 using FP-OA')
            results = opt.solve(model, strategy='OA',
                                init_strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)
            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), 6.00976, places=2)

    def test_FP_OA_MINLP3_simple(self):
        """Test the FP-OA algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP3()
            print('\n Solving SimpleMINLP3 using FP-OA')
            results = opt.solve(model, strategy='OA',
                                init_strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.cost.expr), -5.512, places=2)

    def test_FP_OA_Proposal(self):
        """Test the FP-OA algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving ProposalModel using FP-OA')
            results = opt.solve(model, strategy='OA',
                                init_strategy='FP',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertIs(results.solver.termination_condition,
                          TerminationCondition.optimal)
            self.assertAlmostEqual(value(model.obj.expr), 0.66555, places=2)


if __name__ == '__main__':
    unittest.main()

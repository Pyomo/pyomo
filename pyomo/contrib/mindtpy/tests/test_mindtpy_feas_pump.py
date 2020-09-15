"""Tests for the MindtPy solver plugin."""
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

    def get_config(self, solver):
        config = solver.CONFIG
        return config

    # def test_FP_8PP(self):
    #     """Test the extended cutting plane decomposition algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         model = EightProcessFlowsheet()
    #         print('\n Solving 8PP problem with extended cutting plane')
    #         results = opt.solve(model, strategy='feas_pump',
    #                             mip_solver=required_solvers[1],
    #                             nlp_solver=required_solvers[0],
    #                             bound_tolerance=1E-5,
    #                             tee=True)
    #         log_infeasible_constraints(model)
    #         self.assertTrue(is_feasible(model, self.get_config(opt)))

    # def test_FP_simpleMINLP(self):
    #     """Test the extended cutting plane decomposition algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         model = SimpleMINLP()
    #         print('\n Solving 8PP problem with feasibility pump')
    #         results = opt.solve(model, strategy='feas_pump',
    #                             mip_solver=required_solvers[1],
    #                             nlp_solver=required_solvers[0],
    #                             bound_tolerance=1E-5,
    #                             tee=True)
    #         log_infeasible_constraints(model)
    #         self.assertTrue(is_feasible(model, self.get_config(opt)))

    # def test_FP_Feasibility_Pump1(self):
    #     """Test the extended cutting plane decomposition algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         model = Feasibility_Pump1()
    #         print('\n Solving Feasibility_Pump1 with feasibility pump')
    #         results = opt.solve(model, strategy='feas_pump',
    #                             mip_solver=required_solvers[1],
    #                             nlp_solver=required_solvers[0],
    #                             bound_tolerance=1E-5,
    #                             tee=True)
    #         log_infeasible_constraints(model)
    #         self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_Feasibility_Pump2(self):
        """Test the extended cutting plane decomposition algorithm.
        TODO: the fixed_nlp is an LP"""
        with SolverFactory('mindtpy') as opt:
            model = Feasibility_Pump2()
            print('\n Solving Feasibility_Pump2 with feasibility pump')
            results = opt.solve(model, strategy='feas_pump',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                bound_tolerance=1E-3,
                                tee=True,
                                solver_tee=True)
            log_infeasible_constraints(model)
            self.assertTrue(is_feasible(model, self.get_config(opt)))


'''
    def test_feas_pump_8PP(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving feasibility pump')
            results = opt.solve(model, strategy='feas_pump',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_feas_pump_8PP_init_max_binary(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving feasibility pump')
            results = opt.solve(model, strategy='feas_pump',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_feas_pump_MINLP_simple(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving feasibility pump')
            results = opt.solve(model, strategy='feas_pump',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_feas_pump_MINLP2_simple(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP2()
            print('\n Solving feasibility pump')
            results = opt.solve(model, strategy='feas_pump',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_feas_pump_MINLP3_simple(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP3()
            print('\n Solving feasibility pump')
            results = opt.solve(model, strategy='feas_pump',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_feas_pump_Proposal(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving feasibility pump')
            results = opt.solve(model, strategy='feas_pump',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_feas_pump_Proposal_with_int_cuts(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = ProposalModel()
            print('\n Solving feasibility pump')
            results = opt.solve(model, strategy='feas_pump',
                                mip_solver=required_solvers[1],
                                nlp_solver=required_solvers[0],
                                iteration_limit=30)

            self.assertTrue(is_feasible(model, self.get_config(opt)))
'''

if __name__ == "__main__":
    unittest.main()

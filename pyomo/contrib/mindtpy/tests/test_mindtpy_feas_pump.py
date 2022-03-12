# -*- coding: utf-8 -*-
"""Tests for the MindtPy solver."""
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.from_proposal import ProposalModel
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import ConstraintQualificationExample
from pyomo.contrib.mindtpy.tests.online_doc_example import OnlineDocExample
from pyomo.environ import SolverFactory, value
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

model_list = [EightProcessFlowsheet(convex=True),
              ConstraintQualificationExample(),
              Feasibility_Pump1(),
              Feasibility_Pump2(),
              SimpleMINLP(),
              SimpleMINLP2(),
              SimpleMINLP3(),
              ProposalModel(),
              OnlineDocExample()
              ]


@unittest.skipIf(not subsolvers_available,
                 'Required subsolvers %s are not available'
                 % (required_solvers,))
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver."""

    def get_config(self, solver):
        config = solver.CONFIG
        return config

    def test_FP(self):
        """Test the feasibility pump algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='FP',
                                    mip_solver=required_solvers[1],
                                    nlp_solver=required_solvers[0],
                                    absolute_bound_tolerance=1E-5)
                log_infeasible_constraints(model)
                self.assertTrue(is_feasible(model, self.get_config(opt)))

    def test_FP_OA_8PP(self):
        """Test the FP-OA algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='OA',
                                    init_strategy='FP',
                                    mip_solver=required_solvers[1],
                                    nlp_solver=required_solvers[0],
                                    # absolute_bound_tolerance=1E-5
                                    )
                self.assertIn(results.solver.termination_condition,
                              [TerminationCondition.optimal, TerminationCondition.feasible])
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=1)


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
"""Tests for global LP/NLP in the MindtPy solver."""
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.nonconvex1 import Nonconvex1
from pyomo.contrib.mindtpy.tests.nonconvex2 import Nonconvex2
from pyomo.contrib.mindtpy.tests.nonconvex3 import Nonconvex3
from pyomo.contrib.mindtpy.tests.nonconvex4 import Nonconvex4
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
from pyomo.contrib.mcpp import pyomo_mcpp

required_solvers = ('baron', 'cplex_persistent')
if not all(SolverFactory(s).available(False) for s in required_solvers):
    subsolvers_available = False
elif not SolverFactory('baron').license_is_valid():
    subsolvers_available = False
else:
    subsolvers_available = True


model_list = [EightProcessFlowsheet(convex=False),
              Nonconvex1(),
              Nonconvex2(),
              Nonconvex3(),
              Nonconvex4(),
              ]


@unittest.skipIf(not subsolvers_available,
                 'Required subsolvers %s are not available'
                 % (required_solvers,))
@unittest.skipIf(not pyomo_mcpp.mcpp_available(),
                 'MC++ is not available')
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def test_GOA(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='GOA',
                                    mip_solver=required_solvers[1],
                                    nlp_solver=required_solvers[0],
                                    single_tree=True)

                self.assertIn(results.solver.termination_condition, [
                    TerminationCondition.optimal, TerminationCondition.feasible])
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2)

    def test_GOA_tabu_list(self):
        """Test the global outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            for model in model_list:
                results = opt.solve(model, strategy='GOA',
                                    mip_solver=required_solvers[1],
                                    nlp_solver=required_solvers[0],
                                    single_tree=True,
                                    use_tabu_list=True)

                self.assertIn(results.solver.termination_condition, [
                    TerminationCondition.optimal, TerminationCondition.feasible])
                self.assertAlmostEqual(
                    value(model.objective.expr), model.optimal_value, places=2)


if __name__ == '__main__':
    unittest.main()

"""Tests for the MINDT solver plugin."""
from math import fabs

import pyomo.core.base.symbolic
import pyutilib.th as unittest
from pyomo.contrib.mindtpy.tests.fo9 import build_model
from pyomo.environ import SolverFactory, value

# from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
# model = EightProcessFlowsheet()


required_solvers = ('ipopt', 'cplex')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


class TestMindtPy(unittest.TestCase):
    """Tests for the MINDT solver plugin."""

    def test_model(self):
        """Test the MindtPy implementation."""
        with SolverFactory('mindtpy') as opt:
            print('\n Solving problem with selected decomposition strategy')
            mip_options = {'threads': 4}
            opt.solve(build_model(),
                      strategy='OA', init_strategy='initial_binary',
                      mip_solver=required_solvers[1], iteration_limit=13,
                      mip_solver_args={'options': mip_options},
                      nlp_solver=required_solvers[0])
            # model.pprint()

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            # self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)


if __name__ == "__main__":
    unittest.main()

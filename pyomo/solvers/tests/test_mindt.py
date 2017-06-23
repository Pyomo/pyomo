"""Tests for the MINDT solver plugin."""

import pyutilib.th as unittest
from pyomo.environ import SolverFactory, value
from pyomo.solvers.tests.models.eight_process_prob import EightProcessFlowsheet

required_solvers = ('ipopt', 'gurobi')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False

diff_tol = 1e-4


class TestMINDT(unittest.TestCase):
    """Tests for the MINDT solver plugin."""

    @unittest.skipIf(not subsolvers_available,
                     "Required subsolvers {} are not available"
                     .format(required_solvers))
    def test_OA(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindt') as opt:
            model = EightProcessFlowsheet()
            opt.solve(model, strategy='OA')

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(abs(value(model.cost.expr) - 68) <= 1E-2)

    @unittest.skipIf(not subsolvers_available,
                     "Required subsolvers {} are not available"
                     .format(required_solvers))
    def test_PSC(self):
        """Test the partial surrogate cuts decomposition algorithm."""
        with SolverFactory('mindt') as opt:
            model = EightProcessFlowsheet()
            opt.solve(model, strategy='PSC')

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(abs(value(model.cost.expr) - 68) <= 1E-2)


if __name__ == "__main__":
    unittest.main()

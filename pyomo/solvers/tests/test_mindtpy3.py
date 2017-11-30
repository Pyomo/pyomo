"""Tests for the MINDT solver plugin."""

import pyutilib.th as unittest
from pyomo.environ import SolverFactory, value
from pyomo.solvers.tests.models.batchdes import *

required_solvers = ('ipopt', 'gurobi')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False

diff_tol = 1e-4

__author__ = "David Bernal <https://github.com/bernalde>"


class TestMindtPy(unittest.TestCase):
    """Tests for the MINDT solver plugin."""

    @unittest.skipIf(not subsolvers_available,
                     "Required subsolvers {} are not available"
                     .format(required_solvers))
    def test_OA(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            print('\n Solving problem with Outer Approximation')
            opt.solve(model, strategy='PSC', init_strategy = 'rNLP')

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
    #
    # @unittest.skipIf(not subsolvers_available,
    #                  "Required subsolvers {} are not available"
    #                  .format(required_solvers))
    # def test_PSC(self):
    #     """Test the partial surrogate cuts decomposition algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         print('\n Solving problem with Partial Surrogate Cuts')
    #         opt.solve(model, strategy='PSC', init_strategy = 'rNLP')
    #
    #         # self.assertIs(results.solver.termination_condition,
    #         #               TerminationCondition.optimal)
    #         self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)
    #
    # @unittest.skipIf(not subsolvers_available,
    #                  "Required subsolvers {} are not available"
    #                  .format(required_solvers))
    # def test_GBD(self):
    #     """Test the generalized Benders Decomposition algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         print('\n Solving problem with Generalized Benders Decomposition')
    #         opt.solve(model, strategy='GBD', init_strategy = 'rNLP')
    #
    #         # self.assertIs(results.solver.termination_condition,
    #         #               TerminationCondition.optimal)
    #         self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)
    #
    # @unittest.skipIf(not subsolvers_available,
    #                  "Required subsolvers {} are not available"
    #                  .format(required_solvers))
    # def test_ECP(self):
    #     """Test the Extended Cutting Planes algorithm."""
    #     with SolverFactory('mindtpy') as opt:
    #         print('\n Solving problem with Extended Cutting Planes')
    #         opt.solve(model, strategy='ECP', init_strategy = 'rNLP')
    #
    #         # self.assertIs(results.solver.termination_condition,
    #         #               TerminationCondition.optimal)
    #         self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)


if __name__ == "__main__":
    unittest.main()

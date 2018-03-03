"""Tests for the MINDT solver plugin."""
from math import fabs

import pyutilib.th as unittest

from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet

from pyomo.environ import SolverFactory, value

required_solvers = ('ipopt', 'gurobi')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False

import pyomo.core.base.symbolic


@unittest.skipIf(not subsolvers_available,
                 "Required subsolvers %s are not available"
                 % (required_solvers,))
@unittest.skipIf(not pyomo.core.base.symbolic.differentiate_available,
                 "Symbolic differentiation is not available")


class TestMindtPy(unittest.TestCase):
    """Tests for the MINDT solver plugin."""

    def test_OA(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving problem with Outer Approximation')
            opt.solve(model, strategy='OA', init_strategy = 'rNLP', mip = 'gurobi')

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(fabs(value(model.cost.expr) - 68) <= 1E-2)


    def test_PSC(self):
        """Test the partial surrogate cuts decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving problem with Partial Surrogate Cuts')
            opt.solve(model, strategy='PSC', init_strategy = 'rNLP', mip = 'gurobi')

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(fabs(value(model.cost.expr) - 68) <= 1E-2)


    def test_GBD(self):
        """Test the generalized Benders Decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving problem with Generalized Benders Decomposition')
            opt.solve(model, strategy='GBD', init_strategy = 'rNLP', mip = 'gurobi')

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(fabs(value(model.cost.expr) - 68) <= 1E-2)

    def test_ECP(self):
        """Test the Extended Cutting Planes algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = EightProcessFlowsheet()
            print('\n Solving problem with Extended Cutting Planes')
            opt.solve(model, strategy='ECP', init_strategy = 'rNLP', mip = 'gurobi')

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(fabs(value(model.cost.expr) - 68) <= 1E-2)


if __name__ == "__main__":
    unittest.main()

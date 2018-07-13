"""Tests for the MINDT solver plugin."""
from math import fabs

import pyomo.core.base.symbolic
import pyutilib.th as unittest
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP
from pyomo.environ import SolverFactory, value

required_solvers = ('ipopt', 'gurobi')
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
    """Tests for the MINDT solver plugin."""

    def test_OA(self):
        """Test the outer approximation decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving problem with Outer Approximation')
            opt.solve(model, strategy='OA', init_strategy='initial_binary',
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0])

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)

    def test_PSC(self):
        """Test the partial surrogate cuts decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving problem with Partial Surrogate Cuts')
            opt.solve(model, strategy='PSC', init_strategy='initial_binary',
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0])

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)

    def test_GBD(self):
        """Test the generalized Benders Decomposition algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving problem with Generalized Benders Decomposition')
            opt.solve(model, strategy='GBD', init_strategy='initial_binary',
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0])

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)

    def test_ECP(self):
        """Test the Extended Cutting Planes algorithm."""
        with SolverFactory('mindtpy') as opt:
            model = SimpleMINLP()
            print('\n Solving problem with Extended Cutting Planes')
            opt.solve(model, strategy='ECP', init_strategy='initial_binary',
                      ECP_tolerance=1E-4,
                      mip_solver=required_solvers[1],
                      nlp_solver=required_solvers[0])

            # self.assertIs(results.solver.termination_condition,
            #               TerminationCondition.optimal)
            self.assertTrue(abs(value(model.cost.expr) - 3.5) <= 1E-2)
    #


if __name__ == "__main__":
    unittest.main()

"""Tests the BARON interface."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, Objective, Var, log10,
                           minimize)
from pyomo.opt import SolverFactory, TerminationCondition

# check if BARON is available
from pyomo.solvers.tests.solvers import test_solver_cases
baron_available = test_solver_cases('baron', 'bar').available


@unittest.skipIf(not baron_available,
                 "The 'BARON' solver is not available")
class BaronTest(unittest.TestCase):
    """Test the BARON interface."""

    def test_log10(self):
        # Tests the special transformation for log10
        with SolverFactory("baron") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=log10(m.x) >= 2)
            m.obj = Objective(expr=m.x, sense=minimize)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)

    def test_abs(self):
        # Tests the special transformation for abs
        with SolverFactory("baron") as opt:

            m = ConcreteModel()
            m.x = Var(bounds=(-100,1))
            m.c = Constraint(expr=abs(m.x) >= 2)
            m.obj = Objective(expr=m.x, sense=minimize)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)

    def test_pow(self):
        # Tests the special transformation for x ^ y (both variables)
        with SolverFactory("baron") as opt:

            m = ConcreteModel()
            m.x = Var(bounds=(10,100))
            m.y = Var(bounds=(1,10))
            m.c = Constraint(expr=m.x ** m.y >= 20)
            m.obj = Objective(expr=m.x, sense=minimize)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)


if __name__ == '__main__':
    unittest.main()

"""Tests the BARON interface."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, Objective, Var, log10,
                           minimize)
from pyomo.opt import SolverFactory, TerminationCondition

# I don't know how you figure out if BARON is available
baron_available = True


class BaronTest(unittest.TestCase):
    """Test the BARON interface."""

    @unittest.skipIf(not baron_available,
                     "The 'BARON' solver is not available")
    def test_log10(self):
        """Tests the log10 functionality."""
        with SolverFactory("baron") as opt:

            m = ConcreteModel()
            m.x = Var()
            m.c = Constraint(expr=log10(m.x) >= 2)
            m.obj = Objective(expr=m.x, sense=minimize)

            results = opt.solve(m)

            self.assertEqual(results.solver.termination_condition,
                             TerminationCondition.optimal)


if __name__ == '__main__':
    unittest.main()

"""Tests for the GDPopt solver plugin."""
from math import fabs

import pyutilib.th as unittest

from pyomo.contrib.gdpopt.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.environ import SolverFactory, value

required_solvers = ('ipopt', 'gurobi')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


class TestGDPopt(unittest.TestCase):
    """Tests for the GDPopt solver plugin."""

    @unittest.skipIf(not subsolvers_available,
                     "Required subsolvers {} are not available"
                     .format(required_solvers))
    def test_LOA(self):
        """Test logic-based outer approximation."""
        with SolverFactory('gdpopt') as opt:
            model = EightProcessFlowsheet()
            opt.solve(model, strategy='LOA')

            self.assertTrue(fabs(value(model.profit.expr) - 68) <= 1E-2)


if __name__ == '__main__':
    unittest.main()

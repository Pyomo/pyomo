"""Tests for the GDPopt solver plugin."""
from math import fabs

import pyutilib.th as unittest

from pyomo.contrib.gdpopt.tests.eight_process_problem import \
    build_eight_process_flowsheet
from pyomo.environ import SolverFactory, value

required_solvers = ('ipopt', 'cbc')
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
class TestGDPopt(unittest.TestCase):
    """Tests for the GDPopt solver plugin."""

    def test_LOA(self):
        """Test logic-based outer approximation."""
        with SolverFactory('gdpopt') as opt:
            model = build_eight_process_flowsheet()
            opt.solve(model, strategy='LOA', mip='cbc')

            self.assertTrue(fabs(value(model.profit.expr) - 68) <= 1E-2)

    def test_LOA_maxBinary(self):
        """Test logic-based OA with max_binary initialization."""
        with SolverFactory('gdpopt') as opt:
            model = build_eight_process_flowsheet()
            opt.solve(model, strategy='LOA', init_strategy='max_binary',
                      mip='cbc')

            self.assertTrue(fabs(value(model.profit.expr) - 68) <= 1E-2)


if __name__ == '__main__':
    unittest.main()

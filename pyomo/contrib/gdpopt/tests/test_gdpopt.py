"""Tests for the GDPopt solver plugin."""
from math import fabs

import pyomo.core.base.symbolic
import pyutilib.th as unittest
from pyomo.contrib.gdpopt.tests.eight_process_problem import \
    build_eight_process_flowsheet
from pyomo.environ import SolverFactory, value

required_solvers = ('ipopt', 'cbc')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


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
            opt.solve(model, strategy='LOA',
                      mip=required_solvers[1],
                      nlp=required_solvers[0])

            self.assertTrue(fabs(value(model.profit.expr) - 68) <= 1E-2)

    def test_LOA_maxBinary(self):
        """Test logic-based OA with max_binary initialization."""
        with SolverFactory('gdpopt') as opt:
            model = build_eight_process_flowsheet()
            opt.solve(model, strategy='LOA', init_strategy='max_binary',
                      mip=required_solvers[1],
                      nlp=required_solvers[0])

            self.assertTrue(fabs(value(model.profit.expr) - 68) <= 1E-2)

    def test_LOA_custom_disjuncts(self):
        """Test logic-based OA with custom disjuncts initialization."""
        with SolverFactory('gdpopt') as opt:
            model = build_eight_process_flowsheet()
            initialize = [
                # Use units 1, 4, 7, 8
                [model.use_unit_1or2.disjuncts[0],
                 model.use_unit_4or5ornot.disjuncts[0],
                 model.use_unit_6or7ornot.disjuncts[1],
                 model.use_unit_8ornot.disjuncts[0]],
                # Use units 2, 3, 6, 8
                [model.use_unit_1or2.disjuncts[1],
                 model.use_unit_3ornot.disjuncts[0],
                 model.use_unit_6or7ornot.disjuncts[0],
                 model.use_unit_8ornot.disjuncts[0]]
            ]
            opt.solve(model, strategy='LOA', init_strategy='custom_disjuncts',
                      custom_init_disjuncts=initialize,
                      mip=required_solvers[1],
                      nlp=required_solvers[0])

            self.assertTrue(fabs(value(model.profit.expr) - 68) <= 1E-2)


if __name__ == '__main__':
    unittest.main()

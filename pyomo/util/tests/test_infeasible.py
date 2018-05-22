# -*- coding: UTF-8 -*-
"""Tests infeasible model debugging utilities."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Var, Constraint)
from pyomo.util.infeasible import (
    log_infeasible_constraints, log_infeasible_bounds,
    log_active_constraints, log_close_to_bounds)


class TestInfeasible(unittest.TestCase):
    """Tests infeasible model debugging utilities."""

    def test_log_infeasible_constraints(self):
        """Test for logging of infeasible constraints."""
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.c = Constraint(expr=m.x >= 2)
        m.c2 = Constraint(expr=m.x == 4)
        m.c3 = Constraint(expr=m.x <= 0)
        log_infeasible_constraints(m)
        m.x.setlb(2)
        m.x.setub(0)
        log_infeasible_bounds(m)
        log_active_constraints(m)
        log_close_to_bounds(m)


if __name__ == '__main__':
    unittest.main()

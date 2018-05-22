# -*- coding: UTF-8 -*-
"""Tests infeasible model debugging utilities."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Var, Constraint)
from pyomo.util.infeasible import log_infeasible_constraints


class TestInfeasible(unittest.TestCase):
    """Tests infeasible model debugging utilities."""

    def test_log_infeasible_constraints(self):
        """Test for logging of infeasible constraints."""
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.c = Constraint(expr=m.x >= 2)
        log_infeasible_constraints(m)


if __name__ == '__main__':
    unittest.main()

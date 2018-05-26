# -*- coding: UTF-8 -*-
"""Tests infeasible model debugging utilities."""
import logging

from six import StringIO

import pyutilib.th as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Var
from pyomo.util.infeasible import (log_active_constraints, log_close_to_bounds,
                                   log_infeasible_bounds,
                                   log_infeasible_constraints)


class TestInfeasible(unittest.TestCase):
    """Tests infeasible model debugging utilities."""

    def test_log_infeasible_constraints(self):
        """Test for logging of infeasible constraints."""
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.c = Constraint(expr=m.x >= 2)
        m.c2 = Constraint(expr=m.x == 4)
        m.c3 = Constraint(expr=m.x <= 0)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_infeasible_constraints(m)
        expected_output = [
            "CONSTR c: 1 < 2.0",
            "CONSTR c2: 1 != 4.0",
            "CONSTR c3: 1 > 0.0"]
        self.assertEqual(output.getvalue().splitlines(), expected_output)

        m.x.setlb(2)
        m.x.setub(0)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_infeasible_bounds(m)
        expected_output = ["VAR x: 1 < LB 2",
                           "VAR x: 1 > UB 0"]
        self.assertEqual(output.getvalue().splitlines(), expected_output)

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_active_constraints(m)
        expected_output = ["c active",
                           "c2 active",
                           "c3 active"]
        self.assertEqual(output.getvalue().splitlines(), expected_output)

        m.y = Var(bounds=(0, 2), initialize=1.9999999)
        m.c4 = Constraint(expr=m.y >= m.y.value)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_close_to_bounds(m)
        expected_output = ["y near UB of 2",
                           "c4 near LB"]
        self.assertEqual(output.getvalue().splitlines(), expected_output)


if __name__ == '__main__':
    unittest.main()

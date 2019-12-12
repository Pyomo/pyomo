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

    def build_model(self):
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.c = Constraint(expr=m.x >= 2)
        m.c2 = Constraint(expr=m.x == 4)
        m.c3 = Constraint(expr=m.x <= 0)
        m.y = Var(bounds=(0, 2), initialize=1.9999999)
        m.c4 = Constraint(expr=m.y >= m.y.value)
        m.z = Var(bounds=(0, 6))
        m.c5 = Constraint(expr=m.z >= 5)
        return m

    def test_log_infeasible_constraints(self):
        """Test for logging of infeasible constraints."""
        m = self.build_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_infeasible_constraints(m)
        expected_output = [
            "CONSTR c: 2.0 </= 1",
            "CONSTR c2: 1 =/= 4.0",
            "CONSTR c3: 1 </= 0.0",
            "CONSTR c5: 5.0 <?= missing variable value"]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_infeasible_bounds(self):
        """Test for logging of infeasible variable bounds."""
        m = self.build_model()
        m.x.setlb(2)
        m.x.setub(0)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_infeasible_bounds(m)
        expected_output = ["VAR x: 1 < LB 2",
                           "VAR x: 1 > UB 0"]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_active_constraints(self):
        """Test for logging of active constraints."""
        m = self.build_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_active_constraints(m)
        expected_output = ["c active",
                           "c2 active",
                           "c3 active",
                           "c4 active",
                           "c5 active"]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_close_to_bounds(self):
        """Test logging of variables and constraints near bounds."""
        m = self.build_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_close_to_bounds(m)
        expected_output = ["y near UB of 2",
                           "c4 near LB",
                           "Skipping CONSTR c5: missing variable value."]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_infeasible_constraints_verbose_expressions(self):
        """Test for logging of infeasible constraints."""
        m = self.build_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_infeasible_constraints(m, log_expression=True)
        expected_output = [
            "CONSTR c: 2.0 </= 1", "  2.0 </= x",
            "CONSTR c2: 1 =/= 4.0", "  x =/= 4.0",
            "CONSTR c3: 1 </= 0.0", "  x </= 0.0",
            "CONSTR c5: 5.0 <?= missing variable value", "  5.0 <?= z"]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_infeasible_constraints_verbose_variables(self):
        """Test for logging of infeasible constraints."""
        m = self.build_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_infeasible_constraints(m, log_variables=True)
        expected_output = [
            "CONSTR c: 2.0 </= 1", "  VAR x: 1",
            "CONSTR c2: 1 =/= 4.0", "  VAR x: 1",
            "CONSTR c3: 1 </= 0.0", "  VAR x: 1",
            "CONSTR c5: 5.0 <?= missing variable value", "  VAR z: None"]
        self.assertEqual(expected_output, output.getvalue().splitlines())


if __name__ == '__main__':
    unittest.main()

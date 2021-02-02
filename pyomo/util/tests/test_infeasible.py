# -*- coding: utf-8 -*-
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""Tests infeasible model debugging utilities."""
import logging

from six import StringIO

import pyutilib.th as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Var, inequality
from pyomo.util.infeasible import (log_active_constraints, log_close_to_bounds,
                                   log_infeasible_bounds,
                                   log_infeasible_constraints)


class TestInfeasible(unittest.TestCase):
    """Tests infeasible model debugging utilities."""

    def build_model(self):
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.c1 = Constraint(expr=m.x >= 2)
        m.c2 = Constraint(expr=m.x == 4)
        m.c3 = Constraint(expr=m.x <= 0)
        m.y = Var(bounds=(0, 2), initialize=1.9999999)
        m.c4 = Constraint(expr=m.y >= m.y.value)
        m.z = Var(bounds=(0, 6))
        m.c5 = Constraint(expr=inequality(5, m.z, 10), doc="Range infeasible")
        m.c6 = Constraint(expr=m.x + m.y <= 6, doc="Feasible")
        m.c7 = Constraint(expr=m.z == 6, doc="Equality infeasible")
        m.c8 = Constraint(expr=inequality(3, m.x, 6), doc="Range lb infeasible")
        m.c9 = Constraint(expr=inequality(0, m.x, 0.5), doc="Range ub infeasible")
        m.c10 = Constraint(expr=m.y >= 3, doc="Inactive")
        m.c10.deactivate()
        m.c11 = Constraint(expr=m.y <= m.y.value)
        m.yy = Var(bounds=(0, 1), initialize=1E-7, doc="Close to lower bound")
        m.y3 = Var(bounds=(0, 1E-7), initialize=0, doc="Bounds too close")
        m.y4 = Var(bounds=(0, 1), initialize=2, doc="Fixed out of bounds.")
        m.y4.fix()
        return m

    def test_log_infeasible_constraints(self):
        """Test for logging of infeasible constraints."""
        m = self.build_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_infeasible_constraints(m)
        expected_output = [
            "CONSTR c1: 2.0 </= 1",
            "CONSTR c2: 1 =/= 4.0",
            "CONSTR c3: 1 </= 0.0",
            "CONSTR c5: 5.0 <?= missing variable value <?= 10.0",
            "CONSTR c7: missing variable value =?= 6.0",
            "CONSTR c8: 3.0 </= 1 <= 6.0",
            "CONSTR c9: 0.0 <= 1 </= 0.5",
        ]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_infeasible_bounds(self):
        """Test for logging of infeasible variable bounds."""
        m = self.build_model()
        m.x.setlb(2)
        m.x.setub(0)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util', logging.INFO):
            log_infeasible_bounds(m)
        expected_output = [
            "VAR x: 1 >/= LB 2", "VAR x: 1 </= UB 0", "VAR y4: 2 </= UB 1",
        ]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_active_constraints(self):
        """Test for logging of active constraints."""
        m = self.build_model()
        depr = StringIO()
        output = StringIO()
        with LoggingIntercept(depr, 'pyomo'):
            with LoggingIntercept(output, 'pyomo.util', logging.INFO):
                log_active_constraints(m)
        self.assertIn("log_active_constraints is deprecated.", depr.getvalue())
        expected_output = [
            "c1 active", "c2 active", "c3 active", "c4 active",
            "c5 active", "c6 active", "c7 active", "c8 active",
            "c9 active", "c11 active"
        ]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_close_to_bounds(self):
        """Test logging of variables and constraints near bounds."""
        m = self.build_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_close_to_bounds(m)
        expected_output = [
            "y near UB of 2", "yy near LB of 0", "c4 near LB",
            "Skipping CONSTR c5: missing variable value.",
            "c11 near UB",
        ]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_infeasible_constraints_verbose_expressions(self):
        """Test for logging of infeasible constraints."""
        m = self.build_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_infeasible_constraints(m, log_expression=True)
        expected_output = [
            "CONSTR c1: 2.0 </= 1", "  - EXPR: 2.0 </= x",
            "CONSTR c2: 1 =/= 4.0", "  - EXPR: x =/= 4.0",
            "CONSTR c3: 1 </= 0.0", "  - EXPR: x </= 0.0",
            "CONSTR c5: 5.0 <?= missing variable value <?= 10.0", "  - EXPR: 5.0 <?= z <?= 10.0",
            "CONSTR c7: missing variable value =?= 6.0", "  - EXPR: z =?= 6.0",
            "CONSTR c8: 3.0 </= 1 <= 6.0", "  - EXPR: 3.0 </= x <= 6.0",
            "CONSTR c9: 0.0 <= 1 </= 0.5", "  - EXPR: 0.0 <= x </= 0.5",
        ]
        self.assertEqual(expected_output, output.getvalue().splitlines())

    def test_log_infeasible_constraints_verbose_variables(self):
        """Test for logging of infeasible constraints."""
        m = self.build_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.infeasible', logging.INFO):
            log_infeasible_constraints(m, log_variables=True)
        expected_output = [
            "CONSTR c1: 2.0 </= 1", "  - VAR x: 1",
            "CONSTR c2: 1 =/= 4.0", "  - VAR x: 1",
            "CONSTR c3: 1 </= 0.0", "  - VAR x: 1",
            "CONSTR c5: 5.0 <?= missing variable value <?= 10.0", "  - VAR z: None",
            "CONSTR c7: missing variable value =?= 6.0", "  - VAR z: None",
            "CONSTR c8: 3.0 </= 1 <= 6.0", "  - VAR x: 1",
            "CONSTR c9: 0.0 <= 1 </= 0.5", "  - VAR x: 1",
        ]
        self.assertEqual(expected_output, output.getvalue().splitlines())


if __name__ == '__main__':
    unittest.main()

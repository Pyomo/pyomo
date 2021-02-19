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

import logging

from six import StringIO

import pyutilib.th as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Constraint, Var, inequality
from pyomo.util.blockutil import (
    log_model_constraints,
)

class TestBlockutil(unittest.TestCase):
    """Tests block utilities."""

    def test_log_model_constraints(self):
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

        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util', logging.INFO):
            log_model_constraints(m)
        expected_output = [
            "c1 active", "c2 active", "c3 active", "c4 active",
            "c5 active", "c6 active", "c7 active", "c8 active",
            "c9 active", "c11 active"
        ]
        self.assertEqual(expected_output, output.getvalue().splitlines())

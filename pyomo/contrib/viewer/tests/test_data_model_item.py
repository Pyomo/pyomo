##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# This software is distributed under the 3-clause BSD License.
##############################################################################
"""
Test data model items for QTreeView. These tests should work even without PyQt.
"""

import pyutilib.th as unittest
from pyomo.environ import *
from pyomo.contrib.viewer.model_browser import ComponentDataItem

class TestDataModelItem(unittest.TestCase):
    def setUp(self):
        # Borrowed this test model from the trust region tests
        m = ConcreteModel()
        m.z = Var(range(3), domain=Reals, initialize=2.)
        m.x = Var(range(2), initialize=2.)
        m.x[1] = 1.0

        m.b1 = Block()
        m.b1.e1 = Expression(expr=m.x[0] + m.x[1])

        def blackbox(a,b):
            return sin(a-b)
        self.bb = ExternalFunction(blackbox)

        m.obj = Objective(
            expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
                + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6 # + m.bb(m.x[0],m.x[1])
            )
        m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + self.bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
        m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))
        m.c3 = Constraint(expr=m.x[1] == 3)
        self.m = m.clone()

    def test_expr_calc(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.b1.e1)
        cdi.calculate()
        assert(abs(cdi.get("value")-3) < 0.0001)

    def test_cons_calc(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.c3)
        cdi.calculate()
        assert(abs(cdi.get("residual") - 2) < 0.0001)

    def test_var_get_value(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.x[1])
        assert(abs(cdi.get("value") - 1) < 0.0001)

    def test_var_get_bounds(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.x[1])
        self.m.x[1].setlb(0)
        self.m.x[1].setub(10)
        assert(abs(cdi.get("lb") - 0) < 0.0001)
        assert(abs(cdi.get("ub") - 10) < 0.0001)

    def test_var_set_bounds(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.x[1])
        cdi.set("lb", 2)
        cdi.set("ub", 8)
        assert(abs(cdi.get("lb") - 2) < 0.0001)
        assert(abs(cdi.get("ub") - 8) < 0.0001)

    def test_var_fixed_bounds(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.x[1])
        cdi.set("fixed", True)
        assert(cdi.get("fixed"))
        cdi.set("fixed", False)
        assert(not cdi.get("fixed"))

    def test_degrees_of_freedom(self):
        import pyomo.contrib.viewer.report as rpt
        # this should hit everything in report.  It only exists to calculate
        # degrees of freedom for display in the ui
        assert(rpt.degrees_of_freedom(self.m)==2)

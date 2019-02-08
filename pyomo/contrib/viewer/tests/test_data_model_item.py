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

        self.m = m.clone()

    def test_expr_calc(self):
        cdi = ComponentDataItem(parent=None, ui_setup=None, o=self.m.b1.e1)
        cdi.calculate()
        assert(abs(cdi.get("value")-3) < 0.0001)

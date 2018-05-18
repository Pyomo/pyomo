"""Tests the variable aggregation module."""
import pyutilib.th as unittest

from pyomo.environ import (ConcreteModel, Constraint, RangeSet,
                           TransformationFactory, Var, value)


class TestVarAggregate(unittest.TestCase):
    """Tests variable aggregation."""

    def test_var_aggregate(self):
        """Test for transitivity in a variable equality set."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.v4 = Var(initialize=4)
        m.c1 = Constraint(expr=m.v1 == m.v2)
        m.c2 = Constraint(expr=m.v2 == m.v3)
        m.c3 = Constraint(expr=m.v3 == m.v4)
        m.v2.fix()

        m.s = RangeSet(4)
        m.x = Var(m.s, initialize=5)
        m.c = Constraint(m.s)
        m.c.add(1, expr=m.x[1] == m.x[3])
        m.c.add(2, expr=m.x[2] == m.x[4])
        m.c.add(3, expr=m.x[2] == m.x[3])
        m.c.add(4, expr=m.x[1] == 1)

        m.y = Var([1, 2], initialize=3)
        m.c_too = Constraint(expr=m.y[1] == m.y[2])

        m.z1 = Var()
        m.z2 = Var()
        m.ignore_me = Constraint(expr=m.y[1] + m.z1 + m.z2 <= 0)

        TransformationFactory('contrib.aggregate_vars').apply_to(m)
        m.pprint()
        from pprint import pprint
        z_to_vars = m._var_aggregator_info.z_to_vars
        pprint({z.name: list(v.name for v in z_to_vars[z]) for z in z_to_vars})
        # self.assertTrue(m.v1.fixed)
        # self.assertTrue(m.v2.fixed)
        # self.assertTrue(m.v3.fixed)
        # self.assertTrue(m.v4.fixed)
        # self.assertEquals(value(m.v4), 2)
        #
        # self.assertTrue(m.x[1].fixed)
        # self.assertTrue(m.x[2].fixed)
        # self.assertTrue(m.x[3].fixed)
        # self.assertTrue(m.x[4].fixed)
        # self.assertEquals(value(m.x[4]), 1)
        #
        # self.assertFalse(m.y[1].fixed)
        # self.assertFalse(m.y[2].fixed)
        # self.assertFalse(m.z1.fixed)
        # self.assertFalse(m.z2.fixed)
        # m.display()


if __name__ == '__main__':
    unittest.main()

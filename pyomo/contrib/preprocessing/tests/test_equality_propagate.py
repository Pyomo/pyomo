"""Tests the equality set propagation module."""
import pyutilib.th as unittest

from pyomo.environ import (ConcreteModel, Constraint, RangeSet,
                           TransformationFactory, Var, value)


class TestEqualityPropagate(unittest.TestCase):
    """Tests equality set variable attribute propagation."""

    def test_fixed_var_propagate(self):
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

        TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        self.assertEquals(value(m.v4), 2)

        self.assertTrue(m.x[1].fixed)
        self.assertTrue(m.x[2].fixed)
        self.assertTrue(m.x[3].fixed)
        self.assertTrue(m.x[4].fixed)
        self.assertEquals(value(m.x[4]), 1)

        self.assertFalse(m.y[1].fixed)
        self.assertFalse(m.y[2].fixed)
        self.assertFalse(m.z1.fixed)
        self.assertFalse(m.z2.fixed)
        # m.display()

    def test_fixed_var_propagate_backwards(self):
        """Test backwards propagation through equality set."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.v4 = Var(initialize=4)
        m.c1 = Constraint(expr=m.v1 == m.v2)
        m.c2 = Constraint(expr=m.v2 == m.v3)
        m.c3 = Constraint(expr=m.v3 == m.v4)
        m.v4.fix()
        # check to make sure that all the v's have the same equality set. John
        # had found a logic error.
        TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
        # m.display()
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        self.assertEquals(value(m.v4), 4)

    def test_var_fix_revert(self):
        """Test to make sure that variable fixing reversion works."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.v4 = Var(initialize=4)
        m.c1 = Constraint(expr=m.v1 == m.v2)
        m.c2 = Constraint(expr=m.v2 == m.v3)
        m.c3 = Constraint(expr=m.v3 == m.v4)
        m.v2.fix()
        fvp = TransformationFactory('contrib.propagate_fixed_vars')
        fvp.apply_to(m, tmp=True)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        fvp.revert(m)
        self.assertFalse(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertFalse(m.v4.fixed)

    def test_var_bound_propagate(self):
        """Test for transitivity in a variable equality set."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1, bounds=(1, 3))
        m.v2 = Var(initialize=2, bounds=(0, 8))
        m.v3 = Var(initialize=3, bounds=(2, 4))
        m.v4 = Var(initialize=4, bounds=(0, 5))
        m.c1 = Constraint(expr=m.v1 == m.v2)
        m.c2 = Constraint(expr=m.v2 == m.v3)
        m.c3 = Constraint(expr=m.v3 == m.v4)

        m.s = RangeSet(4)
        m.x = Var(m.s, initialize=5)
        m.x[2].setlb(-1)
        m.c = Constraint(m.s)
        m.c.add(1, expr=m.x[1] == m.x[3])
        m.c.add(2, expr=m.x[2] == m.x[4])
        m.c.add(3, expr=m.x[2] == m.x[3])
        m.c.add(4, expr=m.x[1] == 1)

        m.y = Var([1, 2], initialize=3)
        m.y[1].setub(3)
        m.y[2].setub(15)
        m.c_too = Constraint(expr=m.y[1] == m.y[2])

        m.z1 = Var(bounds=(1, 2))
        m.z2 = Var(bounds=(3, 4))
        m.ignore_me = Constraint(expr=m.y[1] + m.z1 + m.z2 <= 0)

        TransformationFactory('contrib.propagate_eq_var_bounds').apply_to(m)

        self.assertEquals(value(m.v1.lb), 2)
        self.assertEquals(value(m.v1.lb), value(m.v2.lb))
        self.assertEquals(value(m.v1.lb), value(m.v3.lb))
        self.assertEquals(value(m.v1.lb), value(m.v4.lb))
        self.assertEquals(value(m.v1.ub), 3)
        self.assertEquals(value(m.v1.ub), value(m.v2.ub))
        self.assertEquals(value(m.v1.ub), value(m.v3.ub))
        self.assertEquals(value(m.v1.ub), value(m.v4.ub))

        for i in m.s:
            self.assertEquals(value(m.x[i].lb), -1)

        self.assertEquals(value(m.y[1].ub), 3)
        self.assertEquals(value(m.y[2].ub), 3)
        self.assertEquals(value(m.y[1].lb), None)
        self.assertEquals(value(m.y[1].lb), None)

        self.assertEquals(value(m.z1.ub), 2)
        self.assertEquals(value(m.z2.ub), 4)
        self.assertEquals(value(m.z1.lb), 1)
        self.assertEquals(value(m.z2.lb), 3)

    def test_var_bound_propagate_crossover(self):
        """Test for error message when variable bound crosses over."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1, bounds=(1, 3))
        m.v2 = Var(initialize=5, bounds=(4, 8))
        m.c1 = Constraint(expr=m.v1 == m.v2)
        xfrm = TransformationFactory('contrib.propagate_eq_var_bounds')
        with self.assertRaises(ValueError):
            xfrm.apply_to(m)

    def test_var_bound_propagate_revert(self):
        """Test to make sure bound propagation revert works."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1, bounds=(1, 3))
        m.v2 = Var(initialize=2, bounds=(0, 8))
        m.v3 = Var(initialize=3, bounds=(2, 4))
        m.v4 = Var(initialize=4, bounds=(0, 5))
        m.c1 = Constraint(expr=m.v1 == m.v2)
        m.c2 = Constraint(expr=m.v2 == m.v3)
        m.c3 = Constraint(expr=m.v3 == m.v4)
        xfrm = TransformationFactory('contrib.propagate_eq_var_bounds')
        xfrm.apply_to(m, tmp=True)
        self.assertEquals(value(m.v1.lb), 2)
        self.assertEquals(value(m.v1.lb), value(m.v2.lb))
        self.assertEquals(value(m.v1.lb), value(m.v3.lb))
        self.assertEquals(value(m.v1.lb), value(m.v4.lb))
        self.assertEquals(value(m.v1.ub), 3)
        self.assertEquals(value(m.v1.ub), value(m.v2.ub))
        self.assertEquals(value(m.v1.ub), value(m.v3.ub))
        self.assertEquals(value(m.v1.ub), value(m.v4.ub))
        xfrm.revert(m)
        self.assertEquals(value(m.v1.lb), 1)
        self.assertEquals(value(m.v2.lb), 0)
        self.assertEquals(value(m.v3.lb), 2)
        self.assertEquals(value(m.v4.lb), 0)
        self.assertEquals(value(m.v1.ub), 3)
        self.assertEquals(value(m.v2.ub), 8)
        self.assertEquals(value(m.v3.ub), 4)
        self.assertEquals(value(m.v4.ub), 5)


if __name__ == '__main__':
    unittest.main()

"""Tests the equality set propagation module."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, TransformationFactory,
                           Var, value)

__author__ = "Qi Chen <qichen at andrew.cmu.edu>"


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
        # check to make sure that all the v's have the same equality set. John
        # had found a logic error.
        TransformationFactory('core.propagate_fixed_vars').apply_to(m)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        self.assertEquals(value(m.v4), 2)
        # m.display()

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
        fvp = TransformationFactory('core.propagate_fixed_vars')
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
        TransformationFactory('core.propagate_eq_var_bounds').apply_to(m)
        self.assertEquals(value(m.v1.lb), 2)
        self.assertEquals(value(m.v1.lb), value(m.v2.lb))
        self.assertEquals(value(m.v1.lb), value(m.v3.lb))
        self.assertEquals(value(m.v1.lb), value(m.v4.lb))
        self.assertEquals(value(m.v1.ub), 3)
        self.assertEquals(value(m.v1.ub), value(m.v2.ub))
        self.assertEquals(value(m.v1.ub), value(m.v3.ub))
        self.assertEquals(value(m.v1.ub), value(m.v4.ub))

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
        xfrm = TransformationFactory('core.propagate_eq_var_bounds')
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

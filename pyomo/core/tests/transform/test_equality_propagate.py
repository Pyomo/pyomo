"""Tests the equality set propagation module."""
from pyomo.core.plugins.transform.equality_propagate import FixedVarPropagator
from pyomo.environ import ConcreteModel, Var, Constraint
import pyutilib.th as unittest

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
        FixedVarPropagator().apply_to(m)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        # m.display()

    def test_var_fix_revert(self):
        """Test to make sure that reversion works."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.v4 = Var(initialize=4)
        m.c1 = Constraint(expr=m.v1 == m.v2)
        m.c2 = Constraint(expr=m.v2 == m.v3)
        m.c3 = Constraint(expr=m.v3 == m.v4)
        m.v2.fix()
        fvp = FixedVarPropagator()
        fvp.apply_to(m, tmp=True)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        fvp.revert()
        self.assertFalse(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertFalse(m.v3.fixed)
        self.assertFalse(m.v4.fixed)


if __name__ == '__main__':
    unittest.main()

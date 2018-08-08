"""Tests initialization of uninitialized variables."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, TransformationFactory,
                           value, Var)


class TestInitVars(unittest.TestCase):
    """Tests initialization of uninitialized variables."""

    def test_midpoint_var_init(self):
        """Test midpoint initialization."""
        m = ConcreteModel()
        m.v1 = Var()
        m.v2 = Var()
        m.v3 = Var()
        m.v4 = Var()
        m.v5 = Var(initialize=2)
        m.v5.fix()
        m.v6 = Var(initialize=3)
        m.v2.setlb(2)
        m.v3.setub(2)
        m.v4.setlb(0)
        m.v4.setub(2)

        TransformationFactory('contrib.init_vars_midpoint').apply_to(m)
        self.assertEquals(value(m.v1), 0)
        self.assertEquals(value(m.v2), 2)
        self.assertEquals(value(m.v3), 2)
        self.assertEquals(value(m.v4), 1)
        self.assertEquals(value(m.v5), 2)
        self.assertEquals(value(m.v6), 3)

        TransformationFactory('contrib.init_vars_midpoint').apply_to(
            m, overwrite=True)
        self.assertEquals(value(m.v1), 0)
        self.assertEquals(value(m.v2), 2)
        self.assertEquals(value(m.v3), 2)
        self.assertEquals(value(m.v4), 1)
        self.assertEquals(value(m.v5), 2)
        self.assertEquals(value(m.v6), 0)

    def test_zero_var_init(self):
        """Test zero initialization."""
        m = ConcreteModel()
        m.v1 = Var()
        m.v2 = Var()
        m.v3 = Var()
        m.v4 = Var()
        m.v5 = Var(initialize=2)
        m.v5.fix()
        m.v6 = Var(initialize=3)
        m.v2.setlb(2)
        m.v3.setub(-2)
        m.v4.setlb(0)
        m.v4.setub(2)

        TransformationFactory('contrib.init_vars_zero').apply_to(m)
        self.assertEquals(value(m.v1), 0)
        self.assertEquals(value(m.v2), 2)
        self.assertEquals(value(m.v3), -2)
        self.assertEquals(value(m.v4), 0)
        self.assertEquals(value(m.v5), 2)
        self.assertEquals(value(m.v6), 3)

        TransformationFactory('contrib.init_vars_zero').apply_to(
            m, overwrite=True)
        self.assertEquals(value(m.v1), 0)
        self.assertEquals(value(m.v2), 2)
        self.assertEquals(value(m.v3), -2)
        self.assertEquals(value(m.v4), 0)
        self.assertEquals(value(m.v5), 2)
        self.assertEquals(value(m.v6), 0)


if __name__ == '__main__':
    unittest.main()

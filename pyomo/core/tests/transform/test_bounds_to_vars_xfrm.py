"""Tests explicit bound to variable bound transformation module."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, TransformationFactory,
                           Var, value)

__author__ = "Qi Chen <https://github.com/qtothec>"


class TestConstraintToVarBoundTransform(unittest.TestCase):
    """Tests explicit to variable bound transformation."""

    def test_fixed_var_propagate(self):
        """Test for transitivity in a variable equality set."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.v4 = Var(initialize=4)
        m.v5 = Var(initialize=5)
        m.c1 = Constraint(expr=m.v1 == 2)
        m.c2 = Constraint(expr=m.v2 >= -2)
        m.c3 = Constraint(expr=m.v3 <= 5)
        m.c4 = Constraint(expr=m.v4 <= m.v5)
        m.v5.fix()

        m2 = TransformationFactory(
            'core.constraints_to_var_bounds').create_using(m)
        self.assertEquals(value(m2.v1.lb), 2)
        self.assertEquals(value(m2.v1.ub), 2)
        # at this point in time, do not expect for v1 to be fixed
        self.assertFalse(m2.v1.fixed)

        self.assertEquals(value(m2.v2.lb), -2)
        self.assertFalse(m2.v2.has_ub())

        self.assertEquals(value(m2.v3.ub), 5)
        self.assertFalse(m2.v3.has_lb())

        self.assertEquals(value(m2.v4.ub), 5)
        self.assertFalse(m2.v4.has_lb())

        del m2  # to keep from accidentally using it below

        TransformationFactory('core.constraints_to_var_bounds').apply_to(m)
        self.assertEquals(value(m.v1.lb), 2)
        self.assertEquals(value(m.v1.ub), 2)
        # at this point in time, do not expect for v1 to be fixed
        self.assertFalse(m.v1.fixed)

        self.assertEquals(value(m.v2.lb), -2)
        self.assertFalse(m.v2.has_ub())

        self.assertEquals(value(m.v3.ub), 5)
        self.assertFalse(m.v3.has_lb())

        self.assertEquals(value(m.v4.ub), 5)
        self.assertFalse(m.v4.has_lb())


if __name__ == '__main__':
    unittest.main()

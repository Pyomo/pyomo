"""Tests explicit bound to variable bound transformation module."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, TransformationFactory,
                           Var, value)


class TestConstraintToVarBoundTransform(unittest.TestCase):
    """Tests explicit bound to variable bound transformation."""

    def test_constraint_to_var_bound(self):
        """Test converting explicit constraints into variable bounds."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.v4 = Var(initialize=4)
        m.v5 = Var(initialize=5)
        m.v6 = Var()
        m.c1 = Constraint(expr=m.v1 == 2)
        m.c2 = Constraint(expr=m.v2 >= -2)
        m.c3 = Constraint(expr=m.v3 <= 5)
        m.c4 = Constraint(expr=m.v4 <= m.v5)
        m.v5.fix()
        m.c6 = Constraint(expr=m.v6 >= 2)

        m2 = TransformationFactory(
            'contrib.constraints_to_var_bounds').create_using(m)
        self.assertEqual(value(m2.v1.lb), 2)
        self.assertEqual(value(m2.v1.ub), 2)
        self.assertTrue(m2.v1.fixed)

        self.assertEqual(value(m2.v2.lb), -2)
        self.assertFalse(m2.v2.has_ub())

        self.assertEqual(value(m2.v3.ub), 5)
        self.assertFalse(m2.v3.has_lb())

        self.assertEqual(value(m2.v4.ub), 5)
        self.assertFalse(m2.v4.has_lb())

        self.assertEqual(value(m2.v6.lb), 2)
        self.assertFalse(m2.v6.has_ub())
        self.assertEqual(value(m2.v6, exception=False), None)

        del m2  # to keep from accidentally using it below

        TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m)
        self.assertEqual(value(m.v1.lb), 2)
        self.assertEqual(value(m.v1.ub), 2)
        self.assertTrue(m.v1.fixed)

        self.assertEqual(value(m.v2.lb), -2)
        self.assertFalse(m.v2.has_ub())

        self.assertEqual(value(m.v3.ub), 5)
        self.assertFalse(m.v3.has_lb())

        self.assertEqual(value(m.v4.ub), 5)
        self.assertFalse(m.v4.has_lb())

    def test_skip_trivial_constraints(self):
        """Tests handling of zero coefficients."""
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.c = Constraint(expr=m.x * m.y == m.z)
        m.z.fix(0)
        m.y.fix(0)
        TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m)
        self.assertEqual(m.c.body.polynomial_degree(), 1)
        self.assertTrue(m.c.active)
        self.assertFalse(m.x.has_lb())
        self.assertFalse(m.x.has_ub())

    def test_detect_fixed_false(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x == 3)
        TransformationFactory('contrib.constraints_to_var_bounds').apply_to(
            m, detect_fixed=False)
        self.assertFalse(m.c.active)
        self.assertTrue(m.x.has_lb())
        self.assertTrue(m.x.has_ub())
        self.assertFalse(m.x.fixed)


if __name__ == '__main__':
    unittest.main()

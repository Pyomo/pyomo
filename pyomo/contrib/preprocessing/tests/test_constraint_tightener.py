"""Tests the Bounds Tightening module."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, TransformationFactory, Var, value)


class TestIntervalTightener(unittest.TestCase):
    """Tests Bounds Tightening."""

    def test_constraint_bound_tightening(self):
        # Check for no coefficients
        m = ConcreteModel()
        m.v1 = Var(initialize=7, bounds=(7, 10))
        m.v2 = Var(initialize=2, bounds=(2, 5))
        m.v3 = Var(initialize=6, bounds=(6, 9))
        m.v4 = Var(initialize=1, bounds=(1, 1))
        m.c1 = Constraint(expr=m.v1 >= m.v2 + m.v3 + m.v4 + 1)

        self.assertEqual(value(m.c1.upper), 0)
        self.assertFalse(m.c1.has_lb())
        TransformationFactory('core.tighten_constraints_from_vars').apply_to(m)
        self.assertEqual(value(m.c1.upper), 0)
        self.assertEqual(value(m.c1.lower), 0)

    def test_less_than_constraint(self):
        m = ConcreteModel()
        m.v1 = Var(initialize=7, bounds=(7, 10))
        m.v2 = Var(initialize=2, bounds=(2, 5))
        m.v3 = Var(initialize=6, bounds=(6, 9))
        m.v4 = Var(initialize=1, bounds=(1, 1))
        m.c1 = Constraint(expr=m.v1 <= m.v2 + m.v3 + m.v4)

        self.assertEqual(value(m.c1.upper), 0)
        self.assertFalse(m.c1.has_lb())
        TransformationFactory('core.tighten_constraints_from_vars').apply_to(m)
        self.assertEqual(value(m.c1.upper), 0)
        self.assertEqual(value(m.c1.lower), -8)

    def test_constraint_with_coef(self):
        """Test with coefficient on constraint."""
        m = ConcreteModel()
        m.v1 = Var(initialize=7, bounds=(7, 10))
        m.v2 = Var(initialize=2, bounds=(2, 5))
        m.v3 = Var(initialize=6, bounds=(6, 9))
        m.v4 = Var(initialize=1, bounds=(1, 1))
        m.c1 = Constraint(expr=m.v1 <= 2 * m.v2 + m.v3 + m.v4)

        self.assertEqual(value(m.c1.upper), 0)
        self.assertFalse(m.c1.has_lb())
        TransformationFactory('core.tighten_constraints_from_vars').apply_to(m)
        self.assertEqual(value(m.c1.upper), -1)
        self.assertEqual(value(m.c1.lower), -13)

    def test_unbounded_var(self):
        """test with unbounded variables"""
        m = ConcreteModel()
        m.v1 = Var(initialize=7)
        m.v2 = Var(initialize=2, bounds=(2, 5))
        m.v3 = Var(initialize=6, bounds=(6, 9))
        m.v4 = Var(initialize=1, bounds=(1, 1))
        m.c1 = Constraint(expr=m.v1 <= 2 * m.v2 + m.v3 + m.v4)

        self.assertEqual(value(m.c1.upper), 0)
        self.assertFalse(m.c1.has_lb())
        TransformationFactory('core.tighten_constraints_from_vars').apply_to(m)
        self.assertEqual(value(m.c1.upper), 0)
        self.assertFalse(m.c1.has_lb())

    def test_unbounded_one_direction(self):
        """Unbounded in one direction"""
        m = ConcreteModel()
        m.v1 = Var(initialize=7, bounds=(-float('inf'), 10))
        m.v2 = Var(initialize=2, bounds=(2, 5))
        m.v3 = Var(initialize=6, bounds=(6, 9))
        m.v4 = Var(initialize=1, bounds=(1, 1))
        m.c1 = Constraint(expr=m.v1 <= 2 * m.v2 + m.v3 + m.v4)

        self.assertEqual(value(m.c1.upper), 0)
        self.assertFalse(m.c1.has_lb())
        TransformationFactory('core.tighten_constraints_from_vars').apply_to(m)
        self.assertEqual(value(m.c1.upper), -1)
        self.assertFalse(m.c1.has_lb())

    def test_ignore_nonlinear(self):
        m = ConcreteModel()
        m.v1 = Var()
        m.c1 = Constraint(expr=m.v1 * m.v1 >= 2)

        self.assertEqual(value(m.c1.lower), 2)
        self.assertFalse(m.c1.has_ub())
        TransformationFactory('core.tighten_constraints_from_vars').apply_to(m)
        self.assertEqual(value(m.c1.lower), 2)
        self.assertFalse(m.c1.has_ub())


if __name__ == '__main__':
    unittest.main()

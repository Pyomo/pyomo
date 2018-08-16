"""Tests the zero sum propagation module."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, TransformationFactory,
                           Var, NonNegativeReals, NonPositiveReals, Binary)


class TestZeroSumPropagate(unittest.TestCase):
    """Tests zero sum propagation."""

    def test_non_negative_propagate(self):
        """Test zero sum propagation for non-negative vars."""
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        m.v2 = Var(initialize=2, domain=NonNegativeReals)
        m.v3 = Var(initialize=3, domain=NonNegativeReals)
        m.v4 = Var(initialize=4, domain=NonNegativeReals)
        m.c1 = Constraint(expr=m.v1 == m.v2 + m.v3 + m.v4)
        m.v1.fix()
        # Because v1 = 0, all the other values must be zero
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        del m

        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        m.v2 = Var(initialize=2, domain=NonNegativeReals)
        m.v3 = Var(initialize=3, domain=NonNegativeReals)
        m.v4 = Var(initialize=4, domain=NonNegativeReals)
        m.c1 = Constraint(expr=m.v2 + m.v3 + m.v4 == m.v1)
        m.v1.fix()
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        del m

        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        m.v2 = Var(initialize=2, domain=NonNegativeReals)
        m.v3 = Var(initialize=3, domain=NonNegativeReals)
        m.v4 = Var(initialize=4, domain=NonNegativeReals)
        m.c1 = Constraint(expr=m.v1 >= m.v2 + m.v3 + m.v4)
        m.v1.fix()
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)

    def test_not_transform_improperly(self):
        """Tests that invalid constraints are not transformed."""
        m = ConcreteModel()
        m.v1 = Var(initialize=0, domain=Binary)
        m.c1 = Constraint(expr=-1 * m.v1 <= 0)
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        self.assertFalse(m.v1.fixed)

    def test_non_positive_propagate(self):
        """Tests zero sum propagation for non-positive vars."""
        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        m.v2 = Var(initialize=-2, domain=NonPositiveReals)
        m.v3 = Var(initialize=-3, domain=NonPositiveReals)
        m.v4 = Var(initialize=-4, domain=NonPositiveReals)
        m.c1 = Constraint(expr=m.v1 == m.v2 + m.v3 + m.v4)
        m.v1.fix()
        # Because v1 = 0, all the other values must be zero
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        del m

        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        m.v2 = Var(initialize=-2, domain=NonPositiveReals)
        m.v3 = Var(initialize=-3, domain=NonPositiveReals)
        m.v4 = Var(initialize=-4, domain=NonPositiveReals)
        m.c1 = Constraint(expr=m.v2 + m.v3 + m.v4 == m.v1)
        m.v1.fix()
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)
        del m

        m = ConcreteModel()
        m.v1 = Var(initialize=0)
        m.v2 = Var(initialize=-2, domain=NonPositiveReals)
        m.v3 = Var(initialize=-3, domain=NonPositiveReals)
        m.v4 = Var(initialize=-4, domain=NonPositiveReals)
        m.c1 = Constraint(expr=m.v1 <= m.v2 + m.v3 + m.v4)
        m.v1.fix()
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        self.assertTrue(m.v1.fixed)
        self.assertTrue(m.v2.fixed)
        self.assertTrue(m.v3.fixed)
        self.assertTrue(m.v4.fixed)


if __name__ == '__main__':
    unittest.main()

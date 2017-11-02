"""Tests the Bounds Tightening module."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, TransformationFactory,
                           Var, NonNegativeReals, NonPositiveReals)

__author__ = "Sunjeev Kale <https://github.com/sjkale>"


class TestBoundsTightener(unittest.TestCase):
    """Tests Bounds Tightening."""

    def test_constraint_bound_tightening(self):

        #Check for no coefficients
        m = ConcreteModel()
        m.v1 = Var(initialize=7, bounds=(7,10))
        m.v2 = Var(initialize=2, bounds=(2,5))
        m.v3 = Var(initialize=6, bounds=(6,9))
        m.v4 = Var(initialize=1, bounds=(1,1))
        m.c1 = Constraint(expr=m.v1 >= m.v2 + m.v3 + m.v4)

        TransformationFactory('core.bounds_tightener').apply_to(m)
        self.assertTrue(m.c1._upper == 0)
        self.assertTrue(m.c1._lower == -1)
        del m

        m = ConcreteModel()
        m.v1 = Var(initialize=7, bounds=(7,10))
        m.v2 = Var(initialize=2, bounds=(2,5))
        m.v3 = Var(initialize=6, bounds=(6,9))
        m.v4 = Var(initialize=1, bounds=(1,1))
        m.c1 = Constraint(expr=m.v1 <= m.v2 + m.v3 + m.v4)

        TransformationFactory('core.bounds_tightener').apply_to(m)
        self.assertTrue(m.c1._upper == 0)
        self.assertTrue(m.c1._lower == -8)
        del m

        #test for coefficients
        m = ConcreteModel()
        m.v1 = Var(initialize=7, bounds=(7,10))
        m.v2 = Var(initialize=2, bounds=(2,5))
        m.v3 = Var(initialize=6, bounds=(6,9))
        m.v4 = Var(initialize=1, bounds=(1,1))
        m.c1 = Constraint(expr=m.v1 <= 2 *  m.v2 + m.v3 + m.v4)

        TransformationFactory('core.bounds_tightener').apply_to(m)
        self.assertTrue(m.c1._upper == -1)
        self.assertTrue(m.c1._lower == -13)
        del m

        #test for unbounded variables
        m = ConcreteModel()
        m.v1 = Var(initialize=7)
        m.v2 = Var(initialize=2, bounds=(2,5))
        m.v3 = Var(initialize=6, bounds=(6,9))
        m.v4 = Var(initialize=1, bounds=(1,1))
        m.c1 = Constraint(expr=m.v1 <= 2 *  m.v2 + m.v3 + m.v4)

        TransformationFactory('core.bounds_tightener').apply_to(m)
        self.assertTrue(m.c1._upper == 0)
        self.assertTrue(m.c1._lower == -float('inf'))
        del m

        #test for coefficients
        m = ConcreteModel()
        m.v1 = Var(initialize=7, bounds=(-float('inf'),10))
        m.v2 = Var(initialize=2, bounds=(2,5))
        m.v3 = Var(initialize=6, bounds=(6,9))
        m.v4 = Var(initialize=1, bounds=(1,1))
        m.c1 = Constraint(expr=m.v1 <= 2 *  m.v2 + m.v3 + m.v4)

        TransformationFactory('core.bounds_tightener').apply_to(m)
        self.assertTrue(m.c1._upper == -1)
        self.assertTrue(m.c1._lower == -float('inf'))
        del m





if __name__ == '__main__':
    unittest.main()

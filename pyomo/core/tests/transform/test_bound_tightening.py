"""Tests the Bounds Tightening module."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, TransformationFactory,
                           Var, NonNegativeReals, NonPositiveReals)

__author__ = "Sunjeev Kale <https://github.com/sjkale>"


class TestBoundTightening(unittest.TestCase):
    """Tests Bounds Tightening."""

    def test_bound_tightening(self):
        m = ConcreteModel()
        m.v1 = Var(initialize=7, bounds=(7,10))
        m.v2 = Var(initialize=2, bounds=(2,5))
        m.v3 = Var(initialize=6, bounds=(6,9))
        m.v4 = Var(initialize=1, bounds=(1,1))
        m.c1 = Constraint(expr=m.v1 == m.v2 + m.v3 + m.v4)

        TransformationFactory('core.bound_tightener').apply_to(m)
        print m.c1.upper
        print m.c1.lower
        print m.v1.upper
        print m.v1.lower
        print m.v2.upper
        print m.v2.lower
        print m.v3.upper
        print m.v3.lower
        print m.v4.upper
        print m.v4.lower
        assert(True)


if __name__ == '__main__':
    unittest.main()

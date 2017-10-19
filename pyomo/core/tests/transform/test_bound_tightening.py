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
        m.v2 = Var(initialize=2, domain=(2,5))
        m.v3 = Var(initialize=6, domain=(6,9))
        m.v4 = Var(initialize=1, domain=(1,1))
        m.c1 = Constraint(expr=m.v1 == m.v2 + m.v3 + m.v4)

        TransformationFactory('core.bound_tightener').apply_to(m)

        

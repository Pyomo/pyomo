# -*- coding: UTF-8 -*-
"""Tests disjunct fixing."""
import pyutilib.th as unittest
from pyomo.environ import (Block,
                           Constraint, ConcreteModel, TransformationFactory,
                           RangeSet, NonNegativeReals, Var)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error


class TestFixDisjuncts(unittest.TestCase):
    """Tests fixing of disjuncts."""

    def test_fix_disjunct(self):
        """Test for partial deactivation of a indexed disjunction."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x >= 2)
        m.d3 = Disjunct()
        m.d3.c = Constraint(expr=m.x >= 3)
        m.d4 = Disjunct()
        m.d4.c = Constraint(expr=m.x >= 4)

        @m.Disjunction([0, 1])
        def disj(m, i):
            if i == 0:
                return [m.d1, m.d2]
            else:
                return [m.d3, m.d4]

        m.d1.indicator_var.fix(0)
        m.d2.indicator_var.fix(1)
        TransformationFactory('gdp.fix_disjuncts').apply_to(
            m, targets=m.disj[0])
        TransformationFactory('gdp.bigm').apply_to(m)
        m.pprint()


if __name__ == '__main__':
    unittest.main()

"""Tests explicit bound to variable bound transformation module."""
import pyutilib.th as unittest
from pyomo.contrib.gdp_bounds.plugins.compute_bounds import (disjunctive_lb,
                                                             disjunctive_ub)
from pyomo.environ import (ConcreteModel, Constraint, Objective,
                           TransformationFactory, Var, value)
from pyomo.gdp import Disjunct, Disjunction


class TestGDPBounds(unittest.TestCase):
    """Tests disjunctive variable bounds implementation."""

    def test_enable_bounds(self):
        """Test enabling disjunctive bounds."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 8))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 2)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 4)
        m.disj = Disjunction(expr=[m.d1, m.d2])
        m.obj = Objective(expr=m.x)
        TransformationFactory('contrib.enable_disjunctive_bounds').apply_to(m)
        self.assertTrue(hasattr(m.d1, '_disjunctive_bounds'))
        self.assertTrue(hasattr(m.d2, '_disjunctive_bounds'))
        self.assertTrue(hasattr(m.d1, 'disjunctive_var_constraints'))
        self.assertTrue(hasattr(m.d2, 'disjunctive_var_constraints'))

    def test_compute_bounds(self):
        """Test computation of disjunctive bounds."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 8))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 2)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 4)
        m.disj = Disjunction(expr=[m.d1, m.d2])
        m.obj = Objective(expr=m.x)
        TransformationFactory('contrib.compute_disjunctive_bounds').apply_to(m)
        self.assertEquals(m.d1._disjunctive_bounds[m.x], (2, 8))
        self.assertEquals(m.d2._disjunctive_bounds[m.x], (0, 4))
        self.assertEquals(disjunctive_lb(m.x, m.d1), 2)
        self.assertEquals(disjunctive_ub(m.x, m.d1), 8)
        self.assertEquals(disjunctive_lb(m.x, m.d2), 0)
        self.assertEquals(disjunctive_ub(m.x, m.d2), 4)


if __name__ == '__main__':
    unittest.main()

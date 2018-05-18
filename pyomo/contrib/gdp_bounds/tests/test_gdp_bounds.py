"""Tests explicit bound to variable bound transformation module."""
import pyutilib.th as unittest
from pyomo.contrib.gdp_bounds.plugins.compute_bounds import (disjunctive_lb,
                                                             disjunctive_ub)
from pyomo.core import ComponentMap
from pyomo.environ import (ConcreteModel, Constraint, Objective,
                           TransformationFactory, Var, value)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import check_available_solvers

solvers = check_available_solvers('cbc')


class TestGDPBounds(unittest.TestCase):
    """Tests disjunctive variable bounds implementation."""

    @unittest.skipIf('cbc' not in solvers, "CBC solver not available")
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
        TransformationFactory('contrib.compute_disj_var_bounds').apply_to(m)
        self.assertEquals(m.d1._disj_var_bounds[m.x], (2, 8))
        self.assertEquals(m.d2._disj_var_bounds[m.x], (0, 4))
        self.assertEquals(disjunctive_lb(m.x, m.d1), 2)
        self.assertEquals(disjunctive_ub(m.x, m.d1), 8)
        self.assertEquals(disjunctive_lb(m.x, m.d2), 0)
        self.assertEquals(disjunctive_ub(m.x, m.d2), 4)
        self.assertEquals(len(m.d1._disjunctive_var_constraints), 2)
        self.assertEquals(len(m.d2._disjunctive_var_constraints), 2)
        self.assertIs(m.d1._disjunctive_var_constraints[1].body, m.x)
        self.assertEquals(m.d1._disjunctive_var_constraints[1].lower, 2)
        self.assertIs(m.d1._disjunctive_var_constraints[2].body, m.x)
        self.assertEquals(m.d1._disjunctive_var_constraints[2].upper, 8)
        self.assertIs(m.d2._disjunctive_var_constraints[1].body, m.x)
        self.assertEquals(m.d2._disjunctive_var_constraints[1].lower, 0)
        self.assertIs(m.d2._disjunctive_var_constraints[2].body, m.x)
        self.assertEquals(m.d2._disjunctive_var_constraints[2].upper, 4)


if __name__ == '__main__':
    unittest.main()

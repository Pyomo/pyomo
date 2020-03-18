"""Tests explicit bound to variable bound transformation module."""
import pyutilib.th as unittest
from pyomo.contrib.gdp_bounds.info import (
    disjunctive_lb, disjunctive_ub)
from pyomo.environ import (ConcreteModel, Constraint, Objective,
                           TransformationFactory, Var)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import check_available_solvers

solvers = check_available_solvers('cbc')


class TestGDPBounds(unittest.TestCase):
    """Tests disjunctive variable bounds implementation."""

    @unittest.skipIf('cbc' not in solvers, "CBC solver not available")
    def test_compute_bounds_obbt(self):
        """Test computation of disjunctive bounds."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 8))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 2)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x <= 4)
        m.disj = Disjunction(expr=[m.d1, m.d2])
        m.obj = Objective(expr=m.x)
        TransformationFactory('contrib.compute_disj_var_bounds').apply_to(m, solver='cbc')
        self.assertEqual(m.d1._disj_var_bounds[m.x], (2, 8))
        self.assertEqual(m.d2._disj_var_bounds[m.x], (0, 4))
        self.assertEqual(disjunctive_lb(m.x, m.d1), 2)
        self.assertEqual(disjunctive_ub(m.x, m.d1), 8)
        self.assertEqual(disjunctive_lb(m.x, m.d2), 0)
        self.assertEqual(disjunctive_ub(m.x, m.d2), 4)

    @unittest.skipIf('cbc' not in solvers, "CBC solver not available")
    def test_compute_bounds_obbt_prune_disjunct(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 2)
        m.d1.c2 = Constraint(expr=m.x <= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x + 3 == 0)
        m.disj = Disjunction(expr=[m.d1, m.d2])
        m.obj = Objective(expr=m.x)
        TransformationFactory('contrib.compute_disj_var_bounds').apply_to(m, solver='cbc')
        self.assertFalse(m.d1.active)
        self.assertEqual(m.d1.indicator_var, 0)
        self.assertTrue(m.d1.indicator_var.fixed)

    def test_compute_bounds_fbbt(self):
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
        self.assertEqual(m.d1._disj_var_bounds[m.x], (2, 8))
        self.assertEqual(m.d2._disj_var_bounds[m.x], (0, 4))
        self.assertEqual(disjunctive_lb(m.x, m.d1), 2)
        self.assertEqual(disjunctive_ub(m.x, m.d1), 8)
        self.assertEqual(disjunctive_lb(m.x, m.d2), 0)
        self.assertEqual(disjunctive_ub(m.x, m.d2), 4)

    def test_nested_fbbt(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 8))
        m.y = Var(bounds=(0, 8))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x >= 2)
        m.d1.innerD1 = Disjunct()
        m.d1.innerD1.c = Constraint(expr=m.y >= m.x + 3)
        m.d1.innerD2 = Disjunct()
        m.d1.innerD2.c = Constraint(expr=m.y <= m.x - 4)
        m.d1.innerDisj = Disjunction(expr=[m.d1.innerD1, m.d1.innerD2])
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x == 3)
        m.disj = Disjunction(expr=[m.d1, m.d2])
        TransformationFactory('contrib.compute_disj_var_bounds').apply_to(m)
        self.assertEqual(disjunctive_lb(m.y, m.d1), 0)
        self.assertEqual(disjunctive_lb(m.y, m.d1.innerD1), 5)
        self.assertEqual(disjunctive_ub(m.y, m.d1.innerD2), 4)


if __name__ == '__main__':
    unittest.main()

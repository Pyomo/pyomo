import pyomo.environ as pe
from pyomo.contrib import appsi
from pyomo.common import unittest
from pyomo.contrib.coramin.utils.compare_models import (
    is_relaxation,
    is_equivalent,
    _attempt_presolve,
)
from pyomo.core.expr.compare import assertExpressionsEqual, compare_expressions


highs_available = appsi.solvers.Highs().available()


class TestCompareModels(unittest.TestCase):
    @unittest.skipUnless(highs_available, 'HiGHS is not available')
    def test_compare_models_1(self):
        m1 = pe.ConcreteModel()
        m1.x = x = pe.Var(bounds=(-5, 4))
        m1.y = y = pe.Var(bounds=(0, 7))

        m1.c1 = pe.Constraint(expr=x + y == 1)

        m2 = pe.ConcreteModel()
        m2.c1 = pe.Constraint(expr=x + y <= 1)
        m2.c2 = pe.Constraint(expr=x + y >= 1)

        opt = appsi.solvers.Highs()

        self.assertTrue(is_equivalent(m1, m2, opt))
        m2.c2.deactivate()
        self.assertFalse(is_equivalent(m1, m2, opt))
        self.assertTrue(is_relaxation(m2, m1, opt))
        self.assertFalse(is_relaxation(m1, m2, opt))

    def _get_model(self):
        m = pe.ConcreteModel()
        m.x1 = pe.Var()
        m.x2 = pe.Var()
        m.x3 = pe.Var(bounds=(-3, 3))
        m.x4 = pe.Var()
        m.c1 = pe.Constraint(expr=m.x1 + m.x2 + m.x3 + m.x4 == 1)
        m.c2 = pe.Constraint(expr=m.x2 + m.x3 + m.x4 == 1)
        m.c3 = pe.Constraint(expr=m.x3 + m.x4 == 1)
        m.c4 = pe.Constraint(expr=m.x4 == 1)
        return m

    def _compare_expressions(self, got, options):
        success = False
        for exp in options:
            if compare_expressions(got, exp):
                success = True
                break
        return success

    def test_presolve1(self):
        m = self._get_model()
        success = _attempt_presolve(m, [m.x3])
        self.assertTrue(success)
        self.assertTrue(m.c1.active)
        self.assertTrue(m.c2.active)
        self.assertFalse(m.c3.active)
        self.assertTrue(m.c4.active)
        self.assertTrue(
            self._compare_expressions(
                m.c1.body,
                [m.x1 + m.x2 + 1, m.x2 + m.x1 + 1, 1 + m.x1 + m.x2, 1 + m.x2 + m.x1],
            )
        )
        self.assertTrue(self._compare_expressions(m.c2.body, [m.x2 + 1, 1 + m.x2]))
        self.assertEqual(m.c1.lb, 1)
        self.assertEqual(m.c1.ub, 1)
        self.assertEqual(m.c2.lb, 1)
        self.assertEqual(m.c2.ub, 1)
        self.assertEqual(len(m.bound_constraints), 1)
        self.assertTrue(
            self._compare_expressions(m.bound_constraints[1].body, [1 - m.x4])
        )
        self.assertEqual(m.bound_constraints[1].lb, -3)
        self.assertEqual(m.bound_constraints[1].ub, 3)

    def test_presolve2(self):
        m = self._get_model()
        success = _attempt_presolve(m, [m.x3, m.x4])
        self.assertTrue(success)
        self.assertTrue(m.c1.active)
        self.assertTrue(m.c2.active)
        self.assertFalse(m.c3.active)
        self.assertFalse(m.c4.active)
        self.assertTrue(
            self._compare_expressions(
                m.c1.body,
                [m.x1 + m.x2 + 1, m.x2 + m.x1 + 1, 1 + m.x1 + m.x2, 1 + m.x2 + m.x1],
            )
        )
        self.assertTrue(self._compare_expressions(m.c2.body, [m.x2 + 1, 1 + m.x2]))
        self.assertEqual(m.c1.lb, 1)
        self.assertEqual(m.c1.ub, 1)
        self.assertEqual(m.c2.lb, 1)
        self.assertEqual(m.c2.ub, 1)
        self.assertEqual(len(m.bound_constraints), 1)
        self.assertEqual(m.bound_constraints[1].body, 0)
        self.assertEqual(m.bound_constraints[1].lb, -3)
        self.assertEqual(m.bound_constraints[1].ub, 3)

    def test_presolve3(self):
        m = self._get_model()
        success = _attempt_presolve(m, [m.x3, m.x4, m.x2])
        self.assertTrue(success)
        self.assertTrue(m.c1.active)
        self.assertFalse(m.c2.active)
        self.assertFalse(m.c3.active)
        self.assertFalse(m.c4.active)
        self.assertTrue(self._compare_expressions(m.c1.body, [m.x1 + 1, 1 + m.x1]))
        self.assertEqual(m.c1.lb, 1)
        self.assertEqual(m.c1.ub, 1)
        self.assertEqual(len(m.bound_constraints), 1)
        self.assertEqual(m.bound_constraints[1].body, 0)
        self.assertEqual(m.bound_constraints[1].lb, -3)
        self.assertEqual(m.bound_constraints[1].ub, 3)

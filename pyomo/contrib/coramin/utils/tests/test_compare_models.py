import pyomo.environ as pe
from pyomo.contrib import appsi
from pyomo.common import unittest
from pyomo.contrib.coramin.utils.compare_models import is_relaxation, is_equivalent


class TestCompareModels(unittest.TestCase):
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

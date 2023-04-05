from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.fbbt.tests.test_fbbt import FbbtTestBase
from pyomo.common.errors import InfeasibleConstraintException
import math

pe = pyo


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestFbbt(FbbtTestBase, unittest.TestCase):
    def setUp(self) -> None:
        self.it = appsi.fbbt.IntervalTightener()
        self.tightener = self.it.perform_fbbt


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestFBBTPersistent(unittest.TestCase):
    def test_persistent(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.w = pe.Var()
        m.xl = pe.Param(mutable=True)
        m.xu = pe.Param(mutable=True)
        m.yl = pe.Param(mutable=True)
        m.yu = pe.Param(mutable=True)
        m.c1 = pe.Constraint(expr=m.w >= m.xl * m.y + m.x * m.yl - m.xl * m.yl)
        m.c2 = pe.Constraint(expr=m.w >= m.xu * m.y + m.x * m.yu - m.xu * m.yu)
        m.c3 = pe.Constraint(expr=m.w <= m.xu * m.y + m.x * m.yl - m.xu * m.yl)
        m.c4 = pe.Constraint(expr=m.w <= m.x * m.yu + m.xl * m.y - m.xl * m.yu)

        m.xl.value = -2
        m.xu.value = 2
        m.yl.value = -2
        m.yu.value = 2
        m.x.setlb(m.xl.value)
        m.x.setub(m.xu.value)
        m.y.setlb(m.yl.value)
        m.y.setub(m.yu.value)
        it = appsi.fbbt.IntervalTightener()
        it.perform_fbbt(m, symbolic_solver_labels=True)
        self.assertAlmostEqual(m.w.lb, -12)
        self.assertAlmostEqual(m.w.ub, 12)

        m.xl.value = -1
        m.xu.value = 1
        m.x.setlb(m.xl.value)
        m.x.setub(m.xu.value)
        it.perform_fbbt_with_seed(m, m.x)
        self.assertAlmostEqual(m.w.lb, -6)
        self.assertAlmostEqual(m.w.ub, 6)

        m.xl.value = -0.5
        m.xu.value = 0.5
        m.x.setlb(m.xl.value)
        m.x.setub(m.xu.value)
        it.perform_fbbt(m)
        self.assertAlmostEqual(m.w.lb, -3)
        self.assertAlmostEqual(m.w.ub, 3)

        m.yl.value = -1
        m.yu.value = 1
        m.y.setlb(m.yl.value)
        m.y.setub(m.yu.value)
        del m.c1
        del m.c2
        it.perform_fbbt(m)
        self.assertAlmostEqual(m.w.lb, -3)
        self.assertAlmostEqual(m.w.ub, 1.5)

    def test_sync_after_infeasible(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1, 1))
        m.y = pe.Var()
        m.c1 = pe.Constraint(expr=m.x == m.y)
        m.c2 = pe.Constraint(expr=m.y == 2)
        it = appsi.fbbt.IntervalTightener()
        try:
            it.perform_fbbt(m)
            was_infeasible = False
        except InfeasibleConstraintException:
            was_infeasible = True
        self.assertTrue(was_infeasible)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 1)
        self.assertAlmostEqual(m.y.lb, 1)
        self.assertAlmostEqual(m.y.ub, 1)

        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1, 1))
        m.y = pe.Var()
        m.c1 = pe.Constraint(expr=m.x == m.y)
        m.c2 = pe.Constraint(expr=m.y == 2)
        it = appsi.fbbt.IntervalTightener()
        try:
            it.perform_fbbt_with_seed(m, m.x)
            was_infeasible = False
        except InfeasibleConstraintException:
            was_infeasible = True
        self.assertTrue(was_infeasible)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 1)
        self.assertAlmostEqual(m.y.lb, 1)
        self.assertAlmostEqual(m.y.ub, 1)

    def test_deactivated_constraints(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.c1 = pe.Constraint(expr=m.x == 1)
        m.c2 = pe.Constraint(expr=m.y == m.x)
        it = appsi.fbbt.IntervalTightener()
        it.config.deactivate_satisfied_constraints = True
        it.perform_fbbt(m)
        self.assertFalse(m.c1.active)
        self.assertFalse(m.c2.active)
        m.c2.activate()
        m.x.setlb(0)
        m.x.setub(2)
        m.y.setlb(None)
        m.y.setub(None)
        it.perform_fbbt(m)
        self.assertTrue(m.c2.active)
        self.assertAlmostEqual(m.y.lb, 0)
        self.assertAlmostEqual(m.y.ub, 2)

    def test_named_exprs(self):
        m = pe.ConcreteModel()
        m.a = pe.Set(initialize=[1, 2, 3])
        m.x = pe.Var(m.a, bounds=(0, None))
        m.e = pe.Expression(m.a)
        for i in m.a:
            m.e[i].expr = i * m.x[i]
        m.c = pe.Constraint(expr=sum(m.e.values()) == 0)
        it = appsi.fbbt.IntervalTightener()
        it.perform_fbbt(m)
        for x in m.x.values():
            self.assertAlmostEqual(x.lb, 0)
            self.assertAlmostEqual(x.ub, 0)

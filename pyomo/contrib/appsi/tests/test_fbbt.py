from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.expr.numeric_expr import ProductExpression, UnaryFunctionExpression, LinearExpression
import math
from pyomo.contrib import appsi
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData
from typing import Sequence
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestFBBT(unittest.TestCase):
    def compare(self, first, second):
        if first is None or second is None:
            self.assertEqual(first, second)
        else:
            self.assertAlmostEqual(first, second)

    def run_fbbt_and_compare(self, m: _BlockData, vars_to_check: Sequence[_GeneralVarData]):
        m2 = m.clone()
        fbbt(m)
        it = appsi.fbbt.IntervalTightener()
        it.perform_fbbt(m2)
        for v in vars_to_check:
            v2 = m2.find_component(v)
            self.compare(v.lb, v2.lb)
            self.compare(v.ub, v2.ub)

    def run_fbbt_with_infeasible_constraint(self, m):
        m2 = m.clone()
        it = appsi.fbbt.IntervalTightener()
        with self.assertRaises(InfeasibleConstraintException):
            fbbt(m)
        with self.assertRaises(InfeasibleConstraintException):
            it.perform_fbbt(m2)

    def test_add(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.p = pyo.Param(mutable=True)
                m.p.value = 1
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.x+m.y+(m.p+1), lower=cl, upper=cu))
                self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_sub1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.x-m.y, lower=cl, upper=cu))
                self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_sub2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.y-m.x, lower=cl, upper=cu))
                self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_mul(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.x*m.y, lower=cl, upper=cu))
                self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_div1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.x/m.y, lower=cl, upper=cu))
                self.run_fbbt_and_compare(m, [m.y, m.y])

    def test_div2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.y/m.x, lower=cl, upper=cu))
                self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_pow1(self):
        x_bounds = [(0, 2.8), (0.5, 2.8), (1, 2.8), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (-2.5, 0), (0, 2.8), (1, 2.8), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.x**m.y, lower=cl, upper=cu))
                if xl > 0 and cu <= 0:
                    self.run_fbbt_with_infeasible_constraint(m)
                else:
                    self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_pow2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.y**m.x, lower=cl, upper=cu))
                self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_x_sq(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**2 == m.y)

        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(1)
        m.y.setub(4)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(0)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(-0.5)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(-1)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(0)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(-1)
        self.run_fbbt_with_infeasible_constraint(m)

        m.y.setub(0)
        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_pow5(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var(bounds=(0.5, 1))
        m.c = pyo.Constraint(expr=2**m.x == m.y)

        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_x_pow_minus_2(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**(-2) == m.y)

        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.y.setlb(-5)
        m.y.setub(-1)
        self.run_fbbt_with_infeasible_constraint(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(0)
        self.run_fbbt_with_infeasible_constraint(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(1)
        m.y.setlb(0.25)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(0)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(0)
        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_x_cubed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**3 == m.y)

        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(1)
        m.y.setub(8)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-8)
        m.y.setub(8)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(8)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-8)
        m.y.setub(-1)
        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_x_pow_minus_3(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**(-3) == m.y)

        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.y.setlb(-1)
        m.y.setub(-0.125)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(0)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-1)
        m.y.setub(1)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.y.setlb(0.125)
        m.y.setub(1)
        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_pow4(self):
        y_bounds = [(0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1), (0, 0.5)]
        exp_vals = [-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]
        for yl, yu in y_bounds:
            for _exp_val in exp_vals:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var()
                m.y = pyo.Var(bounds=(yl, yu))
                m.c = pyo.Constraint(expr=m.x**_exp_val == m.y)
                self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_sqrt(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.sqrt(m.x) == m.y)

        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(-1)
        self.run_fbbt_with_infeasible_constraint(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(0)
        m.y.setlb(None)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(2)
        m.y.setlb(1)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m.x.setlb(None)
        m.x.setub(0)
        m.y.setlb(None)
        m.y.setub(None)
        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_exp(self):
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var()
            m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.exp(m.x), lower=cl, upper=cu))
            self.run_fbbt_and_compare(m, [m.x])

    def test_log(self):
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var()
            m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.log(m.x), lower=cl, upper=cu))
            self.run_fbbt_and_compare(m, [m.x])

    def test_log10(self):
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var()
            m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.log10(m.x), lower=cl, upper=cu))
            self.run_fbbt_and_compare(m, [m.x])

    def test_sin(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(-math.pi/2, math.pi/2))
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.sin(m.x), lower=-0.5, upper=0.5))
        self.run_fbbt_and_compare(m, [m.x])

        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.sin(m.x), lower=-0.5, upper=0.5))
        self.run_fbbt_and_compare(m, [m.x])

    def test_cos(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(0, math.pi))
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.cos(m.x), lower=-0.5, upper=0.5))
        self.run_fbbt_and_compare(m, [m.x])

        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.cos(m.x), lower=-0.5, upper=0.5))
        self.run_fbbt_and_compare(m, [m.x])

    def test_tan(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(-math.pi/2, math.pi/2))
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.tan(m.x), lower=-0.5, upper=0.5))
        self.run_fbbt_and_compare(m, [m.x])

        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.tan(m.x), lower=-0.5, upper=0.5))
        self.run_fbbt_and_compare(m, [m.x])

    def test_asin(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.asin(m.x), lower=-0.5, upper=0.5))
        self.run_fbbt_and_compare(m, [m.x])

    def test_acos(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.acos(m.x), lower=1, upper=2))
        self.run_fbbt_and_compare(m, [m.x])

    def test_atan(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.atan(m.x), lower=-0.5, upper=0.5))
        self.run_fbbt_and_compare(m, [m.x])

    def test_multiple_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-3, 3))
        m.y = pyo.Var(bounds=(0, None))
        m.z = pyo.Var()
        m.c = pyo.ConstraintList()
        m.c.add(m.x + m.y >= -1)
        m.c.add(m.x + m.y <= -1)
        m.c.add(m.y - m.x*m.z <= 2)
        m.c.add(m.y - m.x*m.z >= -2)
        m.c.add(m.x + m.z == 1)
        self.run_fbbt_and_compare(m, [m.x, m.y, m.z])

    def test_multiple_constraints2(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-3, 3))
        m.y = pyo.Var(bounds=(None, 0))
        m.z = pyo.Var()
        m.c = pyo.ConstraintList()
        m.c.add(-m.x - m.y >= -1)
        m.c.add(-m.x - m.y <= -1)
        m.c.add(-m.y - m.x*m.z >= -2)
        m.c.add(-m.y - m.x*m.z <= 2)
        m.c.add(-m.x - m.z == 1)
        self.run_fbbt_and_compare(m, [m.x, m.y, m.z])

    def test_binary(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Binary)
        m.y = pyo.Var(domain=pyo.Binary)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1.5)
        self.run_fbbt_and_compare(m, [m.x, m.y])

        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Binary)
        m.y = pyo.Var(domain=pyo.Binary)
        m.c = pyo.Constraint(expr=m.x + m.y <= 0.5)
        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_always_feasible(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1,2))
        m.y = pyo.Var(bounds=(1,2))
        m.c = pyo.Constraint(expr=m.x + m.y >= 0)
        m2 = m.clone()
        it = appsi.fbbt.IntervalTightener()
        fbbt(m)
        it.perform_fbbt(m2)
        self.assertTrue(m.c.active)
        self.assertTrue(m2.c.active)
        fbbt(m, deactivate_satisfied_constraints=True)
        it.config.deactivate_satisfied_constraints = True
        it.perform_fbbt(m2)
        self.assertFalse(m.c.active)
        self.assertFalse(m2.c.active)

    def test_iteration_limit(self):
        m = pyo.ConcreteModel()
        m.x_set = pyo.Set(initialize=[0, 1, 2], ordered=True)
        m.c_set = pyo.Set(initialize=[0, 1], ordered=True)
        m.x = pyo.Var(m.x_set)
        m.c = pyo.Constraint(m.c_set)
        m.c[0] = m.x[0] == m.x[1]
        m.c[1] = m.x[1] == m.x[2]
        m.x[2].setlb(-1)
        m.x[2].setub(1)
        m2 = m.clone()
        fbbt(m, max_iter=1)
        it = appsi.fbbt.IntervalTightener()
        it.config.max_iter = 1
        it.perform_fbbt(m2)
        self.compare(m.x[0].lb, m2.x[0].lb)
        self.compare(m.x[0].ub, m2.x[0].ub)
        self.compare(m.x[1].lb, m2.x[1].lb)
        self.compare(m.x[1].ub, m2.x[1].ub)
        self.compare(m.x[2].lb, m2.x[2].lb)
        self.compare(m.x[2].ub, m2.x[2].ub)

    def test_inf_bounds(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Constraint(expr=m.x + m.y == m.z)
        self.run_fbbt_and_compare(m, [m.x, m.y, m.z])

    def test_encountered_bugs1(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(-0.035, -0.035))
        m.y = pyo.Var(bounds=(-0.023, -0.023))
        m.c = pyo.Constraint(expr=m.x**2 + m.y**2 <= 0.0256)
        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_encountered_bugs2(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(within=pyo.Integers)
        m.y = pyo.Var(within=pyo.Integers)
        m.c = pyo.Constraint(expr=m.x + m.y == 1)
        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_encountered_bugs3(self):
        xl = 0.033689710575092756
        xu = 0.04008169994804723
        yl = 0.03369608678342047
        yu = 0.04009243987444148

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(xl, xu))
        m.y = pyo.Var(bounds=(yl, yu))

        m.c = pyo.Constraint(expr=m.x == pyo.sin(m.y))

        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_negative_power(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Constraint(expr=(m.x**2 + m.y**2)**(-0.5) == m.z)
        self.run_fbbt_and_compare(m, [m.x, m.y, m.z])
        self.assertAlmostEqual(m.z.lb, 0)
        self.assertEqual(m.z.ub, None)

    def test_linear_expression(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1, 2))
        m.y = pyo.Var()
        m.p = pyo.Param(initialize=3, mutable=True)
        e = LinearExpression(constant=1, linear_coefs=[1, m.p - 1], linear_vars=[m.x, m.y])
        m.c = pyo.Constraint(expr=e == 0)
        self.run_fbbt_and_compare(m, [m.x, m.y])

    def test_long_sum(self):
        N = 30
        for n in range(N):
            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setub(None)
            m.c = pyo.Constraint(expr=sum(m.x.values()) == 1)
            it = appsi.fbbt.IntervalTightener()
            it.perform_fbbt(m)
            self.assertAlmostEqual(m.x[n].ub, 1)

            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setlb(None)
            m.c = pyo.Constraint(expr=sum(m.x.values()) == 1)
            it = appsi.fbbt.IntervalTightener()
            it.perform_fbbt(m)
            self.assertAlmostEqual(m.x[n].lb, -28)

    def test_long_linear_expression(self):
        N = 30
        for n in range(N):
            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setub(None)
            m.c = pyo.Constraint(expr=LinearExpression(constant=0, linear_coefs=[1]*N, linear_vars=list(m.x.values())) == 1)
            it = appsi.fbbt.IntervalTightener()
            it.perform_fbbt(m)
            self.assertAlmostEqual(m.x[n].ub, 1)

            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setlb(None)
            m.c = pyo.Constraint(expr=LinearExpression(constant=0, linear_coefs=[1]*N, linear_vars=list(m.x.values())) == 1)
            it = appsi.fbbt.IntervalTightener()
            it.perform_fbbt(m)
            self.assertAlmostEqual(m.x[n].lb, -28)

    def test_long_linear_expression2(self):
        N = 30
        for n in range(N):
            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setlb(None)
            m.x[n].setub(None)
            m.c = pyo.Constraint(expr=LinearExpression(constant=1, linear_coefs=[1]*N, linear_vars=list(m.x.values())) == 1)
            it = appsi.fbbt.IntervalTightener()
            it.perform_fbbt(m)
            self.assertAlmostEqual(m.x[n].lb, -29)
            self.assertAlmostEqual(m.x[n].ub, 0)

    def test_quadratic_as_product(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], bounds=(-2, 6))
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c1 = pyo.Constraint(expr=m.x[1]*m.x[1] + m.x[2]*m.x[2] == m.y)
        m.c2 = pyo.Constraint(expr=m.x[1]**2 + m.x[2]**2 == m.z)
        self.run_fbbt_and_compare(m, [m.x[1], m.x[2], m.y, m.z])

        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], bounds=(-2, 6))
        m.c = pyo.Constraint(expr=m.x[1]*m.x[1] + m.x[2]*m.x[2] == 0)
        self.run_fbbt_and_compare(m, [m.x[1], m.x[2]])

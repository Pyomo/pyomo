#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.expr.numeric_expr import (ProductExpression,
                                          UnaryFunctionExpression)
import math
from six import StringIO


class DummyExpr(ProductExpression):
    pass


class TestFBBT(unittest.TestCase):
    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
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
                new_bounds = fbbt(m)
                self.assertEqual(new_bounds[m.x], (pyo.value(m.x.lb), pyo.value(m.x.ub)))
                self.assertEqual(new_bounds[m.y], (pyo.value(m.y.lb), pyo.value(m.y.ub)))
                x = np.linspace(pyo.value(m.x.lb), pyo.value(m.x.ub), 100)
                z = np.linspace(pyo.value(m.c.lower), pyo.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = z - _x - m.p.value - 1
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_sub1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.x-m.y, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pyo.value(m.x.lb), pyo.value(m.x.ub), 100)
                z = np.linspace(pyo.value(m.c.lower), pyo.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = _x - z
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_sub2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.y-m.x, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pyo.value(m.x.lb), pyo.value(m.x.ub), 100)
                z = np.linspace(pyo.value(m.c.lower), pyo.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = z + _x
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_mul(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.x*m.y, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pyo.value(m.x.lb) + 1e-6, pyo.value(m.x.ub), 100, endpoint=False)
                z = np.linspace(pyo.value(m.c.lower), pyo.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = z / _x
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_div1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.x/m.y, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pyo.value(m.x.lb), pyo.value(m.x.ub), 100)
                z = np.linspace(pyo.value(m.c.lower) + 1e-6, pyo.value(m.c.upper), 100, endpoint=False)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = _x / z
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_div2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.y/m.x, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pyo.value(m.x.lb), pyo.value(m.x.ub), 100)
                z = np.linspace(pyo.value(m.c.lower), pyo.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = _x * z
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
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
                    with self.assertRaises(InfeasibleConstraintException):
                        fbbt(m)
                else:
                    fbbt(m)
                    x = np.linspace(pyo.value(m.x.lb) + 1e-6, pyo.value(m.x.ub), 100, endpoint=False)
                    z = np.linspace(pyo.value(m.c.lower) + 1e-6, pyo.value(m.c.upper), 100, endpoint=False)
                    if m.y.lb is None:
                        yl = -np.inf
                    else:
                        yl = m.y.lb
                    if m.y.ub is None:
                        yu = np.inf
                    else:
                        yu = m.y.ub
                    for _x in x:
                        _y = np.log(abs(z)) / np.log(abs(_x))
                        self.assertTrue(np.all(yl <= _y))
                        self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_pow2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(expr=pyo.inequality(body=m.y**m.x, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pyo.value(m.x.lb) + 1e-6, pyo.value(m.x.ub), 100, endpoint=False)
                z = np.linspace(pyo.value(m.c.lower) + 1e-6, pyo.value(m.c.upper), 100, endpoint=False)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = np.exp(np.log(abs(z)) / _x)
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    def test_x_sq(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**2 == m.y)

        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, None)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(1)
        m.y.setub(4)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(0)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(-0.5)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(-1)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(0)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, -1)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(-1)
        with self.assertRaises(InfeasibleConstraintException):
            fbbt(m)

        m.y.setub(0)
        fbbt(m)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 0)

    def test_pow5(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var(bounds=(0.5, 1))
        m.c = pyo.Constraint(expr=2**m.x == m.y)

        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -1)
        self.assertAlmostEqual(m.x.ub, 0)

    def test_x_pow_minus_2(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**(-2) == m.y)

        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, None)

        m.y.setlb(-5)
        m.y.setub(-1)
        with self.assertRaises(InfeasibleConstraintException):
            fbbt(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(0)
        with self.assertRaises(InfeasibleConstraintException):
            fbbt(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(1)
        m.y.setlb(0.25)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(0)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(0)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, -1)

    def test_x_cubed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**3 == m.y)

        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, None)
        self.assertEqual(m.y.ub, None)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(1)
        m.y.setub(8)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-8)
        m.y.setub(8)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(8)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -5.0**(1.0/3.0))
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-8)
        m.y.setub(-1)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, -1)

    def test_x_pow_minus_3(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**(-3) == m.y)

        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, None)
        self.assertEqual(m.y.ub, None)

        m.y.setlb(-1)
        m.y.setub(-0.125)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, -1)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(0)
        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertAlmostEqual(m.x.ub, -1)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-1)
        m.y.setub(1)
        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

        m.y.setlb(0.125)
        m.y.setub(1)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_pow4(self):
        y_bounds = [(0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1), (0, 0.5)]
        exp_vals = [-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]
        for yl, yu in y_bounds:
            for _exp_val in exp_vals:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var()
                m.y = pyo.Var(bounds=(yl, yu))
                m.c = pyo.Constraint(expr=m.x**_exp_val == m.y)
                fbbt(m)
                y = np.linspace(pyo.value(m.y.lb) + 1e-6, pyo.value(m.y.ub), 100, endpoint=True)
                if m.x.lb is None:
                    xl = -np.inf
                else:
                    xl = m.x.lb
                if m.x.ub is None:
                    xu = np.inf
                else:
                    xu = m.x.ub
                _x = np.exp(np.log(y) / _exp_val)
                self.assertTrue(np.all(xl <= _x))
                self.assertTrue(np.all(xu >= _x))

    def test_sqrt(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.sqrt(m.x) == m.y)

        fbbt(m)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, None)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(-1)
        with self.assertRaises(InfeasibleConstraintException):
            fbbt(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(0)
        m.y.setlb(None)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, 0)
        self.assertAlmostEqual(m.x.ub, 0)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(2)
        m.y.setlb(1)
        fbbt(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 4)

        m.x.setlb(None)
        m.x.setub(0)
        m.y.setlb(None)
        m.y.setub(None)
        fbbt(m)
        self.assertAlmostEqual(m.y.lb, 0)
        self.assertAlmostEqual(m.y.ub, 0)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_exp(self):
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var()
            m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.exp(m.x), lower=cl, upper=cu))
            fbbt(m)
            if pyo.value(m.c.lower) <= 0:
                _cl = 1e-6
            else:
                _cl = pyo.value(m.c.lower)
            z = np.linspace(_cl, pyo.value(m.c.upper), 100)
            if m.x.lb is None:
                xl = -np.inf
            else:
                xl = pyo.value(m.x.lb)
            if m.x.ub is None:
                xu = np.inf
            else:
                xu = pyo.value(m.x.ub)
            x = np.log(z)
            self.assertTrue(np.all(xl <= x))
            self.assertTrue(np.all(xu >= x))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_log(self):
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var()
            m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.log(m.x), lower=cl, upper=cu))
            fbbt(m)
            z = np.linspace(pyo.value(m.c.lower), pyo.value(m.c.upper), 100)
            if m.x.lb is None:
                xl = -np.inf
            else:
                xl = pyo.value(m.x.lb)
            if m.x.ub is None:
                xu = np.inf
            else:
                xu = pyo.value(m.x.ub)
            x = np.exp(z)
            self.assertTrue(np.all(xl <= x))
            self.assertTrue(np.all(xu >= x))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_log10(self):
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var()
            m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.log10(m.x), lower=cl, upper=cu))
            fbbt(m)
            z = np.linspace(pyo.value(m.c.lower), pyo.value(m.c.upper), 100)
            if m.x.lb is None:
                xl = -np.inf
            else:
                xl = pyo.value(m.x.lb)
            if m.x.ub is None:
                xu = np.inf
            else:
                xu = pyo.value(m.x.ub)
            x = 10**z
            print(xl, xu, cl, cu)
            print(x)
            self.assertTrue(np.all(xl <= x))
            self.assertTrue(np.all(xu >= x))

    def test_sin(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(-math.pi/2, math.pi/2))
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.sin(m.x), lower=-0.5, upper=0.5))
        fbbt(m.c)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.asin(-0.5))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.asin(0.5))

        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.sin(m.x), lower=-0.5, upper=0.5))
        fbbt(m.c)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

    def test_cos(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(0, math.pi))
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.cos(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.acos(0.5))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.acos(-0.5))

        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.cos(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

    def test_tan(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(-math.pi/2, math.pi/2))
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.tan(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.atan(-0.5))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.atan(0.5))

        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.tan(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

    def test_asin(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.asin(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.sin(-0.5))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.sin(0.5))

    def test_acos(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.acos(m.x), lower=1, upper=2))
        fbbt(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.cos(2))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.cos(1))

    def test_atan(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.atan(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.tan(-0.5))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.tan(0.5))

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
        fbbt(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), -1, 8)
        self.assertAlmostEqual(pyo.value(m.x.ub), -1, 8)
        self.assertAlmostEqual(pyo.value(m.y.lb), 0, 8)
        self.assertAlmostEqual(pyo.value(m.y.ub), 0, 8)
        self.assertAlmostEqual(pyo.value(m.z.lb), 2, 8)
        self.assertAlmostEqual(pyo.value(m.z.ub), 2, 8)

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
        fbbt(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), 1, 8)
        self.assertAlmostEqual(pyo.value(m.x.ub), 1, 8)
        self.assertAlmostEqual(pyo.value(m.y.lb), 0, 8)
        self.assertAlmostEqual(pyo.value(m.y.ub), 0, 8)
        self.assertAlmostEqual(pyo.value(m.z.lb), -2, 8)
        self.assertAlmostEqual(pyo.value(m.z.ub), -2, 8)

    def test_binary(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Binary)
        m.y = pyo.Var(domain=pyo.Binary)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1.5)
        fbbt(m)
        self.assertEqual(pyo.value(m.x.lb), 1)
        self.assertEqual(pyo.value(m.x.ub), 1)
        self.assertEqual(pyo.value(m.y.lb), 1)
        self.assertEqual(pyo.value(m.y.ub), 1)

        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Binary)
        m.y = pyo.Var(domain=pyo.Binary)
        m.c = pyo.Constraint(expr=m.x + m.y <= 0.5)
        fbbt(m)
        self.assertEqual(pyo.value(m.x.lb), 0)
        self.assertEqual(pyo.value(m.x.ub), 0)
        self.assertEqual(pyo.value(m.y.lb), 0)
        self.assertEqual(pyo.value(m.y.ub), 0)

    def test_always_feasible(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1,2))
        m.y = pyo.Var(bounds=(1,2))
        m.c = pyo.Constraint(expr=m.x + m.y >= 0)
        fbbt(m)
        self.assertTrue(m.c.active)
        fbbt(m, deactivate_satisfied_constraints=True)
        self.assertFalse(m.c.active)

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
        fbbt(m, max_iter=1)
        self.assertEqual(m.x[1].lb, -1)
        self.assertEqual(m.x[1].ub, 1)
        self.assertEqual(m.x[0].lb, None)
        self.assertEqual(m.x[0].ub, None)

    def test_inf_bounds_on_expr(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.y = pyo.Var()
        lb, ub = compute_bounds_on_expr(m.x + m.y)
        self.assertEqual(lb, None)
        self.assertEqual(ub, None)

    def test_skip_unknown_expression1(self):

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1,1))
        m.y = pyo.Var()
        expr = DummyExpr([m.x, m.y])
        m.c = pyo.Constraint(expr=expr == 1)

        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo.contrib.fbbt.fbbt'):
            new_bounds = fbbt(m)

        self.assertEqual(pyo.value(m.x.lb), 1)
        self.assertEqual(pyo.value(m.x.ub), 1)
        self.assertEqual(pyo.value(m.y.lb), None)
        self.assertEqual(pyo.value(m.y.ub), None)
        self.assertIn("Unsupported expression type for FBBT", OUT.getvalue())

    def test_skip_unknown_expression2(self):
        def dummy_unary_expr(x):
            return 0.5*x

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0,4))
        expr = UnaryFunctionExpression((m.x,), name='dummy_unary_expr', fcn=dummy_unary_expr)
        m.c = pyo.Constraint(expr=expr == 1)

        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo.contrib.fbbt.fbbt'):
            new_bounds = fbbt(m)

        self.assertEqual(pyo.value(m.x.lb), 0)
        self.assertEqual(pyo.value(m.x.ub), 4)
        self.assertIn("Unsupported expression type for FBBT", OUT.getvalue())

    def test_compute_expr_bounds(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1,1))
        m.y = pyo.Var(bounds=(-1,1))
        e = m.x + m.y
        lb, ub = compute_bounds_on_expr(e)
        self.assertAlmostEqual(lb, -2, 14)
        self.assertAlmostEqual(ub, 2, 14)

    def test_encountered_bugs1(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(-0.035, -0.035))
        m.y = pyo.Var(bounds=(-0.023, -0.023))
        m.c = pyo.Constraint(expr=m.x**2 + m.y**2 <= 0.0256)
        fbbt(m.c)
        self.assertEqual(m.x.lb, -0.035)
        self.assertEqual(m.x.ub, -0.035)
        self.assertEqual(m.y.lb, -0.023)
        self.assertEqual(m.y.ub, -0.023)

    def test_encountered_bugs2(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(within=pyo.Integers)
        m.y = pyo.Var(within=pyo.Integers)
        m.c = pyo.Constraint(expr=m.x + m.y == 1)
        fbbt(m.c)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, None)
        self.assertEqual(m.y.ub, None)

    def test_encountered_bugs3(self):
        xl = 0.033689710575092756
        xu = 0.04008169994804723
        yl = 0.03369608678342047
        yu = 0.04009243987444148

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(xl, xu))
        m.y = pyo.Var(bounds=(yl, yu))

        m.c = pyo.Constraint(expr=m.x == pyo.sin(m.y))

        fbbt(m.c)

        self.assertAlmostEqual(m.x.lb, xl)
        self.assertAlmostEqual(m.x.ub, xu)
        self.assertAlmostEqual(m.y.lb, yl)
        self.assertAlmostEqual(m.y.ub, yu)

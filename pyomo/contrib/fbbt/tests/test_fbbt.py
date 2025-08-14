#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr, _before_child_handlers
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.fileutils import find_library
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.expr.numeric_expr import (
    ProductExpression,
    UnaryFunctionExpression,
    LinearExpression,
)
import math
import platform
from io import StringIO

flib = find_library("asl_external_demo")
is_pypy = platform.python_implementation().lower().startswith('pypy')


class DummyExpr(ProductExpression):
    pass


class FbbtTestBase(object):
    """
    These tests are set up weird, but it is for a good reason.
    The FBBT code is duplicated in pyomo.contrib.appsi for
    improved performance. We want to keep this version because
    it does not require building an extension. However, when we
    fix a bug in one module, we want to ensure we fix that bug
    in the other module. Therefore, we use this base class
    for testing both modules. The only difference in the
    derived classes is self.tightener attribute.
    """

    def test_add(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        x_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        c_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.p = pyo.Param(mutable=True)
                m.p.value = 1
                m.c = pyo.Constraint(
                    expr=pyo.inequality(body=m.x + m.y + (m.p + 1), lower=cl, upper=cu)
                )
                self.tightener(m)
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

    def test_sub1(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        x_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        c_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(
                    expr=pyo.inequality(body=m.x - m.y, lower=cl, upper=cu)
                )
                self.tightener(m)
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

    def test_sub2(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        x_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        c_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(
                    expr=pyo.inequality(body=m.y - m.x, lower=cl, upper=cu)
                )
                self.tightener(m)
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

    def test_mul(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        x_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        c_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(
                    expr=pyo.inequality(body=m.x * m.y, lower=cl, upper=cu)
                )
                self.tightener(m)
                x = np.linspace(
                    pyo.value(m.x.lb) + 1e-6, pyo.value(m.x.ub), 100, endpoint=False
                )
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

    def test_div1(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        x_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        c_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(
                    expr=pyo.inequality(body=m.x / m.y, lower=cl, upper=cu)
                )
                self.tightener(m)
                x = np.linspace(pyo.value(m.x.lb), pyo.value(m.x.ub), 100)
                z = np.linspace(
                    pyo.value(m.c.lower) + 1e-6,
                    pyo.value(m.c.upper),
                    100,
                    endpoint=False,
                )
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

    def test_div2(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        x_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        c_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(
                    expr=pyo.inequality(body=m.y / m.x, lower=cl, upper=cu)
                )
                self.tightener(m)
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

    def test_pow1(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        x_bounds = [(0, 2.8), (0.5, 2.8), (1, 2.8), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (-2.5, 0), (0, 2.8), (1, 2.8), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(
                    expr=pyo.inequality(body=m.x**m.y, lower=cl, upper=cu)
                )
                if xl > 0 and cu <= 0:
                    with self.assertRaises(InfeasibleConstraintException):
                        self.tightener(m)
                else:
                    self.tightener(m)
                    x = np.linspace(
                        pyo.value(m.x.lb) + 1e-6, pyo.value(m.x.ub), 100, endpoint=False
                    )
                    z = np.linspace(
                        pyo.value(m.c.lower) + 1e-6,
                        pyo.value(m.c.upper),
                        100,
                        endpoint=False,
                    )
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

    def test_pow2(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        x_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var(bounds=(xl, xu))
                m.y = pyo.Var()
                m.c = pyo.Constraint(
                    expr=pyo.inequality(body=m.y**m.x, lower=cl, upper=cu)
                )
                self.tightener(m)
                x = np.linspace(
                    pyo.value(m.x.lb) + 1e-6, pyo.value(m.x.ub), 100, endpoint=False
                )
                z = np.linspace(
                    pyo.value(m.c.lower) + 1e-6,
                    pyo.value(m.c.upper),
                    100,
                    endpoint=False,
                )
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                y = np.exp(np.split(np.log(np.abs(z)), len(z)) / x)
                self.assertTrue(np.all(yl <= y))
                self.assertTrue(np.all(yu >= y))

    def test_x_sq(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**2 == m.y)

        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, None)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(1)
        m.y.setub(4)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(0)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(-0.5)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(-1)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(0)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, -1)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(-1)
        with self.assertRaises(InfeasibleConstraintException):
            self.tightener(m)

        m.y.setub(0)
        self.tightener(m)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, 0)

    def test_pow5(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var(bounds=(0.5, 1))
        m.c = pyo.Constraint(expr=2**m.x == m.y)

        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -1)
        self.assertAlmostEqual(m.x.ub, 0)

    def test_x_pow_minus_2(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x ** (-2) == m.y)

        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, None)

        m.y.setlb(-5)
        m.y.setub(-1)
        with self.assertRaises(InfeasibleConstraintException):
            self.tightener(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(0)
        with self.assertRaises(InfeasibleConstraintException):
            self.tightener(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(1)
        m.y.setlb(0.25)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(0)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(0)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, -1)

    def test_x_cubed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x**3 == m.y)

        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, None)
        self.assertEqual(m.y.ub, None)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(1)
        m.y.setub(8)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-8)
        m.y.setub(8)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(8)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -(5.0 ** (1.0 / 3.0)))
        self.assertAlmostEqual(m.x.ub, 2)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-8)
        m.y.setub(-1)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, -1)

    def test_x_pow_minus_3(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x ** (-3) == m.y)

        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, None)
        self.assertEqual(m.y.ub, None)

        m.y.setlb(-1)
        m.y.setub(-0.125)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -2)
        self.assertAlmostEqual(m.x.ub, -1)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(0)
        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertAlmostEqual(m.x.ub, -1)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-1)
        m.y.setub(1)
        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

        m.y.setlb(0.125)
        m.y.setub(1)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 2)

    def test_pow4(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        y_bounds = [(0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1), (0, 0.5)]
        exp_vals = [-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]
        for yl, yu in y_bounds:
            for _exp_val in exp_vals:
                m = pyo.Block(concrete=True)
                m.x = pyo.Var()
                m.y = pyo.Var(bounds=(yl, yu))
                m.c = pyo.Constraint(expr=m.x**_exp_val == m.y)
                self.tightener(m)
                y = np.linspace(
                    pyo.value(m.y.lb) + 1e-6, pyo.value(m.y.ub), 100, endpoint=True
                )
                if m.x.lb is None:
                    xl = -np.inf
                else:
                    xl = m.x.lb
                if m.x.ub is None:
                    xu = np.inf
                else:
                    xu = m.x.ub
                _x = np.exp(np.log(y) / _exp_val)
                self.assertTrue(np.all(xl - 1e-14 <= _x))
                self.assertTrue(np.all(xu + 1e-14 >= _x))

    def test_sqrt(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.sqrt(m.x) == m.y)

        self.tightener(m)
        self.assertEqual(m.x.lb, 0)
        self.assertEqual(m.x.ub, None)
        self.assertEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, None)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(-1)
        with self.assertRaises(InfeasibleConstraintException):
            self.tightener(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(0)
        m.y.setlb(None)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 0)
        self.assertAlmostEqual(m.x.ub, 0)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setub(2)
        m.y.setlb(1)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 4)

        m.x.setlb(None)
        m.x.setub(0)
        m.y.setlb(None)
        m.y.setub(None)
        self.tightener(m)
        self.assertAlmostEqual(m.y.lb, 0)
        self.assertAlmostEqual(m.y.ub, 0)

    def test_exp(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var()
            m.c = pyo.Constraint(
                expr=pyo.inequality(body=pyo.exp(m.x), lower=cl, upper=cu)
            )
            self.tightener(m)
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

    def test_abs(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=abs(m.x) == m.y)

        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)
        self.assertAlmostEqual(m.y.lb, 0)
        self.assertEqual(m.y.ub, None)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(2)
        m.y.setub(4)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -4)
        self.assertAlmostEqual(m.x.ub, 4)
        self.assertAlmostEqual(m.y.lb, 2)
        self.assertAlmostEqual(m.y.ub, 4)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(4)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -4)
        self.assertAlmostEqual(m.x.ub, 4)
        self.assertAlmostEqual(m.y.lb, 0)
        self.assertAlmostEqual(m.y.ub, 4)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(-4)
        with self.assertRaises(InfeasibleConstraintException):
            self.tightener(m)

        m.x.setlb(None)
        m.x.setub(None)
        m.y.setlb(-5)
        m.y.setub(0)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 0)
        self.assertAlmostEqual(m.x.ub, 0)
        self.assertAlmostEqual(m.y.lb, 0)
        self.assertAlmostEqual(m.y.ub, 0)

        m.x.setlb(-5)
        m.x.setub(5)
        m.y.setlb(None)
        m.y.setub(None)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -5)
        self.assertAlmostEqual(m.x.ub, 5)
        self.assertAlmostEqual(m.y.lb, 0)
        self.assertAlmostEqual(m.y.ub, 5)

        m.x.setlb(-5)
        m.x.setub(-2)
        m.y.setlb(None)
        m.y.setub(None)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, -5)
        self.assertAlmostEqual(m.x.ub, -2)
        self.assertAlmostEqual(m.y.lb, 2)
        self.assertAlmostEqual(m.y.ub, 5)

        m.x.setlb(2)
        m.x.setub(5)
        m.y.setlb(None)
        m.y.setub(None)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 2)
        self.assertAlmostEqual(m.x.ub, 5)
        self.assertAlmostEqual(m.y.lb, 2)
        self.assertAlmostEqual(m.y.ub, 5)

    def test_log(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        c_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var()
            m.c = pyo.Constraint(
                expr=pyo.inequality(body=pyo.log(m.x), lower=cl, upper=cu)
            )
            self.tightener(m)
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
            self.assertTrue(np.all(xl - 1e-14 <= x))
            self.assertTrue(np.all(xu + 1e-14 >= x))

    def test_log10(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available')
        c_bounds = [
            (-2.5, 2.8),
            (-2.5, -0.5),
            (0.5, 2.8),
            (-2.5, 0),
            (0, 2.8),
            (-2.5, -1),
            (1, 2.8),
            (-1, -0.5),
            (0.5, 1),
        ]
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var()
            m.c = pyo.Constraint(
                expr=pyo.inequality(body=pyo.log10(m.x), lower=cl, upper=cu)
            )
            self.tightener(m)
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
            self.assertTrue(np.all(xl - 1e-14 <= x))
            self.assertTrue(np.all(xu + 1e-14 >= x))

    def test_sin(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(-math.pi / 2, math.pi / 2))
        m.c = pyo.Constraint(
            expr=pyo.inequality(body=pyo.sin(m.x), lower=-0.5, upper=0.5)
        )
        self.tightener(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.asin(-0.5))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.asin(0.5))

        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(
            expr=pyo.inequality(body=pyo.sin(m.x), lower=-0.5, upper=0.5)
        )
        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

    def test_cos(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(0, math.pi))
        m.c = pyo.Constraint(
            expr=pyo.inequality(body=pyo.cos(m.x), lower=-0.5, upper=0.5)
        )
        self.tightener(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.acos(0.5))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.acos(-0.5))

        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(
            expr=pyo.inequality(body=pyo.cos(m.x), lower=-0.5, upper=0.5)
        )
        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

    def test_tan(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(-math.pi / 2, math.pi / 2))
        m.c = pyo.Constraint(
            expr=pyo.inequality(body=pyo.tan(m.x), lower=-0.5, upper=0.5)
        )
        self.tightener(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.atan(-0.5))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.atan(0.5))

        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(
            expr=pyo.inequality(body=pyo.tan(m.x), lower=-0.5, upper=0.5)
        )
        self.tightener(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

    def test_asin(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(
            expr=pyo.inequality(body=pyo.asin(m.x), lower=-0.5, upper=0.5)
        )
        self.tightener(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.sin(-0.5))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.sin(0.5))

    def test_acos(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=pyo.inequality(body=pyo.acos(m.x), lower=1, upper=2))
        self.tightener(m)
        self.assertAlmostEqual(pyo.value(m.x.lb), math.cos(2))
        self.assertAlmostEqual(pyo.value(m.x.ub), math.cos(1))

    def test_atan(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var()
        m.c = pyo.Constraint(
            expr=pyo.inequality(body=pyo.atan(m.x), lower=-0.5, upper=0.5)
        )
        self.tightener(m)
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
        m.c.add(m.y - m.x * m.z <= 2)
        m.c.add(m.y - m.x * m.z >= -2)
        m.c.add(m.x + m.z == 1)
        self.tightener(m)
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
        m.c.add(-m.y - m.x * m.z >= -2)
        m.c.add(-m.y - m.x * m.z <= 2)
        m.c.add(-m.x - m.z == 1)
        self.tightener(m)
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
        self.tightener(m)
        self.assertEqual(pyo.value(m.x.lb), 1)
        self.assertEqual(pyo.value(m.x.ub), 1)
        self.assertEqual(pyo.value(m.y.lb), 1)
        self.assertEqual(pyo.value(m.y.ub), 1)

        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Binary)
        m.y = pyo.Var(domain=pyo.Binary)
        m.c = pyo.Constraint(expr=m.x + m.y <= 0.5)
        self.tightener(m)
        self.assertEqual(pyo.value(m.x.lb), 0)
        self.assertEqual(pyo.value(m.x.ub), 0)
        self.assertEqual(pyo.value(m.y.lb), 0)
        self.assertEqual(pyo.value(m.y.ub), 0)

    def test_always_feasible(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1, 2))
        m.y = pyo.Var(bounds=(1, 2))
        m.c = pyo.Constraint(expr=m.x + m.y >= 0)
        self.tightener(m)
        self.assertTrue(m.c.active)
        if self.tightener is fbbt:
            self.tightener(m, deactivate_satisfied_constraints=True)
        else:
            self.it.config.deactivate_satisfied_constraints = True
            self.tightener(m)
            self.it.config.deactivate_satisfied_constraints = False
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
        if self.tightener is fbbt:
            self.tightener(m, max_iter=1)
        else:
            self.it.config.max_iter = 1
            self.tightener(m)
            self.it.config.max_iter = 10
        self.assertEqual(m.x[1].lb, -1)
        self.assertEqual(m.x[1].ub, 1)
        self.assertEqual(m.x[0].lb, None)
        self.assertEqual(m.x[0].ub, None)

    def test_inf_bounds_on_expr(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Constraint(expr=m.x + m.y == m.z)
        self.tightener(m)
        self.assertEqual(m.z.lb, None)
        self.assertEqual(m.z.ub, None)

    def test_skip_unknown_expression1(self):
        if self.tightener is not fbbt:
            raise unittest.SkipTest(
                'Appsi FBBT does not support unknown expressions yet'
            )

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1, 1))
        m.y = pyo.Var()
        expr = DummyExpr([m.x, m.y])
        m.c = pyo.Constraint(expr=expr == 1)

        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo.contrib.fbbt.fbbt'):
            new_bounds = self.tightener(m)

        self.assertEqual(pyo.value(m.x.lb), 1)
        self.assertEqual(pyo.value(m.x.ub), 1)
        self.assertEqual(pyo.value(m.y.lb), None)
        self.assertEqual(pyo.value(m.y.ub), None)
        self.assertIn("Unsupported expression type for FBBT", OUT.getvalue())

    def test_skip_unknown_expression2(self):
        if self.tightener is not fbbt:
            raise unittest.SkipTest(
                'Appsi FBBT does not support unknown expressions yet'
            )

        def dummy_unary_expr(x):
            return 0.5 * x

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 4))
        expr = UnaryFunctionExpression(
            (m.x,), name='dummy_unary_expr', fcn=dummy_unary_expr
        )
        m.c = pyo.Constraint(expr=expr == 1)

        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo.contrib.fbbt.fbbt'):
            new_bounds = self.tightener(m)

        self.assertEqual(pyo.value(m.x.lb), 0)
        self.assertEqual(pyo.value(m.x.ub), 4)
        self.assertIn("Unsupported expression type for FBBT", OUT.getvalue())

    def test_compute_expr_bounds(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.y = pyo.Var(bounds=(-1, 1))
        e = m.x + m.y
        lb, ub = compute_bounds_on_expr(e)
        self.assertAlmostEqual(lb, -2, 14)
        self.assertAlmostEqual(ub, 2, 14)

    def test_encountered_bugs1(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(bounds=(-0.035, -0.035))
        m.y = pyo.Var(bounds=(-0.023, -0.023))
        m.c = pyo.Constraint(expr=m.x**2 + m.y**2 <= 0.0256)
        self.tightener(m)
        self.assertEqual(m.x.lb, -0.035)
        self.assertEqual(m.x.ub, -0.035)
        self.assertEqual(m.y.lb, -0.023)
        self.assertEqual(m.y.ub, -0.023)

    def test_encountered_bugs2(self):
        m = pyo.Block(concrete=True)
        m.x = pyo.Var(within=pyo.Integers)
        m.y = pyo.Var(within=pyo.Integers)
        m.c = pyo.Constraint(expr=m.x + m.y == 1)
        self.tightener(m)
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

        self.tightener(m)

        self.assertAlmostEqual(m.x.lb, xl)
        self.assertAlmostEqual(m.x.ub, xu)
        self.assertAlmostEqual(m.y.lb, yl)
        self.assertAlmostEqual(m.y.ub, yu)

    def test_negative_power(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.c = pyo.Constraint(expr=(m.x**2 + m.y**2) ** (-0.5) == m.z)
        self.tightener(m)
        self.assertAlmostEqual(m.z.lb, 0)
        self.assertIsNone(m.z.ub)

    def test_linear_expression(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(1, 2))
        m.y = pyo.Var()
        m.p = pyo.Param(initialize=3, mutable=True)
        e = LinearExpression(
            constant=1, linear_coefs=[1, m.p - 1], linear_vars=[m.x, m.y]
        )
        m.c = pyo.Constraint(expr=e == 0)
        self.tightener(m)
        self.assertAlmostEqual(m.y.lb, -1.5)
        self.assertAlmostEqual(m.y.ub, -1)

    def test_quadratic_as_product(self):
        m1 = pyo.ConcreteModel()
        m1.x = pyo.Var([1, 2], bounds=(-2, 6))
        m1.y = pyo.Var()
        m1.c = pyo.Constraint(expr=m1.x[1] * m1.x[1] + m1.x[2] * m1.x[2] == m1.y)

        m2 = pyo.ConcreteModel()
        m2.x = pyo.Var([1, 2], bounds=(-2, 6))
        m2.y = pyo.Var()
        m2.c = pyo.Constraint(expr=m2.x[1] ** 2 + m2.x[2] ** 2 == m2.y)

        self.tightener(m1)
        self.tightener(m2)

        self.assertAlmostEqual(m1.y.lb, m2.y.lb)
        self.assertAlmostEqual(m1.y.ub, m2.y.ub)

        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], bounds=(-2, 6))
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x[1] * m.x[1] + m.x[2] * m.x[2] == 0)
        self.tightener(m)
        self.assertAlmostEqual(m.x[1].lb, 0)
        self.assertAlmostEqual(m.x[1].ub, 0)
        self.assertAlmostEqual(m.x[2].lb, 0)
        self.assertAlmostEqual(m.x[2].ub, 0)

    def test_long_sum(self):
        N = 30
        for n in range(N):
            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setub(None)
            m.c = pyo.Constraint(expr=sum(m.x.values()) == 1)
            self.tightener(m)
            self.assertAlmostEqual(m.x[n].ub, 1)

            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setlb(None)
            m.c = pyo.Constraint(expr=sum(m.x.values()) == 1)
            self.tightener(m)
            self.assertAlmostEqual(m.x[n].lb, -28)

    def test_long_linear_expression(self):
        N = 30
        for n in range(N):
            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setub(None)
            m.c = pyo.Constraint(
                expr=LinearExpression(
                    constant=0, linear_coefs=[1] * N, linear_vars=list(m.x.values())
                )
                == 1
            )
            self.tightener(m)
            self.assertAlmostEqual(m.x[n].ub, 1)

            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setlb(None)
            m.c = pyo.Constraint(
                expr=LinearExpression(
                    constant=0, linear_coefs=[1] * N, linear_vars=list(m.x.values())
                )
                == 1
            )
            self.tightener(m)
            self.assertAlmostEqual(m.x[n].lb, -28)

    def test_long_linear_expression2(self):
        N = 30
        for n in range(N):
            m = pyo.ConcreteModel()
            m.a = pyo.Set(initialize=list(range(N)))
            m.x = pyo.Var(m.a, bounds=(0, 1))
            m.x[n].setlb(None)
            m.x[n].setub(None)
            m.c = pyo.Constraint(
                expr=LinearExpression(
                    constant=1, linear_coefs=[1] * N, linear_vars=list(m.x.values())
                )
                == 1
            )
            self.tightener(m)
            self.assertAlmostEqual(m.x[n].lb, -29)
            self.assertAlmostEqual(m.x[n].ub, 0)

    def test_fixed_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var(bounds=(1, 2))
        m.x.fix(1)
        m.c = pyo.Constraint(expr=m.z == m.x * m.y)

        self.tightener(m)

        self.assertAlmostEqual(m.x.lb, 1)
        self.assertAlmostEqual(m.x.ub, 1)
        self.assertAlmostEqual(m.y.lb, 1)
        self.assertAlmostEqual(m.y.ub, 2)

    @unittest.skipUnless(flib, 'Could not find the "asl_external_demo.so" library')
    @unittest.skipIf(is_pypy, 'Cannot evaluate external functions under pypy')
    def test_external_function(self):
        if self.tightener is not fbbt:
            raise unittest.SkipTest(
                'Appsi FBBT does not support unknown expressions yet'
            )

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.y = pyo.Var(bounds=(0, 5))
        m.p = pyo.Param(initialize=1)
        m.q = pyo.Param(initialize=3)
        m.ef = pyo.ExternalFunction(library=flib, function="demo_function")

        m.con1 = pyo.Constraint(expr=m.ef("sum", m.x, m.y) >= 1)

        # No change due to variable EF
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 0)
        self.assertAlmostEqual(m.x.ub, 1)
        self.assertAlmostEqual(m.y.lb, 0)
        self.assertAlmostEqual(m.y.ub, 5)

        m.con2 = pyo.Constraint(expr=m.ef("sum", m.p, m.q) - m.y >= 1)

        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 0)
        self.assertAlmostEqual(m.x.ub, 1)
        self.assertAlmostEqual(m.y.lb, 0)
        self.assertAlmostEqual(m.y.ub, 3)

    def test_named_expr(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, None))
        m.y = pyo.Var(bounds=(1, 6))
        m.e_const = pyo.Expression(expr=3)
        m.e_var = pyo.Expression(expr=m.y + m.e_const)

        m.c = pyo.Constraint(expr=m.x**2 == m.e_var)

        self.tightener(m)
        self.tightener(m)
        self.assertAlmostEqual(m.x.lb, 2)
        self.assertAlmostEqual(m.x.ub, 3)


class TestFBBT(FbbtTestBase, unittest.TestCase):
    def setUp(self) -> None:
        self.tightener = fbbt

    def test_ranged_expression(self):
        # The python version of FBBT is slightly more flexible than
        # APPSI's cmodel (it allows - and correctly handles -
        # RangedExpressions with variable lower / upper bounds).  If we
        # ever port that functionality into APPSI, then this test can be
        # moved into the base class.
        m = pyo.ConcreteModel()
        m.l = pyo.Var(bounds=(2, None))
        m.x = pyo.Var()
        m.u = pyo.Var(bounds=(None, 8))
        m.c = pyo.Constraint(expr=pyo.inequality(m.l, m.x, m.u))
        self.tightener(m)
        self.tightener(m)
        self.assertEqual(m.l.bounds, (2, 8))
        self.assertEqual(m.x.bounds, (2, 8))
        self.assertEqual(m.u.bounds, (2, 8))

        m = pyo.ConcreteModel()
        m.l = pyo.Var(bounds=(2, None))
        m.x = pyo.Var(bounds=(3, 7))
        m.u = pyo.Var(bounds=(None, 8))
        m.c = pyo.Constraint(expr=pyo.inequality(m.l, m.x, m.u))
        self.tightener(m)
        self.tightener(m)
        self.assertEqual(m.l.bounds, (2, 7))
        self.assertEqual(m.x.bounds, (3, 7))
        self.assertEqual(m.u.bounds, (3, 8))

    @unittest.skipUnless(numpy_available, "Test requires numpy")
    def test_numpy_leaves(self):
        m = pyo.ConcreteModel()
        m.l = pyo.Var(bounds=(2, None))
        m.x = pyo.Var()
        m.u = pyo.Var(bounds=(None, 8))
        m.c = pyo.Constraint(
            expr=pyo.inequality(m.l + np.int32(1), m.x, m.u - np.float64(1))
        )

        # Remove the numpy types so we can test that automatic numeric
        # type registrations
        old = [(t, _before_child_handlers.pop(t, None)) for t in (np.int32, np.float64)]

        try:
            self.tightener(m)
            self.tightener(m)
            self.assertEqual(m.l.bounds, (2, 6.0))
            self.assertEqual(m.x.bounds, (3, 7.0))
            self.assertEqual(m.u.bounds, (4, 8.0))
        finally:
            for t, fcn in old:
                if fcn is None:
                    _before_child_handlers.pop(t, None)
                else:
                    _before_child_handlers[t] = fcn

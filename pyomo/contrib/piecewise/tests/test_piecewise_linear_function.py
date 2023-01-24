#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
    assertExpressionsEqual, assertExpressionsStructurallyEqual)
from pyomo.environ import ConcreteModel, Constraint, log, Var

np, numpy_available = attempt_import('numpy')
scipy, scipy_available = attempt_import('scipy')

class TestPiecewiseLinearFunction2D(unittest.TestCase):
    def make_ln_x_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        def f(x):
            return log(x)
        m.f = f

        def f1(x):
            return (log(3)/2)*x - log(3)/2
        m.f1 = f1
        def f2(x):
            return (log(2)/3)*x + log(3/2)
        m.f2 = f2
        def f3(x):
            return (log(5/3)/4)*x + log(6/((5/3)**(3/2)))
        m.f3 = f3

        return m

    def check_ln_x_approx(self, pw, x):
        self.assertEqual(len(pw.simplices), 3)
        self.assertEqual(len(pw.linear_functions), 3)
        # indices of extreme points.
        simplices = [(0, 1), (1, 2), (2, 3)]
        for idx, simplex in enumerate(simplices):
            self.assertEqual(pw.simplices[idx], simplices[idx])

        assertExpressionsEqual(self, pw.linear_functions[0](x),
                               (log(3)/2)*x - log(3)/2, places=7)
        assertExpressionsEqual(self, pw.linear_functions[1](x),
                               (log(2)/3)*x + log(3/2), places=7)
        assertExpressionsEqual(self, pw.linear_functions[2](x),
                               (log(5/3)/4)*x + log(6/((5/3)**(3/2))),
                               places=7)

    def check_x_squared_approx(self, pw, x):
        self.assertEqual(len(pw.simplices), 3)
        self.assertEqual(len(pw.linear_functions), 3)
        # indices of extreme points.
        simplices = [(0, 1), (1, 2), (2, 3)]
        for idx, simplex in enumerate(simplices):
            self.assertEqual(pw.simplices[idx], simplices[idx])

        assertExpressionsStructurallyEqual(self, pw.linear_functions[0](x),
                                           4*x - 3, places=7)
        assertExpressionsStructurallyEqual(self, pw.linear_functions[1](x),
                                           9*x - 18, places=7)
        assertExpressionsStructurallyEqual(self, pw.linear_functions[2](x),
                                           16*x - 60, places=7)

    def test_pw_linear_approx_of_ln_x_simplices(self):
        m = self.make_ln_x_model()
        simplices = [(1, 3), (3, 6), (6, 10)]
        m.pw = PiecewiseLinearFunction(simplices=simplices, function=m.f)
        self.check_ln_x_approx(m.pw, m.x)

    def test_pw_linear_approx_of_ln_x_points(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(points=[1, 3, 6, 10], function=m.f)
        self.check_ln_x_approx(m.pw, m.x)

    def test_pw_linear_approx_of_ln_x_linear_funcs(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(simplices=[(1, 3), (3, 6), (6, 10)],
                                       linear_functions=[m.f1, m.f2, m.f3])
        self.check_ln_x_approx(m.pw, m.x)

    def test_use_pw_function_in_constraint(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(simplices=[(1, 3), (3, 6), (6, 10)],
                                       linear_functions=[m.f1, m.f2, m.f3])
        m.c = Constraint(expr=m.pw(m.x) <= 1)
        self.assertEqual(str(m.c.body), "pw(x)")

    def test_evaluate_pw_function(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(simplices=[(1, 3), (3, 6), (6, 10)],
                                       linear_functions=[m.f1, m.f2, m.f3])
        self.assertAlmostEqual(m.pw(1), 0)
        self.assertAlmostEqual(m.pw(2), m.f1(2))
        self.assertAlmostEqual(m.pw(3), log(3))
        self.assertAlmostEqual(m.pw(4.5), m.f2(4.5))
        self.assertAlmostEqual(m.pw(9.2), m.f3(9.2))
        self.assertAlmostEqual(m.pw(10), log(10))

    def test_indexed_pw_linear_function_approximate_over_simplices(self):
        m = self.make_ln_x_model()
        m.z = Var([1, 2], bounds=(-10, 10))
        def g1(x):
            return x**2
        def g2(x):
            return log(x)
        m.funcs = {1: g1, 2: g2}
        simplices = [(1, 3), (3, 6), (6, 10)]
        m.pw = PiecewiseLinearFunction([1, 2], simplices=simplices,
                                       function_rule=lambda m, i: m.funcs[i])
        self.check_ln_x_approx(m.pw[2], m.z[2])
        self.check_x_squared_approx(m.pw[1], m.z[1])

    def test_indexed_pw_linear_function_approximate_over_points(self):
        m = self.make_ln_x_model()
        m.z = Var([1, 2], bounds=(-10, 10))
        def g1(x):
            return x**2
        def g2(x):
            return log(x)
        m.funcs = {1: g1, 2: g2}
        def silly_pts_rule(m, i):
            return [1, 3, 6, 10]
        m.pw = PiecewiseLinearFunction([1, 2], points=silly_pts_rule,
                                       function_rule=lambda m, i: m.funcs[i])
        self.check_ln_x_approx(m.pw[2], m.z[2])
        self.check_x_squared_approx(m.pw[1], m.z[1])

    def test_indexed_pw_linear_function_linear_funcs_and_simplices(self):
        m = self.make_ln_x_model()
        m.z = Var([1, 2], bounds=(-10, 10))
        def silly_simplex_rule(m, i):
            return [(1, 3), (3, 6), (6, 10)]
        def h1(x):
            return 4*x - 3
        def h2(x):
            return 9*x - 18
        def h3(x):
            return 16*x - 60
        def silly_linear_func_rule(m, i):
            return [h1, h2, h3]
        m.pw = PiecewiseLinearFunction([1, 2], simplices=silly_simplex_rule,
                                       linear_functions=silly_linear_func_rule)
        self.check_x_squared_approx(m.pw[1], m.z[1])
        self.check_x_squared_approx(m.pw[2], m.z[2])

class TestPiecewiseLinearFunction3D(unittest.TestCase):
    @unittest.skipUnless(scipy_available and numpy_available,
                         "scipy and/or numpy are not available")
    def make_model(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 3))
        m.x2 = Var(bounds=(1, 7))
        # Here's a cute paraboloid:
        def f(x, y):
            return x**2 + y**2
        m.f = f
        return m

    def check_pw_linear_approximation(self, m):
        self.assertEqual(len(m.pw.simplices), 4)
        self.assertEqual(len(m.pw.linear_functions), 4)

        assertExpressionsStructurallyEqual(self,
                                           m.pw.linear_functions[0](m.x1, m.x2),
                                           3*m.x1 + 5*m.x2 - 4, places=7)
        assertExpressionsStructurallyEqual(self,
                                           m.pw.linear_functions[1](m.x1, m.x2),
                                           3*m.x1 + 5*m.x2 - 4, places=7)
        assertExpressionsStructurallyEqual(self,
                                           m.pw.linear_functions[2](m.x1, m.x2),
                                           3*m.x1 + 11*m.x2 - 28, places=7)
        assertExpressionsStructurallyEqual(self,
                                           m.pw.linear_functions[3](m.x1, m.x2),
                                           3*m.x1 + 11*m.x2 - 28, places=7)

    def test_pw_linear_approx_of_paraboloid_points(self):
        m = self.make_model()
        m.pw = PiecewiseLinearFunction(points=[(0, 1), (0, 4), (0, 7),
                                               (3, 1), (3, 4), (3, 7)],
                                       function=m.f)
        self.check_pw_linear_approximation(m)

    def test_pw_linear_approx_of_paraboloid_simplices(self):
        m = self.make_model()
        m.pw = PiecewiseLinearFunction(function=m.f,
                                       simplices=[[(0, 1), (0, 4), (3, 1)],
                                                  [(0, 1), (3, 4), (3, 1)],
                                                  [(3, 4), (3, 7), (0, 7)],
                                                  [(0, 7), (0, 4), (3, 4)]])
        self.check_pw_linear_approximation(m)

    def test_pw_linear_approx_of_paraboloid_linear_funcs(self):
        m = self.make_model()
        def f1(x1, x2):
            return 3*x1 + 5*x2 - 4
        def f2(x1, x2):
            return 3*x1 + 11*x2 - 28
        m.pw = PiecewiseLinearFunction(simplices=[[(0, 1), (0, 4), (3, 1)],
                                                  [(0, 1), (3, 4), (3, 1)],
                                                  [(3, 4), (3, 7), (0, 7)],
                                                  [(0, 7), (0, 4), (3, 4)]],
                                       linear_functions=[f1, f1, f2, f2])
        self.check_pw_linear_approximation(m)

    def test_use_pw_linear_approx_in_constraint(self):
        m = self.make_model()
        def f1(x1, x2):
            return 3*x1 + 5*x2 - 4
        def f2(x1, x2):
            return 3*x1 + 11*x2 - 28
        m.pw = PiecewiseLinearFunction(simplices=[[(0, 1), (0, 4), (3, 1)],
                                                  [(0, 1), (3, 4), (3, 1)],
                                                  [(3, 4), (3, 7), (0, 7)],
                                                  [(0, 7), (0, 4), (3, 4)]],
                                       linear_functions=[f1, f1, f2, f2])

        m.c = Constraint(expr=m.pw(m.x1, m.x2) <= 5)
        self.assertEqual(str(m.c.body), "pw(x1, x2)")

    def test_evaluate_pw_linear_function(self):
        m = self.make_model()
        def f1(x1, x2):
            return 3*x1 + 5*x2 - 4
        def f2(x1, x2):
            return 3*x1 + 11*x2 - 28
        simplices = [[(0, 1), (0, 4), (3, 1)],
                     [(0, 1), (3, 4), (3, 1)],
                     [(3, 4), (3, 7), (0, 7)],
                     [(0, 7), (0, 4), (3, 4)]]
        m.pw = PiecewiseLinearFunction(simplices=simplices,
                                       linear_functions=[f1, f1, f2, f2])
        # check it's equal to the original function at all the extreme points of
        # the simplices
        for (x1, x2) in m.pw.points:
            self.assertAlmostEqual(m.pw(x1, x2), m.f(x1, x2))
        # check some points in the approximation
        self.assertAlmostEqual(m.pw(1, 3), f1(1, 3))
        self.assertAlmostEqual(m.pw(2.5, 6), f2(2.5, 6))
        self.assertAlmostEqual(m.pw(0.2, 4.3), f2(0.2, 4.3))

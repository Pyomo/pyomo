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
from pyomo.environ import ConcreteModel, log, Var

np, numpy_available = attempt_import('numpy')
scipy, scipy_available = attempt_import('scipy')

class TestPiecewiseLinearFunction2D(unittest.TestCase):
    def make_ln_x_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        def f(x):
            return log(x)
        m.f = f

        return m

    def check_ln_x_approx(self, m):
        self.assertEqual(len(m.pw.simplices), 3)
        self.assertEqual(len(m.pw.linear_functions), 3)
        # indices of extreme points.
        simplices = [(0, 1), (1, 2), (2, 3)]
        for idx, simplex in enumerate(simplices):
            self.assertEqual(m.pw.simplices[idx], simplices[idx])

        assertExpressionsEqual(self, m.pw.linear_functions[0](m.x),
                               (log(3)/2)*m.x - log(3)/2, places=7)
        assertExpressionsEqual(self, m.pw.linear_functions[1](m.x),
                               (log(2)/3)*m.x + log(3/2), places=7)
        assertExpressionsEqual(self, m.pw.linear_functions[2](m.x),
                               (log(5/3)/4)*m.x + log(6/((5/3)**(3/2))),
                               places=7)

    def test_pw_linear_approx_of_ln_x_simplices(self):
        m = self.make_ln_x_model()
        simplices = [(1, 3), (3, 6), (6, 10)]
        m.pw = PiecewiseLinearFunction(simplices=simplices, function=m.f)
        self.check_ln_x_approx(m)

    def test_pw_linear_approx_of_ln_x_points(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(points=[1, 3, 6, 10], function=m.f)
        self.check_ln_x_approx(m)

    def test_pw_linear_approx_of_ln_x_linear_funcs(self):
        m = self.make_ln_x_model()
        def f1(x):
            return (log(3)/2)*x - log(3)/2
        m.f1 = f1
        def f2(x):
            return (log(2)/3)*x + log(3/2)
        m.f2 = f2
        def f3(x):
            return (log(5/3)/4)*x + log(6/((5/3)**(3/2)))
        m.f3 = f3
                    
        m.pw = PiecewiseLinearFunction(simplices=[(1, 3), (3, 6), (6, 10)],
                                       linear_functions=[m.f1, m.f2, m.f3])
        self.check_ln_x_approx(m)

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
                                           3*m.x1 + 5*m.x2 - 4)
        assertExpressionsStructurallyEqual(self,
                                           m.pw.linear_functions[1](m.x1, m.x2),
                                           3*m.x1 + 5*m.x2 - 4)
        assertExpressionsStructurallyEqual(self,
                                           m.pw.linear_functions[2](m.x1, m.x2),
                                           3*m.x1 + 11*m.x2 - 28)
        assertExpressionsStructurallyEqual(self,
                                           m.pw.linear_functions[3](m.x1, m.x2),
                                           3*m.x1 + 11*m.x2 - 28)

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

        #m.c = Constraint(expr=m.pw(m.x1, m.x2) <= 5)


# def alternate_thing():
#     def f(x, y):
#         return x**2 + y**2

#     m.pw = PiecewiseLinearFunction(points=[...],
#                                    function=f,
#                                    #nargs=2,
#     )
#     m.c = Constraint(expr=m.pw(m.x, m.y) <= 5)

#     #

#     PLF(points, function)
#     PLF(simplices, function?)
#     PLF(simplices, linear_expression_list) # ESJ: Should these be
                                             # expressions or functions?
#     PLF(table_data) ... {point: val}

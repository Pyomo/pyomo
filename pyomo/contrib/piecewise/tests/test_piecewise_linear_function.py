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

from io import StringIO
import logging
import pickle

from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.environ import ConcreteModel, Constraint, log, Var


np, numpy_available = attempt_import('numpy')
scipy, scipy_available = attempt_import('scipy')


def f(x):
    return log(x)


def f1(x):
    return (log(3) / 2) * x - log(3) / 2


def f2(x):
    return (log(2) / 3) * x + log(3 / 2)


def f3(x):
    return (log(5 / 3) / 4) * x + log(6 / ((5 / 3) ** (3 / 2)))


class TestPiecewiseLinearFunction2D(unittest.TestCase):
    def make_ln_x_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        m.f = f
        m.f1 = f1
        m.f2 = f2
        m.f3 = f3

        return m

    def check_ln_x_approx(self, pw, x):
        self.assertEqual(len(pw._simplices), 3)
        self.assertEqual(len(pw._linear_functions), 3)
        # indices of extreme points.
        simplices = [(0, 1), (1, 2), (2, 3)]
        for idx, simplex in enumerate(simplices):
            self.assertEqual(pw._simplices[idx], simplices[idx])

        assertExpressionsEqual(
            self, pw._linear_functions[0](x), (log(3) / 2) * x - log(3) / 2, places=7
        )
        assertExpressionsEqual(
            self, pw._linear_functions[1](x), (log(2) / 3) * x + log(3 / 2), places=7
        )
        assertExpressionsEqual(
            self,
            pw._linear_functions[2](x),
            (log(5 / 3) / 4) * x + log(6 / ((5 / 3) ** (3 / 2))),
            places=7,
        )

    def check_x_squared_approx(self, pw, x):
        self.assertEqual(len(pw._simplices), 3)
        self.assertEqual(len(pw._linear_functions), 3)
        # indices of extreme points.
        simplices = [(0, 1), (1, 2), (2, 3)]
        for idx, simplex in enumerate(simplices):
            self.assertEqual(pw._simplices[idx], simplices[idx])

        assertExpressionsStructurallyEqual(
            self, pw._linear_functions[0](x), 4 * x - 3, places=7
        )
        assertExpressionsStructurallyEqual(
            self, pw._linear_functions[1](x), 9 * x - 18, places=7
        )
        assertExpressionsStructurallyEqual(
            self, pw._linear_functions[2](x), 16 * x - 60, places=7
        )

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
        m.pw = PiecewiseLinearFunction(
            simplices=[(1, 3), (3, 6), (6, 10)], linear_functions=[m.f1, m.f2, m.f3]
        )
        self.check_ln_x_approx(m.pw, m.x)

    def test_use_pw_function_in_constraint(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(
            simplices=[(1, 3), (3, 6), (6, 10)], linear_functions=[m.f1, m.f2, m.f3]
        )
        m.c = Constraint(expr=m.pw(m.x) <= 1)
        self.assertEqual(str(m.c.body.expr), "pw(x)")

    def test_evaluate_pw_function(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(
            simplices=[(1, 3), (3, 6), (6, 10)], linear_functions=[m.f1, m.f2, m.f3]
        )
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
        m.pw = PiecewiseLinearFunction(
            [1, 2], simplices=simplices, function_rule=lambda m, i: m.funcs[i]
        )
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

        m.pw = PiecewiseLinearFunction(
            [1, 2], points=silly_pts_rule, function_rule=lambda m, i: m.funcs[i]
        )
        self.check_ln_x_approx(m.pw[2], m.z[2])
        self.check_x_squared_approx(m.pw[1], m.z[1])

    def test_indexed_pw_linear_function_linear_funcs_and_simplices(self):
        m = self.make_ln_x_model()
        m.z = Var([1, 2], bounds=(-10, 10))

        def silly_simplex_rule(m, i):
            return [(1, 3), (3, 6), (6, 10)]

        def h1(x):
            return 4 * x - 3

        def h2(x):
            return 9 * x - 18

        def h3(x):
            return 16 * x - 60

        def silly_linear_func_rule(m, i):
            return [h1, h2, h3]

        m.pw = PiecewiseLinearFunction(
            [1, 2],
            simplices=silly_simplex_rule,
            linear_functions=silly_linear_func_rule,
        )
        self.check_x_squared_approx(m.pw[1], m.z[1])
        self.check_x_squared_approx(m.pw[2], m.z[2])

    def test_pickle(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(points=[1, 3, 6, 10], function=m.f)
        m.c = Constraint(expr=m.pw(m.x) >= 0.35)

        # pickle and unpickle
        unpickle = pickle.loads(pickle.dumps(m))

        # Check that the pprint is equal
        m_buf = StringIO()
        m.pprint(ostream=m_buf)
        m_output = m_buf.getvalue()

        unpickle_buf = StringIO()
        unpickle.pprint(ostream=unpickle_buf)
        unpickle_output = unpickle_buf.getvalue()
        self.assertMultiLineEqual(m_output, unpickle_output)


# Here's a cute paraboloid:
def g(x, y):
    return x**2 + y**2


class TestPiecewiseLinearFunction3D(unittest.TestCase):
    simplices = [
        [(0, 1), (0, 4), (3, 4)],
        [(0, 1), (3, 4), (3, 1)],
        [(3, 4), (3, 7), (0, 7)],
        [(0, 7), (0, 4), (3, 4)],
    ]

    def make_model(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 3))
        m.x2 = Var(bounds=(1, 7))
        m.g = g
        return m

    def check_pw_linear_approximation(self, m):
        self.assertEqual(len(m.pw._simplices), 4)
        for i, simplex in enumerate(m.pw._simplices):
            for idx in simplex:
                self.assertIn(m.pw._points[idx], self.simplices[i])

        self.assertEqual(len(m.pw._linear_functions), 4)

        assertExpressionsStructurallyEqual(
            self,
            m.pw._linear_functions[0](m.x1, m.x2),
            3 * m.x1 + 5 * m.x2 - 4,
            places=7,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.pw._linear_functions[1](m.x1, m.x2),
            3 * m.x1 + 5 * m.x2 - 4,
            places=7,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.pw._linear_functions[2](m.x1, m.x2),
            3 * m.x1 + 11 * m.x2 - 28,
            places=7,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.pw._linear_functions[3](m.x1, m.x2),
            3 * m.x1 + 11 * m.x2 - 28,
            places=7,
        )

    @unittest.skipUnless(
        scipy_available and numpy_available, "scipy and/or numpy are not available"
    )
    def test_pw_linear_approx_of_paraboloid_points(self):
        m = self.make_model()
        m.pw = PiecewiseLinearFunction(
            points=[(0, 1), (0, 4), (0, 7), (3, 1), (3, 4), (3, 7)], function=m.g
        )
        self.check_pw_linear_approximation(m)

    @unittest.skipUnless(numpy_available, "numpy are not available")
    def test_pw_linear_approx_of_paraboloid_simplices(self):
        m = self.make_model()
        m.pw = PiecewiseLinearFunction(function=m.g, simplices=self.simplices)
        self.check_pw_linear_approximation(m)

    def test_pw_linear_approx_of_paraboloid_linear_funcs(self):
        m = self.make_model()

        def g1(x1, x2):
            return 3 * x1 + 5 * x2 - 4

        def g2(x1, x2):
            return 3 * x1 + 11 * x2 - 28

        m.pw = PiecewiseLinearFunction(
            simplices=self.simplices, linear_functions=[g1, g1, g2, g2]
        )
        self.check_pw_linear_approximation(m)

    def test_use_pw_linear_approx_in_constraint(self):
        m = self.make_model()

        def g1(x1, x2):
            return 3 * x1 + 5 * x2 - 4

        def g2(x1, x2):
            return 3 * x1 + 11 * x2 - 28

        m.pw = PiecewiseLinearFunction(
            simplices=self.simplices, linear_functions=[g1, g1, g2, g2]
        )

        m.c = Constraint(expr=m.pw(m.x1, m.x2) <= 5)
        self.assertEqual(str(m.c.body.expr), "pw(x1, x2)")
        self.assertIs(m.c.body.expr.pw_linear_function, m.pw)

    @unittest.skipUnless(numpy_available, "numpy is not available")
    def test_evaluate_pw_linear_function(self):
        # NOTE: This test requires numpy because it is used to check which
        # simplex a point is in
        m = self.make_model()

        def g1(x1, x2):
            return 3 * x1 + 5 * x2 - 4

        def g2(x1, x2):
            return 3 * x1 + 11 * x2 - 28

        m.pw = PiecewiseLinearFunction(
            simplices=self.simplices, linear_functions=[g1, g1, g2, g2]
        )
        # check it's equal to the original function at all the extreme points of
        # the simplices
        for x1, x2 in m.pw._points:
            self.assertAlmostEqual(m.pw(x1, x2), m.g(x1, x2))
        # check some points in the approximation
        self.assertAlmostEqual(m.pw(1, 3), g1(1, 3))
        self.assertAlmostEqual(m.pw(2.5, 6), g2(2.5, 6))
        self.assertAlmostEqual(m.pw(0.2, 4.3), g2(0.2, 4.3))


class TestTriangulationProducesDegenerateSimplices(unittest.TestCase):
    simplices = [
        # right bottom back cube with vertices (10, 11, 13, 14, 19, 20, 22, 23)
        (11, 22, 14, 13),
        (11, 10, 22, 13),
        (11, 23, 22, 20),
        (11, 23, 22, 14),
        (11, 19, 10, 22),
        (11, 19, 22, 20),
        # right bottom front cube with vertices (9, 10, 12, 13, 18, 19, 21, 22)
        (21, 10, 22, 13),
        (21, 12, 10, 13),
        (21, 19, 10, 22),
        (21, 19, 10, 18),
        (21, 9, 12, 10),
        (21, 9, 10, 18),
        # left bottom front cube with vertices (0, 1, 3, 4, 9, 10, 12, 13)
        (1, 12, 10, 13),
        (1, 4, 12, 13),
        (1, 9, 12, 10),
        (1, 9, 12, 0),
        (1, 3, 4, 12),
        (1, 3, 12, 0),
        # left bottom back cube with vertices (1, 2, 4, 5, 10, 11, 13, 14)
        (1, 10, 14, 13),
        (1, 11, 10, 14),
        (1, 4, 14, 13),
        (1, 11, 14, 2),
        (1, 5, 4, 14),
        (1, 5, 14, 2),
        # left top front cube with vertices (3, 4, 6, 7, 12, 13, 15, 16)
        (7, 4, 12, 13),
        (7, 16, 12, 13),
        (7, 3, 4, 12),
        (7, 3, 12, 6),
        (7, 15, 16, 12),
        (7, 15, 12, 6),
        # left top back cube with vertices (4, 5, 7, 8, 13, 14, 16, 17)
        (17, 4, 14, 13),
        (17, 16, 4, 13),
        (17, 7, 16, 4),
        (17, 5, 4, 14),
        (17, 5, 4, 8),
        (17, 7, 4, 8),
        # right top front cube with vertices (12, 13, 15, 16, 21, 22, 24, 25)
        (25, 12, 22, 13),
        (25, 16, 12, 13),
        (25, 15, 16, 12),
        (25, 21, 12, 22),
        (25, 21, 12, 24),
        (25, 15, 12, 24),
        # right top back cube with vertices (13, 14, 16, 17, 22, 23, 25, 26)
        (25, 22, 14, 13),
        (25, 16, 14, 13),
        (25, 17, 16, 14),
        (25, 23, 22, 14),
        (25, 23, 14, 26),
        (25, 17, 14, 26),
    ]

    def make_model(self):
        m = ConcreteModel()

        m.f = lambda x1, x2, y: x1 * x2 + y
        # This is a 2x2 stack of cubes, so there are 8 total cubes, each of which
        # will get divided into 6 simplices.
        m.points = [
            (-2.0, 0.0, 1.0),
            (-2.0, 0.0, 4.0),
            (-2.0, 0.0, 7.0),
            (-2.0, 1.5, 1.0),
            (-2.0, 1.5, 4.0),
            (-2.0, 1.5, 7.0),
            (-2.0, 3.0, 1.0),
            (-2.0, 3.0, 4.0),
            (-2.0, 3.0, 7.0),
            (-1.5, 0.0, 1.0),
            (-1.5, 0.0, 4.0),
            (-1.5, 0.0, 7.0),
            (-1.5, 1.5, 1.0),
            (-1.5, 1.5, 4.0),
            (-1.5, 1.5, 7.0),
            (-1.5, 3.0, 1.0),
            (-1.5, 3.0, 4.0),
            (-1.5, 3.0, 7.0),
            (-1.0, 0.0, 1.0),
            (-1.0, 0.0, 4.0),
            (-1.0, 0.0, 7.0),
            (-1.0, 1.5, 1.0),
            (-1.0, 1.5, 4.0),
            (-1.0, 1.5, 7.0),
            (-1.0, 3.0, 1.0),
            (-1.0, 3.0, 4.0),
            (-1.0, 3.0, 7.0),
        ]
        return m

    @unittest.skipUnless(
        scipy_available and numpy_available, "scipy and/or numpy are not available"
    )
    def test_degenerate_simplices_filtered(self):
        m = self.make_model()
        pw = m.approx = PiecewiseLinearFunction(points=m.points, function=m.f)

        # check that all the points got used
        self.assertEqual(len(pw._points), 27)
        for p_model, p_pw in zip(m.points, pw._points):
            self.assertEqual(p_model, p_pw)

        # Started with a 2x2 grid of cubes, and each is divided into 6
        # simplices. It's crazy degenerate in terms of *how* this is done, but
        # that's the point of this test.
        self.assertEqual(len(pw._simplices), 48)
        for simplex, baseline in zip(sorted(self.simplices), sorted(pw._simplices)):
            self.assertEqual(simplex, baseline)

        # We're not checking 48 linear functions...

    @unittest.skipUnless(
        scipy_available and numpy_available, "scipy and/or numpy are not available"
    )
    def test_redundant_points_logged(self):
        m = self.make_model()
        # add a redundant point
        m.points.append((-2, 0, 1))

        out = StringIO()
        with LoggingIntercept(out, 'contrib.piecewise', level=logging.INFO):
            m.approx = PiecewiseLinearFunction(points=m.points, function=m.f)

        self.assertIn(
            "The Delaunay triangulation dropped the point with index 27 "
            "from the triangulation",
            out.getvalue(),
        )

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

from io import StringIO
import logging
import pickle

from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction, Triangulation
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

    def test_pw_linear_approx_of_ln_x_tabular_data(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(
            tabular_data={1: 0, 3: log(3), 6: log(6), 10: log(10)}
        )
        self.check_ln_x_approx(m.pw, m.x)

    def test_pw_linear_approx_of_ln_x_j1(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(
            points=[1, 3, 6, 10], triangulation=Triangulation.J1, function=m.f
        )
        self.check_ln_x_approx(m.pw, m.x)
        # we disregard their request because it's 1D
        self.assertEqual(m.pw.triangulation, Triangulation.AssumeValid)

    def test_pw_linear_approx_of_ln_x_user_defined_segments(self):
        m = self.make_ln_x_model()
        m.pw = PiecewiseLinearFunction(
            simplices=[[1, 3], [3, 6], [6, 10]], function=m.f
        )
        self.check_ln_x_approx(m.pw, m.x)
        self.assertEqual(m.pw.triangulation, Triangulation.Unknown)

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

    def test_indexed_pw_linear_function_tabular_data(self):
        m = self.make_ln_x_model()
        m.z = Var([1, 2], bounds=(-10, 10))

        def silly_tabular_data_rule(m, i):
            if i == 1:
                return {1: 1, 3: 9, 6: 36, 10: 100}
            if i == 2:
                return {1: 0, 3: log(3), 6: log(6), 10: log(10)}

        m.pw = PiecewiseLinearFunction(
            [1, 2], tabular_data_rule=silly_tabular_data_rule
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

    @unittest.skipUnless(numpy_available, "numpy is not available")
    def test_pw_linear_approx_of_paraboloid_j1(self):
        m = self.make_model()
        m.pw = PiecewiseLinearFunction(
            points=[
                (0, 1),
                (0, 4),
                (0, 7),
                (3, 1),
                (3, 4),
                (3, 7),
                (4, 1),
                (4, 4),
                (4, 7),
            ],
            function=m.g,
            triangulation=Triangulation.OrderedJ1,
        )
        self.assertEqual(len(m.pw._simplices), 8)
        self.assertEqual(m.pw.triangulation, Triangulation.OrderedJ1)

    @unittest.skipUnless(scipy_available, "scipy is not available")
    def test_pw_linear_approx_tabular_data(self):
        m = self.make_model()

        m.pw = PiecewiseLinearFunction(
            tabular_data={
                (0, 1): g(0, 1),
                (0, 4): g(0, 4),
                (0, 7): g(0, 7),
                (3, 1): g(3, 1),
                (3, 4): g(3, 4),
                (3, 7): g(3, 7),
            }
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
    cube_extreme_pt_indices = [
        {10, 11, 13, 14, 19, 20, 22, 23},  # right bottom back
        {9, 10, 12, 13, 18, 19, 21, 22},  # right bottom front
        {0, 1, 3, 4, 9, 10, 12, 13},  # left bottom front
        {1, 2, 4, 5, 10, 11, 13, 14},  # left bottom back
        {3, 4, 6, 7, 12, 13, 15, 16},  # left top front
        {4, 5, 7, 8, 13, 14, 16, 17},  # left top back
        {12, 13, 15, 16, 21, 22, 24, 25},  # right top front
        {13, 14, 16, 17, 22, 23, 25, 26},  # right top back
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
        simplex_in_cube = {idx: 0 for idx in range(8)}
        for simplex in pw._simplices:
            for i, vertex_set in enumerate(self.cube_extreme_pt_indices):
                if set(simplex).issubset(vertex_set):
                    simplex_in_cube[i] += 1
            # verify the simplex is full-dimensional
            pts = np.array([pw._points[j] for j in simplex]).transpose()
            A = pts[:, 1:] - np.append(pts[:, :2], pts[:, [0]], axis=1)
            self.assertNotEqual(np.linalg.det(A), 0)

        # Check that they are 6 to a cube, as expected
        for num in simplex_in_cube.values():
            self.assertEqual(num, 6)

    @unittest.skipUnless(
        scipy_available and numpy_available, "scipy and/or numpy are not available"
    )
    def test_redundant_points_logged(self):
        m = self.make_model()
        # add a redundant point
        m.points.append((-2, 0, 1))

        out = StringIO()
        with LoggingIntercept(
            out, 'pyomo.contrib.piecewise.piecewise_linear_function', level=logging.INFO
        ):
            m.approx = PiecewiseLinearFunction(points=m.points, function=m.f)

        self.assertIn(
            "The Delaunay triangulation dropped the point with index 27 "
            "from the triangulation",
            out.getvalue(),
        )

    @unittest.skipUnless(numpy_available, "numpy is not available")
    def test_user_given_degenerate_simplex_error(self):
        m = self.make_model()
        with self.assertRaisesRegex(
            ValueError,
            "When calculating the hyperplane approximation over the simplex "
            "with index 0, the matrix was unexpectedly singular. This "
            "likely means that this simplex is degenerate",
        ):
            m.pw = PiecewiseLinearFunction(
                simplices=[
                    (
                        (-2.0, 0.0, 1.0),
                        (-2.0, 0.0, 4.0),
                        (-2.0, 1.5, 1.0),
                        (-2.0, 1.5, 4.0),
                    )
                ],
                function=m.f,
            )

    @unittest.skipUnless(
        scipy_available and numpy_available, "scipy and/or numpy are not available"
    )
    def test_simplex_not_numerically_full_rank_but_determinant_nonzero(self):
        m = ConcreteModel()

        def f(x3, x6, x9, x4):
            return -x6 * (0.01 * x4 * x9 + x3) + 0.98 * x3

        points = [
            (0, 0.85, 1.2, 0),
            (0.07478, 0.86396, 1.8668, 5),
            (0, 0.85, 1.8668, 0),
            (0.07478, 0.86396, 2.18751, 5),
            (0, 0.86396, 1.2, 0),
            (0.07478, 0.87971, 2.18751, 5),
            (0, 0.87971, 1.2, 0),
            (0.07478, 0.89001, 2.18751, 5),
            (0.07478, 0.85, 1.2, 0),
            (0.28333, 0.86396, 2.18751, 5),
            (0.07478, 0.86396, 1.2, 0),
            (0.28333, 0.89001, 2.18751, 5),
            (0.28333, 0.85, 1.2, 0),
            (0.31332, 0.89001, 2.18751, 5),
            (0.31332, 0.85, 1.2, 0),
            (1.2, 0.89001, 2.18751, 5),
            (0, 0.89001, 1.2, 0),
            (0.07478, 0.91727, 1.8668, 5),
            (0, 0.89001, 1.8668, 0),
            (0.07478, 0.91727, 2.18751, 5),
            (0, 0.91727, 1.2, 0),
            (0.07478, 0.93, 2.18751, 5),
            (0.07478, 0.89001, 1.2, 0),
            (0.28333, 0.91727, 2.18751, 5),
            (0.07478, 0.91727, 1.2, 0),
            (0.28333, 0.93, 2.18751, 5),
            (0.28333, 0.89001, 1.2, 0),
            (0.31332, 0.93, 2.18751, 5),
            (0.31332, 0.89001, 1.2, 0),
            (1.2, 0.93, 2.18751, 5),
            (0, 0.85, 2.18751, 0),
            (0.07478, 0.86396, 3.49134, 5),
            (0, 0.85, 3.49134, 0),
            (0.07478, 0.86396, 4, 5),
            (0, 0.86396, 2.18751, 0),
            (0.07478, 0.87971, 4, 5),
            (0, 0.87971, 2.18751, 0),
            (0.07478, 0.89001, 4, 5),
            (0.07478, 0.85, 2.18751, 0),
            (0.28333, 0.86396, 4, 5),
            (0.07478, 0.86396, 2.18751, 0),
            (0.28333, 0.89001, 4, 5),
            (0.28333, 0.85, 2.18751, 0),
            (0.31332, 0.89001, 4, 5),
            (0.31332, 0.85, 2.18751, 0),
            (1.2, 0.89001, 4, 5),
            (0, 0.89001, 2.18751, 0),
            (0.07478, 0.91727, 3.49134, 5),
            (0, 0.89001, 3.49134, 0),
            (0.07478, 0.91727, 4, 5),
            (0, 0.91727, 2.18751, 0),
            (0.07478, 0.93, 3.49134, 5),
            (0, 0.91727, 3.49134, 0),
            (0.07478, 0.93, 4, 5),
            (0.07478, 0.89001, 2.18751, 0),
            (0.28333, 0.91727, 4, 5),
            (0.07478, 0.91727, 2.18751, 0),
            (0.28333, 0.93, 4, 5),
            (0.28333, 0.89001, 2.18751, 0),
            (0.31332, 0.93, 4, 5),
            (0.31332, 0.89001, 2.18751, 0),
            (1.2, 0.93, 4, 5),
        ]
        m.pw = PiecewiseLinearFunction(points=points, function=f)

        # The big win is if the above runs, but we'll check the approximation
        # computationally at least, to make sure that at all the points we gave,
        # the pw linear approximation evaluates to the same value as the
        # original nonlinear function.
        for pt in points:
            self.assertAlmostEqual(m.pw(*pt), f(*pt))

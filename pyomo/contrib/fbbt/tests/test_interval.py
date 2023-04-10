import math
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.errors import InfeasibleConstraintException
import pyomo.contrib.fbbt.interval as interval

try:
    isfinite = math.isfinite
except AttributeError:
    # isfinite() was added to math in Python 3.2
    def isfinite(x):
        return not (math.isnan(x) or math.isinf(x))


class IntervalTestBase(object):
    """
    These tests are set up weird, but it is for a good reason.
    The interval arithmetic code is duplicated in pyomo.contrib.appsi for
    improved performance. We want to keep this version because
    it does not require building an extension. However, when we
    fix a bug in one module, we want to ensure we fix that bug
    in the other module. Therefore, we use this base class
    for testing both modules. The only difference in the
    derived classes is in the self.add, self.sub,
    self.mul, etc. attributes.
    """

    def setUp(self):
        if numpy_available:
            np.random.seed(0)

    def test_add(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        xl = -2.5
        xu = 2.8
        yl = -3.2
        yu = 2.7
        zl, zu = self.add(xl, xu, yl, yu)
        x = np.linspace(xl, xu, 100)
        y = np.linspace(yl, yu, 100)
        for _x in x:
            _z = _x + y
            self.assertTrue(np.all(zl <= _z))
            self.assertTrue(np.all(zu >= _z))

    def test_sub(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        xl = -2.5
        xu = 2.8
        yl = -3.2
        yu = 2.7
        zl, zu = self.sub(xl, xu, yl, yu)
        x = np.linspace(xl, xu, 100)
        y = np.linspace(yl, yu, 100)
        for _x in x:
            _z = _x - y
            self.assertTrue(np.all(zl <= _z))
            self.assertTrue(np.all(zu >= _z))

    def test_mul(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        xl = -2.5
        xu = 2.8
        yl = -3.2
        yu = 2.7
        zl, zu = self.mul(xl, xu, yl, yu)
        x = np.linspace(xl, xu, 100)
        y = np.linspace(yl, yu, 100)
        for _x in x:
            _z = _x * y
            self.assertTrue(np.all(zl <= _z))
            self.assertTrue(np.all(zu >= _z))

    def test_inv(self):
        lb, ub = self.inv(0.1, 0.2, 1e-8)
        self.assertAlmostEqual(lb, 5)
        self.assertAlmostEqual(ub, 10)

        lb, ub = self.inv(0, 0.1, 1e-8)
        self.assertEqual(lb, 10)
        self.assertEqual(ub, interval.inf)

        lb, ub = self.inv(0, 0, 1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = self.inv(-0.1, 0, 1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, -10)

        lb, ub = self.inv(-0.2, -0.1, 1e-8)
        self.assertAlmostEqual(lb, -10)
        self.assertAlmostEqual(ub, -5)

        lb, ub = self.inv(0, -1e-16, 1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = self.inv(1e-16, 0, 1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = self.inv(-1, 1, 1e-8)
        self.assertAlmostEqual(lb, -interval.inf)
        self.assertAlmostEqual(ub, interval.inf)

    def test_div(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        x_bounds = [(np.random.uniform(-5, -2), np.random.uniform(2, 5))]
        y_bounds = [
            (np.random.uniform(-5, -2), np.random.uniform(2, 5)),
            (0, np.random.uniform(2, 5)),
            (np.random.uniform(0, 2), np.random.uniform(2, 5)),
            (np.random.uniform(-5, -2), 0),
            (np.random.uniform(-5, -2), np.random.uniform(-2, 0)),
        ]
        for xl, xu in x_bounds:
            for yl, yu in y_bounds:
                zl, zu = self.div(xl, xu, yl, yu, 1e-8)
                x = np.linspace(xl, xu, 100)
                y = np.linspace(yl, yu, 100)
                for _x in x:
                    _z = _x / y
                    self.assertTrue(np.all(zl <= _z))
                    self.assertTrue(np.all(zu >= _z))

    def test_div_edge_cases(self):
        lb, ub = self.div(0, -1e-16, 0, 0, 1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = self.div(0, 1e-16, 0, 0, 1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = self.div(-1e-16, 0, 0, 0, 1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = self.div(1e-16, 0, 0, 0, 1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

    def test_pow(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        x_bounds = [
            (np.random.uniform(0, 2), np.random.uniform(2, 5)),
            (0, np.random.uniform(2, 5)),
            (0, 0),
        ]
        y_bounds = [
            (np.random.uniform(-5, -2), np.random.uniform(2, 5)),
            (0, np.random.uniform(2, 5)),
            (np.random.uniform(0, 2), np.random.uniform(2, 5)),
            (np.random.uniform(-5, -2), 0),
            (np.random.uniform(-5, -2), np.random.uniform(-2, 0)),
        ]
        for xl, xu in x_bounds:
            for yl, yu in y_bounds:
                zl, zu = self.power(xl, xu, yl, yu, 1e-8)
                if xl == 0 and xu == 0 and yu < 0:
                    self.assertEqual(zl, -interval.inf)
                    self.assertEqual(zu, interval.inf)
                x = np.linspace(xl, xu, 100)
                y = np.linspace(yl, yu, 100)
                for _x in x:
                    _z = _x**y
                    self.assertTrue(np.all(zl - 1e-14 <= _z))
                    self.assertTrue(np.all(zu + 1e-14 >= _z))

        x_bounds = [
            (np.random.uniform(-5, -2), np.random.uniform(2, 5)),
            (np.random.uniform(-5, -2), np.random.uniform(-2, 0)),
            (np.random.uniform(-5, -2), 0),
        ]
        y_bounds = list(range(-4, 4))
        for xl, xu in x_bounds:
            for yl in y_bounds:
                yu = yl
                zl, zu = self.power(xl, xu, yl, yu, 1e-8)
                x = np.linspace(xl, xu, 100, endpoint=False)
                y = yl
                _z = x**y
                self.assertTrue(np.all(zl <= _z))
                self.assertTrue(np.all(zu >= _z))

    def test_pow2(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        xl = np.linspace(-2, 2, 9)
        xu = np.linspace(-2, 2, 9)
        yl = np.linspace(-2, 2, 9)
        yu = np.linspace(-2, 2, 9)
        for _xl in xl:
            for _xu in xu:
                if _xl > _xu:
                    continue
                for _yl in yl:
                    for _yu in yu:
                        if _yl > _yu:
                            continue
                        if _xl == 0 and _xu == 0 and _yu < 0:
                            lb, ub = self.power(_xl, _xu, _yl, _yu, 1e-8)
                            self.assertEqual(lb, -interval.inf)
                            self.assertEqual(ub, interval.inf)
                        elif (
                            _yl == _yu
                            and _yl != round(_yl)
                            and (_xu < 0 or (_xu < 0 and _yu < 0))
                        ):
                            with self.assertRaises(
                                (
                                    InfeasibleConstraintException,
                                    interval.IntervalException,
                                )
                            ):
                                lb, ub = self.power(_xl, _xu, _yl, _yu, 1e-8)
                        else:
                            lb, ub = self.power(_xl, _xu, _yl, _yu, 1e-8)
                            if isfinite(lb) and isfinite(ub):
                                nan_fill = 0.5 * (lb + ub)
                            elif isfinite(lb):
                                nan_fill = lb + 1
                            elif isfinite(ub):
                                nan_fill = ub - 1
                            else:
                                nan_fill = 0
                            x = np.linspace(_xl, _xu, 30)
                            y = np.linspace(_yl, _yu, 30)
                            z = x ** np.split(y, len(y))
                            z[np.isnan(z)] = nan_fill
                            all_values = z
                            estimated_lb = all_values.min()
                            estimated_ub = all_values.max()
                            self.assertTrue(lb - 1e-8 <= estimated_lb)
                            self.assertTrue(ub + 1e-8 >= estimated_ub)

    def test_exp(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        xl = -2.5
        xu = 2.8
        zl, zu = self.exp(xl, xu)
        x = np.linspace(xl, xu, 100)
        _z = np.exp(x)
        self.assertTrue(np.all(zl <= _z))
        self.assertTrue(np.all(zu >= _z))

    def test_log(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        x_bounds = [
            (np.random.uniform(0, 2), np.random.uniform(2, 5)),
            (0, np.random.uniform(2, 5)),
            (0, 0),
        ]
        for xl, xu in x_bounds:
            zl, zu = self.log(xl, xu)
            x = np.linspace(xl, xu, 100)
            _z = np.log(x)
            self.assertTrue(np.all(zl <= _z))
            self.assertTrue(np.all(zu >= _z))

    def test_cos(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        lbs = np.linspace(-2 * math.pi, 2 * math.pi, 10)
        ubs = np.linspace(-2 * math.pi, 2 * math.pi, 10)
        for xl in lbs:
            for xu in ubs:
                if xu >= xl:
                    zl, zu = self.cos(xl, xu)
                    x = np.linspace(xl, xu, 100)
                    _z = np.cos(x)
                    self.assertTrue(np.all(zl <= _z + 1e-14))
                    self.assertTrue(np.all(zu >= _z - 1e-14))

    def test_sin(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        lbs = np.linspace(-2 * math.pi, 2 * math.pi, 10)
        ubs = np.linspace(-2 * math.pi, 2 * math.pi, 10)
        for xl in lbs:
            for xu in ubs:
                if xu >= xl:
                    zl, zu = self.sin(xl, xu)
                    x = np.linspace(xl, xu, 100)
                    _z = np.sin(x)
                    self.assertTrue(np.all(zl <= _z + 1e-14))
                    self.assertTrue(np.all(zu >= _z - 1e-14))

    def test_tan(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        lbs = np.linspace(-2 * math.pi, 2 * math.pi, 10)
        ubs = np.linspace(-2 * math.pi, 2 * math.pi, 10)
        for xl in lbs:
            for xu in ubs:
                if xu >= xl:
                    zl, zu = self.tan(xl, xu)
                    x = np.linspace(xl, xu, 100)
                    _z = np.tan(x)
                    self.assertTrue(np.all(zl <= _z + 1e-14))
                    self.assertTrue(np.all(zu >= _z - 1e-14))

    def test_asin(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        yl, yu = self.asin(-0.5, 0.5, -interval.inf, interval.inf, 1e-8)
        self.assertEqual(yl, -interval.inf)
        self.assertEqual(yu, interval.inf)
        yl, yu = self.asin(-0.5, 0.5, -math.pi, math.pi, 1e-8)
        self.assertAlmostEqual(yl, -math.pi, 12)
        self.assertAlmostEqual(yu, math.pi, 12)
        yl, yu = self.asin(-0.5, 0.5, -math.pi / 2, math.pi / 2, 1e-8)
        self.assertAlmostEqual(yl, math.asin(-0.5))
        self.assertAlmostEqual(yu, math.asin(0.5))
        yl, yu = self.asin(-0.5, 0.5, -math.pi / 2 - 0.1, math.pi / 2 + 0.1, 1e-8)
        self.assertAlmostEqual(yl, math.asin(-0.5))
        self.assertAlmostEqual(yu, math.asin(0.5))
        yl, yu = self.asin(-0.5, 0.5, -math.pi / 2 + 0.1, math.pi / 2 - 0.1, 1e-8)
        self.assertAlmostEqual(yl, math.asin(-0.5))
        self.assertAlmostEqual(yu, math.asin(0.5))
        yl, yu = self.asin(-0.5, 0.5, -1.5 * math.pi, 1.5 * math.pi, 1e-8)
        self.assertAlmostEqual(yl, -3.6651914291880920, 12)
        self.assertAlmostEqual(yu, 3.6651914291880920, 12)
        yl, yu = self.asin(-0.5, 0.5, -1.5 * math.pi - 0.1, 1.5 * math.pi + 0.1, 1e-8)
        self.assertAlmostEqual(yl, -3.6651914291880920, 12)
        self.assertAlmostEqual(yu, 3.6651914291880920, 12)
        yl, yu = self.asin(-0.5, 0.5, -1.5 * math.pi + 0.1, 1.5 * math.pi - 0.1, 1e-8)
        self.assertAlmostEqual(yl, -3.6651914291880920, 12)
        self.assertAlmostEqual(yu, 3.6651914291880920, 12)

    def test_acos(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        yl, yu = self.acos(-0.5, 0.5, -interval.inf, interval.inf, 1e-8)
        self.assertEqual(yl, -interval.inf)
        self.assertEqual(yu, interval.inf)
        yl, yu = self.acos(-0.5, 0.5, -0.5 * math.pi, 0.5 * math.pi, 1e-8)
        self.assertAlmostEqual(yl, -0.5 * math.pi, 12)
        self.assertAlmostEqual(yu, 0.5 * math.pi, 12)
        yl, yu = self.acos(-0.5, 0.5, 0, math.pi, 1e-8)
        self.assertAlmostEqual(yl, math.acos(0.5))
        self.assertAlmostEqual(yu, math.acos(-0.5))
        yl, yu = self.acos(-0.5, 0.5, 0 - 0.1, math.pi + 0.1, 1e-8)
        self.assertAlmostEqual(yl, math.acos(0.5))
        self.assertAlmostEqual(yu, math.acos(-0.5))
        yl, yu = self.acos(-0.5, 0.5, 0 + 0.1, math.pi - 0.1, 1e-8)
        self.assertAlmostEqual(yl, math.acos(0.5))
        self.assertAlmostEqual(yu, math.acos(-0.5))
        yl, yu = self.acos(-0.5, 0.5, -math.pi, 0, 1e-8)
        self.assertAlmostEqual(yl, -math.acos(-0.5), 12)
        self.assertAlmostEqual(yu, -math.acos(0.5), 12)
        yl, yu = self.acos(-0.5, 0.5, -math.pi - 0.1, 0 + 0.1, 1e-8)
        self.assertAlmostEqual(yl, -math.acos(-0.5), 12)
        self.assertAlmostEqual(yu, -math.acos(0.5), 12)
        yl, yu = self.acos(-0.5, 0.5, -math.pi + 0.1, 0 - 0.1, 1e-8)
        self.assertAlmostEqual(yl, -math.acos(-0.5), 12)
        self.assertAlmostEqual(yu, -math.acos(0.5), 12)

    def test_atan(self):
        if not numpy_available:
            raise unittest.SkipTest('Numpy is not available.')
        yl, yu = self.atan(-0.5, 0.5, -interval.inf, interval.inf)
        self.assertEqual(yl, -interval.inf)
        self.assertEqual(yu, interval.inf)
        yl, yu = self.atan(-0.5, 0.5, -0.1, 0.1)
        self.assertAlmostEqual(yl, -0.1, 12)
        self.assertAlmostEqual(yu, 0.1, 12)
        yl, yu = self.atan(-0.5, 0.5, -0.5 * math.pi + 0.1, math.pi / 2 - 0.1)
        self.assertAlmostEqual(yl, math.atan(-0.5), 12)
        self.assertAlmostEqual(yu, math.atan(0.5), 12)
        yl, yu = self.atan(-0.5, 0.5, -1.5 * math.pi + 0.1, 1.5 * math.pi - 0.1)
        self.assertAlmostEqual(yl, math.atan(-0.5) - math.pi, 12)
        self.assertAlmostEqual(yu, math.atan(0.5) + math.pi, 12)

    def test_encountered_bugs(self):
        lb, ub = self._inverse_power1(
            88893.4225, 88893.4225, 2, 2, 298.15, 298.15, 1e-8
        )
        self.assertAlmostEqual(lb, 298.15)
        self.assertAlmostEqual(ub, 298.15)

        lb, ub = self._inverse_power1(2.56e-6, 2.56e-6, 2, 2, -0.0016, -0.0016, 1e-12)
        self.assertAlmostEqual(lb, -0.0016)
        self.assertAlmostEqual(ub, -0.0016)

        lb, ub = self._inverse_power1(
            -1, -1e-12, 2, 2, -interval.inf, interval.inf, 1e-8
        )
        self.assertAlmostEqual(lb, 0)
        self.assertAlmostEqual(ub, 0)

        lb, ub = self.mul(0, 0, -interval.inf, interval.inf)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)


class TestInterval(IntervalTestBase, unittest.TestCase):
    def setUp(self):
        super(TestInterval, self).setUp()
        self.add = interval.add
        self.sub = interval.sub
        self.mul = interval.mul
        self.inv = interval.inv
        self.div = interval.div
        self.power = interval.power
        self.exp = interval.exp
        self.log = interval.log
        self.log10 = interval.log10
        self.sin = interval.sin
        self.cos = interval.cos
        self.tan = interval.tan
        self.asin = interval.asin
        self.acos = interval.acos
        self.atan = interval.atan
        self._inverse_power1 = interval._inverse_power1
        self._inverse_power2 = interval._inverse_power2

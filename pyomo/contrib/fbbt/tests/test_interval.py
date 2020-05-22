import math
import pyutilib.th as unittest
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.errors import InfeasibleConstraintException
import pyomo.contrib.fbbt.interval as interval

try:
    isfinite = math.isfinite
except AttributeError:
    # isfinite() was added to math in Python 3.2
    def isfinite(x):
        return not (math.isnan(x) or math.isinf(x))

class TestInterval(unittest.TestCase):
    def setUp(self):
        if numpy_available:
            np.random.seed(0)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_add(self):
        xl = -2.5
        xu = 2.8
        yl = -3.2
        yu = 2.7
        zl, zu = interval.add(xl, xu, yl, yu)
        x = np.linspace(xl, xu, 100)
        y = np.linspace(yl, yu, 100)
        for _x in x:
            _z = _x + y
            self.assertTrue(np.all(zl <= _z))
            self.assertTrue(np.all(zu >= _z))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_sub(self):
        xl = -2.5
        xu = 2.8
        yl = -3.2
        yu = 2.7
        zl, zu = interval.sub(xl, xu, yl, yu)
        x = np.linspace(xl, xu, 100)
        y = np.linspace(yl, yu, 100)
        for _x in x:
            _z = _x - y
            self.assertTrue(np.all(zl <= _z))
            self.assertTrue(np.all(zu >= _z))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_mul(self):
        xl = -2.5
        xu = 2.8
        yl = -3.2
        yu = 2.7
        zl, zu = interval.mul(xl, xu, yl, yu)
        x = np.linspace(xl, xu, 100)
        y = np.linspace(yl, yu, 100)
        for _x in x:
            _z = _x * y
            self.assertTrue(np.all(zl <= _z))
            self.assertTrue(np.all(zu >= _z))

    def test_inv(self):
        lb, ub = interval.inv(0.1, 0.2, feasibility_tol=1e-8)
        self.assertAlmostEqual(lb, 5)
        self.assertAlmostEqual(ub, 10)

        lb, ub = interval.inv(0, 0.1, feasibility_tol=1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = interval.inv(0, 0, feasibility_tol=1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = interval.inv(-0.1, 0, feasibility_tol=1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = interval.inv(-0.2, -0.1, feasibility_tol=1e-8)
        self.assertAlmostEqual(lb, -10)
        self.assertAlmostEqual(ub, -5)

        lb, ub = interval.inv(0, -1e-16, feasibility_tol=1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertEqual(ub, interval.inf)

        lb, ub = interval.inv(1e-16, 0, feasibility_tol=1e-8)
        self.assertEqual(lb, -interval.inf)
        self.assertAlmostEqual(ub, interval.inf)

        lb, ub = interval.inv(-1, 1, feasibility_tol=1e-8)
        self.assertAlmostEqual(lb, -interval.inf)
        self.assertAlmostEqual(ub, interval.inf)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_div(self):
        x_bounds = [(np.random.uniform(-5, -2), np.random.uniform(2, 5))]
        y_bounds = [(np.random.uniform(-5, -2), np.random.uniform(2, 5)),
                    (0, np.random.uniform(2, 5)),
                    (np.random.uniform(0, 2), np.random.uniform(2, 5)),
                    (np.random.uniform(-5, -2), 0),
                    (np.random.uniform(-5, -2), np.random.uniform(-2, 0))]
        for xl, xu in x_bounds:
            for yl, yu in y_bounds:
                zl, zu = interval.div(xl, xu, yl, yu, feasibility_tol=1e-8)
                x = np.linspace(xl, xu, 100)
                y = np.linspace(yl, yu, 100)
                for _x in x:
                    _z = _x / y
                    self.assertTrue(np.all(zl <= _z))
                    self.assertTrue(np.all(zu >= _z))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_pow(self):
        x_bounds = [(np.random.uniform(0, 2), np.random.uniform(2, 5)),
                    (0, np.random.uniform(2, 5)),
                    (0, 0)]
        y_bounds = [(np.random.uniform(-5, -2), np.random.uniform(2, 5)),
                    (0, np.random.uniform(2, 5)),
                    (np.random.uniform(0, 2), np.random.uniform(2, 5)),
                    (np.random.uniform(-5, -2), 0),
                    (np.random.uniform(-5, -2), np.random.uniform(-2, 0))]
        for xl, xu in x_bounds:
            for yl, yu in y_bounds:
                zl, zu = interval.power(xl, xu, yl, yu)
                x = np.linspace(xl, xu, 100)
                y = np.linspace(yl, yu, 100)
                for _x in x:
                    _z = _x ** y
                    self.assertTrue(np.all(zl <= _z))
                    self.assertTrue(np.all(zu >= _z))

        x_bounds = [(np.random.uniform(-5, -2), np.random.uniform(2, 5)),
                    (np.random.uniform(-5, -2), np.random.uniform(-2, 0)),
                    (np.random.uniform(-5, -2), 0)]
        y_bounds = list(range(-4, 4))
        for xl, xu in x_bounds:
            for yl in y_bounds:
                yu = yl
                zl, zu = interval.power(xl, xu, yl, yu)
                x = np.linspace(xl, xu, 100, endpoint=False)
                y = yl
                _z = x ** y
                self.assertTrue(np.all(zl <= _z))
                self.assertTrue(np.all(zu >= _z))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_pow2(self):
        xl = np.linspace(-2, 2, 17)
        xu = np.linspace(-2, 2, 17)
        yl = np.linspace(-2, 2, 17)
        yu = np.linspace(-2, 2, 17)
        for _xl in xl:
            for _xu in xu:
                if _xl > _xu:
                    continue
                for _yl in yl:
                    for _yu in yu:
                        if _yl > _yu:
                            continue
                        if _yl == _yu and _yl != round(_yl) and _xu < 0:
                            with self.assertRaises(InfeasibleConstraintException):
                                lb, ub = interval.power(_xl, _xu, _yl, _yu)
                        else:
                            lb, ub = interval.power(_xl, _xu, _yl, _yu)
                            if isfinite(lb) and isfinite(ub):
                                nan_fill = 0.5*(lb + ub)
                            elif isfinite(lb):
                                nan_fill = lb + 1
                            elif isfinite(ub):
                                nan_fill = ub - 1
                            else:
                                nan_fill = 0
                            x = np.linspace(_xl, _xu, 17)
                            y = np.linspace(_yl, _yu, 17)
                            all_values = list()
                            for _x in x:
                                z = _x**y
                                #np.nan_to_num(z, copy=False, nan=nan_fill, posinf=np.inf, neginf=-np.inf)
                                tmp = []
                                for _z in z:
                                    if math.isnan(_z):
                                        tmp.append(nan_fill)
                                    else:
                                        tmp.append(_z)
                                all_values.append(np.array(tmp))
                            all_values = np.array(all_values)
                            estimated_lb = all_values.min()
                            estimated_ub = all_values.max()
                            self.assertTrue(lb - 1e-8 <= estimated_lb)
                            self.assertTrue(ub + 1e-8 >= estimated_ub)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_exp(self):
        xl = -2.5
        xu = 2.8
        zl, zu = interval.exp(xl, xu)
        x = np.linspace(xl, xu, 100)
        _z = np.exp(x)
        self.assertTrue(np.all(zl <= _z))
        self.assertTrue(np.all(zu >= _z))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_log(self):
        x_bounds = [(np.random.uniform(0, 2), np.random.uniform(2, 5)),
                    (0, np.random.uniform(2, 5)),
                    (0, 0)]
        for xl, xu in x_bounds:
            zl, zu = interval.log(xl, xu)
            x = np.linspace(xl, xu, 100)
            _z = np.log(x)
            self.assertTrue(np.all(zl <= _z))
            self.assertTrue(np.all(zu >= _z))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_cos(self):
        lbs = np.linspace(-2*math.pi, 2*math.pi, 10)
        ubs = np.linspace(-2*math.pi, 2*math.pi, 10)
        for xl in lbs:
            for xu in ubs:
                if xu >= xl:
                    zl, zu = interval.cos(xl, xu)
                    x = np.linspace(xl, xu, 100)
                    _z = np.cos(x)
                    self.assertTrue(np.all(zl <= _z))
                    self.assertTrue(np.all(zu >= _z))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_sin(self):
        lbs = np.linspace(-2*math.pi, 2*math.pi, 10)
        ubs = np.linspace(-2*math.pi, 2*math.pi, 10)
        for xl in lbs:
            for xu in ubs:
                if xu >= xl:
                    zl, zu = interval.sin(xl, xu)
                    x = np.linspace(xl, xu, 100)
                    _z = np.sin(x)
                    self.assertTrue(np.all(zl <= _z))
                    self.assertTrue(np.all(zu >= _z))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_tan(self):
        lbs = np.linspace(-2*math.pi, 2*math.pi, 10)
        ubs = np.linspace(-2*math.pi, 2*math.pi, 10)
        for xl in lbs:
            for xu in ubs:
                if xu >= xl:
                    zl, zu = interval.tan(xl, xu)
                    x = np.linspace(xl, xu, 100)
                    _z = np.tan(x)
                    self.assertTrue(np.all(zl <= _z))
                    self.assertTrue(np.all(zu >= _z))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_asin(self):
        yl, yu = interval.asin(-0.5, 0.5, -interval.inf, interval.inf, feasibility_tol=1e-8)
        self.assertEqual(yl, -interval.inf)
        self.assertEqual(yu, interval.inf)
        yl, yu = interval.asin(-0.5, 0.5, -math.pi, math.pi, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, -math.pi, 12)
        self.assertAlmostEqual(yu, math.pi, 12)
        yl, yu = interval.asin(-0.5, 0.5, -math.pi/2, math.pi/2, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, math.asin(-0.5))
        self.assertAlmostEqual(yu, math.asin(0.5))
        yl, yu = interval.asin(-0.5, 0.5, -math.pi/2-0.1, math.pi/2+0.1, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, math.asin(-0.5))
        self.assertAlmostEqual(yu, math.asin(0.5))
        yl, yu = interval.asin(-0.5, 0.5, -math.pi/2+0.1, math.pi/2-0.1, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, math.asin(-0.5))
        self.assertAlmostEqual(yu, math.asin(0.5))
        yl, yu = interval.asin(-0.5, 0.5, -1.5*math.pi, 1.5*math.pi, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, -3.6651914291880920, 12)
        self.assertAlmostEqual(yu, 3.6651914291880920, 12)
        yl, yu = interval.asin(-0.5, 0.5, -1.5*math.pi-0.1, 1.5*math.pi+0.1, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, -3.6651914291880920, 12)
        self.assertAlmostEqual(yu, 3.6651914291880920, 12)
        yl, yu = interval.asin(-0.5, 0.5, -1.5*math.pi+0.1, 1.5*math.pi-0.1, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, -3.6651914291880920, 12)
        self.assertAlmostEqual(yu, 3.6651914291880920, 12)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_acos(self):
        yl, yu = interval.acos(-0.5, 0.5, -interval.inf, interval.inf, feasibility_tol=1e-8)
        self.assertEqual(yl, -interval.inf)
        self.assertEqual(yu, interval.inf)
        yl, yu = interval.acos(-0.5, 0.5, -0.5*math.pi, 0.5*math.pi, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, -0.5*math.pi, 12)
        self.assertAlmostEqual(yu, 0.5*math.pi, 12)
        yl, yu = interval.acos(-0.5, 0.5, 0, math.pi, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, math.acos(0.5))
        self.assertAlmostEqual(yu, math.acos(-0.5))
        yl, yu = interval.acos(-0.5, 0.5, 0-0.1, math.pi+0.1, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, math.acos(0.5))
        self.assertAlmostEqual(yu, math.acos(-0.5))
        yl, yu = interval.acos(-0.5, 0.5, 0+0.1, math.pi-0.1, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, math.acos(0.5))
        self.assertAlmostEqual(yu, math.acos(-0.5))
        yl, yu = interval.acos(-0.5, 0.5, -math.pi, 0, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, -math.acos(-0.5), 12)
        self.assertAlmostEqual(yu, -math.acos(0.5), 12)
        yl, yu = interval.acos(-0.5, 0.5, -math.pi-0.1, 0+0.1, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, -math.acos(-0.5), 12)
        self.assertAlmostEqual(yu, -math.acos(0.5), 12)
        yl, yu = interval.acos(-0.5, 0.5, -math.pi+0.1, 0-0.1, feasibility_tol=1e-8)
        self.assertAlmostEqual(yl, -math.acos(-0.5), 12)
        self.assertAlmostEqual(yu, -math.acos(0.5), 12)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_atan(self):
        yl, yu = interval.atan(-0.5, 0.5, -interval.inf, interval.inf)
        self.assertEqual(yl, -interval.inf)
        self.assertEqual(yu, interval.inf)
        yl, yu = interval.atan(-0.5, 0.5, -0.1, 0.1)
        self.assertAlmostEqual(yl, -0.1, 12)
        self.assertAlmostEqual(yu, 0.1, 12)
        yl, yu = interval.atan(-0.5, 0.5, -0.5*math.pi+0.1, math.pi/2-0.1)
        self.assertAlmostEqual(yl, math.atan(-0.5), 12)
        self.assertAlmostEqual(yu, math.atan(0.5), 12)
        yl, yu = interval.atan(-0.5, 0.5, -1.5*math.pi+0.1, 1.5*math.pi-0.1)
        self.assertAlmostEqual(yl, math.atan(-0.5)-math.pi, 12)
        self.assertAlmostEqual(yu, math.atan(0.5)+math.pi, 12)

    def test_encountered_bugs(self):
        lb, ub = interval._inverse_power1(88893.4225, 88893.4225, 2, 2, 298.15, 298.15, feasibility_tol=1e-8)
        self.assertAlmostEqual(lb, 298.15)
        self.assertAlmostEqual(ub, 298.15)

        lb, ub = interval._inverse_power1(2.56e-6, 2.56e-6, 2, 2, -0.0016, -0.0016, 1e-12)
        self.assertAlmostEqual(lb, -0.0016)
        self.assertAlmostEqual(ub, -0.0016)

        lb, ub = interval._inverse_power1(-1, -1e-12, 2, 2, -interval.inf, interval.inf, feasibility_tol=1e-8)
        self.assertAlmostEqual(lb, 0)
        self.assertAlmostEqual(ub, 0)

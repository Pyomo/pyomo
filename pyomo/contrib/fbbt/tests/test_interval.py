import pyutilib.th as unittest
import math
import pyomo.contrib.fbbt.interval as interval
try:
    import numpy as np
    numpy_available = True
    np.random.seed(0)
except ImportError:
    numpy_available = False


class TestInterval(unittest.TestCase):
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
        yl, yu = interval.asin(-0.5, 0.5, -interval.inf, interval.inf)
        self.assertEqual(yl, -interval.inf)
        self.assertEqual(yu, interval.inf)
        yl, yu = interval.asin(-0.5, 0.5, -math.pi, math.pi)
        self.assertAlmostEqual(yl, -math.pi, 12)
        self.assertAlmostEqual(yu, math.pi, 12)
        yl, yu = interval.asin(-0.5, 0.5, -math.pi/2, math.pi/2)
        self.assertAlmostEqual(yl, math.asin(-0.5))
        self.assertAlmostEqual(yu, math.asin(0.5))
        yl, yu = interval.asin(-0.5, 0.5, -math.pi/2-0.1, math.pi/2+0.1)
        self.assertAlmostEqual(yl, math.asin(-0.5))
        self.assertAlmostEqual(yu, math.asin(0.5))
        yl, yu = interval.asin(-0.5, 0.5, -math.pi/2+0.1, math.pi/2-0.1)
        self.assertAlmostEqual(yl, math.asin(-0.5))
        self.assertAlmostEqual(yu, math.asin(0.5))
        yl, yu = interval.asin(-0.5, 0.5, -1.5*math.pi, 1.5*math.pi)
        self.assertAlmostEqual(yl, -3.6651914291880920, 12)
        self.assertAlmostEqual(yu, 3.6651914291880920, 12)
        yl, yu = interval.asin(-0.5, 0.5, -1.5*math.pi-0.1, 1.5*math.pi+0.1)
        self.assertAlmostEqual(yl, -3.6651914291880920, 12)
        self.assertAlmostEqual(yu, 3.6651914291880920, 12)
        yl, yu = interval.asin(-0.5, 0.5, -1.5*math.pi+0.1, 1.5*math.pi-0.1)
        self.assertAlmostEqual(yl, -3.6651914291880920, 12)
        self.assertAlmostEqual(yu, 3.6651914291880920, 12)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_acos(self):
        yl, yu = interval.acos(-0.5, 0.5, -interval.inf, interval.inf)
        self.assertEqual(yl, -interval.inf)
        self.assertEqual(yu, interval.inf)
        yl, yu = interval.acos(-0.5, 0.5, -0.5*math.pi, 0.5*math.pi)
        self.assertAlmostEqual(yl, -0.5*math.pi, 12)
        self.assertAlmostEqual(yu, 0.5*math.pi, 12)
        yl, yu = interval.acos(-0.5, 0.5, 0, math.pi)
        self.assertAlmostEqual(yl, math.acos(0.5))
        self.assertAlmostEqual(yu, math.acos(-0.5))
        yl, yu = interval.acos(-0.5, 0.5, 0-0.1, math.pi+0.1)
        self.assertAlmostEqual(yl, math.acos(0.5))
        self.assertAlmostEqual(yu, math.acos(-0.5))
        yl, yu = interval.acos(-0.5, 0.5, 0+0.1, math.pi-0.1)
        self.assertAlmostEqual(yl, math.acos(0.5))
        self.assertAlmostEqual(yu, math.acos(-0.5))
        yl, yu = interval.acos(-0.5, 0.5, -math.pi, 0)
        self.assertAlmostEqual(yl, -math.acos(-0.5), 12)
        self.assertAlmostEqual(yu, -math.acos(0.5), 12)
        yl, yu = interval.acos(-0.5, 0.5, -math.pi-0.1, 0+0.1)
        self.assertAlmostEqual(yl, -math.acos(-0.5), 12)
        self.assertAlmostEqual(yu, -math.acos(0.5), 12)
        yl, yu = interval.acos(-0.5, 0.5, -math.pi+0.1, 0-0.1)
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

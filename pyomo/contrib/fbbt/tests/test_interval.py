import unittest
import pyomo.contrib.fbbt.interval as interval
try:
    import numpy as np
    numpy_available = True
    np.random.seed(0)
except ImportError:
    numpy_available = False


class TestInterval(unittest.TestCase):
    def test_add(self):
        if not numpy_available:
            raise unittest.SkipTest
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

    def test_sub(self):
        if not numpy_available:
            raise unittest.SkipTest
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

    def test_mul(self):
        if not numpy_available:
            raise unittest.SkipTest
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

    def test_div(self):
        if not numpy_available:
            raise unittest.SkipTest
        x_bounds = [(np.random.uniform(-5, -2), np.random.uniform(2, 5))]
        y_bounds = [(np.random.uniform(-5, -2), np.random.uniform(2, 5)),
                    (0, np.random.uniform(2, 5)),
                    (np.random.uniform(0, 2), np.random.uniform(2, 5)),
                    (np.random.uniform(-5, -2), 0),
                    (np.random.uniform(-5, -2), np.random.uniform(-2, 0))]
        for xl, xu in x_bounds:
            for yl, yu in y_bounds:
                zl, zu = interval.div(xl, xu, yl, yu)
                x = np.linspace(xl, xu, 100)
                y = np.linspace(yl, yu, 100)
                for _x in x:
                    _z = _x / y
                    self.assertTrue(np.all(zl <= _z))
                    self.assertTrue(np.all(zu >= _z))

    def test_pow(self):
        if not numpy_available:
            raise unittest.SkipTest
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

    def test_exp(self):
        if not numpy_available:
            raise unittest.SkipTest
        xl = -2.5
        xu = 2.8
        zl, zu = interval.exp(xl, xu)
        x = np.linspace(xl, xu, 100)
        _z = np.exp(x)
        self.assertTrue(np.all(zl <= _z))
        self.assertTrue(np.all(zu >= _z))

    def test_log(self):
        if not numpy_available:
            raise unittest.SkipTest
        x_bounds = [(np.random.uniform(0, 2), np.random.uniform(2, 5)),
                    (0, np.random.uniform(2, 5)),
                    (0, 0)]
        for xl, xu in x_bounds:
            zl, zu = interval.log(xl, xu)
            x = np.linspace(xl, xu, 100)
            _z = np.log(x)
            self.assertTrue(np.all(zl <= _z))
            self.assertTrue(np.all(zu >= _z))

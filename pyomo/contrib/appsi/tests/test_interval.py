from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
import pyomo.common.unittest as unittest
import math
from pyomo.contrib.fbbt.tests.test_interval import IntervalTestBase


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestInterval(IntervalTestBase, unittest.TestCase):
    def setUp(self):
        super(TestInterval, self).setUp()
        self.add = cmodel.py_interval_add
        self.sub = cmodel.py_interval_sub
        self.mul = cmodel.py_interval_mul
        self.inv = cmodel.py_interval_inv
        self.div = cmodel.py_interval_div
        self.power = cmodel.py_interval_power
        self.exp = cmodel.py_interval_exp
        self.log = cmodel.py_interval_log
        self.log10 = cmodel.py_interval_log10
        self.sin = cmodel.py_interval_sin
        self.cos = cmodel.py_interval_cos
        self.tan = cmodel.py_interval_tan
        self.asin = cmodel.py_interval_asin
        self.acos = cmodel.py_interval_acos
        self.atan = cmodel.py_interval_atan
        self._inverse_power1 = cmodel._py_inverse_power1
        self._inverse_power2 = cmodel._py_inverse_power2


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestCInterval(unittest.TestCase):
    def test_pow_with_inf(self):
        x_list = [
            0,
            -math.inf,
            math.inf,
            -3,
            3,
            -2,
            2,
            -1,
            1,
            -2.5,
            2.5,
            -0.5,
            0.5,
            -1.5,
            1.5,
        ]
        y_list = list(x_list)
        for x in x_list:
            for y in y_list:
                try:
                    expected = x**y
                    if type(expected) is complex:
                        expect_error = True
                        expected = None
                    else:
                        expect_error = False
                except ZeroDivisionError as e:
                    expected = None
                    expect_error = True
                if expect_error:
                    with self.assertRaises(ValueError):
                        cmodel._pow_with_inf(x, y)
                else:
                    got = cmodel._pow_with_inf(x, y)
                    self.assertAlmostEqual(expected, got)

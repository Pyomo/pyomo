from pyomo.contrib.fbbt import interval
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
import pyomo.common.unittest as unittest
import math
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.errors import InfeasibleConstraintException


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestInterval(unittest.TestCase):
    def setUp(self) -> None:
        if numpy_available:
            np.random.seed(0)

    def assertAlmostEqual(self, first, second, **kwargs):
        if math.isfinite(first) and math.isfinite(second):
            super(TestInterval, self).assertAlmostEqual(first, second, **kwargs)
        elif first == -math.inf:
            self.assertEqual(second, -cmodel.inf)
        elif first == math.inf:
            self.assertEqual(second, cmodel.inf)
        elif second == -math.inf:
            self.assertEqual(first, -cmodel.inf)
        else:
            self.assertEqual(first, cmodel.inf)

    def test_add(self):
        xl = -2.5
        xu = 2.8
        yl = -3.2
        yu = 2.7
        zl1, zu1 = interval.add(xl, xu, yl, yu)
        zl2, zu2 = cmodel.py_interval_add(xl, xu, yl, yu)
        self.assertAlmostEqual(zl1, zl2)
        self.assertAlmostEqual(zu1, zu2)

    def test_sub(self):
        xl = -2.5
        xu = 2.8
        yl = -3.2
        yu = 2.7
        zl1, zu1 = interval.sub(xl, xu, yl, yu)
        zl2, zu2 = cmodel.py_interval_sub(xl, xu, yl, yu)
        self.assertAlmostEqual(zl1, zl2)
        self.assertAlmostEqual(zu1, zu2)

    def test_mul(self):
        xl = -2.5
        xu = 2.8
        yl = -3.2
        yu = 2.7
        zl1, zu1 = interval.mul(xl, xu, yl, yu)
        zl2, zu2 = cmodel.py_interval_mul(xl, xu, yl, yu)
        self.assertAlmostEqual(zl1, zl2)
        self.assertAlmostEqual(zu1, zu2)

    def test_inv(self):
        lb1, ub1 = interval.inv(0.1, 0.2, feasibility_tol=1e-8)
        lb2, ub2 = cmodel.py_interval_inv(0.1, 0.2, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.inv(0, 0.1, feasibility_tol=1e-8)
        lb2, ub2 = cmodel.py_interval_inv(0, 0.1, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.inv(0, 0, feasibility_tol=1e-8)
        lb2, ub2 = cmodel.py_interval_inv(0, 0, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.inv(-0.1, 0, feasibility_tol=1e-8)
        lb2, ub2 = cmodel.py_interval_inv(-0.1, 0, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.inv(-0.2, -0.1, feasibility_tol=1e-8)
        lb2, ub2 = cmodel.py_interval_inv(-0.2, -0.1, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.inv(0, -1e-16, feasibility_tol=1e-8)
        lb2, ub2 = cmodel.py_interval_inv(0, -1e-16, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.inv(1e-16, 0, feasibility_tol=1e-8)
        lb2, ub2 = cmodel.py_interval_inv(1e-16, 0, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.inv(-1, 1, feasibility_tol=1e-8)
        lb2, ub2 = cmodel.py_interval_inv(-1, 1, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

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
                zl1, zu1 = interval.div(xl, xu, yl, yu, feasibility_tol=1e-8)
                zl2, zu2 = cmodel.py_interval_div(xl, xu, yl, yu, 1e-8)
                self.assertAlmostEqual(zl1, zl2)
                self.assertAlmostEqual(zu1, zu2)

    def test_div_edge_cases(self):
        lb1, ub1 = interval.div(0, -1e-16, 0, 0, 1e-8)
        lb2, ub2 = cmodel.py_interval_div(0, -1e-16, 0, 0, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.div(0, 1e-16, 0, 0, 1e-8)
        lb2, ub2 = cmodel.py_interval_div(0, 1e-16, 0, 0, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.div(-1e-16, 0, 0, 0, 1e-8)
        lb2, ub2 = cmodel.py_interval_div(-1e-16, 0, 0, 0, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

        lb1, ub1 = interval.div(1e-16, 0, 0, 0, 1e-8)
        lb2, ub2 = cmodel.py_interval_div(1e-16, 0, 0, 0, 1e-8)
        self.assertAlmostEqual(lb1, lb2)
        self.assertAlmostEqual(ub1, ub2)

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
                zl1, zu1 = interval.power(xl, xu, yl, yu, feasibility_tol=1e-8)
                zl2, zu2 = cmodel.py_interval_power(xl, xu, yl, yu, 1e-8)
                self.assertAlmostEqual(zl1, zl2)
                self.assertAlmostEqual(zu1, zu2)

        x_bounds = [(np.random.uniform(-5, -2), np.random.uniform(2, 5)),
                    (np.random.uniform(-5, -2), np.random.uniform(-2, 0)),
                    (np.random.uniform(-5, -2), 0)]
        y_bounds = list(range(-4, 4))
        for xl, xu in x_bounds:
            for yl in y_bounds:
                yu = yl
                zl1, zu1 = interval.power(xl, xu, yl, yu, feasibility_tol=1e-8)
                zl2, zu2 = cmodel.py_interval_power(xl, xu, yl, yu, 1e-8)
                self.assertAlmostEqual(zl1, zl2)
                self.assertAlmostEqual(zu1, zu2)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_pow2(self):
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
                            lb1, ub1 = interval.power(_xl, _xu, _yl, _yu, feasibility_tol=1e-8)
                            lb2, ub2 = cmodel.py_interval_power(_xl, _xu, _yl, _yu, 1e-8)
                            self.assertAlmostEqual(lb1, lb2)
                            self.assertAlmostEqual(ub1, ub2)
                        elif _yl == _yu and _yl != round(_yl) and (_xu < 0 or (_xu < 0 and _yu < 0)):
                            with self.assertRaises((InfeasibleConstraintException, interval.IntervalException)):
                                lb, ub = interval.power(_xl, _xu, _yl, _yu, feasibility_tol=1e-8)
                            with self.assertRaises(ValueError):
                                lb, ub = cmodel.py_interval_power(_xl, _xu, _yl, _yu, 1e-8)
                        else:
                            lb1, ub1 = interval.power(_xl, _xu, _yl, _yu, feasibility_tol=1e-8)
                            lb2, ub2 = cmodel.py_interval_power(_xl, _xu, _yl, _yu, 1e-8)
                            self.assertAlmostEqual(lb1, lb2)
                            self.assertAlmostEqual(ub1, ub2)

    def test_exp(self):
        xl = -2.5
        xu = 2.8
        zl1, zu1 = interval.exp(xl, xu)
        zl2, zu2 = cmodel.py_interval_exp(xl, xu)
        self.assertAlmostEqual(zl1, zl2)
        self.assertAlmostEqual(zu1, zu2)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_log(self):
        x_bounds = [(np.random.uniform(0, 2), np.random.uniform(2, 5)),
                    (0, np.random.uniform(2, 5)),
                    (0, 0)]
        for xl, xu in x_bounds:
            zl1, zu1 = interval.log(xl, xu)
            zl2, zu2 = cmodel.py_interval_log(xl, xu)
            self.assertAlmostEqual(zl1, zl2)
            self.assertAlmostEqual(zu1, zu2)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_cos(self):
        lbs = np.linspace(-2*math.pi, 2*math.pi, 10)
        ubs = np.linspace(-2*math.pi, 2*math.pi, 10)
        for xl in lbs:
            for xu in ubs:
                if xu >= xl:
                    zl1, zu1 = interval.cos(xl, xu)
                    zl2, zu2 = cmodel.py_interval_cos(xl, xu)
                    self.assertAlmostEqual(zl1, zl2)
                    self.assertAlmostEqual(zu1, zu2)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_sin(self):
        lbs = np.linspace(-2*math.pi, 2*math.pi, 10)
        ubs = np.linspace(-2*math.pi, 2*math.pi, 10)
        for xl in lbs:
            for xu in ubs:
                if xu >= xl:
                    zl1, zu1 = interval.sin(xl, xu)
                    zl2, zu2 = cmodel.py_interval_sin(xl, xu)
                    self.assertAlmostEqual(zl1, zl2)
                    self.assertAlmostEqual(zu1, zu2)

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_tan(self):
        lbs = np.linspace(-2*math.pi, 2*math.pi, 10)
        ubs = np.linspace(-2*math.pi, 2*math.pi, 10)
        for xl in lbs:
            for xu in ubs:
                if xu >= xl:
                    zl1, zu1 = interval.tan(xl, xu)
                    zl2, zu2 = cmodel.py_interval_tan(xl, xu)
                    self.assertAlmostEqual(zl1, zl2)
                    self.assertAlmostEqual(zu1, zu2)

    def test_asin(self):
        yl1, yu1 = interval.asin(-0.5, 0.5, -interval.inf, interval.inf, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_asin(-0.5, 0.5, -cmodel.inf, cmodel.inf, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.asin(-0.5, 0.5, -math.pi, math.pi, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_asin(-0.5, 0.5, -math.pi, math.pi, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.asin(-0.5, 0.5, -math.pi/2, math.pi/2, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_asin(-0.5, 0.5, -math.pi/2, math.pi/2, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.asin(-0.5, 0.5, -math.pi/2-0.1, math.pi/2+0.1, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_asin(-0.5, 0.5, -math.pi/2-0.1, math.pi/2+0.1, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.asin(-0.5, 0.5, -math.pi/2+0.1, math.pi/2-0.1, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_asin(-0.5, 0.5, -math.pi/2+0.1, math.pi/2-0.1, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.asin(-0.5, 0.5, -1.5*math.pi, 1.5*math.pi, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_asin(-0.5, 0.5, -1.5*math.pi, 1.5*math.pi, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.asin(-0.5, 0.5, -1.5*math.pi-0.1, 1.5*math.pi+0.1, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_asin(-0.5, 0.5, -1.5*math.pi-0.1, 1.5*math.pi+0.1, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.asin(-0.5, 0.5, -1.5*math.pi+0.1, 1.5*math.pi-0.1, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_asin(-0.5, 0.5, -1.5*math.pi+0.1, 1.5*math.pi-0.1, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

    def test_acos(self):
        yl1, yu1 = interval.acos(-0.5, 0.5, -interval.inf, interval.inf, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_acos(-0.5, 0.5, -cmodel.inf, cmodel.inf, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.acos(-0.5, 0.5, -0.5*math.pi, 0.5*math.pi, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_acos(-0.5, 0.5, -0.5*math.pi, 0.5*math.pi, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.acos(-0.5, 0.5, 0, math.pi, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_acos(-0.5, 0.5, 0, math.pi, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.acos(-0.5, 0.5, 0-0.1, math.pi+0.1, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_acos(-0.5, 0.5, 0-0.1, math.pi+0.1, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.acos(-0.5, 0.5, 0+0.1, math.pi-0.1, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_acos(-0.5, 0.5, 0+0.1, math.pi-0.1, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.acos(-0.5, 0.5, -math.pi, 0, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_acos(-0.5, 0.5, -math.pi, 0, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.acos(-0.5, 0.5, -math.pi-0.1, 0+0.1, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_acos(-0.5, 0.5, -math.pi-0.1, 0+0.1, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.acos(-0.5, 0.5, -math.pi+0.1, 0-0.1, feasibility_tol=1e-8)
        yl2, yu2 = cmodel.py_interval_acos(-0.5, 0.5, -math.pi+0.1, 0-0.1, 1e-8)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

    def test_atan(self):
        yl1, yu1 = interval.atan(-0.5, 0.5, -interval.inf, interval.inf)
        yl2, yu2 = cmodel.py_interval_atan(-0.5, 0.5, -interval.inf, interval.inf)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.atan(-0.5, 0.5, -0.1, 0.1)
        yl2, yu2 = cmodel.py_interval_atan(-0.5, 0.5, -0.1, 0.1)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.atan(-0.5, 0.5, -0.5*math.pi+0.1, math.pi/2-0.1)
        yl2, yu2 = cmodel.py_interval_atan(-0.5, 0.5, -0.5*math.pi+0.1, math.pi/2-0.1)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

        yl1, yu1 = interval.atan(-0.5, 0.5, -1.5*math.pi+0.1, 1.5*math.pi-0.1)
        yl2, yu2 = cmodel.py_interval_atan(-0.5, 0.5, -1.5*math.pi+0.1, 1.5*math.pi-0.1)
        self.assertAlmostEqual(yl1, yl2)
        self.assertAlmostEqual(yu1, yu2)

    def test_encountered_bugs(self):
        lb1, ub1 = interval._inverse_power1(88893.4225, 88893.4225, 2, 2, 298.15, 298.15, feasibility_tol=1e-8)
        lb2, ub2 = cmodel._py_inverse_power1(88893.4225, 88893.4225, 2, 2, 298.15, 298.15, 1e-8)

        lb1, ub1 = interval._inverse_power1(2.56e-6, 2.56e-6, 2, 2, -0.0016, -0.0016, 1e-12)
        lb2, ub2 = cmodel._py_inverse_power1(2.56e-6, 2.56e-6, 2, 2, -0.0016, -0.0016, 1e-12)

        lb1, ub1 = interval._inverse_power1(-1, -1e-12, 2, 2, -interval.inf, interval.inf, feasibility_tol=1e-8)
        lb2, ub2 = cmodel._py_inverse_power1(-1, -1e-12, 2, 2, -interval.inf, interval.inf, 1e-8)

        lb1, ub1 = interval.mul(0, 0, -interval.inf, interval.inf)
        lb2, ub2 = cmodel.py_interval_mul(0, 0, -interval.inf, interval.inf)

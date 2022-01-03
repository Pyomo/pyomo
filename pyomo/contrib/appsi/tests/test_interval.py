from pyomo.contrib.fbbt import interval
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
import pyomo.common.unittest as unittest
import math
from pyomo.common.dependencies import numpy as np, numpy_available


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

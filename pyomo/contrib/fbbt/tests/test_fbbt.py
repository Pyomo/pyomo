import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
from pyomo.core.expr.numeric_expr import ProductExpression, UnaryFunctionExpression
import math
import logging
import io
try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False


class DummyExpr(ProductExpression):
    pass


class TestFBBT(unittest.TestCase):
    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_add(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.p = pe.Param(mutable=True)
                m.p.value = 1
                m.c = pe.Constraint(expr=pe.inequality(body=m.x+m.y+(m.p+1), lower=cl, upper=cu))
                new_bounds = fbbt(m)
                self.assertEqual(new_bounds[m.x], (pe.value(m.x.lb), pe.value(m.x.ub)))
                self.assertEqual(new_bounds[m.y], (pe.value(m.y.lb), pe.value(m.y.ub)))
                x = np.linspace(pe.value(m.x.lb), pe.value(m.x.ub), 100)
                z = np.linspace(pe.value(m.c.lower), pe.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = z - _x - m.p.value - 1
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_sub1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.x-m.y, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pe.value(m.x.lb), pe.value(m.x.ub), 100)
                z = np.linspace(pe.value(m.c.lower), pe.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = _x - z
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_sub2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.y-m.x, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pe.value(m.x.lb), pe.value(m.x.ub), 100)
                z = np.linspace(pe.value(m.c.lower), pe.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = z + _x
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_mul(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.x*m.y, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pe.value(m.x.lb) + 1e-6, pe.value(m.x.ub), 100, endpoint=False)
                z = np.linspace(pe.value(m.c.lower), pe.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = z / _x
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_div1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.x/m.y, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pe.value(m.x.lb), pe.value(m.x.ub), 100)
                z = np.linspace(pe.value(m.c.lower) + 1e-6, pe.value(m.c.upper), 100, endpoint=False)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = _x / z
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_div2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.y/m.x, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pe.value(m.x.lb), pe.value(m.x.ub), 100)
                z = np.linspace(pe.value(m.c.lower), pe.value(m.c.upper), 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = _x * z
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_pow1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.x**m.y, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pe.value(m.x.lb) + 1e-6, pe.value(m.x.ub), 100, endpoint=False)
                z = np.linspace(pe.value(m.c.lower) + 1e-6, pe.value(m.c.upper), 100, endpoint=False)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = np.log(abs(z)) / np.log(abs(_x))
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_pow2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.y**m.x, lower=cl, upper=cu))
                fbbt(m)
                x = np.linspace(pe.value(m.x.lb) + 1e-6, pe.value(m.x.ub), 100, endpoint=False)
                z = np.linspace(pe.value(m.c.lower) + 1e-6, pe.value(m.c.upper), 100, endpoint=False)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = np.exp(np.log(abs(z)) / _x)
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))
                    _y = - _y
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_exp(self):
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pe.Block(concrete=True)
            m.x = pe.Var()
            m.c = pe.Constraint(expr=pe.inequality(body=pe.exp(m.x), lower=cl, upper=cu))
            fbbt(m)
            if pe.value(m.c.lower) <= 0:
                _cl = 1e-6
            else:
                _cl = pe.value(m.c.lower)
            z = np.linspace(_cl, pe.value(m.c.upper), 100)
            if m.x.lb is None:
                xl = -np.inf
            else:
                xl = pe.value(m.x.lb)
            if m.x.ub is None:
                xu = np.inf
            else:
                xu = pe.value(m.x.ub)
            x = np.log(z)
            self.assertTrue(np.all(xl <= x))
            self.assertTrue(np.all(xu >= x))

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_log(self):
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pe.Block(concrete=True)
            m.x = pe.Var()
            m.c = pe.Constraint(expr=pe.inequality(body=pe.log(m.x), lower=cl, upper=cu))
            fbbt(m)
            z = np.linspace(pe.value(m.c.lower), pe.value(m.c.upper), 100)
            if m.x.lb is None:
                xl = -np.inf
            else:
                xl = pe.value(m.x.lb)
            if m.x.ub is None:
                xu = np.inf
            else:
                xu = pe.value(m.x.ub)
            x = np.exp(z)
            self.assertTrue(np.all(xl <= x))
            self.assertTrue(np.all(xu >= x))

    def test_sin(self):
        m = pe.Block(concrete=True)
        m.x = pe.Var(bounds=(-math.pi/2, math.pi/2))
        m.c = pe.Constraint(expr=pe.inequality(body=pe.sin(m.x), lower=-0.5, upper=0.5))
        fbbt(m.c)
        self.assertAlmostEqual(pe.value(m.x.lb), math.asin(-0.5))
        self.assertAlmostEqual(pe.value(m.x.ub), math.asin(0.5))

        m = pe.Block(concrete=True)
        m.x = pe.Var()
        m.c = pe.Constraint(expr=pe.inequality(body=pe.sin(m.x), lower=-0.5, upper=0.5))
        fbbt(m.c)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

    def test_cos(self):
        m = pe.Block(concrete=True)
        m.x = pe.Var(bounds=(0, math.pi))
        m.c = pe.Constraint(expr=pe.inequality(body=pe.cos(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertAlmostEqual(pe.value(m.x.lb), math.acos(0.5))
        self.assertAlmostEqual(pe.value(m.x.ub), math.acos(-0.5))

        m = pe.Block(concrete=True)
        m.x = pe.Var()
        m.c = pe.Constraint(expr=pe.inequality(body=pe.cos(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

    def test_tan(self):
        m = pe.Block(concrete=True)
        m.x = pe.Var(bounds=(-math.pi/2, math.pi/2))
        m.c = pe.Constraint(expr=pe.inequality(body=pe.tan(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertAlmostEqual(pe.value(m.x.lb), math.atan(-0.5))
        self.assertAlmostEqual(pe.value(m.x.ub), math.atan(0.5))

        m = pe.Block(concrete=True)
        m.x = pe.Var()
        m.c = pe.Constraint(expr=pe.inequality(body=pe.tan(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertEqual(m.x.lb, None)
        self.assertEqual(m.x.ub, None)

    def test_asin(self):
        m = pe.Block(concrete=True)
        m.x = pe.Var()
        m.c = pe.Constraint(expr=pe.inequality(body=pe.asin(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertAlmostEqual(pe.value(m.x.lb), math.sin(-0.5))
        self.assertAlmostEqual(pe.value(m.x.ub), math.sin(0.5))

    def test_acos(self):
        m = pe.Block(concrete=True)
        m.x = pe.Var()
        m.c = pe.Constraint(expr=pe.inequality(body=pe.acos(m.x), lower=1, upper=2))
        fbbt(m)
        self.assertAlmostEqual(pe.value(m.x.lb), math.cos(2))
        self.assertAlmostEqual(pe.value(m.x.ub), math.cos(1))

    def test_atan(self):
        m = pe.Block(concrete=True)
        m.x = pe.Var()
        m.c = pe.Constraint(expr=pe.inequality(body=pe.atan(m.x), lower=-0.5, upper=0.5))
        fbbt(m)
        self.assertAlmostEqual(pe.value(m.x.lb), math.tan(-0.5))
        self.assertAlmostEqual(pe.value(m.x.ub), math.tan(0.5))

    def test_multiple_constraints(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-3, 3))
        m.y = pe.Var(bounds=(0, None))
        m.z = pe.Var()
        m.c = pe.ConstraintList()
        m.c.add(m.x + m.y >= -1)
        m.c.add(m.x + m.y <= -1)
        m.c.add(m.y - m.x*m.z <= 2)
        m.c.add(m.y - m.x*m.z >= -2)
        m.c.add(m.x + m.z == 1)
        fbbt(m)
        self.assertAlmostEqual(pe.value(m.x.lb), -1, 8)
        self.assertAlmostEqual(pe.value(m.x.ub), -1, 8)
        self.assertAlmostEqual(pe.value(m.y.lb), 0, 8)
        self.assertAlmostEqual(pe.value(m.y.ub), 0, 8)
        self.assertAlmostEqual(pe.value(m.z.lb), 2, 8)
        self.assertAlmostEqual(pe.value(m.z.ub), 2, 8)

    def test_multiple_constraints2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-3, 3))
        m.y = pe.Var(bounds=(None, 0))
        m.z = pe.Var()
        m.c = pe.ConstraintList()
        m.c.add(-m.x - m.y >= -1)
        m.c.add(-m.x - m.y <= -1)
        m.c.add(-m.y - m.x*m.z >= -2)
        m.c.add(-m.y - m.x*m.z <= 2)
        m.c.add(-m.x - m.z == 1)
        fbbt(m)
        self.assertAlmostEqual(pe.value(m.x.lb), 1, 8)
        self.assertAlmostEqual(pe.value(m.x.ub), 1, 8)
        self.assertAlmostEqual(pe.value(m.y.lb), 0, 8)
        self.assertAlmostEqual(pe.value(m.y.ub), 0, 8)
        self.assertAlmostEqual(pe.value(m.z.lb), -2, 8)
        self.assertAlmostEqual(pe.value(m.z.ub), -2, 8)

    def test_binary(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(domain=pe.Binary)
        m.y = pe.Var(domain=pe.Binary)
        m.c = pe.Constraint(expr=m.x + m.y >= 1.5)
        fbbt(m)
        self.assertEqual(pe.value(m.x.lb), 1)
        self.assertEqual(pe.value(m.x.ub), 1)
        self.assertEqual(pe.value(m.y.lb), 1)
        self.assertEqual(pe.value(m.y.ub), 1)

        m = pe.ConcreteModel()
        m.x = pe.Var(domain=pe.Binary)
        m.y = pe.Var(domain=pe.Binary)
        m.c = pe.Constraint(expr=m.x + m.y <= 0.5)
        fbbt(m)
        self.assertEqual(pe.value(m.x.lb), 0)
        self.assertEqual(pe.value(m.x.ub), 0)
        self.assertEqual(pe.value(m.y.lb), 0)
        self.assertEqual(pe.value(m.y.ub), 0)

    def test_always_feasible(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1,2))
        m.y = pe.Var(bounds=(1,2))
        m.c = pe.Constraint(expr=m.x + m.y >= 0)
        fbbt(m)
        self.assertTrue(m.c.active)
        fbbt(m, deactivate_satisfied_constraints=True)
        self.assertFalse(m.c.active)

    def test_no_update_bounds(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0, 1))
        m.y = pe.Var(bounds=(1, 1))
        m.c = pe.Constraint(expr=m.x + m.y == 1)
        new_bounds = fbbt(m, update_variable_bounds=False)
        self.assertEqual(pe.value(m.x.lb), 0)
        self.assertEqual(pe.value(m.x.ub), 1)
        self.assertEqual(pe.value(m.y.lb), 1)
        self.assertEqual(pe.value(m.y.ub), 1)
        self.assertAlmostEqual(new_bounds[m.x][0], 0, 12)
        self.assertAlmostEqual(new_bounds[m.x][1], 0, 12)
        self.assertAlmostEqual(new_bounds[m.y][0], 1, 12)
        self.assertAlmostEqual(new_bounds[m.y][1], 1, 12)

    @unittest.skip('This test passes locally, but not on travis or appveyor. I will add an issue.')
    def test_skip_unknown_expression1(self):

        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(1,1))
        m.y = pe.Var()
        expr = DummyExpr([m.x, m.y])
        m.c = pe.Constraint(expr=expr == 1)
        logging_io = io.StringIO()
        handler = logging.StreamHandler(stream=logging_io)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('pyomo.contrib.fbbt.fbbt')
        logger.addHandler(handler)
        new_bounds = fbbt(m)
        handler.flush()
        self.assertEqual(pe.value(m.x.lb), 1)
        self.assertEqual(pe.value(m.x.ub), 1)
        self.assertEqual(pe.value(m.y.lb), None)
        self.assertEqual(pe.value(m.y.ub), None)
        a = "Unsupported expression type for FBBT"
        b = logging_io.getvalue()
        a = a.strip()
        b = b.strip()
        self.assertTrue(b.startswith(a))
        logger.removeHandler(handler)

    @unittest.skip('This test passes locally, but not on travis or appveyor. I will add an issue.')
    def test_skip_unknown_expression2(self):
        def dummy_unary_expr(x):
            return 0.5*x

        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(0,4))
        expr = UnaryFunctionExpression((m.x,), name='dummy_unary_expr', fcn=dummy_unary_expr)
        m.c = pe.Constraint(expr=expr == 1)
        logging_io = io.StringIO()
        handler = logging.StreamHandler(stream=logging_io)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger('pyomo.contrib.fbbt.fbbt')
        logger.addHandler(handler)
        new_bounds = fbbt(m)
        handler.flush()
        self.assertEqual(pe.value(m.x.lb), 0)
        self.assertEqual(pe.value(m.x.ub), 4)
        a = "Unsupported expression type for FBBT"
        b = logging_io.getvalue()
        a = a.strip()
        b = b.strip()
        self.assertTrue(b.startswith(a))
        logger.removeHandler(handler)

    def test_compute_expr_bounds(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1,1))
        m.y = pe.Var(bounds=(-1,1))
        e = m.x + m.y
        lb, ub = compute_bounds_on_expr(e)
        self.assertAlmostEqual(lb, -2, 14)
        self.assertAlmostEqual(ub, 2, 14)

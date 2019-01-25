import unittest
import pyomo.environ as pe
from pyomo.contrib.fbbt.fbbt import fbbt
import numpy as np


class TestFBBT(unittest.TestCase):
    def test_add(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.x+m.y, lower=cl, upper=cu))
                fbbt(m.c)
                x = np.linspace(m.x.lb, m.x.ub, 100)
                z = np.linspace(m.c.lower, m.c.upper, 100)
                if m.y.lb is None:
                    yl = -np.inf
                else:
                    yl = m.y.lb
                if m.y.ub is None:
                    yu = np.inf
                else:
                    yu = m.y.ub
                for _x in x:
                    _y = z - _x
                    self.assertTrue(np.all(yl <= _y))
                    self.assertTrue(np.all(yu >= _y))

    def test_sub1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.x-m.y, lower=cl, upper=cu))
                fbbt(m.c)
                x = np.linspace(m.x.lb, m.x.ub, 100)
                z = np.linspace(m.c.lower, m.c.upper, 100)
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

    def test_sub2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.y-m.x, lower=cl, upper=cu))
                fbbt(m.c)
                x = np.linspace(m.x.lb, m.x.ub, 100)
                z = np.linspace(m.c.lower, m.c.upper, 100)
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

    def test_mul(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.x*m.y, lower=cl, upper=cu))
                fbbt(m.c)
                x = np.linspace(m.x.lb + 1e-6, m.x.ub, 100, endpoint=False)
                z = np.linspace(m.c.lower, m.c.upper, 100)
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

    def test_div1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.x/m.y, lower=cl, upper=cu))
                fbbt(m.c)
                x = np.linspace(m.x.lb, m.x.ub, 100)
                z = np.linspace(m.c.lower + 1e-6, m.c.upper, 100, endpoint=False)
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

    def test_div2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.y/m.x, lower=cl, upper=cu))
                fbbt(m.c)
                x = np.linspace(m.x.lb, m.x.ub, 100)
                z = np.linspace(m.c.lower, m.c.upper, 100)
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

    def test_pow1(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.x**m.y, lower=cl, upper=cu))
                fbbt(m.c)
                x = np.linspace(m.x.lb + 1e-6, m.x.ub, 100, endpoint=False)
                z = np.linspace(m.c.lower + 1e-6, m.c.upper, 100, endpoint=False)
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

    def test_pow2(self):
        x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for xl, xu in x_bounds:
            for cl, cu in c_bounds:
                m = pe.Block(concrete=True)
                m.x = pe.Var(bounds=(xl, xu))
                m.y = pe.Var()
                m.c = pe.Constraint(expr=pe.inequality(body=m.y**m.x, lower=cl, upper=cu))
                fbbt(m.c)
                x = np.linspace(m.x.lb + 1e-6, m.x.ub, 100, endpoint=False)
                z = np.linspace(m.c.lower + 1e-6, m.c.upper, 100, endpoint=False)
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

    def test_exp(self):
        c_bounds = [(-2.5, 2.8), (0.5, 2.8), (0, 2.8), (1, 2.8), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pe.Block(concrete=True)
            m.x = pe.Var()
            m.c = pe.Constraint(expr=pe.inequality(body=pe.exp(m.x), lower=cl, upper=cu))
            fbbt(m.c)
            if m.c.lower <= 0:
                _cl = 1e-6
            else:
                _cl = m.c.lower
            z = np.linspace(_cl, m.c.upper, 100)
            if m.x.lb is None:
                xl = -np.inf
            else:
                xl = m.x.lb
            if m.x.ub is None:
                xu = np.inf
            else:
                xu = m.x.ub
            x = np.log(z)
            self.assertTrue(np.all(xl <= x))
            self.assertTrue(np.all(xu >= x))

    def test_log(self):
        c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
        for cl, cu in c_bounds:
            m = pe.Block(concrete=True)
            m.x = pe.Var()
            m.c = pe.Constraint(expr=pe.inequality(body=pe.log(m.x), lower=cl, upper=cu))
            fbbt(m.c)
            z = np.linspace(m.c.lower, m.c.upper, 100)
            if m.x.lb is None:
                xl = -np.inf
            else:
                xl = m.x.lb
            if m.x.ub is None:
                xu = np.inf
            else:
                xu = m.x.ub
            x = np.exp(z)
            self.assertTrue(np.all(xl <= x))
            self.assertTrue(np.all(xu >= x))

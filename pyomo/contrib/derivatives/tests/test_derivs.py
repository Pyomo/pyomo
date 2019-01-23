import unittest
import pyomo.environ as pe
from pyomo.contrib.derivatives.differentiate import reverse_ad


tol = 5


def approx_deriv(expr, wrt, delta=0.001):
    wrt.value += delta
    val1 = pe.value(expr)
    wrt.value -= 2*delta
    val2 = pe.value(expr)
    wrt.value += delta
    return (val1 - val2) / (2*delta)


class TestDerivs(unittest.TestCase):
    def test_prod(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        m.y = pe.Var(initialize=3.0)
        e = m.x * m.y
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_sum(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        m.y = pe.Var(initialize=3.0)
        e = 2.0*m.x + 3.0*m.y - m.x
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_div(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        m.y = pe.Var(initialize=3.0)
        e = m.x / m.y
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_pow(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        m.y = pe.Var(initialize=3.0)
        e = m.x ** m.y
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_exp(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        e = pe.exp(m.x)
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_log(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        e = pe.log(m.x)
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_ad, reverse_sd


tol = 6


def approx_deriv(expr, wrt, delta=0.001):
    numerator = 0
    wrt.value += 2*delta
    numerator -= pe.value(expr)
    wrt.value -= delta
    numerator += 8*pe.value(expr)
    wrt.value -= 2*delta
    numerator -= 8*pe.value(expr)
    wrt.value -= delta
    numerator += pe.value(expr)
    wrt.value += 2*delta
    return numerator / (12*delta)


class TestDerivs(unittest.TestCase):
    def test_prod(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        m.y = pe.Var(initialize=3.0)
        e = m.x * m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y], pe.value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_sum(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        m.y = pe.Var(initialize=3.0)
        e = 2.0*m.x + 3.0*m.y - m.x*m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y], pe.value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_div(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        m.y = pe.Var(initialize=3.0)
        e = m.x / m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y], pe.value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_pow(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        m.y = pe.Var(initialize=3.0)
        e = m.x ** m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y], pe.value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_sqrt(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        m.y = pe.Var(initialize=3.0)
        e = pe.sqrt(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_exp(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        e = pe.exp(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_log(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        e = pe.log(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_sin(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        e = pe.sin(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_cos(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        e = pe.cos(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_tan(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        e = pe.tan(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_asin(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=0.5)
        e = pe.asin(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_acos(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=0.5)
        e = pe.acos(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_atan(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2.0)
        e = pe.atan(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_nested(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=2)
        m.y = pe.Var(initialize=3)
        m.p = pe.Param(initialize=0.5, mutable=True)
        e = pe.exp(m.x**m.p + 3.2*m.y - 12)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y], pe.value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.p], pe.value(symbolic[m.p]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)
        self.assertAlmostEqual(derivs[m.p], approx_deriv(e, m.p), tol)

    def test_expressiondata(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(initialize=3)
        m.e = pe.Expression(expr=m.x * 2)

        @m.Expression([1, 2])
        def e2(m, i):
            if i == 1:
                return m.x + 4
            else:
                return m.x ** 2
        m.o = pe.Objective(expr=m.e + 1 + m.e2[1] + m.e2[2])
        derivs = reverse_ad(m.o.expr)
        symbolic = reverse_sd(m.o.expr)
        self.assertAlmostEqual(derivs[m.x], pe.value(symbolic[m.x]), tol)

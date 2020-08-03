#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest
from pyomo.environ import ConcreteModel, Var, Param, Objective, ExternalFunction, Expression, value, sqrt, exp, log, log10, cos, tan, sin, acos, asin, atan
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_ad, reverse_sd
from pyomo.common.getGSL import find_GSL


tol = 6


def approx_deriv(expr, wrt, delta=0.001):
    numerator = 0
    wrt.value += 2*delta
    numerator -=  value(expr)
    wrt.value -= delta
    numerator += 8* value(expr)
    wrt.value -= 2*delta
    numerator -= 8* value(expr)
    wrt.value -= delta
    numerator +=  value(expr)
    wrt.value += 2*delta
    return numerator / (12*delta)


class TestDerivs(unittest.TestCase):
    def test_prod(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        m.y =  Var(initialize=3.0)
        e = m.x * m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y],  value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_sum(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        m.y =  Var(initialize=3.0)
        e = 2.0*m.x + 3.0*m.y - m.x*m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y],  value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_div(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        m.y =  Var(initialize=3.0)
        e = m.x / m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y],  value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_pow(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        m.y =  Var(initialize=3.0)
        e = m.x ** m.y
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y],  value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

    def test_sqrt(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        m.y =  Var(initialize=3.0)
        e =  sqrt(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_exp(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        e =  exp(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_log(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        e =  log(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_log10(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        e =  log10(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_sin(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        e =  sin(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_cos(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        e =  cos(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_tan(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        e =  tan(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_asin(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=0.5)
        e =  asin(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_acos(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=0.5)
        e =  acos(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_atan(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2.0)
        e =  atan(m.x)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)

    def test_nested(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=2)
        m.y =  Var(initialize=3)
        m.p =  Param(initialize=0.5, mutable=True)
        e =  exp(m.x**m.p + 3.2*m.y - 12)
        derivs = reverse_ad(e)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol+3)
        self.assertAlmostEqual(derivs[m.y],  value(symbolic[m.y]), tol+3)
        self.assertAlmostEqual(derivs[m.p],  value(symbolic[m.p]), tol+3)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)
        self.assertAlmostEqual(derivs[m.p], approx_deriv(e, m.p), tol)

    def test_expressiondata(self):
        m =  ConcreteModel()
        m.x =  Var(initialize=3)
        m.e =  Expression(expr=m.x * 2)

        @m.Expression([1, 2])
        def e2(m, i):
            if i == 1:
                return m.x + 4
            else:
                return m.x ** 2
        m.o =  Objective(expr=m.e + 1 + m.e2[1] + m.e2[2])
        derivs = reverse_ad(m.o.expr)
        symbolic = reverse_sd(m.o.expr)
        self.assertAlmostEqual(derivs[m.x],  value(symbolic[m.x]), tol)

    def test_multiple_named_expressions(self):
        m =  ConcreteModel()
        m.x =  Var()
        m.y =  Var()
        m.x.value = 1
        m.y.value = 1
        m.E =  Expression(expr=m.x*m.y)
        e = m.E - m.E
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], 0)
        self.assertAlmostEqual(derivs[m.y], 0)
        symbolic = reverse_sd(e)
        self.assertAlmostEqual( value(symbolic[m.x]), 0)
        self.assertAlmostEqual( value(symbolic[m.y]), 0)

    def test_external(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest('Could not find the amplgsl.dll library')

        m =  ConcreteModel()
        m.hypot =  ExternalFunction(library=DLL, function='gsl_hypot')
        m.x =  Var(initialize=0.5)
        m.y =  Var(initialize=1.5)
        e = 2 * m.hypot(m.x, m.x*m.y)
        derivs = reverse_ad(e)
        self.assertAlmostEqual(derivs[m.x], approx_deriv(e, m.x), tol)
        self.assertAlmostEqual(derivs[m.y], approx_deriv(e, m.y), tol)

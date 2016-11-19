#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyutilib.th as unittest

from pyomo.util import DeveloperError
from pyomo.core import *
from pyomo.core.base.symbolic import (
    differentiate, NondifferentiableError,
    _sympy_available, _map_sympy2pyomo,
)

def s(e):
    return str(e).replace(' ','')

@unittest.skipIf( not _sympy_available,
                  "Symbolic derivatives require teh sympy package" )
class SymbolicDericatives(unittest.TestCase):
    def test_single_derivatives(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = differentiate(1, wrt=m.x)
        self.assertIn(type(e), (int,float))
        self.assertEqual(e, 0)

        e = differentiate(m.x, wrt=m.x)
        self.assertIn(type(e), (int,float))
        self.assertEqual(e, 1)

        e = differentiate(m.x**2, wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(2.*m.x))

        e = differentiate(m.y, wrt=m.x)
        self.assertIn(type(e), (int,float))
        self.assertEqual(e, 0)

        e = differentiate(m.x*m.y, wrt=m.x)
        self.assertIs(e, m.y)
        self.assertEqual(s(e), s(m.y))

        e = differentiate(m.x**2*m.y, wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(2.*m.y*m.x))

        e = differentiate(m.x**2/m.y, wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(2.*m.x*m.y**-1.))

    def test_single_derivative_list(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = differentiate(1, wrt_list=[m.x])
        self.assertIs(type(e), list)
        self.assertEqual(len(e), 1)
        e = e[0]
        self.assertIn(type(e), (int,float))
        self.assertEqual(e, 0)

        e = differentiate(m.x, wrt_list=[m.x])
        self.assertIs(type(e), list)
        self.assertEqual(len(e), 1)
        e = e[0]
        self.assertIn(type(e), (int,float))
        self.assertEqual(e, 1)

        e = differentiate(m.x**2, wrt_list=[m.x])
        self.assertIs(type(e), list)
        self.assertEqual(len(e), 1)
        e = e[0]
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(2.*m.x))

        e = differentiate(m.y, wrt_list=[m.x])
        self.assertIs(type(e), list)
        self.assertEqual(len(e), 1)
        e = e[0]
        self.assertIn(type(e), (int,float))
        self.assertEqual(e, 0)

        e = differentiate(m.x*m.y, wrt_list=[m.x])
        self.assertIs(type(e), list)
        self.assertEqual(len(e), 1)
        e = e[0]
        self.assertIs(e, m.y)
        self.assertEqual(s(e), s(m.y))

        e = differentiate(m.x**2*m.y, wrt_list=[m.x])
        self.assertIs(type(e), list)
        self.assertEqual(len(e), 1)
        e = e[0]
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(2.*m.y*m.x))

        e = differentiate(m.x**2/m.y, wrt_list=[m.x])
        self.assertIs(type(e), list)
        self.assertEqual(len(e), 1)
        e = e[0]
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(2.*m.x*m.y**-1.))


    def test_trig_fuctions(self):
        m = ConcreteModel()
        m.x = Var()

        e = differentiate(sin(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(cos(m.x)))

        e = differentiate(cos(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(-1.0*sin(m.x)))

        e = differentiate(tan(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(1.+tan(m.x)**2.))

        e = differentiate(sinh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(cosh(m.x)))

        e = differentiate(cosh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(sinh(m.x)))

        e = differentiate(tanh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(1.-tanh(m.x)**2.))


        e = differentiate(asin(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s((1.-m.x**2.)**-0.5))

        e = differentiate(acos(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(-1.*(1.-m.x**2.)**-0.5))

        e = differentiate(atan(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s((1.+m.x**2.)**-1.))

        e = differentiate(asinh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s((1.+m.x**2)**-.5))

        e = differentiate(acosh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s((-1.+m.x**2.)**-.5))

        e = differentiate(atanh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s((1.-m.x**2.)**-1.))


    def test_intrinsic_fuctions(self):
        m = ConcreteModel()
        m.x = Var()

        e = differentiate(log(m.x), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(m.x**-1.))

        e = differentiate(log10(log10(m.x)), wrt=m.x)
        self.assertTrue(e.is_expression())
        self.assertEqual(s(e), s(1./log(10)*m.x**-1.*log(m.x)**-1.))


    def test_nondifferentiable(self):
        m = ConcreteModel()
        m.x = Var()

        self.assertRaisesRegexp(
            NondifferentiableError,
            "The sub-expression '.*' is not differentiable with respect to x",
            differentiate, ceil(m.x), wrt=m.x)

        self.assertRaisesRegexp(
            NondifferentiableError,
            "The sub-expression '.*' is not differentiable with respect to x",
            differentiate, floor(m.x), wrt=m.x)

    def test_errors(self):
        m = ConcreteModel()
        m.x = Var()

        self.assertRaisesRegexp(
            ValueError,
            "Must specify exactly one of wrt and wrt_list",
            differentiate, m.x, wrt=m.x, wrt_list=[m.x])

        x = pyomo.core.base.symbolic.sympy.Symbol('x')
        class bogus(object):
            def __init__(self):
                self._args = (x,)
        self.assertRaisesRegexp(
            DeveloperError,
            "sympy expression .* not found in the operator map",
            _map_sympy2pyomo, bogus(), {x:m.x})

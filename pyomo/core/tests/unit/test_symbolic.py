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

from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (ConcreteModel, Var, Param, Set, NonNegativeReals,
                           Expression, RangeSet, sin, cos, tan, sinh, cosh,
                           tanh, asin, acos, atan, asinh, acosh, atanh,
                           log, log10, exp, sqrt, ceil, floor)
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import PyomoSympyBimap, sympy_available, sympy2pyomo_expression


def s(e):
    return str(e).replace(' ','').replace('1.0','1').replace('2.0','2')

@unittest.skipIf( not sympy_available,
                  "Symbolic derivatives require the sympy package" )
class SymbolicDerivatives(unittest.TestCase):

    def test_single_derivatives1(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = differentiate(1, wrt=m.x)
        self.assertIn(type(e), (int,float))
        self.assertEqual(e, 0)

    def test_single_derivatives2(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = differentiate(m.x, wrt=m.x)
        self.assertIn(type(e), (int,float))
        self.assertEqual(e, 1)

    def test_single_derivatives3(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = differentiate(m.x**2, wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(2.*m.x))

    def test_single_derivatives4(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = differentiate(m.y, wrt=m.x)
        self.assertIn(type(e), (int,float))
        self.assertEqual(e, 0)

    def test_single_derivatives5(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = differentiate(m.x*m.y, wrt=m.x)
        self.assertIs(e, m.y)
        self.assertEqual(s(e), s(m.y))

    def test_single_derivatives6(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = differentiate(m.x**2*m.y, wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(2.*m.x*m.y))

    def test_single_derivatives7(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = differentiate(m.x**2/m.y, wrt=m.x)
        self.assertTrue(e.is_expression_type())
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
        self.assertTrue(e.is_expression_type())
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
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(2.*m.x*m.y))

        e = differentiate(m.x**2/m.y, wrt_list=[m.x])
        self.assertIs(type(e), list)
        self.assertEqual(len(e), 1)
        e = e[0]
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(2.*m.x*m.y**-1.))


    def test_trig_fuctions(self):
        m = ConcreteModel()
        m.x = Var()

        e = differentiate(sin(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(cos(m.x)))

        e = differentiate(cos(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(-1.0*sin(m.x)))

        e = differentiate(tan(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(1.+tan(m.x)**2.))

        e = differentiate(sinh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(cosh(m.x)))

        e = differentiate(cosh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(sinh(m.x)))

        e = differentiate(tanh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(1.0-tanh(m.x)**2.0))


        e = differentiate(asin(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s((1.0 + (-1.0)*m.x**2.)**-0.5))

        e = differentiate(acos(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(-1.*(1.+ (-1.0)*m.x**2.)**-0.5))

        e = differentiate(atan(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s((1.+m.x**2.)**-1.))

        e = differentiate(asinh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s((1.+m.x**2)**-.5))

        e = differentiate(acosh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s((-1.+m.x**2.)**-.5))

        e = differentiate(atanh(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s((1.+(-1.0)*m.x**2.)**-1.))


    def test_intrinsic_functions1(self):
        m = ConcreteModel()
        m.x = Var()

        e = differentiate(log(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(m.x**-1.))

    def test_intrinsic_functions2(self):
        m = ConcreteModel()
        m.x = Var()

        e = differentiate(exp(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(exp(m.x)))

    def test_intrinsic_functions3(self):
        m = ConcreteModel()
        m.x = Var()

        e = differentiate(exp(2 * m.x), wrt=m.x)
        self.assertEqual(s(e), s(2. * exp(2. * m.x)))

    def test_intrinsic_functions4(self):
        m = ConcreteModel()
        m.x = Var()

        e = differentiate(log10(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(m.x**-1.0 * (1.0/log(10))))

    def test_intrinsic_functions5(self):
        m = ConcreteModel()
        m.x = Var()

        e = differentiate(log10(log10(m.x)), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(m.x**-1.0 * (1.0/log(10)) * log(m.x)**-1.0))

    def test_sqrt_function(self):
        m = ConcreteModel()
        m.x = Var()

        e = differentiate(sqrt(m.x), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(0.5 * m.x**-0.5))

    def test_abs_and_complex(self):
        m = ConcreteModel()
        m.x = Var()

        # Unless we force sympy to know that X is real, it will return a
        # complex expression.  This tests issue #1139.
        e = differentiate(abs(m.x**2), wrt=m.x)
        self.assertTrue(e.is_expression_type())
        self.assertEqual(s(e), s(2 * m.x))

    def test_param(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(mutable=True, initialize=5)

        e = differentiate(m.p*m.x, wrt=m.x)
        self.assertIs(type(e), float)
        self.assertEqual(e, 5.0)

    def test_Expression_component(self):
        m = ConcreteModel()
        m.s = Set(initialize=['A', 'B'])
        m.x = Var(m.s, domain=NonNegativeReals)

        def y_rule(m, s):
            return m.x[s] * 2
        m.y = Expression(m.s, rule=y_rule)

        expr = 1 - m.y['A'] ** 2
        jacs = differentiate(expr, wrt_list=[m.x['A'], m.x['B']])
        self.assertEqual(str(jacs[0]), "-8.0*x[A]")
        self.assertEqual(str(jacs[1]), "0.0")

        expr = 1 - m.y['B'] ** 2
        jacs = differentiate(expr, wrt_list=[m.x['A'], m.x['B']])
        self.assertEqual(str(jacs[0]), "0.0")
        self.assertEqual(str(jacs[1]), "-8.0*x[B]")

    def test_jacobian(self):
        m = ConcreteModel()
        m.I = RangeSet(4)
        m.x = Var(m.I)

        idxMap = {}
        jacs = []
        for i in m.I:
            idxMap[i] = len(jacs)
            jacs.append(m.x[i])

        expr = m.x[1]+m.x[2]*m.x[3]**2
        ans = differentiate(expr, wrt_list=jacs)

        self.assertEqual(len(ans), len(m.I))
        self.assertEqual(str(ans[0]), "1.0")
        self.assertEqual(str(ans[1]), "x[3]**2.0")
        self.assertEqual(str(ans[2]), "2.0*x[2]*x[3]")
        # 0 calculated by bypassing sympy
        self.assertEqual(str(ans[3]), "0.0")

    def test_hessian(self):
        m = ConcreteModel()
        m.I = RangeSet(4)
        m.x = Var(m.I)

        idxMap = {}
        hessian = []
        for i in m.I:
            for j in m.I:
                idxMap[i,j] = len(hessian)
                hessian.append((m.x[i], m.x[j]))

        expr = m.x[1]+m.x[2]*m.x[3]**2
        ans = differentiate(expr, wrt_list=hessian)

        self.assertEqual(len(ans), len(m.I)**2)
        for i in m.I:
            for j in m.I:
                self.assertEqual(str(ans[idxMap[i,j]]), str(ans[idxMap[j,i]]))
        # 0 calculated by sympy
        self.assertEqual(str(ans[idxMap[1,1]]), "0.0")
        self.assertEqual(str(ans[idxMap[2,2]]), "0.0")
        self.assertEqual(str(ans[idxMap[3,3]]), "2.0*x[2]")
        # 0 calculated by bypassing sympy
        self.assertEqual(str(ans[idxMap[4,4]]), "0.0")
        self.assertEqual(str(ans[idxMap[2,3]]), "2.0*x[3]")

    def test_nondifferentiable(self):
        m = ConcreteModel()
        m.foo = Var()

        self.assertRaisesRegexp(
            NondifferentiableError,
            "The sub-expression '.*' is not differentiable "
            "with respect to .*foo",
            differentiate, ceil(m.foo), wrt=m.foo)

        self.assertRaisesRegexp(
            NondifferentiableError,
            "The sub-expression '.*' is not differentiable "
            "with respect to .*foo",
            differentiate, floor(m.foo), wrt=m.foo)

    def test_errors(self):
        m = ConcreteModel()
        m.x = Var()

        self.assertRaisesRegexp(
            ValueError,
            "Must specify exactly one of wrt and wrt_list",
            differentiate, m.x, wrt=m.x, wrt_list=[m.x])

        obj_map = PyomoSympyBimap()
        class bogus(object):
            def __init__(self):
                self._args = (obj_map.getSympySymbol(m.x),)
        self.assertRaisesRegexp(
            DeveloperError,
            "sympy expression .* not found in the operator map",
            sympy2pyomo_expression, bogus(), obj_map)


class SymbolicDerivatives_importTest(unittest.TestCase):
    def test_sympy_avail_flag(self):
        if sympy_available:
            import sympy
        else:
            with self.assertRaises(ImportError):
                import sympy

if __name__ == "__main__":
    unittest.main()

#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyutilib.th as unittest
#from nose.tools import nottest
import pyomo.repn

_campl_available = False
try:
    # Disabling these tests
    #import pyomo.repn.plugins.ampl.cAmpl as cAmpl
    _campl_available = False
except ImportError:
    pass

class MockFixedValue:
    value = 42
    def __init__(self, v = 42):
        self.value = v
    def is_fixed(self):
        return True

@unittest.skipUnless(_campl_available, "C AMPL module required")
class CAmplBasicTest(unittest.TestCase):
    def testNone(self):
        with self.assertRaises(ValueError) as cm:
            pyomo.repn.generate_ampl_repn(None)

    def testVar(self):
        testname = 'testname'

        var = pyomo.core.base.var._VarData(testname, None, None)
        var_ar = pyomo.repn.generate_ampl_repn(var)

        self.assertIsInstance(var_ar, pyomo.core.ampl.ampl_representation)

        self.assertEquals({testname:1.0}, var_ar._linear_terms_coef)

        self.assertEquals(1, len(var_ar._linear_terms_var))
        self.assertIsInstance(var_ar._linear_terms_var[testname], pyomo.core.base.var._VarData)

    def testExpressionBase(self):
        exp = pyomo.core.base.expr._ExpressionBase('name', 0, [])
        with self.assertRaises(ValueError) as cm:
            exp_ar = pyomo.repn.generate_ampl_repn(exp)

    def testSumExpression(self):
        exp = pyomo.core.base.expr._SumExpression()
        exp_ar = pyomo.repn.generate_ampl_repn(exp)

        self.assertIsInstance(exp_ar, pyomo.core.ampl.ampl_representation)

    def testProductExpression(self):
        x = pyomo.core.base.var.Var()
        y = pyomo.core.base.var.Var()
        exp = x * y
        exp_ar = pyomo.repn.generate_ampl_repn(exp)

        self.assertIsInstance(exp_ar, pyomo.core.ampl.ampl_representation)
        self.assertIs(exp_ar._nonlinear_expr, exp)
        self.assertTrue(exp_ar.is_nonlinear())

    def testProductExpressionZeroDiv(self):
        exp = pyomo.core.base.expr._ProductExpression()
        exp._numerator = [MockFixedValue(1)]
        exp._denominator = [MockFixedValue(0)]
        with self.assertRaises(ZeroDivisionError) as cm:
            exp_ar = pyomo.repn.generate_ampl_repn(exp)

    def testPowExpressionNoneArgs(self):
        exp = pyomo.core.base.expr._PowExpression([None, None])
        with self.assertRaises(ValueError) as cm:
            exp_ar = pyomo.repn.generate_ampl_repn(exp)

    def testPowExpressionConstants(self):
        v1 = MockFixedValue(2)
        v2 = MockFixedValue(3)

        exp = pyomo.core.base.expr._PowExpression([v1, v2])
        exp_ar = pyomo.repn.generate_ampl_repn(exp)

        self.assertIsNotNone(exp_ar)
        self.assertIsInstance(exp_ar, pyomo.core.ampl.ampl_representation)

        self.assertEquals(exp_ar._constant, 8)

    def testPowExpressionExp1(self):
        v1 = pyomo.core.base.var.Var()
        v2 = MockFixedValue(1)

        exp = pyomo.core.base.expr._PowExpression([v1, v2])
        exp_ar = pyomo.repn.generate_ampl_repn(exp)

        self.assertIsNotNone(exp_ar)
        self.assertIsInstance(exp_ar, pyomo.core.ampl.ampl_representation)
        self.assertEquals(exp_ar._linear_terms_var, pyomo.repn.generate_ampl_repn(v1)._linear_terms_var)

    def testPowExpressionExp0(self):
        v1 = pyomo.core.base.var.Var()
        v2 = MockFixedValue(0)

        exp = pyomo.core.base.expr._PowExpression([v1, v2])
        exp_ar = pyomo.repn.generate_ampl_repn(exp)

        self.assertIsNotNone(exp_ar)
        self.assertIsInstance(exp_ar, pyomo.core.ampl.ampl_representation)
        self.assertEquals(exp_ar._constant, 1)

    def testPowExpressionNonlinear(self):
        v1 = pyomo.core.base.var.Var(initialize=2)
        v2 = pyomo.core.base.var.Var(initialize=3)

        exp = pyomo.core.base.expr._PowExpression([v1, v2])
        exp_ar = pyomo.repn.generate_ampl_repn(exp)

        self.assertIsNotNone(exp_ar)
        self.assertIsInstance(exp_ar, pyomo.core.ampl.ampl_representation)
        self.assertIsInstance(exp_ar._nonlinear_expr, pyomo.core.base.expr._PowExpression)

    def testIntrinsicFunctionExpressionEmpty(self):
        exp = pyomo.core.base.expr._IntrinsicFunctionExpression('testname', 0, [], None)
        with self.assertRaises(AssertionError) as cm:
            exp_ar = pyomo.repn.generate_ampl_repn(exp)

    def testIntrinsicFunctionExpressionNoneArg(self):
        exp = pyomo.core.base.expr._IntrinsicFunctionExpression('sum', 1, [None], sum)
        with self.assertRaises(ValueError) as cm:
            exp_ar = pyomo.repn.generate_ampl_repn(exp)

    def testAbsFunctionExpression(self):
        exp = pyomo.core.base.expr._AbsExpression([2])
        with self.assertRaises(ValueError) as cm:
            exp_ar = pyomo.repn.generate_ampl_repn(exp)

    def testIntrinsicFunctionExpression(self):
        exp = pyomo.core.base.expr._IntrinsicFunctionExpression('sum', 1, [MockFixedValue()], sum)
        exp_ar = pyomo.repn.generate_ampl_repn(exp)

        self.assertIsNotNone(exp_ar)
        self.assertIsInstance(exp_ar, pyomo.core.ampl.ampl_representation)

        self.assertEquals(type(exp), type(exp_ar._nonlinear_expr))
        self.assertEquals(exp.name, exp_ar._nonlinear_expr.name)
        self.assertEquals(0, len(exp_ar._nonlinear_vars))

    def testFixedValue(self):
        val = MockFixedValue()
        val_ar = pyomo.repn.generate_ampl_repn(val)

        self.assertIsInstance(val_ar, pyomo.core.ampl.ampl_representation)
        self.assertEquals(MockFixedValue.value, val_ar._constant)

    def testCombinedProductSum(self):
        x = pyomo.core.base.var.Var()
        y = pyomo.core.base.var.Var()
        z = pyomo.core.base.var.Var()
        exp = x * y + z

        exp_ar = pyomo.repn.generate_ampl_repn(exp)

        self.assertIsInstance(exp_ar, pyomo.core.ampl.ampl_representation)
        self.assertTrue(exp_ar.is_nonlinear())

if __name__ == "__main__":
    unittest.main(verbosity=2)

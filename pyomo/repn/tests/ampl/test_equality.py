#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import types

import pyutilib.th as unittest

from pyomo.repn.tests.ampl.helper import MockFixedValue
from pyomo.core import *

_campl_available = False
#FIXME: Disabling C AMPL tests until we decide whether to keep C AMPL module
#       around and keep it up to date with nl writer changes. (ZBF)
#try:
#    import pyomo.core.ampl.cAmpl as cAmpl
#    _campl_available = True
#    cgar = pyomo.core.ampl.cAmpl.generate_ampl_repn
#    gar = pyomo.core.ampl.ampl.py_generate_ampl_repn
#except ImportError:
from pyomo.repn.ampl_repn import _generate_ampl_repn as gar
from pyomo.repn.ampl_repn import AmplRepn

class _GenericAmplRepnEqualityTests(unittest.TestCase):
    def setUp(self):
        # Helper function for param init
        def q_initialize(model, i):
            return [2,3,5,7,11,13,17,19,23,29][i-1]
        # Need a model so that variables can be distinguishable
        self.model = ConcreteModel()
        self.model.s = RangeSet(10)
        self.model.p = Param(default=42)
        self.model.q = Param(self.model.s, initialize=q_initialize)
        self.model.w = Var(self.model.s)
        self.model.x = Var()
        self.model.y = Var()
        self.model.z = Var(self.model.s)
        
    def tearDown(self):
        self.model = None

    def assertAmplRepnMatch(self, rep1, rep2):
        self.assertEqual(rep1, rep2)
        self.assertEqual(rep1.is_constant(), rep2.is_constant())
        self.assertEqual(rep1.is_linear(), rep2.is_linear())
        self.assertEqual(rep1.is_nonlinear(), rep2.is_nonlinear())
        self.assertEqual(rep1.needs_sum(), rep2.needs_sum())

class AmplRepnEqualityTests(_GenericAmplRepnEqualityTests):
    """Serves as a test class for the AmplRepn.__eq__ method."""

    def testBasicEquality(self):
        self.assertEqual(AmplRepn(), AmplRepn())

    def testBasicInequality(self):
        self.assertNotEqual(AmplRepn(), None)

    def testVarEquality(self):
        self.assertEqual(gar(self.model.x), gar(self.model.x))

    def testVarInequality(self):
        self.assertNotEqual(gar(self.model.x), gar(self.model.y))

    def testVarCoefInequality(self):
        self.assertNotEqual(gar(self.model.x), gar(2.0 * self.model.x))

    def testFixedValueEquality(self):
        self.assertEqual(gar(MockFixedValue()), gar(MockFixedValue()))

    def testFixedValueInequality(self):
        self.assertNotEqual(gar(MockFixedValue(1)), gar(MockFixedValue(2)))

    def testProductEquality(self):
        expr = self.model.x * self.model.y
        self.assertEqual(gar(expr), gar(expr))

    def testProductInequality(self):
        e1 = self.model.x * self.model.x
        e2 = self.model.y * self.model.y
        self.assertNotEqual(gar(e1), gar(e2))

    def testSumEquality(self):
        expr = self.model.x + self.model.y
        self.assertEqual(gar(expr), gar(expr))

    def testSumInequality(self):
        e1 = self.model.x + self.model.y
        e2 = self.model.x + self.model.x
        self.assertNotEqual(gar(e1), gar(e2))

    def testSumCommutativeEquality(self):
        e1 = self.model.x + self.model.y
        e2 = self.model.y + self.model.x
        self.assertEqual(gar(e1), gar(e2))

    def testPowEquality(self):
        expr = self.model.x ** 2
        self.assertEqual(gar(expr), gar(expr))

    def testPowInequality(self):
        e1 = self.model.x ** 2
        e2 = self.model.x ** 3
        self.assertNotEqual(gar(e1), gar(e2))

    def testIntrinsicEquality(self):
        fns = [sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, log, exp]
        for fn in fns:
            expr = fn(self.model.x)
            self.assertEqual(gar(expr), gar(expr))

    def testIntrinsicInequality(self):
        e1 = sin(self.model.x)
        e2 = cos(self.model.x)
        self.assertNotEqual(gar(e1), gar(e2))

    def testCompoundEquality(self):
        expr = self.model.x + self.model.y + self.model.x * self.model.y
        self.assertEqual(gar(expr), gar(expr))

    def testMoreCompoundEquality(self):
        expr = ((self.model.x + self.model.y) * self.model.x) ** 2
        self.assertEqual(gar(expr), gar(expr))

    def testCompoundCoefficientInequality(self):
        e1 = self.model.x * self.model.y
        e2 = 2.0 * self.model.x * self.model.y
        self.assertNotEqual(gar(e1), gar(e2))
    
    def testQuotientEquality(self):
        expr = self.model.x / self.model.y
        self.assertEqual(gar(expr), gar(expr))

    def testCompoundQuotientEquality(self):
        expr = self.model.y + self.model.x / self.model.y + self.model.y ** 2
        self.assertEqual(gar(expr), gar(expr))

    def testQuotientInequality(self):
        e1 = self.model.x / self.model.y
        e2 = self.model.y / self.model.x
        self.assertNotEqual(gar(e1), gar(e2))

    def testSumEquality(self):
        expr = sum(self.model.z[i] for i in self.model.s)
        self.assertEqual(gar(expr), gar(expr))

    def testCompoundSumEquality(self):
        expr = sum(self.model.z[i] for i in self.model.s) + self.model.x / self.model.y
        self.assertEqual(gar(expr), gar(expr))

@unittest.skipUnless(_campl_available, "C AMPL module required")
class CAmplEqualityCompatTests(_GenericAmplRepnEqualityTests):
    """Tests whether the Python and cAmpl implementations produce identical repns."""

    def testEnvironment(self):
        # Simply ensure the import environment is working.
        with self.assertRaises(ValueError) as cm:
            cAmpl.generate_ampl_repn(None, None, None)

    def testEnvironmentTypes(self):
        self.assertEqual(type(gar), types.FunctionType)
        self.assertEqual(type(cgar), types.BuiltinFunctionType)

    def testFixedValue(self):
        expr = MockFixedValue()
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testVar(self):
        expr = self.model.x
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testProduct(self):
        expr = self.model.x * self.model.y
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testSum(self):
        expr = self.model.x + self.model.y
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testSumCommutative(self):
        e1 = self.model.x + self.model.y
        e2 = self.model.y + self.model.x
        self.assertAmplRepnMatch(gar(e1), cgar(e2))

    def testPow(self):
        expr = self.model.x ** 2
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testIntrinsic(self):
        fns = [sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, log, exp]
        for fn in fns:
            expr = fn(self.model.x)
            self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testCompound(self):
        expr = self.model.x + self.model.y + self.model.x * self.model.y
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testMoreCompound(self):
        expr = ((self.model.x + self.model.y) * self.model.x) ** 2
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testQuotient(self):
        expr = self.model.x / self.model.y
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testCompoundQuotient(self):
        expr = self.model.y + self.model.x / self.model.y + self.model.y ** 2
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testSum(self):
        expr = sum(self.model.z[i] for i in self.model.s)
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testCompoundSum(self):
        expr = sum(self.model.z[i] for i in self.model.s) + self.model.x / self.model.y
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testSumExpression(self):
        expr = sum(self.model.w[i] / self.model.z[i] ** 3 for i in self.model.s)
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testSumExpressionParam(self):
        expr = sum(value(self.model.q[i]) / self.model.z[i] ** 3 for i in self.model.s)
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testCantilvrConstraintExpr(self):
        # originally from pyomo.data.cute cantilvr model.
        expr = sum(value(self.model.q[i]) / self.model.z[i] ** 3 for i in self.model.s) - 1.0
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

    def testCantilvrObjective(self):
        # originally from pyomo.data.cute cantilvr model.
        # exposes problem in linear product handling, if present.
        expr = sum(self.model.z[i] for i in self.model.s) * 0.0624
        self.assertAmplRepnMatch(gar(expr), cgar(expr))

if __name__ == "__main__":
    unittest.main(verbosity=2)

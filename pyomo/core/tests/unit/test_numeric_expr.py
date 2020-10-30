#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for expression generation
#

import copy
import pickle
import math
import os
from collections import defaultdict

from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
from pyutilib.th import nottest

from pyomo.environ import ConcreteModel, AbstractModel, RangeSet, Var, Param, Set, Constraint, ConstraintList, Expression, Objective, Reals, ExternalFunction, PositiveReals, log10, exp, floor, ceil, log, cos, sin, tan, acos, asin, atan, sinh, cosh, tanh, acosh, asinh, atanh, sqrt, value, quicksum, sum_product, is_fixed, is_constant
from pyomo.kernel import variable, expression, objective
from pyomo.core.expr.numvalue import (NumericConstant, as_numeric,
                                      native_numeric_types,
                                      is_potentially_variable, polynomial_degree)
from pyomo.core.expr.numeric_expr import (
    ExpressionBase, UnaryFunctionExpression, SumExpression, PowExpression,
    ProductExpression, NegationExpression, linear_expression,
    MonomialTermExpression, LinearExpression, DivisionExpression,
    NPV_NegationExpression, NPV_ProductExpression, 
    NPV_PowExpression, NPV_DivisionExpression,
    decompose_term, clone_counter, nonlinear_expression,
    _MutableLinearExpression, _MutableSumExpression, _decompose_linear_terms,
    LinearDecompositionError,
)
import pyomo.core.expr.logical_expr as logical_expr
from pyomo.core.expr.visitor import (expression_to_string, 
                                     clone_expression)
from pyomo.core.expr.current import Expr_if
from pyomo.core.base.label import NumericLabeler
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr import expr_common
from pyomo.core.base.var import _GeneralVarData

from pyomo.repn import generate_standard_repn

class TestExpression_EvaluateNumericConstant(unittest.TestCase):

    def setUp(self):
        # Do we expect arithmetic operations to return expressions?
        self.expectExpression = False
        # Do we expect relational tests to return constant expressions?
        self.expectConstExpression = True

    def create(self, val, domain):
        # Create the type of expression term that we are testing
        return NumericConstant(val)

    @nottest
    def value_test(self, exp, val, expectExpression=None):
        """ Test the value of the expression. """
        #
        # Override the class value of 'expectExpression'
        #
        if expectExpression is None:
            expectExpression = self.expectExpression
        #
        # Confirm whether 'exp' is an expression
        #
        self.assertEqual(isinstance(exp,ExpressionBase), expectExpression)
        #
        # Confirm that 'exp' has the expected value
        #
        self.assertEqual(value(exp), val)

    @nottest
    def relation_test(self, exp, val, expectConstExpression=None):
        """ Test a relationship expression. """
        #
        # Override the class value of 'expectConstExpression'
        #
        if expectConstExpression is None:
            expectConstExpression = self.expectConstExpression
        #
        # Confirm that this is a relational expression
        #
        self.assertTrue(isinstance(exp, ExpressionBase))
        self.assertTrue(exp.is_relational())
        #
        # Check that the expression evaluates correctly
        #
        self.assertEqual(exp(), val)
        #
        # Check that the expression evaluates correctly in a Boolean context
        #
        try:
            if expectConstExpression:
                #
                # The relational expression should be a constant.
                #
                # Check that 'val' equals the boolean value of the expression.
                #
                self.assertEqual(bool(exp), val)
                #
                # The 'chainedInequality' value is None
                #
                if logical_expr._using_chained_inequality:
                    self.assertIsNone(logical_expr._chainedInequality.prev)
            else:
                #
                # The relational expression may not be constant
                #
                # Check that the expression evaluates to 'val'
                #
                if logical_expr._using_chained_inequality:
                    self.assertEqual(bool(exp), True)
                else:
                    self.assertEqual(bool(exp), val)
                #
                # Check that the 'chainedInequality' value is the current expression
                #
                if logical_expr._using_chained_inequality:
                    self.assertIs(exp,logical_expr._chainedInequality.prev)
        finally:
            #
            # TODO: Why would we get here?  Because the expression isn't constant?
            #
            # Check that the 'chainedInequality' value is None
            #
            if logical_expr._using_chained_inequality:
                logical_expr._chainedInequality.prev = None

    def test_lt(self):
        #
        # Test the 'less than' operator
        #
        a=self.create(1.3, Reals)
        b=self.create(2.0, Reals)
        self.relation_test(a<b, True)
        self.relation_test(a<a, False) 
        self.relation_test(b<a, False)
        self.relation_test(a<2.0, True)
        self.relation_test(a<1.3, False)
        self.relation_test(b<1.3, False)
        self.relation_test(1.3<b, True)
        self.relation_test(1.3<a, False)
        self.relation_test(2.0<a, False)

    def test_gt(self):
        #
        # Test the 'greater than' operator
        #
        a=self.create(1.3, Reals)
        b=self.create(2.0, Reals)
        self.relation_test(a>b, False)
        self.relation_test(a>a, False)
        self.relation_test(b>a, True)
        self.relation_test(a>2.0, False)
        self.relation_test(a>1.3, False)
        self.relation_test(b>1.3, True)
        self.relation_test(1.3>b, False)
        self.relation_test(1.3>a, False)
        self.relation_test(2.0>a, True)

    def test_eq(self):
        #
        # Test the 'equals' operator
        #
        a=self.create(1.3, Reals)
        b=self.create(2.0, Reals)
        self.relation_test(a==b, False, True)
        self.relation_test(a==a, True, True)
        self.relation_test(b==a, False, True)
        self.relation_test(a==2.0, False, True)
        self.relation_test(a==1.3, True, True)
        self.relation_test(b==1.3, False, True)
        self.relation_test(1.3==b, False, True)
        self.relation_test(1.3==a, True, True)
        self.relation_test(2.0==a, False, True)

    def test_arithmetic(self):
        #
        #
        # Test binary arithmetic operators
        #
        a=self.create(-0.5, Reals)
        b=self.create(2.0, Reals)
        self.value_test(a-b, -2.5)
        self.value_test(a+b, 1.5)
        self.value_test(a*b, -1.0)
        self.value_test(b/a, -4.0)
        self.value_test(a**b, 0.25)

        self.value_test(a-2.0, -2.5)
        self.value_test(a+2.0, 1.5)
        self.value_test(a*2.0, -1.0)
        self.value_test(b/(0.5), 4.0)
        self.value_test(a**2.0, 0.25)

        self.value_test(0.5-b, -1.5)
        self.value_test(0.5+b, 2.5)
        self.value_test(0.5*b, 1.0)
        self.value_test(2.0/a, -4.0)
        self.value_test((0.5)**b, 0.25)

        self.value_test(-a, 0.5)
        self.value_test(+a, -0.5, False)        # This doesn't generate an expression
        self.value_test(abs(-a), 0.5)


class TestExpression_EvaluateVarData(TestExpression_EvaluateNumericConstant):

    def setUp(self):
        #
        # Create Model
        #
        TestExpression_EvaluateNumericConstant.setUp(self)
        #
        # Create model instance
        #
        self.expectExpression = True
        self.expectConstExpression = False

    def create(self, val, domain):
        tmp=_GeneralVarData()
        tmp.domain = domain
        tmp.value=val
        return tmp


class TestExpression_EvaluateVar(TestExpression_EvaluateNumericConstant):

    def setUp(self):
        #
        # Create Model
        #
        TestExpression_EvaluateNumericConstant.setUp(self)
        #
        # Create model instance
        #
        self.expectExpression = True
        self.expectConstExpression = False

    def create(self, val, domain):
        tmp=Var(name="unknown",domain=domain)
        tmp.construct()
        tmp.value=val
        return tmp


class TestExpression_EvaluateFixedVar(TestExpression_EvaluateNumericConstant):

    def setUp(self):
        #
        # Create Model
        #
        TestExpression_EvaluateNumericConstant.setUp(self)
        #
        # Create model instance
        #
        self.expectExpression = True
        self.expectConstExpression = False

    def create(self, val, domain):
        tmp=Var(name="unknown", domain=domain)
        tmp.construct()
        tmp.fixed=True
        tmp.value=val
        return tmp


class TestExpression_EvaluateImmutableParam(TestExpression_EvaluateNumericConstant):

    def setUp(self):
        #
        # Create Model
        #
        TestExpression_EvaluateNumericConstant.setUp(self)
        #
        # Create model instance
        #
        self.expectExpression = False
        self.expectConstExpression = True

    def create(self, val, domain):
        tmp=Param(default=val, mutable=False, within=domain)
        tmp.construct()
        return tmp


class TestExpression_Evaluate_MutableParam(TestExpression_EvaluateNumericConstant):

    def setUp(self):
        #
        # Create Model
        #
        TestExpression_EvaluateNumericConstant.setUp(self)
        #
        # Create model instance
        #
        self.expectExpression = True
        self.expectConstExpression = False

    def create(self, val, domain):
        tmp=Param(default=val, mutable=True, within=domain)
        tmp.construct()
        return tmp


class TestExpression_Intrinsic(unittest.TestCase):

    def test_abs_numval(self):
        e = abs(1.5)
        self.assertAlmostEqual(value(e), 1.5)
        e = abs(-1.5)
        self.assertAlmostEqual(value(e), 1.5)

    def test_abs_param(self):
        m = ConcreteModel()
        m.p = Param(initialize=1.5)
        e = abs(m.p)
        self.assertAlmostEqual(value(e), 1.5)
        m.q = Param(initialize=-1.5)
        e = abs(m.q)
        self.assertAlmostEqual(value(e), 1.5)

    def test_abs_mutableparam(self):
        m = ConcreteModel()
        m.p = Param(initialize=0, mutable=True)
        m.p.value = 1.5
        e = abs(m.p)
        self.assertAlmostEqual(value(e), 1.5)
        m.p.value = -1.5
        e = abs(m.p)
        self.assertAlmostEqual(value(e), 1.5)
        self.assertIs(e.is_potentially_variable(), False)

    def test_ceil_numval(self):
        e = ceil(1.5)
        self.assertAlmostEqual(value(e), 2.0)
        e = ceil(-1.5)
        self.assertAlmostEqual(value(e), -1.0)

    def test_ceil_param(self):
        m = ConcreteModel()
        m.p = Param(initialize=1.5)
        e = ceil(m.p)
        self.assertAlmostEqual(value(e), 2.0)
        m.q = Param(initialize=-1.5)
        e = ceil(m.q)
        self.assertAlmostEqual(value(e), -1.0)

    def test_ceil_mutableparam(self):
        m = ConcreteModel()
        m.p = Param(initialize=0, mutable=True)
        m.p.value = 1.5
        e = ceil(m.p)
        self.assertAlmostEqual(value(e), 2.0)
        m.p.value = -1.5
        e = ceil(m.p)
        self.assertAlmostEqual(value(e), -1.0)
        self.assertIs(e.is_potentially_variable(), False)

    def test_ceil(self):
        m = ConcreteModel()
        m.v = Var()
        e = ceil(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1.5
        self.assertAlmostEqual(value(e), 2.0)
        m.v.value = -1.5
        self.assertAlmostEqual(value(e), -1.0)
        self.assertIs(e.is_potentially_variable(), True)

    def test_floor(self):
        m = ConcreteModel()
        m.v = Var()
        e = floor(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1.5
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = -1.5
        self.assertAlmostEqual(value(e), -2.0)

    def test_exp(self):
        m = ConcreteModel()
        m.v = Var()
        e = exp(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1
        self.assertAlmostEqual(value(e), math.e)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 1.0)

    def test_log(self):
        m = ConcreteModel()
        m.v = Var()
        e = log(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1
        self.assertAlmostEqual(value(e), 0)
        m.v.value = math.e
        self.assertAlmostEqual(value(e), 1)

    def test_log10(self):
        m = ConcreteModel()
        m.v = Var()
        e = log10(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1
        self.assertAlmostEqual(value(e), 0)
        m.v.value = 10
        self.assertAlmostEqual(value(e), 1)

    def test_pow(self):
        m = ConcreteModel()
        m.v = Var()
        m.p = Param(mutable=True)
        e = pow(m.v,m.p)
        self.assertEqual(e.__class__, PowExpression)
        m.v.value = 2
        m.p.value = 0
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = 2
        m.p.value = 1
        self.assertAlmostEqual(value(e), 2.0)

    def test_sqrt(self):
        m = ConcreteModel()
        m.v = Var()
        e = sqrt(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = 4
        self.assertAlmostEqual(value(e), 2.0)

    def test_sin(self):
        m = ConcreteModel()
        m.v = Var()
        e = sin(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = math.pi/2.0 
        self.assertAlmostEqual(value(e), 1.0)

    def test_cos(self):
        m = ConcreteModel()
        m.v = Var()
        e = cos(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = math.pi/2.0 
        self.assertAlmostEqual(value(e), 0.0)

    def test_tan(self):
        m = ConcreteModel()
        m.v = Var()
        e = tan(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = math.pi/4.0 
        self.assertAlmostEqual(value(e), 1.0)

    def test_asin(self):
        m = ConcreteModel()
        m.v = Var()
        e = asin(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), math.pi/2.0)

    def test_acos(self):
        m = ConcreteModel()
        m.v = Var()
        e = acos(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 0.0 
        self.assertAlmostEqual(value(e), math.pi/2.0)

    def test_atan(self):
        m = ConcreteModel()
        m.v = Var()
        e = atan(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), math.pi/4.0)

    def test_sinh(self):
        m = ConcreteModel()
        m.v = Var()
        e = sinh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), (math.e-1.0/math.e)/2.0)

    def test_cosh(self):
        m = ConcreteModel()
        m.v = Var()
        e = cosh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 1.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), (math.e+1.0/math.e)/2.0)

    def test_tanh(self):
        m = ConcreteModel()
        m.v = Var()
        e = tanh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), (math.e-1.0/math.e)/(math.e+1.0/math.e))

    def test_asinh(self):
        m = ConcreteModel()
        m.v = Var()
        e = asinh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = (math.e-1.0/math.e)/2.0
        self.assertAlmostEqual(value(e), 1.0)

    def test_acosh(self):
        m = ConcreteModel()
        m.v = Var()
        e = acosh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 1.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = (math.e+1.0/math.e)/2.0
        self.assertAlmostEqual(value(e), 1.0)

    def test_atanh(self):
        m = ConcreteModel()
        m.v = Var()
        e = atanh(m.v)
        self.assertEqual(e.__class__, UnaryFunctionExpression)
        m.v.value = 0.0
        self.assertAlmostEqual(value(e), 0.0)
        m.v.value = (math.e-1.0/math.e)/(math.e+1.0/math.e)
        self.assertAlmostEqual(value(e), 1.0)


class TestNumericValue(unittest.TestCase):

    def test_asnum(self):
        try:
            as_numeric(None)
            self.fail("test_asnum - expected TypeError")
        except TypeError:
            pass

    def test_vals(self):
        #
        # Check that we can get the value from a numeric constant
        #
        a = NumericConstant(1.1)
        b = float(value(a))
        self.assertEqual(b,1.1)
        b = int(value(a))
        self.assertEqual(b,1)

    def test_ops(self):
        #
        # Verify that we can compare the value of numeric constants
        #
        a = NumericConstant(1.1)
        b = NumericConstant(2.2)
        c = NumericConstant(-2.2)
        #a <= b
        self.assertEqual(a() <= b(), True)
        self.assertEqual(a() >= b(), False)
        self.assertEqual(a() == b(), False)
        self.assertEqual(abs(a() + b()-3.3) <= 1e-7, True)
        self.assertEqual(abs(b() - a()-1.1) <= 1e-7, True)
        self.assertEqual(abs(b() * 3-6.6) <= 1e-7, True)
        self.assertEqual(abs(b() / 2-1.1) <= 1e-7, True)
        self.assertEqual(abs(abs(-b())-2.2) <= 1e-7, True)
        self.assertEqual(abs(c()), 2.2)
        #
        # Check that we can get the string representation for a numeric
        # constant.
        #
        self.assertEqual(str(c), "-2.2")

    def test_var(self):
        M = ConcreteModel()
        M.x = Var()
        e = M.x + 2
        self.assertRaises(ValueError, value, M.x)
        self.assertEqual(e(exception=False), None)


class TestGenerate_SumExpression(unittest.TestCase):

    def test_simpleSum(self):
        # a + b
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        e = m.a + m.b
        #
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)

        self.assertRaises(KeyError, e.arg, 3)
        self.assertIs(e.arg(-1), m.b)

    def test_simpleSum_API(self):
        m = ConcreteModel()
        m.a = Var()
        m.b = Var()
        e = m.a + m.b
        e += (2*m.a)
        self.assertIs(e.nargs(), 3)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(type(e.arg(2)), MonomialTermExpression)
        self.assertEqual(id(e.arg(-1)), id(e.arg(2)))

    def test_constSum(self):
        # a + 5
        m = AbstractModel()
        m.a = Var()
        e = m.a + 5
        #
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), 5)
        self.assertEqual(e.size(), 3)

        e = 5 + m.a
        #
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), 5)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e.size(), 3)

    def test_nestedSum(self):
        #
        # Check the structure of nested sums
        #
        expectedType = SumExpression

        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #           +
        #          / \
        #         +   5
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = e1 + 5
        #
        self.assertIs(type(e), expectedType)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(2), 5)
        self.assertEqual(e.size(), 4)

        #       + 
        #      / \ 
        #     5   +
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = 5 + e1
        #
        self.assertIs(type(e), expectedType)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(2), 5)
        self.assertEqual(e.size(), 4)

        #           +
        #          / \
        #         +   c
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = e1 + m.c
        #
        self.assertIs(type(e), expectedType)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(2), m.c)
        self.assertEqual(e.size(), 4)

        #       + 
        #      / \ 
        #     c   +
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = m.c + e1
        #
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(2), m.c)
        self.assertEqual(e.size(), 4)

        #            +
        #          /   \
        #         +     +
        #        / \   / \
        #       a   b c   d
        e1 = m.a + m.b
        e2 = m.c + m.d
        e = e1 + e2
        #
        self.assertIs(type(e), expectedType)
        self.assertEqual(e.nargs(), 4)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(2), m.c)
        self.assertIs(e.arg(3), m.d)
        self.assertEqual(e.size(), 5)

    def test_nestedSum2(self):
        #
        # Check the structure of nested sums
        #
        expectedType = SumExpression

        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #           +
        #          / \
        #         *   c
        #        / \
        #       2   +
        #          / \
        #         a   b 
        e1 = m.a + m.b
        e = 2*e1 + m.c

        self.assertIs(type(e), expectedType)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0).arg(1), e1)
        self.assertIs(e.arg(1), m.c)
        self.assertEqual(e.size(), 7)

        #         *
        #        / \
        #       3   +
        #          / \
        #         *   c
        #        / \
        #       2   +
        #          / \
        #         a   b 
        e1 = m.a + m.b
        e = 3*(2*e1 + m.c)

        self.assertIs(type(e.arg(1)), expectedType)
        self.assertEqual(e.arg(1).nargs(), 2)
        self.assertIs(e.arg(1).arg(0).arg(1), e1)
        self.assertIs(e.arg(1).arg(1), m.c)
        self.assertEqual(e.size(), 9)

    def test_trivialSum(self):
        #
        # Check that adding zero doesn't change the expression
        #
        m = AbstractModel()
        m.a = Var()
        e = m.a + 0
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)

        e = 0 + m.a
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)
        #
        # Adding zero to a viewsum will not change the sum
        #
        e = m.a + m.a
        f = e + 0
        self.assertEqual(id(e), id(f))

    def test_sumOf_nestedTrivialProduct(self):
        #
        # Check sums with nested products
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()

        #       +
        #      / \
        #     *   b
        #    / \
        #   a   5
        e1 = m.a * 5
        e = e1 + m.b
        #
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0).arg(0), 5)
        self.assertIs(e.arg(0).arg(1), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       +
        #      / \
        #     b   *
        #        / \
        #       a   5
        e = m.b + e1
        #
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1).arg(0), 5)
        self.assertIs(e.arg(1).arg(1), m.a)
        self.assertEqual(e.size(), 5)

        #            +
        #          /   \
        #         *     +
        #        / \   / \
        #       a   5 b   c
        e2 = m.b + m.c
        e = e1 + e2
        #
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.c)
        self.assertIs(e.arg(2).arg(0), 5)
        self.assertIs(e.arg(2).arg(1), m.a)
        self.assertEqual(e.size(), 6)

        #            +
        #          /   \
        #         +     *
        #        / \   / \
        #       b   c a   5
        e2 = m.b + m.c
        e = e2 + e1

        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.c)
        self.assertIs(e.arg(2).arg(0), 5)
        self.assertIs(e.arg(2).arg(1), m.a)
        self.assertEqual(e.size(), 6)

    def test_simpleDiff(self):
        #
        # Check the structure of a simple difference with two variables
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()

        #    -
        #   / \
        #  a   b
        e = m.a - m.b
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.b)

    def test_constDiff(self):
        #
        # Check the structure of a simple difference with a constant
        #
        m = AbstractModel()
        m.a = Var()

        #    -
        #   / \
        #  a   5
        e = m.a - 5
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertEqual(e.arg(1), -5)
        self.assertEqual(e.size(), 3)

        #    -
        #   / \
        #  5   a
        e = 5 - m.a
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), 5)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertIs(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.a)
        self.assertEqual(e.size(), 5)

    def test_paramDiff(self):
        #
        # Check the structure of a simple difference with a constant
        #
        m = AbstractModel()
        m.a = Var()
        m.p = Param()

        #    -
        #   / \
        #  a   p
        e = m.a - m.p
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(type(e.arg(1)), NPV_NegationExpression)
        self.assertIs(e.arg(1).arg(0), m.p)
        self.assertEqual(e.size(), 4)

        #      -
        #     / \
        #  m.p   a
        e = m.p - m.a
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.p)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.a)
        self.assertEqual(e.size(), 5)

    def test_constparamDiff(self):
        #
        # Check the structure of a simple difference with a constant
        #
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(initialize=0)

        #    -
        #   / \
        #  a   p
        e = m.a - m.p
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)

        #      -
        #     / \
        #  m.p   a
        e = m.p - m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), -1)
        self.assertIs(e.arg(1), m.a)

    def test_termDiff(self):
        #
        # Check the structure of a simple difference with a term
        #
        m = ConcreteModel()
        m.a = Var()

        #
        #   -
        #  / \
        # 5   *
        #    / \
        #   2   a
        #

        e = 5 - 2*m.a

        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.arg(0), 5)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -2)
        self.assertIs(e.arg(1).arg(1), m.a)

    def test_nestedDiff(self):
        #
        # Check the structure of nested differences
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #       -
        #      / \
        #     -   5
        #    / \
        #   a   b
        e1 = m.a - m.b
        e = e1 - 5
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1).__class__, MonomialTermExpression)
        self.assertIs(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertIs(e.arg(2), -5)
        self.assertEqual(e.size(), 6)

        #       -
        #      / \
        #     5   -
        #        / \
        #       a   b
        e1 = m.a - m.b
        e = 5 - e1
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), 5)
        self.assertIs(type(e.arg(1)), NegationExpression)
        self.assertIs(e.arg(1).arg(0), e1)
        self.assertEqual(e.size(), 8)

        #       -
        #      / \
        #     -   c
        #    / \
        #   a   b
        e1 = m.a - m.b
        e = e1 - m.c
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1).__class__, MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertIs(type(e.arg(2)), MonomialTermExpression)
        self.assertEqual(e.arg(2).arg(0), -1)
        self.assertIs(e.arg(2).arg(1), m.c)
        self.assertEqual(e.size(), 8)

        #       -
        #      / \
        #     c   -
        #        / \
        #       a   b
        e1 = m.a - m.b
        e = m.c - e1
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), m.c)
        self.assertIs(type(e.arg(1)), NegationExpression)
        self.assertIs(e.arg(1).arg(0), e1)
        self.assertEqual(e.size(), 8)

        #            -
        #          /   \
        #         -     -
        #        / \   / \
        #       a   b c   d
        e1 = m.a - m.b
        e2 = m.c - m.d
        e = e1 - e2
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertIs(e.arg(2).arg(0), e2)
        self.assertEqual(e.size(), 11)

    def test_negation_param(self):
        #
        # Check logic for negations with uninitialize params
        #
        m = AbstractModel()
        m.p = Param()
        e = - m.p
        self.assertIs(type(e), NPV_NegationExpression)
        e = - e
        self.assertTrue(isinstance(e, Param))

    def test_negation_mutableparam(self):
        #
        # Check logic for negations with mutable params
        #
        m = AbstractModel()
        m.p = Param(mutable=True, initialize=1.0)
        e = - m.p
        self.assertIs(type(e), NPV_NegationExpression)
        e = - e
        self.assertTrue(isinstance(e, Param))

    def test_negation_terms(self):
        #
        # Check logic for negations with mutable params
        #
        m = AbstractModel()
        m.v = Var()
        m.p = Param(mutable=True, initialize=1.0)
        e = - m.p*m.v
        self.assertIs(type(e), MonomialTermExpression)
        self.assertIs(type(e.arg(0)), NPV_NegationExpression)
        e = - e
        self.assertIs(type(e), MonomialTermExpression)
        self.assertIs(type(e.arg(0)), NPV_NegationExpression)
        #
        e = - 5*m.v
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.arg(0), -5)
        e = - e
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.arg(0), 5)

    def test_trivialDiff(self):
        #
        # Check that subtracting zero doesn't change the expression
        #
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(mutable=True)

        # a - 0
        e = m.a - 0
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)

        # 0 - a
        e = 0 - m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), -1)
        self.assertIs(e.arg(1), m.a)

        # p - 0
        e = m.p - 0
        self.assertIs(type(e), type(m.p))
        self.assertIs(e, m.p)

        # 0 - p
        e = 0 - m.p
        self.assertIs(type(e), NPV_NegationExpression)
        self.assertEqual(e.nargs(), 1)
        self.assertIs(e.arg(0), m.p)

        # 0 - 5*a
        e = 0 - 5*m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), -5)

        # 0 - p*a
        e = 0 - m.p*m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), NPV_NegationExpression)
        self.assertIs(e.arg(0).arg(0), m.p)

        # 0 - a*a
        e = 0 - m.a*m.a
        self.assertIs(type(e), NegationExpression)
        self.assertEqual(e.nargs(), 1)
        self.assertIs(type(e.arg(0)), ProductExpression)

    def test_sumOf_nestedTrivialProduct2(self):
        #
        # Check the structure of sum of products
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()

        #       -
        #      / \
        #     *   b
        #    / \
        #   a   5
        e1 = m.a * 5
        e = e1 - m.b
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), e1)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 7)

        #       -
        #      / \
        #     b   *
        #        / \
        #       a   5
        e1 = m.a * 5
        e = m.b - e1
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1).arg(0), -5)
        self.assertIs(e.arg(1).arg(1), m.a)
        self.assertEqual(e.size(), 5)

        #            -
        #          /   \
        #         *     -
        #        / \   / \
        #       a   5 b   c
        e1 = m.a * 5
        e2 = m.b - m.c
        e = e1 - e2
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), e1)
        self.assertIs(type(e.arg(1)), NegationExpression)
        self.assertIs(e.arg(1).arg(0), e2)
        self.assertEqual(e.size(), 10)

        #            -
        #          /   \
        #         -     *
        #        / \   / \
        #       b   c a   5
        e1 = m.a * 5
        e2 = m.b - m.c
        e = e2 - e1
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.c)
        self.assertIs(e.arg(2).arg(0), -5)
        self.assertIs(e.arg(2).arg(1), m.a)
        self.assertEqual(e.size(), 8)

    def test_sumOf_nestedTrivialProduct2(self):
        #
        # Check the structure of sum of products
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.p = Param(initialize=5, mutable=True)

        #       -
        #      / \
        #     *   b
        #    / \
        #   a   5
        e1 = m.a * m.p
        e = e1 - m.b
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), e1)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 7)

        #       -
        #      / \
        #     b   *
        #        / \
        #       a   5
        e1 = m.a * m.p
        e = m.b - e1
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertIs(type(e.arg(1).arg(0)), NPV_NegationExpression)
        self.assertIs(e.arg(1).arg(0).arg(0), m.p)
        self.assertIs(e.arg(1).arg(1), m.a)
        self.assertEqual(e.size(), 6)

        #            -
        #          /   \
        #         *     -
        #        / \   / \
        #       a   5 b   c
        e1 = m.a * m.p
        e2 = m.b - m.c
        e = e1 - e2
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), e1)
        self.assertIs(type(e.arg(1)), NegationExpression)
        self.assertIs(e.arg(1).arg(0), e2)
        self.assertEqual(e.size(), 10)

        #            -
        #          /   \
        #         -     *
        #        / \   / \
        #       b   c a   5
        e1 = m.a * m.p
        e2 = m.b - m.c
        e = e2 - e1
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.c)
        self.assertIs(type(e.arg(2)), MonomialTermExpression)
        self.assertIs(type(e.arg(2).arg(0)), NPV_NegationExpression)
        self.assertIs(e.arg(2).arg(0).arg(0), m.p)
        self.assertIs(e.arg(2).arg(1), m.a)
        self.assertEqual(e.size(), 9)


class TestGenerate_ProductExpression(unittest.TestCase):

    def test_simpleProduct(self):
        #
        # Check the structure of a simple product of variables
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()

        #    *
        #   / \
        #  a   b
        e = m.a * m.b
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)

    def test_constProduct(self):
        #
        # Check the structure of a simple product with a constant
        #
        m = AbstractModel()
        m.a = Var()

        #    *
        #   / \
        #  a   5
        e = m.a * 5
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 5)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e.size(), 3)

        #    *
        #   / \
        #  5   a
        e = 5 * m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), 5)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e.size(), 3)

    def test_nestedProduct(self):
        #
        # Check the structure of nested products
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #       *
        #      / \
        #     *   5
        #    / \
        #   a   b
        e1 = m.a * m.b
        e = e1 * 5
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(1), 5)
        self.assertIs(type(e.arg(0)), ProductExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       *
        #      / \
        #     5   *
        #        / \
        #       a   b
        e1 = m.a * m.b
        e = 5 * e1
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 5)
        self.assertIs(type(e.arg(1)), ProductExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       *
        #      / \
        #     *   c
        #    / \
        #   a   b
        e1 = m.a * m.b
        e = e1 * m.c
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(1), m.c)
        self.assertIs(type(e.arg(0)), ProductExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       *
        #      / \
        #     c   *
        #        / \
        #       a   b
        e1 = m.a * m.b
        e = m.c * e1
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.c)
        self.assertIs(type(e.arg(1)), ProductExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #            *
        #          /   \
        #         *     *
        #        / \   / \
        #       a   b c   d
        e1 = m.a * m.b
        e2 = m.c * m.d
        e = e1 * e2
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), ProductExpression)
        self.assertIs(type(e.arg(1)), ProductExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertIs(e.arg(1).arg(0), m.c)
        self.assertIs(e.arg(1).arg(1), m.d)
        self.assertEqual(e.size(), 7)

    def test_nestedProduct2(self):
        #
        # Check the structure of nested products
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #
        # Check the structure of nested products
        #
        #            *
        #          /   \
        #         +     +
        #        / \   / \
        #       c    +    d
        #           / \
        #          a   b
        e1 = m.a + m.b
        e2 = m.c + e1
        e3 = e1 + m.d
        e = e2 * e3

        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)

        self.assertIs(type(e.arg(0)), SumExpression)
        self.assertIs(e.arg(0).nargs(), 3)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertIs(e.arg(0).arg(2), m.c)

        self.assertIs(type(e.arg(1)), SumExpression)
        self.assertIs(e.arg(1).nargs(), 2)
        self.assertIs(e.arg(1).arg(1), m.d)
        self.assertEqual(e.size(), 10)

        #
        # Check the structure of nested products
        #
        #            *
        #          /   \
        #         *     *
        #        / \   / \
        #       c    +    d
        #           / \
        #          a   b
        e1 = m.a + m.b
        e2 = m.c * e1
        e3 = e1 * m.d
        e = e2 * e3
        #
        self.assertEqual(e.size(), 11)
        #
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)

        self.assertIs(type(e.arg(0)), ProductExpression)
        self.assertIs(e.arg(0).nargs(), 2)
        self.assertIs(e.arg(0).arg(0), m.c)

        self.assertIs(type(e.arg(0).arg(1)), SumExpression)
        self.assertIs(e.arg(0).arg(1).nargs(), 2)
        self.assertIs(e.arg(0).arg(1).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1).arg(1), m.b)

        self.assertIs(type(e.arg(1)), ProductExpression)
        self.assertIs(e.arg(1).nargs(), 2)
        self.assertIs(e.arg(1).arg(1), m.d)

        self.assertIs(type(e.arg(1).arg(0)), SumExpression)
        self.assertIs(e.arg(1).arg(0).nargs(), 2)
        self.assertIs(e.arg(1).arg(0).arg(0), m.a)
        self.assertIs(e.arg(1).arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 11)

    def test_nestedProduct3(self):
        #
        # Check the structure of nested products
        #
        m = AbstractModel()
        m.a = Param(mutable=True)
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #       *
        #      / \
        #     *   5
        #    / \
        #   3   b
        e1 = 3 * m.b
        e = e1 * 5
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 15)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)

        #       *
        #      / \
        #     *   5
        #    / \
        #   a   b
        e1 = m.a * m.b
        e = e1 * 5
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), NPV_ProductExpression)
        self.assertEqual(e.arg(0).arg(0), 5)
        self.assertIs(e.arg(0).arg(1), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       *
        #      / \
        #     5   *
        #        / \
        #       3   b
        e1 = 3 * m.b
        e = 5 * e1
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 15)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)

        #       *
        #      / \
        #     5   *
        #        / \
        #       a   b
        e1 = m.a * m.b
        e = 5 * e1
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), NPV_ProductExpression)
        self.assertEqual(e.arg(0).arg(0), 5)
        self.assertIs(e.arg(0).arg(1), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       *
        #      / \
        #     *   c
        #    / \
        #   a   b
        e1 = m.a * m.b
        e = e1 * m.c
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(1), m.c)
        self.assertIs(type(e.arg(0)), MonomialTermExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       *
        #      / \
        #     c   *
        #        / \
        #       a   b
        e1 = m.a * m.b
        e = m.c * e1
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.c)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #            *
        #          /   \
        #         *     *
        #        / \   / \
        #       a   b c   d
        e1 = m.a * m.b
        e2 = m.c * m.d
        e = e1 * e2
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), MonomialTermExpression)
        self.assertIs(type(e.arg(1)), ProductExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertIs(e.arg(1).arg(0), m.c)
        self.assertIs(e.arg(1).arg(1), m.d)
        self.assertEqual(e.size(), 7)


    def test_trivialProduct(self):
        #
        # Check that multiplying by zero gives zero
        #
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(initialize=0)
        m.q = Param(initialize=1)

        e = m.a * 0
        self.assertIs(type(e), int)
        self.assertEqual(e, 0)

        e = 0 * m.a
        self.assertIs(type(e), int)
        self.assertEqual(e, 0)

        e = m.a * m.p
        self.assertIs(type(e), int)
        self.assertEqual(e, 0)

        e = m.p * m.a
        self.assertIs(type(e), int)
        self.assertEqual(e, 0)

        #
        # Check that multiplying by one gives the original expression
        #
        e = m.a * 1
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)

        e = 1 * m.a
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)

        e = m.a * m.q
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)

        e = m.q * m.a
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)

        #
        # Check that numeric constants are simply muliplied out
        #
        e = NumericConstant(3) * NumericConstant(2)
        self.assertIs(type(e), int)
        self.assertEqual(e, 6)

    def test_simpleDivision(self):
        #
        # Check the structure of a simple division with variables
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()

        #    /
        #   / \
        #  a   b
        e = m.a / m.b
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)

    def test_constDivision(self):
        #
        # Check the structure of a simple division with a constant
        #
        m = AbstractModel()
        m.a = Var()

        #    /
        #   / \
        #  a   5
        e = m.a / 5
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertAlmostEqual(e.arg(0), 0.2)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e.size(), 3)

        #    /
        #   / \
        #  5   a
        e = 5 / m.a
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 5)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e.size(), 3)

    def test_nestedDivision(self):
        #
        # Check the structure of nested divisions
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #       /
        #      / \
        #     *   5
        #    / \
        #   3   b
        e1 = 3 * m.b
        e = e1 / 5
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 3./5)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)

        #       /
        #      / \
        #     /   5
        #    / \
        #   a   b
        e1 = m.a / m.b
        e = e1 / 5
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(1), 5)
        self.assertIs(type(e.arg(0)), DivisionExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       /
        #      / \
        #     5   /
        #        / \
        #       a   b
        e1 = m.a / m.b
        e = 5 / e1
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 5)
        self.assertIs(type(e.arg(1)), DivisionExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       /
        #      / \
        #     /   c
        #    / \
        #   a   b
        e1 = m.a / m.b
        e = e1 / m.c
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(1), m.c)
        self.assertIs(type(e.arg(0)), DivisionExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #       /
        #      / \
        #     c   /
        #        / \
        #       a   b
        e1 = m.a / m.b
        e = m.c / e1
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.c)
        self.assertIs(type(e.arg(1)), DivisionExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)

        #            /
        #          /   \
        #         /     /
        #        / \   / \
        #       a   b c   d
        e1 = m.a / m.b
        e2 = m.c / m.d
        e = e1 / e2
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), DivisionExpression)
        self.assertIs(type(e.arg(1)), DivisionExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertIs(e.arg(1).arg(0), m.c)
        self.assertIs(e.arg(1).arg(1), m.d)
        self.assertEqual(e.size(), 7)

    def test_trivialDivision(self):
        #
        # Check that dividing by zero generates an exception
        #
        m = AbstractModel()
        m.a = Var()
        m.p = Param()
        m.q = Param(initialize=2)
        m.r = Param(mutable=True)
        self.assertRaises(ZeroDivisionError, m.a.__div__, 0)

        #
        # Check that dividing zero by anything non-zero gives zero
        #
        e = 0 / m.a
        self.assertIs(type(e), int)
        self.assertAlmostEqual(e, 0.0)

        #
        # Check that dividing by one 1 gives the original expression
        #
        e = m.a / 1
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)

        #
        # Check the structure dividing 1 by an expression
        #
        e = 1 / m.a
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), 1)
        self.assertIs(e.arg(1), m.a)

        #
        # Check the structure dividing 1 by an expression
        #
        e = 1 / m.p
        self.assertIs(type(e), NPV_DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 1)
        self.assertIs(e.arg(1), m.p)

        #
        # Check the structure dividing 1 by an expression
        #
        e = 1 / m.q
        self.assertIs(type(e), NPV_DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 1)
        self.assertIs(e.arg(1), m.q)

        #
        # Check the structure dividing 1 by an expression
        #
        e = 1 / m.r
        self.assertIs(type(e), NPV_DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 1)
        self.assertIs(e.arg(1), m.r)

        #
        # Check that dividing two non-zero constants gives a constant
        #
        e = NumericConstant(3) / NumericConstant(2)
        self.assertIs(type(e), float)
        self.assertEqual(e, 1.5)


class TestPrettyPrinter_oldStyle(unittest.TestCase):

    _save = None

    def setUp(self):
        # This class tests the Pyomo 5.x expression trees
        TestPrettyPrinter_oldStyle._save = expr_common.TO_STRING_VERBOSE
        expr_common.TO_STRING_VERBOSE = True

    def tearDown(self):
        expr_common.TO_STRING_VERBOSE = TestPrettyPrinter_oldStyle._save

    def test_sum(self):
        #
        # Print simple sum
        #
        model = ConcreteModel()
        model.a = Var()
        model.p = Param(mutable=True)

        expr = 5 + model.a + model.a
        self.assertEqual("sum(5, a, a)", str(expr))

        expr += 5
        self.assertEqual("sum(5, a, a, 5)", str(expr))

        expr = 2 + model.p
        self.assertEqual("sum(2, p)", str(expr))

    def test_linearsum(self):
        #
        # Print simple sum
        #
        model = ConcreteModel()
        A = range(5)
        model.a = Var(A)
        model.p = Param(A, initialize=2, mutable=True)

        expr = quicksum(i*model.a[i] for i in A)
        self.assertEqual("sum(a[1], prod(2, a[2]), prod(3, a[3]), prod(4, a[4]))", str(expr))

        expr = quicksum((i-2)*model.a[i] for i in A)
        self.assertEqual("sum(prod(-2, a[0]), prod(-1, a[1]), a[3], prod(2, a[4]))", str(expr))

        expr = quicksum(model.a[i] for i in A)
        self.assertEqual("sum(a[0], a[1], a[2], a[3], a[4])", str(expr))

        model.p[1].value = 0
        model.p[3].value = 3
        expr = quicksum(model.p[i]*model.a[i] if i != 3 else model.p[i] for i in A)
        self.assertEqual("sum(3, prod(2, a[0]), prod(2, a[2]), prod(2, a[4]))", expression_to_string(expr, compute_values=True))
        self.assertEqual("sum(p[3], prod(p[0], a[0]), prod(p[1], a[1]), prod(p[2], a[2]), prod(p[4], a[4]))", expression_to_string(expr, compute_values=False))

    def test_expr(self):
        #
        # Print simple expressions with products and divisions
        #
        model = ConcreteModel()
        model.a = Var()

        expr = 5 * model.a * model.a
        self.assertEqual("prod(mon(5, a), a)", str(expr))

        # This returns an integer, which has no pprint().
        #expr = expr*0
        #buf = StringIO()
        #EXPR.pprint(ostream=buf)
        #self.assertEqual("0.0", buf.getvalue())

        expr = 5 * model.a / model.a
        self.assertEqual( "div(mon(5, a), a)",
                          str(expr) )

        expr = expr / model.a
        self.assertEqual( "div(div(mon(5, a), a), a)",
                          str(expr) )

        expr = 5 * model.a / model.a / 2
        self.assertEqual( "div(div(mon(5, a), a), 2)",
                          str(expr) )

    def test_other(self):
        #
        # Print other stuff
        #
        model = ConcreteModel()
        model.a = Var()
        model.x = ExternalFunction(library='foo.so', function='bar')

        expr = model.x(model.a, 1, "foo", [])
        self.assertEqual("x(a, 1, 'foo', '[]')", str(expr))

    def test_inequality(self):
        #
        # Print inequalities
        #
        model = ConcreteModel()
        model.a = Var()

        expr = 5 < model.a
        self.assertEqual( "5.0  <  a", str(expr) )

        expr = model.a >= 5
        self.assertEqual( "5.0  <=  a", str(expr) )

        expr = expr < 10
        self.assertEqual( "5.0  <=  a  <  10.0", str(expr) )

        expr = 5 <= model.a + 5
        self.assertEqual( "5.0  <=  sum(a, 5)", str(expr) )

        expr = expr < 10
        self.assertEqual( "5.0  <=  sum(a, 5)  <  10.0", str(expr) )

    def test_equality(self):
        #
        # Print equality
        #
        model = ConcreteModel()
        model.a = Var()
        model.b = Param(initialize=5, mutable=True)

        expr = model.a == model.b
        self.assertEqual( "a  ==  b", str(expr) )

        expr = model.b == model.a
        self.assertEqual( "b  ==  a", str(expr) )

        # NB: since there is no "reverse equality" operator, explicit
        # constants will always show up second.
        expr = 5 == model.a
        self.assertEqual( "a  ==  5.0", str(expr) )

        expr = model.a == 10
        self.assertEqual( "a  ==  10.0", str(expr) )

        expr = 5 == model.a + 5
        self.assertEqual( "sum(a, 5)  ==  5.0", str(expr) )

        expr = model.a + 5 == 5
        self.assertEqual( "sum(a, 5)  ==  5.0", str(expr) )

    def test_getitem(self):
        m = ConcreteModel()
        m.I = RangeSet(1,9)
        m.x = Var(m.I, initialize=lambda m,i: i+1)
        m.P = Param(m.I, initialize=lambda m,i: 10-i, mutable=True)
        t = IndexTemplate(m.I)

        e = m.x[t+m.P[t+1]] + 3
        self.assertEqual("sum(getitem(x, sum({I}, getitem(P, sum({I}, 1)))), 3)", str(e))

    def test_small_expression(self):
        #
        # Print complex
        #
        model = AbstractModel()
        model.a = Var()
        model.b = Param(initialize=2, mutable=True)
        instance=model.create_instance()
        expr = instance.a+1
        expr = expr-1
        expr = expr*instance.a
        expr = expr/instance.a
        expr = expr**instance.b
        expr = 1-expr
        expr = 1+expr
        expr = 2*expr
        expr = 2/expr
        expr = 2**expr
        expr = - expr
        expr = + expr
        expr = abs(expr)
        self.assertEqual(
            "abs(neg(pow(2, div(2, prod(2, sum(1, neg(pow(div(prod(sum(a, 1, -1), a), a), b)), 1))))))",
            str(expr) )


class TestPrettyPrinter_newStyle(unittest.TestCase):

    _save = None

    def setUp(self):
        # This class tests the Pyomo 5.x expression trees
        TestPrettyPrinter_oldStyle._save = expr_common.TO_STRING_VERBOSE
        expr_common.TO_STRING_VERBOSE = False

    def tearDown(self):
        expr_common.TO_STRING_VERBOSE = TestPrettyPrinter_oldStyle._save

    def test_sum(self):
        #
        # Print sum
        #
        model = ConcreteModel()
        model.a = Var()
        model.p = Param(mutable=True)

        expr = 5 + model.a + model.a
        self.assertIs(type(expr), SumExpression)
        self.assertEqual("5 + a + a", str(expr))

        expr += 5
        self.assertIs(type(expr), SumExpression)
        self.assertEqual("5 + a + a + 5", str(expr))

        expr = 2 + model.p
        self.assertEqual("2 + p", str(expr))

        expr = 2 - model.p
        self.assertEqual("2 - p", str(expr))

    def test_linearsum(self):
        #
        # Print simple sum
        #
        model = ConcreteModel()
        A = range(5)
        model.a = Var(A)
        model.p = Param(A, initialize=2, mutable=True)

        expr = quicksum(i*model.a[i] for i in A) + 3
        self.assertEqual("a[1] + 2*a[2] + 3*a[3] + 4*a[4] + 3", str(expr))
        self.assertEqual("a[1] + 2*a[2] + 3*a[3] + 4*a[4] + 3", expression_to_string(expr, compute_values=True))

        expr = quicksum((i-2)*model.a[i] for i in A) + 3
        self.assertEqual("- 2.0*a[0] - a[1] + a[3] + 2*a[4] + 3", str(expr))
        self.assertEqual("- 2.0*a[0] - a[1] + a[3] + 2*a[4] + 3", expression_to_string(expr, compute_values=True))

        expr = quicksum(model.a[i] for i in A) + 3
        self.assertEqual("a[0] + a[1] + a[2] + a[3] + a[4] + 3", str(expr))

        expr = quicksum(model.p[i]*model.a[i] for i in A)
        self.assertEqual("2*a[0] + 2*a[1] + 2*a[2] + 2*a[3] + 2*a[4]", expression_to_string(expr, compute_values=True))
        self.assertEqual("p[0]*a[0] + p[1]*a[1] + p[2]*a[2] + p[3]*a[3] + p[4]*a[4]", expression_to_string(expr, compute_values=False))
        self.assertEqual("p[0]*a[0] + p[1]*a[1] + p[2]*a[2] + p[3]*a[3] + p[4]*a[4]", str(expr))

        model.p[1].value = 0
        model.p[3].value = 3
        expr = quicksum(model.p[i]*model.a[i] if i != 3 else model.p[i] for i in A)
        self.assertEqual("3 + 2*a[0] + 2*a[2] + 2*a[4]", expression_to_string(expr, compute_values=True))
        expr = quicksum(model.p[i]*model.a[i] if i != 3 else -3 for i in A)
        self.assertEqual("-3 + p[0]*a[0] + p[1]*a[1] + p[2]*a[2] + p[4]*a[4]", expression_to_string(expr, compute_values=False))
        
    def test_negation(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()

        e = M.x*(1 + M.y)
        e = - e
        self.assertEqual("- x*(1 + y)", expression_to_string(e))

        M.x = -1
        M.x.fixed = True
        self.assertEqual("(1 + y)", expression_to_string(e, compute_values=True))

    def test_prod(self):
        #
        # Print expressions
        #
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()

        expr = 5 * model.a * model.a
        self.assertEqual("5*a*a", str(expr))

        # This returns an integer, which has no pprint().
        #expr = expr*0
        #buf = StringIO()
        #EXPR.pprint(ostream=buf)
        #self.assertEqual("0.0", buf.getvalue())

        expr = 5 * model.a / model.a
        self.assertEqual( "5*a/a",
                          str(expr) )

        expr = expr / model.a
        self.assertEqual( "5*a/a/a",
                          str(expr) )

        expr = 5 * model.a / (model.a * model.a)
        self.assertEqual( "5*a/(a*a)",
                          str(expr) )

        expr = 5 * model.a / model.a / 2
        self.assertEqual( "5*a/a/2",
                          str(expr) )

        expr = model.a * model.b
        model.a = 1
        model.a.fixed = True
        self.assertEqual( "b", expression_to_string(expr, compute_values=True))

    def test_inequality(self):
        #
        # Print inequalities
        #
        model = ConcreteModel()
        model.a = Var()

        expr = 5 < model.a
        self.assertEqual( "5.0  <  a", str(expr) )

        expr = model.a >= 5
        self.assertEqual( "5.0  <=  a", str(expr) )

        expr = expr < 10
        self.assertEqual( "5.0  <=  a  <  10.0", str(expr) )

        expr = 5 <= model.a + 5
        self.assertEqual( "5.0  <=  a + 5", str(expr) )

        expr = expr < 10
        self.assertEqual( "5.0  <=  a + 5  <  10.0", str(expr) )

    def test_equality(self):
        #
        # Print equalities
        #
        model = ConcreteModel()
        model.a = Var()
        model.b = Param(initialize=5, mutable=True)

        expr = model.a == model.b
        self.assertEqual( "a  ==  b", str(expr) )

        expr = model.b == model.a
        self.assertEqual( "b  ==  a", str(expr) )

        # NB: since there is no "reverse equality" operator, explicit
        # constants will always show up second.
        expr = 5 == model.a
        self.assertEqual( "a  ==  5.0", str(expr) )

        expr = model.a == 10
        self.assertEqual( "a  ==  10.0", str(expr) )

        expr = 5 == model.a + 5
        self.assertEqual( "a + 5  ==  5.0", str(expr) )

        expr = model.a + 5 == 5
        self.assertEqual( "a + 5  ==  5.0", str(expr) )


    def test_linear(self):
        #
        # Print linear
        #
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.p = Param(initialize=2, mutable=True)

        expr = m.x - m.p*m.y
        self.assertEqual( "x - p*y", str(expr) )

        expr = m.x - m.p*m.y + 5
        self.assertIs(type(expr), SumExpression)
        self.assertEqual( "x - p*y + 5", str(expr) )

        expr = m.x - m.p*m.y - 5
        self.assertIs(type(expr), SumExpression)
        self.assertEqual( "x - p*y - 5", str(expr) )

        expr = m.x - m.p*m.y - 5 + m.p
        self.assertIs(type(expr), SumExpression)
        self.assertEqual( "x - p*y - 5 + p", str(expr) )

    def test_expr_if(self):
        m = ConcreteModel()
        m.a = Var()
        m.b = Var()
        expr = Expr_if(IF=m.a + m.b < 20, THEN=m.a, ELSE=m.b)
        self.assertEqual("Expr_if( ( a + b  <  20.0 ), then=( a ), else=( b ) )", str(expr))
        expr = Expr_if(IF=m.a + m.b < 20, THEN=1, ELSE=m.b)
        self.assertEqual("Expr_if( ( a + b  <  20.0 ), then=( 1 ), else=( b ) )", str(expr))

    def test_getitem(self):
        m = ConcreteModel()
        m.I = RangeSet(1,9)
        m.x = Var(m.I, initialize=lambda m,i: i+1)
        m.P = Param(m.I, initialize=lambda m,i: 10-i, mutable=True)
        t = IndexTemplate(m.I)

        e = m.x[t+m.P[t+1]] + 3
        self.assertEqual("x[{I} + P[{I} + 1]] + 3", str(e))

    def test_associativity_rules(self):
        m = ConcreteModel()
        m.w = Var()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        self.assertEqual(str( m.z+m.x+m.y ), "z + x + y")
        self.assertEqual(str( (m.z+m.x)+m.y ), "z + x + y")
        # FIXME: Pyomo currently returns "z + y + x"
        # self.assertEqual(str( m.z+(m.x+m.y) ), "z + x + y")
        self.assertEqual(str( (m.w+m.z)+(m.x+m.y) ), "w + z + x + y")

        self.assertEqual(str( (m.z/m.x)/(m.y/m.w) ), "z/x/(y/w)")

        self.assertEqual(str( m.z/m.x/m.y ), "z/x/y")
        self.assertEqual(str( (m.z/m.x)/m.y ), "z/x/y")
        self.assertEqual(str( m.z/(m.x/m.y) ), "z/(x/y)")

        self.assertEqual(str( m.z*m.x/m.y ), "z*x/y")
        self.assertEqual(str( (m.z*m.x)/m.y ), "z*x/y")
        self.assertEqual(str( m.z*(m.x/m.y) ), "z*(x/y)")

        self.assertEqual(str( m.z/m.x*m.y ), "z/x*y")
        self.assertEqual(str( (m.z/m.x)*m.y ), "z/x*y")
        self.assertEqual(str( m.z/(m.x*m.y) ), "z/(x*y)")

        self.assertEqual(str( m.x**m.y**m.z ), "x**(y**z)")
        self.assertEqual(str( (m.x**m.y)**m.z ), "(x**y)**z")
        self.assertEqual(str( m.x**(m.y**m.z) ), "x**(y**z)")


    def test_small_expression(self):
        #
        # Print complex expression
        #
        model = AbstractModel()
        model.a = Var()
        model.b = Param(initialize=2, mutable=True)
        instance=model.create_instance()
        expr = instance.a+1
        expr = expr-1
        expr = expr*instance.a
        expr = expr/instance.a
        expr = expr**instance.b
        expr = 1-expr
        expr = 1+expr
        expr = 2*expr
        expr = 2/expr
        expr = 2**expr
        expr = - expr
        expr = + expr
        expr = abs(expr)
        self.assertEqual(
            "abs(- 2**(2/(2*(1 - ((a + 1 - 1)*a/a)**b + 1))))",
            str(expr) )

    def test_large_expression(self):
        #
        # Diff against a large model
        #
        def c1_rule(model):
            return (1.0,model.b[1],None)
        def c2_rule(model):
            return (None,model.b[1],0.0)
        def c3_rule(model):
            return (0.0,model.b[1],1.0)
        def c4_rule(model):
            return (3.0,model.b[1])
        def c5_rule(model, i):
            return (model.b[i],0.0)

        def c6a_rule(model):
            return 0.0 <= model.c
        def c7a_rule(model):
            return model.c <= 1.0
        def c7b_rule(model):
            return model.c >= 1.0
        def c8_rule(model):
            return model.c == 2.0
        def c9a_rule(model):
            return model.A+model.A <= model.c
        def c9b_rule(model):
            return model.A+model.A >= model.c
        def c10a_rule(model):
            return model.c <= model.B+model.B
        def c11_rule(model):
            return model.c == model.A+model.B
        def c15a_rule(model):
            return model.A <= model.A*model.d
        def c16a_rule(model):
            return model.A*model.d <= model.B

        def c12_rule(model):
            return model.c == model.d
        def c13a_rule(model):
            return model.c <= model.d
        def c14a_rule(model):
            return model.c >= model.d

        def cl_rule(model, i):
            if i > 10:
                return ConstraintList.End
            return i* model.c >= model.d

        def o2_rule(model, i):
            return model.b[i]
        model=AbstractModel()
        model.a = Set(initialize=[1,2,3])
        model.b = Var(model.a,initialize=1.1,within=PositiveReals)
        model.c = Var(initialize=2.1, within=PositiveReals)
        model.d = Var(initialize=3.1, within=PositiveReals)
        model.e = Var(initialize=4.1, within=PositiveReals)
        model.A = Param(default=-1, mutable=True)
        model.B = Param(default=-2, mutable=True)
        #model.o1 = Objective()
        model.o2 = Objective(model.a,rule=o2_rule)
        model.o3 = Objective(model.a,model.a)
        model.c1 = Constraint(rule=c1_rule)
        model.c2 = Constraint(rule=c2_rule)
        model.c3 = Constraint(rule=c3_rule)
        model.c4 = Constraint(rule=c4_rule)
        model.c5 = Constraint(model.a,rule=c5_rule)

        model.c6a = Constraint(rule=c6a_rule)
        model.c7a = Constraint(rule=c7a_rule)
        model.c7b = Constraint(rule=c7b_rule)
        model.c8 = Constraint(rule=c8_rule)
        model.c9a = Constraint(rule=c9a_rule)
        model.c9b = Constraint(rule=c9b_rule)
        model.c10a = Constraint(rule=c10a_rule)
        model.c11 = Constraint(rule=c11_rule)
        model.c15a = Constraint(rule=c15a_rule)
        model.c16a = Constraint(rule=c16a_rule)

        model.c12 = Constraint(rule=c12_rule)
        model.c13a = Constraint(rule=c13a_rule)
        model.c14a = Constraint(rule=c14a_rule)

        model.cl = ConstraintList(rule=cl_rule)

        instance=model.create_instance()
        OUTPUT=open(currdir+"varpprint.out","w")
        instance.pprint(ostream=OUTPUT)
        OUTPUT.close()
        self.assertFileEqualsBaseline( currdir+"varpprint.out",
                                       currdir+"varpprint.txt" )

    def test_labeler(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.z = Var()
        M.a = Var(range(3))
        M.p = Param(range(3), initialize=2)
        M.q = Param(range(3), initialize=3, mutable=True)

        e = M.x*M.y + sum_product(M.p, M.a) + quicksum(M.q[i]*M.a[i] for i in M.a) / M.x
        self.assertEqual(
            str(e),
            "x*y + (2*a[0] + 2*a[1] + 2*a[2]) + (q[0]*a[0] + q[1]*a[1] + q[2]*a[2])/x")
        self.assertEqual(
            e.to_string(),
            "x*y + (2*a[0] + 2*a[1] + 2*a[2]) + (q[0]*a[0] + q[1]*a[1] + q[2]*a[2])/x")
        self.assertEqual(
            e.to_string(compute_values=True),
            "x*y + (2*a[0] + 2*a[1] + 2*a[2]) + (3*a[0] + 3*a[1] + 3*a[2])/x")

        labeler = NumericLabeler('x')
        self.assertEqual(
            expression_to_string(e, labeler=labeler),
            "x1*x2 + (2*x3 + 2*x4 + 2*x5) + (q[0]*x3 + q[1]*x4 + q[2]*x5)/x1")

        from pyomo.core.expr.symbol_map import SymbolMap
        labeler = NumericLabeler('x')
        smap = SymbolMap(labeler)
        self.assertEqual(
            expression_to_string(e, smap=smap),
            "x1*x2 + (2*x3 + 2*x4 + 2*x5) + (q[0]*x3 + q[1]*x4 + q[2]*x5)/x1")
        self.assertEqual(
            expression_to_string(e, smap=smap, compute_values=True),
            "x1*x2 + (2*x3 + 2*x4 + 2*x5) + (3*x3 + 3*x4 + 3*x5)/x1")

#
# TODO:What is this checking?
#
class TestInplaceExpressionGeneration(unittest.TestCase):

    def setUp(self):
        # This class tests the Pyomo 5.x expression trees

        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        self.m = m

    def tearDown(self):
        self.m = None

    def test_iadd(self):
        m = self.m
        x = 0

        x += m.a
        self.assertIs(type(x), type(m.a))

        x += m.a
        self.assertIs(type(x), SumExpression)
        self.assertEqual(x.nargs(), 2)

        x += m.b
        self.assertIs(type(x), SumExpression)
        self.assertEqual(x.nargs(), 3)


    def test_isub(self):
        m = self.m

        x = m.a
        x -= 0
        self.assertIs(type(x), type(m.a))

        x = 0
        x -= m.a
        self.assertIs(type(x), MonomialTermExpression)
        self.assertEqual(x.nargs(), 2)

        x -= m.a
        self.assertIs(type(x), SumExpression)
        self.assertEqual(x.nargs(), 2)

        x -= m.a
        self.assertIs(type(x), SumExpression)
        self.assertEqual(x.nargs(), 3)

        x -= m.b
        self.assertIs(type(x), SumExpression)
        self.assertEqual(x.nargs(), 4)

    def test_imul(self):
        m = self.m
        x = 1

        x *= m.a
        self.assertIs(type(x), type(m.a))

        x *= m.a
        self.assertIs(type(x), ProductExpression)
        self.assertEqual(x.nargs(), 2)

        x *= m.a
        self.assertIs(type(x), ProductExpression)
        self.assertEqual(x.nargs(), 2)

    def test_idiv(self):
        m = self.m
        x = 1

        x /= m.a
        self.assertIs(type(x), DivisionExpression)
        self.assertEqual(x.arg(0), 1)
        self.assertIs(x.arg(1), m.a)

        x /= m.a
        self.assertIs(type(x), DivisionExpression)
        self.assertIs(type(x.arg(0)), DivisionExpression)
        self.assertIs(x.arg(0).arg(1), m.a)
        self.assertIs(x.arg(1), m.a)

    def test_ipow(self):
        m = self.m
        x = 1

        x **= m.a
        self.assertIs(type(x), PowExpression)
        self.assertEqual(x.nargs(), 2)
        self.assertEqual(value(x.arg(0)), 1)
        self.assertIs(x.arg(1), m.a)

        x **= m.b
        self.assertIs(type(x), PowExpression)
        self.assertEqual(x.nargs(), 2)
        self.assertIs(type(x.arg(0)), PowExpression)
        self.assertIs(x.arg(1), m.b)
        self.assertEqual(x.nargs(), 2)
        self.assertEqual(value(x.arg(0).arg(0)), 1)
        self.assertIs(x.arg(0).arg(1), m.a)

        # If someone else holds a reference to the expression, we still
        # need to clone it:
        x = 1 ** m.a
        y = x
        x **= m.b
        self.assertIs(type(y), PowExpression)
        self.assertEqual(y.nargs(), 2)
        self.assertEqual(value(y.arg(0)), 1)
        self.assertIs(y.arg(1), m.a)

        self.assertIs(type(x), PowExpression)
        self.assertEqual(x.nargs(), 2)
        self.assertIs(type(x.arg(0)), PowExpression)
        self.assertIs(x.arg(1), m.b)
        self.assertEqual(x.nargs(), 2)
        self.assertEqual(value(x.arg(0).arg(0)), 1)
        self.assertIs(x.arg(0).arg(1), m.a)


class TestGeneralExpressionGeneration(unittest.TestCase):

    def test_invalidIndexing(self):
        #
        # Check for errors when generating expressions with invalid indices
        #
        m = AbstractModel()
        m.A = Set()
        m.p = Param(m.A, mutable=True)
        m.x = Var(m.A)
        m.z = Var()

        try:
            m.p * 2
            self.fail("Expected m.p*2 to raise a TypeError")
        except TypeError:
            pass

        try:
            m.x * 2
            self.fail("Expected m.x*2 to raise a TypeError")
        except TypeError:
            pass

        try:
            2 * m.p
            self.fail("Expected 2*m.p to raise a TypeError")
        except TypeError:
            pass

        try:
            2 * m.x
            self.fail("Expected 2*m.x to raise a TypeError")
        except TypeError:
            pass

        try:
            m.z * m.p
            self.fail("Expected m.z*m.p to raise a TypeError")
        except TypeError:
            pass
        except ValueError:
            pass

        try:
            m.z * m.x
            self.fail("Expected m.z*m.x to raise a TypeError")
        except TypeError:
            pass

    def test_negation(self):
        #
        # Test negation logic for various expressions
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()

        e = -m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), -1)
        self.assertIs(e.arg(1), m.a)

        e1 = m.a - m.b
        e = -e1
        self.assertIs(type(e), NegationExpression)
        self.assertIs(e.arg(0), e1)
        self.assertIs(type(e.arg(0)), SumExpression)

        e1 = m.a * m.b
        e = -e1
        self.assertIs(type(e), NegationExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)

        e1 = sin(m.a)
        e = -e1
        self.assertIs(type(e), NegationExpression)
        self.assertIs(type(e.arg(0)), UnaryFunctionExpression)


class TestExprConditionalContext(unittest.TestCase):


    def tearDown(self):
        # Make sure errors here don't bleed over to other tests
        if logical_expr._using_chained_inequality:
            logical_expr._chainedInequality.prev = None

    def checkCondition(self, expr, expectedValue):
        try:
            if expr:
                if not logical_expr._using_chained_inequality and expectedValue != True:
                    self.fail("__nonzero__ returned the wrong condition value"
                              " (expected %s)" % expectedValue)
            else:
                if expectedValue != False:
                    self.fail("__nonzero__ returned the wrong condition value"
                              " (expected %s)" % expectedValue)
            if expectedValue is None:
                self.fail("Expected ValueError because component was undefined")
        except ValueError:
            if expectedValue is not None:
                raise
        finally:
            if logical_expr._using_chained_inequality:
                logical_expr._chainedInequality.prev = None

    def test_immutable_paramConditional(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=False)
        #
        try:
            self.checkCondition(model.p > 0, True)
            if not logical_expr._using_chained_inequality:
                self.fail("Expected ValueError because the parameter is unconstructed.")
        except ValueError:
            pass
        #self.checkCondition(model.p >= 0, True)
        #self.checkCondition(model.p < 1, True)
        #self.checkCondition(model.p <= 1, True)
        #self.checkCondition(model.p == 0, None)

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the parameter is initialized
        #
        try:
            self.checkCondition(model.p > 0, True)
            if not logical_expr._using_chained_inequality:
                self.fail("Expected ValueError because the parameter is unconstructed.")
        except ValueError:
            pass
        #self.checkCondition(model.p >= 0, True)
        #self.checkCondition(model.p < 1, True)
        #self.checkCondition(model.p <= 1, True)
        #self.checkCondition(model.p == 0, None)

        instance = model.create_instance()
        self.checkCondition(instance.p > 0, True)
        self.checkCondition(instance.p > 2, False)
        self.checkCondition(instance.p >= 1, True)
        self.checkCondition(instance.p >= 2, False)
        self.checkCondition(instance.p < 2, True)
        self.checkCondition(instance.p < 0, False)
        self.checkCondition(instance.p <= 1, True)
        self.checkCondition(instance.p <= 0, False)
        self.checkCondition(instance.p == 1, True)
        self.checkCondition(instance.p == 2, False)

    def test_immutable_paramConditional_reversed(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=False)
        #
        # TODO: Inequalities evaluate True when the parameter is unconstructed?
        #
        self.checkCondition(0 < model.p, True)
        self.checkCondition(0 <= model.p, True)
        self.checkCondition(1 > model.p, True)
        self.checkCondition(1 >= model.p, True)
        self.checkCondition(0 == model.p, None)

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the parameter is initialized
        #
        self.checkCondition(0 < instance.p, True)
        self.checkCondition(2 < instance.p, False)
        self.checkCondition(1 <= instance.p, True)
        self.checkCondition(instance.p > 0, True)
        self.checkCondition(instance.p > 2, False)
        self.checkCondition(instance.p >= 1, True)
        self.checkCondition(instance.p >= 2, False)
        self.checkCondition(instance.p < 2, True)
        self.checkCondition(instance.p < 0, False)
        self.checkCondition(instance.p <= 1, True)
        self.checkCondition(instance.p <= 0, False)
        self.checkCondition(instance.p == 1, True)
        self.checkCondition(instance.p == 2, False)

    def test_immutable_paramConditional_reversed(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=False)
        #
        try:
            self.checkCondition(0 < model.p, True)
            if not logical_expr._using_chained_inequality:
                self.fail("Expected ValueError because the parameter value is being accessed before it is constructed.")
        except ValueError:
            pass
        #self.checkCondition(0 <= model.p, True)
        #self.checkCondition(1 > model.p, True)
        #self.checkCondition(1 >= model.p, True)
        #self.checkCondition(0 == model.p, None)

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the parameter is initialized
        #
        self.checkCondition(0 < instance.p, True)
        self.checkCondition(2 < instance.p, False)
        self.checkCondition(1 <= instance.p, True)
        self.checkCondition(2 <= instance.p, False)
        self.checkCondition(2 > instance.p, True)
        self.checkCondition(0 > instance.p, False)
        self.checkCondition(1 >= instance.p, True)
        self.checkCondition(0 >= instance.p, False)
        self.checkCondition(1 == instance.p, True)
        self.checkCondition(2 == instance.p, False)

    def test_mutable_paramConditional(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=True)
        #
        try:
            self.checkCondition(model.p > 0, True)
            if not logical_expr._using_chained_inequality:
                self.fail("Expected ValueError because the parameter value is being accessed before it is constructed.")
        except ValueError:
            pass
        #self.checkCondition(model.p >= 0, True)
        #self.checkCondition(model.p < 1, True)
        #self.checkCondition(model.p <= 1, True)
        #self.checkCondition(model.p == 0, None)

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the parameter is initialized
        #
        self.checkCondition(instance.p > 0, True)
        self.checkCondition(instance.p > 2, False)
        self.checkCondition(instance.p >= 1, True)
        self.checkCondition(instance.p >= 2, False)
        self.checkCondition(instance.p < 2, True)
        self.checkCondition(instance.p < 0, False)
        self.checkCondition(instance.p <= 1, True)
        self.checkCondition(instance.p <= 0, False)
        self.checkCondition(instance.p == 1, True)
        self.checkCondition(instance.p == 2, False)

    def test_mutable_paramConditional_reversed(self):
        model = AbstractModel()
        model.p = Param(initialize=1.0, mutable=True)
        #
        try:
            self.checkCondition(0 < model.p, True)
            if not logical_expr._using_chained_inequality:
                self.fail("Expected ValueError because the parameter value is being accessed before it is constructed.")
        except ValueError:
            pass
        #self.checkCondition(0 <= model.p, True)
        #self.checkCondition(1 > model.p, True)
        #self.checkCondition(1 >= model.p, True)
        #self.checkCondition(0 == model.p, None)

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the parameter is initialized
        #
        self.checkCondition(0 < instance.p, True)
        self.checkCondition(2 < instance.p, False)
        self.checkCondition(1 <= instance.p, True)
        self.checkCondition(2 <= instance.p, False)
        self.checkCondition(2 > instance.p, True)
        self.checkCondition(0 > instance.p, False)
        self.checkCondition(1 >= instance.p, True)
        self.checkCondition(0 >= instance.p, False)
        self.checkCondition(1 == instance.p, True)
        self.checkCondition(2 == instance.p, False)

    def test_varConditional(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        #
        try:
            self.checkCondition(model.v > 0, True)
            if not logical_expr._using_chained_inequality:
                self.fail("Expected ValueError because the variable value is being accessed before it is constructed.")
        except:
            pass
        #self.checkCondition(model.v >= 0, True)
        #self.checkCondition(model.v < 1, True)
        #self.checkCondition(model.v <= 1, True)
        #self.checkCondition(model.v == 0, None)

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the variable is initialized
        #
        self.checkCondition(instance.v > 0, True)
        self.checkCondition(instance.v > 2, False)
        self.checkCondition(instance.v >= 1, True)
        self.checkCondition(instance.v >= 2, False)
        self.checkCondition(instance.v < 2, True)
        self.checkCondition(instance.v < 0, False)
        self.checkCondition(instance.v <= 1, True)
        self.checkCondition(instance.v <= 0, False)
        self.checkCondition(instance.v == 1, True)
        self.checkCondition(instance.v == 2, False)

    def test_varConditional_reversed(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        #
        try:
            self.checkCondition(0 < model.v, True)
            if not logical_expr._using_chained_inequality:
                self.fail("Expected ValueError because the variable value is being accessed before it is constructed.")
        except:
            pass
        #self.checkCondition(0 <= model.v, True)
        #self.checkCondition(1 > model.v, True)
        #self.checkCondition(1 >= model.v, True)
        #self.checkCondition(0 == model.v, None)

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the variable is initialized
        #
        self.checkCondition(0 < instance.v, True)
        self.checkCondition(2 < instance.v, False)
        self.checkCondition(1 <= instance.v, True)
        self.checkCondition(2 <= instance.v, False)
        self.checkCondition(2 > instance.v, True)
        self.checkCondition(0 > instance.v, False)
        self.checkCondition(1 >= instance.v, True)
        self.checkCondition(0 >= instance.v, False)
        self.checkCondition(1 == instance.v, True)
        self.checkCondition(2 == instance.v, False)

    def test_eval_sub_varConditional(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        #
        # The value() function generates an exception when the variable is unconstructed!
        #
        try:
            self.checkCondition(value(model.v) > 0, None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(value(model.v) >= 0, None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(value(model.v) < 1, None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(value(model.v) <= 1, None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(value(model.v) == 0, None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the variable is initialized
        #
        self.checkCondition(value(instance.v) > 0, True)
        self.checkCondition(value(instance.v) > 2, False)
        self.checkCondition(value(instance.v) >= 1, True)
        self.checkCondition(value(instance.v) >= 2, False)
        self.checkCondition(value(instance.v) < 2, True)
        self.checkCondition(value(instance.v) < 0, False)
        self.checkCondition(value(instance.v) <= 1, True)
        self.checkCondition(value(instance.v) <= 0, False)
        self.checkCondition(value(instance.v) == 1, True)
        self.checkCondition(value(instance.v) == 2, False)

    def test_eval_sub_varConditional_reversed(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        #
        # The value() function generates an exception when the variable is unconstructed!
        #
        try:
            self.checkCondition(0 < value(model.v), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(0 <= value(model.v), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(1 > value(model.v), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(1 >= value(model.v), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(0 == value(model.v), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the variable is initialized
        #
        self.checkCondition(0 < value(instance.v), True)
        self.checkCondition(2 < value(instance.v), False)
        self.checkCondition(1 <= value(instance.v), True)
        self.checkCondition(2 <= value(instance.v), False)
        self.checkCondition(2 > value(instance.v), True)
        self.checkCondition(0 > value(instance.v), False)
        self.checkCondition(1 >= value(instance.v), True)
        self.checkCondition(0 >= value(instance.v), False)
        self.checkCondition(1 == value(instance.v), True)
        self.checkCondition(2 == value(instance.v), False)

    def test_eval_varConditional(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        #
        # The value() function generates an exception when the variable is unconstructed!
        #
        try:
            self.checkCondition(value(model.v > 0), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(value(model.v >= 0), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(value(model.v == 0), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass

        instance = model.create_instance()
        self.checkCondition(value(instance.v > 0), True)
        self.checkCondition(value(instance.v > 2), False)
        self.checkCondition(value(instance.v >= 1), True)
        self.checkCondition(value(instance.v >= 2), False)
        self.checkCondition(value(instance.v == 1), True)
        self.checkCondition(value(instance.v == 2), False)

    def test_eval_varConditional_reversed(self):
        model = AbstractModel()
        model.v = Var(initialize=1.0)
        #
        # The value() function generates an exception when the variable is unconstructed!
        #
        try:
            self.checkCondition(value(0 < model.v), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(value(0 <= model.v), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass
        try:
            self.checkCondition(value(0 == model.v), None)
            self.fail("Expected ValueError because component was undefined")
        except ValueError:
            pass

        instance = model.create_instance()
        #
        # Inequalities evaluate normally when the variable is initialized
        #
        self.checkCondition(value(0 < instance.v), True)
        self.checkCondition(value(2 < instance.v), False)
        self.checkCondition(value(1 <= instance.v), True)
        self.checkCondition(value(2 <= instance.v), False)
        self.checkCondition(value(1 == instance.v), True)
        self.checkCondition(value(2 == instance.v), False)


class TestPolynomialDegree(unittest.TestCase):

    def setUp(self):
        # This class tests the Pyomo 5.x expression trees
        def d_fn(model):
            return model.c+model.c
        self.model = AbstractModel()
        self.model.a = Var(initialize=1.0)
        self.model.b = Var(initialize=2.0)
        self.model.c = Param(initialize=0, mutable=True)
        self.model.d = Param(initialize=d_fn, mutable=True)
        self.model.e = Param(mutable=True)
        self.instance = self.model.create_instance()

    def tearDown(self):
        self.model = None
        self.instance = None

    def test_param(self):
        #
        # Check that a parameter has degree 0
        #
        self.assertEqual(self.model.d.polynomial_degree(), 0)

    def test_var(self):
        #
        # Check that a non-fixed variable has degree 1
        #
        self.model.a.fixed = False
        self.assertEqual(self.model.a.polynomial_degree(), 1)
        #
        # Check that a fixed variable has degree 0
        #
        self.model.a.fixed = True
        self.assertEqual(self.model.a.polynomial_degree(), 0)

    def test_simple_sum(self):
        #
        # A sum of parameters has degree 0
        #
        expr = self.model.c + self.model.d
        self.assertEqual(expr.polynomial_degree(), 0)
        #
        # A sum of variables has degree 1
        #
        expr = self.model.a + self.model.b
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = False
        #
        # A sum of fixed variables has degree 0
        #
        expr = self.model.a + self.model.c
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)

    def test_linearsum(self):
        m = ConcreteModel()
        A = range(5)
        m.v = Var(A)

        e = quicksum(m.v[i] for i in A)
        self.assertEqual(e.polynomial_degree(), 1) 

        e = quicksum(i*m.v[i] for i in A)
        self.assertEqual(e.polynomial_degree(), 1) 

        e = quicksum(1 for i in A)
        self.assertEqual(polynomial_degree(e), 0) 

        e = quicksum((1 for i in A), linear=True)
        self.assertTrue(e.__class__ in native_numeric_types)

    def test_relational_ops(self):
        #
        # TODO: Should a relational expression have a polynomial degree?
        #
        # A relational expression with parameters has degree 0
        #
        expr = self.model.c < self.model.d
        self.assertEqual(expr.polynomial_degree(), 0)
        #
        # A relational expression with variables has degree 1
        #
        expr = self.model.a <= self.model.d
        self.assertEqual(expr.polynomial_degree(), 1)
        #
        # A relational expression with variable products has degree 2
        #
        expr = self.model.a * self.model.a >= self.model.b
        self.assertEqual(expr.polynomial_degree(), 2)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = False
        #
        expr = self.model.a > self.model.a * self.model.b
        self.assertEqual(expr.polynomial_degree(), 2)
        self.model.b.fixed = True
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.b.fixed = False
        #
        expr = self.model.a == self.model.a * self.model.b
        self.assertEqual(expr.polynomial_degree(), 2)
        self.model.b.fixed = True
        self.assertEqual(expr.polynomial_degree(), 1)

    def test_simple_product(self):
        #
        # A product of parameters has degree 0
        #
        expr = self.model.c * self.model.d
        self.assertEqual(expr.polynomial_degree(), 0)
        #
        # A product of variables has degree 2
        #
        expr = self.model.a * self.model.b
        self.assertEqual(expr.polynomial_degree(), 2)
        #
        # A product of a variable and parameter has degree 1
        #
        expr = self.model.a * self.model.c
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        self.model.a.fixed = False
        #
        # A fraction with a variable and parameter has degree 1
        #
        expr = self.model.a / self.model.c
        self.assertEqual(expr.polynomial_degree(), 1)
        #
        # A fraction with a variable in the denominator has degree None.
        # This indicates that it is not a polyomial.
        #
        expr = self.model.c / self.model.a
        self.assertEqual(expr.polynomial_degree(), None)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        self.model.a.fixed = False

    def test_nested_expr(self):
        #
        # Verify that nested expressions compute polynomial degrees appropriately
        #
        expr1 = self.model.c * self.model.d
        expr2 = expr1 + expr1
        self.assertEqual(expr2.polynomial_degree(), 0)

        expr1 = self.model.a * self.model.b
        expr2 = expr1 + expr1
        self.assertEqual(expr2.polynomial_degree(), 2)
        self.model.a.fixed = True
        self.assertEqual(expr2.polynomial_degree(), 1)
        self.model.a.fixed = False

        expr1 = self.model.c + self.model.d
        expr2 = expr1 * expr1
        self.assertEqual(expr2.polynomial_degree(), 0)

        expr1 = self.model.a + self.model.b
        expr2 = expr1 * expr1
        self.assertEqual(expr2.polynomial_degree(), 2)
        self.model.a.fixed = True
        self.assertEqual(expr2.polynomial_degree(), 2)
        self.model.b.fixed = True
        self.assertEqual(expr2.polynomial_degree(), 0)

    def test_misc_operators(self):
        #
        # Check that polynomial checks work with Negation
        #
        expr = -(self.model.a * self.model.b)
        self.assertEqual(expr.polynomial_degree(), 2)

    def test_nonpolynomial_abs(self):
        #
        # Check that an expression containing abs() is not a polynomial
        #
        expr1 = abs(self.model.a * self.model.b)
        self.assertEqual(expr1.polynomial_degree(), None)

        expr2 = self.model.a + self.model.b * abs(self.model.b)
        self.assertEqual(expr2.polynomial_degree(), None)

        expr3 = self.model.a * ( self.model.b + abs(self.model.b) )
        self.assertEqual(expr3.polynomial_degree(), None)
        #
        # Fixing variables should turn intrinsic functions into constants
        #
        # Fixing 'a' still leaves a non-constant expression
        #
        self.model.a.fixed = True
        self.assertEqual(expr1.polynomial_degree(), None)
        self.assertEqual(expr2.polynomial_degree(), None)
        self.assertEqual(expr3.polynomial_degree(), None)
        #
        # Fixing 'a' and 'b' creates a constant expression
        #
        self.model.b.fixed = True
        self.assertEqual(expr1.polynomial_degree(), 0)
        self.assertEqual(expr2.polynomial_degree(), 0)
        self.assertEqual(expr3.polynomial_degree(), 0)
        #
        # Fixing 'b' still leaves a non-constant expression for expr1
        #
        self.model.a.fixed = False
        self.assertEqual(expr1.polynomial_degree(), None)
        self.assertEqual(expr2.polynomial_degree(), 1)
        self.assertEqual(expr3.polynomial_degree(), 1)

    def test_nonpolynomial_pow(self):
        m = self.instance
        #
        # A power with a variable exponent is not a polynomial
        #
        expr = pow(m.a, m.b)
        self.assertEqual(expr.polynomial_degree(), None)
        #
        # A power with a constant exponent is not a polynomial
        #
        m.b.fixed = True
        self.assertEqual(expr.polynomial_degree(), 2)
        m.b.value = 0
        self.assertEqual(expr.polynomial_degree(), 0)
        #
        # A power with a constant base and exponent is a constant
        #
        m.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        #
        # A power with a constant base and variable exponent is not a polynomial
        #
        m.b.fixed = False
        self.assertEqual(expr.polynomial_degree(), None)
        #
        # Confirm that pow() expresses the correct degree
        #
        m.a.fixed = False

        expr = pow(m.a, 1)
        self.assertEqual(expr.polynomial_degree(), 1)

        expr = pow(m.a, 2)
        self.assertEqual(expr.polynomial_degree(), 2)

        expr = pow(m.a*m.a, 2)
        self.assertEqual(expr.polynomial_degree(), 4)
        #
        # A non-integer exponent is not a polynomial
        #
        expr = pow(m.a*m.a, 2.1)
        self.assertEqual(expr.polynomial_degree(), None)
        #
        # A negative exponent is not a polynomial
        #
        expr = pow(m.a*m.a, -1)
        self.assertEqual(expr.polynomial_degree(), None)
        #
        # A nonpolynomial base is not a polynomial if the exponent is nonzero
        #
        expr = pow(2**m.a, 1)
        self.assertEqual(expr.polynomial_degree(), None)

        expr = pow(2**m.a, 0)
        self.assertEqual(expr, 1)
        self.assertEqual(as_numeric(expr).polynomial_degree(), 0)
        #
        # With an undefined exponent, the polynomial degree is None
        #
        expr = pow(m.a, m.e)
        self.assertEqual(expr.polynomial_degree(), None)

    def test_Expr_if(self):
        m = self.instance
        #
        # When IF conditional is constant, then polynomial degree is propigated
        #
        expr = Expr_if(1,m.a**3,m.a**2)
        self.assertEqual(expr.polynomial_degree(), 3)
        m.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        m.a.fixed = False

        expr = Expr_if(0,m.a**3,m.a**2)
        self.assertEqual(expr.polynomial_degree(), 2)
        m.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        m.a.fixed = False
        #
        # When IF conditional is variable, then polynomial degree is propagated
        #
        expr = Expr_if(m.a,m.b,m.b**2)
        self.assertEqual(expr.polynomial_degree(), None)
        m.a.fixed = True
        m.a.value = 1
        self.assertEqual(expr.polynomial_degree(), 1)
        #
        # When IF conditional is uninitialized
        #
        # A constant expression has degree 0
        #
        expr = Expr_if(m.e,1,0)
        self.assertEqual(expr.polynomial_degree(), 0)
        #
        # A nonconstant expression has degree if both arguments have the
        # same degree, as long as the IF is fixed (even if it is not
        # defined)
        #
        expr = Expr_if(m.e,m.a,0)
        self.assertEqual(expr.polynomial_degree(), 0)
        expr = Expr_if(m.e,5*m.b,1+m.b)
        self.assertEqual(expr.polynomial_degree(), 1)
        #
        # A nonconstant expression has degree None because
        # m.e is an uninitialized parameter
        #
        expr = Expr_if(m.e,m.b,0)
        self.assertEqual(expr.polynomial_degree(), None)


#
# TODO: Confirm that this checks for entangled expressions.
#
class EntangledExpressionErrors(unittest.TestCase):

    def test_sumexpr_add_entangled(self):
        x = Var()
        e = x*2 + 1
        e + 1

    def test_entangled_test1(self):
        self.m = ConcreteModel()
        self.m.a = Var()
        self.m.b = Var()
        self.m.c = Var()
        self.m.d = Var()

        e1 = self.m.a + self.m.b

        #print(e1)
        #print(e1_)
        #print("--")
        e2 = self.m.c + e1

        #print(e1)
        #print(e1_)
        #print(e2)
        #print(e2_)
        #print("--")
        e3 = self.m.d + e1

        self.assertEqual( e1.nargs(), 2)
        self.assertEqual( e2.nargs(), 3)
        self.assertEqual( e3.nargs(), 2)

        self.assertNotEqual( id(e2.arg(2)), id(e3.arg(1).arg(1)))


class TestSummationExpression(unittest.TestCase):

    def setUp(self):
        # This class tests the Pyomo 5.x expression trees

        self.m = ConcreteModel()
        self.m.I = RangeSet(5)
        self.m.a = Var(self.m.I, initialize=5)
        self.m.b = Var(self.m.I, initialize=10)
        self.m.p = Param(self.m.I, initialize=1, mutable=True)
        self.m.q = Param(self.m.I, initialize=3, mutable=False)

    def tearDown(self):
        self.m = None

    def test_summation1(self):
        e = sum_product(self.m.a)
        self.assertEqual( e(), 25 )
        self.assertIs(type(e), LinearExpression)
        self.assertEqual( id(self.m.a[1]), id(e.linear_vars[0]) )
        self.assertEqual( id(self.m.a[2]), id(e.linear_vars[1]) )
        self.assertEqual(e.size(), 1)

    def test_summation2(self):
        e = sum_product(self.m.p, self.m.a)
        self.assertEqual( e(), 25 )
        self.assertIs(type(e), LinearExpression)
        self.assertEqual( id(self.m.a[1]), id(e.linear_vars[0]) )
        self.assertEqual( id(self.m.a[2]), id(e.linear_vars[1]) )
        self.assertEqual(e.size(), 1)

    def test_summation3(self):
        e = sum_product(self.m.q, self.m.a)
        self.assertEqual( e(), 75 )
        self.assertIs(type(e), LinearExpression)
        self.assertEqual( id(self.m.a[1]), id(e.linear_vars[0]) )
        self.assertEqual( id(self.m.a[2]), id(e.linear_vars[1]) )
        self.assertEqual(e.size(), 1)

    def test_summation4(self):
        e = sum_product(self.m.a, self.m.b)
        self.assertEqual( e(), 250 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual( id(self.m.a[1]), id(e.arg(0).arg(0)) )
        self.assertEqual( id(self.m.a[2]), id(e.arg(1).arg(0)) )
        self.assertEqual(e.size(), 16)

    def test_summation5(self):
        e = sum_product(self.m.b, denom=self.m.a)
        self.assertEqual( e(), 10 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.size(), 16)

    def test_summation6(self):
        e = sum_product(self.m.a, denom=self.m.p)
        self.assertEqual( e(), 25 )
        self.assertIs(type(e), LinearExpression)
        #self.assertEqual( id(self.m.a[1]), id(e.arg(0).arg(0)) )
        #self.assertEqual( id(self.m.a[2]), id(e.arg(1).arg(0)) )
        #self.assertEqual(e.size(), 21)

    def test_summation7(self):
        e = sum_product(self.m.p, self.m.q, index=self.m.I)
        self.assertEqual( e(), 15 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual( e.nargs(), 5)
        self.assertEqual(e.size(), 16)

    def test_summation_compression(self):
        e1 = sum_product(self.m.a)
        e2 = sum_product(self.m.b)
        e = e1+e2
        self.assertEqual( e(), 75 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual( e.nargs(), 2)
        self.assertEqual(e.size(), 3)


class TestSumExpression(unittest.TestCase):

    def setUp(self):
        # This class tests the Pyomo 5.x expression trees

        self.m = ConcreteModel()
        self.m.I = RangeSet(5)
        self.m.a = Var(self.m.I, initialize=5)
        self.m.b = Var(self.m.I, initialize=10)
        self.m.p = Param(self.m.I, initialize=1, mutable=True)
        self.m.q = Param(self.m.I, initialize=3, mutable=False)

    def tearDown(self):
        self.m = None

    def test_summation1(self):
        e = quicksum((self.m.a[i] for i in self.m.a), linear=False)
        self.assertEqual( e(), 25 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual( id(self.m.a[1]), id(e.arg(0)) )
        self.assertEqual( id(self.m.a[2]), id(e.arg(1)) )
        self.assertEqual(e.size(), 6)
        #
        e = quicksum(self.m.a[i] for i in self.m.a)
        self.assertEqual( e(), 25 )
        self.assertIs(type(e), LinearExpression)

    def test_summation2(self):
        e = quicksum((self.m.p[i]*self.m.a[i] for i in self.m.a), linear=False)
        self.assertEqual( e(), 25 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual( id(self.m.a[1]), id(e.arg(0).arg(1)) )
        self.assertEqual( id(self.m.a[2]), id(e.arg(1).arg(1)) )
        self.assertEqual(e.size(), 16)
        #
        e = quicksum(self.m.p[i]*self.m.a[i] for i in self.m.a)
        self.assertEqual( e(), 25 )
        self.assertIs(type(e), LinearExpression)

    def test_summation3(self):
        e = quicksum((self.m.q[i]*self.m.a[i] for i in self.m.a), linear=False)
        self.assertEqual( e(), 75 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual( id(self.m.a[1]), id(e.arg(0).arg(1)) )
        self.assertEqual( id(self.m.a[2]), id(e.arg(1).arg(1)) )
        self.assertEqual(e.size(), 16)
        #
        e = quicksum(self.m.q[i]*self.m.a[i] for i in self.m.a)
        self.assertEqual( e(), 75 )
        self.assertIs(type(e), LinearExpression)

    def test_summation4(self):
        e = quicksum(self.m.a[i]*self.m.b[i] for i in self.m.a)
        self.assertEqual( e(), 250 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual( id(self.m.a[1]), id(e.arg(0).arg(0)) )
        self.assertEqual( id(self.m.a[2]), id(e.arg(1).arg(0)) )
        self.assertEqual(e.size(), 16)

    def test_summation5(self):
        e = quicksum(self.m.b[i]/self.m.a[i] for i in self.m.a)
        self.assertEqual( e(), 10 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.size(), 16)

    def test_summation6(self):
        e = quicksum((self.m.a[i]/self.m.p[i] for i in self.m.a), linear=False)
        self.assertEqual( e(), 25 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual( id(self.m.a[1]), id(e.arg(0).arg(1)) )
        self.assertEqual( id(self.m.a[2]), id(e.arg(1).arg(1)) )
        self.assertEqual(e.size(), 26)
        #
        e = quicksum(self.m.a[i]/self.m.p[i] for i in self.m.a)
        self.assertEqual( e(), 25 )
        self.assertIs(type(e), LinearExpression)

    def test_summation7(self):
        e = quicksum((self.m.p[i]*self.m.q[i] for i in self.m.I), linear=False)
        self.assertEqual( e(), 15 )
        self.assertIs(type(e), SumExpression)
        self.assertEqual( e.nargs(), 5)
        self.assertEqual(e.size(), 16)
        #
        e = quicksum(self.m.p[i]*self.m.q[i] for i in self.m.I)
        self.assertEqual( e(), 15 )
        self.assertIs(type(e), SumExpression)


class TestCloneExpression(unittest.TestCase):

    def setUp(self):
        # This class tests the Pyomo 5.x expression trees

        self.m = ConcreteModel()
        self.m.a = Var(initialize=5)
        self.m.b = Var(initialize=10)
        self.m.p = Param(initialize=1, mutable=True)

    def tearDown(self):
        self.m = None

    def test_numeric(self):
        with clone_counter() as counter:
            start = counter.count
            e_ = 1
            e = clone_expression(e_)
            self.assertEqual(id(e), id(e_))
            e = clone_expression(self.m.p)
            self.assertEqual(id(e), id(self.m.p))
            #
            total = counter.count - start
            self.assertEqual(total, 2)
        
    def test_Expression(self):
        #
        # Identify variables when there are duplicates
        #
        m = ConcreteModel()
        m.a = Var(initialize=1)
        m.b = Var(initialize=2)
        m.e = Expression(expr=3*m.a)
        m.E = Expression([0,1], initialize={0:3*m.a, 1:4*m.b})

        with clone_counter() as counter:
            start = counter.count
            expr1 = m.e + m.E[1] 
            expr2 = expr1.clone()
            self.assertEqual( expr1(), 11 )
            self.assertEqual( expr2(), 11 )
            self.assertNotEqual( id(expr1),       id(expr2) )
            self.assertNotEqual( id(expr1._args_), id(expr2._args_) )
            self.assertEqual( id(expr1.arg(0)), id(expr2.arg(0)) )
            self.assertEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            #
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_ExpressionX(self):
        #
        # Identify variables when there are duplicates
        #
        m = ConcreteModel()
        m.a = Var(initialize=1)
        m.b = Var(initialize=2)
        m.e = Expression(expr=3*m.a)
        m.E = Expression([0,1], initialize={0:3*m.a, 1:4*m.b})

        with clone_counter() as counter:
            start = counter.count
            expr1 = m.e + m.E[1] 
            expr2 = copy.deepcopy(expr1)
            self.assertEqual( expr1(), 11 )
            self.assertEqual( expr2(), 11 )
            self.assertNotEqual( id(expr1),       id(expr2) )
            self.assertNotEqual( id(expr1._args_), id(expr2._args_) )
            self.assertNotEqual( id(expr1.arg(0)), id(expr2.arg(0)) )
            self.assertNotEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            #
            total = counter.count - start
            self.assertEqual(total, 0)

    def test_SumExpression(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = self.m.a + self.m.b
            expr2 = expr1.clone()
            self.assertEqual( expr1(), 15 )
            self.assertEqual( expr2(), 15 )
            self.assertNotEqual( id(expr1),       id(expr2) )
            self.assertNotEqual( id(expr1._args_), id(expr2._args_) )
            self.assertEqual( id(expr1.arg(0)), id(expr2.arg(0)) )
            self.assertEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            expr1 += self.m.b
            self.assertEqual( expr1(), 25 )
            self.assertEqual( expr2(), 15 )
            self.assertNotEqual( id(expr1),       id(expr2) )
            self.assertNotEqual( id(expr1._args_), id(expr2._args_) )
            self.assertEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            self.assertEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            #
            total = counter.count - start
            self.assertEqual(total, 1)
            
    def test_SumExpressionX(self):
        with clone_counter() as counter:
            start = counter.count
            expr1 = self.m.a + self.m.b
            expr2 = copy.deepcopy(expr1)
            self.assertEqual( expr1(), 15 )
            self.assertEqual( expr2(), 15 )
            self.assertNotEqual( id(expr1),       id(expr2) )
            self.assertNotEqual( id(expr1._args_), id(expr2._args_) )
            self.assertNotEqual( id(expr1.arg(0)), id(expr2.arg(0)) )
            self.assertNotEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            expr1 += self.m.b
            self.assertEqual( expr1(), 25 )
            self.assertEqual( expr2(), 15 )
            self.assertNotEqual( id(expr1),       id(expr2) )
            self.assertNotEqual( id(expr1._args_), id(expr2._args_) )
            self.assertNotEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            self.assertNotEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            #
            total = counter.count - start
            self.assertEqual(total, 0)
            
    def test_SumExpressionY(self):
        self.m = ConcreteModel()
        A = range(5)
        self.m.a = Var(A, initialize=5)
        self.m.b = Var(initialize=10)

        with clone_counter() as counter:
            start = counter.count
            expr1 = quicksum(self.m.a[i] for i in self.m.a)
            expr2 = copy.deepcopy(expr1)
            self.assertEqual( expr1(), 25 )
            self.assertEqual( expr2(), 25 )
            self.assertNotEqual( id(expr1),        id(expr2) )
            self.assertEqual( id(expr1._args_), id(expr2._args_) )
            self.assertNotEqual( id(expr1.linear_vars[0]), id(expr2.linear_vars[0]) )
            self.assertNotEqual( id(expr1.linear_vars[1]), id(expr2.linear_vars[1]) )
            expr1 += self.m.b
            self.assertEqual( expr1(), 35 )
            self.assertEqual( expr2(), 25 )
            self.assertNotEqual( id(expr1),        id(expr2) )
            self.assertNotEqual( id(expr1._args_), id(expr2._args_) )
            #
            total = counter.count - start
            self.assertEqual(total, 0)
            
    def test_ProductExpression_mult(self):
        with clone_counter() as counter:
            start = counter.count
            #
            expr1 = self.m.a * self.m.b
            expr2 = expr1.clone()
            self.assertEqual( expr1(), 50 )
            self.assertEqual( expr2(), 50 )
            self.assertNotEqual( id(expr1),      id(expr2) )
            self.assertEqual( id(expr1._args_), id(expr2._args_) )
            self.assertEqual( id(expr1.arg(0)), id(expr2.arg(0)) )
            self.assertEqual( id(expr1.arg(1)), id(expr2.arg(1)) )

            expr1 *= self.m.b
            self.assertEqual( expr1(), 500 )
            self.assertEqual( expr2(), 50 )
            self.assertNotEqual( id(expr1),                 id(expr2) )
            self.assertNotEqual( id(expr1._args_),           id(expr2._args_) )
            self.assertEqual( id(expr1.arg(0)._args_), id(expr2._args_) )
            self.assertEqual( id(expr1.arg(1)),           id(expr2.arg(1)) )
            self.assertEqual( id(expr1.arg(0).arg(0)),  id(expr2.arg(0)) )
            self.assertEqual( id(expr1.arg(0).arg(1)),  id(expr2.arg(1)) )

            expr1 = self.m.a * (self.m.b + self.m.a)
            expr2 = expr1.clone()
            self.assertEqual( expr1(), 75 )
            self.assertEqual( expr2(), 75 )
            # Note that since one of the args is a sum expression, the _args_
            # in the sum is a *list*, which will be duplicated by deepcopy.
            # This will cause the two args in the Product to be different.
            self.assertNotEqual( id(expr1),      id(expr2) )
            self.assertNotEqual( id(expr1._args_), id(expr2._args_) )
            self.assertEqual( id(expr1.arg(0)), id(expr2.arg(0)) )
            self.assertNotEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            #
            total = counter.count - start
            self.assertEqual(total, 2)

    def test_ProductExpression_div(self):
        with clone_counter() as counter:
            start = counter.count
            #
            expr1 = self.m.a / self.m.b
            expr2 = expr1.clone()
            self.assertEqual( expr1(), 0.5 )
            self.assertEqual( expr2(), 0.5 )
            self.assertNotEqual( id(expr1),       id(expr2) )
            # Note: _args_ are the same because tuples are not copied
            self.assertEqual( id(expr1._args_),    id(expr2._args_) )
            self.assertEqual( id(expr1.arg(0)), id(expr2.arg(0)) )
            self.assertEqual( id(expr1.arg(1)), id(expr2.arg(1)) )

            expr1 /= self.m.b
            self.assertEqual( expr1(), 0.05 )
            self.assertEqual( expr2(), 0.5 )
            self.assertNotEqual( id(expr1),                 id(expr2) )
            self.assertNotEqual( id(expr1._args_),           id(expr2._args_) )
            self.assertEqual( id(expr1.arg(0)._args_),  id(expr2._args_) )
            self.assertEqual( id(expr1.arg(0).arg(0)),  id(expr2.arg(0)) )
            self.assertEqual( id(expr1.arg(0).arg(1)),  id(expr2.arg(1)) )

            expr1 = self.m.a / (self.m.b + self.m.a)
            expr2 = expr1.clone()
            self.assertEqual( expr1(), 1/3. )
            self.assertEqual( expr2(), 1/3. )
            # Note that since one of the args is a sum expression, the _args_
            # in the sum is a *list*, which will be duplicated by deepcopy.
            # This will cause the two args in the Product to be different.
            self.assertNotEqual( id(expr1),      id(expr2) )
            self.assertNotEqual( id(expr1._args_), id(expr2._args_) )
            self.assertEqual( id(expr1.arg(0)), id(expr2.arg(0)) )
            self.assertNotEqual( id(expr1.arg(1)), id(expr2.arg(1)) )
            #
            total = counter.count - start
            self.assertEqual(total, 2)

    def test_sumOfExpressions(self):
        with clone_counter() as counter:
            start = counter.count
            #
            expr1 = self.m.a * self.m.b + self.m.a * self.m.a
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 75)
            self.assertEqual(expr2(), 75)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(expr1.arg(0)(), expr2.arg(0)())
            self.assertEqual(expr1.arg(1)(), expr2.arg(1)())
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            expr1 += self.m.b
            self.assertEqual(expr1(), 85)
            self.assertEqual(expr2(), 75)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertEqual(expr1.nargs(), 3)
            self.assertEqual(expr2.nargs(), 2)
            self.assertEqual(expr1.arg(0)(), 50)
            self.assertEqual(expr1.arg(1)(), 25)
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            #
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_productOfExpressions(self):
        with clone_counter() as counter:
            start = counter.count
            #
            expr1 = (self.m.a + self.m.b) * (self.m.a + self.m.a)
            expr2 = expr1.clone()
            self.assertEqual(expr1(), 150)
            self.assertEqual(expr2(), 150)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            self.assertEqual(expr1.arg(0)(), expr2.arg(0)())
            self.assertEqual(expr1.arg(1)(), expr2.arg(1)())

            self.assertEqual(expr1.arg(0).nargs(), 2)
            self.assertEqual(expr2.arg(0).nargs(), 2)
            self.assertEqual(expr1.arg(1).nargs(), 2)
            self.assertEqual(expr2.arg(1).nargs(), 2)

            self.assertIs( expr1.arg(0).arg(0),
                           expr2.arg(0).arg(0) )
            self.assertIs( expr1.arg(0).arg(1),
                           expr2.arg(0).arg(1) )
            self.assertIs( expr1.arg(1).arg(0),
                           expr2.arg(1).arg(0) )

            expr1 *= self.m.b
            self.assertEqual(expr1(), 1500)
            self.assertEqual(expr2(), 150)
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))

            self.assertIs(type(expr1.arg(0)), type(expr2))
            self.assertEqual(expr1.arg(0)(), expr2())

            self.assertEqual(expr1.nargs(), 2)
            self.assertEqual(expr2.nargs(), 2)
            #
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_productOfExpressions_div(self):
        with clone_counter() as counter:
            start = counter.count
            #
            expr1 = (self.m.a + self.m.b) / (self.m.a + self.m.a)
            expr2 = expr1.clone()

            self.assertNotEqual(id(expr1), id(expr2))
            self.assertNotEqual(id(expr1._args_), id(expr2._args_))
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))
            self.assertEqual(expr1.arg(0)(), expr2.arg(0)())
            self.assertEqual(expr1.arg(1)(), expr2.arg(1)())

            self.assertEqual(expr1.nargs(), 2)
            self.assertEqual(expr2.nargs(), 2)
            self.assertEqual(expr1.arg(0).nargs(), 2)
            self.assertEqual(expr2.arg(0).nargs(), 2)
            self.assertEqual(expr1.arg(1).nargs(), 2)
            self.assertEqual(expr2.arg(1).nargs(), 2)

            self.assertIs( expr1.arg(0).arg(0), expr2.arg(0).arg(0) )
            self.assertIs( expr1.arg(0).arg(1), expr2.arg(0).arg(1) )

            expr1 /= self.m.b
            self.assertAlmostEqual(expr1(), .15)
            self.assertAlmostEqual(expr2(), 1.5)
            self.assertNotEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            self.assertNotEqual(id(expr1.arg(1)), id(expr2.arg(1)))

            self.assertIs(type(expr1.arg(0)), type(expr2))
            self.assertAlmostEqual(expr1.arg(0)(), expr2())

            self.assertEqual(expr1.nargs(), 2)
            self.assertEqual(expr2.nargs(), 2)
            #
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_Expr_if(self):
        with clone_counter() as counter:
            start = counter.count
            #
            expr1 = Expr_if(IF=self.m.a + self.m.b < 20, THEN=self.m.a, ELSE=self.m.b)
            expr2 = expr1.clone()
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertEqual(expr1(), value(self.m.a))
            self.assertEqual(expr2(), value(self.m.a))
            self.assertNotEqual(id(expr1._if), id(expr2._if))
            self.assertEqual(id(expr1._then), id(expr2._then))
            self.assertEqual(id(expr1._else), id(expr2._else))
            self.assertEqual(expr1._if(), expr2._if())
            self.assertEqual(expr1._then(), expr2._then())
            self.assertEqual(expr1._else(), expr2._else())
            #
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_getitem(self):
        # Testing cloning of the abs() function
        with clone_counter() as counter:
            start = counter.count
            #
            m = ConcreteModel()
            m.I = RangeSet(1,9)
            m.x = Var(m.I, initialize=lambda m,i: i+1)
            m.P = Param(m.I, initialize=lambda m,i: 10-i, mutable=True)
            t = IndexTemplate(m.I)

            e = m.x[t+m.P[t+1]] + 3
            e_ = e.clone()
            self.assertEqual("x[{I} + P[{I} + 1]] + 3", str(e_))
            #
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_other(self):
        # Testing cloning of the abs() function
        with clone_counter() as counter:
            start = counter.count
            #
            model = ConcreteModel()
            model.a = Var()
            model.x = ExternalFunction(library='foo.so', function='bar')
            e = model.x(2*model.a, 1, "foo", [])
            e_ = e.clone()
            self.assertEqual(type(e_), type(e))
            self.assertEqual(type(e_.arg(0)), type(e.arg(0)))
            self.assertEqual(type(e_.arg(1)), type(e.arg(1)))
            self.assertEqual(type(e_.arg(2)), type(e.arg(2)))
            self.assertEqual(type(e_.arg(3)), type(e.arg(3)))
            #
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_abs(self):
        # Testing cloning of the abs() function
        with clone_counter() as counter:
            start = counter.count
            #
            expr1 = abs(self.m.a)
            expr2 = expr1.clone()
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertEqual(expr1(), value(self.m.a))
            self.assertEqual(expr2(), value(self.m.a))
            self.assertEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            #
            total = counter.count - start
            self.assertEqual(total, 1)

    def test_sin(self):
        # Testing cloning of intrinsic functions
        with clone_counter() as counter:
            start = counter.count
            #
            expr1 = sin(self.m.a)
            expr2 = expr1.clone()
            self.assertNotEqual(id(expr1), id(expr2))
            self.assertEqual(expr1(), math.sin(value(self.m.a)))
            self.assertEqual(expr2(), math.sin(value(self.m.a)))
            self.assertEqual(id(expr1.arg(0)), id(expr2.arg(0)))
            #
            total = counter.count - start
            self.assertEqual(total, 1)


#
# Fixed               - Expr has a fixed value
# Constant            - Expr only contains constants and immutable parameters
# PotentiallyVariable - Expr contains one or more variables
#
class TestIsFixedIsConstant(unittest.TestCase):

    def setUp(self):
        # This class tests the Pyomo 5.x expression trees

        def d_fn(model):
            return model.c+model.c
        self.model = AbstractModel()
        self.model.a = Var(initialize=1.0)
        self.model.b = Var(initialize=2.0)
        self.model.c = Param(initialize=1, mutable=True)
        self.model.d = Param(initialize=d_fn, mutable=True)
        self.model.e = Param(initialize=d_fn, mutable=False)
        self.model.f = Param(initialize=0, mutable=True)
        self.model.g = Var(initialize=0)
        self.instance = self.model.create_instance()

    def tearDown(self):
        self.model = None
        self.instance = None

    def test_simple_sum(self):
        #
        # Sum of mutable parameters:  fixed, not constant, not pvar
        #
        expr = self.instance.c + self.instance.d
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        #
        # Sum of mutable and immutable parameters:  fixed, not constant, not pvar
        #
        expr = self.instance.e + self.instance.d
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        #
        # Sum of immutable parameters:  fixed, constant, not pvar
        #
        expr = self.instance.e + self.instance.e
        self.assertEqual(is_fixed(expr), True)
        self.assertEqual(is_constant(expr), True)
        self.assertEqual(is_potentially_variable(expr), False)
        #
        # Sum of unfixed variables:  not fixed, not constant, pvar
        #
        expr = self.instance.a + self.instance.b
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Sum of fixed and unfixed variables:  not fixed, not constant, pvar
        #
        self.instance.a.fixed = True
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Sum of fixed variables:  fixed, not constant, pvar
        #
        self.instance.b.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

    def test_linear_sum(self):
        m = ConcreteModel()
        A = range(5)
        m.v = Var(A)

        e = quicksum(m.v[i] for i in A)
        self.assertEqual(e.is_fixed(), False)
        for i in A:
            m.v[i].fixed = True
        self.assertEqual(e.is_fixed(), True)

        with linear_expression() as e:
            e += 1
        self.assertEqual(len(e.linear_vars), 0)
        self.assertEqual(e.is_fixed(), True)

    def test_simple_product(self):
        #
        # Product of mutable parameters:  fixed, not constant, not pvar
        #
        expr = self.instance.c * self.instance.d
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        #
        # Product of unfixed variable and mutable parameter:  not fixed, not constant, pvar
        #
        expr = self.instance.a * self.instance.c
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Product of unfixed variable and zero parameter: fixed, not constant, pvar
        #
        expr = self.instance.f * self.instance.b
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Product of unfixed variables:  not fixed, not constant, pvar
        #
        expr = self.instance.a * self.instance.b
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Product of fixed and unfixed variables:  not fixed, not constant, pvar
        #
        self.instance.a.fixed = True
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Product of fixed variables:  fixed, not constant, pvar
        #
        self.instance.b.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.instance.a.fixed = False
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Product of unfixed variable and fixed zero variable: fixed, not constant, pvar
        #
        expr = self.instance.a * self.instance.g
        self.instance.a.fixed = False
        self.instance.g.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        #
        # Fraction of unfixed variable and mutable parameter:  not fixed, not constant, pvar
        #
        expr = self.instance.a / self.instance.c
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Fraction of fixed variable and mutable parameter:  fixed, not constant, pvar
        #
        self.instance.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        self.instance.a.fixed = False
        #
        # Fraction of unfixed variable and mutable parameter:  not fixed, not constant, pvar
        #
        expr = self.instance.c / self.instance.a
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Fraction of fixed variable and mutable parameter:  fixed, not constant, pvar
        #
        self.instance.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

    def test_misc_operators(self):
        #
        # Negation of product of unfixed variables:  not fixed, not constant, pvar
        #
        expr = -(self.instance.a * self.instance.b)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

    def test_polynomial_external_func(self):
        model = ConcreteModel()
        model.a = Var()
        model.p = Param(initialize=1, mutable=True)
        model.x = ExternalFunction(library='foo.so', function='bar')

        expr = model.x(2*model.a, 1, "foo", [])
        self.assertEqual(expr.polynomial_degree(), None)

        expr = model.x(2*model.p, 1, "foo", [])
        self.assertEqual(expr.polynomial_degree(), 0)

    def test_getitem(self):
        m = ConcreteModel()
        m.I = RangeSet(1,9)
        m.x = Var(m.I, initialize=lambda m,i: i+1)
        m.P = Param(m.I, initialize=lambda m,i: 10-i, mutable=True)
        t = IndexTemplate(m.I)

        e = m.x[t]
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(e.is_fixed(), False)

        e = m.x[t+m.P[t+1]] + 3
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(e.is_fixed(), False)

        for i in m.I:
            m.x[i].fixed = True
        e = m.x[t]
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(e.is_fixed(), True)

        e = m.x[t+m.P[t+1]] + 3
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(e.is_fixed(), True)


        e = m.P[t+1] + 3
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(e.is_fixed(), True)

    def test_nonpolynomial_abs(self):
        #
        # abs() is fixed or constant depending on its arguments
        #
        expr1 = abs(self.instance.a * self.instance.b)
        self.assertEqual(expr1.is_fixed(), False)
        self.assertEqual(expr1.is_constant(), False)
        self.assertEqual(expr1.is_potentially_variable(), True)

        expr2 = self.instance.a + self.instance.b * abs(self.instance.b)
        self.assertEqual(expr2.is_fixed(), False)
        self.assertEqual(expr2.is_constant(), False)
        self.assertEqual(expr2.is_potentially_variable(), True)

        expr3 = self.instance.a * ( self.instance.b + abs(self.instance.b) )
        self.assertEqual(expr3.is_fixed(), False)
        self.assertEqual(expr3.is_constant(), False)
        self.assertEqual(expr3.is_potentially_variable(), True)

        #
        # Fixing variables should turn intrinsic functions into constants
        #
        self.instance.a.fixed = True
        self.assertEqual(expr1.is_fixed(), False)
        self.assertEqual(expr1.is_constant(), False)
        self.assertEqual(expr1.is_potentially_variable(), True)
        self.assertEqual(expr2.is_fixed(), False)
        self.assertEqual(expr2.is_constant(), False)
        self.assertEqual(expr2.is_potentially_variable(), True)
        self.assertEqual(expr3.is_fixed(), False)
        self.assertEqual(expr3.is_constant(), False)
        self.assertEqual(expr3.is_potentially_variable(), True)

        self.instance.b.fixed = True
        self.assertEqual(expr1.is_fixed(), True)
        self.assertEqual(expr1.is_constant(), False)
        self.assertEqual(expr1.is_potentially_variable(), True)
        self.assertEqual(expr2.is_fixed(), True)
        self.assertEqual(expr2.is_constant(), False)
        self.assertEqual(expr2.is_potentially_variable(), True)
        self.assertEqual(expr3.is_fixed(), True)
        self.assertEqual(expr3.is_constant(), False)
        self.assertEqual(expr3.is_potentially_variable(), True)

        self.instance.a.fixed = False
        self.assertEqual(expr1.is_fixed(), False)
        self.assertEqual(expr1.is_constant(), False)
        self.assertEqual(expr1.is_potentially_variable(), True)
        self.assertEqual(expr2.is_fixed(), False)
        self.assertEqual(expr2.is_constant(), False)
        self.assertEqual(expr2.is_potentially_variable(), True)
        self.assertEqual(expr3.is_fixed(), False)
        self.assertEqual(expr3.is_constant(), False)
        self.assertEqual(expr3.is_potentially_variable(), True)

    def test_nonpolynomial_pow(self):
        m = self.instance
        #
        # A power with a mutable base and exponent: fixed, not constant, not pvar
        #
        expr = pow(m.d, m.e)
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        #
        # A power with a variable exponent: not fixed, not constant, pvar
        #
        expr = pow(m.a, m.b)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # A power with a constant exponent: not fixed, not constant, pvar
        #
        m.b.fixed = True
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        m.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        m.b.fixed = False
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        m.a.fixed = False

        expr = pow(m.a, 1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        expr = pow(m.a, 2)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        expr = pow(m.a*m.a, 2)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        expr = pow(m.a*m.a, 2.1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        expr = pow(m.a*m.a, -1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        expr = pow(2**m.a, 1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        expr = pow(2**m.a, 0)
        self.assertEqual(is_fixed(expr), True)
        self.assertEqual(is_constant(expr), True)
        self.assertEqual(expr, 1)
        self.assertEqual(as_numeric(expr).polynomial_degree(), 0)

    def test_Expr_if(self):
        m = self.instance

        expr = Expr_if(1,m.a,m.e)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = False

        expr = Expr_if(0,m.a,m.e)
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), True)
        # BUG
        #self.assertEqual(expr.is_potentially_variable(), False)
        m.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), True)
        # BUG
        #self.assertEqual(expr.is_potentially_variable(), False)
        m.a.fixed = False

        expr = Expr_if(m.a,m.b,m.b)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = True
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = False

    def test_LinearExpr(self):
        m = self.instance

        expr = m.a + m.b
        self.assertIs(type(expr), SumExpression)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        m.a.fix(1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        m.b.fix(1)
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        m.a.unfix()
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        m.a.unfix()

        expr -= m.a
        self.assertEqual(expr.is_fixed(), False)   # With a simple tree, the terms do not cancel
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

        expr -= m.b
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

    def test_expression(self):
        m = ConcreteModel()
        m.x = Expression()
        e = m.x
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)

        e = m.x + 1
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)

        e = m.x**2
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)

        e = m.x**2/(m.x + 1)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)

    def test_external_func(self):
        m = ConcreteModel()
        m.a = Var(initialize=1)
        m.p = Param(initialize=1, mutable=True)
        m.x = ExternalFunction(library='foo.so', function='bar')

        e = m.x(m.a, 1, "foo bar", [])
        self.assertEqual(e.is_potentially_variable(), True)
        e = m.x(m.p, 1, "foo bar", [])
        self.assertEqual(e.is_potentially_variable(), False)


# NOTE: These are fairly weak coverage tests.  
# It's probably worth confirming the final linear expression that is generated.
class TestLinearExpression(unittest.TestCase):

    def test_sum_other(self):
        m = ConcreteModel()
        m.v = Var(range(5))
        m.p = Param(mutable=True, initialize=2)

        with linear_expression() as e:
            e = e + 2
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = e + m.p*(1+m.v[0])
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = e + m.v[0]
            self.assertIs(e.__class__, _MutableLinearExpression)

            e = 2 + e
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = m.p*(1+m.v[0]) + e
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = m.v[0] + e
            self.assertIs(e.__class__, _MutableLinearExpression)

            e = e - 2
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = e - m.p(1+m.v[0])
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = e - m.v[0]
            self.assertIs(e.__class__, _MutableLinearExpression)

            e = 2 - e
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = m.p*(1+m.v[0]) - e
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = m.v[0] - e
            self.assertIs(e.__class__, _MutableLinearExpression)

        with linear_expression() as e:
            e += m.v[0]*m.v[1]
            self.assertIs(e.__class__, SumExpression)

        with linear_expression() as e:
            e = e + m.v[0]*m.v[1]
            self.assertIs(e.__class__, SumExpression)

        with linear_expression() as e:
            e = m.v[0]*m.v[1] + e
            self.assertIs(e.__class__, SumExpression)

    def test_mul_other(self):
        m = ConcreteModel()
        m.v = Var(range(5), initialize=1)
        m.p = Param(initialize=2, mutable=True)

        with linear_expression() as e:
            e += 1
            e = 2 * e
            self.assertEqual("2", str(e))
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = (1+m.v[0]) * e
            self.assertEqual("2 + 2*v[0]", str(e))
            self.assertIs(e.__class__, _MutableLinearExpression)
            try:
                e = m.v[0] * e
                self.fail("Expecting ValueError")
            except ValueError:
                pass

        with linear_expression() as e:
            e += 1
            e = e * m.p
            self.assertEqual("p", str(e))
            self.assertIs(e.__class__, _MutableLinearExpression)

        with linear_expression() as e:
            e += 1
            e = e * 0
            self.assertEqual(e.constant, 0)
            self.assertIs(e.__class__, _MutableLinearExpression)

        with linear_expression() as e:
            e += m.v[0]
            e = e * 2
            self.assertEqual("2*v[0]", str(e))
            self.assertIs(e.__class__, _MutableLinearExpression)

        with linear_expression() as e:
            e += 1
            e *= m.v[0]*m.v[1]
            self.assertIs(e.__class__, ProductExpression)

        with linear_expression() as e:
            e += 1
            e = e * (m.v[0]*m.v[1])
            self.assertIs(e.__class__, ProductExpression)

        with linear_expression() as e:
            e += 1
            e = (m.v[0]*m.v[1]) * e
            self.assertIs(e.__class__, ProductExpression)

    def test_div(self):
        m = ConcreteModel()
        m.v = Var(range(5), initialize=1)
        m.p = Param(initialize=2, mutable=True)

        with linear_expression() as e:
            e += m.v[0]
            e /= 2
            self.assertEqual("0.5*v[0]", str(e))
            self.assertIs(e.__class__, _MutableLinearExpression)

        with linear_expression() as e:
            e += m.v[0]
            e /= m.p
            self.assertEqual("(1/p)*v[0]", str(e))
            self.assertIs(e.__class__, _MutableLinearExpression)

        with linear_expression() as e:
            e += 1
            try:
                e /= m.v[0]
                self.fail("Expected ValueError")
            except:
                pass

    def test_div_other(self):
        m = ConcreteModel()
        m.v = Var(range(5), initialize=1)
        m.p = Param(initialize=2, mutable=True)

        with linear_expression() as e:
            e += m.v[0]
            try:
                e = 1 / e
                self.fail("Expected ValueError")
            except:
                pass

        with linear_expression() as e:
            e += 1
            e = 1 / e
            self.assertEqual("1.0",str(e))

    def test_negation_other(self):
        m = ConcreteModel()
        m.v = Var(range(5))

        with linear_expression() as e:
            e = 2 - e
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = - e
            self.assertIs(e.__class__, _MutableLinearExpression)

    def test_pow_other(self):
        m = ConcreteModel()
        m.v = Var(range(5))

        with linear_expression() as e:
            e = 2**e
            self.assertIs(e.__class__, NPV_PowExpression)
            e = m.v[0] + m.v[1]
            e = m.v[0]**e
            self.assertIs(e.__class__, PowExpression)


class TestNonlinearExpression(unittest.TestCase):

    def test_sum_other(self):
        m = ConcreteModel()
        m.v = Var(range(5))

        with nonlinear_expression() as e:
            e_ = 2 + m.v[0]
            self.assertIs(e_.__class__, SumExpression)
            e += e_
            self.assertIs(e.__class__, _MutableSumExpression)
            self.assertEqual(e.nargs(), 2)


#
# Test the logic of _decompose_linear_terms
#
class TestLinearDecomp(unittest.TestCase):

    def setUp(self):
        #
        # A hack to setup the _LinearExpression.vtypes data
        #
        #try:
        #    l = LinearExpression()
        #    l._combine_expr(None,None)
        #except:
        #    pass
        pass

    def test_numeric(self):
        self.assertEqual(list(_decompose_linear_terms(2.0)), [(2.0,None)])

    def test_NPV(self):
        M = ConcreteModel()
        M.q = Param(initialize=2)
        self.assertEqual(list(_decompose_linear_terms(M.q)), [(M.q,None)])

    def test_var(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(list(_decompose_linear_terms(M.v)), [(1,M.v)])

    def test_simple(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(list(_decompose_linear_terms(2*M.v)), [(2,M.v)])

    def test_sum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        self.assertEqual(list(_decompose_linear_terms(2+M.v)), [(2,None), (1,M.v)])
        self.assertEqual(list(_decompose_linear_terms(M.q+M.v)), [(2,None), (1,M.v)])
        self.assertEqual(list(_decompose_linear_terms(M.v+M.q)), [(1,M.v), (2,None)])
        self.assertEqual(list(_decompose_linear_terms(M.w+M.v)), [(1,M.w), (1,M.v)])

    def test_prod(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        self.assertEqual(list(_decompose_linear_terms(2*M.v)), [(2,M.v)])
        self.assertEqual(list(_decompose_linear_terms(M.q*M.v)), [(2,M.v)])
        self.assertEqual(list(_decompose_linear_terms(M.v*M.q)), [(2,M.v)])
        self.assertRaises(LinearDecompositionError, list, _decompose_linear_terms(M.w*M.v))

    def test_negation(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(list(_decompose_linear_terms(-M.v)), [(-1,M.v)])
        self.assertEqual(list(_decompose_linear_terms(-(2+M.v))), [(-2,None), (-1,M.v)])

    def test_reciprocal(self):
        M = ConcreteModel()
        M.v = Var()
        M.q = Param(initialize=2)
        self.assertRaises(LinearDecompositionError, list, _decompose_linear_terms(1/M.v))
        self.assertEqual(list(_decompose_linear_terms(1/M.q)), [(0.5,None)])
        
    def test_multisum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        e = SumExpression([2])
        self.assertEqual(list(_decompose_linear_terms(e)), [(2,None)])
        e = SumExpression([2,M.v])
        self.assertEqual(list(_decompose_linear_terms(e)), [(2,None), (1,M.v)])
        e = SumExpression([2,M.q+M.v])
        self.assertEqual(list(_decompose_linear_terms(e)), [(2,None), (2,None), (1,M.v)])
        e = SumExpression([2,M.q+M.v,M.w])
        self.assertEqual(list(_decompose_linear_terms(e)), [(2,None), (2,None), (1,M.v), (1,M.w)])
        

#
# Test the logic of decompose_term()
#
class Test_decompose_linear_terms(unittest.TestCase):

    def test_numeric(self):
        self.assertEqual(decompose_term(2.0), (True,[(2.0,None)]))

    def test_NPV(self):
        M = ConcreteModel()
        M.q = Param(initialize=2)
        self.assertEqual(decompose_term(M.q), (True, [(M.q,None)]))

    def test_var(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(decompose_term(M.v), (True, [(1,M.v)]))

    def test_simple(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(decompose_term(2*M.v), (True, [(2,M.v)]))

    def test_sum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        self.assertEqual(decompose_term(2+M.v),   (True, [(2,None), (1,M.v)]))
        self.assertEqual(decompose_term(M.q+M.v), (True, [(2,None), (1,M.v)]))
        self.assertEqual(decompose_term(M.v+M.q), (True, [(1,M.v), (2,None)]))
        self.assertEqual(decompose_term(M.v+M.w), (True, [(1,M.v), (1,M.w)]))

    def test_prod(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        self.assertEqual(decompose_term(2*M.v),   (True, [(2,M.v)]))
        self.assertEqual(decompose_term(M.q*M.v), (True, [(2,M.v)]))
        self.assertEqual(decompose_term(M.v*M.q), (True, [(2,M.v)]))
        self.assertEqual(decompose_term(M.w*M.v), (False, None))

    def test_negation(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(decompose_term(-M.v),     (True, [(-1,M.v)]))
        self.assertEqual(decompose_term(-(2+M.v)), (True, [(-2,None), (-1,M.v)]))

    def test_reciprocal(self):
        M = ConcreteModel()
        M.v = Var()
        M.q = Param(initialize=2)
        M.p = Param(initialize=2, mutable=True)
        self.assertEqual(decompose_term(1/M.v), (False, None))
        self.assertEqual(decompose_term(1/M.q), (True, [(0.5,None)]))
        e = 1/M.p
        self.assertEqual(decompose_term(e), (True, [(e,None)]))
        
    def test_multisum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=3)
        e = SumExpression([2])
        self.assertEqual(decompose_term(e), (True, [(2,None)]))
        e = SumExpression([2,M.v])
        self.assertEqual(decompose_term(e), (True, [(2,None), (1,M.v)]))
        e = SumExpression([2,M.q+M.v])
        self.assertEqual(decompose_term(e), (True, [(2,None), (3,None), (1,M.v)]))
        e = SumExpression([2,M.q+M.v,M.w])
        self.assertEqual(decompose_term(e), (True, [(2,None), (3,None), (1,M.v), (1,M.w)]))

    def test_linear(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        with linear_expression() as e:
            e += 2
            #
            # When the linear expression is constant, then it will be
            # identified as not potentially variable, and the expression returned
            # will be itself.
            #
            self.assertEqual(decompose_term(e), (True, [(e,None)]))
            e += M.v
            self.assertEqual(decompose_term(-e), (True, [(-2,None), (-1,M.v)]))
        
def x_(m,i):
    return i+1
def P_(m,i):
    return 10-i

#
# Test pickle logic
#
class Test_pickle(unittest.TestCase):

    def test_simple(self):
        M = ConcreteModel()
        M.v = Var()
        e = 2*M.v
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        flag, terms = decompose_term(e_)
        self.assertTrue(flag)
        self.assertEqual(terms[0][0], 2)
        self.assertEqual(str(terms[0][1]), str(M.v))

    def test_sum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        e = M.v+M.q
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        flag, terms = decompose_term(e_)
        self.assertTrue(flag)
        self.assertEqual(terms[0][0], 1)
        self.assertEqual(str(terms[0][1]), str(M.v))
        self.assertEqual(terms[1][0], 2)
        self.assertEqual(terms[1][1], None)

    def Xtest_Sum(self):
        M = ConcreteModel()
        A = range(5)
        M.v = Var(A)
        e = quicksum(M.v[i] for i in M.v)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        flag, terms = decompose_term(e_)
        self.assertTrue(flag)
        self.assertEqual(terms[0][0], 1)
        self.assertEqual(str(terms[0][1]), str(M.v[0]))
        self.assertEqual(terms[1][0], 1)

    def test_prod(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        e = M.v*M.q
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        flag, terms = decompose_term(e_)
        self.assertTrue(flag)
        self.assertEqual(terms[0][0], 2)
        self.assertEqual(str(terms[0][1]), str(M.v))

    def test_negation(self):
        M = ConcreteModel()
        M.v = Var()
        e = -(2+M.v)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        flag, terms = decompose_term(e_)
        self.assertTrue(flag)
        self.assertEqual(terms[0][0], -2)
        self.assertEqual(terms[0][1], None)
        self.assertEqual(terms[1][0], -1)
        self.assertEqual(str(terms[1][1]), str(M.v))

    def test_reciprocal(self):
        M = ConcreteModel()
        M.v = Var()
        M.q = Param(initialize=2)
        M.p = Param(initialize=2, mutable=True)
        e = 1/M.p
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        flag, terms = decompose_term(e_)
        self.assertTrue(flag)
        self.assertEqual(terms[0][0], 0.5)
        self.assertEqual(terms[0][1], None)
        
    def test_multisum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=3)
        e = SumExpression([2,M.q+M.v,M.w])
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        flag, terms = decompose_term(e_)
        self.assertTrue(flag)
        self.assertEqual(terms[0][0], 2)
        self.assertEqual(terms[0][1], None)
        self.assertEqual(terms[1][0], 3)
        self.assertEqual(terms[1][1], None)
        self.assertEqual(terms[2][0], 1)
        self.assertEqual(str(terms[2][1]), str(M.v))
        self.assertEqual(terms[3][0], 1)
        self.assertEqual(str(terms[3][1]), str(M.w))

    def test_linear(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        with linear_expression() as e:
            e += 2
            #
            # When the linear expression is constant, then it will be
            # identified as not potentially variable, and the expression returned
            # will be itself.
            #
            self.assertEqual(decompose_term(e), (True, [(e,None)]))
            e += M.v
            s = pickle.dumps(-e)
            e_ = pickle.loads(s)
            flag, terms = decompose_term(e_)
            self.assertTrue(flag)
            self.assertEqual(terms[0][0], -2)
            self.assertEqual(terms[0][1], None)
            self.assertEqual(terms[1][0], -1)
            self.assertEqual(str(terms[1][1]), str(M.v))
        
    def test_ExprIf(self):
        M = ConcreteModel()
        M.v = Var()
        e = Expr_if(M.v, 1, 0)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertEqual(type(e.arg(0)), type(e_.arg(0)))
        self.assertEqual(e.arg(1), e_.arg(1))
        self.assertEqual(e.arg(2), e_.arg(2))

    def test_getitem(self):
        m = ConcreteModel()
        m.I = RangeSet(1,9)
        m.x = Var(m.I, initialize=x_)
        m.P = Param(m.I, initialize=P_, mutable=True)
        t = IndexTemplate(m.I)

        e = m.x[t+m.P[t+1]] + 3
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertEqual("x[{I} + P[{I} + 1]] + 3", str(e))

    def test_abs(self):
        M = ConcreteModel()
        M.v = Var()
        e = abs(M.v)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertEqual(str(e), str(e_))

    def test_sin(self):
        M = ConcreteModel()
        M.v = Var()
        e = sin(M.v)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertEqual(str(e), str(e_))

    def test_external_fcn(self):
        model = ConcreteModel()
        model.a = Var()
        model.x = ExternalFunction(library='foo.so', function='bar')
        e = model.x(model.a, 1, "foo", [])
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertEqual(type(e_), type(e))
        self.assertEqual(type(e_.arg(0)), type(e.arg(0)))
        self.assertEqual(type(e_.arg(1)), type(e.arg(1)))
        self.assertEqual(type(e_.arg(2)), type(e.arg(2)))

#
# Every class that is duck typed to be a named expression
# should be tested here.
#
class TestNamedExpressionDuckTyping(unittest.TestCase):

    def check_api(self, obj):
        self.assertTrue(hasattr(obj, 'nargs'))
        self.assertTrue(hasattr(obj, 'arg'))
        self.assertTrue(hasattr(obj, 'args'))
        self.assertTrue(hasattr(obj, '__call__'))
        self.assertTrue(hasattr(obj, 'to_string'))
        self.assertTrue(hasattr(obj, '_precedence'))
        self.assertTrue(hasattr(obj, '_to_string'))
        self.assertTrue(hasattr(obj, 'clone'))
        self.assertTrue(hasattr(obj, 'create_node_with_local_data'))
        self.assertTrue(hasattr(obj, 'is_constant'))
        self.assertTrue(hasattr(obj, 'is_fixed'))
        self.assertTrue(hasattr(obj, '_is_fixed'))
        self.assertTrue(hasattr(obj, 'is_potentially_variable'))
        self.assertTrue(hasattr(obj, 'is_named_expression_type'))
        self.assertTrue(hasattr(obj, 'is_expression_type'))
        self.assertTrue(hasattr(obj, 'polynomial_degree'))
        self.assertTrue(hasattr(obj, '_compute_polynomial_degree'))
        self.assertTrue(hasattr(obj, '_apply_operation'))

    def test_Objective(self):
        M = ConcreteModel()
        M.x = Var()
        M.e = Objective(expr=M.x)
        self.check_api(M.e)

    def test_Expression(self):
        M = ConcreteModel()
        M.x = Var()
        M.e = Expression(expr=M.x)
        self.check_api(M.e)

    def test_ExpressionIndex(self):
        M = ConcreteModel()
        M.x = Var()
        M.e = Expression([0])
        M.e[0] = M.x
        self.check_api(M.e[0])

    def test_expression(self):
        x = variable()
        e = expression()
        e.expr = x
        self.check_api(e)

    def test_objective(self):
        x = variable()
        e = objective()
        e.expr = x
        self.check_api(e)


class TestNumValueDuckTyping(unittest.TestCase):

    def check_api(self, obj):
        self.assertTrue(hasattr(obj, 'is_fixed'))
        self.assertTrue(hasattr(obj, 'is_constant'))
        self.assertTrue(hasattr(obj, 'is_parameter_type'))
        self.assertTrue(hasattr(obj, 'is_potentially_variable'))
        self.assertTrue(hasattr(obj, 'is_variable_type'))
        self.assertTrue(hasattr(obj, 'is_named_expression_type'))
        self.assertTrue(hasattr(obj, 'is_expression_type'))
        self.assertTrue(hasattr(obj, '_compute_polynomial_degree'))
        self.assertTrue(hasattr(obj, '__call__'))
        self.assertTrue(hasattr(obj, 'to_string'))

    def test_Param(self):
        M = ConcreteModel()
        M.x = Param()
        self.check_api(M.x)

    def test_MutableParam(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        self.check_api(M.x)

    def test_MutableParamIndex(self):
        M = ConcreteModel()
        M.x = Param([0], initialize=10, mutable=True)
        self.check_api(M.x[0])

    def test_Var(self):
        M = ConcreteModel()
        M.x = Var()
        self.check_api(M.x)

    def test_VarIndex(self):
        M = ConcreteModel()
        M.x = Var([0])
        self.check_api(M.x[0])

    def test_variable(self):
        x = variable()
        self.check_api(x)

class TestDirect_LinearExpression(unittest.TestCase):

    def test_LinearExpression_Param(self):
        m = ConcreteModel()
        N = 10
        S = list(range(1,N+1))
        m.x = Var(S, initialize=lambda m,i: 1.0/i)
        m.P = Param(S, initialize=lambda m,i: i)
        m.obj = Objective(expr=LinearExpression(constant=1.0, linear_coefs=[m.P[i] for i in S], linear_vars=[m.x[i] for i in S]))

        # test that the expression evaluates correctly
        self.assertAlmostEqual(value(m.obj), N+1)

        # test that the standard repn can be constructed
        repn = generate_standard_repn(m.obj.expr)
        self.assertAlmostEqual(repn.constant, 1.0)
        self.assertTrue(len(repn.linear_coefs) == N)
        self.assertTrue(len(repn.linear_vars) == N)

    def test_LinearExpression_Number(self):
        m = ConcreteModel()
        N = 10
        S = list(range(1,N+1))
        m.x = Var(S, initialize=lambda m,i: 1.0/i)
        m.obj = Objective(expr=LinearExpression(constant=1.0, linear_coefs=[i for i in S], linear_vars=[m.x[i] for i in S]))

        # test that the expression evaluates correctly
        self.assertAlmostEqual(value(m.obj), N+1)

        # test that the standard repn can be constructed
        repn = generate_standard_repn(m.obj.expr)
        self.assertAlmostEqual(repn.constant, 1.0)
        self.assertTrue(len(repn.linear_coefs) == N)
        self.assertTrue(len(repn.linear_vars) == N)

    def test_LinearExpression_MutableParam(self):
        m = ConcreteModel()
        N = 10
        S = list(range(1,N+1))
        m.x = Var(S, initialize=lambda m,i: 1.0/i)
        m.P = Param(S, initialize=lambda m,i: i, mutable=True)
        m.obj = Objective(expr=LinearExpression(constant=1.0, linear_coefs=[m.P[i] for i in S], linear_vars=[m.x[i] for i in S]))

        # test that the expression evaluates correctly
        self.assertAlmostEqual(value(m.obj), N+1)

        # test that the standard repn can be constructed
        repn = generate_standard_repn(m.obj.expr)
        self.assertAlmostEqual(repn.constant, 1.0)
        self.assertTrue(len(repn.linear_coefs) == N)
        self.assertTrue(len(repn.linear_vars) == N)

    def test_LinearExpression_expression(self):
        m = ConcreteModel()
        N = 10
        S = list(range(1,N+1))
        m.x = Var(S, initialize=lambda m,i: 1.0/i)
        m.P = Param(S, initialize=lambda m,i: i, mutable=True)
        m.obj = Objective(expr=LinearExpression(constant=1.0, linear_coefs=[i*m.P[i] for i in S], linear_vars=[m.x[i] for i in S]))

        # test that the expression evaluates correctly
        self.assertAlmostEqual(value(m.obj), sum(i for i in S)+1)

        # test that the standard repn can be constructed
        repn = generate_standard_repn(m.obj.expr)
        self.assertAlmostEqual(repn.constant, 1.0)
        self.assertTrue(len(repn.linear_coefs) == N)
        self.assertTrue(len(repn.linear_vars) == N)

    def test_LinearExpression_polynomial_degree(self):
        m = ConcreteModel()
        m.S = RangeSet(2)
        m.var_1 = Var(initialize=0)
        m.var_2 = Var(initialize=0)
        m.var_3 = Var(m.S, initialize=0)

        def con_rule(model):
            return model.var_1 - (model.var_2 + sum_product(defaultdict(lambda: 6), model.var_3)) <= 0

        m.c1 = Constraint(rule=con_rule)

        m.var_1.fix(1)
        m.var_2.fix(1)
        m.var_3.fix(1)

        self.assertTrue(is_fixed(m.c1.body))
        self.assertEqual(polynomial_degree(m.c1.body), 0)

    def test_LinearExpression_is_fixed(self):
        m = ConcreteModel()
        m.S = RangeSet(2)
        m.var_1 = Var(initialize=0)
        m.var_2 = Var(initialize=0)
        m.var_3 = Var(m.S, initialize=0)

        def con_rule(model):
            return model.var_1 - (model.var_2 + sum_product(defaultdict(lambda: 6), model.var_3)) <= 0

        m.c1 = Constraint(rule=con_rule)

        m.var_1.fix(1)
        m.var_2.fix(1)

        self.assertFalse(is_fixed(m.c1.body))
        self.assertEqual(polynomial_degree(m.c1.body), 1)


if __name__ == "__main__":
    unittest.main()

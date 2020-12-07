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

import pickle
import os
import six
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest

from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Param, Constraint, inequality, display
import pyomo.core.expr.logical_expr as logical_expr
from pyomo.core.expr.logical_expr import (
    InequalityExpression, EqualityExpression, RangedExpression,
)

class TestGenerate_RelationalExpression(unittest.TestCase):

    def setUp(self):
        m = AbstractModel()
        m.I = Set()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.x = Var(m.I)
        self.m = m

    def tearDown(self):
        self.m = None

    def test_simpleEquality(self):
        #
        # Check the structure of a simple equality statement
        #
        m = self.m
        e = m.a == m.b
        self.assertIs(type(e), EqualityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)

    def test_equalityErrors(self):
        #
        # Check equality errors
        #
        m = self.m
        e = m.a == m.b
        #       =
        #      / \
        #     =   5
        #    / \
        #   a   b
        # Python 2.7 supports better testing of exceptions
        if sys.hexversion >= 0x02070000:
            self.assertRaisesRegexp(TypeError, "EqualityExpression .*"
                                   "sub-expressions is a relational",
                                   e.__eq__, m.a)
            self.assertRaisesRegexp(TypeError, "EqualityExpression .*"
                                   "sub-expressions is a relational",
                                   m.a.__eq__, e)

            # NB: cannot test the reverse here: _VarArray (correctly)
            # does not define __eq__
            self.assertRaisesRegexp(TypeError, "Argument .*"
                                    "is an indexed numeric value",
                                    m.a.__eq__, m.x)
        else:
            self.assertRaises(TypeError, e.__eq__, m.a)
            self.assertRaises(TypeError, m.a.__eq__, e)
            self.assertRaises(TypeError, m.a.__eq__, m.x)

        try:
            e == m.a
            self.fail("expected nested equality expression to raise TypeError")
        except TypeError:
            pass

        try:
            m.a == e
            self.fail("expected nested equality expression to raise TypeError")
        except TypeError:
            pass

        #
        # Test expression with an indexed variable
        #
        try:
            m.x == m.a
            self.fail("expected use of indexed variable to raise TypeError")
        except TypeError:
            pass

        try:
            m.a == m.x
            self.fail("expected use of indexed variable to raise TypeError")
        except TypeError:
            pass

    def test_simpleInequality1(self):
        #
        # Check the structure of a simple inequality
        #
        m = self.m
        #    <
        #   / \
        #  a   b
        e = m.a < m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        #self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, True)

        #    <=
        #   / \
        #  a   b
        e = m.a <= m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        #self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, False)

        #    >
        #   / \
        #  a   b
        e = m.a > m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        #self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, True)

        #    >=
        #   / \
        #  a   b
        e = m.a >= m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        #self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, False)

    def test_simpleInequality2(self):
        #
        # Check the structure of a simple inequality
        #
        m = self.m
        #    <
        #   / \
        #  a   b
        e = inequality(lower=m.a, body=m.b, strict=True)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        #self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, True)

        #    <=
        #   / \
        #  a   b
        e = inequality(lower=m.a, upper=m.b)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        #self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, False)

        #    >
        #   / \
        #  a   b
        e = inequality(lower=m.b, upper=m.a, strict=True)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        #self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, True)

        #    >=
        #   / \
        #  a   b
        e = m.a >= m.b
        e = inequality(body=m.b, upper=m.a)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        #self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, False)

        try:
            inequality(None, None)
            self.fail("expected invalid inequality error.")
        except ValueError:
            pass

        try:
            inequality(m.a, None)
            self.fail("expected invalid inequality error.")
        except ValueError:
            pass


class TestGenerate_RangedExpression(unittest.TestCase):

    def setUp(self):
        m = AbstractModel()
        m.I = Set()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.x = Var(m.I)
        self.m = m

    def tearDown(self):
        self.m = None

    def test_compoundInequality(self):
        m = self.m
        #       <
        #      / \
        #     <   c
        #    / \
        #   a   b
        e = inequality(m.a, m.b, m.c, strict=True)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(2), m.c)
        #self.assertEqual(len(e._strict), 2)
        self.assertEqual(e._strict[0], True)
        self.assertEqual(e._strict[1], True)

        #       <=
        #      / \
        #     <=  c
        #    / \
        #   a   b
        e = inequality(m.a, m.b, m.c)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(2), m.c)
        #self.assertEqual(len(e._strict), 2)
        self.assertEqual(e._strict[0], False)
        self.assertEqual(e._strict[1], False)

        #       >
        #      / \
        #     >   c
        #    / \
        #   a   b
        e = inequality(upper=m.c, body=m.b, lower=m.a, strict=True)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(2), m.c)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(0), m.a)
        #self.assertEqual(len(e._strict), 2)
        self.assertEqual(e._strict[0], True)
        self.assertEqual(e._strict[1], True)

        #       >=
        #      / \
        #     >=  c
        #    / \
        #   a   b
        e = inequality(upper=m.c, body=m.b, lower=m.a)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(2), m.c)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(0), m.a)
        #self.assertEqual(len(e._strict), 2)
        self.assertEqual(e._strict[0], False)
        self.assertEqual(e._strict[1], False)

        #       <=
        #      / \
        #     <=  0
        #    / \
        #   0   a
        e = inequality(0, m.a, 0)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(2), 0)
        self.assertIs(e.arg(1), m.a)
        self.assertIs(e.arg(0), 0)
        #self.assertEqual(len(e._strict), 2)
        self.assertEqual(e._strict[0], False)
        self.assertEqual(e._strict[1], False)

        #       <
        #      / \
        #     <  0
        #    / \
        #   0   a
        e = inequality(0, m.a, 0, True)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(2), 0)
        self.assertIs(e.arg(1), m.a)
        self.assertIs(e.arg(0), 0)
        #self.assertEqual(len(e._strict), 2)
        self.assertEqual(e._strict[0], True)
        self.assertEqual(e._strict[1], True)

    def test_val1(self):
        m = ConcreteModel()
        m.v = Var(initialize=2)

        e = inequality(0, m.v, 2)
        self.assertEqual(e.__nonzero__(), True)
        e = inequality(0, m.v, 1)
        self.assertEqual(e.__nonzero__(), False)
        e = inequality(0, m.v, 2, strict=True)
        self.assertEqual(e.__nonzero__(), False)

    def test_val2(self):
        m = ConcreteModel()
        m.v = Var(initialize=2)

        e = 1 < m.v
        e = e <= 2
        self.assertEqual(e.__nonzero__(), True)
        e = 1 <= m.v
        e = e < 2
        self.assertEqual(e.__nonzero__(), False)


@unittest.skipIf(not logical_expr._using_chained_inequality, "Skipping tests of chained inequalities")
class TestGenerate_ChainedRelationalExpression(unittest.TestCase):

    def setUp(self):
        m = AbstractModel()
        m.I = Set()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.x = Var(m.I)
        self.m = m

    def tearDown(self):
        self.m = None

    def test_eval_compoundInequality(self):
        #
        # Evaluate a compound inequality
        #
        m = ConcreteModel()
        m.x = Var(initialize=0)

        # Implicit equalities
        self.assertTrue( 0 <= m.x <= 0 )
        self.assertIsNone(logical_expr._chainedInequality.prev)
        self.assertFalse( 1 <= m.x <= 1 )
        self.assertIsNone(logical_expr._chainedInequality.prev)
        self.assertFalse( -1 <= m.x <= -1 )
        self.assertIsNone(logical_expr._chainedInequality.prev)

        # Chained inequalities
        self.assertTrue( 0 <= m.x <= 1 )
        self.assertIsNone(logical_expr._chainedInequality.prev)
        self.assertFalse( 1 <= m.x <= 2 )
        self.assertIsNone(logical_expr._chainedInequality.prev)
        self.assertTrue( -1 <= m.x <= 0 )
        self.assertIsNone(logical_expr._chainedInequality.prev)

        # Chained inequalities
        self.assertFalse( 0 < m.x <= 1 )
        self.assertIsNone(logical_expr._chainedInequality.prev)
        self.assertTrue( 0 <= m.x < 1 )
        self.assertIsNone(logical_expr._chainedInequality.prev)
        self.assertFalse( 0 < m.x < 1 )
        self.assertIsNone(logical_expr._chainedInequality.prev)

    def test_compoundInequality_errors(self):
        #
        # Evaluate errors in a compound inequality
        #
        m = ConcreteModel()
        m.x = Var()

        #       >=
        #      / \
        #     <=  0
        #    / \
        #   0   x
        try:
            0 <= m.x >= 0
            self.fail("expected construction of relational expression to "
                      "generate a TypeError")
        except TypeError as e:
            self.assertIn(
                "Relational expression used in an unexpected Boolean context.",
                str(e) )

        #       >=
        #      / \
        #     <=  1
        #    / \
        #   0   x
        try:
            0 <= m.x >= 1
            self.fail("expected construction of relational expression to "
                      "generate a TypeError")
        except TypeError as e:
            self.assertIn(
                "Attempting to form a compound inequality with two lower bounds",
                str(e) )

        #       <=
        #      / \
        #     >=  1
        #    / \
        #   0   x
        try:
            0 >= m.x <= 1
            self.fail("expected construction of relational expression to "
                      "generate a TypeError")
        except TypeError as e:
            self.assertIn(
                "Attempting to form a compound inequality with two upper bounds",
                str(e) )

        #
        # Confirm error when 
        self.assertTrue(m.x <= 0)
        self.assertIsNotNone(logical_expr._chainedInequality.prev)
        try:
            m.x == 5
            self.fail("expected construction of relational expression to "
                      "generate a TypeError")
        except TypeError as e:
            self.assertIn(
                "Relational expression used in an unexpected Boolean context.",
                str(e) )

        self.assertTrue(m.x <= 0)
        self.assertIsNotNone(logical_expr._chainedInequality.prev)
        try:
            m.x*2 <= 5
            self.fail("expected construction of relational expression to "
                      "generate a TypeError")
        except TypeError as e:
            self.assertIn(
                "Relational expression used in an unexpected Boolean context.",
                str(e) )

        #
        # Error because expression is being detected in an unusual context
        #
        self.assertTrue(m.x <= 0)
        self.assertIsNotNone(logical_expr._chainedInequality.prev)
        try:
            m.x <= 0
            self.fail("expected construction of relational expression to "
                      "generate a TypeError")
        except TypeError as e:
            self.assertIn(
                "Relational expression used in an unexpected Boolean context.",
                str(e) )

    def test_inequalityErrors(self):
        #
        # Check inequality errors
        #
        m = self.m
        e = m.a <= m.b <= m.c
        e1 = m.a == m.b
        if sys.hexversion >= 0x02070000:
            # Python 2.7 supports better testing of exceptions
            #
            # Check error with indexed variable
            #
            self.assertRaisesRegexp(TypeError, "Argument .*"
                                    "is an indexed numeric value",
                                    m.a.__lt__, m.x)
            self.assertRaisesRegexp(TypeError, "Argument .*"
                                    "is an indexed numeric value",
                                    m.a.__gt__, m.x)

            #
            # Check error with more than two inequalities
            #
            self.assertRaisesRegexp(TypeError, "Cannot create an InequalityExpression where one of the sub-expressions is an equality or ranged expression:.*", e.__lt__, m.c)
            self.assertRaisesRegexp(TypeError, "Cannot create an InequalityExpression where one of the sub-expressions is an equality or ranged expression:.*", e.__gt__, m.c)

            #
            # Check error when both expressions are relational
            #
            self.assertRaisesRegexp(TypeError, "InequalityExpression .*"
                                   "one of the sub-expressions is an equality or ranged expression",
                                   e.__lt__, m.a < m.b)
            self.assertRaisesRegexp(TypeError, "InequalityExpression .*"
                                   "one of the sub-expressions is an equality or ranged expression",
                                   m.a.__lt__, e1)
            self.assertRaisesRegexp(TypeError, "InequalityExpression .*"
                                   "one of the sub-expressions is an equality or ranged expression",
                                   m.a.__gt__, e1)
        else:
            self.assertRaises(TypeError, m.a.__lt__, m.x)
            self.assertRaises(TypeError, m.a.__gt__, m.x)
            self.assertRaises(TypeError, e.__lt__, m.c)
            self.assertRaises(TypeError, e.__gt__, m.c)
            self.assertRaises(TypeError, e.__lt__, m.a < m.b)
            self.assertRaises(TypeError, m.a.__lt__, e1)
            self.assertRaises(TypeError, m.a.__gt__, e1)

        #
        # Check error with indexed variable
        #
        try:
            m.x < m.a
            self.fail("expected use of indexed variable to raise TypeError")
        except TypeError:
            pass
        try:
            m.a < m.x
            self.fail("expected use of indexed variable to raise TypeError")
        except TypeError:
            pass

        #
        # Check error with more than two relational expressions
        #
        try:
            e < m.c
            self.fail("expected 4-term inequality to raise ValueError")
        except TypeError:
            pass
        try:
            m.c < e
            self.fail("expected 4-term inequality to raise ValueError")
        except TypeError:
            pass
        try:
            e1 = m.a < m.b
            e < e1
            self.fail("expected inequality of inequalities to raise TypeError")
        except TypeError:
            pass
        try:
            m.a < (m.a == m.b)
            self.fail("expected equality within inequality to raise TypeError")
        except TypeError:
            pass
        try:
            m.a > (m.a == m.b)
            self.fail("expected equality within inequality to raise TypeError")
        except TypeError:
            pass


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

    def test_relational_ops(self):
        #
        # Relation of mutable parameters:  fixed, not constant, pvar
        #
        expr = self.instance.c < self.instance.d
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        expr = inequality(0, self.instance.c, self.instance.d)
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        expr = self.instance.c == self.instance.d
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        #
        # Relation of unfixed variable and mutable parameters:  not fixed, not constant, pvar
        #
        expr = self.instance.a <= self.instance.d
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = inequality(0, self.instance.a, self.instance.d)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = self.instance.a == self.instance.d
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Relation of unfixed variables:  not fixed, not constant, pvar
        #
        expr = self.instance.a * self.instance.a >= self.instance.b
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = self.instance.a * self.instance.a == self.instance.b
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = inequality(self.instance.b, self.instance.a * self.instance.a, 0)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Relation of fixed and unfixed variables:  not fixed, not constant, pvar
        #
        self.instance.a.fixed = True
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        #
        # Relation of fixed variables:  fixed, not constant, pvar
        #
        self.instance.b.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)


class TestMultiArgumentExpressions(unittest.TestCase):

    def test_double_sided_ineq(self):
        m = ConcreteModel()
        m.s = Set(initialize=[1.0,2.0,3.0,4.0,5.0])

        m.vmin = Param(m.s, initialize=lambda m,i: i)
        m.vmax = Param(m.s, initialize=lambda m,i: i**2)

        m.v = Var(m.s)

        def _con(m, i):
            return inequality(m.vmin[i]**2, m.v[i], m.vmax[i]**2)
        m.con = Constraint(m.s, rule=_con)

        OUT = six.StringIO()
        for i in m.s:
            OUT.write(str(_con(m,i)))
            OUT.write("\n")
        display(m.con, ostream=OUT)

        if logical_expr._using_chained_inequality:
            reference="""1.0  <=  v[1.0]  <=  1.0
4.0  <=  v[2.0]  <=  16.0
9.0  <=  v[3.0]  <=  81.0
16.0  <=  v[4.0]  <=  256.0
25.0  <=  v[5.0]  <=  625.0
con : Size=5
    Key : Lower : Body : Upper
    1.0 :   1.0 : None :   1.0
    2.0 :   4.0 : None :  16.0
    3.0 :   9.0 : None :  81.0
    4.0 :  16.0 : None : 256.0
    5.0 :  25.0 : None : 625.0
"""
        else:
            reference="""1.0  <=  v[1.0]  <=  1.0
4.0  <=  v[2.0]  <=  16.0
9.0  <=  v[3.0]  <=  81.0
16.0  <=  v[4.0]  <=  256.0
25.0  <=  v[5.0]  <=  625.0
con : Size=5
    Key : Lower : Body : Upper
    1.0 :   1.0 : None :   1.0
    2.0 :   4.0 : None :  16.0
    3.0 :   9.0 : None :  81.0
    4.0 :  16.0 : None : 256.0
    5.0 :  25.0 : None : 625.0
"""
        self.assertEqual(OUT.getvalue(), reference)


#
# Test pickle logic
#
class Test_pickle(unittest.TestCase):


    def test_ineq(self):
        M = ConcreteModel()
        M.v = Var()
        e = M.v >= 0
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertEqual(str(e), str(e_))

    def test_range(self):
        M = ConcreteModel()
        M.v = Var()
        e = inequality(0, M.v, 1)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertEqual(str(e), str(e_))

    def test_eq(self):
        M = ConcreteModel()
        M.v = Var()
        e = M.v == 0
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertEqual(str(e), str(e_))


if __name__ == "__main__":
    unittest.main()

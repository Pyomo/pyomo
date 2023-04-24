#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
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
import io
import sys
from os.path import abspath, dirname

currdir = dirname(abspath(__file__)) + os.sep

import pyomo.common.unittest as unittest

from pyomo.environ import (
    AbstractModel,
    ConcreteModel,
    Set,
    Var,
    Param,
    Constraint,
    inequality,
    display,
)
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
    InequalityExpression,
    EqualityExpression,
    RangedExpression,
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
        with self.assertRaisesRegex(
            TypeError,
            "Attempting to use a non-numeric type "
            r"\(EqualityExpression\) in a numeric expression context.",
        ):
            e == m.a
        with self.assertRaisesRegex(
            TypeError,
            "Attempting to use a non-numeric type "
            r"\(EqualityExpression\) in a numeric expression context.",
        ):
            m.a == e

        #
        # Test expression with an indexed variable
        #
        with self.assertRaisesRegex(
            TypeError, "Argument .* is an indexed numeric value"
        ):
            m.x == m.a
        with self.assertRaisesRegex(
            TypeError, "Argument .* is an indexed numeric value"
        ):
            m.a == m.x

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
        # self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, True)

        #    <=
        #   / \
        #  a   b
        e = m.a <= m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        # self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, False)

        #    >
        #   / \
        #  a   b
        e = m.a > m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        # self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, True)

        #    >=
        #   / \
        #  a   b
        e = m.a >= m.b
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        # self.assertEqual(len(e._strict), 1)
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
        # self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, True)

        #    <=
        #   / \
        #  a   b
        e = inequality(lower=m.a, upper=m.b)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        # self.assertEqual(len(e._strict), 1)
        self.assertEqual(e._strict, False)

        #    >
        #   / \
        #  a   b
        e = inequality(lower=m.b, upper=m.a, strict=True)
        self.assertIs(type(e), InequalityExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1), m.a)
        # self.assertEqual(len(e._strict), 1)
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
        # self.assertEqual(len(e._strict), 1)
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
        # self.assertEqual(len(e._strict), 2)
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
        # self.assertEqual(len(e._strict), 2)
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
        # self.assertEqual(len(e._strict), 2)
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
        # self.assertEqual(len(e._strict), 2)
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
        # self.assertEqual(len(e._strict), 2)
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
        # self.assertEqual(len(e._strict), 2)
        self.assertEqual(e._strict[0], True)
        self.assertEqual(e._strict[1], True)

    def test_val1(self):
        m = ConcreteModel()
        m.v = Var(initialize=2)

        e = inequality(0, m.v, 2)
        self.assertEqual(value(e), True)
        e = inequality(0, m.v, 1)
        self.assertEqual(value(e), False)
        e = inequality(0, m.v, 2, strict=True)
        self.assertEqual(value(e), False)

    def test_val2(self):
        m = ConcreteModel()
        m.v = Var(initialize=2)

        e = 1 < m.v
        e = e <= 2
        self.assertEqual(value(e), True)
        e = 1 <= m.v
        e = e < 2
        self.assertEqual(value(e), False)


#
# Fixed               - Expr has a fixed value
# Constant            - Expr only contains constants and immutable parameters
# PotentiallyVariable - Expr contains one or more variables
#
class TestIsFixedIsConstant(unittest.TestCase):
    def setUp(self):
        # This class tests the Pyomo 5.x expression trees

        def d_fn(model):
            return model.c + model.c

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
        m.s = Set(initialize=[1.0, 2.0, 3.0, 4.0, 5.0])

        m.vmin = Param(m.s, initialize=lambda m, i: i)
        m.vmax = Param(m.s, initialize=lambda m, i: i**2)

        m.v = Var(m.s)

        def _con(m, i):
            return inequality(m.vmin[i] ** 2, m.v[i], m.vmax[i] ** 2)

        m.con = Constraint(m.s, rule=_con)

        OUT = io.StringIO()
        for i in m.s:
            OUT.write(str(_con(m, i)))
            OUT.write("\n")
        display(m.con, ostream=OUT)

        reference = """1.0  <=  v[1.0]  <=  1.0
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

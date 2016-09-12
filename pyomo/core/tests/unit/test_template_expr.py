#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#

import pyutilib.th as unittest

from pyomo.environ import ConcreteModel, RangeSet, Param, Var, Set
from pyomo.core.base import expr as EXPR
from pyomo.core.base import expr_common
from pyomo.core.base.template_expr import (
    IndexTemplate, 
    substitute_template_expression, 
    substitute_template_with_param,
    substitute_template_with_index,
)

class ExpressionObjectTester(object):
    def setUp(self):
        self.m = m = ConcreteModel()
        m.I = RangeSet(1,9)
        m.J = RangeSet(10,19)
        m.x = Var(m.I, initialize=lambda m,i: i+1)
        m.P = Param(m.I, initialize=lambda m,i: 10-i, mutable=True)
        m.p = Param(m.I, m.J, initialize=lambda m,i,j: 100*i+j)
        m.s = Set(m.I, initialize=lambda m,i:range(i))

    def test_template_scalar(self):
        m = self.m
        t = IndexTemplate(m.I)
        e = m.x[t]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.x)
        self.assertEqual(e._args, (t,))
        self.assertFalse(e.is_constant())
        self.assertFalse(e.is_fixed())
        self.assertEqual(e.polynomial_degree(), 1)
        t.set_value(5)
        self.assertEqual(e(), 6)
        self.assertIs(e.resolve_template(), m.x[5])
        t.set_value(None)

        e = m.p[t,10]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.p)
        self.assertEqual(e._args, (t,10))
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertEqual(e.polynomial_degree(), 0)
        t.set_value(5)
        self.assertEqual(e(), 510)
        self.assertIs(e.resolve_template(), m.p[5,10])
        t.set_value(None)

        e = m.p[5,t]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.p)
        self.assertEqual(e._args, (5,t))
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertEqual(e.polynomial_degree(), 0)
        t.set_value(10)
        self.assertEqual(e(), 510)
        self.assertIs(e.resolve_template(), m.p[5,10])
        t.set_value(None)

        e = m.s[t]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.s)
        self.assertEqual(e._args, (t,))
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertEqual(e.polynomial_degree(), 0)
        t.set_value(5)
        self.assertRaises(TypeError, e)
        self.assertIs(e.resolve_template(), m.s[5])
        t.set_value(None)


    def test_template_operation(self):
        m = self.m
        t = IndexTemplate(m.I)
        e = m.x[t+m.P[5]]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(e._args[0]._args[1], m.P[5])


    def test_nested_template_operation(self):
        m = self.m
        t = IndexTemplate(m.I)
        e = m.x[t+m.P[t+1]]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(type(e._args[0]._args[1]), EXPR._GetItemExpression)
        self.assertIs(type(e._args[0]._args[1]._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[1]._args[0]._args[0], t)


    def test_template_name(self):
        m = self.m
        t = IndexTemplate(m.I)

        E = m.x[t+m.P[1+t]] + m.P[1]
        self.assertEqual( str(E), "x( {I} + P( 1 + {I} ) ) + P[1]" )

        E = m.x[t+m.P[1+t]**2.]**2. + m.P[1]
        self.assertEqual( str(E), "x( {I} + P( 1 + {I} )**2.0 )**2.0 + P[1]" )


    def test_template_in_expression(self):
        m = self.m
        t = IndexTemplate(m.I)

        E = m.x[t+m.P[t+1]] + m.P[1]
        self.assertIs(type(E), EXPR._SumExpression)
        e = E._args[0]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(type(e._args[0]._args[1]), EXPR._GetItemExpression)
        self.assertIs(type(e._args[0]._args[1]._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[1]._args[0]._args[0], t)

        E = m.P[1] + m.x[t+m.P[t+1]]
        self.assertIs(type(E), EXPR._SumExpression)
        e = E._args[1]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(type(e._args[0]._args[1]), EXPR._GetItemExpression)
        self.assertIs(type(e._args[0]._args[1]._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[1]._args[0]._args[0], t)

        E = m.x[t+m.P[t+1]] + 1
        self.assertIs(type(E), EXPR._SumExpression)
        e = E._args[0]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(type(e._args[0]._args[1]), EXPR._GetItemExpression)
        self.assertIs(type(e._args[0]._args[1]._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[1]._args[0]._args[0], t)

        E = 1 + m.x[t+m.P[t+1]]
        self.assertIs(type(E), EXPR._SumExpression)
        # Note: in coopr3, the 1 is held in a separate attribute (so
        # len(_args) is 1), whereas in pyomo4 the constant is a proper
        # argument.  The -1 index works for both modes.
        e = E._args[-1]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(type(e._args[0]._args[1]), EXPR._GetItemExpression)
        self.assertIs(type(e._args[0]._args[1]._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[1]._args[0]._args[0], t)


    def test_clone(self):
        m = self.m
        t = IndexTemplate(m.I)

        E_base = m.x[t+m.P[t+1]] + m.P[1]
        E = E_base.clone()
        self.assertIs(type(E), EXPR._SumExpression)
        e = E._args[0]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIsNot(e, E_base._args[0])
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(type(e._args[0]._args[1]), EXPR._GetItemExpression)
        self.assertIs(type(e._args[0]._args[1]),
                      type(E_base._args[0]._args[0]._args[1]))
        self.assertIsNot(e._args[0]._args[1],
                         E_base._args[0]._args[0]._args[1])
        self.assertIs(type(e._args[0]._args[1]._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[1]._args[0]._args[0], t)

        E_base = m.P[1] + m.x[t+m.P[t+1]]
        E = E_base.clone()
        self.assertIs(type(E), EXPR._SumExpression)
        e = E._args[1]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIsNot(e, E_base._args[0])
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(type(e._args[0]._args[1]), EXPR._GetItemExpression)
        self.assertIs(type(e._args[0]._args[1]),
                      type(E_base._args[1]._args[0]._args[1]))
        self.assertIsNot(e._args[0]._args[1],
                         E_base._args[1]._args[0]._args[1])
        self.assertIs(type(e._args[0]._args[1]._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[1]._args[0]._args[0], t)

        E_base = m.x[t+m.P[t+1]] + 1
        E = E_base.clone()
        self.assertIs(type(E), EXPR._SumExpression)
        e = E._args[0]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIsNot(e, E_base._args[0])
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(type(e._args[0]._args[1]), EXPR._GetItemExpression)
        self.assertIs(type(e._args[0]._args[1]),
                      type(E_base._args[0]._args[0]._args[1]))
        self.assertIsNot(e._args[0]._args[1],
                         E_base._args[0]._args[0]._args[1])
        self.assertIs(type(e._args[0]._args[1]._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[1]._args[0]._args[0], t)

        E_base = 1 + m.x[t+m.P[t+1]]
        E = E_base.clone()
        self.assertIs(type(E), EXPR._SumExpression)
        # Note: in coopr3, the 1 is held in a separate attribute (so
        # len(_args) is 1), whereas in pyomo4 the constant is a proper
        # argument.  The -1 index works for both modes.
        e = E._args[-1]
        self.assertIs(type(e), EXPR._GetItemExpression)
        self.assertIsNot(e, E_base._args[0])
        self.assertIs(e._base, m.x)
        self.assertEqual(len(e._args), 1)
        self.assertIs(type(e._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[0], t)
        self.assertIs(type(e._args[0]._args[1]), EXPR._GetItemExpression)
        self.assertIs(type(e._args[0]._args[1]),
                      type(E_base._args[-1]._args[0]._args[1]))
        self.assertIsNot(e._args[0]._args[1],
                         E_base._args[-1]._args[0]._args[1])
        self.assertIs(type(e._args[0]._args[1]._args[0]), EXPR._SumExpression)
        self.assertIs(e._args[0]._args[1]._args[0]._args[0], t)
        

class TestTemplate_expressionObjects_coopr3\
      ( ExpressionObjectTester, unittest.TestCase ):
    def setUp(self):
        # This class tests the Coopr 3.x expression trees
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        ExpressionObjectTester.setUp(self)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)


class TestTemplate_expressionObjects_pyomo4\
      ( ExpressionObjectTester, unittest.TestCase ):
    def setUp(self):
        # This class tests the Pyomo 4.x expression trees
        EXPR.set_expression_tree_format(expr_common.Mode.pyomo4_trees)
        ExpressionObjectTester.setUp(self)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

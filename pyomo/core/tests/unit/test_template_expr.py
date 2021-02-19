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

import pyutilib.th as unittest

from pyomo.environ import (
    ConcreteModel, AbstractModel, RangeSet, Param, Var, Set, value,
    Integers,
)
import pyomo.core.expr.current as EXPR
from pyomo.core.expr.template_expr import (
    IndexTemplate,
    TemplateExpressionError,
    _GetItemIndexer,
    resolve_template,
    templatize_constraint,
    substitute_template_expression,
    substitute_getitem_with_param,
    substitute_template_with_value,
)


class TestTemplateExpressions(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()
        m.I = RangeSet(1,9)
        m.J = RangeSet(10,19)
        m.x = Var(m.I, initialize=lambda m,i: i+1)
        m.P = Param(m.I, initialize=lambda m,i: 10-i, mutable=True)
        m.p = Param(m.I, m.J, initialize=lambda m,i,j: 100*i+j)
        m.s = Set(m.I, initialize=lambda m,i:range(i))

    def test_nonTemplates(self):
        m = self.m
        self.assertIs(resolve_template(m.x[1]), m.x[1])
        e = m.x[1] + m.x[2]
        self.assertIs(resolve_template(e), e)

    def test_IndexTemplate(self):
        m = self.m
        i = IndexTemplate(m.I)
        with self.assertRaisesRegex(
                TemplateExpressionError,
                "Evaluating uninitialized IndexTemplate"):
            value(i)

        self.assertEqual(str(i), "{I}")

        i.set_value(5)
        self.assertEqual(value(i), 5)
        self.assertIs(resolve_template(i), 5)

    def test_template_scalar(self):
        m = self.m
        t = IndexTemplate(m.I)
        e = m.x[t]
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.args, (m.x, t))
        self.assertFalse(e.is_constant())
        self.assertFalse(e.is_fixed())
        self.assertEqual(e.polynomial_degree(), 1)
        self.assertEqual(str(e), "x[{I}]")
        t.set_value(5)
        v = e()
        self.assertIn(type(v), (int, float))
        self.assertEqual(v, 6)
        self.assertIs(resolve_template(e), m.x[5])
        t.set_value()

        e = m.p[t,10]
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.args, (m.p,t,10))
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(str(e), "p[{I},10]")
        t.set_value(5)
        v = e()
        self.assertIn(type(v), (int, float))
        self.assertEqual(v, 510)
        self.assertIs(resolve_template(e), m.p[5,10])
        t.set_value()

        e = m.p[5,t]
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.args, (m.p,5,t))
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(str(e), "p[5,{I}]")
        t.set_value(10)
        v = e()
        self.assertIn(type(v), (int, float))
        self.assertEqual(v, 510)
        self.assertIs(resolve_template(e), m.p[5,10])
        t.set_value()

    def test_template_scalar_with_set(self):
        m = self.m
        t = IndexTemplate(m.I)
        e = m.s[t]
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.args, (m.s,t))
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertEqual(e.polynomial_degree(), 0)
        self.assertEqual(str(e), "s[{I}]")
        t.set_value(5)
        v = e()
        self.assertIs(v, m.s[5])
        self.assertIs(resolve_template(e), m.s[5])
        t.set_value()

    def test_template_operation(self):
        m = self.m
        t = IndexTemplate(m.I)
        e = m.x[t+m.P[5]]
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(e.arg(1).arg(1), m.P[5])
        self.assertEqual(str(e), "x[{I} + P[5]]")

    def test_nested_template_operation(self):
        m = self.m
        t = IndexTemplate(m.I)
        e = m.x[t+m.P[t+1]]
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.GetItemExpression)
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)
        self.assertEqual(str(e), "x[{I} + P[{I} + 1]]")

    def test_block_templates(self):
        m = ConcreteModel()
        m.T = RangeSet(3)
        @m.Block(m.T)
        def b(b, i):
            b.x = Var(initialize=i)

            @b.Block(m.T)
            def bb(bb, j):
                bb.I =RangeSet(i*j)
                bb.y = Var(bb.I, initialize=lambda m,i:i)
        t = IndexTemplate(m.T)
        e = m.b[t].x
        self.assertIs(type(e), EXPR.GetAttrExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), EXPR.GetItemExpression)
        self.assertIs(e.arg(0).arg(0), m.b)
        self.assertEqual(e.arg(0).nargs(), 2)
        self.assertIs(e.arg(0).arg(1), t)
        self.assertEqual(str(e), "b[{T}].x")
        t.set_value(2)
        v = e()
        self.assertIn(type(v), (int, float))
        self.assertEqual(v, 2)
        self.assertIs(resolve_template(e), m.b[2].x)
        t.set_value()

        e = m.b[t].bb[t].y[1]
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(str(e), "b[{T}].bb[{T}].y[1]")
        t.set_value(2)
        v = e()
        self.assertIn(type(v), (int, float))
        self.assertEqual(v, 1)
        self.assertIs(resolve_template(e), m.b[2].bb[2].y[1])

    def test_template_name(self):
        m = self.m
        t = IndexTemplate(m.I)

        E = m.x[t+m.P[1+t]] + m.P[1]
        self.assertEqual( str(E), "x[{I} + P[1 + {I}]] + P[1]")

        E = m.x[t+m.P[1+t]**2.]**2. + m.P[1]
        self.assertEqual( str(E), "x[{I} + P[1 + {I}]**2.0]**2.0 + P[1]")

    def test_template_in_expression(self):
        m = self.m
        t = IndexTemplate(m.I)

        E = m.x[t+m.P[t+1]] + m.P[1]
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(0)
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.GetItemExpression)
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)

        E = m.P[1] + m.x[t+m.P[t+1]]
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(1)
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.GetItemExpression)
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)

        E = m.x[t+m.P[t+1]] + 1
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(0)
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.GetItemExpression)
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)

        E = 1 + m.x[t+m.P[t+1]]
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(E.nargs()-1)
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.GetItemExpression)
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)

    def test_clone(self):
        m = self.m
        t = IndexTemplate(m.I)

        E_base = m.x[t+m.P[t+1]] + m.P[1]
        E = E_base.clone()
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(0)
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertIsNot(e, E_base.arg(0))
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.GetItemExpression)
        self.assertIs(type(e.arg(1).arg(1)),
                      type(E_base.arg(0).arg(1).arg(1)))
        self.assertIsNot(e.arg(1).arg(1),
                         E_base.arg(0).arg(1).arg(1))
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)

        E_base = m.P[1] + m.x[t+m.P[t+1]]
        E = E_base.clone()
        self.assertTrue(isinstance(E, EXPR.SumExpressionBase))
        e = E.arg(1)
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertIsNot(e, E_base.arg(0))
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.GetItemExpression)
        self.assertIs(type(e.arg(1).arg(1)),
                      type(E_base.arg(1).arg(1).arg(1)))
        self.assertIsNot(e.arg(1).arg(1),
                         E_base.arg(1).arg(1).arg(1))
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)

        E_base = m.x[t+m.P[t+1]] + 1
        E = E_base.clone()
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(0)
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertIsNot(e, E_base.arg(0))
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.GetItemExpression)
        self.assertIs(type(e.arg(1).arg(1)),
                      type(E_base.arg(0).arg(1).arg(1)))
        self.assertIsNot(e.arg(1).arg(1),
                         E_base.arg(0).arg(1).arg(1))
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)

        E_base = 1 + m.x[t+m.P[t+1]]
        E = E_base.clone()
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(-1)
        self.assertIs(type(e), EXPR.GetItemExpression)
        self.assertIsNot(e, E_base.arg(0))
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.GetItemExpression)
        self.assertIs(type(e.arg(1).arg(1)),
                      type(E_base.arg(-1).arg(1).arg(1)))
        self.assertIsNot(e.arg(1).arg(1),
                         E_base.arg(-1).arg(1).arg(1))
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)


class TestTemplatizeRule(unittest.TestCase):
    def test_simple_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.x = Var(m.I)
        @m.Constraint(m.I)
        def c(m, i):
            return m.x[i] <= 0

        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, m.I)
        self.assertEqual(str(template), "x[_1]  <=  0.0")
        # Test that the RangeSet iterator was put back
        self.assertEqual(list(m.I), list(range(1,4)))
        # Evaluate the template
        indices[0].set_value(2)
        self.assertEqual(str(resolve_template(template)), 'x[2]  <=  0.0')

    def test_simple_rule_nonfinite_set(self):
        m = ConcreteModel()
        m.x = Var(Integers, dense=False)
        @m.Constraint(Integers)
        def c(m, i):
            return m.x[i] <= 0

        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, Integers)
        self.assertEqual(str(template), "x[_1]  <=  0.0")
        # Evaluate the template
        indices[0].set_value(2)
        self.assertEqual(str(resolve_template(template)), 'x[2]  <=  0.0')

    def test_simple_abstract_rule(self):
        m = AbstractModel()
        m.I = RangeSet(3)
        m.x = Var(m.I)
        @m.Constraint(m.I)
        def c(m, i):
            return m.x[i] <= 0

        # Note: the constraint can be abstract, but the Set/Var must
        # have been constructed (otherwise accessing the Set raises an
        # exception)

        with self.assertRaisesRegex(
                ValueError, ".*has not been constructed"):
            template, indices = templatize_constraint(m.c)

        m.I.construct()
        m.x.construct()
        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, m.I)
        self.assertEqual(str(template), "x[_1]  <=  0.0")

    def test_simple_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.x = Var(m.I,m.J)
        @m.Constraint(m.I)
        def c(m, i):
            return sum(m.x[i,j] for j in m.J) <= 0

        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 1)
        self.assertIs(indices[0]._set, m.I)
        self.assertEqual(
            template.to_string(verbose=True),
            "templatesum(getitem(x, _1, _2), iter(_2, J))  <=  0.0"
        )
        self.assertEqual(
            str(template),
            "SUM(x[_1,_2] for _2 in J)  <=  0.0"
        )
        # Evaluate the template
        indices[0].set_value(2)
        self.assertEqual(
            str(resolve_template(template)),
            'x[2,1] + x[2,2] + x[2,3]  <=  0.0'
        )

    def test_nested_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.K = Set(m.I, initialize={1:[10], 2:[10,20], 3:[10,20,30]})
        m.x = Var(m.I,m.J,[10,20,30])
        @m.Constraint()
        def c(m):
            return sum( sum(m.x[i,j,k] for k in m.K[i])
                        for j in m.J for i in m.I) <= 0

        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 0)
        self.assertEqual(
            template.to_string(verbose=True),
            "templatesum("
            "templatesum(getitem(x, _2, _1, _3), iter(_3, getitem(K, _2))), "
            "iter(_1, J), iter(_2, I))  <=  0.0"
        )
        self.assertEqual(
            str(template),
            "SUM(SUM(x[_2,_1,_3] for _3 in K[_2]) "
            "for _1 in J for _2 in I)  <=  0.0"
        )
        # Evaluate the template
        self.assertEqual(
            str(resolve_template(template)),
            'x[1,1,10] + '
            '(x[2,1,10] + x[2,1,20]) + '
            '(x[3,1,10] + x[3,1,20] + x[3,1,30]) + '
            '(x[1,2,10]) + '
            '(x[2,2,10] + x[2,2,20]) + '
            '(x[3,2,10] + x[3,2,20] + x[3,2,30]) + '
            '(x[1,3,10]) + '
            '(x[2,3,10] + x[2,3,20]) + '
            '(x[3,3,10] + x[3,3,20] + x[3,3,30])  <=  0.0'
        )

    def test_multidim_nested_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.JI = m.J*m.I
        m.K = Set(m.I, initialize={1:[10], 2:[10,20], 3:[10,20,30]})
        m.x = Var(m.I,m.J,[10,20,30])
        @m.Constraint()
        def c(m):
            return sum( sum(m.x[i,j,k] for k in m.K[i])
                        for j,i in m.JI) <= 0

        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 0)
        self.assertEqual(
            template.to_string(verbose=True),
            "templatesum("
            "templatesum(getitem(x, _2, _1, _3), iter(_3, getitem(K, _2))), "
            "iter(_1, _2, JI))  <=  0.0"
        )
        self.assertEqual(
            str(template),
            "SUM(SUM(x[_2,_1,_3] for _3 in K[_2]) "
            "for _1, _2 in JI)  <=  0.0"
        )
        # Evaluate the template
        self.assertEqual(
            str(resolve_template(template)),
            'x[1,1,10] + '
            '(x[2,1,10] + x[2,1,20]) + '
            '(x[3,1,10] + x[3,1,20] + x[3,1,30]) + '
            '(x[1,2,10]) + '
            '(x[2,2,10] + x[2,2,20]) + '
            '(x[3,2,10] + x[3,2,20] + x[3,2,30]) + '
            '(x[1,3,10]) + '
            '(x[2,3,10] + x[2,3,20]) + '
            '(x[3,3,10] + x[3,3,20] + x[3,3,30])  <=  0.0'
        )

    def test_multidim_nested_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.JI = m.J*m.I
        m.K = Set(m.I, initialize={1:[10], 2:[10,20], 3:[10,20,30]})
        m.x = Var(m.I,m.J,[10,20,30])
        @m.Constraint()
        def c(m):
            return sum( sum(m.x[i,j,k] for k in m.K[i])
                        for j,i in m.JI) <= 0

        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 0)
        self.assertEqual(
            template.to_string(verbose=True),
            "templatesum("
            "templatesum(getitem(x, _2, _1, _3), iter(_3, getitem(K, _2))), "
            "iter(_1, _2, JI))  <=  0.0"
        )
        self.assertEqual(
            str(template),
            "SUM(SUM(x[_2,_1,_3] for _3 in K[_2]) "
            "for _1, _2 in JI)  <=  0.0"
        )
        # Evaluate the template
        self.assertEqual(
            str(resolve_template(template)),
            'x[1,1,10] + '
            '(x[2,1,10] + x[2,1,20]) + '
            '(x[3,1,10] + x[3,1,20] + x[3,1,30]) + '
            '(x[1,2,10]) + '
            '(x[2,2,10] + x[2,2,20]) + '
            '(x[3,2,10] + x[3,2,20] + x[3,2,30]) + '
            '(x[1,3,10]) + '
            '(x[2,3,10] + x[2,3,20]) + '
            '(x[3,3,10] + x[3,3,20] + x[3,3,30])  <=  0.0'
        )

    def test_multidim_nested_getattr_sum_rule(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.J = RangeSet(3)
        m.JI = m.J*m.I
        m.K = Set(m.I, initialize={1:[10], 2:[10,20], 3:[10,20,30]})
        m.x = Var(m.I,m.J,[10,20,30])
        @m.Block(m.I)
        def b(b, i):
            b.K = RangeSet(10, 10*i, 10)
        @m.Constraint()
        def c(m):
            return sum( sum(m.x[i,j,k] for k in m.b[i].K)
                        for j,i in m.JI) <= 0

        template, indices = templatize_constraint(m.c)
        self.assertEqual(len(indices), 0)
        self.assertEqual(
            template.to_string(verbose=True),
            "templatesum("
            "templatesum(getitem(x, _2, _1, _3), "
            "iter(_3, getattr(getitem(b, _2), 'K'))), "
            "iter(_1, _2, JI))  <=  0.0"
        )
        self.assertEqual(
            str(template),
            "SUM(SUM(x[_2,_1,_3] for _3 in b[_2].K) "
            "for _1, _2 in JI)  <=  0.0"
        )
        # Evaluate the template
        self.assertEqual(
            str(resolve_template(template)),
            'x[1,1,10] + '
            '(x[2,1,10] + x[2,1,20]) + '
            '(x[3,1,10] + x[3,1,20] + x[3,1,30]) + '
            '(x[1,2,10]) + '
            '(x[2,2,10] + x[2,2,20]) + '
            '(x[3,2,10] + x[3,2,20] + x[3,2,30]) + '
            '(x[1,3,10]) + '
            '(x[2,3,10] + x[2,3,20]) + '
            '(x[3,3,10] + x[3,3,20] + x[3,3,30])  <=  0.0'
        )


class TestTemplateSubstitution(unittest.TestCase):

    def setUp(self):
        self.m = m = ConcreteModel()
        m.TRAY = Set(initialize=range(5))
        m.TIME = Set(bounds=(0,10), initialize=range(10))
        m.y = Var(initialize=1)
        m.x = Var(m.TIME, m.TRAY, initialize=lambda _m,i,j: i)
        m.dxdt = Var(m.TIME, m.TRAY, initialize=lambda _m,i,j: 2*i)

    def test_simple_substitute_param(self):
        def diffeq(m,t, i):
            return m.dxdt[t, i] == t*m.x[t, i-1]**2 + m.y**2 + \
                m.x[t, i+1] + m.x[t, i-1]

        m = self.m
        t = IndexTemplate(m.TIME)
        e = diffeq(m, t, 2)

        self.assertTrue( isinstance(e, EXPR.ExpressionBase) )

        _map = {}
        E = substitute_template_expression(
            e, substitute_getitem_with_param, _map )
        self.assertIsNot(e,E)

        self.assertEqual( len(_map), 3 )

        idx1 = _GetItemIndexer( m.x[t,1] )
        self.assertEqual( idx1.nargs(), 2 )
        self.assertIs( idx1.base, m.x )
        self.assertIs( idx1.arg(0), t )
        self.assertEqual( idx1.arg(1), 1 )
        self.assertIn( idx1, _map )

        idx2 = _GetItemIndexer( m.dxdt[t,2] )
        self.assertEqual( idx2.nargs(), 2 )
        self.assertIs( idx2.base, m.dxdt )
        self.assertIs( idx2.arg(0), t )
        self.assertEqual( idx2.arg(1), 2 )
        self.assertIn( idx2, _map )

        idx3 = _GetItemIndexer( m.x[t,3] )
        self.assertEqual( idx3.nargs(), 2 )
        self.assertIs( idx3.base, m.x )
        self.assertIs( idx3.arg(0), t )
        self.assertEqual( idx3.arg(1), 3 )
        self.assertIn( idx3, _map )

        self.assertFalse( idx1 == idx2 )
        self.assertFalse( idx1 == idx3 )
        self.assertFalse( idx2 == idx3 )

        idx4 = _GetItemIndexer( m.x[t,2] )
        self.assertNotIn( idx4, _map )

        t.set_value(5)
        self.assertEqual((e.arg(0)(), e.arg(1)()), (10,136))

        self.assertEqual(
            str(E),
            'dxdt[{TIME},2]  ==  {TIME}*x[{TIME},1]**2 + y**2 + x[{TIME},3] + x[{TIME},1]' )

        _map[idx1].set_value( value(m.x[value(t), 1]) )
        _map[idx2].set_value( value(m.dxdt[value(t), 2]) )
        _map[idx3].set_value( value(m.x[value(t), 3]) )
        self.assertEqual((E.arg(0)(), E.arg(1)()), (10,136))

        _map[idx1].set_value( 12 )
        _map[idx2].set_value( 34 )
        self.assertEqual((E.arg(0)(), E.arg(1)()), (34,738))


    def test_simple_substitute_index(self):
        def diffeq(m,t, i):
            return m.dxdt[t, i] == t * m.x[t, i] ** 2 + m.y**2

        m = self.m
        t = IndexTemplate(m.TIME)
        e = diffeq(m,t, 2)
        t.set_value(5)

        self.assertTrue( isinstance(e, EXPR.ExpressionBase) )
        self.assertEqual((e.arg(0)(), e.arg(1)()), (10,126))

        E = substitute_template_expression(e, substitute_template_with_value)
        self.assertIsNot(e,E)

        self.assertEqual(
            str(E),
            'dxdt[5,2]  ==  5.0*x[5,2]**2 + y**2' )

if __name__ == "__main__":
    unittest.main()

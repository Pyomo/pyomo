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

# TODO: How do we defer so this doesn't mess up everything?
import docplex.cp.model as cp

import pyomo.common.unittest as unittest

from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
    AlwaysIn, Step, Pulse
)
from pyomo.contrib.cp.repn.docplex_writer import LogicalToDoCplex

from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import equivalent, exactly, atleast, atmost
from pyomo.core.expr.relational_expr import NotEqualExpression

from pyomo.environ import (
    ConcreteModel, RangeSet, Var, BooleanVar, Constraint, LogicalConstraint,
    PositiveIntegers, Binary, NonNegativeIntegers, NegativeIntegers,
    NonPositiveIntegers, Integers, inequality, Expression
)

from pytest import set_trace

class CommonTest(unittest.TestCase):
    def get_visitor(self):
        docplex_model= cp.CpoModel()
        return LogicalToDoCplex(docplex_model, symbolic_solver_labels=True)

    def get_model(self):
        m = ConcreteModel()
        m.I = RangeSet(10)
        m.a = Var(m.I)
        m.x = Var(within=PositiveIntegers, bounds=(6,8))
        m.i = IntervalVar(optional=True)
        m.i2 = IntervalVar([1, 2], optional=False)

        m.b = BooleanVar()
        m.b2 = BooleanVar(['a', 'b', 'c'])

        return m

class TestCPExpressionWalker_AlgebraicExpressions(CommonTest):
    def test_write_addition(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x + m.i.start_time + m.i2[2].length <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i2[2].length), visitor.var_map)

        cpx_x = visitor.var_map[id(m.x)]
        cpx_i = visitor.var_map[id(m.i)]
        cpx_i2 = visitor.var_map[id(m.i2[2])]
        self.assertTrue(expr[1].equals(cpx_x + cp.start_of(cpx_i) +
                                       cp.length_of(cpx_i2)))

    def test_write_subtraction(self):
        m = self.get_model()
        m.a.domain = Binary
        m.c = Constraint(expr=m.x - m.a[1] <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a[1]), visitor.var_map)

        x = visitor.var_map[id(m.x)]
        a1 = visitor.var_map[id(m.a[1])]

        self.assertTrue(expr[1].equals(x + (-1 * a1)))

    def test_write_product(self):
        m = self.get_model()
        m.a.domain = PositiveIntegers
        m.c = Constraint(expr=m.x*(m.a[1] + 1) <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a[1]), visitor.var_map)

        x = visitor.var_map[id(m.x)]
        a1 = visitor.var_map[id(m.a[1])]

        self.assertTrue(expr[1].equals(x*(a1 + 1)))

    def test_write_floating_point_division(self):
        m = self.get_model()
        m.a.domain = NonNegativeIntegers
        m.c = Constraint(expr=m.x/(m.a[1] + 1) <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a[1]), visitor.var_map)

        x = visitor.var_map[id(m.x)]
        a1 = visitor.var_map[id(m.a[1])]

        self.assertTrue(expr[1].equals(x/(a1 + 1)))

    def test_write_power_expression(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x**2 <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        cpx_x = visitor.var_map[id(m.x)]
        # .equals checks the equality of two expressions in docplex.
        self.assertTrue(expr[1].equals(cpx_x**2))

    def test_write_absolute_value_expression(self):
        m = self.get_model()
        m.a.domain = NegativeIntegers
        m.c = Constraint(expr=abs(m.a[1]) + 1 <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.a[1]), visitor.var_map)

        a1 = visitor.var_map[id(m.a[1])]

        self.assertTrue(expr[1].equals(cp.abs(a1) + 1))

    def test_write_min_expression(self):
        m = self.get_model()
        m.a.domain = NonPositiveIntegers
        m.c = Constraint(expr=MinExpression([m.a[i] for i in m.I]) >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]

        self.assertTrue(expr[1].equals(cp.min(a[i] for i in m.I)))

    def test_write_max_expression(self):
        m = self.get_model()
        m.a.domain = NonPositiveIntegers
        m.c = Constraint(expr=MaxExpression([m.a[i] for i in m.I]) >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]

        self.assertTrue(expr[1].equals(cp.max(a[i] for i in m.I)))

    def test_indirection_single_index(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = Constraint(expr=m.a[m.x] >= 3.5)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        a = []
        # only need indices 6, 7, and 8 from a, since that's what x is capable
        # of selecting.
        for idx in [6, 7, 8]:
            v = m.a[idx]
            self.assertIn(id(v), visitor.var_map)
            a.append(visitor.var_map[id(v)])
        # since x is between 6 and 8, we subtract 6 from it for it to be the
        # right index
        self.assertTrue(expr[1].equals(cp.element(a, 0 + 1 *(x - 6) // 1)))


class TestCPExpressionWalker_LogicalExpressions(CommonTest):
    def test_write_logical_and(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.b.land(m.b2['b']))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.b2['b']), visitor.var_map)

        b = visitor.var_map[id(m.b)]
        b2b = visitor.var_map[id(m.b2['b'])]

        self.assertTrue(expr[1].equals(cp.logical_and(b, b2b)))

    def test_write_logical_or(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.b.lor(m.i.is_present))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        i = visitor.var_map[id(m.i)]

        self.assertTrue(expr[1].equals(cp.logical_or(b, cp.presence_of(i))))

    def test_write_xor(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.b.xor(m.i2[2].start_time >= 5))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        i22 = visitor.var_map[id(m.i2[2])]

        # [ESJ 9/22/22]: This isn't the greatest test because there's no direct
        # translation so how we choose to represent this could change.
        self.assertTrue(expr[1].equals(
            cp.count([b, cp.less_or_equal(5, cp.start_of(i22))], True) == 1))

    def test_write_logical_not(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=~m.b2['a'])
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b2['a']), visitor.var_map)
        b2a = visitor.var_map[id(m.b2['a'])]

        self.assertTrue(expr[1].equals(cp.logical_not(b2a)))

    def test_equivalence(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=equivalent(~m.b2['a'], m.b))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.b2['a']), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        b2a = visitor.var_map[id(m.b2['a'])]

        self.assertTrue(expr[1].equals(cp.equal(cp.logical_not(b2a), b)))

    def test_implication(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.b2['a'].implies(~m.b))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.b2['a']), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        b2a = visitor.var_map[id(m.b2['a'])]

        self.assertTrue(expr[1].equals(cp.if_then(b2a, cp.logical_not(b))))

    def test_equality(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=m.b.implies(m.a[3] == 4))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.a[3]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        a3 = visitor.var_map[id(m.a[3])]

        self.assertTrue(expr[1].equals(cp.if_then(b, cp.equal(a3, 4))))

    def test_inequality(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=m.b.implies(m.a[3] >= m.a[4]))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.a[3]), visitor.var_map)
        self.assertIn(id(m.a[4]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        a3 = visitor.var_map[id(m.a[3])]
        a4 = visitor.var_map[id(m.a[4])]

        self.assertTrue(expr[1].equals(cp.if_then(b, cp.less_or_equal(a4, a3))))

    def test_ranged_inequality(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = Constraint(expr=inequality(3, m.a[2], 5))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.a[2]), visitor.var_map)
        a2 = visitor.var_map[id(m.a[2])]

        self.assertTrue(expr[1].equals(cp.range(a2, 3, 5)))

    def test_not_equal(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=m.b.implies(
            NotEqualExpression([m.a[3], m.a[4]])))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.a[3]), visitor.var_map)
        self.assertIn(id(m.a[4]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        a3 = visitor.var_map[id(m.a[3])]
        a4 = visitor.var_map[id(m.a[4])]

        self.assertTrue(expr[1].equals(cp.if_then(b, a3 !=  a4)))

    def test_exactly_expression(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=exactly(3, [m.a[i] == 4 for i in m.I]))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]

        self.assertTrue(expr[1].equals(
            cp.equal(cp.count([a[i] == 4 for i in m.I], True), 3)))

    def test_atleast_expression(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=atleast(3, [m.a[i] == 4 for i in m.I]))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]

        self.assertTrue(expr[1].equals(
            cp.greater_or_equal(cp.count([a[i] == 4 for i in m.I], True), 3)))

    def test_atmost_expression(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=atmost(3, [m.a[i] == 4 for i in m.I]))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]

        self.assertTrue(expr[1].equals(
            cp.less_or_equal(cp.count([a[i] == 4 for i in m.I], True), 3)))

    def test_interval_var_is_present(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=m.i.is_present.implies(m.a[1] == 5))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.a[1]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        a1 = visitor.var_map[id(m.a[1])]
        i = visitor.var_map[id(m.i)]

        self.assertTrue(expr[1].equals(
            cp.if_then(cp.presence_of(i), a1 == 5)))

    def test_interval_var_is_present_indirection(self):
        m = self.get_model()
        m.a.domain = Integers
        m.y = Var(domain=Integers, bounds=[1,2])

        m.c = LogicalConstraint(expr=m.i2[m.y].is_present.implies(m.a[1] >= 7))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.a[1]), visitor.var_map)
        a1 = visitor.var_map[id(m.a[1])]

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]

        self.assertTrue(expr[1].equals(
            cp.if_then(cp.element([cp.presence_of(i21), cp.presence_of(i22)],
                                  0 + 1 * (y - 1) // 1) == True,
                       cp.less_or_equal(7, a1))))

    def test_is_present_indirection_and_length(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1,2])

        m.c = LogicalConstraint(
            expr=m.i2[m.y].is_present.land(m.i2[m.y].length >= 7))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]

        self.assertTrue(expr[1].equals(
            cp.logical_and(
                cp.element([cp.presence_of(i21), cp.presence_of(i22)],
                           0 + 1 * (y - 1) // 1) == True,
                cp.less_or_equal(7, cp.element([cp.length_of(i21),
                                                cp.length_of(i22)],
                                               0 + 1*(y - 1) // 1)))))

class TestCPExpressionWalker_PrecedenceExpressions(CommonTest):
    def test_start_before_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.start_time.before(m.i2[1].start_time))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.start_before_start(i, i21, 0)))

    def test_start_before_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.start_time.before(m.i2[1].end_time,
                                                           delay=3))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.start_before_end(i, i21, 3)))

    def test_end_before_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.before(m.i2[1].start_time,
                                                         delay=-2))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.end_before_start(i, i21, -2)))

    def test_end_before_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.before(m.i2[1].end_time,
                                                         delay=6))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.end_before_end(i, i21, 6)))

    def test_start_at_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.start_time.at(m.i2[1].start_time))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.start_at_start(i, i21, 0)))

    def test_start_at_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.start_time.at(m.i2[1].end_time,
                                                           delay=3))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.start_at_end(i, i21, 3)))

    def test_end_at_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.at(m.i2[1].start_time,
                                                         delay=-2))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.end_at_start(i, i21, -2)))

    def test_end_at_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.at(m.i2[1].end_time,
                                                         delay=6))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.end_at_end(i, i21, 6)))

    ##
    # Tests for precedence constraints with indirection
    ##

    def test_indirection_before_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1,2])
        m.c = LogicalConstraint(expr=m.i2[m.y].start_time.before(m.i.end_time,
                                                                 delay=3))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]

        self.assertTrue(expr[1].equals(
            cp.element([cp.start_of(i21), cp.start_of(i22)],
                       0 + 1 * (y-1) // 1) + 3 <= cp.end_of(i)))

    def test_indirection_after_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1,2])
        m.c = LogicalConstraint(expr=m.i2[m.y].start_time.after(m.i.end_time,
                                                                delay=-2))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]

        self.assertTrue(expr[1].equals(
            cp.end_of(i) + (-2) <=
            cp.element([cp.start_of(i21), cp.start_of(i22)],
                       0 + 1 * (y-1) // 1)))

    def test_indirection_at_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1,2])
        m.c = LogicalConstraint(expr=m.i2[m.y].start_time.at(m.i.end_time,
                                                             delay=4))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]

        self.assertTrue(expr[1].equals(
            cp.element([cp.start_of(i21), cp.start_of(i22)],
                       0 + 1 * (y-1) // 1) == cp.end_of(i) + 4))

    def test_before_indirection_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1,2])
        m.c = LogicalConstraint(expr=m.i.start_time.before(m.i2[m.y].end_time,
                                                           delay=-4))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]

        self.assertTrue(expr[1].equals(
            cp.start_of(i) <= cp.element([cp.end_of(i21), cp.end_of(i22)],
                                         0 + 1 * (y-1) // 1) + (- 4)))

    def test_after_indirection_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1,2])
        m.c = LogicalConstraint(expr=m.i.start_time.after(m.i2[m.y].end_time))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]

        self.assertTrue(expr[1].equals(
            cp.element([cp.end_of(i21), cp.end_of(i22)],
                       0 + 1 * (y-1) // 1) <= cp.start_of(i) + 0))

    def test_at_indirection_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1,2])
        m.c = LogicalConstraint(expr=m.i.start_time.at(m.i2[m.y].end_time,
                                                       delay=-6))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]

        self.assertTrue(expr[1].equals(
            cp.start_of(i) == cp.element([cp.end_of(i21), cp.end_of(i22)],
                                         0 + 1 * (y-1) // 1) + (- 6)))

    def test_double_indirection_before_constraint(self):
        m = self.get_model()
        # add interval var x can index
        m.i3 = IntervalVar([(1,3), (1,4), (1,5)], length=4, optional=True)
        m.y = Var(domain=Integers, bounds=[1,2])
        m.c = LogicalConstraint(
            expr=m.i3[1, m.x - 3].start_time.before(m.i2[m.y].end_time))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i3[1,3]), visitor.var_map)
        self.assertIn(id(m.i3[1,4]), visitor.var_map)
        self.assertIn(id(m.i3[1,5]), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        x = visitor.var_map[id(m.x)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i33 = visitor.var_map[id(m.i3[1,3])]
        i34 = visitor.var_map[id(m.i3[1,4])]
        i35 = visitor.var_map[id(m.i3[1,5])]

        self.assertTrue(expr[1].equals(
            cp.element([cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)],
                       0 + 1 * (x + (-3) - 3) // 1) <=
            cp.element([cp.end_of(i21), cp.end_of(i22)],
                       0 + 1 * (y - 1) // 1)))

    def test_double_indirection_after_constraint(self):
        m = self.get_model()
        # add interval var x can index
        m.i3 = IntervalVar([(1,3), (1,4), (1,5)], length=4, optional=True)
        m.y = Var(domain=Integers, bounds=[1,2])
        m.c = LogicalConstraint(
            expr=m.i3[1, m.x - 3].start_time.after(m.i2[m.y].end_time))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i3[1,3]), visitor.var_map)
        self.assertIn(id(m.i3[1,4]), visitor.var_map)
        self.assertIn(id(m.i3[1,5]), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        x = visitor.var_map[id(m.x)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i33 = visitor.var_map[id(m.i3[1,3])]
        i34 = visitor.var_map[id(m.i3[1,4])]
        i35 = visitor.var_map[id(m.i3[1,5])]

        print(expr[1])
        self.assertTrue(expr[1].equals(
            cp.element([cp.end_of(i21), cp.end_of(i22)],
                       0 + 1 * (y - 1) // 1) <=
            cp.element([cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)],
                       0 + 1 * (x + (-3) - 3) // 1)))

    def test_double_indirection_at_constraint(self):
        m = self.get_model()
        # add interval var x can index
        m.i3 = IntervalVar([(1,3), (1,4), (1,5)], length=4, optional=True)
        m.y = Var(domain=Integers, bounds=[1,2])
        m.c = LogicalConstraint(
            expr=m.i3[1, m.x - 3].start_time.at(m.i2[m.y].end_time))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i3[1,3]), visitor.var_map)
        self.assertIn(id(m.i3[1,4]), visitor.var_map)
        self.assertIn(id(m.i3[1,5]), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        x = visitor.var_map[id(m.x)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i33 = visitor.var_map[id(m.i3[1,3])]
        i34 = visitor.var_map[id(m.i3[1,4])]
        i35 = visitor.var_map[id(m.i3[1,5])]

        self.assertTrue(expr[1].equals(
            cp.element([cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)],
                       0 + 1 * (x + (-3) - 3) // 1) ==
            cp.element([cp.end_of(i21), cp.end_of(i22)],
                       0 + 1 * (y - 1) // 1)))


class TestCPExpressionWalker_CumulFuncExpressions(CommonTest):
    def test_always_in(self):
        m = self.get_model()
        f = Pulse(m.i, height=3) + Step(m.i2[1].start_time, height=2) - \
            Step(m.i2[2].end_time, height=-1)
        m.c = LogicalConstraint(expr=f.within((0, 3), (0, 10)))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]

        self.assertTrue(expr[1].equals(cp.always_in(cp.pulse(i, 3) +
                                                    cp.step_at_start(i21, 2) -
                                                    cp.step_at_end(i22, -1), 0,
                                                    3, 0, 10)))


class TestCPExpressionWalker_NamedExpressions(CommonTest):
    def test_named_expression(self):
        m = self.get_model()
        m.e = Expression(expr=m.x**2 + 7)
        m.c = Constraint(expr=m.e <= 32)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]

        self.assertTrue(expr[1].equals(x**2 + 7))

    def test_repeated_named_expression(self):
        m = self.get_model()
        m.e = Expression(expr=m.x**2 + 7)
        m.c = Constraint(expr=m.e - 8*m.e <= 32)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]

        self.assertTrue(expr[1].equals(x**2 + 7 + (-1) *(8*(x**2 + 7))))


class TestCPExpressionWalker_Vars(CommonTest):
    def test_complain_about_non_integer_vars(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.is_present.implies(m.a[1] == 5))

        visitor = self.get_visitor()
        with self.assertRaisesRegexp(
                ValueError,
                "The LogicalToDoCplex writer can only support integer- or "
                "Boolean-valued variables. Cannot write Var a\[1\] with domain "
                "Reals"):
            expr = visitor.walk_expression((m.c.expr, m.c, 0))

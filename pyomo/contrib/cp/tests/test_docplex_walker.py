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

import pyomo.common.unittest as unittest

from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
    AlwaysIn,
    Step,
    Pulse,
)
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex

from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import equivalent, exactly, atleast, atmost
from pyomo.core.expr.relational_expr import NotEqualExpression

from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Var,
    BooleanVar,
    Constraint,
    LogicalConstraint,
    PositiveIntegers,
    Binary,
    NonNegativeIntegers,
    NegativeIntegers,
    NonPositiveIntegers,
    Integers,
    inequality,
    Expression,
    Reals,
    Set,
    Param,
)

try:
    import docplex.cp.model as cp

    docplex_available = True
except:
    docplex_available = False


class CommonTest(unittest.TestCase):
    def get_visitor(self):
        docplex_model = cp.CpoModel()
        return LogicalToDoCplex(docplex_model, symbolic_solver_labels=True)

    def get_model(self):
        m = ConcreteModel()
        m.I = RangeSet(10)
        m.a = Var(m.I)
        m.x = Var(within=PositiveIntegers, bounds=(6, 8))
        m.i = IntervalVar(optional=True)
        m.i2 = IntervalVar([1, 2], optional=False)

        m.b = BooleanVar()
        m.b2 = BooleanVar(['a', 'b', 'c'])

        return m


@unittest.skipIf(not docplex_available, "docplex is not available")
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
        self.assertTrue(
            expr[1].equals(cpx_x + cp.start_of(cpx_i) + cp.length_of(cpx_i2))
        )

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
        m.c = Constraint(expr=m.x * (m.a[1] + 1) <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a[1]), visitor.var_map)

        x = visitor.var_map[id(m.x)]
        a1 = visitor.var_map[id(m.a[1])]

        self.assertTrue(expr[1].equals(x * (a1 + 1)))

    def test_write_floating_point_division(self):
        m = self.get_model()
        m.a.domain = NonNegativeIntegers
        m.c = Constraint(expr=m.x / (m.a[1] + 1) <= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a[1]), visitor.var_map)

        x = visitor.var_map[id(m.x)]
        a1 = visitor.var_map[id(m.a[1])]

        self.assertTrue(expr[1].equals(x / (a1 + 1)))

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

    def test_expression_with_mutable_param(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers, bounds=(2, 3))
        m.p = Param(initialize=4, mutable=True)
        e = m.p * m.x

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]

        self.assertTrue(expr[1].equals(4 * x))


@unittest.skipIf(not docplex_available, "docplex is not available")
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
        self.assertTrue(
            expr[1].equals(cp.count([b, cp.less_or_equal(5, cp.start_of(i22))], 1) == 1)
        )

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
        m.c = LogicalConstraint(expr=m.b.implies(NotEqualExpression([m.a[3], m.a[4]])))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.a[3]), visitor.var_map)
        self.assertIn(id(m.a[4]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        a3 = visitor.var_map[id(m.a[3])]
        a4 = visitor.var_map[id(m.a[4])]

        self.assertTrue(expr[1].equals(cp.if_then(b, a3 != a4)))

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

        self.assertTrue(
            expr[1].equals(cp.equal(cp.count([a[i] == 4 for i in m.I], 1), 3))
        )

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

        self.assertTrue(
            expr[1].equals(
                cp.greater_or_equal(cp.count([a[i] == 4 for i in m.I], 1), 3)
            )
        )

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

        self.assertTrue(
            expr[1].equals(cp.less_or_equal(cp.count([a[i] == 4 for i in m.I], 1), 3))
        )

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

        self.assertTrue(expr[1].equals(cp.if_then(cp.presence_of(i), a1 == 5)))

    def test_interval_var_is_present_indirection(self):
        m = self.get_model()
        m.a.domain = Integers
        m.y = Var(domain=Integers, bounds=[1, 2])

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

        self.assertTrue(
            expr[1].equals(
                cp.if_then(
                    cp.element(
                        [cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1
                    )
                    == True,
                    cp.less_or_equal(7, a1),
                )
            )
        )

    def test_is_present_indirection_and_length(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])

        m.c = LogicalConstraint(expr=m.i2[m.y].is_present.land(m.i2[m.y].length >= 7))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]

        self.assertTrue(
            expr[1].equals(
                cp.logical_and(
                    cp.element(
                        [cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1
                    )
                    == True,
                    cp.less_or_equal(
                        7,
                        cp.element(
                            [cp.length_of(i21), cp.length_of(i22)], 0 + 1 * (y - 1) // 1
                        ),
                    ),
                )
            )
        )

    def test_handle_getattr_lor(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=(1, 2))

        e = m.i2[m.y].is_present.lor(~m.b)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.b), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        b = visitor.var_map[id(m.b)]

        self.assertTrue(
            expr[1].equals(
                cp.logical_or(
                    cp.element(
                        [cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1
                    )
                    == True,
                    cp.logical_not(b),
                )
            )
        )

    def test_handle_getattr_xor(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=(1, 2))

        e = m.i2[m.y].is_present.xor(m.b)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.b), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        b = visitor.var_map[id(m.b)]

        self.assertTrue(
            expr[1].equals(
                cp.equal(
                    cp.count(
                        [
                            cp.element(
                                [cp.presence_of(i21), cp.presence_of(i22)],
                                0 + 1 * (y - 1) // 1,
                            )
                            == True,
                            b,
                        ],
                        1,
                    ),
                    1,
                )
            )
        )

    def test_handle_getattr_equivalent_to(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=(1, 2))

        e = m.i2[m.y].is_present.equivalent_to(~m.b)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.b), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        b = visitor.var_map[id(m.b)]

        self.assertTrue(
            expr[1].equals(
                cp.equal(
                    cp.element(
                        [cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1
                    )
                    == True,
                    cp.logical_not(b),
                )
            )
        )

    def test_logical_or_on_indirection(self):
        m = ConcreteModel()
        m.b = BooleanVar([2, 3, 4, 5])
        m.x = Var(domain=Integers, bounds=(3, 5))

        e = m.b[m.x].lor(m.x == 5)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.b[3]), visitor.var_map)
        self.assertIn(id(m.b[4]), visitor.var_map)
        self.assertIn(id(m.b[5]), visitor.var_map)

        x = visitor.var_map[id(m.x)]
        b3 = visitor.var_map[id(m.b[3])]
        b4 = visitor.var_map[id(m.b[4])]
        b5 = visitor.var_map[id(m.b[5])]

        self.assertTrue(
            expr[1].equals(
                cp.logical_or(
                    cp.element([b3, b4, b5], 0 + 1 * (x - 3) // 1) == True,
                    cp.equal(x, 5),
                )
            )
        )

    def test_logical_xor_on_indirection(self):
        m = ConcreteModel()
        m.b = BooleanVar([2, 3, 4, 5])
        m.b[4].fix(False)
        m.x = Var(domain=Integers, bounds=(3, 5))

        e = m.b[m.x].xor(m.x == 5)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.b[3]), visitor.var_map)
        self.assertIn(id(m.b[5]), visitor.var_map)

        x = visitor.var_map[id(m.x)]
        b3 = visitor.var_map[id(m.b[3])]
        b5 = visitor.var_map[id(m.b[5])]

        self.assertTrue(
            expr[1].equals(
                cp.equal(
                    cp.count(
                        [
                            cp.element([b3, False, b5], 0 + 1 * (x - 3) // 1) == True,
                            cp.equal(x, 5),
                        ],
                        1,
                    ),
                    1,
                )
            )
        )

    def test_using_precedence_expr_as_boolean_expr(self):
        m = self.get_model()
        e = m.b.implies(m.i2[2].start_time.before(m.i2[1].start_time))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)

        b = visitor.var_map[id(m.b)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]

        self.assertTrue(
            expr[1].equals(cp.if_then(b, cp.start_of(i22) + 0 <= cp.start_of(i21)))
        )

    def test_using_precedence_expr_as_boolean_expr_positive_delay(self):
        m = self.get_model()
        e = m.b.implies(m.i2[2].start_time.before(m.i2[1].start_time, delay=4))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)

        b = visitor.var_map[id(m.b)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]

        self.assertTrue(
            expr[1].equals(cp.if_then(b, cp.start_of(i22) + 4 <= cp.start_of(i21)))
        )

    def test_using_precedence_expr_as_boolean_expr_negative_delay(self):
        m = self.get_model()
        e = m.b.implies(m.i2[2].start_time.at(m.i2[1].start_time, delay=-3))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)

        b = visitor.var_map[id(m.b)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]

        self.assertTrue(
            expr[1].equals(cp.if_then(b, cp.start_of(i22) + (-3) == cp.start_of(i21)))
        )


@unittest.skipIf(not docplex_available, "docplex is not available")
class TestCPExpressionWalker_IntervalVars(CommonTest):
    def test_interval_var_fixed_presences_correct(self):
        m = self.get_model()

        m.silly = LogicalConstraint(expr=m.i.is_present)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.silly.expr, m.silly, 0))
        self.assertIn(id(m.i), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        # Check that docplex knows it's optional
        self.assertTrue(i.is_optional())

        # Now fix it to absent
        m.i.is_present.fix(False)
        m.c = LogicalConstraint(expr=m.i.is_present.lor(m.i2[1].start_time == 2))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i2[1]), visitor.var_map)
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertIn(id(m.i), visitor.var_map)
        i = visitor.var_map[id(m.i)]

        # Check that we passed on the presence info to docplex
        self.assertTrue(i.is_absent())
        self.assertTrue(i21.is_present())
        # Not testing the expression here because sometime we might optimize out
        # the presence_of call for fixed absent vars, but for now I haven't.

    def test_interval_var_fixed_length(self):
        m = ConcreteModel()
        m.i = IntervalVar(start=(2, 7), end=(6, 11), optional=True)
        m.i.length.fix(4)
        m.silly = LogicalConstraint(expr=m.i.is_present)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.silly.expr, m.silly, 0))

        self.assertIn(id(m.i), visitor.var_map)
        i = visitor.var_map[id(m.i)]

        self.assertTrue(i.is_optional())
        self.assertEqual(i.get_length(), (4, 4))
        self.assertEqual(i.get_start(), (2, 7))
        self.assertEqual(i.get_end(), (6, 11))

    def test_interval_var_fixed_start_and_end(self):
        m = ConcreteModel()
        m.i = IntervalVar(start=(3, 7), end=(6, 10))
        m.i.start_time.fix(3)
        m.i.end_time.fix(6)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.i, m.i, 0))

        self.assertIn(id(m.i), visitor.var_map)
        i = visitor.var_map[id(m.i)]

        self.assertFalse(i.is_optional())
        self.assertEqual(i.get_start(), (3, 3))
        self.assertEqual(i.get_end(), (6, 6))


@unittest.skipIf(not docplex_available, "docplex is not available")
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
        m.c = LogicalConstraint(expr=m.i.start_time.before(m.i2[1].end_time, delay=3))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.start_before_end(i, i21, 3)))

    def test_end_before_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.before(m.i2[1].start_time, delay=-2))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.end_before_start(i, i21, -2)))

    def test_end_before_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.before(m.i2[1].end_time, delay=6))
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
        m.c = LogicalConstraint(expr=m.i.start_time.at(m.i2[1].end_time, delay=3))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.start_at_end(i, i21, 3)))

    def test_end_at_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.at(m.i2[1].start_time, delay=-2))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]

        self.assertTrue(expr[1].equals(cp.end_at_start(i, i21, -2)))

    def test_end_at_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.at(m.i2[1].end_time, delay=6))
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
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i2[m.y].start_time.before(m.i.end_time, delay=3))

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

        self.assertTrue(
            expr[1].equals(
                cp.element([cp.start_of(i21), cp.start_of(i22)], 0 + 1 * (y - 1) // 1)
                + 3
                <= cp.end_of(i)
            )
        )

    def test_indirection_after_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i2[m.y].start_time.after(m.i.end_time, delay=-2))

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

        self.assertTrue(
            expr[1].equals(
                cp.end_of(i) + (-2)
                <= cp.element(
                    [cp.start_of(i21), cp.start_of(i22)], 0 + 1 * (y - 1) // 1
                )
            )
        )

    def test_indirection_at_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i2[m.y].start_time.at(m.i.end_time, delay=4))

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

        self.assertTrue(
            expr[1].equals(
                cp.element([cp.start_of(i21), cp.start_of(i22)], 0 + 1 * (y - 1) // 1)
                == cp.end_of(i) + 4
            )
        )

    def test_before_indirection_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(
            expr=m.i.start_time.before(m.i2[m.y].end_time, delay=-4)
        )

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

        self.assertTrue(
            expr[1].equals(
                cp.start_of(i) + (-4)
                <= cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)
            )
        )

    def test_after_indirection_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
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

        self.assertTrue(
            expr[1].equals(
                cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1) + 0
                <= cp.start_of(i)
            )
        )

    def test_at_indirection_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i.start_time.at(m.i2[m.y].end_time, delay=-6))

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

        self.assertTrue(
            expr[1].equals(
                cp.start_of(i) + (-6)
                == cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)
            )
        )

    def test_double_indirection_before_constraint(self):
        m = self.get_model()
        # add interval var x can index
        m.i3 = IntervalVar([(1, 3), (1, 4), (1, 5)], length=4, optional=True)
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(
            expr=m.i3[1, m.x - 3].start_time.before(m.i2[m.y].end_time)
        )

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i3[1, 3]), visitor.var_map)
        self.assertIn(id(m.i3[1, 4]), visitor.var_map)
        self.assertIn(id(m.i3[1, 5]), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        x = visitor.var_map[id(m.x)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i33 = visitor.var_map[id(m.i3[1, 3])]
        i34 = visitor.var_map[id(m.i3[1, 4])]
        i35 = visitor.var_map[id(m.i3[1, 5])]

        self.assertTrue(
            expr[1].equals(
                cp.element(
                    [cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)],
                    0 + 1 * (x + (-3) - 3) // 1,
                )
                <= cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)
            )
        )

    def test_double_indirection_after_constraint(self):
        m = self.get_model()
        # add interval var x can index
        m.i3 = IntervalVar([(1, 3), (1, 4), (1, 5)], length=4, optional=True)
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(
            expr=m.i3[1, m.x - 3].start_time.after(m.i2[m.y].end_time)
        )

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i3[1, 3]), visitor.var_map)
        self.assertIn(id(m.i3[1, 4]), visitor.var_map)
        self.assertIn(id(m.i3[1, 5]), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        x = visitor.var_map[id(m.x)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i33 = visitor.var_map[id(m.i3[1, 3])]
        i34 = visitor.var_map[id(m.i3[1, 4])]
        i35 = visitor.var_map[id(m.i3[1, 5])]

        self.assertTrue(
            expr[1].equals(
                cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)
                <= cp.element(
                    [cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)],
                    0 + 1 * (x + (-3) - 3) // 1,
                )
            )
        )

    def test_double_indirection_at_constraint(self):
        m = self.get_model()
        # add interval var x can index
        m.i3 = IntervalVar([(1, 3), (1, 4), (1, 5)], length=4, optional=True)
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i3[1, m.x - 3].start_time.at(m.i2[m.y].end_time))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i3[1, 3]), visitor.var_map)
        self.assertIn(id(m.i3[1, 4]), visitor.var_map)
        self.assertIn(id(m.i3[1, 5]), visitor.var_map)

        y = visitor.var_map[id(m.y)]
        x = visitor.var_map[id(m.x)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i33 = visitor.var_map[id(m.i3[1, 3])]
        i34 = visitor.var_map[id(m.i3[1, 4])]
        i35 = visitor.var_map[id(m.i3[1, 5])]

        self.assertTrue(
            expr[1].equals(
                cp.element(
                    [cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)],
                    0 + 1 * (x + (-3) - 3) // 1,
                )
                == cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)
            )
        )

    def test_indirection_nonconstant_step_size(self):
        m = ConcreteModel()

        def param_rule(m, i):
            return i + 1

        m.p = Param([1, 3, 4], initialize=param_rule)
        m.x = Var(within={1, 3, 4})
        e = m.p[m.x]

        visitor = self.get_visitor()
        with self.assertRaisesRegex(
            ValueError,
            r"Variable indirection 'p\[x\]' is over a discrete domain "
            "without a constant step size. This is not supported.",
        ):
            expr = visitor.walk_expression((e, e, 0))

    def test_indirection_with_param(self):
        m = ConcreteModel()

        def param_rule(m, i):
            return i + 1

        m.p = Param([1, 3, 5], initialize=param_rule)
        m.x = Var(within={1, 3, 5})
        m.a = Var(domain=Integers, bounds=(0, 100))

        e = m.p[m.x] / m.a
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        a = visitor.var_map[id(m.a)]

        self.assertTrue(expr[1].equals(cp.element([2, 4, 6], 0 + 1 * (x - 1) // 2) / a))


@unittest.skipIf(not docplex_available, "docplex is not available")
class TestCPExpressionWalker_CumulFuncExpressions(CommonTest):
    def test_always_in(self):
        m = self.get_model()
        f = (
            Pulse((m.i, 3))
            + Step(m.i2[1].start_time, height=2)
            - Step(m.i2[2].end_time, height=-1)
            + Step(3, height=4)
        )
        m.c = LogicalConstraint(expr=f.within((0, 3), (0, 10)))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)

        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]

        self.assertTrue(
            expr[1].equals(
                cp.always_in(
                    cp.pulse(i, 3)
                    + cp.step_at_start(i21, 2)
                    - cp.step_at_end(i22, -1)
                    + cp.step_at(3, 4),
                    interval=(0, 10),
                    min=0,
                    max=3,
                )
            )
        )


@unittest.skipIf(not docplex_available, "docplex is not available")
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
        m.c = Constraint(expr=m.e - 8 * m.e <= 32)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]

        self.assertTrue(expr[1].equals(x**2 + 7 + (-1) * (8 * (x**2 + 7))))


@unittest.skipIf(not docplex_available, "docplex is not available")
class TestCPExpressionWalker_Vars(CommonTest):
    def test_complain_about_non_integer_vars(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.is_present.implies(m.a[1] == 5))

        visitor = self.get_visitor()
        with self.assertRaisesRegex(
            ValueError,
            "The LogicalToDoCplex writer can only support integer- or "
            r"Boolean-valued variables. Cannot write Var 'a\[1\]' with "
            "domain 'Reals'",
        ):
            expr = visitor.walk_expression((m.c.expr, m.c, 0))

    def test_fixed_integer_var(self):
        m = self.get_model()
        m.a.domain = Integers
        m.a[1].fix(3)
        m.c = Constraint(expr=m.a[1] + m.a[2] >= 4)

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        self.assertIn(id(m.a[2]), visitor.var_map)
        a2 = visitor.var_map[id(m.a[2])]

        self.assertTrue(expr[1].equals(3 + a2))

    def test_fixed_boolean_var(self):
        m = self.get_model()
        m.b.fix(False)
        m.b2['a'].fix(True)
        m.c = LogicalConstraint(expr=m.b.lor(m.b2['a'].land(m.b2['b'])))

        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))

        self.assertIn(id(m.b2['b']), visitor.var_map)
        b2b = visitor.var_map[id(m.b2['b'])]

        self.assertTrue(expr[1].equals(cp.logical_or(False, cp.logical_and(True, b2b))))

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
        self.assertTrue(expr[1].equals(cp.element(a, 0 + 1 * (x - 6) // 1)))

    def test_indirection_multi_index_second_constant(self):
        m = self.get_model()
        m.z = Var(m.I, m.I, domain=Integers)

        e = m.z[m.x, 3]

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        z = {}
        for i in [6, 7, 8]:
            self.assertIn(id(m.z[i, 3]), visitor.var_map)
            z[i, 3] = visitor.var_map[id(m.z[i, 3])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]

        self.assertTrue(
            expr[1].equals(
                cp.element([z[i, 3] for i in [6, 7, 8]], 0 + 1 * (x - 6) // 1)
            )
        )

    def test_indirection_multi_index_first_constant(self):
        m = self.get_model()
        m.z = Var(m.I, m.I, domain=Integers)

        e = m.z[3, m.x]

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        z = {}
        for i in [6, 7, 8]:
            self.assertIn(id(m.z[3, i]), visitor.var_map)
            z[3, i] = visitor.var_map[id(m.z[3, i])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]

        self.assertTrue(
            expr[1].equals(
                cp.element([z[3, i] for i in [6, 7, 8]], 0 + 1 * (x - 6) // 1)
            )
        )

    def test_indirection_multi_index_neither_constant_same_var(self):
        m = self.get_model()
        m.z = Var(m.I, m.I, domain=Integers)

        e = m.z[m.x, m.x]

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        z = {}
        for i in [6, 7, 8]:
            for j in [6, 7, 8]:
                self.assertIn(id(m.z[i, j]), visitor.var_map)
                z[i, j] = visitor.var_map[id(m.z[i, j])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]

        self.assertTrue(
            expr[1].equals(
                cp.element(
                    [z[i, j] for i in [6, 7, 8] for j in [6, 7, 8]],
                    0 + 1 * (x - 6) // 1 + 3 * (x - 6) // 1,
                )
            )
        )

    def test_indirection_multi_index_neither_constant_diff_vars(self):
        m = self.get_model()
        m.z = Var(m.I, m.I, domain=Integers)
        m.y = Var(within=[1, 3, 5])

        e = m.z[m.x, m.y]

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        z = {}
        for i in [6, 7, 8]:
            for j in [1, 3, 5]:
                self.assertIn(id(m.z[i, 3]), visitor.var_map)
                z[i, j] = visitor.var_map[id(m.z[i, j])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertIn(id(m.y), visitor.var_map)
        y = visitor.var_map[id(m.y)]

        self.assertTrue(
            expr[1].equals(
                cp.element(
                    [z[i, j] for i in [6, 7, 8] for j in [1, 3, 5]],
                    0 + 1 * (x - 6) // 1 + 3 * (y - 1) // 2,
                )
            )
        )

    def test_indirection_expression_index(self):
        m = self.get_model()
        m.a.domain = Integers
        m.y = Var(within=[1, 3, 5])

        e = m.a[m.x - m.y]

        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        a = {}
        for i in range(1, 8):
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertIn(id(m.y), visitor.var_map)
        y = visitor.var_map[id(m.y)]

        self.assertTrue(
            expr[1].equals(
                cp.element([a[i] for i in range(1, 8)], 0 + 1 * (x + -1 * y - 1) // 1)
            )
        )

    def test_indirection_fails_with_non_finite_index_domain(self):
        m = self.get_model()
        m.a.domain = Integers
        # release the bounds
        m.x.setlb(None)
        m.x.setub(None)
        m.c = Constraint(expr=m.a[m.x] >= 0)

        visitor = self.get_visitor()
        with self.assertRaisesRegex(
            ValueError,
            r"Variable indirection 'a\[x\]' contains argument 'x', "
            "which is not restricted to a finite discrete domain",
        ):
            expr = visitor.walk_expression((m.c.body, m.c, 0))

    def test_indirection_invalid_index_domain(self):
        m = self.get_model()
        m.a.domain = Integers
        m.a.bounds = (6, 8)
        m.y = Var(within=Integers, bounds=(0, 10))

        e = m.a[m.y]

        visitor = self.get_visitor()
        with self.assertRaisesRegex(
            ValueError,
            r"Variable indirection 'a\[y\]' permits an index '0' "
            "that is not a valid key.",
        ):
            expr = visitor.walk_expression((e, e, 0))

    def test_infinite_domain_var(self):
        m = ConcreteModel()
        m.Evens = RangeSet(ranges=(NumericRange(0, None, 2), NumericRange(0, None, -2)))
        m.x = Var(domain=m.Evens)
        e = m.x**2

        visitor = self.get_visitor()
        with self.assertRaisesRegex(
            ValueError,
            "The LogicalToDoCplex writer does not support "
            "infinite discrete domains. Cannot "
            "write Var 'x' with domain 'Evens'",
        ):
            expr = visitor.walk_expression((e, e, 0))

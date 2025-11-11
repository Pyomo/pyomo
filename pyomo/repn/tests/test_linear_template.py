#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import unittest

import pyomo.core.base.constraint as constraint
import pyomo.core.base.objective as objective

from pyomo.common.errors import InvalidConstraintError
from pyomo.core.base.enums import SortComponents
from pyomo.repn.linear_template import LinearTemplateRepnVisitor
from pyomo.repn.util import TemplateVarRecorder

from pyomo.environ import *


class TestLinearTemplate(unittest.TestCase):
    def setUp(self):
        self._memo = (
            constraint.TEMPLATIZE_CONSTRAINTS,
            objective.TEMPLATIZE_OBJECTIVES,
        )
        constraint.TEMPLATIZE_CONSTRAINTS = True
        objective.TEMPLATIZE_OBJECTIVES = True
        var_recorder = TemplateVarRecorder({}, SortComponents.deterministic)
        self.visitor = LinearTemplateRepnVisitor({}, var_recorder=var_recorder)

    def tearDown(self):
        constraint.TEMPLATIZE_CONSTRAINTS, objective.TEMPLATIZE_OBJECTIVES = self._memo

    def _build_evaluator(self, expr):
        repn = self.visitor.walk_expression(expr)
        return repn._build_evaluator(
            self.visitor.symbolmap, self.visitor.expr_cache, 1, 1, False
        )

    def _build_evaluator_fcn(self, expr, args):
        repn = self.visitor.walk_expression(expr)
        return repn._build_evaluator_fcn(
            args, self.visitor.symbolmap, self.visitor.expr_cache, False
        )

    def _eval(self, obj):
        return self.visitor.expand_expression(obj, obj.template_expr())

    def test_repn_to_string(self):
        m = ConcreteModel()
        m.x = Var(range(3))
        m.p = Param(range(3), initialize={0: 5}, mutable=True, default=1)

        e = m.p[0] * m.x[0] + m.x[1] + 10

        repn = self.visitor.walk_expression(e)
        self.assertEqual(
            str(repn),
            "LinearTemplateRepn(mult=1, const=10, linear={0: 5, 1: 1}, "
            "linear_sum=[], nonlinear=None)",
        )

        @m.Objective()
        def obj(m):
            return sum(m.p[i] * m.x[i] for i in m.p.index_set())

        e = m.obj.template_expr()[0]
        repn = self.visitor.walk_expression(e)
        self.assertEqual(
            str(repn),
            "LinearTemplateRepn(mult=1, const=0, linear={}, "
            "linear_sum=[LinearTemplateRepn(mult=p[_1], const=0, "
            "linear={%s: 1}, linear_sum=[], nonlinear=None), "
            "[(_1)], [(0, 1, 2)]], nonlinear=None)"
            % (list(self.visitor.expr_cache)[-1],),
        )

    def test_no_indirection(self):
        m = ConcreteModel()
        m.x = Var()

        @m.Constraint()
        def c(m):
            return m.x <= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, None)
        self.assertEqual(ub, 0)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(['linear_indices.append(0)', 'linear_data.append(1)'], ans)

    def test_single_var_no_loop(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])

        @m.Constraint(m.x.index_set())
        def c(m, i):
            return m.x[i] <= 1

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[1].to_bounded_expression()
        self.assertEqual(lb, None)
        self.assertEqual(ub, 1)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            ['linear_indices.append(x1[x2])', 'linear_data.append(1)'], ans
        )

        @m.Constraint(m.x.index_set())
        def d(m, i):
            return i * m.x[i] <= 1

        self.assertTrue(hasattr(m.d[1], 'template_expr'))
        lb, body, ub = m.d[1].to_bounded_expression()
        self.assertEqual(lb, None)
        self.assertEqual(ub, 1)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            ['linear_indices.append(x1[x3])', 'linear_data.append(x3)'], ans
        )

    def test_two_var_const_no_loop(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        m.y = Var([1, 2, 3, 4])

        @m.Constraint(m.x.index_set())
        def c(m, i):
            return m.x[i] + 10 - m.y[i + 1] >= 0

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 10)
        self.assertEqual(
            [
                'linear_indices.append(x1[x2])',
                'linear_data.append(1)',
                'linear_indices.append(x3[x2 + 1])',
                'linear_data.append(-1)',
            ],
            ans,
        )

        @m.Constraint(m.x.index_set())
        def d(m, i):
            return m.x[i] + i - m.y[i + 1] >= 0

        self.assertTrue(hasattr(m.d[1], 'template_expr'))
        lb, body, ub = m.d[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'const += x4',
                'linear_indices.append(x1[x4])',
                'linear_data.append(1)',
                'linear_indices.append(x3[x4 + 1])',
                'linear_data.append(-1)',
            ],
            ans,
        )

    def test_explicit_sum(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.x = Var(range(5))

        @m.Constraint(m.I)
        def c(m, i):
            return sum([m.x[i - 1], m.x[i], m.x[i + 1]]) >= 0

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'linear_indices.append(x1[x2 - 1])',
                'linear_data.append(1)',
                'linear_indices.append(x1[x2])',
                'linear_data.append(1)',
                'linear_indices.append(x1[x2 + 1])',
                'linear_data.append(1)',
            ],
            ans,
        )

    def test_sum_one_var(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])

        @m.Constraint()
        def c(m):
            return sum(m.x[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x2 in x3:',
                '    linear_indices.append(x1[x2])',
                '    linear_data.append(1)',
            ],
            ans,
        )

        @m.Constraint()
        def d(m):
            return sum(m.x[i] + 2 for i in m.x) >= 0

        self.assertTrue(hasattr(m.d, 'template_expr'))
        lb, body, ub = m.d.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 6)
        self.assertEqual(
            [
                'for x4 in x3:',
                '    linear_indices.append(x1[x4])',
                '    linear_data.append(1)',
            ],
            ans,
        )

        m.p = Param(initialize=2, mutable=True)

        @m.Constraint()
        def e(m):
            return sum(m.x[i] + m.p for i in m.x) >= 0

        self.assertTrue(hasattr(m.e, 'template_expr'))
        lb, body, ub = m.e.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 6)
        self.assertEqual(
            [
                'for x5 in x3:',
                '    linear_indices.append(x1[x5])',
                '    linear_data.append(1)',
            ],
            ans,
        )

        m.q = Param([1, 2, 3], initialize={1: 10, 2: 20, 3: 30})

        @m.Constraint()
        def e(m):
            return sum(m.x[i] + m.q[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.e, 'template_expr'))
        lb, body, ub = m.e.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x6 in x3:',
                '    const += x7[x6]',
                '    linear_indices.append(x1[x6])',
                '    linear_data.append(1)',
            ],
            ans,
        )

    def test_sum_two_var(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        m.y = Var([1, 2, 3])

        @m.Constraint()
        def c(m):
            return sum(m.x[i] + m.y[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x2 in x4:',
                '    linear_indices.append(x1[x2])',
                '    linear_data.append(1)',
                '    linear_indices.append(x3[x2])',
                '    linear_data.append(1)',
            ],
            ans,
        )

        @m.Constraint()
        def d(m):
            return sum(m.x[i] + 2 + m.y[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.d, 'template_expr'))
        lb, body, ub = m.d.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 6)
        self.assertEqual(
            [
                'for x5 in x4:',
                '    linear_indices.append(x1[x5])',
                '    linear_data.append(1)',
                '    linear_indices.append(x3[x5])',
                '    linear_data.append(1)',
            ],
            ans,
        )

        m.p = Param(initialize=2, mutable=True)

        @m.Constraint()
        def e(m):
            return sum(m.x[i] + m.p + m.y[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.e, 'template_expr'))
        lb, body, ub = m.e.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 6)
        self.assertEqual(
            [
                'for x6 in x4:',
                '    linear_indices.append(x1[x6])',
                '    linear_data.append(1)',
                '    linear_indices.append(x3[x6])',
                '    linear_data.append(1)',
            ],
            ans,
        )

    def test_sum_with_multiplier(self):
        m = ConcreteModel()
        m.w = Var([1, 2, 3])
        m.x = Var([1, 2, 3])
        m.y = Var([1, 2, 3])
        m.z = Var([1, 2, 3])

        @m.Constraint()
        def c(m):
            return (
                sum(4 * (m.w[i] + 2 * m.x[i]) + 3 * (m.y[i] + 2 * m.z[i]) for i in m.x)
                >= 0
            )

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x2 in x6:',
                '    linear_indices.append(x1[x2])',
                '    linear_data.append(4)',
                '    linear_indices.append(x3[x2])',
                '    linear_data.append(8)',
                '    linear_indices.append(x4[x2])',
                '    linear_data.append(3)',
                '    linear_indices.append(x5[x2])',
                '    linear_data.append(6)',
            ],
            ans,
        )

    def test_nested_sum_with_multiplier(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.J = Set(initialize=[1, 2])
        m.x = Var([1, 2, 3], [1, 2])
        m.y = Var([1, 2, 3], [1, 2])

        @m.Constraint()
        def c(m):
            return (
                sum(
                    4 * sum(m.x[i, j] for j in m.J) + 3 * sum(m.y[i, j] for j in m.J)
                    for i in m.I
                )
                >= 0
            )

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x2 in x7:',
                '    for x3 in x4:',
                '        linear_indices.append(x1[x2,x3])',
                '        linear_data.append(4)',
                '    for x6 in x4:',
                '        linear_indices.append(x5[x2,x6])',
                '        linear_data.append(3)',
            ],
            ans,
        )

        @m.Constraint()
        def d(m):
            return (
                sum(
                    4 * sum(j * m.x[i, j] for j in m.J)
                    + 3 * sum(i * m.y[i, j] for j in m.J)
                    for i in m.I
                )
                >= 0
            )

        self.assertTrue(hasattr(m.d, 'template_expr'))
        lb, body, ub = m.d.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x9 in x7:',
                '    for x8 in x4:',
                '        linear_indices.append(x1[x9,x8])',
                '        linear_data.append(x8*4)',
                '    for x10 in x4:',
                '        linear_indices.append(x5[x9,x10])',
                '        linear_data.append(x9*3)',
            ],
            ans,
        )

    def test_filter_0_coef(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.J = Set(initialize=[1, 2])
        m.x = Var([1, 2, 3], [1, 2])
        m.y = Var([1, 2, 3], [1, 2])
        m.p = Param(mutable=True, initialize=0)

        @m.Constraint()
        def c(m):
            return sum(m.x[i] + m.y[i] - m.x[i] - m.p * m.y[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x2,x3 in x5:',
                '    linear_indices.append(x1[x2,x3])',
                '    linear_data.append(1)',
                '    linear_indices.append(x4[x2,x3])',
                '    linear_data.append(1)',
                '    linear_indices.append(x1[x2,x3])',
                '    linear_data.append(-1)',
            ],
            ans,
        )

        @m.Constraint()
        def d(m):
            return (
                sum(
                    4 * sum(j * m.x[i, j] for j in m.J)
                    + 0 * sum(m.y[i, j] for j in m.J)
                    for i in m.I
                )
                >= 0
            )

        self.assertTrue(hasattr(m.d, 'template_expr'))
        lb, body, ub = m.d.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x7 in x10:',
                '    for x6 in x8:',
                '        linear_indices.append(x1[x7,x6])',
                '        linear_data.append(x6*4)',
            ],
            ans,
        )

        @m.Constraint()
        def e(m):
            return (
                sum(
                    4 * sum(j * m.x[i, j] for j in m.J)
                    + 0 * sum(i * m.y[i, j] for j in m.J)
                    for i in m.I
                )
                >= 0
            )

        self.assertTrue(hasattr(m.e, 'template_expr'))
        lb, body, ub = m.e.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x12 in x10:',
                '    for x11 in x8:',
                '        linear_indices.append(x1[x12,x11])',
                '        linear_data.append(x11*4)',
                '    for x13 in x8:',
                '        linear_indices.append(x4[x12,x13])',
                '        linear_data.append(x12*0)',
            ],
            ans,
        )

    def test_iter_nonfinite_component(self):
        m = ConcreteModel()
        m.x = Var(NonNegativeIntegers, dense=False)
        m.p = Param(mutable=True, initialize=0)
        m.x[1] = 1
        m.x[2] = 2

        @m.Constraint()
        def c(m):
            return sum(m.p * m.x[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual([], ans)

        m.p = 1
        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x2 in x3:',
                '    linear_indices.append(x1[x2])',
                '    linear_data.append(1)',
            ],
            ans,
        )

    def test_set_of_sets(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.J = Set(m.I, initialize=lambda m, i: range(i))
        m.x = Var([1, 2], [0, 1])

        @m.Constraint(m.I)
        def c(m, i):
            return sum(m.x[i, j] for j in m.J[i]) == 2

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[1].to_bounded_expression()
        self.assertEqual(lb, 2)
        self.assertEqual(ub, 2)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            [
                'for x3 in x4[x2]:',
                '    linear_indices.append(x1[x2,x3])',
                '    linear_data.append(1)',
            ],
            ans,
        )

    def test_general_nonlinear(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        m.y = Var([1, 2, 3])

        @m.Constraint()
        def c(m):
            return sum(m.x[i] ** 2 for i in m.x) >= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        with self.assertRaisesRegex(
            InvalidConstraintError,
            "LinearTemplateRepn does not support constraints containing "
            "general nonlinear terms.",
        ):
            ans, const = self._build_evaluator(body)

    def test_monomial_expr(self):
        m = ConcreteModel()
        m.x = Var()

        @m.Constraint([1, 2, 3])
        def c(m, i):
            return (0, 5 * m.x, i)

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[2].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(str(ub), '_1')

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(['linear_indices.append(0)', 'linear_data.append(5)'], ans)

    def test_fcn_no_sum_expr(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        m.y = Var([1, 2, 3, 4])

        @m.Constraint(m.x.index_set())
        def c(m, i):
            return m.x[i] + 10 - m.y[i + 1] >= 0

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        fcn, fname = self._build_evaluator_fcn(body, ())
        self.assertEqual(fname, 'build_expr')
        self.assertEqual(
            '''def build_expr(linear_indices, linear_data, ):
    linear_indices.append(x1[x2])
    linear_data.append(1)
    linear_indices.append(x3[x2 + 1])
    linear_data.append(-1)
    return 10''',
            fcn,
        )

    def test_fcn_sum_expr_no_const(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3, 4])

        @m.Constraint()
        def c(m):
            return sum(m.x[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        fcn, fname = self._build_evaluator_fcn(body, ())
        self.assertEqual(fname, 'build_expr')
        self.assertEqual(
            '''def build_expr(linear_indices, linear_data, ):
    for x2 in x3:
        linear_indices.append(x1[x2])
        linear_data.append(1)
    return 0''',
            fcn,
        )

    def test_fcn_sum_expr_const(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3, 4])

        @m.Constraint()
        def c(m):
            return sum(m.x[i] + i for i in m.x) >= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        fcn, fname = self._build_evaluator_fcn(body, ())
        self.assertEqual(fname, 'build_expr')
        self.assertEqual(
            '''def build_expr(linear_indices, linear_data, ):
    const = 0
    for x2 in x3:
        const += x2
        linear_indices.append(x1[x2])
        linear_data.append(1)
    return const''',
            fcn,
        )

    def test_fcn_const(self):
        m = ConcreteModel()
        m.x = Var()
        m.x.fix(4)

        @m.Constraint()
        def c(m):
            return m.x + 5 >= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        fcn, fname = self._build_evaluator_fcn(body, ())
        self.assertEqual(fname, None)
        self.assertEqual(9, fcn(None, None))

    def test_fcn_with_outer_const(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I)
        m.y = Var([1, 2, 3])
        m.p = Param(m.I, initialize=lambda m, i: i)

        @m.Constraint(m.I)
        def c(m, i):
            return 5 + m.x[i] + m.p[i] + sum(m.y.values()) >= 0

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        fcn, fname = self._build_evaluator_fcn(body, ())
        self.assertEqual(fname, 'build_expr')
        self.assertEqual(
            """def build_expr(linear_indices, linear_data, ):
    linear_indices.append(x1[x2])
    linear_data.append(1)
    for x5 in x6:
        linear_indices.append(x4[x5])
        linear_data.append(1)
    return 5 + x3[x2]""",
            fcn,
        )

    def test_fcn_with_outer_and_inner_const(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I)
        m.y = Var([1, 2, 3])
        m.p = Param(m.I, initialize=lambda m, i: i)

        @m.Constraint(m.I)
        def c(m, i):
            return 5 + m.x[i] + m.p[i] + sum(m.y[j] + j for j in m.y) >= 0

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        fcn, fname = self._build_evaluator_fcn(body, ())
        self.assertEqual(fname, 'build_expr')
        self.assertEqual(
            """def build_expr(linear_indices, linear_data, ):
    const = 5 + x3[x2]
    linear_indices.append(x1[x2])
    linear_data.append(1)
    for x5 in x6:
        const += x5
        linear_indices.append(x4[x5])
        linear_data.append(1)
    return const""",
            fcn,
        )

    def test_fcn_with_nonfinite(self):
        m = ConcreteModel()
        m.x = Var(Any, dense=False)

        @m.Constraint()
        def c(m):
            return 5 + sum(m.x.values()) >= 0

        m.x[1]
        m.x[3]
        m.x[5]

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        fcn, fname = self._build_evaluator_fcn(body, ())
        self.assertEqual(fname, 'build_expr')
        self.assertEqual(
            """def build_expr(linear_indices, linear_data, ):
    for x2 in x3:
        linear_indices.append(x1[x2])
        linear_data.append(1)
    return 5""",
            fcn,
        )

    def test_fcn_explicit_sum(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])

        @m.Constraint()
        def c(m):
            return sum(i * m.x[i] for i in [1, 2, 3]) >= 0

        self.assertTrue(hasattr(m.c, 'template_expr'))
        lb, body, ub = m.c.to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        fcn, fname = self._build_evaluator_fcn(body, ())
        self.assertEqual(fname, 'build_expr')
        self.assertEqual(
            """def build_expr(linear_indices, linear_data, ):
    linear_indices.append(0)
    linear_data.append(1)
    linear_indices.append(1)
    linear_data.append(2)
    linear_indices.append(2)
    linear_data.append(3)
    return 0""",
            fcn,
        )

    def test_eval_no_sum_expr(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        m.y = Var([1, 2, 3, 4])

        @m.Constraint(m.x.index_set())
        def c(m, i):
            return m.x[i] + 10 - m.y[i + 1] >= 0

        const, var_list, coef_list, lb, ub = self._eval(m.c[1])
        self.assertEqual(const, 10)
        self.assertEqual(var_list, [0, 4])
        self.assertEqual(coef_list, [1, -1])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)
        const, var_list, coef_list, lb, ub = self._eval(m.c[2])
        self.assertEqual(const, 10)
        self.assertEqual(var_list, [1, 5])
        self.assertEqual(coef_list, [1, -1])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

    def test_eval_sum_expr_no_const(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3, 4])

        @m.Constraint()
        def c(m):
            return sum(m.x[i] for i in m.x) >= 0

        const, var_list, coef_list, lb, ub = self._eval(m.c)
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [0, 1, 2, 3])
        self.assertEqual(coef_list, [1, 1, 1, 1])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

    def test_eval_sum_expr_const(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3, 4])

        @m.Constraint()
        def c(m):
            return sum(m.x[i] + i for i in m.x) >= 0

        const, var_list, coef_list, lb, ub = self._eval(m.c)
        self.assertEqual(const, 10)
        self.assertEqual(var_list, [0, 1, 2, 3])
        self.assertEqual(coef_list, [1, 1, 1, 1])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

    def test_eval_const(self):
        m = ConcreteModel()
        m.x = Var()
        m.x.fix(4)

        @m.Constraint()
        def c(m):
            return m.x + 5 >= 0

        const, var_list, coef_list, lb, ub = self._eval(m.c)
        self.assertEqual(const, 9)
        self.assertEqual(var_list, [])
        self.assertEqual(coef_list, [])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

    def test_eval_with_outer_const(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I)
        m.y = Var([1, 2, 3])
        m.p = Param(m.I, initialize=lambda m, i: i)

        @m.Constraint(m.I)
        def c(m, i):
            return 5 + m.x[i] + m.p[i] + sum(m.y.values()) >= 0

        const, var_list, coef_list, lb, ub = self._eval(m.c[1])
        self.assertEqual(const, 6)
        self.assertEqual(var_list, [0, 2, 3, 4])
        self.assertEqual(coef_list, [1, 1, 1, 1])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)
        const, var_list, coef_list, lb, ub = self._eval(m.c[2])
        self.assertEqual(const, 7)
        self.assertEqual(var_list, [1, 2, 3, 4])
        self.assertEqual(coef_list, [1, 1, 1, 1])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

    def test_eval_with_outer_and_inner_const(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I)
        m.y = Var([1, 2, 3])
        m.p = Param(m.I, initialize=lambda m, i: i)

        @m.Constraint(m.I)
        def c(m, i):
            return 5 + m.x[i] + m.p[i] + sum(m.y[j] + j for j in m.y) >= 0

        const, var_list, coef_list, lb, ub = self._eval(m.c[1])
        self.assertEqual(const, 12)
        self.assertEqual(var_list, [0, 2, 3, 4])
        self.assertEqual(coef_list, [1, 1, 1, 1])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)
        const, var_list, coef_list, lb, ub = self._eval(m.c[2])
        self.assertEqual(const, 13)
        self.assertEqual(var_list, [1, 2, 3, 4])
        self.assertEqual(coef_list, [1, 1, 1, 1])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

    def test_eval_with_nonfinite(self):
        m = ConcreteModel()
        m.x = Var(Any, dense=False)

        @m.Constraint()
        def c(m):
            return 5 + sum(m.x.values()) >= 0

        m.x[1]
        m.x[3]
        m.x[5]

        const, var_list, coef_list, lb, ub = self._eval(m.c)
        self.assertEqual(const, 5)
        self.assertEqual(var_list, [0, 1, 2])
        self.assertEqual(coef_list, [1, 1, 1])
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

    def test_eval_explicit_sum(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])

        @m.Constraint()
        def c(m):
            return sum(i * m.x[i] for i in [1, 2, 3]) >= 10

        const, var_list, coef_list, lb, ub = self._eval(m.c)
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [0, 1, 2])
        self.assertEqual(coef_list, [1, 2, 3])
        self.assertEqual(lb, 10)
        self.assertEqual(ub, None)

    def test_eval_ranged_const(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var([1, 2])

        @m.Constraint(m.I)
        def c(m, i):
            return inequality(2, m.x[i], 4)

        const, var_list, coef_list, lb, ub = self._eval(m.c[1])
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [0])
        self.assertEqual(coef_list, [1])
        self.assertEqual(lb, 2)
        self.assertEqual(ub, 4)
        const, var_list, coef_list, lb, ub = self._eval(m.c[2])
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [1])
        self.assertEqual(coef_list, [1])
        self.assertEqual(lb, 2)
        self.assertEqual(ub, 4)

    def test_eval_ranged_expr(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I)
        m.p = Param(m.I, initialize=lambda m, i: i * 2)

        @m.Constraint(m.I)
        def c(m, i):
            return inequality(m.p[i], m.x[i], 2 * m.p[i])

        const, var_list, coef_list, lb, ub = self._eval(m.c[1])
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [0])
        self.assertEqual(coef_list, [1])
        self.assertEqual(lb, 2)
        self.assertEqual(ub, 4)
        const, var_list, coef_list, lb, ub = self._eval(m.c[2])
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [1])
        self.assertEqual(coef_list, [1])
        self.assertEqual(lb, 4)
        self.assertEqual(ub, 8)

    def test_eval_ranged_fixed_expr(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I)
        m.p = Var([1, 2, 3], initialize=lambda m, i: i * 2)
        m.p[1].fix()
        m.p[2].fix()

        @m.Constraint(m.I)
        def c(m, i):
            return inequality(m.p[i], m.x[i], 2 * m.p[i])

        const, var_list, coef_list, lb, ub = self._eval(m.c[1])
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [0])
        self.assertEqual(coef_list, [1])
        self.assertEqual(lb, 2)
        self.assertEqual(ub, 4)
        const, var_list, coef_list, lb, ub = self._eval(m.c[2])
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [1])
        self.assertEqual(coef_list, [1])
        self.assertEqual(lb, 4)
        self.assertEqual(ub, 8)

    def test_eval_ranged_unfixed_expr(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I)
        m.p = Var([1, 2, 3], initialize=lambda m, i: i * 2)
        m.p.fix()

        @m.Constraint(m.I)
        def c(m, i):
            return inequality(m.p[i], m.x[i], 2 * m.p[i + 1])

        m.p[1].unfix()
        m.p[3].unfix()
        with self.assertRaisesRegex(
            RuntimeError, r"Constraint c\[1\] has non-fixed lower bound"
        ):
            self._eval(m.c[1])
        with self.assertRaisesRegex(
            RuntimeError, r"Constraint c\[2\] has non-fixed upper bound"
        ):
            self._eval(m.c[2])

    def test_eval_default_param_expr(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.x = Var(m.I)
        m.p = Param(m.I, initialize={2: 5}, default=10)

        @m.Constraint(m.I)
        def c(m, i):
            return m.x[i] >= m.p[i]

        const, var_list, coef_list, lb, ub = self._eval(m.c[1])
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [0])
        self.assertEqual(coef_list, [1])
        self.assertEqual(lb, 10)
        self.assertEqual(ub, None)
        const, var_list, coef_list, lb, ub = self._eval(m.c[2])
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [1])
        self.assertEqual(coef_list, [1])
        self.assertEqual(lb, 5)
        self.assertEqual(ub, None)
        const, var_list, coef_list, lb, ub = self._eval(m.c[3])
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [2])
        self.assertEqual(coef_list, [1])
        self.assertEqual(lb, 10)
        self.assertEqual(ub, None)

    def test_eval_objective(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I)
        m.obj = Objective(expr=sum(i + (i * 3) * m.x[i] for i in m.I))

        const, var_list, coef_list, lb, ub = self._eval(m.obj)
        self.assertEqual(const, 3)
        self.assertEqual(var_list, [0, 1])
        self.assertEqual(coef_list, [3, 6])
        self.assertEqual(lb, None)
        self.assertEqual(ub, None)

    def test_skip(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2])
        m.x = Var(m.I)

        @m.Constraint()
        def c(m):
            return Constraint.Skip

        const, var_list, coef_list, lb, ub = self._eval(m.c)
        self.assertEqual(const, 0)
        self.assertEqual(var_list, [])
        self.assertEqual(coef_list, [])
        self.assertEqual(lb, None)
        self.assertEqual(ub, None)

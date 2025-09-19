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
        self.assertEqual(ans, ['linear_indices.append(0)', 'linear_data.append(1)'])

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
            ans, ['linear_indices.append(x1[x2])', 'linear_data.append(1)']
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
            ans, ['linear_indices.append(x1[x3])', 'linear_data.append(x3)']
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
            ans,
            [
                'linear_indices.append(x1[x2])',
                'linear_data.append(1)',
                'linear_indices.append(x3[x2 + 1])',
                'linear_data.append(-1)',
            ],
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
            ans,
            [
                'const += x4',
                'linear_indices.append(x1[x4])',
                'linear_data.append(1)',
                'linear_indices.append(x3[x4 + 1])',
                'linear_data.append(-1)',
            ],
        )

    def test_sum_one_var(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])

        @m.Constraint(m.x.index_set())
        def c(m, i):
            return sum(m.x[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            ans,
            [
                'for x2 in x3:',
                '    linear_indices.append(x1[x2])',
                '    linear_data.append(1)',
            ],
        )

        @m.Constraint(m.x.index_set())
        def d(m, i):
            return sum(m.x[i] + 2 for i in m.x) >= 0

        self.assertTrue(hasattr(m.d[1], 'template_expr'))
        lb, body, ub = m.d[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 6)
        self.assertEqual(
            ans,
            [
                'for x4 in x3:',
                '    linear_indices.append(x1[x4])',
                '    linear_data.append(1)',
            ],
        )

        m.p = Param(initialize=2, mutable=True)

        @m.Constraint(m.x.index_set())
        def e(m, i):
            return sum(m.x[i] + m.p for i in m.x) >= 0

        self.assertTrue(hasattr(m.e[1], 'template_expr'))
        lb, body, ub = m.e[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 6)
        self.assertEqual(
            ans,
            [
                'for x5 in x3:',
                '    linear_indices.append(x1[x5])',
                '    linear_data.append(1)',
            ],
        )

        m.q = Param([1, 2, 3], initialize={1: 10, 2: 20, 3: 30})

        @m.Constraint(m.x.index_set())
        def e(m, i):
            return sum(m.x[i] + m.q[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.e[1], 'template_expr'))
        lb, body, ub = m.e[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            ans,
            [
                'for x6 in x3:',
                '    const += x7[x6]',
                '    linear_indices.append(x1[x6])',
                '    linear_data.append(1)',
            ],
        )

    def test_sum_two_var(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        m.y = Var([1, 2, 3])

        @m.Constraint(m.x.index_set())
        def c(m, i):
            return sum(m.x[i] + m.y[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.c[1], 'template_expr'))
        lb, body, ub = m.c[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 0)
        self.assertEqual(
            ans,
            [
                'for x2 in x4:',
                '    linear_indices.append(x1[x2])',
                '    linear_data.append(1)',
                '    linear_indices.append(x3[x2])',
                '    linear_data.append(1)',
            ],
        )

        @m.Constraint(m.x.index_set())
        def d(m, i):
            return sum(m.x[i] + 2 + m.y[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.d[1], 'template_expr'))
        lb, body, ub = m.d[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 6)
        self.assertEqual(
            ans,
            [
                'for x5 in x4:',
                '    linear_indices.append(x1[x5])',
                '    linear_data.append(1)',
                '    linear_indices.append(x3[x5])',
                '    linear_data.append(1)',
            ],
        )

        m.p = Param(initialize=2, mutable=True)

        @m.Constraint(m.x.index_set())
        def e(m, i):
            return sum(m.x[i] + m.p + m.y[i] for i in m.x) >= 0

        self.assertTrue(hasattr(m.e[1], 'template_expr'))
        lb, body, ub = m.e[1].to_bounded_expression()
        self.assertEqual(lb, 0)
        self.assertEqual(ub, None)

        ans, const = self._build_evaluator(body)
        self.assertEqual(const, 6)
        self.assertEqual(
            ans,
            [
                'for x6 in x4:',
                '    linear_indices.append(x1[x6])',
                '    linear_data.append(1)',
                '    linear_indices.append(x3[x6])',
                '    linear_data.append(1)',
            ],
        )

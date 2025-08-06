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


from math import isnan
import unittest

from pyomo.core.expr import SumExpression, MonomialTermExpression
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import Any, ConcreteModel, log, Param, Var
from pyomo.repn.parameterized import ParameterizedQuadraticRepnVisitor
from pyomo.repn.tests.test_linear import VisitorConfig
from pyomo.repn.util import InvalidNumber

nan = float('nan')


def build_test_model():
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.p = Param(initialize=1, mutable=True)

    return m


class TestParameterizedQuadratic(unittest.TestCase):
    def test_constant_literal(self):
        """
        Ensure ParameterizedQuadraticRepnVisitor(*args, wrt=[]) works
        like QuadraticRepnVisitor.
        """
        expr = 2

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 2)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        self.assertEqual(repn.to_expression(visitor), 2)

    def test_constant_param(self):
        m = build_test_model()
        m.p.set_value(2)
        expr = 2 + m.p

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 4)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), 4)

    def test_binary_sum_identical_terms(self):
        m = build_test_model()
        expr = m.x + m.x

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), 2 * m.x)

    def test_binary_sum_identical_terms_wrt_x(self):
        m = build_test_model()
        expr = m.x + m.x

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.x])
        # note: covers walker_exitNode for case where
        #       constant is a fixed expression
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, m.x + m.x)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), m.x + m.x)

    def test_binary_sum_nonidentical_terms(self):
        m = build_test_model()
        expr = m.x + m.y

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 1, id(m.y): 1})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), m.x + m.y)

    def test_binary_sum_nonidentical_terms_wrt_x(self):
        m = build_test_model()
        expr = m.x + m.y

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.x])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.y): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, m.x)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), m.y + m.x)

    def test_ternary_sum_with_product(self):
        m = build_test_model()
        e = m.x + m.z * m.y + m.z

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.z): m.z, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.z): 1, id(m.y): 2})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear), 2)
        self.assertEqual(repn.linear[id(m.x)], 1)
        self.assertEqual(repn.linear[id(m.z)], 1)
        self.assertEqual(len(repn.quadratic), 1)
        self.assertEqual(repn.quadratic[(id(m.z), id(m.y))], 1)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), m.z * m.y + m.x + m.z)

    def test_ternary_sum_with_product_wrt_z(self):
        m = build_test_model()
        e = m.x + m.z * m.y + m.z

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.z])
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertIs(repn.constant, m.z)
        self.assertEqual(len(repn.linear), 2)
        self.assertEqual(repn.linear[id(m.x)], 1)
        self.assertIs(repn.linear[id(m.y)], m.z)
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), m.x + m.z * m.y + m.z)

    def test_nonlinear_wrt_x(self):
        m = build_test_model()
        expr = log(m.x)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.x])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, log(m.x))
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), log(m.x))

    def test_linear_constant_coeffs(self):
        m = build_test_model()
        e = 2 + 3 * m.x

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 2)
        self.assertEqual(repn.linear, {id(m.x): 3})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), 3 * m.x + 2)

    def test_linear_constant_coeffs_wrt_x(self):
        m = build_test_model()
        e = 2 + 3 * m.x

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.x])
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 2 + 3 * m.x)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), 2 + 3 * m.x)

    def test_quadratic(self):
        m = build_test_model()
        e = 2 + 3 * m.x + 4 * m.x**2

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 2)
        self.assertEqual(repn.linear, {id(m.x): 3})
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 4})
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), 4 * m.x**2 + 3 * m.x + 2
        )

    def test_product_quadratic_quadratic(self):
        m = build_test_model()
        e = (2 + 3 * m.x + 4 * m.x**2) * (5 + 6 * m.x + 7 * m.x**2)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        QE4 = 4 * m.x**2
        QE7 = 7 * m.x**2
        LE3 = MonomialTermExpression((3, m.x))
        LE6 = MonomialTermExpression((6, m.x))
        NL = +QE4 * (QE7 + LE6) + (LE3) * (QE7)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 10)
        self.assertEqual(repn.linear, {id(m.x): 27})
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 52})
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), NL + 52 * m.x**2 + 27 * m.x + 10
        )

    def test_product_quadratic_quadratic_2(self):
        m = build_test_model()
        e = (2 + 3 * m.x + 4 * m.x**2) * (5 + 6 * m.x + 7 * m.x**2)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        visitor.expand_nonlinear_products = False
        repn = visitor.walk_expression(e)

        NL = (4 * m.x**2 + 3 * m.x + 2) * (7 * m.x**2 + 6 * m.x + 5)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(self, repn.to_expression(visitor), NL)

    def test_product_linear_linear(self):
        m = build_test_model()
        e = (1 + 2 * m.x + 3 * m.y) * (4 + 5 * m.x + 6 * m.y)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 4)
        self.assertEqual(repn.linear, {id(m.x): 13, id(m.y): 18})
        self.assertEqual(
            repn.quadratic,
            {(id(m.x), id(m.x)): 10, (id(m.y), id(m.y)): 18, (id(m.x), id(m.y)): 27},
        )
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (10 * m.x**2 + 27 * (m.x * m.y) + 18 * m.y**2 + 13 * m.x + 18 * m.y + 4),
        )

    def test_product_linear_linear_wrt_y(self):
        m = build_test_model()
        e = (1 + 2 * m.x + 3 * m.y) * (4 + 5 * m.x + 6 * m.y)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, (1 + 3 * m.y) * (4 + 6 * m.y))
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(
            self, repn.linear[id(m.x)], (4 + 6 * m.y) * 2 + (1 + 3 * m.y) * 5
        )
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 10})
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (
                10 * m.x**2
                + ((4 + 6 * m.y) * 2 + (1 + 3 * m.y) * 5) * m.x
                + (1 + 3 * m.y) * (4 + 6 * m.y)
            ),
        )

    def test_product_linear_linear_const_0(self):
        m = build_test_model()
        expr = (0 + 3 * m.x + 4 * m.y) * (5 + 3 * m.x + 7 * m.y)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 15, id(m.y): 20})
        self.assertEqual(
            repn.quadratic,
            {(id(m.x), id(m.x)): 9, (id(m.x), id(m.y)): 33, (id(m.y), id(m.y)): 28},
        )
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            9 * m.x**2 + 33 * (m.x * m.y) + 28 * m.y**2 + 15 * m.x + 20 * m.y,
        )

    def test_product_linear_quadratic(self):
        m = build_test_model()
        expr = (5 + 3 * m.x + 7 * m.y) * (1 + 3 * m.x + 4 * m.y + 8 * m.y * m.x)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        NL = (3 * m.x + 7 * m.y + 5) * (8 * (m.x * m.y) + 3 * m.x + 4 * m.y + 1)

        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(self, repn.to_expression(visitor), NL)

        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 5)
        self.assertEqual(repn.linear, {id(m.x): 18, id(m.y): 27})
        self.assertEqual(
            repn.quadratic,
            {(id(m.x), id(m.y)): 73, (id(m.x), id(m.x)): 9, (id(m.y), id(m.y)): 28},
        )
        assertExpressionsEqual(
            self, repn.nonlinear, (3 * m.x + 7 * m.y) * (8 * (m.x * m.y))
        )
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (
                73 * (m.x * m.y)
                + 9 * m.x**2
                + 28 * m.y**2
                + 18 * m.x
                + 27 * m.y
                + 5
                + (3 * m.x + 7 * m.y) * (8 * (m.x * m.y))
            ),
        )

    def test_product_linear_quadratic_wrt_x(self):
        m = build_test_model()
        expr = (0 + 3 * m.x + 4 * m.y + 8 * m.y * m.x) * (5 + 3 * m.x + 7 * m.y)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.x])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.y): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 3 * m.x * (5 + 3 * m.x))
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(
            self, repn.linear[id(m.y)], (5 + 3 * m.x) * (4 + 8 * m.x) + 21 * m.x
        )
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(
            self, repn.quadratic[id(m.y), id(m.y)], (4 + 8 * m.x) * 7
        )
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (4 + 8 * m.x) * 7 * m.y**2
            + ((5 + 3 * m.x) * (4 + 8 * m.x) + 21 * m.x) * m.y
            + 3 * m.x * (5 + 3 * m.x),
        )

    def test_product_nonlinear_var_expand_false(self):
        m = build_test_model()
        e = (m.x + m.y + log(m.x)) * m.x

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        visitor.expand_nonlinear_products = False
        repn = visitor.walk_expression(e)

        NL = (log(m.x) + (m.x + m.y)) * m.x

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(self, repn.to_expression(visitor), NL)

    def test_product_nonlinear_var_expand_true(self):
        m = build_test_model()
        e = (m.x + m.y + log(m.x)) * m.x

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        NL = log(m.x) * m.x

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 1, (id(m.x), id(m.y)): 1})
        assertExpressionsEqual(self, repn.nonlinear, NL)

    def test_product_nonlinear_var_2_expand_false(self):
        m = build_test_model()
        e = m.x * (m.x + m.y + log(m.x) + 2)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        visitor.expand_nonlinear_products = False
        repn = visitor.walk_expression(e)

        NL = m.x * (log(m.x) + (m.x + m.y + 2))

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(self, repn.to_expression(visitor), NL)

    def test_product_nonlinear_var_2_expand_true(self):
        m = build_test_model()
        e = m.x * (m.x + m.y + log(m.x) + 2)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        NL = m.x * log(m.x)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 1, (id(m.x), id(m.y)): 1})
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), m.x**2 + m.x * m.y + 2 * m.x + NL
        )

    def test_zero_elimination(self):
        m = ConcreteModel()
        m.x = Var(range(4))
        e = 0 * m.x[0] + 0 * m.x[1] * m.x[2] + 0 * log(m.x[3])

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(
            cfg.var_map,
            {
                id(m.x[0]): m.x[0],
                id(m.x[1]): m.x[1],
                id(m.x[2]): m.x[2],
                id(m.x[3]): m.x[3],
            },
        )
        self.assertEqual(
            cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1, id(m.x[2]): 2, id(m.x[3]): 3}
        )
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, 0 * log(m.x[3]))
        assertExpressionsEqual(self, repn.to_expression(visitor), 0 * log(m.x[3]))

    def test_uninitialized_param_expansion(self):
        m = ConcreteModel()
        m.x = Var(range(4))
        m.p = Param(mutable=True, within=Any, initialize=None)
        e = m.p * m.x[0] + m.p * m.x[1] * m.x[2] + m.p * log(m.x[3])

        cfg = VisitorConfig()
        repn = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[]).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(
            cfg.var_map,
            {
                id(m.x[0]): m.x[0],
                id(m.x[1]): m.x[1],
                id(m.x[2]): m.x[2],
                id(m.x[3]): m.x[3],
            },
        )
        self.assertEqual(
            cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1, id(m.x[2]): 2, id(m.x[3]): 3}
        )
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x[0]): InvalidNumber(None)})
        self.assertEqual(
            repn.quadratic, {(id(m.x[1]), id(m.x[2])): InvalidNumber(None)}
        )
        self.assertEqual(repn.nonlinear, InvalidNumber(None))

    def test_zero_times_var(self):
        m = build_test_model()
        e = 0 * m.x

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), 0)

    def test_square_linear(self):
        m = build_test_model()
        expr = (1 + 3 * m.x + 4 * m.y) ** 2

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 1)
        self.assertEqual(repn.linear, {id(m.x): 6, id(m.y): 8})
        self.assertEqual(
            repn.quadratic,
            {(id(m.x), id(m.x)): 9, (id(m.y), id(m.y)): 16, (id(m.x), id(m.y)): 24},
        )
        self.assertEqual(repn.nonlinear, None)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            9 * m.x**2 + 16 * m.y**2 + 24 * (m.x * m.y) + 6 * m.x + 8 * m.y + 1,
        )

    def test_square_linear_wrt_y(self):
        m = build_test_model()
        expr = (1 + 3 * m.x + 4 * m.y) ** 2

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, (1 + 4 * m.y) ** 2)
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(self, repn.linear[id(m.x)], 3 * (2 * (1 + 4 * m.y)))
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 9})
        self.assertEqual(repn.nonlinear, None)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (9 * m.x**2 + 3 * (2 * (1 + 4 * m.y)) * m.x + (1 + 4 * m.y) ** 2),
        )

    def test_square_linear_float(self):
        m = build_test_model()
        expr = (1 + 3 * m.x + 4 * m.y) ** 2.0

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 1)
        self.assertEqual(repn.linear, {id(m.x): 6, id(m.y): 8})
        self.assertEqual(
            repn.quadratic,
            {(id(m.x), id(m.x)): 9, (id(m.y), id(m.y)): 16, (id(m.x), id(m.y)): 24},
        )
        self.assertEqual(repn.nonlinear, None)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            9 * m.x**2 + 16 * m.y**2 + 24 * (m.x * m.y) + 6 * m.x + 8 * m.y + 1,
        )

    def test_division_quadratic_nonlinear(self):
        m = build_test_model()
        expr = (1 + 3 * m.x + 4 * log(m.x) * m.y + 4 * m.y**2) / (2 * m.x)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(
            self,
            repn.nonlinear,
            (4 * m.y**2 + 3 * m.x + 1 + log(m.x) * 4 * m.y) / (2 * m.x),
        )
        assertExpressionsEqual(self, repn.to_expression(visitor), repn.nonlinear)

    def test_division_quadratic_nonlinear_wrt_x(self):
        m = build_test_model()
        expr = (1 + 3 * m.x + 4 * log(m.x) * m.y + 4 * m.y**2) / (2 * m.x)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.x])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.y): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, (1 + 3 * m.x) * (1 / (2 * m.x)))
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(
            self, repn.linear[id(m.y)], (4 * log(m.x)) * (1 / (2 * m.x))
        )
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(
            self, repn.quadratic[id(m.y), id(m.y)], 4 * (1 / (2 * m.x))
        )
        self.assertEqual(repn.nonlinear, None)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (4 * (1 / (2 * m.x))) * m.y**2
            + ((4 * log(m.x)) * (1 / (2 * m.x))) * m.y
            + (1 + 3 * m.x) * (1 / (2 * m.x)),
        )

    def test_constant_expr_multiplier(self):
        m = build_test_model()
        expr = 5 * (2 * m.x + m.x**2)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 10})
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 5})
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), 5 * m.x**2 + 10 * m.x)

    def test_0_mult_nan_linear_coeff(self):
        m = build_test_model()
        expr = 0 * (nan * m.x + m.y + log(m.x) + m.y * m.x**2 + 2 * m.x)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 0 * m.y)
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(self, repn.linear[id(m.x)], nan)
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(self, repn.quadratic[id(m.x), id(m.x)], 0 * m.y)
        assertExpressionsEqual(self, repn.nonlinear, (log(m.x)) * 0)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            0 * m.y * m.x**2 + nan * m.x + 0 * m.y + log(m.x) * 0,
        )

    def test_0_mult_nan_quadratic_coeff(self):
        m = build_test_model()
        expr = 0 * (m.x + m.y + log(m.x) + nan * m.x**2 + 2 * m.x)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        # visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 0 * m.y)
        self.assertEqual(repn.linear, {})
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(self, repn.quadratic[id(m.x), id(m.x)], nan)
        assertExpressionsEqual(self, repn.nonlinear, (log(m.x)) * 0)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), nan * m.x**2 + 0 * m.y + log(m.x) * 0
        )

    def test_square_quadratic(self):
        m = build_test_model()
        expr = (1 + m.x + m.y + m.x**2 + m.x * m.y) ** 2.0

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        NL = m.x**2 + m.x * m.y + m.x + m.y + 1

        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 0)
        self.assertEqual(len(repn.linear), 0)
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, NL * NL)
        assertExpressionsEqual(self, repn.to_expression(visitor), NL * NL)

        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(expr)

        NL = (m.x**2 + m.x * m.y) * (m.x**2 + m.x * m.y + m.x + m.y) + (
            m.x + m.y
        ) * (m.x**2 + m.x * m.y)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 1)
        self.assertEqual(repn.linear, {id(m.x): 2, id(m.y): 2})
        self.assertEqual(
            repn.quadratic,
            {(id(m.x), id(m.x)): 3, (id(m.x), id(m.y)): 4, (id(m.y), id(m.y)): 1},
        )
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            NL + 3 * m.x**2 + 4 * (m.x * m.y) + m.y**2 + 2 * m.x + 2 * m.y + 1,
        )

    def test_square_quadratic_wrt_y(self):
        m = build_test_model()
        expr = (1 + m.x + m.y + m.x**2 + m.x * m.y) ** 2.0

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        repn = visitor.walk_expression(expr)

        NL = m.x**2 + (1 + m.y) * m.x + (1 + m.y)

        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 0)
        self.assertEqual(len(repn.linear), 0)
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, NL * NL)
        assertExpressionsEqual(self, repn.to_expression(visitor), NL * NL)

        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(expr)

        NL = m.x**2 * (m.x**2 + (1 + m.y) * m.x) + ((1 + m.y) * m.x) * m.x**2
        QC = 1 + m.y + 1 + m.y + (1 + m.y) * (1 + m.y)
        LC = (1 + m.y) * (1 + m.y) + (1 + m.y) * (1 + m.y)
        CON = (1 + m.y) * (1 + m.y)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, (1 + m.y) * (1 + m.y))
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(
            self, repn.linear[id(m.x)], (1 + m.y) * (1 + m.y) + (1 + m.y) * (1 + m.y)
        )
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(
            self,
            repn.quadratic[id(m.x), id(m.x)],
            1 + m.y + 1 + m.y + (1 + m.y) * (1 + m.y),
        )
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), NL + QC * m.x**2 + LC * m.x + CON
        )

    def test_cube_linear(self):
        m = build_test_model()
        expr = (1 + m.x + m.y) ** 3

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        # cubic expansion not supported
        assertExpressionsEqual(self, repn.nonlinear, (m.x + m.y + 1) ** 3)
        assertExpressionsEqual(self, repn.to_expression(visitor), (m.x + m.y + 1) ** 3)

    def test_nonlinear_product_with_constant_terms(self):
        m = build_test_model()
        # test product of nonlinear expressions where one
        # multiplicand has constant of value 1
        expr = (1 + log(m.x)) * (log(m.x) + m.y**2)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.z])
        repn = visitor.walk_expression(expr)
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(
            self, repn.nonlinear, (log(m.x) + 1) * (log(m.x) + m.y**2)
        )
        assertExpressionsEqual(
            self, repn.to_expression(visitor), (log(m.x) + 1) * (log(m.x) + m.y**2)
        )

        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, {(id(m.y), id(m.y)): 1})
        assertExpressionsEqual(
            self, repn.nonlinear, log(m.x) * (log(m.x) + m.y**2) + log(m.x)
        )
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            log(m.x) * (log(m.x) + m.y**2) + log(m.x) + m.y**2,
        )

    def test_finalize_simplify_coefficients(self):
        m = build_test_model()
        expr = m.x + m.p * m.x**2 + 2 * m.y**2 - m.x - m.p * m.x**2 - m.p * m.z

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.z): m.z})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.z): 1})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 2 * m.y**2)
        self.assertEqual(repn.linear, {id(m.z): -1})
        self.assertEqual(repn.quadratic, None)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), -1 * m.z + 2 * m.y**2)

    def test_factor_multiplier_simplify_coefficients(self):
        m = build_test_model()
        expr = 2 * (m.x + m.x**2 + 2 * m.y**2 - m.x - m.x**2 - m.p * m.z)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        # this tests case where there are zeros in the `linear`
        # and `quadratic` dicts of the unfinalized repn
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.z): m.z})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.z): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertIsNone(repn.nonlinear)
        self.assertEqual(repn.quadratic, None)
        self.assertEqual(repn.linear, {id(m.z): -2})
        assertExpressionsEqual(self, repn.constant, (2 * m.y**2) * 2)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), -2 * m.z + (2 * m.y**2) * 2
        )

    def test_sum_nonlinear_custom_multiplier(self):
        m = build_test_model()
        expr = 2 * (1 + log(m.x)) + (2 * (m.y + m.y**2 + log(m.x)))

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 2 + 2 * (m.y + m.y**2))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, None)
        assertExpressionsEqual(self, repn.nonlinear, 2 * log(m.x) + 2 * log(m.x))
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            2 * log(m.x) + 2 * log(m.x) + 2 + 2 * (m.y + m.y**2),
        )

    def test_negation_linear(self):
        m = build_test_model()
        expr = -(2 + 3 * m.x + 5 * m.x * m.y)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, -2)
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(self, repn.linear[id(m.x)], (3 + 5 * m.y) * -1)
        self.assertIsNone(repn.quadratic)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), (3 + 5 * m.y) * -1 * m.x - 2
        )

    def test_negation_nonlinear_wrt_y_fix_z(self):
        m = build_test_model()
        m.z.fix(2)
        expr = -(
            2
            + 3 * m.x
            + 4 * m.y * m.z
            + 5 * m.x**2 * m.y
            + 6 * m.x * (m.z - 2)
            + m.z**2
            + m.z * log(m.x)
        )

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, (2 + 8 * m.y + 4) * -1)
        self.assertEqual(repn.linear, {id(m.x): -3})
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(self, repn.quadratic[(id(m.x), id(m.x))], -5 * m.y)
        assertExpressionsEqual(self, repn.nonlinear, 2 * log(m.x) * -1)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            +(-5 * m.y) * (m.x**2)
            + (-3) * m.x
            + (2 + 8 * m.y + 4) * (-1)
            + 2 * log(m.x) * -1,
        )

    def test_negation_product_linear_linear(self):
        m = build_test_model()
        expr = -(1 + 2 * m.x + 3 * m.y) * (4 + 5 * m.x + 6 * m.y * 7 * m.z)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(
            self, repn.constant, (1 + 3 * m.y) * (4 + 42 * m.y * m.z) * (-1)
        )
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(
            self,
            repn.linear[id(m.x)],
            ((4 + 42 * m.y * m.z) * 2 + (1 + 3 * m.y) * 5) * -1,
        )
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(self, repn.quadratic[id(m.x), id(m.x)], -10)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (
                -10 * m.x**2
                + ((4 + 42 * m.y * m.z) * 2 + (1 + 3 * m.y) * 5) * -1 * m.x
                + (1 + 3 * m.y) * (4 + 42 * m.y * m.z) * (-1)
            ),
        )

    def test_expanded_monomial_square_term(self):
        m = build_test_model()
        expr = m.x * m.x * m.p

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.z])
        # ensure overcomplication issues with standard repn
        # are not repeated by quadratic repn
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 1})
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), m.x**2)

    def test_sum_bilinear_terms_commute_product(self):
        m = build_test_model()
        expr = m.x * m.y + m.y * m.x

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.y)): 2})
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(self, repn.to_expression(visitor), 2 * (m.x * m.y))

    def test_sum_nonlinear(self):
        m = build_test_model()
        expr = (1 + log(m.x)) + (m.x + m.y + m.y**2 + log(m.x))

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        # tests special case of `repn.append` where multiplier
        # is 1 and both summands have a nonlinear term
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 1 + m.y + m.y**2)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, log(m.x) + log(m.x))
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            log(m.x) + log(m.x) + m.x + (1 + m.y) + m.y**2,
        )

    def test_product_linear_linear_0_nan(self):
        m = build_test_model()
        m.p.set_value(0)
        expr = (m.p + 0 * m.x) * (nan + nan * m.x)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear), 1)
        self.assertTrue(isnan(repn.linear[id(m.x)]))
        self.assertEqual(len(repn.quadratic), 1)
        self.assertTrue(isnan(repn.quadratic[id(m.x), id(m.x)]))
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), nan * m.x**2 + nan * m.x
        )

    def test_product_quadratic_quadratic_nan_0(self):

        m = build_test_model()
        m.p.set_value(0)
        expr = (nan + nan * m.x + nan * m.x**2) * (m.p + 0 * m.x + 0 * m.x**2)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        NL = (nan * m.x**2 + nan * m.x + nan) * (0 * m.x**2 + 0 * m.x)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear), 0)
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(self, repn.to_expression(visitor), NL)

        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(expr)

        NL = (nan * m.x**2) * (0 * m.x**2 + 0 * m.x) + nan * m.x * (0 * m.x**2)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertTrue(isnan(repn.constant))
        self.assertEqual(len(repn.linear), 1)
        self.assertTrue(isnan(repn.linear[id(m.x)]))
        self.assertEqual(len(repn.quadratic), 1)
        self.assertTrue(isnan(repn.quadratic[id(m.x), id(m.x)]))
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), NL + nan * m.x**2 + nan * m.x + nan
        )

    def test_product_quadratic_quadratic_0_nan(self):
        m = build_test_model()
        m.p.set_value(0)
        expr = (m.p + 0 * m.x + 0 * m.x**2) * (nan + nan * m.x + nan * m.x**2)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        NL = (0 * m.x**2 + 0 * m.x) * (nan * m.x**2 + nan * m.x + nan)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear), 0)
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(self, repn.to_expression(visitor), NL)

        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(expr)

        NL = (0 * m.x**2) * (nan * m.x**2 + nan * m.x) + 0 * m.x * (nan * m.x**2)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertTrue(isnan(repn.constant))
        self.assertEqual(len(repn.linear), 1)
        self.assertTrue(isnan(repn.linear[id(m.x)]))
        self.assertEqual(len(repn.quadratic), 1)
        self.assertTrue(isnan(repn.quadratic[id(m.x), id(m.x)]))
        assertExpressionsEqual(self, repn.nonlinear, NL)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), NL + nan * m.x**2 + nan * m.x + nan
        )

    def test_nary_sum_products(self):
        m = build_test_model()
        expr = (
            m.x**2 * (m.z - 1)
            + m.x * (m.y**4 + 0.8)
            - 5 * m.x * m.y * m.z
            + m.x * (m.y + 2)
        )

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(
            self, repn.linear[id(m.x)], m.y**4 + 0.8 + 5 * m.y * m.z * (-1) + (m.y + 2)
        )
        assertExpressionsEqual(self, repn.quadratic[id(m.x), id(m.x)], m.z - 1)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (m.z - 1) * m.x**2
            + (m.y**4 + 0.8 + 5 * m.y * m.z * (-1) + (m.y + 2)) * m.x,
        )

    def test_ternary_product_linear(self):
        m = build_test_model()
        expr = (1 + 2 * m.x) * (3 + 4 * m.y) * (5 + 6 * m.z)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.z): m.z})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.z): 1})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 5 * (3 + 4 * m.y))
        self.assertEqual(len(repn.linear), 2)
        assertExpressionsEqual(self, repn.linear[id(m.x)], 10 * (3 + 4 * m.y))
        assertExpressionsEqual(self, repn.linear[id(m.z)], 6 * (3 + 4 * m.y))
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(
            self, repn.quadratic[id(m.x), id(m.z)], 12 * (3 + 4 * m.y)
        )
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (
                12 * (3 + 4 * m.y) * (m.x * m.z)
                + 10 * (3 + 4 * m.y) * m.x
                + 6 * (3 + 4 * m.y) * m.z
                + 5 * (3 + 4 * m.y)
            ),
        )

    def test_noninteger_pow_linear(self):
        m = build_test_model()
        expr = (1 + 2 * m.x + 3 * m.y) ** 1.5

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, (2 * m.x + 1 + 3 * m.y) ** 1.5)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), (2 * m.x + 1 + 3 * m.y) ** 1.5
        )

    def test_variable_pow_linear(self):
        m = build_test_model()
        expr = (1 + 2 * m.x + 3 * m.y) ** (m.y)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIsNone(repn.quadratic)
        assertExpressionsEqual(self, repn.nonlinear, (2 * m.x + 1 + 3 * m.y) ** m.y)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), (2 * m.x + 1 + 3 * m.y) ** m.y
        )

    def test_pow_integer_fixed_var(self):
        m = build_test_model()
        m.z.fix(2)
        expr = (1 + 2 * m.x + 3 * m.y) ** (m.z)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, (1 + 3 * m.y) ** 2)
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(self, repn.linear[id(m.x)], 2 * (2 * (1 + 3 * m.y)))
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 4})
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            (4 * m.x**2 + (2 * (2 * (1 + 3 * m.y))) * m.x + (1 + 3 * m.y) ** 2),
        )

    def test_repr_parameterized_quadratic_repn(self):
        m = build_test_model()
        expr = 2 + m.x + m.x**2 + log(m.x)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        linear_dict = {id(m.x): 1}
        quad_dict = {(id(m.x), id(m.x)): 1}
        expected_repn_str = (
            "ParameterizedQuadraticRepn("
            "mult=1, "
            "const=2, "
            f"linear={linear_dict}, "
            f"quadratic={quad_dict}, "
            "nonlinear=log(x))"
        )
        self.assertEqual(repr(repn), expected_repn_str)
        self.assertEqual(str(repn), expected_repn_str)

    def test_product_var_linear_wrt_yz(self):
        """
        Test product of Var and quadratic expression.

        Aimed at testing what happens when one multiplicand
        of a product
        has a constant term of 0, and the other has a
        constant term that is an expression.
        """
        m = build_test_model()
        expr = m.x * (m.y + m.x * m.y + m.z)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 0)
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(self, repn.linear[id(m.x)], m.y + m.z)
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(self, repn.quadratic[id(m.x), id(m.x)], m.y)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), m.y * m.x**2 + (m.y + m.z) * m.x
        )

    def test_product_linear_var_wrt_yz(self):
        """
        Test product of Var and quadratic expression.

        Checks what happens when multiplicands of
        `test_product_var_linear` are swapped/commuted.
        """
        m = build_test_model()
        expr = (m.y + m.x * m.y + m.z) * m.x

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.y, m.z])
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 0)
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(self, repn.linear[id(m.x)], m.y + m.z)
        self.assertEqual(len(repn.quadratic), 1)
        assertExpressionsEqual(self, repn.quadratic[id(m.x), id(m.x)], m.y)
        self.assertIsNone(repn.nonlinear)
        assertExpressionsEqual(
            self, repn.to_expression(visitor), m.y * m.x**2 + (m.y + m.z) * m.x
        )

    def test_product_var_quadratic(self):
        """
        Test product of Var and quadratic expression.

        Aimed at testing what happens when one multiplicand
        of a product
        has a constant term of 0, and the other has a
        constant term that is an expression.
        """
        m = build_test_model()
        expr = m.x * (m.y + m.x * m.y + m.z)

        cfg = VisitorConfig()
        visitor = ParameterizedQuadraticRepnVisitor(**cfg, wrt=[m.z])
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(expr)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        assertExpressionsEqual(self, repn.constant, 0 * m.z)
        self.assertEqual(len(repn.linear), 1)
        assertExpressionsEqual(self, repn.linear[id(m.x)], m.z)
        self.assertEqual(len(repn.quadratic), 1)
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.y)): 1})
        assertExpressionsEqual(self, repn.nonlinear, m.x * (m.x * m.y))
        assertExpressionsEqual(
            self,
            repn.to_expression(visitor),
            m.x * m.y + m.z * m.x + 0 * m.z + m.x * (m.x * m.y),
        )

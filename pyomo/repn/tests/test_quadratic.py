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

from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest

from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.expr.numeric_expr import (
    LinearExpression,
    MonomialTermExpression,
    SumExpression,
)
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import InvalidNumber

from pyomo.environ import ConcreteModel, Var, Param, Any, log


class VisitorConfig(object):
    def __init__(self):
        self.subexpr = {}
        self.var_map = {}
        self.var_order = {}
        self.sorter = None

    def __iter__(self):
        return iter((self.subexpr, self.var_map, self.var_order, self.sorter))


class TestQuadratic(unittest.TestCase):
    def test_product(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = 2

        cfg = VisitorConfig()
        visitor = QuadraticRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 2)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, None)
        self.assertEqual(repn.nonlinear, None)

        e = 2 + 3 * m.x

        cfg = VisitorConfig()
        visitor = QuadraticRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 2)
        self.assertEqual(repn.linear, {id(m.x): 3})
        self.assertEqual(repn.quadratic, None)
        self.assertEqual(repn.nonlinear, None)

        e = 2 + 3 * m.x + 4 * m.x**2

        cfg = VisitorConfig()
        visitor = QuadraticRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 2)
        self.assertEqual(repn.linear, {id(m.x): 3})
        self.assertEqual(repn.quadratic, {(id(m.x), id(m.x)): 4})
        self.assertEqual(repn.nonlinear, None)

        e = (2 + 3 * m.x + 4 * m.x**2) * (5 + 6 * m.x + 7 * m.x**2)

        cfg = VisitorConfig()
        visitor = QuadraticRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        QE4 = SumExpression([4 * m.x**2])
        QE7 = SumExpression([7 * m.x**2])
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

        e = (2 + 3 * m.x + 4 * m.x**2) * (5 + 6 * m.x + 7 * m.x**2)

        cfg = VisitorConfig()
        visitor = QuadraticRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = False
        repn = visitor.walk_expression(e)

        NL = (4 * m.x**2 + 3 * m.x + 2) * (7 * m.x**2 + 6 * m.x + 5)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, None)
        assertExpressionsEqual(self, repn.nonlinear, NL)

        e = (1 + 2 * m.x + 3 * m.y) * (4 + 5 * m.x + 6 * m.y)

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)

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
        assertExpressionsEqual(self, repn.nonlinear, None)

        e = (m.x + m.y + log(m.x)) * m.x

        cfg = VisitorConfig()
        visitor = QuadraticRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = False
        repn = visitor.walk_expression(e)

        NL = (log(m.x) + (m.x + m.y)) * m.x

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, None)
        assertExpressionsEqual(self, repn.nonlinear, NL)

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

        e = m.x * (m.x + m.y + log(m.x) + 2)

        cfg = VisitorConfig()
        visitor = QuadraticRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = False
        repn = visitor.walk_expression(e)

        NL = m.x * (log(m.x) + (m.x + m.y) + 2)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, None)
        assertExpressionsEqual(self, repn.nonlinear, NL)

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

    def test_sum(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = SumExpression([])

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, None)
        self.assertEqual(repn.nonlinear, None)

        e += 5

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 5)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.quadratic, None)
        self.assertEqual(repn.nonlinear, None)

        e += m.x

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 5)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.quadratic, None)
        self.assertEqual(repn.nonlinear, None)

        e += m.y**2

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 5)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.quadratic, {(id(m.y), id(m.y)): 1})
        self.assertEqual(repn.nonlinear, None)

        e += m.y**3

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 5)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.quadratic, {(id(m.y), id(m.y)): 1})
        assertExpressionsEqual(self, repn.nonlinear, m.y**3)

        e += 2 * m.x**4

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 5)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.quadratic, {(id(m.y), id(m.y)): 1})
        assertExpressionsEqual(self, repn.nonlinear, m.y**3 + 2 * m.x**4)

        e += 2 * m.y

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 5)
        self.assertEqual(repn.linear, {id(m.x): 1, id(m.y): 2})
        self.assertEqual(repn.quadratic, {(id(m.y), id(m.y)): 1})
        assertExpressionsEqual(self, repn.nonlinear, m.y**3 + 2 * m.x**4)

        e += 3 * m.x * m.y

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 5)
        self.assertEqual(repn.linear, {id(m.x): 1, id(m.y): 2})
        self.assertEqual(repn.quadratic, {(id(m.y), id(m.y)): 1, (id(m.x), id(m.y)): 3})
        assertExpressionsEqual(self, repn.nonlinear, m.y**3 + 2 * m.x**4)

    def test_pow(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        # Check **{int}
        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression((1 + 3 * m.x + 4 * m.y) ** 2)
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

        # Check **{int}
        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(
            (1 + 3 * m.x + 4 * m.y) ** 2.0
        )
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

    def test_zero_elimination(self):
        m = ConcreteModel()
        m.x = Var(range(4))

        e = 0 * m.x[0] + 0 * m.x[1] * m.x[2] + 0 * log(m.x[3])

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
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
        self.assertEqual(repn.quadratic, None)
        self.assertEqual(repn.nonlinear, None)

        m.p = Param(mutable=True, within=Any, initialize=None)
        e = m.p * m.x[0] + m.p * m.x[1] * m.x[2] + m.p * log(m.x[3])

        cfg = VisitorConfig()
        repn = QuadraticRepnVisitor(*cfg).walk_expression(e)
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

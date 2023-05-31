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
from pyomo.core.expr.numeric_expr import LinearExpression, MonomialTermExpression
from pyomo.core.expr.current import Expr_if, inequality
from pyomo.repn.linear import LinearRepn, LinearRepnVisitor
from pyomo.repn.util import InvalidNumber

from pyomo.environ import ConcreteModel, Param, Var, Expression, ExternalFunction, cos

nan = float('nan')


class VisitorConfig(object):
    def __init__(self):
        self.subexpr = {}
        self.var_map = {}
        self.var_order = {}

    def __iter__(self):
        return iter((self.subexpr, self.var_map, self.var_order))


def sum_sq(args, fixed, fgh):
    f = sum(arg**2 for arg in args)
    g = [2 * arg for arg in args]
    h = None
    return f, g, h


class TestLinear(unittest.TestCase):
    def test_finalize(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()

        e = m.x + 2 * m.y - m.x - m.z

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y, id(m.z): m.z})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1, id(m.z): 2})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.y): 2, id(m.z): -1})
        self.assertEqual(repn.nonlinear, None)

        e *= 5

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x, id(m.y): m.y, id(m.z): m.z})
        self.assertEqual(cfg.var_order, {id(m.x): 0, id(m.y): 1, id(m.z): 2})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.y): 10, id(m.z): -5})
        self.assertEqual(repn.nonlinear, None)

        e = 5 * (m.y + m.z**2 + 3 * m.y**3)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.y): m.y, id(m.z): m.z})
        self.assertEqual(cfg.var_order, {id(m.y): 0, id(m.z): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.y): 5})
        assertExpressionsEqual(self, repn.nonlinear, (m.z**2 + 3 * m.y**3) * 5)

    def test_scalars(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(mutable=True, initialize=2)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(3)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 3)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression((-1) ** 0.5)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertStructuredAlmostEqual(repn.constant, InvalidNumber(1j))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(m.p)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 2)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p.set_value(None)
        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(m.p)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p.set_value(nan)
        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(m.p)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p.set_value(1j)
        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(m.p)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, InvalidNumber(1j))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(m.x)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(1)
        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(m.x)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 1)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(None)
        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(m.x)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(nan)
        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(m.x)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(1j)
        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(m.x)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(1j)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_npv(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=4)

        nested_expr = 1 / m.p
        pow_expr = m.p ** (0.5)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 1 / 4)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(pow_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 2)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p = 0

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(pow_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p = -1

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, -1)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(pow_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertStructuredAlmostEqual(repn.constant, InvalidNumber(1j))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_monomial(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(mutable=True, initialize=4)

        const_expr = 3 * m.x
        param_expr = m.p * m.x
        nested_expr = (1 / m.p) * m.x
        pow_expr = (m.p ** (0.5)) * m.x

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(const_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 3})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 4})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 1 / 4})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(pow_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)

        m.p = -1.0

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): -1})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): -1})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(pow_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertStructuredAlmostEqual(repn.linear, {id(m.x): InvalidNumber(1j)})
        self.assertEqual(repn.nonlinear, None)

        m.p = float('nan')

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertStructuredAlmostEqual(repn.linear, {id(m.x): InvalidNumber(nan)})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertStructuredAlmostEqual(repn.linear, {id(m.x): InvalidNumber(nan)})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(pow_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertStructuredAlmostEqual(repn.linear, {id(m.x): InvalidNumber(nan)})
        self.assertEqual(repn.nonlinear, None)

        m.p.set_value(None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertStructuredAlmostEqual(repn.linear, {id(m.x): InvalidNumber(nan)})
        self.assertEqual(repn.nonlinear, None)

        m.p.set_value(4)
        m.x.fix(10)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(const_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 30)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 40)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 2.5)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(pow_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 20)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p = float('nan')

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(pow_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p.set_value(None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(str(repn.constant), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p.set_value(0)
        m.x.fix(10)

        cfg = VisitorConfig()
        with LoggingIntercept() as LOG:
            repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertEqual(LOG.getvalue(), "")

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.set_value(nan)

        cfg = VisitorConfig()
        with LoggingIntercept() as LOG:
            repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertIn(
            "DEPRECATED: Encountered 0*nan in expression tree.", LOG.getvalue()
        )

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_linear(self):
        m = ConcreteModel()
        m.x = Var(range(3))
        m.p = Param(mutable=True, initialize=4)

        e = LinearExpression()

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        e += m.x[0]

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0]})
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x[0]): 1})
        self.assertEqual(repn.nonlinear, None)

        e += 2 * m.x[0]

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0]})
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x[0]): 3})
        self.assertEqual(repn.nonlinear, None)

        e += m.p * m.x[1]

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1]})
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x[0]): 3, id(m.x[1]): 4})
        self.assertEqual(repn.nonlinear, None)

        e += (m.p**0.5) * m.x[1]

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1]})
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x[0]): 3, id(m.x[1]): 6})
        self.assertEqual(repn.nonlinear, None)

        e += 10

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1]})
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 10)
        self.assertEqual(repn.linear, {id(m.x[0]): 3, id(m.x[1]): 6})
        self.assertEqual(repn.nonlinear, None)

        e += 10 * m.p

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1]})
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 50)
        self.assertEqual(repn.linear, {id(m.x[0]): 3, id(m.x[1]): 6})
        self.assertEqual(repn.nonlinear, None)

        m.p = -1

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1]})
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertStructuredAlmostEqual(
            repn.linear, {id(m.x[0]): 3, id(m.x[1]): InvalidNumber(-1 + 1j)}
        )
        self.assertEqual(repn.nonlinear, None)

        m.p = 0
        e += (1 / m.p) * m.x[1]

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1]})
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 10)
        self.assertStructuredAlmostEqual(
            repn.linear, {id(m.x[0]): 3, id(m.x[1]): InvalidNumber(nan)}
        )
        self.assertEqual(repn.nonlinear, None)

        m.x[0].fix(10)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[1]): m.x[1]})
        self.assertEqual(cfg.var_order, {id(m.x[1]): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 40)
        self.assertStructuredAlmostEqual(repn.linear, {id(m.x[1]): InvalidNumber(nan)})
        self.assertEqual(repn.nonlinear, None)

        m.x[1].fix(10)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertStructuredAlmostEqual(repn.constant, InvalidNumber(nan))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        # Test some edge cases

        e = LinearExpression()

        e += m.x[2] + (1 / m.p)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[2]): m.x[2]})
        self.assertEqual(cfg.var_order, {id(m.x[2]): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertStructuredAlmostEqual(repn.constant, InvalidNumber(nan))
        self.assertEqual(repn.linear, {id(m.x[2]): 1})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        cfg.var_map[id(m.x[2])] = m.x[2]
        cfg.var_map[id(m.x[0])] = m.x[0]
        cfg.var_order[id(m.x[2])] = 0
        cfg.var_order[id(m.x[0])] = 1
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x[2]): m.x[2], id(m.x[0]): m.x[0]})
        self.assertEqual(cfg.var_order, {id(m.x[2]): 0, id(m.x[0]): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertStructuredAlmostEqual(repn.constant, InvalidNumber(nan))
        self.assertEqual(repn.linear, {id(m.x[2]): 1})
        self.assertEqual(repn.nonlinear, None)

        e = LinearExpression()
        e += 0 * m.x[1]

        cfg = VisitorConfig()
        with LoggingIntercept() as LOG:
            repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(LOG.getvalue(), "")

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x[1].set_value(nan)

        cfg = VisitorConfig()
        with LoggingIntercept() as LOG:
            repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertIn(
            "DEPRECATED: Encountered 0*nan in expression tree.", LOG.getvalue()
        )

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_trig(self):
        m = ConcreteModel()
        m.x = Var()

        e = cos(m.x)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, cos(m.x))

        m.x.fix(0)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 1)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_named_expr(self):
        m = ConcreteModel()
        m.x = Var(range(3))
        m.e = Expression(expr=sum((i + 2) * m.x[i] for i in range(3)))

        e = m.e * 2

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(len(cfg.subexpr), 1)
        self.assertEqual(cfg.subexpr[id(m.e)][1].multiplier, 1)
        self.assertEqual(cfg.subexpr[id(m.e)][1].constant, 0)
        self.assertEqual(
            cfg.subexpr[id(m.e)][1].linear,
            {id(m.x[0]): 2, id(m.x[1]): 3, id(m.x[2]): 4},
        )
        self.assertEqual(cfg.subexpr[id(m.e)][1].nonlinear, None)

        self.assertEqual(
            cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1], id(m.x[2]): m.x[2]}
        )
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1, id(m.x[2]): 2})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x[0]): 4, id(m.x[1]): 6, id(m.x[2]): 8})
        self.assertEqual(repn.nonlinear, None)

        e = m.e * 2 + 3 * m.e

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(len(cfg.subexpr), 1)
        self.assertEqual(cfg.subexpr[id(m.e)][1].multiplier, 1)
        self.assertEqual(cfg.subexpr[id(m.e)][1].constant, 0)
        self.assertEqual(
            cfg.subexpr[id(m.e)][1].linear,
            {id(m.x[0]): 2, id(m.x[1]): 3, id(m.x[2]): 4},
        )
        self.assertEqual(cfg.subexpr[id(m.e)][1].nonlinear, None)

        self.assertEqual(
            cfg.var_map, {id(m.x[0]): m.x[0], id(m.x[1]): m.x[1], id(m.x[2]): m.x[2]}
        )
        self.assertEqual(cfg.var_order, {id(m.x[0]): 0, id(m.x[1]): 1, id(m.x[2]): 2})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x[0]): 10, id(m.x[1]): 15, id(m.x[2]): 20})
        self.assertEqual(repn.nonlinear, None)

        m = ConcreteModel()
        m.e = Expression(expr=10)

        e = m.e * 2

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(len(cfg.subexpr), 1)
        self.assertEqual(cfg.subexpr[id(m.e)][1], 10)

        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 20)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        e = m.e * 2 + 3 * m.e

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(len(cfg.subexpr), 1)
        self.assertEqual(cfg.subexpr[id(m.e)][1], 10)

        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 50)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_pow_expr(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(mutable=True, initialize=1)

        e = m.x**m.p

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)

        m.p = 0

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 1)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p = 2

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, m.x**2)

        m.x.fix(2)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 4)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p = 1 / 2
        m.x = -1

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertStructuredAlmostEqual(repn.constant, InvalidNumber(1j))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.unfix()
        e = (1 + m.x) ** 2

        cfg = VisitorConfig()
        visitor = LinearRepnVisitor(*cfg)
        visitor.max_exponential_expansion = 2
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, (m.x + 1) * (m.x + 1))

        cfg = VisitorConfig()
        visitor = LinearRepnVisitor(*cfg)
        visitor.max_exponential_expansion = 2
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 1)
        self.assertEqual(repn.linear, {id(m.x): 2})
        assertExpressionsEqual(self, repn.nonlinear, m.x * m.x)

    def test_product(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()

        e = (2 + 3 * m.x + 4 * m.x**2) * (5 + 6 * m.x + 7 * m.x**2)

        cfg = VisitorConfig()
        visitor = LinearRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = True
        repn = visitor.walk_expression(e)

        LE3 = MonomialTermExpression((3, m.x))
        LE6 = MonomialTermExpression((6, m.x))
        NL = (
            2 * (7 * m.x**2)
            + 4 * m.x**2 * (7 * m.x**2 + 6 * m.x + 5)
            + (LE3) * (7 * m.x**2 + LE6)
        )

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertAlmostEqual(repn.constant, 10)
        self.assertEqual(repn.linear, {id(m.x): 27})
        assertExpressionsEqual(self, repn.nonlinear, NL)

        m.x.fix(0)
        m.y.fix(nan)
        e = m.x * m.y

        cfg = VisitorConfig()
        visitor = LinearRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = True
        with LoggingIntercept() as LOG:
            repn = visitor.walk_expression(e)
        self.assertIn(
            'Encountered 0*InvalidNumber(nan) in expression tree.', LOG.getvalue()
        )

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertAlmostEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        e = m.x * (m.y + 2 + m.z)

        cfg = VisitorConfig()
        visitor = LinearRepnVisitor(*cfg)
        visitor.expand_nonlinear_products = True
        with LoggingIntercept() as LOG:
            repn = visitor.walk_expression(e)
        self.assertIn('Encountered 0*nan in expression tree.', LOG.getvalue())

        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.z): m.z})
        self.assertEqual(cfg.var_order, {id(m.z): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertAlmostEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_expr_if(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = Expr_if(m.y >= 5, m.x, m.x**2)
        f = Expr_if(m.y == 5, m.x, m.x**2)
        g = Expr_if(inequality(3, m.y, 5), m.x, m.x**2)

        m.y.fix(2)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, m.x**2)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(f)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, m.x**2)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(g)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, m.x**2)

        m.y.fix(5)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(f)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(g)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)

        m.y.fix(2)
        m.x.fix(3)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 9)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(f)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 9)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(g)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 9)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.y.fix(5)
        m.x.fix(6)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 6)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(f)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 6)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(g)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 6)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.y.fix(None)
        m.x.unfix()

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, m.x**2)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(f)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, m.x**2)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(g)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, m.x**2)

        m.y.unfix()

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.y): m.y, id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.y): 0, id(m.x): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(
            self, repn.nonlinear, Expr_if(IF=m.y >= 5, THEN=m.x, ELSE=m.x**2)
        )

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(f)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.y): m.y, id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.y): 0, id(m.x): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(
            self, repn.nonlinear, Expr_if(IF=m.y == 5, THEN=m.x, ELSE=m.x**2)
        )

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(g)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.y): m.y, id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.y): 0, id(m.x): 1})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(
            self,
            repn.nonlinear,
            Expr_if(IF=inequality(3, m.y, 5), THEN=m.x, ELSE=m.x**2),
        )

    def test_division(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = (2 * m.x + 1) / m.y
        m.y.fix(2)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 1 / 2)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)

        e = m.y / (m.x + 1)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        assertExpressionsEqual(self, repn.nonlinear, 2 / (m.x + 1))

    def test_negation(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()

        e = -(m.x + 2)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, -2)
        self.assertEqual(repn.linear, {id(m.x): -1})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(3)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, -5)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_external(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.sq = ExternalFunction(fgh=sum_sq)

        e = m.sq(2 / m.x, 2 * m.y)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIs(repn.nonlinear, e)

        m.x.fix(2)
        m.y.fix(3)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 37)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(0)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(e)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {})
        self.assertIs(repn.nonlinear, e)

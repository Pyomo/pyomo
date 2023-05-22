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

import math

import pyomo.common.unittest as unittest

from pyomo.repn.linear import LinearRepn, LinearRepnVisitor
from pyomo.environ import ConcreteModel, Param, Var


class VisitorConfig(object):
    def __init__(self):
        self.subexpr = {}
        self.var_map = {}
        self.var_order = {}

    def __iter__(self):
        return iter((self.subexpr, self.var_map, self.var_order))


class TestLinear(unittest.TestCase):
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
        self.assertRegex(str(repn.constant), r'InvalidNumber\(\([-+0-9.e]+\+1j\)\)')
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

    def test_monomial(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(mutable=True, initialize=2)

        const_expr = 3 * m.x
        param_expr = m.p * m.x
        nested_expr = (1 / m.p) * m.x

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
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {id(m.x): m.x})
        self.assertEqual(cfg.var_order, {id(m.x): 0})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 0)
        self.assertEqual(repn.linear, {id(m.x): 1 / 2})
        self.assertEqual(repn.nonlinear, None)

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
        self.assertEqual(repn.constant, 20)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertEqual(repn.constant, 5)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p = float('nan')

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(param_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertTrue(math.isnan(repn.constant))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        cfg = VisitorConfig()
        repn = LinearRepnVisitor(*cfg).walk_expression(nested_expr)
        self.assertEqual(cfg.subexpr, {})
        self.assertEqual(cfg.var_map, {})
        self.assertEqual(cfg.var_order, {})
        self.assertEqual(repn.multiplier, 1)
        self.assertTrue(math.isnan(repn.constant))
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

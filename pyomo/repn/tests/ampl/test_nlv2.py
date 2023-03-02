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
#

import pyomo.common.unittest as unittest

import io
import math
import os

import pyomo.repn.plugins.nl_writer as nl_writer

from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.expr.current import Expr_if, inequality
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
    ConcreteModel,
    Objective,
    Param,
    Var,
    log,
    ExternalFunction,
    Suffix,
    Constraint,
)


class INFO(object):
    def __init__(self, symbolic=False):
        if symbolic:
            self.template = nl_writer.text_nl_debug_template
        else:
            self.template = nl_writer.text_nl_template
        self.subexpression_cache = {}
        self.subexpression_order = []
        self.external_functions = {}
        self.var_map = nl_writer._deterministic_dict()
        self.used_named_expressions = set()
        self.symbolic_solver_labels = symbolic

        self.visitor = nl_writer.AMPLRepnVisitor(
            self.template,
            self.subexpression_cache,
            self.subexpression_order,
            self.external_functions,
            self.var_map,
            self.used_named_expressions,
            self.symbolic_solver_labels,
            True,
        )


class Test_AMPLRepnVisitor(unittest.TestCase):
    def test_divide(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=1)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**2 / m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\nv%s\nn2\n', [id(m.x)]))

        m.p = 2

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((4 / m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 2)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x / m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 0.5})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression(((4 * m.x) / m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((4 * (m.x + 2) / m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 4)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**2 / m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o2\nn0.5\no5\nv%s\nn2\n', [id(m.x)]))

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.x) / m.x, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o3\no43\nv%s\nv%s\n', [id(m.x), id(m.x)]))

    def test_errors_divide_by_0(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((1 / m.p, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 1/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x / m.p, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 1/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression(((3 * m.x) / m.p, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3*x/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((3 * (m.x + 2) / m.p, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3*(x + 2)/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**2 / m.p, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: x**2/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_pow(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=2)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\nv%s\nn2\n', [id(m.x)]))

        m.p = 1
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)

        m.p = 0
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 1)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_divide_by_0_mult_by_0(self):
        # Note: we may elect to deprecate this functionality in the future
        #
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p * (1 / m.p), None, None))
        self.assertIn(
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 1/p\n",
            LOG.getvalue(),
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression(((1 / m.p) * m.p, None, None))
        self.assertIn(
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 1/p\n",
            LOG.getvalue(),
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p * (m.x / m.p), None, None))
        self.assertIn(
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 1/p\n",
            LOG.getvalue(),
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression(
                (m.p * (3 * (m.x + 2) / m.p), None, None)
            )
        self.assertIn(
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3*(x + 2)/p\n",
            LOG.getvalue(),
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p * (m.x**2 / m.p), None, None))
        self.assertIn(
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: x**2/p\n",
            LOG.getvalue(),
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_divide_by_0_halt(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()

        nl_writer.HALT_ON_EVALUATION_ERROR, tmp = (
            True,
            nl_writer.HALT_ON_EVALUATION_ERROR,
        )
        try:
            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((1 / m.p, None, None))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 1/p\n",
            )

            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((m.x / m.p, None, None))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 1/p\n",
            )

            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((3 * (m.x + 2) / m.p, None, None))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(3, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 3*(x + 2)/p\n",
            )

            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((m.x**2 / m.p, None, None))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: x**2/p\n",
            )
        finally:
            nl_writer.HALT_ON_EVALUATION_ERROR = tmp

    def test_errors_negative_frac_pow(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=-1)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p ** (0.5), None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'pow(-1, 0.5)'\n"
            "\tmessage: Pyomo does not support complex numbers\n"
            "\texpression: p**0.5\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(0.5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p**m.x, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'pow(-1, 0.5)'\n"
            "\tmessage: Pyomo does not support complex numbers\n"
            "\texpression: p**x\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_unary_func(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.p), None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'log(0)'\n"
            "\tmessage: math domain error\n"
            "\texpression: log(p)\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_propagate_nan(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()
        m.y = Var()
        m.y.fix(1)

        expr = m.y**2 * m.x**2 * (((3 * m.x) / m.p) * m.x) / m.y

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3*x/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_pow(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** (0.5), None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\nv%s\nn0.5\n', [id(m.x)]))

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** (0.5), None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 2)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_abs(self):
        m = ConcreteModel()
        m.x = Var(initialize=-4)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((abs(m.x), None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o15\nv%s\n', [id(m.x)]))

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((abs(m.x), None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 4)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_unary_func(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.x), None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o43\nv%s\n', [id(m.x)]))

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.x), None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, math.log(4))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_expr_if_lessEq(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        m.y = Var(initialize=4)
        expr = Expr_if(m.x <= 4, m.x**2, m.y)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(
            repn.nonlinear,
            ('o35\no23\nv%s\nn4\no5\nv%s\nn2\nv%s\n', [id(m.x), id(m.x), id(m.y)]),
        )

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 16)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_expr_if_Eq(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        m.y = Var(initialize=4)
        expr = Expr_if(m.x == 4, m.x**2, m.y)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(
            repn.nonlinear,
            ('o35\no24\nv%s\nn4\no5\nv%s\nn2\nv%s\n', [id(m.x), id(m.x), id(m.y)]),
        )

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 16)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_expr_if_ranged(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        m.y = Var(initialize=4)
        expr = Expr_if(inequality(1, m.x, 4), m.x**2, m.y)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(
            repn.nonlinear,
            (
                'o35\no21\no23\nn1\nv%s\no23\nv%s\nn4\no5\nv%s\nn2\nv%s\n',
                [id(m.x), id(m.x), id(m.x), id(m.y)],
            ),
        )

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 16)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(0)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertEqual(repn.nonlinear, None)

    def test_custom_named_expression(self):
        class CustomExpression(ScalarExpression):
            pass

        m = ConcreteModel()
        m.x = Var()
        m.e = CustomExpression()
        m.e.expr = m.x + 3

        expr = m.e + m.e
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 6)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)

        self.assertEqual(len(info.subexpression_cache), 1)
        obj, repn, info = info.subexpression_cache[id(m.e)]
        self.assertIs(obj, m.e)
        self.assertEqual(repn.nl, ('v%s\n', (id(m.e),)))
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 3)
        self.assertEqual(repn.linear, [(id(m.x), 1)])
        self.assertEqual(repn.nonlinear, None)
        self.assertEqual(info, [None, None, False])

    def test_nested_operator_zero_arg(self):
        # This tests an error encountered when developing the nlv2
        # writer where var ids were being dropped then the second
        # argument in a binary operator was 0.  The original case was
        # for expr**p where p was a variable fixed to 0.  However, since
        # then, _handle_pow_operator contains special handling for **0
        # and **1.
        m = ConcreteModel()
        m.x = Var()
        m.p = Param(initialize=0, mutable=True)
        expr = (1 / m.x) == m.p

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o24\no3\nn1\nv%s\nn0\n', [id(m.x)]))


class Test_NLWriter(unittest.TestCase):
    def test_external_function_str_args(self):
        m = ConcreteModel()
        m.x = Var()
        m.e = ExternalFunction(library='tmp', function='test')
        m.o = Objective(expr=m.e(m.x, 'str'))

        # Test explicit newline translation
        OUT = io.StringIO(newline='\r\n')
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertIn(
            "Writing NL file containing string arguments to a "
            "text output stream with line endings other than '\\n' ",
            LOG.getvalue(),
        )

        # Test system-dependent newline translation
        with TempfileManager:
            fname = TempfileManager.create_tempfile()
            with open(fname, 'w') as OUT:
                with LoggingIntercept() as LOG:
                    nl_writer.NLWriter().write(m, OUT)
        if os.linesep == '\n':
            self.assertEqual(LOG.getvalue(), "")
        else:
            self.assertIn(
                "Writing NL file containing string arguments to a "
                "text output stream with line endings other than '\\n' ",
                LOG.getvalue(),
            )

        # Test objects lacking 'tell':
        r, w = os.pipe()
        try:
            OUT = os.fdopen(w, 'w')
            with LoggingIntercept() as LOG:
                nl_writer.NLWriter().write(m, OUT)
            if os.linesep == '\n':
                self.assertEqual(LOG.getvalue(), "")
            else:
                self.assertIn(
                    "Writing NL file containing string arguments to a "
                    "text output stream that does not support tell()",
                    LOG.getvalue(),
                )
        finally:
            OUT.close()
            os.close(r)

    def test_suffix_warning_new_components(self):
        m = ConcreteModel()
        m.junk = Suffix(direction=Suffix.EXPORT)
        m.x = Var()
        m.y = Var()
        m.z = Var([1, 2, 3])
        m.o = Objective(expr=m.x + m.z[2])
        m.c = Constraint(expr=m.y <= 0)
        m.c.deactivate()

        @m.Constraint([1, 2, 3])
        def d(m, i):
            return m.z[i] <= 0

        m.d.deactivate()
        m.d[2].activate()
        m.junk[m.x] = 1

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(LOG.getvalue(), "")

        m.junk[m.y] = 1
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            "model contains export suffix 'junk' that contains 1 component "
            "keys that are not exported as part of the NL file.  Skipping.\n",
            LOG.getvalue(),
        )

        m.junk[m.z] = 1
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            "model contains export suffix 'junk' that contains 3 component "
            "keys that are not exported as part of the NL file.  Skipping.\n",
            LOG.getvalue(),
        )

        m.junk[m.c] = 2
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            "model contains export suffix 'junk' that contains 4 component "
            "keys that are not exported as part of the NL file.  Skipping.\n",
            LOG.getvalue(),
        )

        m.junk[m.d] = 2
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            "model contains export suffix 'junk' that contains 6 component "
            "keys that are not exported as part of the NL file.  Skipping.\n",
            LOG.getvalue(),
        )

        m.junk[5] = 5
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            "model contains export suffix 'junk' that contains 6 component "
            "keys that are not exported as part of the NL file.  Skipping.\n"
            "model contains export suffix 'junk' that contains 1 "
            "keys that are not Var, Constraint, Objective, or the model.  "
            "Skipping.\n",
            LOG.getvalue(),
        )

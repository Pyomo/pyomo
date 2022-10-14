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

import math

import pyomo.repn.plugins.nl_writer as nl_writer

from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.current import Expr_if, inequality
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import ConcreteModel, Param, Var, log


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
            repn = info.visitor.walk_expression(((4*m.x) / m.p, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((4*(m.x + 2) / m.p, None, None))
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
        self.assertEqual(repn.nonlinear,('o2\nn0.5\no5\nv%s\nn2\n', [id(m.x)]))

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.x) / m.x, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear,
                         ('o3\no43\nv%s\nv%s\n', [id(m.x), id(m.x)]))

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
            "\texpression: 1/p\n"
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
            "\texpression: 1/p\n"
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)


        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression(((3*m.x) / m.p, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3*x/p\n"
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((3*(m.x + 2) / m.p, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3*(x + 2)/p\n"
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
            "\texpression: x**2/p\n"
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertTrue(math.isnan(repn.const))
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
            repn = info.visitor.walk_expression((m.p*(1 / m.p), None, None))
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
            repn = info.visitor.walk_expression(((1 / m.p)*m.p, None, None))
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
            repn = info.visitor.walk_expression((m.p*(m.x / m.p), None, None))
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
            repn = info.visitor.walk_expression((m.p*(3*(m.x + 2) / m.p), None, None))
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
            repn = info.visitor.walk_expression((m.p*(m.x**2 / m.p), None, None))
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

        nl_writer.HALT_ON_EVALUATION_ERROR, tmp \
            = True, nl_writer.HALT_ON_EVALUATION_ERROR
        try:
            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((1 / m.p, None, None))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 1/p\n"
            )

            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((m.x / m.p, None, None))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 1/p\n"
            )

            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((3*(m.x + 2) / m.p, None, None))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(3, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 3*(x + 2)/p\n"
        )

            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((m.x**2 / m.p, None, None))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: x**2/p\n"
        )
        finally:
            nl_writer.HALT_ON_EVALUATION_ERROR = tmp

    def test_errors_negative_frac_pow(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=-1)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p**(0.5), None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'pow(-1, 0.5)'\n"
            "\tmessage: Pyomo does not support complex numbers\n"
            "\texpression: p**0.5\n"
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
            "\texpression: p**x\n"
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
            "\texpression: log(p)\n"
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

        expr = m.y**2 * m.x**2 * (((3*m.x)/m.p) * m.x ) / m.y

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3*x/p\n"
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
            repn = info.visitor.walk_expression((m.x**(0.5), None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\nv%s\nn0.5\n', [id(m.x)]))

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**(0.5), None, None))
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
        self.assertEqual(repn.nonlinear,
                         ('o35\no23\nv%s\nn4\no5\nv%s\nn2\nv%s\n',
                          [id(m.x), id(m.x), id(m.y)]))

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
        self.assertEqual(repn.linear, {id(m.y):1})
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
        self.assertEqual(repn.nonlinear,
                         ('o35\no24\nv%s\nn4\no5\nv%s\nn2\nv%s\n',
                          [id(m.x), id(m.x), id(m.y)]))

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
        self.assertEqual(repn.linear, {id(m.y):1})
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
            ('o35\no21\no23\nn1\nv%s\no23\nv%s\nn4\no5\nv%s\nn2\nv%s\n',
             [id(m.x), id(m.x), id(m.x), id(m.y)]))

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
        self.assertEqual(repn.linear, {id(m.y):1})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(0)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y):1})
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

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
from pyomo.environ import ConcreteModel, Param, Var, log

# Note that it is important to use the NL writer's NAN
#nan = nl_writer.nan

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
        )

class Test_AMPLRepnVisitor(unittest.TestCase):

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
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p*(1 / m.p), None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 1/p\n"
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p*(m.x / m.p), None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 1/p\n"
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p*(3*(m.x + 2) / m.p), None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3*(x + 2)/p\n"
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p*(m.x**2 / m.p), None, None))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: x**2/p\n"
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

    def test_eval_const_pow(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=4)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p**(0.5), None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 2)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_const_unary_func(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=4)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.p), None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, math.log(4))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

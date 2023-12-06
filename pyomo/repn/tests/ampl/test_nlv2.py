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
import logging
import math
import os
import re

import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import report_timing
from pyomo.core.expr import Expr_if, inequality, LinearExpression
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
    Any,
    ConcreteModel,
    Objective,
    Param,
    Var,
    log,
    ExternalFunction,
    Suffix,
    Constraint,
    Expression,
)
import pyomo.environ as pyo

_invalid_1j = r'InvalidNumber\((\([-+0-9.e]+\+)?1j\)?\)'


class INFO(object):
    def __init__(self, symbolic=False):
        if symbolic:
            self.template = nl_writer.text_nl_debug_template
        else:
            self.template = nl_writer.text_nl_template
        self.subexpression_cache = {}
        self.subexpression_order = []
        self.external_functions = {}
        self.var_map = {}
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
            None,
        )

    def __enter__(self):
        assert nl_writer.AMPLRepn.ActiveVisitor is None
        nl_writer.AMPLRepn.ActiveVisitor = self.visitor
        return self

    def __exit__(self, exc_type, exc_value, tb):
        assert nl_writer.AMPLRepn.ActiveVisitor is self.visitor
        nl_writer.AMPLRepn.ActiveVisitor = None


class Test_AMPLRepnVisitor(unittest.TestCase):
    def test_divide(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=1)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**2 / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\n%s\nn2\n', [id(m.x)]))

        m.p = 2

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((4 / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 2)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 0.5})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression(((4 * m.x) / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((4 * (m.x + 2) / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 4)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**2 / m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o2\nn0.5\no5\n%s\nn2\n', [id(m.x)]))

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.x) / m.x, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o3\no43\n%s\n%s\n', [id(m.x), id(m.x)]))

    def test_errors_divide_by_0(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((1 / m.p, None, None, 1))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 1/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x / m.p, None, None, 1))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 1/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression(((3 * m.x) / m.p, None, None, 1))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((3 * (m.x + 2) / m.p, None, None, 1))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3*(x + 2)/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**2 / m.p, None, None, 1))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(1, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: x**2/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_pow(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=2)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\n%s\nn2\n', [id(m.x)]))

        m.p = 1
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**m.p, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): 1})
        self.assertEqual(repn.nonlinear, None)

        m.p = 0
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x**m.p, None, None, 1))
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
            repn = info.visitor.walk_expression((m.p * (1 / m.p), None, None, 1))
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
            repn = info.visitor.walk_expression(((1 / m.p) * m.p, None, None, 1))
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
            repn = info.visitor.walk_expression((m.p * (m.x / m.p), None, None, 1))
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
                (m.p * (3 * (m.x + 2) / m.p), None, None, 1)
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
            repn = info.visitor.walk_expression((m.p * (m.x**2 / m.p), None, None, 1))
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

        repn_util.HALT_ON_EVALUATION_ERROR, tmp = (
            True,
            repn_util.HALT_ON_EVALUATION_ERROR,
        )
        try:
            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((1 / m.p, None, None, 1))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 1/p\n",
            )

            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((m.x / m.p, None, None, 1))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 1/p\n",
            )

            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((3 * (m.x + 2) / m.p, None, None, 1))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(3, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 3*(x + 2)/p\n",
            )

            info = INFO()
            with LoggingIntercept() as LOG, self.assertRaises(ZeroDivisionError):
                info.visitor.walk_expression((m.x**2 / m.p, None, None, 1))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: x**2/p\n",
            )
        finally:
            repn_util.HALT_ON_EVALUATION_ERROR = tmp

    def test_errors_negative_frac_pow(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=-1)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p ** (0.5), None, None, 1))
        self.assertEqual(
            LOG.getvalue(),
            "Complex number returned from expression\n"
            "\tmessage: Pyomo AMPLRepnVisitor does not support complex numbers\n"
            "\texpression: p**0.5\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertRegex(str(repn.const), _invalid_1j)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(0.5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p**m.x, None, None, 1))
        self.assertEqual(
            LOG.getvalue(),
            "Complex number returned from expression\n"
            "\tmessage: Pyomo AMPLRepnVisitor does not support complex numbers\n"
            "\texpression: p**x\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertRegex(str(repn.const), _invalid_1j)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_unary_func(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.p), None, None, 1))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'log(0)'\n"
            "\tmessage: math domain error\n"
            "\texpression: log(p)\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_errors_propagate_nan(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=0, domain=Any)
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.y.fix(1)

        expr = m.y**2 * m.x**2 * (((3 * m.x) / m.p) * m.x) / m.y

        with LoggingIntercept() as LOG, INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(
            LOG.getvalue(),
            "Exception encountered evaluating expression 'div(3, 0)'\n"
            "\tmessage: division by zero\n"
            "\texpression: 3/p\n",
        )
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.y.fix(None)
        expr = log(m.y) + 3
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        expr = 3 * m.y
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, InvalidNumber(None))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p.value = None
        expr = 5 * (m.p * m.x + 2 * m.z)
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.z): 10, id(m.x): InvalidNumber(None)})
        self.assertEqual(repn.nonlinear, None)

        expr = m.y * m.x
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): InvalidNumber(None)})
        self.assertEqual(repn.nonlinear, None)

        m.z = Var([1, 2, 3, 4], initialize=lambda m, i: i - 1)
        m.z[1].fix(None)
        expr = m.z[1] - ((m.z[2] * m.z[3]) * m.z[4])
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, InvalidNumber(None))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear[0], 'o16\no2\no2\n%s\n%s\n%s\n')
        self.assertEqual(repn.nonlinear[1], [id(m.z[2]), id(m.z[3]), id(m.z[4])])

        m.z[3].fix(float('nan'))
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, InvalidNumber(None))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

    def test_linearexpression_npv(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)
        m.y = Var(initialize=4)
        m.z = Var(initialize=4)
        m.p = Param(initialize=5, mutable=True)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression(
                (
                    LinearExpression(
                        args=[1, m.p, m.p * m.x, (m.p + 2) * m.y, 3 * m.z, m.p * m.z]
                    ),
                    None,
                    None,
                    1,
                )
            )
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 6)
        self.assertEqual(repn.linear, {id(m.x): 5, id(m.y): 7, id(m.z): 8})
        self.assertEqual(repn.nonlinear, None)

    def test_eval_pow(self):
        m = ConcreteModel()
        m.x = Var(initialize=4)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** (0.5), None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o5\n%s\nn0.5\n', [id(m.x)]))

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.x ** (0.5), None, None, 1))
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
            repn = info.visitor.walk_expression((abs(m.x), None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o15\n%s\n', [id(m.x)]))

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((abs(m.x), None, None, 1))
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
            repn = info.visitor.walk_expression((log(m.x), None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o43\n%s\n', [id(m.x)]))

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((log(m.x), None, None, 1))
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
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(
            repn.nonlinear,
            ('o35\no23\n%s\nn4\no5\n%s\nn2\n%s\n', [id(m.x), id(m.x), id(m.y)]),
        )

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 16)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
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
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(
            repn.nonlinear,
            ('o35\no24\n%s\nn4\no5\n%s\nn2\n%s\n', [id(m.x), id(m.x), id(m.y)]),
        )

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 16)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
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
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(
            repn.nonlinear,
            (
                'o35\no21\no23\nn1\n%s\no23\n%s\nn4\no5\n%s\nn2\n%s\n',
                [id(m.x), id(m.x), id(m.x), id(m.y)],
            ),
        )

        m.x.fix()
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 16)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(5)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.y): 1})
        self.assertEqual(repn.nonlinear, None)

        m.x.fix(0)
        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None, 1))
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
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 6)
        self.assertEqual(repn.linear, {id(m.x): 2})
        self.assertEqual(repn.nonlinear, None)

        self.assertEqual(len(info.subexpression_cache), 1)
        obj, repn, info = info.subexpression_cache[id(m.e)]
        self.assertIs(obj, m.e)
        self.assertEqual(repn.nl, ('%s\n', (id(m.e),)))
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 3)
        self.assertEqual(repn.linear, {id(m.x): 1})
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
            repn = info.visitor.walk_expression((expr, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o24\no3\nn1\n%s\nn0\n', [id(m.x)]))

    def test_duplicate_shared_linear_expressions(self):
        # This tests an issue where AMPLRepn.duplicate() was not copying
        # the linear dict, allowing certain operations (like finalizing
        # a bare expression multiplied by something other than 1) to
        # change the compiled shared expression
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.e = Expression(expr=2 * m.x + 3 * m.y)

        expr1 = 10 * m.e
        expr2 = m.e + 100 * m.x + 100 * m.y

        info = INFO()
        with LoggingIntercept() as LOG:
            repn1 = info.visitor.walk_expression((expr1, None, None, 1))
            repn2 = info.visitor.walk_expression((expr2, None, None, 1))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn1.nl, None)
        self.assertEqual(repn1.mult, 1)
        self.assertEqual(repn1.const, 0)
        self.assertEqual(repn1.linear, {id(m.x): 20, id(m.y): 30})
        self.assertEqual(repn1.nonlinear, None)

        self.assertEqual(repn2.nl, None)
        self.assertEqual(repn2.mult, 1)
        self.assertEqual(repn2.const, 0)
        self.assertEqual(repn2.linear, {id(m.x): 102, id(m.y): 103})
        self.assertEqual(repn2.nonlinear, None)


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
        with LoggingIntercept(level=logging.DEBUG) as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            "model contains export suffix 'junk' that contains 1 component "
            "keys that are not exported as part of the NL file.  Skipping.\n"
            "Skipped component keys:\n\ty\n",
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
        with LoggingIntercept(level=logging.DEBUG) as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            "model contains export suffix 'junk' that contains 3 component "
            "keys that are not exported as part of the NL file.  Skipping.\n"
            "Skipped component keys:\n\ty\n\tz[1]\n\tz[3]\n",
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
        with LoggingIntercept(level=logging.DEBUG) as LOG:
            nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            "model contains export suffix 'junk' that contains 6 component "
            "keys that are not exported as part of the NL file.  Skipping.\n"
            "Skipped component keys:\n\tc\n\td[1]\n\td[3]\n\ty\n\tz[1]\n\tz[3]\n"
            "model contains export suffix 'junk' that contains 1 keys that "
            "are not Var, Constraint, Objective, or the model.  Skipping.\n"
            "Skipped component keys:\n\t5\n",
            LOG.getvalue(),
        )

    def test_log_timing(self):
        # This tests an error possibly reported by #2810
        m = ConcreteModel()
        m.x = Var(range(6))
        m.x[0].domain = pyo.Binary
        m.x[1].domain = pyo.Integers
        m.x[2].domain = pyo.Integers
        m.p = Param(initialize=5, mutable=True)
        m.o1 = Objective([1, 2], rule=lambda m, i: 1)
        m.o2 = Objective(expr=m.x[1] * m.x[2])
        m.c1 = Constraint([1, 2], rule=lambda m, i: sum(m.x.values()) == 1)
        m.c2 = Constraint(expr=m.p * m.x[1] ** 2 + m.x[2] ** 3 <= 100)

        self.maxDiff = None
        OUT = io.StringIO()
        with capture_output() as LOG:
            with report_timing(level=logging.DEBUG):
                nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            """      [+   #.##] Initialized column order
      [+   #.##] Collected suffixes
      [+   #.##] Objective o1
      [+   #.##] Objective o2
      [+   #.##] Constraint c1
      [+   #.##] Constraint c2
      [+   #.##] Categorized model variables: 14 nnz
      [+   #.##] Set row / column ordering: 6 var [3, 1, 2 R/B/Z], 3 con [2, 1 L/NL]
      [+   #.##] Generated row/col labels & comments
      [+   #.##] Wrote NL stream
      [    #.##] Generated NL representation
""",
            re.sub(r'\d\.\d\d\]', '#.##]', LOG.getvalue()),
        )

    def test_linear_constraint_npv_const(self):
        # This tests an error possibly reported by #2810
        m = ConcreteModel()
        m.x = Var([1, 2])
        m.p = Param(initialize=5, mutable=True)
        m.o = Objective(expr=1)
        m.c = Constraint(
            expr=LinearExpression([m.p**2, 5 * m.x[1], 10 * m.x[2]]) <= 0
        )

        OUT = io.StringIO()
        nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 2 1 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 0 	# nonzeros in Jacobian, obj. gradient
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0
n0
O0 0
n1.0
x0
r
1 -25
b
3
3
k1
1
J0 2
0 5
1 10
""",
                OUT.getvalue(),
            )
        )

    def test_indexed_sos_constraints(self):
        # This tests the example from issue #2827
        m = pyo.ConcreteModel()
        m.A = pyo.Set(initialize=[1])
        m.B = pyo.Set(initialize=[1, 2, 3])
        m.C = pyo.Set(initialize=[1])

        m.param_cx = pyo.Param(m.A, initialize={1: 1})
        m.param_cy = pyo.Param(m.B, initialize={1: 2, 2: 3, 3: 1})

        m.x = pyo.Var(m.A, domain=pyo.NonNegativeReals, bounds=(0, 40))
        m.y = pyo.Var(m.B, domain=pyo.NonNegativeIntegers)

        @m.Objective()
        def OBJ(m):
            return sum(m.param_cx[a] * m.x[a] for a in m.A) + sum(
                m.param_cy[b] * m.y[b] for b in m.B
            )

        m.y[3].bounds = (2, 3)

        m.mysos = pyo.SOSConstraint(
            m.C, var=m.y, sos=1, index={1: [2, 3]}, weights={2: 25.0, 3: 18.0}
        )

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT, symbolic_solver_labels=True)
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0        # problem unknown
 4 0 1 0 0      # vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0    # nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0    # network constraints: nonlinear, linear
 0 0 0  # nonlinear vars in constraints, objectives, both
 0 0 0 1        # linear network variables; functions; arith, flags
 0 3 0 0 0      # discrete variables: binary, integer, nonlinear (b,c,o)
 0 4    # nonzeros in Jacobian, obj. gradient
 3 4    # max name lengths: constraints, variables
 0 0 0 0 0      # common exprs: b,c,o,c1,o1
S0 2 sosno
2 1
3 1
S0 2 ref
2 25.0
3 18.0
O0 0    #OBJ
n0
x0      # initial guess
r       #0 ranges (rhs's)
b       #4 bounds (on variables)
0 0 40  #x[1]
2 0     #y[1]
2 0     #y[2]
0 2 3   #y[3]
k3      #intermediate Jacobian column lengths
0
0
0
G0 4    #OBJ
0 1
1 2
2 3
3 1
""",
                OUT.getvalue(),
            )
        )

    @unittest.skipUnless(numpy_available, "test requires numpy")
    def test_nonfloat_constants(self):
        import pyomo.environ as pyo

        v = numpy.array([[8], [3], [6], [11]])
        w = numpy.array([[5], [7], [4], [3]])
        # Create model
        m = pyo.ConcreteModel()
        m.I = pyo.Set(initialize=range(4))
        # Variables: note initialization with non-numeric value
        m.zero = pyo.Param(initialize=numpy.array([0]), mutable=True)
        m.one = pyo.Param(initialize=numpy.array([1]), mutable=True)
        m.x = pyo.Var(m.I, bounds=(m.zero, m.one), domain=pyo.Integers, initialize=True)
        # Params: initialize with 1-member ndarrays
        m.limit = pyo.Param(initialize=numpy.array([14]), mutable=True)
        m.v = pyo.Param(m.I, initialize=v, mutable=True)
        m.w = pyo.Param(m.I, initialize=w, mutable=True)
        # Objective: note use of numpy in coefficients
        m.value = pyo.Objective(expr=pyo.sum_product(m.v, m.x), sense=pyo.maximize)
        # Constraint: note use of numpy in coefficients and RHS
        m.weight = pyo.Constraint(expr=pyo.sum_product(m.w, m.x) <= m.limit)

        OUT = io.StringIO()
        ROW = io.StringIO()
        COL = io.StringIO()
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT, ROW, COL, symbolic_solver_labels=True)
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(ROW.getvalue(), "weight\nvalue\n")
        self.assertEqual(COL.getvalue(), "x[0]\nx[1]\nx[2]\nx[3]\n")
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0       #problem unknown
 4 1 1 0 0     #vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0   #nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0   #network constraints: nonlinear, linear
 0 0 0 #nonlinear vars in constraints, objectives, both
 0 0 0 1       #linear network variables; functions; arith, flags
 0 4 0 0 0     #discrete variables: binary, integer, nonlinear (b,c,o)
 4 4   #nonzeros in Jacobian, obj. gradient
 6 4   #max name lengths: constraints, variables
 0 0 0 0 0     #common exprs: b,c,o,c1,o1
C0     #weight
n0
O0 1   #value
n0
x4     #initial guess
0 1.0  #x[0]
1 1.0  #x[1]
2 1.0  #x[2]
3 1.0  #x[3]
r      #1 ranges (rhs's)
1 14.0 #weight
b      #4 bounds (on variables)
0 0 1  #x[0]
0 0 1  #x[1]
0 0 1  #x[2]
0 0 1  #x[3]
k3     #intermediate Jacobian column lengths
1
2
3
J0 4   #weight
0 5
1 7
2 4
3 3
G0 4   #value
0 8
1 3
2 6
3 11
""",
                OUT.getvalue(),
            )
        )

    def test_presolve_lower_triangular(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] == 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), "")

        self.assertEqual(
            nlinfo.eliminated_vars,
            [
                (m.x[3], nl_writer.AMPLRepn(-4.0, {}, None)),
                (m.x[1], nl_writer.AMPLRepn(4.0, {}, None)),
                (m.x[2], nl_writer.AMPLRepn(3.0, {}, None)),
                (m.x[0], nl_writer.AMPLRepn(5.0, {}, None)),
            ],
        )
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 1 0 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 0 1 	# nonzeros in Jacobian, obj. gradient
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
O0 0
n-4.0
x0
r
b
0 -10 10
k0
G0 1
0 1
""",
                OUT.getvalue(),
            )
        )

    def test_presolve_lower_triangular_fixed(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        # m.c.add(m.x[0] == 5)
        m.x[0].bounds = (5, 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), "")

        self.assertEqual(
            nlinfo.eliminated_vars,
            [
                (m.x[3], nl_writer.AMPLRepn(-4.0, {}, None)),
                (m.x[1], nl_writer.AMPLRepn(4.0, {}, None)),
                (m.x[2], nl_writer.AMPLRepn(3.0, {}, None)),
                (m.x[0], nl_writer.AMPLRepn(5.0, {}, None)),
            ],
        )
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 1 0 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 0 1 	# nonzeros in Jacobian, obj. gradient
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
O0 0
n-4.0
x0
r
b
0 -10 10
k0
G0 1
0 1
""",
                OUT.getvalue(),
            )
        )

    def test_presolve_lower_triangular_implied(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(6), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] == m.x[5])
        m.x[0].bounds = (None, 5)
        m.x[5].bounds = (5, None)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), "")

        self.assertEqual(
            nlinfo.eliminated_vars,
            [
                (m.x[1], nl_writer.AMPLRepn(4.0, {}, None)),
                (m.x[5], nl_writer.AMPLRepn(5.0, {}, None)),
                (m.x[3], nl_writer.AMPLRepn(-4.0, {}, None)),
                (m.x[2], nl_writer.AMPLRepn(3.0, {}, None)),
                (m.x[0], nl_writer.AMPLRepn(5.0, {}, None)),
            ],
        )
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 1 0 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 0 1 	# nonzeros in Jacobian, obj. gradient
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
O0 0
n-4.0
x0
r
b
0 -10 10
k0
G0 1
0 1
""",
                OUT.getvalue(),
            )
        )

    def test_presolve_almost_lower_triangular(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] + 2 * m.x[4] == 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), "")

        self.assertEqual(
            nlinfo.eliminated_vars,
            [
                (m.x[4], nl_writer.AMPLRepn(-12, {id(m.x[1]): 3}, None)),
                (m.x[3], nl_writer.AMPLRepn(-72, {id(m.x[1]): 17}, None)),
                (m.x[2], nl_writer.AMPLRepn(-13, {id(m.x[1]): 4}, None)),
                (m.x[0], nl_writer.AMPLRepn(29, {id(m.x[1]): -6}, None)),
            ],
        )
        # Note: bounds on x[1] are:
        #   min(22/3, 82/17, 23/4, -39/-6) == 4.823529411764706
        #   max(2/3, 62/17, 3/4, -19/-6) == 3.6470588235294117
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 1 0 1 0 0 	# vars, constraints, objectives, ranges, eqns
 0 0 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 0 0 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 0 1 	# nonzeros in Jacobian, obj. gradient
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
O0 0
n-84.0
x0
r
b
0 3.6470588235294117 4.823529411764706
k0
G0 1
0 20
""",
                OUT.getvalue(),
            )
        )

    def test_presolve_almost_lower_triangular_nonlinear(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), bounds=(-10, 10))
        m.obj = Objective(expr=m.x[3] + m.x[4] + pyo.log(m.x[0]))
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] + 2 * m.x[4] == 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)
        m.c.add(2 * (m.x[0] ** 2) + m.x[0] + m.x[2] + 3 * (m.x[3] ** 3) == 10)

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), "")

        self.assertEqual(
            nlinfo.eliminated_vars,
            [
                (m.x[4], nl_writer.AMPLRepn(-12, {id(m.x[1]): 3}, None)),
                (m.x[3], nl_writer.AMPLRepn(-72, {id(m.x[1]): 17}, None)),
                (m.x[2], nl_writer.AMPLRepn(-13, {id(m.x[1]): 4}, None)),
                (m.x[0], nl_writer.AMPLRepn(29, {id(m.x[1]): -6}, None)),
            ],
        )
        # Note: bounds on x[1] are:
        #   min(22/3, 82/17, 23/4, -39/-6) == 4.823529411764706
        #   max(2/3, 62/17, 3/4, -19/-6) == 3.6470588235294117
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 1 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 1 1 1 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 1 1 	# nonzeros in Jacobian, obj. gradient
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0
o0
o2
n2
o5
o0
o2
n-6.0
v0
n29.0
n2
o2
n3
o5
o0
o2
n17.0
v0
n-72.0
n3
O0 0
o0
o43
o0
o2
n-6.0
v0
n29.0
n-84.0
x0
r
4 -6.0
b
0 3.6470588235294117 4.823529411764706
k0
J0 1
0 -2.0
G0 1
0 20.0
""",
                OUT.getvalue(),
            )
        )

    def test_presolve_lower_triangular_out_of_bounds(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(5), domain=pyo.NonNegativeReals)
        m.obj = Objective(expr=m.x[3] + m.x[4])
        m.c = pyo.ConstraintList()
        m.c.add(m.x[0] == 5)
        m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
        m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
        m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)

        OUT = io.StringIO()
        with self.assertRaisesRegex(
            nl_writer.InfeasibleConstraintException,
            r"model contains a trivially infeasible variable 'x\[3\]' "
            r"\(presolved to a value of -4.0 outside bounds \[0, None\]\).",
        ):
            with LoggingIntercept() as LOG:
                nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
        self.assertEqual(LOG.getvalue(), "")

    def test_presolve_named_expressions(self):
        # Test from #3055
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1, bounds=(0, 10))
        m.subexpr = pyo.Expression(pyo.Integers)
        m.subexpr[1] = m.x[1] + m.x[2]
        m.eq = pyo.Constraint(pyo.Integers)
        m.eq[1] = m.x[1] == 7
        m.eq[2] = m.x[3] == 0.1 * m.subexpr[1] * m.x[2]
        m.obj = pyo.Objective(expr=m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 3)

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(
                m, OUT, symbolic_solver_labels=True, linear_presolve=True
            )
        self.assertEqual(LOG.getvalue(), "")

        self.assertEqual(
            nlinfo.eliminated_vars, [(m.x[1], nl_writer.AMPLRepn(7, {}, None))]
        )

        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 2 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 1 2 1 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2 	# nonzeros in Jacobian, obj. gradient
 5 4	# max name lengths: constraints, variables
 0 0 0 1 0	# common exprs: b,c,o,c1,o1
V2 1 1	#subexpr[1]
0 1
n7.0
C0	#eq[2]
o16	#-
o2	#*
o2	#*
n0.1
v2	#subexpr[1]
v0	#x[2]
O0 0	#obj
o54	# sumlist
3	# (n)
o5	#^
n7.0
n2
o5	#^
v0	#x[2]
n2
o5	#^
v1	#x[3]
n3
x2	# initial guess
0 1	#x[2]
1 1	#x[3]
r	#1 ranges (rhs's)
4 0	#eq[2]
b	#2 bounds (on variables)
0 0 10	#x[2]
0 0 10	#x[3]
k1	#intermediate Jacobian column lengths
1
J0 2	#eq[2]
0 0
1 1
G0 2	#obj
0 0
1 0
""",
                OUT.getvalue(),
            )
        )

    def test_scaling(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0)
        m.y = pyo.Var(initialize=0, bounds=(-2e5, 1e5))
        m.z = pyo.Var(initialize=0, bounds=(1e3, None))
        m.v = pyo.Var(initialize=0, bounds=(1e3, 1e3))
        m.w = pyo.Var(initialize=0, bounds=(None, 1e3))
        m.obj = pyo.Objective(expr=m.x**2 + (m.y - 50000) ** 2 + m.z)
        m.c = pyo.ConstraintList()
        m.c.add(100 * m.x + m.y / 100 >= 600)
        m.c.add(1000 * m.w + m.v * m.x <= 100)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        m.dual[m.c[1]] = 0.02

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(
                m, OUT, scale_model=False, linear_presolve=False
            )
        self.assertEqual(LOG.getvalue(), "")

        nl1 = OUT.getvalue()
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 5 2 1 0 0 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 3 1 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 5 3 	# nonzeros in Jacobian, obj. gradient
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0
o2
v1
v0
C1
n0
O0 0
o0
o5
v0
n2
o5
o0
v2
n-50000
n2
d1
1 0.02
x5
0 0
1 0
2 0
3 0
4 0
r
1 100
2 600
b
3
4 1000.0
0 -200000.0 100000.0
2 1000.0
1 1000.0
k4
2
3
4
4
J0 3
0 0
1 0
4 1000
J1 2
0 100
2 0.01
G0 3
0 0
2 0
3 1
""",
                nl1,
            )
        )

        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.v] = 1 / 250
        m.scaling_factor[m.w] = 1 / 500
        # m.scaling_factor[m.x] = 1
        m.scaling_factor[m.y] = -1 / 50000
        m.scaling_factor[m.z] = 1 / 1000
        m.scaling_factor[m.c[1]] = 1 / 10
        m.scaling_factor[m.c[2]] = -1 / 100
        m.scaling_factor[m.obj] = 1 / 100

        OUT = io.StringIO()
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(
                m, OUT, scale_model=True, linear_presolve=False
            )
        self.assertEqual(LOG.getvalue(), "")

        nl2 = OUT.getvalue()

        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 5 2 1 0 0 	# vars, constraints, objectives, ranges, eqns
 1 1 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 2 3 1 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 5 3 	# nonzeros in Jacobian, obj. gradient
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
C0
o2
n-0.01
o2
o3
v1
n0.004
v0
C1
n0
O0 0
o2
n0.01
o0
o5
v0
n2
o5
o0
o3
v2
n-2e-05
n-50000
n2
d1
1 0.002
x5
0 0
1 0.0
2 0.0
3 0.0
4 0.0
r
2 -1.0
2 60.0
b
3
4 4.0
0 -2.0 4.0
2 1.0
1 2.0
k4
2
3
4
4
J0 3
0 0.0
1 0.0
4 -5000.0
J1 2
0 10.0
2 -50.0
G0 3
0 0.0
2 0.0
3 10.0
""",
                nl2,
            )
        )

        # Debugging: this diffs the unscaled & scaled models
        # self.assertEqual(*nl_diff(nl1, nl2))

    def test_named_expressions(self):
        # This tests an error possibly reported by #2810
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.E1 = Expression(expr=3 * (m.x * m.y + m.z))
        m.E2 = Expression(expr=m.z * m.y)
        m.E3 = Expression(expr=m.x * m.z + m.y)
        m.o1 = Objective(expr=m.E1 + m.E2)
        m.o2 = Objective(expr=m.E1**2)
        m.c1 = Constraint(expr=m.E2 + 2 * m.E3 >= 0)
        m.c2 = Constraint(expr=pyo.inequality(0, m.E3**2, 10))

        OUT = io.StringIO()
        nl_writer.NLWriter().write(m, OUT, symbolic_solver_labels=True)

        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 3 2 2 1 0 	# vars, constraints, objectives, ranges, eqns
 2 2 0 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 3 3 3 	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0 	# discrete variables: binary, integer, nonlinear (b,c,o)
 6 6 	# nonzeros in Jacobian, obj. gradient
 2 1	# max name lengths: constraints, variables
 1 1 1 1 1	# common exprs: b,c,o,c1,o1
V3 0 0	#nl(E1)
o2	#*
v0	#x
v1	#y
V4 0 0	#E2
o2	#*
v2	#z
v1	#y
V5 0 0	#nl(E3)
o2	#*
v0	#x
v2	#z
C0	#c1
o0	#+
v4	#E2
o2	#*
n2
v5	#nl(E3)
V6 1 2	#E3
1 1
v5	#nl(E3)
C1	#c2
o5	#^
v6	#E3
n2
O0 0	#o1
o0	#+
o2	#*
n3
v3	#nl(E1)
v4	#E2
V7 1 4	#E1
2 3
o2	#*
n3
v3	#nl(E1)
O1 0	#o2
o5	#^
v7	#E1
n2
x0	# initial guess
r	#2 ranges (rhs's)
2 0	#c1
0 0 10	#c2
b	#3 bounds (on variables)
3	#x
3	#y
3	#z
k2	#intermediate Jacobian column lengths
2
4
J0 3	#c1
0 0
1 2
2 0
J1 3	#c2
0 0
1 0
2 0
G0 3	#o1
0 0
1 0
2 3
G1 3	#o2
0 0
1 0
2 0
""",
                OUT.getvalue(),
            )
        )

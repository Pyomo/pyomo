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

import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
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
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
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
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression(((3 * m.x) / m.p, None, None))
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
            repn = info.visitor.walk_expression((3 * (m.x + 2) / m.p, None, None))
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
            repn = info.visitor.walk_expression((m.x**2 / m.p, None, None))
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

        repn_util.HALT_ON_EVALUATION_ERROR, tmp = (
            True,
            repn_util.HALT_ON_EVALUATION_ERROR,
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
            repn_util.HALT_ON_EVALUATION_ERROR = tmp

    def test_errors_negative_frac_pow(self):
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=-1)
        m.x = Var()

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((m.p ** (0.5), None, None))
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
            repn = info.visitor.walk_expression((m.p**m.x, None, None))
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
            repn = info.visitor.walk_expression((log(m.p), None, None))
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

        info = INFO()
        with LoggingIntercept() as LOG:
            repn = info.visitor.walk_expression((expr, None, None))
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
        repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        expr = 3 * m.y
        repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, InvalidNumber(None))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, None)

        m.p.value = None
        expr = 5 * (m.p * m.x + 2 * m.z)
        repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.z): 10, id(m.x): InvalidNumber(None)})
        self.assertEqual(repn.nonlinear, None)

        expr = m.y * m.x
        repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {id(m.x): InvalidNumber(None)})
        self.assertEqual(repn.nonlinear, None)

        m.z = Var([1, 2, 3, 4], initialize=lambda m, i: i - 1)
        m.z[1].fix(None)
        expr = m.z[1] - ((m.z[2] * m.z[3]) * m.z[4])
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, InvalidNumber(None))
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear[0], 'o16\no2\no2\nv%s\nv%s\nv%s\n')
        self.assertEqual(repn.nonlinear[1], [id(m.z[2]), id(m.z[3]), id(m.z[4])])

        m.z[3].fix(float('nan'))
        with INFO() as info:
            repn = info.visitor.walk_expression((expr, None, None))
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
            repn = info.visitor.walk_expression((expr, None, None))
        self.assertEqual(LOG.getvalue(), "")
        self.assertEqual(repn.nl, None)
        self.assertEqual(repn.mult, 1)
        self.assertEqual(repn.const, 0)
        self.assertEqual(repn.linear, {})
        self.assertEqual(repn.nonlinear, ('o24\no3\nn1\nv%s\nn0\n', [id(m.x)]))

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
            repn1 = info.visitor.walk_expression((expr1, None, None))
            repn2 = info.visitor.walk_expression((expr2, None, None))
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

    def test_linear_constraint_npv_const(self):
        # This tests an error possibly reported by #2810
        m = ConcreteModel()
        m.x = Var([1, 2])
        m.p = Param(initialize=5, mutable=True)
        m.o = Objective(expr=1)
        m.c = Constraint(
            expr=LinearExpression([m.p**2, 5 * m.x[1], 10 * m.x[2]]) == 0
        )

        OUT = io.StringIO()
        nl_writer.NLWriter().write(m, OUT)
        self.assertEqual(
            *nl_diff(
                """g3 1 1 0	# problem unknown
 2 1 1 0 1 	# vars, constraints, objectives, ranges, eqns
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
4 -25
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
        with LoggingIntercept() as LOG:
            nl_writer.NLWriter().write(m, OUT, symbolic_solver_labels=True)
        self.assertEqual(LOG.getvalue(), "")
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

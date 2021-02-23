#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
# Note: the self.mcpp.* functions are all C-style functions implemented
# in the compiled MC++ wrapper library
# Note: argument to pow must be an integer
from __future__ import division

import ctypes
import logging
import os
import six

from pyomo.common.fileutils import Library
from pyomo.core import value, Expression
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.numeric_expr import (
    AbsExpression, LinearExpression, NegationExpression, NPV_AbsExpression,
    NPV_ExternalFunctionExpression, NPV_NegationExpression, NPV_PowExpression,
    NPV_ProductExpression, NPV_ReciprocalExpression, NPV_SumExpression,
    NPV_UnaryFunctionExpression, PowExpression, ProductExpression,
    ReciprocalExpression, SumExpression, UnaryFunctionExpression,
    NPV_DivisionExpression, DivisionExpression,
)
from pyomo.core.expr.visitor import (
    StreamBasedExpressionVisitor, identify_variables,
)
from pyomo.common.collections import ComponentMap

logger = logging.getLogger('pyomo.contrib.mcpp')

path = os.path.dirname(__file__)

__version__ = "19.11.12"


def mcpp_available():
    """True if the MC++ shared object file exists. False otherwise."""
    return Library('mcppInterface').path() is not None


NPV_expressions = (
    NPV_AbsExpression, NPV_ExternalFunctionExpression,
    NPV_NegationExpression, NPV_PowExpression,
    NPV_ProductExpression, NPV_ReciprocalExpression, NPV_SumExpression,
    NPV_UnaryFunctionExpression, NPV_DivisionExpression,
)


def _MCPP_lib():
    """A singleton interface to the MC++ library"""
    if _MCPP_lib._mcpp is not None:
        return _MCPP_lib._mcpp

    _MCPP_lib._mcpp = mcpp = ctypes.CDLL(Library('mcppInterface').path())

    # Version number
    mcpp.get_version.restype = ctypes.c_char_p
    
    mcpp.toString.argtypes = [ctypes.c_void_p]
    mcpp.toString.restype = ctypes.c_char_p

    mcpp.lower.argtypes = [ctypes.c_void_p]
    mcpp.lower.restype = ctypes.c_double

    mcpp.upper.argtypes = [ctypes.c_void_p]
    mcpp.upper.restype = ctypes.c_double

    mcpp.concave.argtypes = [ctypes.c_void_p]
    mcpp.concave.restype = ctypes.c_double

    mcpp.convex.argtypes = [ctypes.c_void_p]
    mcpp.convex.restype = ctypes.c_double

    mcpp.subcc.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcpp.subcc.restype = ctypes.c_double

    mcpp.subcv.argtypes = [ctypes.c_void_p, ctypes.c_int]
    mcpp.subcv.restype = ctypes.c_double

    # Create MC type variable
    mcpp.newVar.argtypes = [ctypes.c_double, ctypes.c_double,
                                 ctypes.c_double, ctypes.c_int, ctypes.c_int]
    mcpp.newVar.restype = ctypes.c_void_p

    # Create MC type constant
    mcpp.newConstant.argtypes = [ctypes.c_double]
    mcpp.newConstant.restype = ctypes.c_void_p

    # Multiply MC objects
    mcpp.multiply.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.multiply.restype = ctypes.c_void_p

    # Add MC objects
    mcpp.add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.add.restype = ctypes.c_void_p

    # pow(x, y) functions
    # y is integer
    mcpp.power.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.power.restype = ctypes.c_void_p
    # y is fractional
    mcpp.powerf.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.powerf.restype = ctypes.c_void_p
    # y is an expression
    mcpp.powerx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.powerx.restype = ctypes.c_void_p

    # sqrt function
    mcpp.mc_sqrt.argtypes = [ctypes.c_void_p]
    mcpp.mc_sqrt.restype = ctypes.c_void_p

    # 1 / MC Variable
    mcpp.reciprocal.argtypes = [ctypes.c_void_p]
    mcpp.reciprocal.restype = ctypes.c_void_p

    # - MC Variable
    mcpp.negation.argtypes = [ctypes.c_void_p]
    mcpp.negation.restype = ctypes.c_void_p

    # fabs(MC Variable)
    mcpp.mc_abs.argtypes = [ctypes.c_void_p]
    mcpp.mc_abs.restype = ctypes.c_void_p

    # sin(MC Variable)
    mcpp.trigSin.argtypes = [ctypes.c_void_p]
    mcpp.trigSin.restype = ctypes.c_void_p

    # cos(MC Variable)
    mcpp.trigCos.argtypes = [ctypes.c_void_p]
    mcpp.trigCos.restype = ctypes.c_void_p

    # tan(MC Variable)
    mcpp.trigTan.argtypes = [ctypes.c_void_p]
    mcpp.trigTan.restype = ctypes.c_void_p

    # asin(MC Variable)
    mcpp.atrigSin.argtypes = [ctypes.c_void_p]
    mcpp.atrigSin.restype = ctypes.c_void_p

    # acos(MC Variable)
    mcpp.atrigCos.argtypes = [ctypes.c_void_p]
    mcpp.atrigCos.restype = ctypes.c_void_p

    # atan(MC Variable)
    mcpp.atrigTan.argtypes = [ctypes.c_void_p]
    mcpp.atrigTan.restype = ctypes.c_void_p

    # exp(MC Variable)
    mcpp.exponential.argtypes = [ctypes.c_void_p]
    mcpp.exponential.restype = ctypes.c_void_p

    # log(MC Variable)
    mcpp.logarithm.argtypes = [ctypes.c_void_p]
    mcpp.logarithm.restype = ctypes.c_void_p

    # Releases object from memory (prevent memory leaks)
    mcpp.release.argtypes = [ctypes.c_void_p]

    # Unary function exception wrapper
    mcpp.try_unary_fcn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    mcpp.try_unary_fcn.restype = ctypes.c_void_p

    # Binary function exception wrapper
    mcpp.try_binary_fcn.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p]
    mcpp.try_binary_fcn.restype = ctypes.c_void_p

    # Error message retrieval
    mcpp.get_last_exception_message.restype = ctypes.c_char_p

    return mcpp

# Initialize the singleton to None
_MCPP_lib._mcpp = None


class MCPP_Error(Exception):
    pass


class MCPP_visitor(StreamBasedExpressionVisitor):
    """Creates an MC++ expression from the corresponding Pyomo expression.

    This class walks a pyomo expression tree and builds up the corresponding
    expression of type McCormick.

    Note on memory management: The MCPP_visitor will return a pointer to
    an MC++ interval object that was dynamically allocated within the C
    interface.  It is the caller's responsibility to call
    `mcpp_lib.release()` on that object to prevent a memory leak

    """

    def __init__(self, expression, improved_var_bounds=None):
        super(MCPP_visitor, self).__init__()
        self.mcpp = _MCPP_lib()
        so_file_version = self.mcpp.get_version()
        if six.PY3:
            so_file_version = so_file_version.decode("utf-8")
        if not so_file_version == __version__:
            raise MCPP_Error(
                "Shared object file version %s is out of date with MC++ interface version %s. "
                "Please rebuild the library." % (so_file_version, __version__)
            )
        self.missing_value_warnings = []
        self.expr = expression
        vars = list(identify_variables(expression, include_fixed=False))
        self.num_vars = len(vars)
        # Map expression variables to MC variables
        self.known_vars = ComponentMap()
        # Map expression variables to their index
        self.var_to_idx = ComponentMap()
        # Pre-register all variables
        inf = float('inf')
        for i, var in enumerate(vars):
            self.var_to_idx[var] = i
            # check if improved variable bound is provided
            if improved_var_bounds is not None:
                lb, ub = improved_var_bounds.get(var, (-inf, inf))
            else:
                lb, ub = -inf, inf
            self.known_vars[var] = self.register_var(var, lb, ub)
        self.refs = None

    def walk_expression(self):
        self.refs = set()
        ans = super(MCPP_visitor, self).walk_expression(self.expr)
        self.refs = None
        return ans

    def exitNode(self, node, data):
        if isinstance(node, ProductExpression):
            ans = self.mcpp.multiply(data[0], data[1])
        elif isinstance(node, SumExpression):
            ans = data[0]
            for arg in data[1:]:
                ans = self.mcpp.add(ans, arg)
        elif isinstance(node, PowExpression):
            if type(node.arg(1)) == int:
                ans = self.mcpp.try_binary_fcn(self.mcpp.power, data[0], data[1])
            elif type(node.arg(1)) == float:
                ans = self.mcpp.try_binary_fcn(self.mcpp.powerf, data[0], data[1])
            else:
                ans = self.mcpp.try_binary_fcn(self.mcpp.powerx, data[0], data[1])
        elif isinstance(node, ReciprocalExpression):
            # Note: unreachable after ReciprocalExpression was removed
            ans = self.mcpp.try_unary_fcn(self.mcpp.reciprocal, data[0])
        elif isinstance(node, DivisionExpression):
            ans = self.mcpp.try_binary_fcn(self.mcpp.divide, data[0], data[1])
        elif isinstance(node, NegationExpression):
            ans = self.mcpp.negation(data[0])
        elif isinstance(node, AbsExpression):
            ans = self.mcpp.try_unary_fcn(self.mcpp.mc_abs, data[0])
        elif isinstance(node, LinearExpression):
            raise NotImplementedError(
                'Quicksum has bugs that prevent proper usage of MC++.')
            # ans = self.mcpp.newConstant(node.constant)
            # for coef, var in zip(node.linear_coefs, node.linear_vars):
            #     ans = self.mcpp.add(
            #         ans,
            #         self.mcpp.multiply(
            #             self.mcpp.newConstant(coef),
            #             self.register_num(var)))
        elif isinstance(node, UnaryFunctionExpression):
            if node.name == "exp":
                ans = self.mcpp.try_unary_fcn(self.mcpp.exponential, data[0])
            elif node.name == "log":
                ans = self.mcpp.try_unary_fcn(self.mcpp.logarithm, data[0])
            elif node.name == "sin":
                ans = self.mcpp.try_unary_fcn(self.mcpp.trigSin, data[0])
            elif node.name == "cos":
                ans = self.mcpp.try_unary_fcn(self.mcpp.trigCos, data[0])
            elif node.name == "tan":
                ans = self.mcpp.try_unary_fcn(self.mcpp.trigTan, data[0])
            elif node.name == "asin":
                ans = self.mcpp.try_unary_fcn(self.mcpp.atrigSin, data[0])
            elif node.name == "acos":
                ans = self.mcpp.try_unary_fcn(self.mcpp.atrigCos, data[0])
            elif node.name == "atan":
                ans = self.mcpp.try_unary_fcn(self.mcpp.atrigTan, data[0])
            elif node.name == "sqrt":
                ans = self.mcpp.try_unary_fcn(self.mcpp.mc_sqrt, data[0])
            else:
                raise NotImplementedError("Unknown unary function: %s" % (node.name,))
        elif isinstance(node, NPV_expressions):
            ans = self.mcpp.newConstant(value(data[0]))
        elif type(node) in nonpyomo_leaf_types:
            ans = self.mcpp.newConstant(node)
        elif not node.is_expression_type():
            ans = self.register_num(node)
        elif type(node) in SubclassOf(Expression) or isinstance(node, _ExpressionData):
            ans = data[0]
        else:
            raise RuntimeError("Unhandled expression type: %s" % (type(node)))

        if ans is None:
            msg = self.mcpp.get_last_exception_message()
            if six.PY3:
                msg = msg.decode("utf-8")
            raise MCPP_Error(msg)

        return ans

    def beforeChild(self, node, child, child_idx):
        if type(child) in nonpyomo_leaf_types:
            # This means the child is POD
            # i.e., int, float, string
            return False, self.mcpp.newConstant(child)
        elif not child.is_expression_type():
            # node is either a Param, Var, or NumericConstant
            return False, self.register_num(child)
        else:
            # this is an expression node
            return True, None

    def acceptChildResult(self, node, data, child_result, child_idx):
        self.refs.add(child_result)
        data.append(child_result)
        return data

    def register_num(self, num):
        """Registers a new number: Param, Var, or NumericConstant."""
        if num.is_fixed():
            return self.mcpp.newConstant(value(num))
        else:
            return self.known_vars[num]

    def register_var(self, var, lb, ub):
        """Registers a new variable."""
        var_idx = self.var_to_idx[var]
        inf = float('inf')

        # Guard against errant None values in lb and ub
        lb = -inf if lb is None else lb
        ub = inf if ub is None else ub

        lb = max(var.lb if var.has_lb() else -inf, lb)
        ub = min(var.ub if var.has_ub() else inf, ub)
        var_val = value(var, exception=False)
        if lb == -inf:
            lb = -500000
            logger.warning(
                'Var %s missing lower bound. Assuming LB of %s'
                % (var.name, lb))
        if ub == inf:
            ub = 500000
            logger.warning(
                'Var %s missing upper bound. Assuming UB of %s'
                % (var.name, ub))
        if var_val is None:
            var_val = (lb + ub) / 2
            self.missing_value_warnings.append(
                'Var %s missing value. Assuming midpoint value of %s'
                % (var.name, var_val))
        return self.mcpp.newVar(
            lb, var_val, ub, self.num_vars, var_idx)

    def finalizeResult(self, node_result):
        # Note, the node_result should NOT be in self.refs
        #    self.refs.remove(node_result)
        assert node_result not in self.refs
        for r in self.refs:
            self.mcpp.release(r)
        return node_result


class McCormick(object):

    """
    This class takes the constructed expression from MCPP_Visitor and
    allows for MC methods to be performed on pyomo expressions.

    __repn__(self): returns a display of an MC expression in the form:
    F: [lower interval : upper interval ] [convex underestimator :
    concave overestimator ] [ (convex subgradient) : (concave subgradient]

    lower(self): returns a float of the lower interval bound that is valid
    across the entire domain

    upper(self): returns a float of the upper interval bound that is valid
    across the entire domain

    concave(self): returns a float of the concave overestimator at the
    current value() of each variable.

    convex(self): returns a float of the convex underestimator at the
    current value() of each variable.

    ##Note: In order to describe the concave and convex relaxations over
    the entire domain, it is necessary to use changePoint() to repeat the
    calculation at different points.

    subcc(self): returns a ComponentMap() that maps the pyomo variables
    to the subgradients of the McCormick concave overestimators at the
    current value() of each variable.

    subcv(self): returns a ComponentMap() that maps the pyomo variables
    to the subgradients of the McCormick convex underestimators at the
    current value() of each variable.

    def changePoint(self, var, point): updates the current value() on the
    pyomo side and the current point on the MC++ side.
                                                                    """

    def __init__(self, expression, improved_var_bounds=None):
        # Guarantee that McCormick objects have mc_expr defined
        self.mc_expr = None

        self.mcpp = _MCPP_lib()
        self.pyomo_expr = expression
        self.visitor = MCPP_visitor(expression, improved_var_bounds)
        self.mc_expr = self.visitor.walk_expression()

    def __del__(self):
        if self.mc_expr is not None:
            self.mcpp.release(self.mc_expr)
            self.mc_expr = None

    def __repn__(self):
        repn = self.mcpp.toString(self.mc_expr)
        if six.PY3:
            repn = repn.decode("utf-8")
        return repn

    def __str__(self):
        return self.__repn__()

    def lower(self):
        return self.mcpp.lower(self.mc_expr)

    def upper(self):
        return self.mcpp.upper(self.mc_expr)

    def concave(self):
        self.warn_if_var_missing_value()
        return self.mcpp.concave(self.mc_expr)

    def convex(self):
        self.warn_if_var_missing_value()
        return self.mcpp.convex(self.mc_expr)

    def subcc(self):
        self.warn_if_var_missing_value()
        ans = ComponentMap()
        for key in self.visitor.var_to_idx:
            i = self.visitor.var_to_idx[key]
            ans[key] = self.mcpp.subcc(self.mc_expr, i)
        return ans

    def subcv(self):
        self.warn_if_var_missing_value()
        ans = ComponentMap()
        for key in self.visitor.var_to_idx:
            i = self.visitor.var_to_idx[key]
            ans[key] = self.mcpp.subcv(self.mc_expr, i)
        return ans

    def changePoint(self, var, point):
        var.set_value(point)
        # WARNING: TODO: this has side effects.  If we do not use a
        # fresh MCPP_visitor, we get segfaults and different results.
        self.visitor = MCPP_visitor(self.visitor.expr)
        self.mcpp.release(self.mc_expr)
        self.mc_expr = self.visitor.walk_expression()

    def warn_if_var_missing_value(self):
        if self.visitor.missing_value_warnings:
            for message in self.visitor.missing_value_warnings:
                logger.warning(message)
            self.visitor.missing_value_warnings = []


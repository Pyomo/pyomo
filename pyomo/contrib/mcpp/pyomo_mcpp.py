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
)
from pyomo.core.expr.visitor import (
    StreamBasedExpressionVisitor, identify_variables,
)
from pyomo.core.kernel.component_map import ComponentMap

logger = logging.getLogger('pyomo.contrib.mcpp')

path = os.path.dirname(__file__)


def mcpp_available():
    """True if the MC++ shared object file exists. False otherwise."""
    return Library('mcppInterface').path() is not None


NPV_expressions = {
    NPV_AbsExpression, NPV_ExternalFunctionExpression,
    NPV_NegationExpression, NPV_PowExpression,
    NPV_ProductExpression, NPV_ReciprocalExpression, NPV_SumExpression,
    NPV_UnaryFunctionExpression}


class MCPP_Error(Exception):
    pass


class MCPP_visitor(StreamBasedExpressionVisitor):
    """Creates an MC++ expression from the corresponding Pyomo expression.

    This class walks a pyomo expression tree and builds up the corresponding
    expression of type McCormick.

    """

    def __init__(self, mcpp_lib, expression, improved_var_bounds=ComponentMap()):
        super(MCPP_visitor, self).__init__()
        self.missing_value_warnings = []
        self.expr = expression
        self.mcpp = mcpp_lib
        self.declare_mcpp_library_calls()
        vars = list(identify_variables(expression, include_fixed=False))
        self.num_vars = len(vars)
        # Map expression variables to MC variables
        self.known_vars = ComponentMap()
        # Map expression variables to their index
        self.var_to_idx = ComponentMap()
        # Pre-register all variables
        for i, var in enumerate(vars):
            self.var_to_idx[var] = i
            # check if improved variable bound is provided
            inf = float('inf')
            lb, ub = improved_var_bounds.get(var, (-inf, inf))
            self.known_vars[var] = self.register_var(var, lb, ub)
        self.refs = set()

    def declare_mcpp_library_calls(self):
        # Create MC type variable
        self.mcpp.newVar.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
        self.mcpp.newVar.restype = ctypes.c_void_p

        # Create MC type constant
        self.mcpp.newConstant.argtypes = [ctypes.c_double]
        self.mcpp.newConstant.restype = ctypes.c_void_p

        # Multiply MC objects
        self.mcpp.multiply.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.mcpp.multiply.restype = ctypes.c_void_p

        # Add MC objects
        self.mcpp.add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.mcpp.add.restype = ctypes.c_void_p

        # pow(x, y) functions
        # y is integer
        self.mcpp.power.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.mcpp.power.restype = ctypes.c_void_p
        # y is fractional
        self.mcpp.powerf.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.mcpp.powerf.restype = ctypes.c_void_p
        # y is an expression
        self.mcpp.powerx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.mcpp.powerx.restype = ctypes.c_void_p

        # sqrt function
        self.mcpp.mc_sqrt.argtypes = [ctypes.c_void_p]
        self.mcpp.mc_sqrt.restype = ctypes.c_void_p

        # 1 / MC Variable
        self.mcpp.reciprocal.argtypes = [ctypes.c_void_p]
        self.mcpp.reciprocal.restype = ctypes.c_void_p

        # - MC Variable
        self.mcpp.negation.argtypes = [ctypes.c_void_p]
        self.mcpp.negation.restype = ctypes.c_void_p

        # fabs(MC Variable)
        self.mcpp.mc_abs.argtypes = [ctypes.c_void_p]
        self.mcpp.mc_abs.restype = ctypes.c_void_p

        # sin(MC Variable)
        self.mcpp.trigSin.argtypes = [ctypes.c_void_p]
        self.mcpp.trigSin.restype = ctypes.c_void_p

        # cos(MC Variable)
        self.mcpp.trigCos.argtypes = [ctypes.c_void_p]
        self.mcpp.trigCos.restype = ctypes.c_void_p

        # tan(MC Variable)
        self.mcpp.trigTan.argtypes = [ctypes.c_void_p]
        self.mcpp.trigTan.restype = ctypes.c_void_p

        # asin(MC Variable)
        self.mcpp.atrigSin.argtypes = [ctypes.c_void_p]
        self.mcpp.atrigSin.restype = ctypes.c_void_p

        # acos(MC Variable)
        self.mcpp.atrigCos.argtypes = [ctypes.c_void_p]
        self.mcpp.atrigCos.restype = ctypes.c_void_p

        # atan(MC Variable)
        self.mcpp.atrigTan.argtypes = [ctypes.c_void_p]
        self.mcpp.atrigTan.restype = ctypes.c_void_p

        # exp(MC Variable)
        self.mcpp.exponential.argtypes = [ctypes.c_void_p]
        self.mcpp.exponential.restype = ctypes.c_void_p

        # log(MC Variable)
        self.mcpp.logarithm.argtypes = [ctypes.c_void_p]
        self.mcpp.logarithm.restype = ctypes.c_void_p

        # Releases object from memory (prevent memory leaks)
        self.mcpp.release.argtypes = [ctypes.c_void_p]

        # Unary function exception wrapper
        self.mcpp.try_unary_fcn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.mcpp.try_unary_fcn.restype = ctypes.c_void_p

        # Binary function exception wrapper
        self.mcpp.try_binary_fcn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.mcpp.try_binary_fcn.restype = ctypes.c_void_p

        # Error message retrieval
        self.mcpp.get_last_exception_message.restype = ctypes.c_char_p

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
            ans = self.mcpp.try_unary_fcn(self.mcpp.reciprocal, data[0])
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
        elif any(isinstance(node, npv) for npv in NPV_expressions):
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

    def beforeChild(self, node, child):
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

    def acceptChildResult(self, node, data, child_result):
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
                'Var %s missing value. Assuming midpoint value of %s' % (var.name, var_val))
        return self.mcpp.newVar(
            lb, var_val, ub, self.num_vars, var_idx)

    def finalizeResult(self, node_result):
        for r in self.refs:
            if r != node_result:
                self.mcpp.release(r)
        self.refs = [node_result, ]
        return node_result

    def __del__(self):
        for r in self.refs:
            self.mcpp.release(r)


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

    def __init__(self, expression, improved_var_bounds=ComponentMap()):
        self.mcpp_lib = ctypes.CDLL(Library('mcppInterface').path())
        self.pyomo_expr = expression
        self.visitor = MCPP_visitor(self.mcpp_lib, expression, improved_var_bounds)
        self.mc_expr = self.visitor.walk_expression(expression)

        self.mcpp_lib.toString.argtypes = [ctypes.c_void_p]
        self.mcpp_lib.toString.restype = ctypes.c_char_p

        self.mcpp_lib.lower.argtypes = [ctypes.c_void_p]
        self.mcpp_lib.lower.restype = ctypes.c_double

        self.mcpp_lib.upper.argtypes = [ctypes.c_void_p]
        self.mcpp_lib.upper.restype = ctypes.c_double

        self.mcpp_lib.concave.argtypes = [ctypes.c_void_p]
        self.mcpp_lib.concave.restype = ctypes.c_double

        self.mcpp_lib.convex.argtypes = [ctypes.c_void_p]
        self.mcpp_lib.convex.restype = ctypes.c_double

        self.mcpp_lib.subcc.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.mcpp_lib.subcc.restype = ctypes.c_double

        self.mcpp_lib.subcv.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.mcpp_lib.subcv.restype = ctypes.c_double

    def __repn__(self):
        repn = self.mcpp_lib.toString(self.mc_expr)
        if six.PY3:
            repn = repn.decode("utf-8")
        return repn

    def __str__(self):
        return self.__repn__()

    def lower(self):
        return self.mcpp_lib.lower(self.mc_expr)

    def upper(self):
        return self.mcpp_lib.upper(self.mc_expr)

    def concave(self):
        self.warn_if_var_missing_value()
        return self.mcpp_lib.concave(self.mc_expr)

    def convex(self):
        self.warn_if_var_missing_value()
        return self.mcpp_lib.convex(self.mc_expr)

    def subcc(self):
        self.warn_if_var_missing_value()
        ans = ComponentMap()
        for key in self.visitor.var_to_idx:
            i = self.visitor.var_to_idx[key]
            ans[key] = self.mcpp_lib.subcc(self.mc_expr, i)
        return ans

    def subcv(self):
        self.warn_if_var_missing_value()
        ans = ComponentMap()
        for key in self.visitor.var_to_idx:
            i = self.visitor.var_to_idx[key]
            ans[key] = self.mcpp_lib.subcv(self.mc_expr, i)
        return ans

    def changePoint(self, var, point):
        var.set_value(point)
        # WARNING: TODO: this has side effects
        self.visitor = MCPP_visitor(self.mcpp_lib, self.pyomo_expr)
        self.mc_expr = self.visitor.walk_expression(self.pyomo_expr)

    def warn_if_var_missing_value(self):
        if self.visitor.missing_value_warnings:
            for message in self.visitor.missing_value_warnings:
                logger.warning(message)
            del self.visitor.missing_value_warnings[:]  # list.clear() does not exist in python 2.7


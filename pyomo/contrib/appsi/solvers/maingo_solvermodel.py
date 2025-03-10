#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math

from pyomo.common.dependencies import attempt_import
from pyomo.core.base.var import ScalarVar
from pyomo.core.base.expression import ScalarExpression
import pyomo.core.expr.expr_common as common
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
    value,
    is_constant,
    is_fixed,
    native_numeric_types,
    native_types,
    nonpyomo_leaf_types,
)
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.repn.util import valid_expr_ctypes_minlp


maingopy, maingopy_available = attempt_import("maingopy")

_plusMinusOne = {1, -1}

LEFT_TO_RIGHT = common.OperatorAssociativity.LEFT_TO_RIGHT
RIGHT_TO_LEFT = common.OperatorAssociativity.RIGHT_TO_LEFT


class ToMAiNGOVisitor(EXPR.ExpressionValueVisitor):
    def __init__(self, variables, idmap):
        super(ToMAiNGOVisitor, self).__init__()
        self.variables = variables
        self.idmap = idmap
        self._pyomo_func_to_maingo_func = {
            "log": maingopy.log,
            "log10": ToMAiNGOVisitor.maingo_log10,
            "sin": maingopy.sin,
            "cos": maingopy.cos,
            "tan": maingopy.tan,
            "cosh": maingopy.cosh,
            "sinh": maingopy.sinh,
            "tanh": maingopy.tanh,
            "asin": maingopy.asin,
            "acos": maingopy.acos,
            "atan": maingopy.atan,
            "exp": maingopy.exp,
            "sqrt": maingopy.sqrt,
            "asinh": ToMAiNGOVisitor.maingo_asinh,
            "acosh": ToMAiNGOVisitor.maingo_acosh,
            "atanh": ToMAiNGOVisitor.maingo_atanh,
        }

    @classmethod
    def maingo_log10(cls, x):
        return maingopy.log(x) / math.log(10)

    @classmethod
    def maingo_asinh(cls, x):
        return maingopy.log(x + maingopy.sqrt(maingopy.pow(x, 2) + 1))

    @classmethod
    def maingo_acosh(cls, x):
        return maingopy.log(x + maingopy.sqrt(maingopy.pow(x, 2) - 1))

    @classmethod
    def maingo_atanh(cls, x):
        return 0.5 * maingopy.log(x + 1) - 0.5 * maingopy.log(1 - x)

    def visit(self, node, values):
        """Visit nodes that have been expanded"""
        for i, val in enumerate(values):
            arg = node._args_[i]

            if arg is None:
                values[i] = "Undefined"
            elif arg.__class__ in native_numeric_types:
                pass
            elif arg.__class__ in nonpyomo_leaf_types:
                values[i] = val
            else:
                parens = False
                if arg.is_expression_type() and node.PRECEDENCE is not None:
                    if arg.PRECEDENCE is None:
                        pass
                    elif node.PRECEDENCE < arg.PRECEDENCE:
                        parens = True
                    elif node.PRECEDENCE == arg.PRECEDENCE:
                        if i == 0:
                            parens = node.ASSOCIATIVITY != LEFT_TO_RIGHT
                        elif i == len(node._args_) - 1:
                            parens = node.ASSOCIATIVITY != RIGHT_TO_LEFT
                        else:
                            parens = True
                if parens:
                    values[i] = val

        if node.__class__ in EXPR.NPV_expression_types:
            return value(node)

        if node.__class__ in {EXPR.ProductExpression, EXPR.MonomialTermExpression}:
            return values[0] * values[1]

        if node.__class__ in {EXPR.SumExpression}:
            return sum(values)

        if node.__class__ in {EXPR.PowExpression}:
            return maingopy.pow(values[0], values[1])

        if node.__class__ in {EXPR.DivisionExpression}:
            return values[0] / values[1]

        if node.__class__ in {EXPR.NegationExpression}:
            return -values[0]

        if node.__class__ in {EXPR.AbsExpression}:
            return maingopy.abs(values[0])

        if node.__class__ in {EXPR.UnaryFunctionExpression}:
            pyomo_func = node.getname()
            maingo_func = self._pyomo_func_to_maingo_func[pyomo_func]
            return maingo_func(values[0])

        if node.__class__ in {ScalarExpression}:
            return values[0]

        raise ValueError(f"Unknown function expression encountered: {node.getname()}")

    def visiting_potential_leaf(self, node):
        """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if node.__class__ in native_types:
            return True, node

        if node.is_expression_type():
            if node.__class__ is EXPR.MonomialTermExpression:
                return True, self._monomial_to_maingo(node)
            if node.__class__ is EXPR.LinearExpression:
                return True, self._linear_to_maingo(node)
            return False, None

        if node.is_component_type():
            if node.ctype not in valid_expr_ctypes_minlp:
                # Make sure all components in active constraints
                # are basic ctypes we know how to deal with.
                raise RuntimeError(
                    "Unallowable component '%s' of type %s found in an active "
                    "constraint or objective.\nMAiNGO cannot export "
                    "expressions with this component type."
                    % (node.name, node.ctype.__name__)
                )

        if node.is_fixed():
            return True, node()
        else:
            assert node.is_variable_type()
            maingo_var_id = self.idmap[id(node)]
            maingo_var = self.variables[maingo_var_id]
            return True, maingo_var

    def _monomial_to_maingo(self, node):
        const, var = node.args
        if const.__class__ not in native_types:
            const = value(const)
        if var.is_fixed():
            return const * var.value
        if not const:
            return 0
        maingo_var = self._var_to_maingo(var)
        if const in _plusMinusOne:
            if const < 0:
                return -maingo_var
            else:
                return maingo_var
        return const * maingo_var

    def _var_to_maingo(self, var):
        maingo_var_id = self.idmap[id(var)]
        maingo_var = self.variables[maingo_var_id]
        return maingo_var

    def _linear_to_maingo(self, node):
        values = [
            (
                self._monomial_to_maingo(arg)
                if (arg.__class__ is EXPR.MonomialTermExpression)
                else (
                    value(arg)
                    if arg.__class__ in native_numeric_types
                    else (
                        self._var_to_maingo(arg)
                        if arg.is_variable_type()
                        else value(arg)
                    )
                )
            )
            for arg in node.args
        ]
        return sum(values)


class SolverModel(maingopy.MAiNGOmodel if maingopy_available else object):
    def __init__(self, var_list, objective, con_list, idmap, logger):
        maingopy.MAiNGOmodel.__init__(self)
        self._var_list = var_list
        self._con_list = con_list
        self._objective = objective
        self._idmap = idmap
        self._logger = logger
        self._no_objective = False

        if self._objective is None:
            self._logger.warning("No objective given, setting a dummy objective of 1.")
            self._no_objective = True

    def build_maingo_objective(self, obj, visitor):
        if self._no_objective:
            return visitor.variables[-1]
        maingo_obj = visitor.dfs_postorder_stack(obj.expr)
        if obj.sense == maximize:
            return -1 * maingo_obj
        return maingo_obj

    def build_maingo_constraints(self, cons, visitor):
        eqs = []
        ineqs = []
        for con in cons:
            if con.equality:
                eqs += [visitor.dfs_postorder_stack(con.body - con.lower)]
            elif con.has_ub() and con.has_lb():
                ineqs += [visitor.dfs_postorder_stack(con.body - con.upper)]
                ineqs += [visitor.dfs_postorder_stack(con.lower - con.body)]
            elif con.has_ub():
                ineqs += [visitor.dfs_postorder_stack(con.body - con.upper)]
            elif con.has_lb():
                ineqs += [visitor.dfs_postorder_stack(con.lower - con.body)]
            else:
                raise ValueError(
                    "Constraint does not have a lower "
                    "or an upper bound: {0} \n".format(con)
                )
        return eqs, ineqs

    def get_variables(self):
        vars = [
            maingopy.OptimizationVariable(
                maingopy.Bounds(var.lb, var.ub), var.type, var.name
            )
            for var in self._var_list
        ]
        if self._no_objective:
            vars += [maingopy.OptimizationVariable(maingopy.Bounds(1, 1), "dummy_obj")]
        return vars

    def get_initial_point(self):
        initial = [
            var.init if not var.init is None else (var.lb + var.ub) / 2.0
            for var in self._var_list
        ]
        if self._no_objective:
            initial += [1]
        return initial

    def evaluate(self, maingo_vars):
        visitor = ToMAiNGOVisitor(maingo_vars, self._idmap)
        result = maingopy.EvaluationContainer()
        result.objective = self.build_maingo_objective(self._objective, visitor)
        eqs, ineqs = self.build_maingo_constraints(self._con_list, visitor)
        result.eq = eqs
        result.ineq = ineqs
        return result

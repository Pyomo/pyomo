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

from pyomo.common.errors import PyomoException
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
import pyomo.core.expr as EXPR
from pyomo.core.base.objective import Objective


class NoFeasibleSolutionError(PyomoException):
    default_message = (
        'A feasible solution was not found, so no solution can be loaded. '
        'Please set opt.config.load_solutions=False and check '
        'results.solution_status and '
        'results.incumbent_objective before loading a solution.'
    )


class NoOptimalSolutionError(PyomoException):
    default_message = (
        'Solver did not find the optimal solution. Set '
        'opt.config.raise_exception_on_nonoptimal_result=False to bypass this error.'
    )


class NoSolutionError(PyomoException):
    default_message = (
        'Solution loader does not currently have a valid solution. Please '
        'check results.termination_condition and/or results.solution_status.'
    )


class NoDualsError(PyomoException):
    default_message = (
        'Solver does not currently have valid duals. Please '
        'check results.termination_condition and/or results.solution_status.'
    )


class NoReducedCostsError(PyomoException):
    default_message = (
        'Solver does not currently have valid reduced costs. Please '
        'check results.termination_condition and/or results.solution_status.'
    )


class IncompatibleModelError(PyomoException):
    default_message = (
        'Model is not compatible with the chosen solver. Please check '
        'the model and solver.'
    )


def get_objective(block):
    """
    Get current active objective on a block. If there is more than one active,
    return an error.
    """
    objective = None
    for obj in block.component_data_objects(
        Objective, descend_into=True, active=True, sort=True
    ):
        if objective is not None:
            raise ValueError('Multiple active objectives found.')
        objective = obj
    return objective


class _VarAndNamedExprCollector(ExpressionValueVisitor):
    def __init__(self):
        self.named_expressions = {}
        self.variables = {}
        self.fixed_vars = {}
        self._external_functions = {}

    def visit(self, node, values):
        pass

    def visiting_potential_leaf(self, node):
        if type(node) in nonpyomo_leaf_types:
            return True, None

        if node.is_variable_type():
            self.variables[id(node)] = node
            if node.is_fixed():
                self.fixed_vars[id(node)] = node
            return True, None

        if node.is_named_expression_type():
            self.named_expressions[id(node)] = node
            return False, None

        if isinstance(node, EXPR.ExternalFunctionExpression):
            self._external_functions[id(node)] = node
            return False, None

        if node.is_expression_type():
            return False, None

        return True, None


_visitor = _VarAndNamedExprCollector()


def collect_vars_and_named_exprs(expr):
    _visitor.__init__()
    _visitor.dfs_postorder_stack(expr)
    return (
        list(_visitor.named_expressions.values()),
        list(_visitor.variables.values()),
        list(_visitor.fixed_vars.values()),
        list(_visitor._external_functions.values()),
    )

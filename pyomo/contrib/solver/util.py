#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
import pyomo.core.expr as EXPR
from pyomo.core.base.objective import Objective
from pyomo.opt.results.solver import (
    SolverStatus,
    TerminationCondition as LegacyTerminationCondition,
)


from pyomo.contrib.solver.results import TerminationCondition, SolutionStatus


def get_objective(block):
    """
    Get current active objective on a block. If there is more than one active,
    return an error.
    """
    obj = None
    for o in block.component_data_objects(
        Objective, descend_into=True, active=True, sort=True
    ):
        if obj is not None:
            raise ValueError('Multiple active objectives found')
        obj = o
    return obj


def check_optimal_termination(results):
    """
    This function returns True if the termination condition for the solver
    is 'optimal'.

    Parameters
    ----------
    results : Pyomo Results object returned from solver.solve

    Returns
    -------
    `bool`
    """
    if hasattr(results, 'solution_status'):
        if results.solution_status == SolutionStatus.optimal and (
            results.termination_condition
            == TerminationCondition.convergenceCriteriaSatisfied
        ):
            return True
    else:
        if results.solver.status == SolverStatus.ok and (
            results.solver.termination_condition == LegacyTerminationCondition.optimal
            or results.solver.termination_condition
            == LegacyTerminationCondition.locallyOptimal
            or results.solver.termination_condition
            == LegacyTerminationCondition.globallyOptimal
        ):
            return True
    return False


def assert_optimal_termination(results):
    """
    This function checks if the termination condition for the solver
    is 'optimal', 'locallyOptimal', or 'globallyOptimal', and the status is 'ok'
    and it raises a RuntimeError exception if this is not true.

    Parameters
    ----------
    results : Pyomo Results object returned from solver.solve
    """
    if not check_optimal_termination(results):
        if hasattr(results, 'solution_status'):
            msg = (
                'Solver failed to return an optimal solution. '
                'Solution status: {}, Termination condition: {}'.format(
                    results.solution_status, results.termination_condition
                )
            )
        else:
            msg = (
                'Solver failed to return an optimal solution. '
                'Solver status: {}, Termination condition: {}'.format(
                    results.solver.status, results.solver.termination_condition
                )
            )
        raise RuntimeError(msg)


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

        if type(node) is EXPR.ExternalFunctionExpression:
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

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.core.expr.current as EXPR
from pyomo.core.expr.numvalue import nonpyomo_leaf_types, native_numeric_types
from copy import deepcopy

from pyomo.opt import TerminationCondition, SolverStatus

_acceptable_termination_conditions = set([
    TerminationCondition.optimal,
    TerminationCondition.globallyOptimal,
    TerminationCondition.locallyOptimal,
])
_infeasible_termination_conditions = set([
    TerminationCondition.infeasible,
    TerminationCondition.invalidProblem,
])


class NORMAL(object): pass
class INFEASIBLE(object): pass
class NONOPTIMAL(object): pass

def verify_successful_solve(results):
    status = results.solver.status
    term = results.solver.termination_condition

    if status == SolverStatus.ok and term in _acceptable_termination_conditions:
        return NORMAL
    elif term in _infeasible_termination_conditions:
        return INFEASIBLE
    else:
        return NONOPTIMAL


class _CloneVisitor(EXPR.ExpressionValueVisitor):

    def __init__(self, clone_leaves=False, memo=None, substitute=None):
        self.clone_leaves = clone_leaves
        self.memo = memo
        self.substitute = substitute

    def visit(self, node, values):
        """ Visit nodes that have been expanded """
        if node.__class__ is EXPR.TermExpression and not values[1].is_variable_type():
            #
            # Turn a TermExpression whose variable has been replaced by a constant into
            # a simple constant expression.
            #
            return values[0] * values[1]
        return node.construct_node( tuple(values), self.memo )

    def visiting_potential_leaf(self, node):
        """ 
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
        if id(node) in self.substitute:
            return True, self.substitute[id(node)]

        if node.__class__ in nonpyomo_leaf_types:
            #
            # Store a native or numeric object
            #
            return True, deepcopy(node, self.memo)

        if not node.is_expression_type():
            #
            # Store a leave object that is cloned
            #
            if self.clone_leaves:
                return True, deepcopy(node, self.memo)
            else:
                return True, node

        return False, None


def clone_without_expression_components(expr, memo=None, clone_leaves=True, substitute=None):
    """A function that is used to clone an expression.

    Cloning is roughly equivalent to calling ``copy.deepcopy``.
    However, the :attr:`clone_leaves` argument can be used to 
    clone only interior (i.e. non-leaf) nodes in the expression
    tree.   Note that named expression objects are treated as
    leaves when :attr:`clone_leaves` is :const:`True`, and hence
    those subexpressions are not cloned.

    This function uses a non-recursive 
    logic, which makes it more scalable than the logic in 
    ``copy.deepcopy``.

    Args:
        expr: The expression that will be cloned.
        memo (dict): A dictionary mapping object ids to 
            objects.  This dictionary has the same semantics as
            the memo object used with ``copy.deepcopy``.  Defaults
            to None, which indicates that no user-defined
            dictionary is used.
        clone_leaves (bool): If True, then leaves are
            cloned along with the rest of the expression. 
            Defaults to :const:`True`.
   
    Returns: 
        The cloned expression.
    """
    if not memo:
        memo = {'__block_scope__': { id(None): False }}
    if substitute is None:
        substitute = {}
    #
    visitor = _CloneVisitor(clone_leaves=clone_leaves, memo=memo, substitute=substitute)
    return visitor.dfs_postorder_stack(expr)


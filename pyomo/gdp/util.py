#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from six import string_types

import pyomo.core.expr.current as EXPR
from pyomo.core.expr.numvalue import nonpyomo_leaf_types, native_numeric_types
from copy import deepcopy

from pyomo.core.base.component import _ComponentBase, ComponentUID
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


def clone_without_expression_components(expr, substitute=None):
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
        substitute (dict): A dictionary mapping object ids to
            objects.  This dictionary has the same semantics as
            the memo object used with ``copy.deepcopy``.  Defaults
            to None, which indicates that no user-defined
            dictionary is used.

    Returns:
        The cloned expression.
    """
    if substitute is None:
        substitute = {}
    #
    visitor = EXPR.ExpressionReplacementVisitor(substitute=substitute,
                                                remove_named_expressions=True)
    return visitor.dfs_postorder_stack(expr)



def target_list(x):
    if isinstance(x, ComponentUID):
        return [ x ]
    elif isinstance(x, (_ComponentBase, string_types)):
        return [ ComponentUID(x) ]
    elif hasattr(x, '__iter__'):
        ans = []
        for i in x:
            if isinstance(i, ComponentUID):
                ans.append(i)
            elif isinstance(i, (_ComponentBase, string_types)):
                ans.append(ComponentUID(i))
            else:
                raise ValueError(
                    "Expected ComponentUID, Component, Component name, "
                    "or list of these.\n\tReceived %s" % (type(i),))

        return ans
    else:
        raise ValueError(
            "Expected ComponentUID, Component, Component name, "
            "or list of these.\n\tReceived %s" % (type(x),))

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
from pyomo.gdp import GDP_Error
from copy import deepcopy

from pyomo.core.base.component import _ComponentBase, ComponentUID
from pyomo.opt import TerminationCondition, SolverStatus
from pyomo.common.deprecation import deprecation_warning


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
    if isinstance(x, _ComponentBase):
        return [ x ]
    elif hasattr(x, '__iter__'):
        ans = []
        for i in x:
            if isinstance(i, _ComponentBase):
                ans.append(i)
            else:
                raise ValueError(
                    "Expected Component or list of Components."
                    "\n\tRecieved %s" % (type(i),))
        return ans
    else:
        raise ValueError(
            "Expected Component or list of Components."
            "\n\tRecieved %s" % (type(x),))

# [ESJ 07/09/2019 Should this be a more general utility function elsewhere?  I'm
#  putting it here for now so that all the gdp transformations can use it.
#  Returns True if child is a node or leaf in the tree rooted at parent, False
#  otherwise. Accepts list of known components in the tree and updates this list
#  to enhance performance in future calls. Note that both child and parent must
#  be blocks!
def is_child_of(parent, child, knownBlocks=None):
    # Note: we can get away with a dictionary and not ComponentMap because we
    # will only store Blocks (or their ilk), and Blocks are hashable (only
    # derivatives of NumericValue are not hashable)
    if knownBlocks is None:
        knownBlocks = {}
    tmp = set()
    node = child
    while True:
        known = knownBlocks.get(node)
        if known:
            knownBlocks.update({c: True for c in tmp})
            return True
        if known is not None and not known:
            knownBlocks.update({c: False for c in tmp})
            return False
        if node is parent:
            knownBlocks.update({c: True for c in tmp})
            return True
        if node is None:
            knownBlocks.update({c: False for c in tmp})
            return False

        tmp.add(node)
        container = node.parent_component()
        if container is node:
            node = node.parent_block()
        else:
            node = container

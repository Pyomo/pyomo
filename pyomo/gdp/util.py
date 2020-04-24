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
from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData
from copy import deepcopy

from pyomo.core.base.component import _ComponentBase, ComponentUID
from pyomo.opt import TerminationCondition, SolverStatus
from pyomo.common.deprecation import deprecation_warning
import sys

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

def get_src_disjunction(xor_constraint):
    """Return the Disjunction corresponding to xor_constraint

    Parameters
    ----------
    xor_constraint: Constraint, which must be the logical constraint 
                    (located on the transformation block) of some 
                    Disjunction
    """
    # NOTE: This is indeed a linear search through the Disjunctions on the
    # model. I am leaving it this way on the assumption that asking XOR
    # constraints for their Disjunction is not going to be a common
    # question. If we ever need efficiency then we should store a reverse
    # map from the XOR constraint to the Disjunction on the transformation
    # block while we do the transformation. And then this method could query
    # that map.
    m = xor_constraint.model()
    for disjunction in m.component_data_objects(Disjunction):
        if disjunction._algebraic_constraint:
            if disjunction._algebraic_constraint() is xor_constraint:
                return disjunction
    raise GDP_Error("It appears that %s is not an XOR or OR constraint "
                    "resulting from transforming a Disjunction."
                    % xor_constraint.name)

def get_src_disjunct(transBlock):
    """Return the Disjunct object whose transformed components are on
    transBlock.

    Parameters
    ----------
    transBlock: _BlockData which is in the relaxedDisjuncts IndexedBlock
                on a transformation block.
    """
    try:
        return transBlock._srcDisjunct()
    except:
        raise GDP_Error("Block %s doesn't appear to be a transformation "
                        "block for a disjunct. No source disjunct found." 
                        "\n\t(original error: %s)" 
                        % (transBlock.name, sys.exc_info()[1]))

def get_src_constraint(transformedConstraint):
    """Return the original Constraint whose transformed counterpart is
    transformedConstraint

    Parameters
    ----------
    transformedConstraint: Constraint, which must be a component on one of 
    the BlockDatas in the relaxedDisjuncts Block of 
    a transformation block
    """
    transBlock = transformedConstraint.parent_block()
    # This should be our block, so if it's not, the user messed up and gave
    # us the wrong thing. If they happen to also have a _constraintMap then
    # the world is really against us.
    if not hasattr(transBlock, "_constraintMap"):
        raise GDP_Error("Constraint %s is not a transformed constraint" 
                        % transformedConstraint.name)
    # if something goes wrong here, it's a bug in the mappings.
    return transBlock._constraintMap['srcConstraints'][transformedConstraint]

def _find_parent_disjunct(constraint):
    # traverse up until we find the disjunct this constraint lives on
    parent_disjunct = constraint.parent_block()
    while not isinstance(parent_disjunct, _DisjunctData):
        if parent_disjunct is None:
            raise GDP_Error(
                "Constraint %s is not on a disjunct and so was not "
                "transformed" % constraint.name)
        parent_disjunct = parent_disjunct.parent_block()

    return parent_disjunct

def _get_constraint_transBlock(constraint):
    parent_disjunct = _find_parent_disjunct(constraint)
    # we know from _find_parent_disjunct that parent_disjunct is a Disjunct,
    # so the below is OK
    transBlock = parent_disjunct._transformation_block
    if transBlock is None:
        raise GDP_Error("Constraint %s is on a disjunct which has not been "
                        "transformed" % constraint.name)
    # if it's not None, it's the weakref we wanted.
    transBlock = transBlock()

    return transBlock

def get_transformed_constraint(srcConstraint):
    """Return the transformed version of srcConstraint

    Parameters
    ----------
    srcConstraint: Constraint, which must be in the subtree of a
                   transformed Disjunct
    """
    transBlock = _get_constraint_transBlock(srcConstraint)

    if hasattr(transBlock, "_constraintMap") and transBlock._constraintMap[
            'transformedConstraints'].get(srcConstraint) is not None:
        return transBlock._constraintMap['transformedConstraints'][
            srcConstraint]
    raise GDP_Error("Constraint %s has not been transformed." 
                    % srcConstraint.name)

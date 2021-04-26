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
from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData, Disjunct

from pyomo.core.base.component import _ComponentBase

from pyomo.core import Block, TraversalStrategy
from pyomo.opt import TerminationCondition, SolverStatus
from six import iterkeys
from weakref import ref as weakref_ref
import logging

logger = logging.getLogger('pyomo.gdp')

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
    for disjunction in m.component_data_objects(Disjunction,
                                                descend_into=(Block, Disjunct)):
        if disjunction._algebraic_constraint:
            if disjunction._algebraic_constraint() is xor_constraint:
                return disjunction
    raise GDP_Error("It appears that '%s' is not an XOR or OR constraint "
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
    if not hasattr(transBlock, "_srcDisjunct") or \
       type(transBlock._srcDisjunct) is not weakref_ref:
        raise GDP_Error("Block '%s' doesn't appear to be a transformation "
                        "block for a disjunct. No source disjunct found."
                        % transBlock.name)
    return transBlock._srcDisjunct()

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
        raise GDP_Error("Constraint '%s' is not a transformed constraint"
                        % transformedConstraint.name)
    # if something goes wrong here, it's a bug in the mappings.
    return transBlock._constraintMap['srcConstraints'][transformedConstraint]

def _find_parent_disjunct(constraint):
    # traverse up until we find the disjunct this constraint lives on
    parent_disjunct = constraint.parent_block()
    while not isinstance(parent_disjunct, _DisjunctData):
        if parent_disjunct is None:
            raise GDP_Error(
                "Constraint '%s' is not on a disjunct and so was not "
                "transformed" % constraint.name)
        parent_disjunct = parent_disjunct.parent_block()

    return parent_disjunct

def _get_constraint_transBlock(constraint):
    parent_disjunct = _find_parent_disjunct(constraint)
    # we know from _find_parent_disjunct that parent_disjunct is a Disjunct,
    # so the below is OK
    transBlock = parent_disjunct._transformation_block
    if transBlock is None:
        raise GDP_Error("Constraint '%s' is on a disjunct which has not been "
                        "transformed" % constraint.name)
    # if it's not None, it's the weakref we wanted.
    transBlock = transBlock()

    return transBlock

def get_transformed_constraints(srcConstraint):
    """Return the transformed version of srcConstraint

    Parameters
    ----------
    srcConstraint: SimpleConstraint or _ConstraintData, which must be in
    the subtree of a transformed Disjunct
    """
    if srcConstraint.is_indexed():
        raise GDP_Error("Argument to get_transformed_constraint should be "
                        "a SimpleConstraint or _ConstraintData. (If you "
                        "want the container for all transformed constraints "
                        "from an IndexedDisjunction, this is the parent "
                        "component of a transformed constraint originating "
                        "from any of its _ComponentDatas.)")
    transBlock = _get_constraint_transBlock(srcConstraint)
    try:
        return transBlock._constraintMap['transformedConstraints'][srcConstraint]
    except:
        logger.error("Constraint '%s' has not been transformed."
                     % srcConstraint.name)
        raise

def _warn_for_active_disjunction(disjunction, disjunct, NAME_BUFFER):
    # this should only have gotten called if the disjunction is active
    assert disjunction.active
    problemdisj = disjunction
    if disjunction.is_indexed():
        for i in sorted(iterkeys(disjunction)):
            if disjunction[i].active:
                # a _DisjunctionData is active, we will yell about
                # it specifically.
                problemdisj = disjunction[i]
                break

    parentblock = problemdisj.parent_block()
    # the disjunction should only have been active if it wasn't transformed
    assert problemdisj.algebraic_constraint is None
    _probDisjName = problemdisj.getname(
        fully_qualified=True, name_buffer=NAME_BUFFER)
    _disjName = disjunct.getname(fully_qualified=True, name_buffer=NAME_BUFFER)
    raise GDP_Error("Found untransformed disjunction '%s' in disjunct '%s'! "
                    "The disjunction must be transformed before the "
                    "disjunct. If you are using targets, put the "
                    "disjunction before the disjunct in the list."
                    % (_probDisjName, _disjName))

def _warn_for_active_disjunct(innerdisjunct, outerdisjunct, NAME_BUFFER):
    assert innerdisjunct.active
    problemdisj = innerdisjunct
    if innerdisjunct.is_indexed():
        for i in sorted(iterkeys(innerdisjunct)):
            if innerdisjunct[i].active:
                # This shouldn't be true, we will complain about it.
                problemdisj = innerdisjunct[i]
                break

    raise GDP_Error("Found active disjunct '{0}' in disjunct '{1}'! Either {0} "
                    "is not in a disjunction or the disjunction it is in "
                    "has not been transformed. {0} needs to be deactivated "
                    "or its disjunction transformed before {1} can be "
                    "transformed.".format(
                        problemdisj.getname(
                            fully_qualified=True, name_buffer = NAME_BUFFER),
                        outerdisjunct.getname(
                            fully_qualified=True,
                            name_buffer=NAME_BUFFER)))


def _warn_for_active_logical_constraint(logical_statement, disjunct, NAME_BUFFER):
    # this should only have gotten called if the logical constraint is active
    assert logical_statement.active
    problem_statement = logical_statement
    if logical_statement.is_indexed():
        for i in logical_statement:
            if logical_statement[i].active:
                # a _LogicalConstraintData is active, we will yell about
                # it specifically.
                problem_statement = logical_statement[i]
                break
        # None of the _LogicalConstraintDatas were actually active. We
        # are OK and we can deactivate the container.
        else:
            logical_statement.deactivate()
            return
    # the logical constraint should only have been active if it wasn't transformed
    _probStatementName = problem_statement.getname(
        fully_qualified=True, name_buffer=NAME_BUFFER)
    raise GDP_Error("Found untransformed logical constraint %s in disjunct %s! "
                    "The logical constraint must be transformed before the "
                    "disjunct. Use the logical_to_linear transformation."
                    % (_probStatementName, disjunct.name))


def check_model_algebraic(instance):
    """Checks if there are any active Disjuncts or Disjunctions reachable via
    active Blocks. If there are not, it returns True. If there are, it issues
    a warning detailing where in the model there are remaining non-algebraic
    components, and returns False.

    Parameters
    ----------
    instance: a Model or Block
    """
    disjunction_set = {i for i in instance.component_data_objects(
        Disjunction, descend_into=(Block, Disjunct), active=None)}
    active_disjunction_set = {i for i in instance.component_data_objects(
        Disjunction, descend_into=(Block, Disjunct), active=True)}
    disjuncts_in_disjunctions = set()
    for i in disjunction_set:
        disjuncts_in_disjunctions.update(i.disjuncts)
    disjuncts_in_active_disjunctions = set()
    for i in active_disjunction_set:
        disjuncts_in_active_disjunctions.update(i.disjuncts)

    for disjunct in instance.component_data_objects(
            Disjunct, descend_into=(Block,),
            descent_order=TraversalStrategy.PostfixDFS):
        # check if it's relaxed
        if disjunct.transformation_block is not None:
            continue
        # It's not transformed, check if we should complain
        elif disjunct.active and _disjunct_not_fixed_true(disjunct) and \
             _disjunct_on_active_block(disjunct):
            # If someone thinks they've transformed the whole instance, but
            # there is still an active Disjunct on the model, we will warn
            # them. In the future this should be the writers' job.)
            if disjunct not in disjuncts_in_disjunctions:
                logger.warning('Disjunct "%s" is currently active, '
                               'but was not found in any Disjunctions. '
                               'This is generally an error as the model '
                               'has not been fully relaxed to a '
                               'pure algebraic form.' % (disjunct.name,))
                return False
            elif disjunct not in disjuncts_in_active_disjunctions:
                logger.warning('Disjunct "%s" is currently active. While '
                               'it participates in a Disjunction, '
                               'that Disjunction is currently deactivated. '
                               'This is generally an error as the '
                               'model has not been fully relaxed to a pure '
                               'algebraic form. Did you deactivate '
                               'the Disjunction without addressing the '
                               'individual Disjuncts?' % (disjunct.name,))
                return False
            else:
                logger.warning('Disjunct "%s" is currently active. It must be '
                               'transformed or deactivated before solving the '
                               'model.' % (disjunct.name,))
                return False
    # We didn't find anything bad.
    return True

def _disjunct_not_fixed_true(disjunct):
    # Return true if the disjunct indicator variable is not fixed to True
    return not (disjunct.indicator_var.fixed and
                disjunct.indicator_var.value == 1)

def _disjunct_on_active_block(disjunct):
    # Check first to make sure that the disjunct is not a descendent of an
    # inactive Block or fixed and deactivated Disjunct, before raising a
    # warning.
    parent_block = disjunct.parent_block()
    while parent_block is not None:
        # deactivated Block
        if parent_block.ctype is Block and not parent_block.active:
            return False
        # properly deactivated Disjunct
        elif (parent_block.ctype is Disjunct and not parent_block.active
              and parent_block.indicator_var.value == 0
              and parent_block.indicator_var.fixed):
            return False
        else:
            # Step up one level in the hierarchy
            parent_block = parent_block.parent_block()
            continue
    return True
